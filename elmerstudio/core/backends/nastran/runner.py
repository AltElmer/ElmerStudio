"""Run a study with NASTRAN-95: mesh -> deck -> solver -> Studio solution folder."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np

from ...builder import step_components, study_steps, study_sweep
from ...meshing import build_mesh
from ...results import write_solution_meta
from .convert import NastranConverter
from .deck import NastranPlan, NastranRefusal, check_model, plan_deck, write_deck
from .install import NastranInstall, find_nastran
from .parser import NastranResult, parse_output

TIMEOUT = float(os.environ.get("ELMERSTUDIO_NASTRAN_TIMEOUT", "1800"))


def preview_files(model, study) -> list[tuple[str, str]]:
    """Generate the NASTRAN input deck for preview without executing the solve."""
    reason = check_model(model, study)
    if reason:
        raise NastranRefusal(reason)
    steps = study_steps(study)
    if not steps:
        raise RuntimeError(f"{study.label} has no study steps")
    comp = model.component
    if comp is None:
        raise RuntimeError("Model has no component")

    mesh = build_mesh(model)
    plan = plan_deck(model, study, steps[0])
    import tempfile
    with tempfile.TemporaryDirectory(prefix="es_nastran_preview_") as d:
        p = os.path.join(d, "case.bdf")
        deck_text = write_deck(plan, mesh, model, p)
    return [("case.bdf", deck_text)]


class NastranStudy:
    """Counterpart of FoamStudy for NASTRAN-95. Managed by StudyRunner."""

    def __init__(self, r, inst: NastranInstall | None = None):
        self.r = r
        self.inst = inst or find_nastran()
        self.proc: subprocess.Popen | None = None
        self.case: str | None = None

    def kill(self):
        p = self.proc
        if p is not None and p.poll() is None:
            self.inst.kill(p, self.case)

    def run(self) -> list[str]:
        from ...study_runner import StudyCancelled, component_dir, component_order, sweep_combinations
        r = self.r
        reason = check_model(r.model, r.study)
        if reason:
            raise NastranRefusal(reason)
        if self.inst is None:
            raise RuntimeError(
                "NASTRAN-95 was not found. Please install NASTRAN-95 and set NASTRAN_ROOT or "
                "configure its location in Preferences."
            )

        steps = study_steps(r.study)
        if not steps:
            raise RuntimeError(f"{r.study.label} has no study steps")

        sweep = study_sweep(r.study)
        combos = sweep_combinations(r.model, sweep)
        pname = next(iter(combos[0]), None) if combos[0] else None
        sdir = r.study_dir()
        if os.path.isdir(sdir):
            shutil.rmtree(sdir, ignore_errors=True)
        os.makedirs(sdir, exist_ok=True)

        order = component_order(r.model)
        plan = [[c for c in order if c in step_components(r.model, step)] for step in steps]
        total = max(1, len(combos) * sum(len(p) for p in plan))
        self.job, t_start = 0, time.time()
        r.emit("message", f"Solver backend: {self.inst.describe()}")

        for vi, combo in enumerate(combos):
            overrides = dict(combo) if combo else None
            if pname:
                desc = ", ".join(f"{k} = {v:g}" for k, v in combo.items())
                r.emit("message", f"Parametric sweep: {desc} ({vi + 1}/{len(combos)})")
                r.emit("sweep", f"{desc} ({vi + 1}/{len(combos)})")

            for si, step in enumerate(steps):
                for comp in plan[si]:
                    if r._stop.is_set():
                        raise StudyCancelled()
                    with r.model.scoped(comp):
                        r.emit("progress", (self.job / total, f"Mesh {comp.label}" if len(order) > 1 else "Meshing"))
                        mesh = build_mesh(r.model, overrides, inst=getattr(r, "inst", None))
                        mesh.comp_tag = comp.tag
                        r.emit("mesh", mesh)

                        folder = os.path.join(component_dir(r.model, sdir, comp.tag), f"sol_{vi}_{si}")
                        self._solve(comp, step, si, vi, combo, pname, mesh, overrides, folder, total, t_start)

        r.emit("message", f"Solution time: {time.time() - t_start:.1f} s")
        return r.folders

    def _solve(self, comp, step, si, vi, combo, pname, mesh, overrides, folder, total, t_start):
        from ...study_runner import StudyCancelled
        r, job = self.r, self.job
        val = combo.get(pname) if pname else None
        case = os.path.join(folder, "nastran")
        os.makedirs(case, exist_ok=True)
        self.case = case

        # 1. Plan and write deck
        P = plan_deck(r.model, r.study, step)
        deck_file = os.path.join(case, "case.inp")
        out_file = os.path.join(case, "case.out")
        err_file = os.path.join(case, "case.err")
        write_deck(P, mesh, r.model, deck_file, overrides)

        # 2. Ensure rf is linked into the case directory
        self.inst.prepare_case(case)

        # 3. Execute NASTRAN
        logs: list[str] = []
        r.emit("message", f"{step.label}: running NASTRAN-95 (SOL {P.sol})...")
        r.emit("log", f"===== NASTRAN-95 SOL {P.sol} =====")
        logs.append(f"===== NASTRAN-95 SOL {P.sol} =====")

        t0 = time.time()
        timed_out = False

        if self.inst.kind == "wsl":
            # WSL launch
            argv = self.inst.argv(case, deck_file)
            self.proc = subprocess.Popen(
                argv, cwd=case, env=self.inst.env(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                errors="replace", bufsize=1, **self.inst.popen_kwargs()
            )
            r._killer = self.kill
            for line in self.proc.stdout:
                line = line.rstrip("\r\n")
                logs.append(line)
                r.emit("log", line)
                if r._stop.is_set():
                    break
                if time.time() - t0 > TIMEOUT:
                    timed_out = True
                    self.kill()
                    break
            self.proc.wait()
            returncode = self.proc.returncode
        else:
            # Native Windows or Linux
            with open(deck_file, "r") as stdin_f, open(out_file, "w", errors="replace") as stdout_f, open(err_file, "w", errors="replace") as stderr_f:
                self.proc = subprocess.Popen(
                    [self.inst.exe], stdin=stdin_f, stdout=stdout_f, stderr=stderr_f,
                    cwd=case, env=self.inst.env(), **self.inst.popen_kwargs()
                )
                r._killer = self.kill
                try:
                    self.proc.wait(timeout=TIMEOUT)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self.kill()
                returncode = self.proc.returncode

            if os.path.isfile(out_file):
                with open(out_file, "r", errors="replace") as f:
                    for line in f:
                        line = line.rstrip("\r\n")
                        logs.append(line)
                        if "PAGE" not in line and "NASTRAN" not in line:
                            r.emit("log", line)

        self.proc = None
        r._killer = None

        if r._stop.is_set():
            raise StudyCancelled()

        # 4. Read output and parse results
        out_content = "\n".join(logs)
        if os.path.isfile(out_file):
            with open(out_file, "r", errors="replace") as f:
                out_content = f.read()

        parsed = parse_output(out_content)

        # 5. Determine status
        status = "converged"
        diagnostics = []

        if timed_out:
            status = "failed"
            diagnostics.append(f"{step.label}: NASTRAN did not finish within {TIMEOUT:g} s")
        elif not parsed.success or parsed.fatal_messages:
            status = "failed"
            diagnostics.extend([f"{step.label}: {msg}" for msg in parsed.fatal_messages])
        elif returncode != 0:
            status = "failed"
            diagnostics.append(f"{step.label}: NASTRAN exited with code {returncode}")
        elif not parsed.displacements and not parsed.temperatures and not parsed.eigenvalues:
            status = "failed"
            diagnostics.append(f"{step.label}: NASTRAN produced no displacements or eigenvalues")

        # 6. Convert to ElmerStudio VTU and mesh.npz
        converter = NastranConverter(mesh, parsed)
        if status != "failed":
            pt_data = converter.extract_point_data()
            converter.write_vtu(pt_data, os.path.join(folder, "results", "case_t0001.vtu"))
            converter.save_mesh_npz(os.path.join(folder, "mesh.npz"))

        # 7. Write solution metadata
        meta = self._meta(P, step, mesh, parsed, status, diagnostics, combo, val, pname, vi, si, t_start, returncode)
        write_solution_meta(folder, meta)

        with open(os.path.join(folder, "model.json"), "w") as f:
            f.write(r.model.to_json())
        with open(os.path.join(folder, "solver.log"), "w", errors="replace") as f:
            f.write("\n".join(logs))

        if status == "failed":
            raise RuntimeError(diagnostics[0] if diagnostics else f"{step.label}: NASTRAN run failed")

        r.folders.append(folder)
        self.job += 1
        r.emit("progress", (self.job / total, "Done"))

    def _meta(self, P: NastranPlan, step, mesh, parsed: NastranResult, status: str, diagnostics: list[str],
              combo, val, pname, vi, si, t_start, returncode: int) -> dict:
        r = self.r
        meta = {
            "kind": step.kind,
            "times": [1.0],
            "freqs": [eg["freq_hz"] for eg in parsed.eigenvalues],
            "neig": len(parsed.eigenvalues),
            "warnings": [],
            "physics": [ph.tag for ph in r.model.component.children if ph.kind == f"physics.{P.physics}"],
            "axi": False,
            "catalog": [],
            "var_domains": {},
            "backend": "nastran",
            "nastran": {
                "sol": P.sol,
                "app": P.app,
                "version": self.inst.version,
                "kind": self.inst.kind,
                "eigenvalues": parsed.eigenvalues,
            },
            "mesh_stats": getattr(mesh, "stats", {}),
            "comp": r.model.component.tag,
            "xrefs": {},
            "eigen": [float(eg["eigenvalue"]) for eg in parsed.eigenvalues],
            "sweep_param": pname,
            "sweep_value": val,
            "sweep_params": dict(combo) if combo else {},
            "step_index": si,
            "sweep_index": vi,
            "tunit": step.get("tunit", "s"),
            "returncode": returncode,
            "elapsed": time.time() - t_start,
            "status": status,
            "diagnostics": diagnostics,
            "params": dict(r.model.parameters(dict(combo) if combo else None))
        }
        meta["selections"] = {n.tag: list(n.selection.entities) for n in r.model.root.walk()
                              if n.selection is not None and n.selection.named}
        for tag, (_lvl, ents) in getattr(mesh.geometry, "named", {}).items():
            meta["selections"].setdefault(tag, [int(e) for e in ents])
        return meta
