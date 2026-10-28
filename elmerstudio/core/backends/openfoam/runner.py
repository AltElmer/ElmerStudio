"""Run a study with OpenFOAM: per sweep value and study step, blockMesh -> fields -> solver -> Studio solution folder.

The solution folder has the same layout as an Elmer run (meta.json with status and diagnostics, model.json,
solver.log, mesh.npz, results/case_t*.vtu), plus the OpenFOAM case in foam/.  Convergence is judged from the
solver log with the same statuses and the same study policy as the Elmer path.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np

from ...builder import step_components, study_steps, study_sweep
from ...results import topology_to_npz, write_solution_meta
from .blocks import analyse
from .case import FoamPlan, FoamRefusal, check_model, plan_case, time_name, write_case, write_fields
from .convert import Converter
from .install import FoamInstall, find_openfoam
from .polymesh import PolyMesh, topology

TIMEOUT = float(os.environ.get("ELMERSTUDIO_FOAM_TIMEOUT", "3600"))   # seconds per OpenFOAM command

RE_TIME = re.compile(r"^Time = (\S+)")
RE_SOLVE = re.compile(r"(\w+):\s+Solving for (\w+), Initial residual = ([-+\d.eE]+), Final residual = ([-+\d.eE]+), "
                      r"No Iterations (\d+)")
RE_SIMPLE = re.compile(r"SIMPLE solution converged in (\S+) iterations")
RE_CONT = re.compile(r"time step continuity errors : sum local = ([-+\d.eE]+), global = ([-+\d.eE]+), "
                     r"cumulative = ([-+\d.eE]+)")
RE_FATAL = re.compile(r"FOAM FATAL|Floating point exception(?! trapping)|FOAM aborting|"
                      r"^#\d+\s+Foam::error::printStack|Segmentation fault|Foam::sigFpe::sigHandler|Foam::sigSegv")


@dataclass
class FoamRunState:
    lines: list = field(default_factory=list)
    conv: list = field(default_factory=list)          # (step index, field, "residual", initial residual, step)
    solves: list = field(default_factory=list)        # (step index, solver, field, initial, final, iterations)
    steps: int = 0
    time: str = ""
    simple_converged: int | None = None
    ended: bool = False
    fatal: list = field(default_factory=list)
    continuity: list = field(default_factory=list)    # (step index, sum local)
    returncode: int | None = None
    timed_out: bool = False
    change: dict | None = None                        # SIMPLE at its limit: relative change of one more iteration
    _seen: set = field(default_factory=set)

    def feed(self, line: str) -> str | None:
        self.lines.append(line)
        s = line.strip()
        m = RE_TIME.match(s)
        if m:
            self.steps += 1
            self.time = m.group(1)
            self._seen = set()
            return "step"
        m = RE_SOLVE.search(s)
        if m:
            solver, fld, r0, r1, it = m.group(1), m.group(2), float(m.group(3)), float(m.group(4)), int(m.group(5))
            self.solves.append((self.steps, solver, fld, r0, r1, it))
            if fld not in self._seen:                  # the first solve of a field per step: its initial residual
                self._seen.add(fld)
                self.conv.append((self.steps, fld, "residual", r0, self.steps))
                return "conv"
            return None
        m = RE_CONT.search(s)
        if m:
            self.continuity.append((self.steps, float(m.group(1))))
            return None
        m = RE_SIMPLE.search(s)
        if m:
            self.simple_converged = int(float(m.group(1)))
            return None
        if s == "End":
            self.ended = True
            return None
        if RE_FATAL.search(s):
            self.fatal.append(s)
            return "error"
        return None


def fatal_message(rs: FoamRunState) -> str:
    """OpenFOAM's own words: the lines after 'FOAM FATAL ... ERROR' up to the blank line."""
    for i, ln in enumerate(rs.lines):
        if "FOAM FATAL" in ln:
            body = []
            for x in rs.lines[i + 1:i + 14]:
                x = x.strip()
                if not x and body:
                    break
                if x and not x.startswith(("From ", "in file", "file:", "    ")):
                    body.append(x)
            return " ".join(body)[:400] or ln.strip()
    return rs.fatal[-1] if rs.fatal else ""


def judge_foam(rs: FoamRunState, P: FoamPlan, outputs: list, label: str) -> tuple[str, list[str]]:
    """('failed', [message]) when OpenFOAM stopped with an error or wrote no result; ('not converged', [problems])
    when SIMPLE ran out of iterations (and its last iteration still changed the solution by more than the
    tolerance), a linear solve ended above its tolerance, or a time step lost continuity; else ('converged', []).

    OpenFOAM normalizes a residual by the field's own variation, so for a field that is uniform or zero (the
    pressure with slip walls, v in a straight channel) the residual is noise and residual control is never met.
    Then the relative change of one more SIMPLE iteration decides, the measure Elmer's nonlinear tolerance uses."""
    if rs.timed_out:
        return "failed", [f"{label}: {P.app} did not finish within {TIMEOUT:g} s and was stopped"]
    if rs.returncode != 0 or rs.fatal or not rs.ended:
        msg = fatal_message(rs) or f"{P.app} exited with code {rs.returncode}"
        return "failed", [f"{label}: {P.app}: {msg}"]
    if not outputs:
        return "failed", [f"{label}: {P.app} wrote no result"]
    problems = []
    if P.app == "simpleFoam":
        if rs.simple_converged is None:
            ch = rs.change or {}
            if not ch or max(ch.values()) > P.tol:
                last = {}
                for st, _sv, fld, r0, _r1, _it in rs.solves:
                    if st == rs.steps:
                        last.setdefault(fld, r0)
                res = ", ".join(f"{k} {v:.3g}" for k, v in last.items())
                chg = (", and one more iteration changed " + ", ".join(f"{k} by {v:.3g}" for k, v in ch.items())
                       if ch else "")
                problems.append(f"SIMPLE reached its maximum of {P.maxit} iterations with initial residuals {res} "
                                f"above the tolerance {P.tol:g}{chg}")
    else:
        bad = [(st, fld, r1, it) for st, _sv, fld, _r0, r1, it in rs.solves if r1 > P.lin_tol * (1 + 1e-9)]
        if bad:
            st, fld, r1, it = max(bad, key=lambda b: b[2] / P.lin_tol)
            problems.append(f"{len(bad)} linear solve(s) ended above the tolerance {P.lin_tol:g}; the worst, {fld} "
                            f"in step {st}, at {r1:.3g} after {it} iterations (maximum {P.lin_maxit})")
    if P.transient and rs.continuity:              # per time step it is dimensionless (dt times |div u|)
        worst = max(c for _s, c in rs.continuity)
        if worst > 1e-4:
            problems.append(f"the continuity error of a time step reached {worst:.3g} (sum local)")
    return ("not converged", [f"{label}: {p}" for p in problems]) if problems else ("converged", [])


def preview_files(model, study) -> list[tuple[str, str]]:
    """The dictionaries a Compute would write for the study's first step, without running anything:
    [(relative path, text)].  The fields in 0/ need the blockMesh mesh, so they are summarized per patch."""
    import tempfile
    reason = check_model(model, study)
    if reason:
        raise FoamRefusal(reason)
    steps = study_steps(study)
    if not steps:
        raise RuntimeError(f"{study.label} has no study steps")
    layout = analyse(model)
    P = plan_case(model, study, steps[0], layout)
    out = []
    with tempfile.TemporaryDirectory(prefix="es_foam_preview_") as d:
        write_case(P, layout, d)
        for sub in ("system", "constant"):
            for fn in sorted(os.listdir(os.path.join(d, sub))):
                with open(os.path.join(d, sub, fn)) as f:
                    out.append((f"{sub}/{fn}", f.read()))
    rows = [f"// written after blockMesh, per patch (values in SI as entered; p is divided by rho = {P.rho:g})"]
    for b, (kind, ex) in sorted(P.bcs.items()):
        vals = ", ".join(f"{k} = {v}" for k, v in ex.items())
        rows.append(f"bnd{b:<4} {P.patch_types[b]:<14} {kind:<16} {vals}   // {P.labels.get(b, 'default')}")
    if P.sdim == 2:
        rows.append("frontAndBack empty")
    out.append((f"{time_name(P.t0) if P.transient else '0'}/ (boundary conditions)", "\n".join(rows) + "\n"))
    return out


def output_dirs(case: str, P: FoamPlan) -> list[tuple[float, str]]:
    """(time, folder) of the results to convert: the last iteration (SIMPLE), the solve (steady heat) or every
    output time (transient)."""
    found = []
    for d in os.listdir(case):
        try:
            t = float(d)
        except ValueError:
            continue
        if os.path.isdir(os.path.join(case, d)):
            found.append((t, os.path.join(case, d)))
    found.sort()
    if not P.transient:
        later = [f for f in found if f[0] > 0]
        return later[-1:]
    out = []
    for t in P.out_times:
        hit = [f for f in found if abs(f[0] - t) <= 1e-9 * max(abs(t), P.dt)]
        if not hit:
            return []
        out.append(hit[0])
    return out


class FoamStudy:
    """The OpenFOAM counterpart of StudyRunner._run; *r* is the StudyRunner (events, stop flag, folders)."""

    def __init__(self, r, inst: FoamInstall | None = None):
        self.r = r
        self.inst = inst or find_openfoam()
        self.proc = None
        self.case = None

    def kill(self):
        p = self.proc
        if p is not None and p.poll() is None:
            self.inst.kill(p, self.case)

    def run(self) -> list[str]:
        from ...study_runner import StudyCancelled, component_dir, component_order, sweep_combinations
        r = self.r
        reason = check_model(r.model, r.study)
        if reason:
            raise FoamRefusal(reason)
        if self.inst is None:
            raise RuntimeError("OpenFOAM was not found. Set OPENFOAM_ROOT to the OpenFOAM installation (a Windows "
                               "tree with its msys64, a Linux project folder, or wsl:/usr/lib/openfoam/openfoamNNNN).")
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
        # each component the step solves is its own OpenFOAM case (in component_order: the Elmer path's order)
        order = component_order(r.model)
        plan = [[c for c in order if c in step_components(r.model, step)] for step in steps]
        total = max(1, len(combos) * sum(len(p) for p in plan))
        self.job, t_start = 0, time.time()
        r.emit("message", f"Solver backend: {self.inst.describe()}")
        for vi, combo in enumerate(combos):
            overrides = dict(combo) if combo else None
            if pname:
                desc = ", ".join(f"{k} = {v:g}" for k, v in combo.items())
                r.emit("message", f"Parametric sweep: {desc}  ({vi + 1}/{len(combos)})")
                r.emit("sweep", f"{desc} ({vi + 1}/{len(combos)})")
            layouts: dict = {}
            for si, step in enumerate(steps):
                for comp in plan[si]:
                    if r._stop.is_set():
                        raise StudyCancelled()
                    with r.model.scoped(comp):
                        if comp.tag not in layouts:
                            r.emit("progress", (self.job / total, f"blockMesh {comp.label}" if len(order) > 1
                                                else "blockMesh"))
                            layouts[comp.tag] = analyse(r.model, overrides)
                        folder = os.path.join(component_dir(r.model, sdir, comp.tag), f"sol_{vi}_{si}")
                        self._solve(comp, step, si, vi, combo, pname, layouts[comp.tag], overrides, folder, total,
                                    t_start)
        r.emit("message", f"Solution time: {time.time() - t_start:.1f} s")
        return r.folders

    def _solve(self, comp, step, si, vi, combo, pname, layout, overrides, folder, total, t_start):
        """One component and study step: blockMesh, fields, solver, judge, solution folder."""
        from ...study_runner import StudyNotConverged
        r, job = self.r, self.job
        val = combo.get(pname) if pname else None
        case = os.path.join(folder, "foam")
        os.makedirs(case, exist_ok=True)
        P = plan_case(r.model, r.study, step, layout, overrides)
        write_case(P, layout, case)
        logs = []
        rs_mesh = self._run("blockMesh", case, logs, job, total, step)
        if rs_mesh.returncode != 0 or rs_mesh.fatal:
            with open(os.path.join(folder, "solver.log"), "w", errors="replace") as f:
                f.write("\n".join(logs))
            raise RuntimeError(f"{step.label}: blockMesh: {fatal_message(rs_mesh) or 'failed'}")
        pm = PolyMesh(case)
        un = pm.patch("unassigned")
        if un is not None and un["nFaces"]:
            raise RuntimeError(f"{step.label}: blockMesh left {un['nFaces']} boundary faces without a patch")
        specs = write_fields(P, pm, case, time_name(P.t0) if P.transient else "0")
        topo, pmap = topology(pm, layout)
        topo.comp_tag = comp.tag
        topology_to_npz(topo, os.path.join(folder, "mesh.npz"))
        r.emit("message", f"blockMesh: {pm.n_cells} cells, {len(pm.points)} points; "
                          f"{step.label}: solving {P.app} (case folder {case})")
        rs = self._run(P.app, case, logs, job, total, step, P)
        if P.app == "simpleFoam" and rs.simple_converged is None and rs.ended and rs.returncode == 0 \
                and not rs.fatal:
            rs.change = self._last_change(case, P, logs, job, total, step)
            if rs.change is not None:
                r.emit("message", f"{step.label}: SIMPLE reached its {P.maxit} iterations; one more "
                                  "iteration changed " + ", ".join(f"{k} by {v:.3g}" for k, v in rs.change.items()))
        outputs = output_dirs(case, P)
        status, diagnostics = judge_foam(rs, P, outputs, step.label)
        if status != "failed":
            conv = Converter(P, pm, specs, topo, pmap)
            for i, (_t, tdir) in enumerate(outputs, 1):
                conv.write_vtu(conv.convert_time(tdir), os.path.join(folder, "results", f"case_t{i:04d}.vtu"))
        meta = self._meta(P, step, layout, pm, rs, status, diagnostics, combo, val, pname, vi, si, t_start)
        meta.update({"mesh_stats": topo.stats, "comp": comp.tag, "xrefs": {}})
        write_solution_meta(folder, meta)
        with open(os.path.join(folder, "model.json"), "w") as f:
            f.write(r.model.to_json())
        with open(os.path.join(folder, "solver.log"), "w", errors="replace") as f:
            f.write("\n".join(logs))
        if status == "failed":
            raise RuntimeError(diagnostics[0])
        if status == "not converged":
            if r.study.get("nonconv", "stop") != "keep":
                raise StudyNotConverged("; ".join(diagnostics) + ". Raise the maximum number of iterations "
                                        "in the study's Solver Configurations, or choose to keep unconverged "
                                        "solutions in the study settings.")
            for d in diagnostics:
                r.emit("warning", f"NOT CONVERGED: {d}")
        r.folders.append(folder)
        self.job += 1
        r.emit("progress", (self.job / total, "Done"))

    def _run(self, command, case, logs, job, total, step, P: FoamPlan | None = None) -> FoamRunState:
        from ...study_runner import StudyCancelled
        r = self.r
        rs = FoamRunState()
        self.case = case
        logs.append(f"===== {command} =====")
        r.emit("log", f"===== {command} =====")
        self.proc = subprocess.Popen(self.inst.argv(command, case), cwd=case, env=self.inst.env(),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                                     bufsize=1, **self.inst.popen_kwargs())
        r._killer = self.kill
        t0 = time.time()
        with open(os.path.join(case, f"log.{command.split()[0]}"), "w", errors="replace") as logf:
            for line in self.proc.stdout:
                line = line.rstrip("\n")
                logf.write(line + "\n")
                logs.append(line)
                ev = rs.feed(line)
                r.emit("log", line)
                if ev == "conv":
                    r.emit("conv", rs.conv[-1])
                elif ev == "step" and P is not None:
                    if P.app == "simpleFoam":
                        frac, txt = rs.steps / max(P.maxit, 1), f"iteration {rs.steps}"
                    elif P.transient:
                        frac, txt = rs.steps / max(P.nsteps, 1), f"t = {rs.time} s ({rs.steps}/{P.nsteps})"
                    else:
                        frac, txt = 1.0, "solve"
                    r.emit("progress", ((job + min(frac, 1.0)) / total, f"{step.label}: {txt}"))
                elif ev == "error":
                    r.emit("error_line", line)
                if r._stop.is_set():
                    break
                if time.time() - t0 > TIMEOUT:
                    rs.timed_out = True
                    self.kill()
                    break
        if r._stop.is_set() or rs.timed_out:
            self.kill()
        self.proc.wait()
        rs.returncode = self.proc.returncode
        self.proc = None
        r._killer = None
        if r._stop.is_set():
            raise StudyCancelled()
        return rs

    def _last_change(self, case, P: FoamPlan, logs, job, total, step) -> dict | None:
        """Run one more SIMPLE iteration from the last one; the relative change of U (of max |U|) and of p (of
        max(|p|, |U|^2), the kinematic dynamic-pressure scale)."""
        from .convert import _read_field
        last = output_dirs(case, P)
        if not last:
            return None
        t, tdir = last[-1]
        ctl = os.path.join(case, "system", "controlDict")
        with open(ctl) as f:
            text = f.read()
        text = re.sub(r"endTime\s+[^;]+;", f"endTime         {int(t) + 1};", text)
        text = re.sub(r"startFrom\s+[^;]+;", "startFrom       latestTime;", text)
        text = re.sub(r"writeInterval\s+[^;]+;", "writeInterval   1;", text)
        with open(ctl, "w", newline="\n") as f:
            f.write(text)
        rs2 = self._run(P.app, case, logs, job, total, step, P)
        nxt = os.path.join(case, time_name(int(t) + 1))
        if rs2.returncode != 0 or rs2.fatal or not os.path.isdir(nxt):
            return None
        from .polymesh import PolyMesh
        n = PolyMesh(case).n_cells
        U0, U1 = _read_field(os.path.join(tdir, "U"), 3, n), _read_field(os.path.join(nxt, "U"), 3, n)
        p0, p1 = _read_field(os.path.join(tdir, "p"), 1, n), _read_field(os.path.join(nxt, "p"), 1, n)
        us = float(np.linalg.norm(U0, axis=1).max()) or 1e-300
        ps = max(float(np.abs(p0).max()), us ** 2)
        return {"U": float(np.linalg.norm(U1 - U0, axis=1).max()) / us, "p": float(np.abs(p1 - p0).max()) / ps}

    def _meta(self, P, step, layout, pm, rs, status, diagnostics, combo, val, pname, vi, si, t_start) -> dict:
        from ... import units
        r = self.r
        meta = {"kind": step.kind, "times": [], "freqs": [], "neig": 0, "warnings": [],
                "physics": [ph.tag for ph in r.model.component.children if ph.kind == f"physics.{P.physics}"],
                "axi": False, "catalog": [], "var_domains": {}, "backend": "openfoam",
                "foam": {"app": P.app, "version": self.inst.version, "launcher": self.inst.kind, "cells": pm.n_cells,
                         "rho": P.rho, "nu": P.mu / P.rho if P.physics == "spf" else None,
                         "DT": P.DT if P.physics == "ht" else None, "iterations": rs.steps,
                         "simple_converged": rs.simple_converged,
                         "continuity": max((c for _s, c in rs.continuity), default=None)},
                "convergence": {"tol": P.tol, "maxit": P.maxit, "lin_tol": P.lin_tol, "lin_maxit": P.lin_maxit}}
        meta["selections"] = {n.tag: list(n.selection.entities) for n in r.model.root.walk()
                              if n.selection is not None and n.selection.named}
        for tag, (_lvl, ents) in getattr(layout.geo, "named", {}).items():
            meta["selections"].setdefault(tag, [int(e) for e in ents])
        if P.transient:
            tfac = units.unit_info(step.get("tunit", "s"))[0]
            meta["times"] = [t / tfac for t in P.out_times]
            meta["t0"] = P.t0
        meta.update({"eigen": [], "sweep_param": pname, "sweep_value": val,
                     "sweep_params": dict(combo) if combo else {}, "step_index": si, "sweep_index": vi,
                     "tunit": step.get("tunit", "s"), "returncode": rs.returncode, "elapsed": time.time() - t_start,
                     "status": status, "diagnostics": diagnostics, "params": dict(r.model.parameters(
                         dict(combo) if combo else None))})
        return meta
