"""Compute a study: (parametric sweep) x (study steps) -> mesh, SIF, ElmerSolver run, solution folder.

Blocking by design; the GUI runs it in a worker thread and receives events through
``on_event(kind, payload)``. ``stop()`` kills the running solver process.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

import numpy as np

from . import units
from .builder import study_steps, study_sweep
from .elmer import ElmerInstall, RunState, partition_command, result_files, solver_command, write_case
from .meshing import build_mesh
from .model import Model, Node
from .results import topology_to_npz, write_solution_meta
from .sif import build_sif, parse_list


def sweep_combinations(model: Model, sweep: Node | None) -> list[dict]:
    """Parameter overrides (SI) for each run of a Parametric Sweep: [{}] without a sweep.

    The first parameter comes from the node's own fields, further ones from its 'More parameters' table;
    'specified' pairs the i-th values (lists must be equally long), 'all' takes every combination."""
    if sweep is None or not str(sweep.get("pname", "")).strip():
        return [{}]
    known = dict(model.parameter_rows())
    rows = [(sweep.get("pname"), sweep.get("pvalues", ""), sweep.get("punit", ""))]
    rows += [(r.get("name", ""), r.get("values", ""), r.get("unit", "")) for r in (sweep.get("more") or [])
             if str(r.get("name", "")).strip()]
    lists = []
    for name, vals, unit in rows:
        name = str(name).strip()
        if name not in known:
            raise RuntimeError(f"Parametric Sweep: '{name}' is not a global parameter")
        v = parse_list(vals, model.parameters())
        if not v:
            raise RuntimeError(f"Parametric Sweep: empty value list for '{name}'")
        u = str(unit or "").strip()
        lists.append((name, [units.to_si(x, u) if u else x for x in v]))
    if sweep.get("stype", "specified") == "all":
        combos = [{}]
        for name, vals in lists:
            combos = [{**c, name: v} for c in combos for v in vals]
        return combos
    n = len(lists[0][1])
    if any(len(vals) != n for _nm, vals in lists):
        raise RuntimeError("Parametric Sweep: specified combinations need value lists of equal length")
    return [{name: vals[i] for name, vals in lists} for i in range(n)]


class StudyCancelled(Exception):
    pass


class StudyRunner:
    def __init__(self, model: Model, study: Node, workdir: str, inst: ElmerInstall | None, on_event=None,
                 mesh_cache=None):
        self.model = model
        self.study = study
        self.workdir = workdir
        self.inst = inst
        self.on_event = on_event or (lambda k, p: None)
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self.mesh_cache = mesh_cache
        self.folders: list[str] = []

    def emit(self, kind, payload=None):
        try:
            self.on_event(kind, payload)
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass

    def study_dir(self) -> str:
        return os.path.join(self.workdir, self.study.tag)

    def run(self) -> list[str]:
        try:
            return self._run()
        finally:
            if self._swept_named:       # named selections follow the geometry; put them back on the unswept one
                try:
                    from .geometry import build_geometry
                    build_geometry(self.model, display=False)
                except Exception:
                    pass

    @property
    def _swept_named(self) -> bool:
        sweep = study_sweep(self.study)
        return sweep is not None and any(n.selection is not None and n.selection.named for n in self.model.root.walk())

    def _run(self) -> list[str]:
        if self.inst is None:
            raise RuntimeError("Elmer was not found. Set the Elmer installation folder in Preferences.")
        steps = study_steps(self.study)
        if not steps:
            raise RuntimeError(f"{self.study.label} has no study steps")
        sweep = study_sweep(self.study)
        combos = sweep_combinations(self.model, sweep)
        pname = next(iter(combos[0]), None) if combos[0] else None
        values = [c.get(pname) if pname else None for c in combos]
        sdir = self.study_dir()
        if os.path.isdir(sdir):
            shutil.rmtree(sdir, ignore_errors=True)
        os.makedirs(sdir, exist_ok=True)
        total = len(values) * len(steps)
        job = 0
        t_start = time.time()
        reuse = sweep is not None and len(values) > 1 and sweep.get("reuse", "auto") != "off"
        previous: dict = {}                # study step index -> (restart file, mesh nodes) of the last sweep run
        for vi, val in enumerate(values):
            overrides = dict(combos[vi]) if combos[vi] else None
            if pname:
                desc = ", ".join(f"{k} = {v:g}" for k, v in combos[vi].items())
                self.emit("message", f"Parametric sweep: {desc}  ({vi + 1}/{len(values)})")
            self.emit("progress", (job / total, "Meshing"))
            if self._stop.is_set():
                raise StudyCancelled()
            if overrides is None and self.mesh_cache is not None:
                mesh = self.mesh_cache
            else:
                mesh = build_mesh(self.model, overrides)
                self.emit("mesh", mesh)
            st = mesh.stats
            self.emit("message", f"Mesh: {st['elements']} domain elements, {st['boundary_elements']} boundary elements, "
                                 f"{st['nodes']} nodes (order {st['order']}).")
            for si, step in enumerate(steps):
                if self._stop.is_set():
                    raise StudyCancelled()
                folder = os.path.join(sdir, f"sol_{vi}_{si}")
                cont, prev = None, previous.get(si)
                if reuse and step.kind == "study.stationary":
                    same = prev is not None and prev[1].shape == np.shape(mesh.nodes) and \
                        np.allclose(prev[1], mesh.nodes, rtol=0, atol=1e-12 * (np.ptp(prev[1]) or 1))
                    cont = "reuse" if same and os.path.exists(prev[0]) else "save"
                sif = build_sif(self.model, self.study, step, mesh.geometry, len(mesh.nodes), overrides,
                                continuation=cont)
                for w in sif.meta.get("warnings", []):
                    self.emit("warning", w)
                write_case(folder, mesh.to_elmer(), sif.text)
                if cont == "reuse":
                    shutil.copyfile(prev[0], os.path.join(folder, "mesh", "previous.result"))
                    self.emit("message", f"{step.label}: starting from the previous parameter's solution")
                topology_to_npz(mesh, os.path.join(folder, "mesh.npz"))
                self.emit("message", f"{step.label}: solving {', '.join(sif.meta['physics'])} "
                                     f"(case folder {folder})")
                rs = self._run_elmer(folder, step, job, total)
                meta = dict(sif.meta)
                meta["selections"] = {n.tag: list(n.selection.entities) for n in self.model.root.walk()
                                      if n.selection is not None and n.selection.named}
                meta.update({"eigen": rs.eigen, "sweep_param": pname, "sweep_value": val,
                             "sweep_params": dict(combos[vi]) if combos[vi] else {},
                             "step_index": si, "sweep_index": vi, "tunit": step.get("tunit", "s"),
                             "returncode": getattr(rs, "returncode", None), "elapsed": time.time() - t_start})
                if step.kind == "study.time":
                    tfac = units.unit_info(step.get("tunit", "s"))[0]
                    meta["times"] = [t / tfac for t in meta.get("times", [])]
                write_solution_meta(folder, meta)
                with open(os.path.join(folder, "solver.log"), "w", errors="replace") as f:
                    f.write("\n".join(rs.lines))
                if getattr(rs, "returncode", 0) != 0 or not result_files(folder):
                    said = [e for e in rs.errors if "ERROR::" in e]           # Elmer's message, not the backtrace
                    err = (said or rs.errors or [f"ElmerSolver exited with code {getattr(rs, 'returncode', '?')}"])[-1]
                    raise RuntimeError(f"{step.label}: {err}")
                self.folders.append(folder)
                if cont:
                    previous[si] = (os.path.join(folder, "mesh", "restart.result"), np.array(mesh.nodes, float))
                job += 1
                self.emit("progress", (job / total, "Done"))
                if rs.eigen:
                    import math
                    fs = ", ".join(f"{math.sqrt(abs(l)) / (2 * math.pi):.6g}" for l in rs.eigen)
                    self.emit("message", f"Eigenfrequencies (Hz): {fs}")
        self.emit("message", f"Solution time: {time.time() - t_start:.1f} s")
        return self.folders

    def _run_elmer(self, folder, step, job, total) -> RunState:
        nproc = int(self.study.get("np", 1) or 1)
        if nproc > 1 and not (self.inst.solver_mpi and self.inst.mpiexec):
            self.emit("warning", "MPI-enabled Elmer not found; running serially.")
            nproc = 1
        pc = partition_command(self.inst, nproc)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if pc:
            subprocess.run(pc, cwd=folder, env=self.inst.env(), capture_output=True, creationflags=flags)
        rs = RunState()
        self._proc = subprocess.Popen(solver_command(self.inst, nproc), cwd=folder, env=self.inst.env(),
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                                      creationflags=flags, bufsize=1)
        for line in self._proc.stdout:
            line = line.rstrip("\n")
            ev = rs.feed(line)
            self.emit("log", line)
            if ev == "conv":
                self.emit("conv", rs.conv[-1])
            elif ev == "step" and rs.step[1]:
                frac = (job + rs.step[0] / rs.step[1]) / total
                self.emit("progress", (frac, f"{step.label}: step {rs.step[0]}/{rs.step[1]}"))
            elif ev == "error":
                self.emit("error_line", line)
            if self._stop.is_set():
                break
        self._proc.wait()
        rs.returncode = self._proc.returncode
        self._proc = None
        if self._stop.is_set():
            raise StudyCancelled()
        return rs
