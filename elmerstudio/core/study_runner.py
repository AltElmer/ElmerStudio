"""Compute a study: (parametric sweep) x (study steps) -> mesh, SIF, ElmerSolver run, solution folder.

Blocking by design; the GUI runs it in a worker thread and receives events through
``on_event(kind, payload)``. ``stop()`` kills the running solver process.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import threading
import time

import numpy as np

from . import units
from .backends import study_backend
from .builder import step_components, study_steps, study_sweep
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


def component_dir(model: Model, study_dir: str, comp_tag: str) -> str:
    """Where a component's solutions of a study go: the study folder for the first component (as files written
    before several components existed), a subfolder named after the component for the others."""
    comps = model.components()
    return study_dir if not comps or comps[0].tag == comp_tag else os.path.join(study_dir, comp_tag)


_XREF = re.compile(r"(?<![\w.])(comp\w*)\.(\w+)(\()?")


def cross_references(model: Model, comp: Node) -> list[tuple[str, str, str, str]]:
    """References from *comp*'s settings to other components: [(text, component tag, name, argument)]; the
    argument is None for a plain name (comp1.T), which equations cannot use."""
    tags = {c.tag for c in model.components()}
    out, seen = [], set()

    def scan(text: str):
        for m in _XREF.finditer(text):
            ctag = m.group(1)
            if ctag not in tags or ctag == comp.tag:
                continue
            if m.group(3):                                  # an operator call: take its parenthesized argument
                depth, i = 1, m.end()
                while i < len(text) and depth:
                    depth += text[i] == "("
                    depth -= text[i] == ")"
                    i += 1
                full, arg = text[m.start():i], text[m.end():i - 1]
            else:
                full, arg = m.group(0), None
            if full not in seen:
                seen.add(full)
                out.append((full, ctag, m.group(2), arg))

    def walk(v):
        if isinstance(v, str):
            if "comp" in v:
                scan(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
    for n in comp.walk():
        walk(n.props)
    return out


def component_order(model: Model) -> list[Node]:
    """Components in solving order: one whose expressions use another component's results comes after it.
    A circular use is an error naming the components."""
    comps = model.components()
    deps = {c.tag: {t for _f, t, _n, _a in cross_references(model, c)} for c in comps}
    order, state = [], {}

    def visit(c, path):
        if state.get(c.tag) == 2:
            return
        if state.get(c.tag) == 1:
            cyc = path[path.index(c.tag):] + [c.tag]
            raise RuntimeError("Components use each other's results in a circle: " + " -> ".join(cyc)
                               + ". One-way couplings between components only.")
        state[c.tag] = 1
        for t in sorted(deps[c.tag]):
            visit(next(x for x in comps if x.tag == t), path + [c.tag])
        state[c.tag] = 2
        order.append(c)
    for c in comps:
        visit(c, [])
    return order


def cross_values(model: Model, comp: Node, solved: dict, overrides=None) -> dict:
    """Values of the other components' coupling operators *comp* uses (comp1.aveop1(T)), evaluated on their
    solutions of this run. Returned as {"__xrefs__": {text: value}} for the Elmer input of *comp*."""
    refs = cross_references(model, comp)
    if not refs:
        return {}
    from .nodes import bind_results
    from .results import Solution
    vals = {}
    for full, ctag, name, arg in refs:
        other = next(c for c in model.components() if c.tag == ctag)
        if arg is None:
            raise RuntimeError(f"{comp.label} uses {full}: a field of {other.label} cannot enter the equations of "
                               f"another component; use a coupling operator of {other.label}, e.g. "
                               f"{ctag}.aveop1(...) (Definitions > Component Couplings)")
        if ctag not in solved:
            raise RuntimeError(f"{comp.label} uses {full}, but {other.label} is not solved before it in this study "
                               "(solve both components in the study, or in an earlier step)")
        with model.scoped(other):
            sol = Solution(solved[ctag])
            sol = bind_results(model, sol)
            if name not in getattr(sol, "operators", {}):
                raise RuntimeError(f"{comp.label} uses {full}: {other.label} has no coupling operator '{name}'")
            v = sol.evaluate(f"{name}({arg})", sol.frame_index("last"))
        vals[full] = float(np.ravel(v)[0])
    return {"__xrefs__": vals}


def initial_solution(model: Model, workdir: str, step: Node, comp: Node, mesh) -> str | None:
    """The restart file of the study a step takes its initial values from (Values of Dependent Variables >
    Solution), for *comp*: its last solution, which must be on the same mesh. None when the step has no such
    setting."""
    tag = str(step.get("initsol", "") or "")
    if not tag:
        return None
    src = model.by_tag(tag)
    if src is None:
        raise RuntimeError(f"{step.label}: the study it takes initial values from no longer exists")
    folders = sorted(glob.glob(os.path.join(component_dir(model, os.path.join(workdir, tag), comp.tag), "sol_*_*")),
                     key=lambda p: tuple(int(x) for x in re.findall(r"\d+", os.path.basename(p))))
    folders = [f for f in folders if os.path.isfile(os.path.join(f, "mesh", "restart.result"))]
    if not folders:
        raise RuntimeError(f"{step.label} starts from {src.label}, which has no solution of {comp.label}: "
                           f"compute {src.label} first")
    last = folders[-1]
    try:
        old = np.load(os.path.join(last, "mesh.npz"))["nodes"]
        same = old.shape == np.shape(mesh.nodes) and np.allclose(old, mesh.nodes, rtol=0,
                                                                  atol=1e-9 * (np.ptp(old) or 1))
    except Exception:  # noqa: BLE001
        same = False
    if not same:
        raise RuntimeError(f"{step.label} starts from {src.label}, whose solution is on another mesh: compute "
                           f"{src.label} again with the current mesh")
    return os.path.join(last, "mesh", "restart.result")


def study_order(model: Model) -> list[Node]:
    """Studies in the order their dependencies need: one that starts from another study's solution after it.
    A circle is an error."""
    sts = model.studies()
    deps = {st.tag: {str(s.get("initsol", "") or "") for s in st.children} - {"", st.tag} for st in sts}
    order, state = [], {}

    def visit(st, path):
        if state.get(st.tag) == 2:
            return
        if state.get(st.tag) == 1:
            raise RuntimeError("Studies start from each other's solutions in a circle: "
                               + " -> ".join(path[path.index(st.tag):] + [st.tag]))
        state[st.tag] = 1
        for t in sorted(deps[st.tag]):
            src = model.by_tag(t)
            if src is not None:
                visit(src, path + [st.tag])
        state[st.tag] = 2
        order.append(st)
    for st in sts:
        visit(st, [])
    return order


class StudyCancelled(Exception):
    pass


class CurrentControl:
    """An Electrode Surface with a total current (COMSOL: Electrode phase potential condition, Total current): the
    study adjusts the electrode's potential phis_<tag> by secant steps until the current entering the electrolyte
    through it, cd.<tag>.Itot (the sum of the nodal reactions), is the set one.  The current rises with the potential
    (anodic positive).  One such electrode per component, in stationary studies."""
    MAXIT, STEP, MAXSTEP = 40, 0.05, 0.5

    def __init__(self, model, comp, step, overrides):
        es = [f for ph in comp.children if ph.kind == "physics.cd" and ph.enabled for f in ph.children
              if f.kind == "cd.es" and f.enabled and f.get("control", "potential") == "current"]
        self.node = es[0] if es else None
        if self.node is None:
            return
        if len(es) > 1:
            raise RuntimeError(f"{es[1].label}: one electrode per component can have a total current; give the "
                               "others an electric potential")
        if step.kind != "study.stationary":
            raise RuntimeError(f"{self.node.label}: a total-current electrode needs a Stationary study step")
        params = model.parameters({k: v for k, v in (overrides or {}).items() if not isinstance(v, dict)})
        ev = lambda e: float(np.real(units.evaluate(str(e), params)))  # noqa: E731
        self.target = ev(self.node.get("Iset", "0[A]"))
        self.phi = ev(self.node.get("phis", "0[V]"))
        self.param = f"phis_{self.node.tag}"
        self.tried: list[tuple[float, float]] = []
        self.model = model

    def overrides(self, base):
        return {**(base or {}), self.param: self.phi}

    def measure(self, folder, mesh) -> float:
        from .nodes import bind_results
        from .results import Solution
        sol = bind_results(self.model, Solution(folder), mesh.geometry)
        return float(np.nanmean(sol.evaluate(f"cd.{self.node.tag}.Itot", sol.frame_index("last"))))

    def update(self, current: float) -> bool:
        """Record the current at the present potential; True when it is the set one, else take the next step."""
        self.tried.append((self.phi, current))
        f = current - self.target
        if abs(f) <= max(1e-6 * abs(self.target), 1e-12 * max(abs(current), 1.0)):
            return True
        if len(self.tried) >= self.MAXIT:
            raise RuntimeError(f"{self.node.label}: the total current {self.target:.6g} A was not reached in "
                               f"{self.MAXIT} solves (last {current:.6g} A at {self.phi:.6g} V)")
        step = -self.STEP if f > 0 else self.STEP
        if len(self.tried) > 1:
            (p0, i0), (p1, i1) = self.tried[-2], self.tried[-1]
            slope = (i1 - i0) / (p1 - p0) if p1 != p0 else 0.0
            if slope > 0 and np.isfinite(slope):
                step = -f / slope
        self.phi += float(np.clip(step, -self.MAXSTEP, self.MAXSTEP))
        return False


class StudyNotConverged(RuntimeError):
    """Iterations reached their maximum without meeting the tolerance (the solution folder is kept, its meta.json
    says status 'not converged' with the diagnostics)."""


def judge_run(rs, sif_meta, folder, label) -> tuple[str, list[str]]:
    """The status of one ElmerSolver run: ('failed', [message]) when Elmer stopped or wrote no result,
    ('not converged', [problems]) when nonlinear or coupling iterations ran out, else ('converged', [])."""
    from .elmer import convergence_problems
    if getattr(rs, "returncode", 0) != 0 or not result_files(folder):
        said = [e for e in rs.errors if "ERROR::" in e or "Numerical Error" in e]   # Elmer's message
        err = (said or rs.errors or [f"ElmerSolver exited with code {getattr(rs, 'returncode', '?')}"])[-1]
        return "failed", [f"{label}: {err}"]
    problems = convergence_problems(rs.conv, sif_meta.get("convergence"))
    return ("not converged", [f"{label}: {p}" for p in problems]) if problems else ("converged", [])


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
        self._killer = None                # a backend's own process-tree kill (OpenFOAM runs under bash)

    def emit(self, kind, payload=None):
        try:
            self.on_event(kind, payload)
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        if self._killer is not None:
            try:
                self._killer()
            except Exception:
                pass
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
                from .geometry import build_geometry
                for comp in self.model.components():
                    try:
                        with self.model.scoped(comp):
                            build_geometry(self.model, display=False)
                    except Exception:
                        pass

    @property
    def _swept_named(self) -> bool:
        sweep = study_sweep(self.study)
        return sweep is not None and any(n.selection is not None and n.selection.named for n in self.model.root.walk())

    def comp_dir(self, comp: Node) -> str:
        return component_dir(self.model, self.study_dir(), comp.tag)

    def _run(self) -> list[str]:
        if study_backend(self.study) == "openfoam":
            from .backends.openfoam.runner import FoamStudy
            return FoamStudy(self).run()
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
        # the components each step solves, ordered so that a component whose expressions use another's results
        # (comp1.aveop1(T)) is solved after it
        order = component_order(self.model)
        plan = [[c for c in order if c in step_components(self.model, step)] for step in steps]
        total = max(1, len(values) * sum(len(p) for p in plan))
        job = 0
        t_start = time.time()
        reuse = sweep is not None and len(values) > 1 and sweep.get("reuse", "auto") != "off"
        previous: dict = {}         # (component, step index) -> (restart file, mesh nodes) of the last sweep run
        cache = self.mesh_cache if isinstance(self.mesh_cache, dict) else (
            {self.model.components()[0].tag: self.mesh_cache} if self.mesh_cache is not None and self.model.components()
            else {})
        for vi, val in enumerate(values):
            overrides = dict(combos[vi]) if combos[vi] else None
            if pname:
                desc = ", ".join(f"{k} = {v:g}" for k, v in combos[vi].items())
                self.emit("message", f"Parametric sweep: {desc}  ({vi + 1}/{len(values)})")
                self.emit("sweep", f"{desc} ({vi + 1}/{len(values)})")
            meshes: dict = {}
            solved: dict = {}                      # component tag -> its latest solution folder in this run
            for si, step in enumerate(steps):
                if self._stop.is_set():
                    raise StudyCancelled()
                for comp in plan[si]:
                    with self.model.scoped(comp):
                        if comp.tag not in meshes:
                            self.emit("progress", (job / total, f"Meshing {comp.label}" if len(order) > 1 else "Meshing"))
                            if overrides is None and comp.tag in cache:
                                mesh = cache[comp.tag]
                            else:
                                mesh = build_mesh(self.model, overrides, inst=self.inst)
                                mesh.comp_tag = comp.tag
                                self.emit("mesh", mesh)
                                for m in mesh.messages:
                                    self.emit("message", m)
                            meshes[comp.tag] = mesh
                            st = mesh.stats
                            self.emit("message", (f"{comp.label}: " if len(order) > 1 else "")
                                      + f"Mesh: {st['elements']} domain elements, {st['boundary_elements']} boundary "
                                      f"elements, {st['nodes']} nodes (order {st['order']}).")
                        mesh = meshes[comp.tag]
                        xvals = cross_values(self.model, comp, solved, overrides)
                        step_overrides = {**(overrides or {}), **xvals} or None
                        if xvals:
                            self.emit("message", f"{comp.label}: " + ", ".join(
                                f"{k} = {v:.6g}" for k, v in xvals["__xrefs__"].items()))
                        folder = os.path.join(self.comp_dir(comp), f"sol_{vi}_{si}")
                        from .deforming import DeformingRun, deforming_setup
                        dsetup = deforming_setup(self.model, step)
                        if dsetup is not None:                     # deforming electrodes: one solve per time step
                            os.makedirs(folder, exist_ok=True)
                            DeformingRun(self, dsetup, step, mesh, folder, step_overrides, combos[vi], (pname, val)).run()
                            self.folders.append(folder)
                            job += 1
                            self.emit("progress", (job / total, "Done"))
                            solved[comp.tag] = folder
                            continue
                        cont, prev = None, previous.get((comp.tag, si))
                        if reuse and step.kind == "study.stationary":
                            same = prev is not None and prev[1].shape == np.shape(mesh.nodes) and \
                                np.allclose(prev[1], mesh.nodes, rtol=0, atol=1e-12 * (np.ptp(prev[1]) or 1))
                            cont = "reuse" if same and os.path.exists(prev[0]) else "save"
                        ctrl = CurrentControl(self.model, comp, step, step_overrides)
                        while True:                    # once, or until a total-current electrode carries its current
                            if ctrl.node is not None:
                                step_overrides = ctrl.overrides(step_overrides)
                            init = initial_solution(self.model, self.workdir, step, comp, mesh)
                            sif = build_sif(self.model, self.study, step, mesh.geometry, len(mesh.nodes), step_overrides,
                                            continuation=cont, save_restart=True, initial_restart=init is not None)
                            for w in sif.meta.get("warnings", []):
                                self.emit("warning", w)
                            write_case(folder, mesh.to_elmer(), sif.text)
                            if init is not None:
                                shutil.copyfile(init, os.path.join(folder, "mesh", "initial.result"))
                                self.emit("message", f"{step.label}: initial values from "
                                                     f"{self.model.by_tag(step.get('initsol')).label}")
                            if cont == "reuse":
                                shutil.copyfile(prev[0], os.path.join(folder, "mesh", "previous.result"))
                                self.emit("message", f"{step.label}: starting from the previous parameter's solution")
                            topology_to_npz(mesh, os.path.join(folder, "mesh.npz"))
                            self.emit("message", f"{step.label}: solving {', '.join(sif.meta['physics'])} "
                                                 f"(case folder {folder})")
                            if step.kind == "study.time":           # probes follow the output steps as they are written
                                live = dict(sif.meta, comp=comp.tag, tunit=step.get("tunit", "s"))
                                tfac = units.unit_info(live["tunit"])[0]
                                live["times"] = [t / tfac for t in live.get("times", [])]
                                self.emit("live", (folder, live))
                            rs = self._run_elmer(folder, step, job, total)
                            meta = dict(sif.meta)
                            meta["selections"] = {n.tag: list(n.selection.entities) for n in self.model.root.walk()
                                                  if n.selection is not None and n.selection.named}
                            # the geometry selections themselves, so that results made later (a Line Graph on a Box Selection)
                            # resolve them as this run's geometry did
                            for tag, (_lvl, ents) in getattr(mesh.geometry, "named", {}).items():
                                meta["selections"].setdefault(tag, [int(e) for e in ents])
                            meta.update({"comp": comp.tag, "xrefs": xvals.get("__xrefs__", {}),
                                         "eigen": rs.eigen, "sweep_param": pname, "sweep_value": val,
                                         "sweep_params": dict(combos[vi]) if combos[vi] else {},
                                         "step_index": si, "sweep_index": vi, "tunit": step.get("tunit", "s"),
                                         "returncode": getattr(rs, "returncode", None), "elapsed": time.time() - t_start})
                            if step.kind == "study.time":
                                tfac = units.unit_info(step.get("tunit", "s"))[0]
                                meta["times"] = [t / tfac for t in meta.get("times", [])]
                            status, diagnostics = judge_run(rs, sif.meta, folder, step.label)
                            meta.update({"status": status, "diagnostics": diagnostics,
                                         # provenance: the inputs this solution was computed from (results evaluate with these
                                         # parameter values, not later edits of the model)
                                         "params": dict(self.model.parameters(overrides))})
                            if ctrl.node is not None:            # the potential this solution was computed with
                                meta["params"][ctrl.param] = ctrl.phi
                            write_solution_meta(folder, meta)
                            with open(os.path.join(folder, "model.json"), "w") as f:
                                f.write(self.model.to_json())
                            with open(os.path.join(folder, "solver.log"), "w", errors="replace") as f:
                                f.write("\n".join(rs.lines))
                            if status == "failed":
                                raise RuntimeError(diagnostics[0])
                            if status == "not converged":
                                if self.study.get("nonconv", "stop") != "keep":
                                    raise StudyNotConverged("; ".join(diagnostics) + ". Raise the maximum number of iterations "
                                                            "or the damping in the study's Solver Configurations, or choose to "
                                                            "keep unconverged solutions in the study settings.")
                                for d in diagnostics:
                                    self.emit("warning", f"NOT CONVERGED: {d}")
                            if ctrl.node is None:
                                break
                            current = ctrl.measure(folder, mesh)
                            self.emit("message", f"{ctrl.node.label}: phis = {ctrl.phi:.6g} V gives I = {current:.6g} A "
                                                  f"(set: {ctrl.target:.6g} A)")
                            if ctrl.update(current):
                                break
                        self.folders.append(folder)
                        if cont:
                            previous[(comp.tag, si)] = (os.path.join(folder, "mesh", "restart.result"), np.array(mesh.nodes, float))
                        job += 1
                        self.emit("progress", (job / total, "Done"))
                        if rs.eigen:
                            import math
                            fs = ", ".join(f"{math.sqrt(abs(l)) / (2 * math.pi):.6g}" for l in rs.eigen)
                            self.emit("message", f"Eigenfrequencies (Hz): {fs}")
                        solved[comp.tag] = folder
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
