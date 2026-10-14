"""Time Dependent studies with deforming electrodes (COMSOL: Deforming Domain + Deforming Electrode Surface).

At every time step the current distribution is solved, as a stationary problem, on the present shape; the
electrodes with *Deforming electrode surface* checked then move along their normal at the dissolution velocity

    v_n = eps * M * i_loc / (n F rho_m)        (anodic current dissolves the metal, cathodic current deposits)

with i_loc the conservative local current density (cd.iloc, from Elmer's reaction loads), and the mesh of the
Deforming Domain follows (mesh_motion.MeshMover).  The connectivity never changes, so each step restarts Elmer from
the previous step's solution, and every output time is one frame of the solution with its own node coordinates.
Second-order (Heun) time integration is the default: a predictor step on the advanced shape, then the average of
the two normal velocities.
"""
from __future__ import annotations

import copy
import os
import re
import shutil
import time

import numpy as np

from . import units
from .elmer import result_files, write_case
from .mesh_motion import MeshMotionError, MeshMover, shape_quality, signed_measures
from .model import selection_entities
from .results import Solution, Topology, topology_to_npz, write_solution_meta
from .sif import build_sif, parse_list


def deforming_setup(model, step):
    """{'ddoms', 'movers', 'sym', 'physics'} when *step* is a Time Dependent step of a model with a Deforming Domain
    (Component > Definitions), else None.  Movers: Electrode Surfaces with 'Deforming electrode surface' checked
    (in solved Current Distribution interfaces) and Prescribed Normal Mesh Velocity boundaries."""
    comp = model.component
    if step.kind != "study.time" or comp is None:
        return None
    defs = comp.child("definitions")
    ddoms = [f for f in (defs.children if defs else []) if f.kind == "dg.ddom" and f.enabled]
    if not ddoms:
        return None
    sel = step.props.get("physics", {}) or {}
    solved = [ph for ph in comp.children if ph.kind.startswith("physics.") and ph.enabled and sel.get(ph.tag, True)]
    movers = [f for ph in solved if ph.kind == "physics.cd" for f in ph.children
              if f.kind == "cd.es" and f.enabled and f.get("moving")]
    movers += [f for f in defs.children if f.kind == "dg.pnmv" and f.enabled]
    if not movers:
        raise RuntimeError(f"{ddoms[0].label}: nothing moves -- check 'Deforming electrode surface' on an Electrode "
                           "Surface, or add a Prescribed Normal Mesh Velocity")
    sym = [f for ph in solved for f in ph.children if f.kind.endswith(".sym") and f.enabled]
    return {"ddoms": ddoms, "movers": movers, "sym": sym, "physics": [ph.tag for ph in solved]}


def output_times(model, step, overrides=None):
    params = model.parameters(overrides)
    tfac = units.unit_info(step.get("tunit", "s"))[0]
    times = [t * tfac for t in parse_list(step.get("times", "range(0,0.1,1)"), params)]
    if len(times) < 2:
        raise RuntimeError("Time Dependent: specify at least two output times, e.g. range(0,0.1,1)")
    if np.any(np.diff(times) <= 0):                     # the steps between output times are adaptive (CFL)
        raise RuntimeError("Time Dependent: output times must be increasing")
    return times, tfac


class DeformingRun:
    """Runs one Time Dependent step with moving electrodes into *folder*; *runner* is the StudyRunner (Elmer, events)."""

    def __init__(self, runner, setup, step, mesh, folder, overrides=None, combo=None, sweep=(None, None)):
        self.r = runner
        self.model = runner.model
        self.study = runner.study
        self.step = step
        self.setup = setup
        self.mesh = copy.copy(mesh)                       # never move the GUI's own mesh
        self.mesh.nodes = np.array(mesh.nodes, float)
        self.folder = folder
        self.overrides = overrides
        self.combo = combo or {}
        self.sweep = sweep
        self.geo = mesh.geometry
        self.work = os.path.join(folder, "run")
        self.prev_restart = None
        self.nsolves = 0

    # ------------------------------------------------------------------ helpers
    def _selections(self):
        out = {n.tag: list(n.selection.entities) for n in self.model.root.walk()
               if n.selection is not None and n.selection.named}
        for tag, (_lvl, ents) in getattr(self.geo, "named", {}).items():     # the Box Selections themselves
            out.setdefault(tag, [int(e) for e in ents])
        return out

    def _entities(self, nodes, level):
        meta = {"selections": self._selections()}
        avail = self.geo.numbers.get(level, [])
        out = set()
        for f in nodes:
            if f.selection is None:
                continue
            if f.selection.all and not f.selection.named:
                out.update(avail)
            else:
                out.update(selection_entities(f, meta))
        return sorted(out)

    def _solve(self, nodes, t, label, full=True):
        from .nodes import bind_results
        r = self.r
        if r._stop.is_set():
            from .study_runner import StudyCancelled
            raise StudyCancelled()
        self.mesh.nodes = nodes
        cont = "reuse" if self.prev_restart else "save"
        sif = build_sif(self.model, self.study, self.step, self.geo, len(nodes), self.overrides, continuation=cont,
                        quasi_static_time=t, skip_post=not full)
        write_case(self.work, self.mesh.to_elmer(), sif.text)
        if self.prev_restart:
            shutil.copyfile(self.prev_restart, os.path.join(self.work, "mesh", "previous.result"))
        topology_to_npz(self.mesh, os.path.join(self.work, "mesh.npz"))
        rs = r._run_elmer(self.work, self.step, 0, 1)
        meta = dict(sif.meta)
        meta.update({"kind": "study.stationary", "selections": self._selections(), "sweep_param": self.sweep[0],
                     "sweep_value": self.sweep[1], "sweep_params": dict(self.combo)})
        write_solution_meta(self.work, meta)
        if getattr(rs, "returncode", 0) != 0 or not result_files(self.work):
            with open(os.path.join(self.folder, "solver.log"), "w", errors="replace") as f:
                f.write("\n".join(rs.lines))
            said = [e for e in rs.errors if "ERROR::" in e]
            err = (said or rs.errors or [f"ElmerSolver exited with code {getattr(rs, 'returncode', '?')}"])[-1]
            raise RuntimeError(f"{self.step.label} ({label}): {err}")
        keep = os.path.join(self.folder, "restart.result")
        src = os.path.join(self.work, "mesh", "restart.result")
        if os.path.exists(src):
            shutil.copyfile(src, keep)
            self.prev_restart = keep
        self.nsolves += 1
        self.lines = rs.lines
        sol = Solution(self.work)
        bind_results(self.model, sol, self.geo)
        return sol

    def _speed(self, sol):
        """Outward normal velocity (m/s; positive = the domain grows, e.g. metal dissolving) at the moving nodes."""
        i = sol.frame_index("last")
        v = np.zeros(len(sol.topo.nodes))
        for f in self.setup["movers"]:
            if f.kind == "cd.es":
                eff = re.sub(r"(?<![\w.])iloc(?![\w(])", "cd.iloc", str(f.get("eff", "1") or "1"))
                expr = f"(cd.{f.tag}.vdis)*({eff})"
            else:
                expr = str(f.get("vn", "0") or "0")
            vals = np.asarray(sol.evaluate(expr, i), float)
            if vals.ndim == 0 or vals.size == 1:
                vals = np.full(len(v), float(np.ravel(vals)[0]))
            for b in self._entities([f], "boundary"):
                for _gt, conn in sol.topo.bnd.get(b, []):
                    ids = np.unique(np.asarray(conn))
                    v[ids] = np.nan_to_num(vals[ids])
        return v

    # ------------------------------------------------------------------ mesh bookkeeping
    def _new_mover(self):
        dd = self.setup["ddoms"][0]
        try:
            self.mover = MeshMover(Topology(mesh_result=self.mesh), self.doms, self.moving,
                                   slide_straight=bool(dd.get("slide", True)), sliding_boundaries=self.sym, axi=self.axi)
        except MeshMotionError as exc:
            raise RuntimeError(f"{dd.label}: {exc}") from None
        nodes = np.asarray(self.mesh.nodes, float)
        self.meas_ref = signed_measures(nodes, self.mover.simp, self.mover.sdim)
        self.q_ref = shape_quality(nodes, self.mover.simp, self.mover.sdim)
        # spacing of the moving boundaries' corner nodes (CFL limit) and their grading (kept when remeshing)
        from .remesh import boundary_chains, chain_profile
        self.chains = {(b, k): ch for b in self.moving for k, ch in enumerate(boundary_chains(self.mesh.bnd.get(b, [])))}
        if not hasattr(self, "profiles"):
            self.profiles = {key: chain_profile(nodes[ch, :2]) for key, ch in self.chains.items()}

    def _quality(self, nodes):
        """Worst element, relative to the reference mesh: shape quality or size, whichever dropped more (<= 0: folded)."""
        m = signed_measures(nodes, self.mover.simp, self.mover.sdim)
        q = shape_quality(nodes, self.mover.simp, self.mover.sdim)
        size = m / np.where(self.meas_ref != 0, self.meas_ref, 1.0)
        shape = q / np.where(self.q_ref != 0, self.q_ref, 1.0)     # signed: sub-elements may be either orientation
        return float(min(np.min(size), np.min(shape)))

    def _remesh(self, nodes, t):
        from .remesh import RemeshError, remesh_2d
        if self.mover.sdim != 2:
            raise RuntimeError(f"{self.step.label}: remeshing is available for 2D and 2D axisymmetric models")
        try:
            new = remesh_2d(self.mesh, nodes, set(self.moving), self.profiles)
        except RemeshError as exc:
            raise RuntimeError(f"{self.step.label}: {exc} at t = {t:.6g} s") from None
        self.mesh = new
        self.mesh_index += 1
        topology_to_npz(self.mesh, os.path.join(self.folder, f"mesh_{self.mesh_index}.npz"))
        self.prev_restart = None                              # a new mesh: start from the initial values again
        self._new_mover()
        self.remeshes += 1
        self.r.emit("message", f"{self.step.label}: remeshed at t = {t / self.tfac:.6g} {self.step.get('tunit', 's')} "
                               f"({len(new.nodes)} nodes)")
        return np.asarray(new.nodes, float)

    def _cfl_dt(self, nodes, v):
        """Largest step that moves no moving node by more than cfl x its boundary spacing (s)."""
        cfl = self.cfl
        best = np.inf
        for ch in self.chains.values():
            P = nodes[ch, :2]
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1) * self.mesh.scale
            h = np.minimum(np.r_[seg[0], seg], np.r_[seg, seg[-1]])
            vv = np.abs(v[ch])
            on = vv > 0
            if on.any():
                best = min(best, float(np.min(cfl * h[on] / vv[on])))
        return best

    def _step(self, nodes, sol, t, dt, heun):
        """(new nodes, solution of the predictor or None); raises _Fold when the mesh would fold."""
        mover = self.mover
        v0 = self._speed(sol)
        d0 = mover.boundary_displacement(nodes, v0, dt)
        if heun:
            pred = nodes + mover.extend(nodes, d0)
            if self._quality(pred) <= 0:
                raise _Fold()
            sol1 = self._solve(pred, t + dt, f"t = {(t + dt) / self.tfac:.6g}, predictor", full=False)
            d1 = mover.boundary_displacement(pred, self._speed(sol1), dt)
            dbnd = 0.5 * (d0 + d1)
        else:
            dbnd = d0
        new = nodes + mover.extend(nodes, dbnd)
        if self._quality(new) <= 0:
            raise _Fold()
        return new

    # ------------------------------------------------------------------ run
    def run(self):
        from .model import is_axisymmetric
        r = self.r
        times, tfac = output_times(self.model, self.step, self.overrides)
        self.tfac = tfac
        sub = max(int(self.step.get("substeps", 1) or 1), 1)
        dt_out = times[1] - times[0]
        dd = self.setup["ddoms"][0]
        heun = dd.get("integrator", "heun") != "euler"
        remesh_on = bool(dd.get("remesh", False))
        try:
            qmin = float(units.evaluate(str(dd.get("qmin", "0.2") or "0.2")))
        except Exception:
            qmin = 0.2
        try:
            self.cfl = float(units.evaluate(str(dd.get("cfl", "0.3") or "0.3")))
        except Exception:
            self.cfl = 0.3
        self.doms = self._entities(self.setup["ddoms"], "domain")
        self.moving = self._entities(self.setup["movers"], "boundary")
        self.sym = self._entities(self.setup["sym"], "boundary")
        self.axi = bool(is_axisymmetric(self.model.component))
        self.mesh_index, self.remeshes = 0, 0
        self._new_mover()
        os.makedirs(os.path.join(self.folder, "results"), exist_ok=True)
        topology_to_npz(self.mesh, os.path.join(self.folder, "mesh.npz"))
        nodes = np.array(self.mesh.nodes, float)
        frames, saved_times, frame_mesh = {}, [], []
        t = times[0]
        t_start = time.time()
        nsteps = 0
        tunit = self.step.get("tunit", "s")

        def save(sol):
            n_out = len(saved_times) + 1
            src = sorted(result_files(self.work))[-1]
            shutil.copyfile(src, os.path.join(self.folder, "results", f"case_t{n_out:04d}.vtu"))
            frames[f"f{n_out - 1}"] = nodes.copy()
            saved_times.append(t / tfac)
            frame_mesh.append(self.mesh_index)

        sol = self._solve(nodes, t, f"t = {t / tfac:.6g}")
        save(sol)
        for j in range(1, len(times)):
            t_end = times[j]
            while t < t_end - 1e-12 * max(abs(t_end), dt_out):
                if remesh_on and self._quality(nodes) < qmin:
                    nodes = self._remesh(nodes, t)
                    sol = self._solve(nodes, t, f"t = {t / tfac:.6g}, remeshed", full=False)
                dt = min(t_end - t, dt_out / sub)
                dt = min(dt, self._cfl_dt(nodes, self._speed(sol)))
                n = int(np.ceil((t_end - t) / dt - 1e-9))
                dt = (t_end - t) / max(n, 1)
                for attempt in range(8):
                    try:
                        new = self._step(nodes, sol, t, dt, heun)
                        break
                    except _Fold:
                        if remesh_on and attempt == 0 and self.mover.sdim == 2:
                            nodes = self._remesh(nodes, t)
                            sol = self._solve(nodes, t, f"t = {t / tfac:.6g}, remeshed", full=False)
                        else:
                            dt /= 2
                else:
                    raise RuntimeError(f"{self.step.label}: the moving mesh folded over at t = {t / tfac:.6g} {tunit}; "
                                       "enable remeshing on the Deforming Domain")
                nodes = new
                t += dt
                nsteps += 1
                r.emit("progress", (min((t - times[0]) / (times[-1] - times[0]), 0.999),
                                    f"{self.step.label}: t = {t / tfac:.6g} {tunit}"))
                sol = self._solve(nodes, t, f"t = {t / tfac:.6g}", full=t >= t_end - 1e-12 * max(abs(t_end), dt_out))
            t = t_end
            save(sol)
        np.savez_compressed(os.path.join(self.folder, "frames.npz"), **frames)
        meta = {"kind": "study.time", "times": saved_times, "t0": saved_times[0], "tunit": tunit,
                "moving_mesh": True, "frame_mesh": frame_mesh, "physics": list(self.setup["physics"]), "axi": self.axi,
                "selections": self._selections(), "sweep_param": self.sweep[0], "sweep_value": self.sweep[1],
                "sweep_params": dict(self.combo), "solves": self.nsolves, "steps": nsteps, "remeshes": self.remeshes,
                "elapsed": time.time() - t_start, "integrator": "heun" if heun else "euler"}
        write_solution_meta(self.folder, meta)
        with open(os.path.join(self.folder, "solver.log"), "w", errors="replace") as f:
            f.write("\n".join(getattr(self, "lines", [])))
        shutil.rmtree(self.work, ignore_errors=True)
        r.emit("message", f"{self.step.label}: {nsteps} time steps, {self.nsolves} current-distribution solves "
                          f"({'Heun' if heun else 'explicit Euler'})"
                          + (f", {self.remeshes} remeshings" if self.remeshes else ""))
        return meta


class _Fold(Exception):
    pass
