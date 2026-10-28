"""Model -> OpenFOAM case: what is refused, how each Studio feature maps onto a boundary condition, and the files.

Studio is SI throughout; OpenFOAM's incompressible solvers use the kinematic pressure p/rho and nu = mu/rho.
Pressures are divided by rho on the way in (boundary values, initial values) and the pressure result is multiplied
by rho on the way out (see results.py).  Heat conduction is written for laplacianFoam as
dT/dt = DT lap(T) + Q/(rho Cp) with DT = k/(rho Cp).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ... import units
from ...builder import physics_nodes, step_components, study_steps
from ...model import Model, Node, is_axisymmetric
from ...physics import COUPLINGS, FEATURES, physics_def
from ...sif import SifContext, SifError, _entities, _features, parse_list, time_groups
from . import dicts
from .blocks import BlockLayout

SUPPORTED_PHYSICS = {"spf": "Laminar Flow", "ht": "Heat Transfer in Solids"}
SUPPORTED_FEATURES = {
    "spf": {"spf.fp", "spf.init", "spf.wall", "spf.inlet", "spf.outlet", "spf.symmetry"},
    "ht": {"ht.solid", "ht.init", "ht.source", "ht.temperature", "ht.heatflux", "ht.insulation", "ht.symmetry"}}
WHY_NOT = {
    "spf.vf": "a Volume Force (a momentum source is not mapped yet)",
    "spf.gravity": "Gravity (a momentum source is not mapped yet)",
    "spf.open": "an Open Boundary (traction-free with backflow has no exact counterpart among OpenFOAM's standard "
                "conditions; use an Outlet)",
    "ht.fluid": "a Fluid domain (convection by a velocity field is not part of laplacianFoam)",
    "ht.radiation": "Surface-to-Ambient Radiation (nonlinear in T; not mapped)",
}
STEP_WHY = {"study.eigen": "an Eigenfrequency study (OpenFOAM has no eigenvalue solver here)",
            "study.freq": "a Frequency Domain study (OpenFOAM solves in time only)"}


class FoamRefusal(RuntimeError):
    """The model cannot be solved faithfully by the OpenFOAM backend (the message says why and what to do)."""


def _fail(msg: str):
    raise FoamRefusal(msg + " Choose Elmer as the solver backend for this study.")


# --------------------------------------------------------------------------- quick check
def check_model(model: Model, study: Node) -> str | None:
    """The first reason the study cannot be solved by OpenFOAM, found without building the geometry; None if none."""
    try:
        _check(model, study)
    except FoamRefusal as exc:
        return str(exc)
    return None


def _check(model: Model, study: Node):
    """Every component the study solves (each one a separate OpenFOAM case), or the active one."""
    comps = model.components()
    if not comps:
        _fail("The model has no component.")
    for step in study_steps(study):
        if step.kind in STEP_WHY:
            _fail(f"{step.label} is {STEP_WHY[step.kind]}.")
        if str(step.get("initsol", "") or ""):
            _fail(f"{step.label} takes its initial values from another study's solution, which the OpenFOAM "
                  "backend does not map.")
    solved = [c for c in comps if any(c in step_components(model, s) for s in study_steps(study))] or [model.component]
    for comp in solved:
        with model.scoped(comp):
            _check_component(model, study, comp)


def _check_component(model: Model, study: Node, comp: Node):
    from ...study_runner import cross_references
    where = f" ({comp.label})" if len(model.components()) > 1 else ""
    if is_axisymmetric(comp):
        _fail(f"2D axisymmetric components{where} are not supported by the OpenFOAM backend (a wedge mesh is not "
              "verified).")
    refs = cross_references(model, comp)
    if refs:
        _fail(f"{comp.label} uses another component's results ({refs[0][0]}); values passed between components "
              "are not mapped to OpenFOAM.")
    steps = study_steps(study)
    for step in steps:
        sel = step.props.get("physics", {})
        active = [ph for ph in physics_nodes(model, comp) if ph.enabled and sel.get(ph.tag, True)]
        if len(active) > 1:
            _fail(f"{step.label} solves {len(active)} physics interfaces together{where} "
                  f"({', '.join(p.label for p in active)}); the OpenFOAM backend solves one at a time.")
        for ph in active:
            pid = physics_def(ph.kind).id
            if pid not in SUPPORTED_PHYSICS:
                _fail(f"{ph.label} is not supported by the OpenFOAM backend, which solves Laminar Flow and "
                      "Heat Transfer in Solids only.")
            if pid == "spf" and ph.get("stokes", False):
                _fail(f"{ph.label}: Stokes flow (neglected inertia) is not mapped to OpenFOAM.")
            for f in ph.children:
                if not f.enabled or f.kind in SUPPORTED_FEATURES[pid]:
                    continue
                if f.kind in FEATURES and FEATURES[f.kind].section == "post":
                    continue
                _fail(f"{f.label} ({ph.label}{', ' + comp.label if where else ''}) is "
                      f"{WHY_NOT.get(f.kind, 'a feature the OpenFOAM backend does not map')}.")
    mp = comp.child("multiphysics")
    for c in (mp.children if mp is not None and mp.enabled else []):
        if c.enabled and c.kind in COUPLINGS:
            _fail(f"{c.label} is a multiphysics coupling; the OpenFOAM backend solves single physics only.")


# --------------------------------------------------------------------------- plan
@dataclass
class FoamPlan:
    physics: str                              # spf | ht
    app: str                                  # simpleFoam | pimpleFoam | laplacianFoam
    transient: bool
    params: dict
    bcs: dict = field(default_factory=dict)   # boundary -> (kind, {name: expression})
    patch_types: dict = field(default_factory=dict)
    init: dict = field(default_factory=dict)  # domain -> {name: expression}
    sources: dict = field(default_factory=dict)  # domain -> Q (W/m^3)
    rho: float = 1.0
    mu: float = 1.0
    k: float = 1.0
    cp: float = 1.0
    tol: float = 1e-7                         # SIMPLE: initial-residual tolerance
    maxit: int = 2000
    lin_tol: float = 1e-10
    lin_maxit: int = 1000
    t0: float = 0.0
    dt: float = 1.0
    nsteps: int = 1
    sub: int = 1
    out_times: list = field(default_factory=list)   # output times (s), after t0
    ddt: str = "steadyState"
    labels: dict = field(default_factory=dict)      # boundary -> feature label (for messages)
    sdim: int = 2

    @property
    def DT(self) -> float:
        return self.k / (self.rho * self.cp)


class _Expr:
    """Studio expressions reduced to SI numbers of parameters and x, y, z (evaluated at face or cell centres)."""

    def __init__(self, ctx: SifContext):
        self.ctx = ctx

    def text(self, expr, what: str, space=True) -> str:
        ctx = self.ctx
        s = ctx._inline_funcs(ctx._expand(str(expr).strip() or "0"))
        try:
            syms = units.free_symbols(s) - set(ctx.params)
        except units.ExprError as exc:
            _fail(f"{what}: {exc}.")
        if "t" in syms:
            _fail(f"{what} depends on time ('{expr}'); time-dependent values are not mapped to OpenFOAM.")
        extra = syms - ({"x", "y", "z"} if space else set())
        if extra:
            dep = "space" if extra <= {"x", "y", "z"} else "the solution or an unknown symbol"
            _fail(f"{what} depends on {dep} ('{expr}': {', '.join(sorted(extra))}); the OpenFOAM backend needs "
                  f"{'a constant' if not space else 'parameters and x, y, z only'} here.")
        try:
            units.evaluate(s, {**ctx.params, "x": 0.0, "y": 0.0, "z": 0.0})
        except Exception as exc:
            _fail(f"{what}: cannot evaluate '{expr}' ({exc}).")
        return s

    def const(self, expr, what: str) -> float:
        return float(np.real(units.evaluate(self.text(expr, what, space=False), self.ctx.params)))


def evaluate_at(expr: str, params: dict, pts: np.ndarray) -> np.ndarray:
    """An expression of parameters and x, y, z (SI) at points (N, 3): an array of N values."""
    v = units.evaluate(expr, {**params, "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]})
    return np.broadcast_to(np.real(np.asarray(v, float)), (len(pts),)).copy()


def _solver_config(study: Node, ph: Node) -> Node | None:
    sols = study.child("study.solvers")
    return next((c for c in (sols.children if sols else []) if c.props.get("physics") == ph.tag), None)


def plan_case(model: Model, study: Node, step: Node, layout: BlockLayout, overrides=None) -> FoamPlan:
    _check(model, study)
    try:
        ctx = SifContext(model, study, step, layout.geo, 0, overrides)
    except SifError as exc:
        raise FoamRefusal(str(exc)) from exc
    ex = _Expr(ctx)
    geo = layout.geo
    ph = ctx.active[0]
    pid = physics_def(ph.kind).id
    doms = list(geo.numbers["domain"])
    phd = _entities(ctx, ph, "domain")
    if set(phd) != set(doms):
        _fail(f"{ph.label} covers domains {sorted(phd)} of {sorted(doms)}; the OpenFOAM backend meshes every domain "
              "and needs the physics on all of them.")
    transient = step.kind == "study.time"
    app = {"spf": "pimpleFoam" if transient else "simpleFoam", "ht": "laplacianFoam"}[pid]
    P = FoamPlan(pid, app, transient, dict(ctx.params), sdim=layout.sdim)

    # ---- materials: one constant value for all domains
    def material(kind, keys):
        vals = {k: {} for k in keys}
        for f in _features(ph, "domain"):
            if f.kind != kind:
                continue
            for d in _entities(ctx, f, "domain", set(phd)):
                for k in keys:
                    try:
                        e = ctx._mat_expr(f, k, d)
                    except SifError as exc:
                        raise FoamRefusal(str(exc)) from exc
                    vals[k][d] = ex.const(e, f"{f.label}: {k} in domain {d}")
        out = {}
        for k in keys:
            got = vals[k]
            missing = [d for d in phd if d not in got]
            if missing:
                _fail(f"{ph.label}: no {kind} feature gives '{k}' in domains {missing}.")
            v = list(got.values())
            if max(v) - min(v) > 1e-12 * max(abs(x) for x in v):
                _fail(f"{ph.label}: '{k}' differs between domains ({', '.join(f'{d}: {x:g}' for d, x in got.items())}); "
                      "the OpenFOAM backend solves with one constant value.")
            if v[0] <= 0:
                _fail(f"{ph.label}: '{k}' must be positive (it is {v[0]:g}).")
            out[k] = v[0]
        return out

    if pid == "spf":
        m = material("spf.fp", ["rho", "mu"])
        P.rho, P.mu = m["rho"], m["mu"]
    else:
        m = material("ht.solid", ["k", "rho", "Cp"])
        P.k, P.rho, P.cp = m["k"], m["rho"], m["Cp"]

    # ---- initial values and sources per domain (later features override earlier ones, as in the .sif)
    init_kind = f"{pid}.init"
    for f in _features(ph, "domain"):
        for d in _entities(ctx, f, "domain", set(phd)):
            if f.kind == init_kind:
                if pid == "spf":
                    P.init[d] = {c: ex.text(f.get(f"u{c}", "0"), f"{f.label}: u{c}") for c in "xyz"}
                    P.init[d]["p"] = ex.text(f.get("p0", "0"), f"{f.label}: pressure")
                else:
                    P.init[d] = {"T": ex.text(f.get("T0", "293.15[K]"), f"{f.label}: temperature")}
            elif f.kind == "ht.source":
                P.sources[d] = ex.const(f.get("Q0", "0"), f"{f.label}: heat source in domain {d}")

    # ---- boundary conditions (default features on the exterior only; later features override earlier ones)
    pb = set(geo.boundaries_of_domains(phd))
    exterior = {b for b in pb if len(set(geo.adj_up["boundary"].get(b, [])) & set(phd)) == 1}
    per: dict = {}
    for f in _features(ph, "boundary"):
        if f.kind in FEATURES and FEATURES[f.kind].section == "post":
            continue
        for b in _entities(ctx, f, "boundary", exterior if f.meta.get("default") else pb):
            per[b] = f
    for b, f in sorted(per.items()):
        if b not in exterior:
            _fail(f"{f.label} acts on interior boundary {b}; a condition between two domains would need a baffle, "
                  "which the OpenFOAM backend does not create.")
        P.labels[b] = f.label
        P.bcs[b], P.patch_types[b] = _bc(f, ex, pid)
    for b in exterior - set(per):
        P.bcs[b], P.patch_types[b] = (("wall", {}), "wall") if pid == "spf" else (("zero", {}), "patch")

    # ---- solver settings
    cfg = _solver_config(study, ph)
    if cfg is not None:
        if int(cfg.get("nl_maxit", 0) or 0) > 0:
            P.maxit = int(cfg.get("nl_maxit"))
        P.tol = ex.const(cfg.get("nl_tol", "1e-7"), "Solver: nonlinear tolerance")
        P.lin_tol = ex.const(cfg.get("tol", "1e-10"), "Solver: linear tolerance")
        P.lin_maxit = int(cfg.get("maxit", 1000) or 1000)
    if transient:
        tfac = units.unit_info(step.get("tunit", "s"))[0]
        times = [t * tfac for t in parse_list(step.get("times", "range(0,0.1,1)"), ctx.params)]
        if len(times) < 2:
            _fail(f"{step.label}: give at least two output times.")
        groups = time_groups(times)
        if len(groups) != 1:
            _fail(f"{step.label}: the output times are not equally spaced; the OpenFOAM backend runs one fixed "
                  "time step.")
        dt, n = groups[0]
        P.sub = max(int(step.get("substeps", 1) or 1), 1)
        P.t0, P.dt, P.nsteps = times[0], dt / P.sub, n * P.sub
        P.out_times = [times[0] + (i + 1) * dt for i in range(n)]
        P.ddt = "backward" if str(step.get("bdf", "2")) == "2" else "Euler"
    return P


def _bc(f: Node, ex: _Expr, pid: str):
    """(kind, expressions) and the patch type of one boundary feature."""
    lab = f.label
    if f.kind == "spf.wall":
        return (("wall", {}) if f.get("cond", "noslip") == "noslip" else ("slip", {})), "wall"
    if f.kind == "spf.inlet":
        if f.get("type", "velocity") == "pressure":
            return ("pressure", {"p": ex.text(f.get("p0", "0"), f"{lab}: pressure")}), "patch"
        if f.get("spec", "normal") == "normal":
            return ("inlet_normal", {"U0": ex.text(f.get("U0", "0"), f"{lab}: normal inflow velocity")}), "patch"
        return ("inlet_field", {c: ex.text(f.get(f"u{c}", "0"), f"{lab}: u{c}") for c in "xyz"}), "patch"
    if f.kind == "spf.outlet":
        kind = "pressure_normal" if f.get("normal", False) else "pressure"
        return (kind, {"p": ex.text(f.get("p0", "0"), f"{lab}: pressure")}), "patch"
    if f.kind == "spf.symmetry":
        return ("symmetry", {}), "symmetryPlane"
    if f.kind == "ht.temperature":
        return ("temp", {"T": ex.text(f.get("T0", "293.15[K]"), f"{lab}: temperature")}), "patch"
    if f.kind in ("ht.insulation", "ht.symmetry"):
        return ("zero", {}), "patch"
    if f.kind == "ht.heatflux":
        if f.get("type", "general") == "general":
            return ("flux", {"q": ex.text(f.get("q0", "0"), f"{lab}: heat flux")}), "patch"
        return ("conv", {"h": ex.text(f.get("h", "10"), f"{lab}: heat transfer coefficient"),
                         "Text": ex.text(f.get("Text", "293.15[K]"), f"{lab}: external temperature")}), "patch"
    _fail(f"{lab} is {WHY_NOT.get(f.kind, 'a feature the OpenFOAM backend does not map')}.")


# --------------------------------------------------------------------------- dictionaries
def _control(P: FoamPlan) -> str:
    if P.app == "simpleFoam":
        start, end, dt, wc, wi = 0, P.maxit, 1, "timeStep", P.maxit
    elif P.transient:
        start, end, dt, wc, wi = P.t0, P.t0 + P.nsteps * P.dt, P.dt, "timeStep", P.sub
    else:                                           # laplacianFoam, steady: one solve
        start, end, dt, wc, wi = 0, 1, 1, "timeStep", 1
    return (dicts.header("dictionary", "controlDict", "system")
            + f"application     {P.app};\nstartFrom       startTime;\nstartTime       {start:.15g};\n"
              f"stopAt          endTime;\nendTime         {end:.15g};\ndeltaT          {dt:.15g};\n"
              f"writeControl    {wc};\nwriteInterval   {wi};\npurgeWrite      0;\nwriteFormat     ascii;\n"
              "writePrecision  15;\nwriteCompression off;\ntimeFormat      general;\ntimePrecision   12;\n"
              "runTimeModifiable false;\n")


def _schemes(P: FoamPlan) -> str:
    ddt = P.ddt if P.transient else "steadyState"
    if P.physics == "spf":
        div = ("    default         none;\n"
               f"    div(phi,U)      {'bounded ' if not P.transient else ''}Gauss linearUpwind grad(U);\n"
               "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n")
    else:
        div = "    default         none;\n"
    return (dicts.header("dictionary", "fvSchemes", "system")
            + f"ddtSchemes\n{{\n    default         {ddt};\n}}\n\ngradSchemes\n{{\n    default         Gauss linear;\n}}\n\n"
              f"divSchemes\n{{\n{div}}}\n\nlaplacianSchemes\n{{\n    default         Gauss linear corrected;\n}}\n\n"
              "interpolationSchemes\n{\n    default         linear;\n}\n\nsnGradSchemes\n{\n    default         corrected;\n}\n")


def _solution(P: FoamPlan) -> str:
    tol, mi = f"{P.lin_tol:.6g}", P.lin_maxit
    if P.physics == "ht":
        return (dicts.header("dictionary", "fvSolution", "system")
                + f"solvers\n{{\n    T\n    {{\n        solver          PCG;\n        preconditioner  DIC;\n"
                  f"        tolerance       {tol};\n        relTol          0;\n        maxIter         {mi};\n    }}\n}}\n\n"
                  "SIMPLE\n{\n    nNonOrthogonalCorrectors 0;\n}\n")
    if P.app == "simpleFoam":
        return (dicts.header("dictionary", "fvSolution", "system")
                + f"solvers\n{{\n    p\n    {{\n        solver          GAMG;\n        smoother        GaussSeidel;\n"
                  f"        tolerance       {tol};\n        relTol          0.01;\n        maxIter         {mi};\n    }}\n"
                  f"    U\n    {{\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
                  f"        tolerance       {tol};\n        relTol          0.1;\n        maxIter         {mi};\n    }}\n}}\n\n"
                  f"SIMPLE\n{{\n    nNonOrthogonalCorrectors 0;\n    consistent      yes;\n    pRefCell        0;\n"
                  f"    pRefValue       0;\n    residualControl\n    {{\n        p               {P.tol:.6g};\n"
                  f"        U               {P.tol:.6g};\n    }}\n}}\n\n"
                  "relaxationFactors\n{\n    equations\n    {\n        U               0.9;\n        \".*\"            0.9;\n"
                  "    }\n}\n")
    return (dicts.header("dictionary", "fvSolution", "system")
            + f"solvers\n{{\n    \"(p|pFinal)\"\n    {{\n        solver          GAMG;\n        smoother        GaussSeidel;\n"
              f"        tolerance       {tol};\n        relTol          0;\n        maxIter         {mi};\n    }}\n"
              f"    \"(U|UFinal)\"\n    {{\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
              f"        tolerance       {tol};\n        relTol          0;\n        maxIter         {mi};\n    }}\n}}\n\n"
              "PIMPLE\n{\n    nOuterCorrectors 1;\n    nCorrectors     2;\n    nNonOrthogonalCorrectors 0;\n"
              "    pRefCell        0;\n    pRefValue       0;\n}\n")


def _fv_options(P: FoamPlan) -> str | None:
    src = {d: q for d, q in P.sources.items() if q != 0}
    if not src:
        return None
    body = ""
    for d, q in sorted(src.items()):
        body += (f"heatSource_dom{d}\n{{\n    type            scalarSemiImplicitSource;\n    selectionMode   cellZone;\n"
                 f"    cellZone        dom{d};\n    volumeMode      specific;\n    injectionRateSuSp\n    {{\n"
                 f"        T               ({q / (P.rho * P.cp):.15g} 0);\n    }}\n}}\n\n")
    return dicts.header("dictionary", "fvOptions", "system") + body


def write_case(P: FoamPlan, layout: BlockLayout, case: str):
    """system/ and constant/ (the fields in 0/ need the mesh: write_fields after blockMesh)."""
    from .blocks import block_mesh_dict
    dicts.write(os.path.join(case, "system", "blockMeshDict"), block_mesh_dict(layout, P.patch_types))
    dicts.write(os.path.join(case, "system", "controlDict"), _control(P))
    dicts.write(os.path.join(case, "system", "fvSchemes"), _schemes(P))
    dicts.write(os.path.join(case, "system", "fvSolution"), _solution(P))
    fo = _fv_options(P)
    if fo:
        dicts.write(os.path.join(case, "system", "fvOptions"), fo)
    if P.physics == "spf":
        dicts.write(os.path.join(case, "constant", "transportProperties"),
                    dicts.header("dictionary", "transportProperties", "constant")
                    + f"transportModel  Newtonian;\n\n// nu = mu/rho = {P.mu:.15g}/{P.rho:.15g}\n"
                      f"nu              {P.mu / P.rho:.15g};\n")
        dicts.write(os.path.join(case, "constant", "turbulenceProperties"),
                    dicts.header("dictionary", "turbulenceProperties", "constant") + "simulationType  laminar;\n")
    else:
        dicts.write(os.path.join(case, "constant", "transportProperties"),
                    dicts.header("dictionary", "transportProperties", "constant")
                    + f"// DT = k/(rho Cp) = {P.k:.15g}/({P.rho:.15g}*{P.cp:.15g})\n"
                      f"DT              DT [0 2 -1 0 0 0 0] {P.DT:.15g};\n")


def _normals(pm, fl) -> np.ndarray:
    p = pm.points[pm.faces[fl]]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return n / np.linalg.norm(n, axis=1)[:, None]


def boundary_specs(P: FoamPlan, pm) -> dict:
    """Per patch: the face list, outward normals, centres (SI) and the evaluated boundary values (SI, not kinematic)."""
    out = {}
    for b, (kind, ex) in P.bcs.items():
        fl = pm.patch_faces(f"bnd{b}")
        C = pm.face_centres[fl]
        n = _normals(pm, fl)
        vals = {k: evaluate_at(e, P.params, C) for k, e in ex.items()}
        own = pm.owner[fl]
        d = np.abs(np.einsum("ij,ij->i", n, C - pm.cell_centres[own]))
        out[b] = {"kind": kind, "faces": fl, "normals": n, "centres": C, "values": vals, "cells": own, "delta": d}
    return out


def _entry(name: str, body: str) -> str:
    return f"    {name}\n    {{\n" + "\n".join(f"        {line}" for line in body.strip().splitlines()) + "\n    }\n"


def write_fields(P: FoamPlan, pm, case: str, start_name: str = "0"):
    """0/U and 0/p (Laminar Flow) or 0/T (heat), with per-face values where they vary."""
    specs = boundary_specs(P, pm)
    dom = pm.cell_zone()
    cc = pm.cell_centres
    tdir = os.path.join(case, start_name)
    empty = _entry("frontAndBack", "type empty;") if P.sdim == 2 else ""

    def internal(name):
        v = np.zeros(pm.n_cells)
        for d, ex in P.init.items():
            sel = dom == d
            if name in ex and np.any(sel):
                v[sel] = evaluate_at(ex[name], P.params, cc[sel])
        return v

    if P.physics == "spf":
        U = np.column_stack([internal(c) for c in "xyz"])
        if P.sdim == 2:
            U[:, 2] = 0.0
        pk = internal("p") / P.rho
        ub, pbody = "", ""
        for b, s in sorted(specs.items()):
            kind, n, nf = s["kind"], s["normals"], len(s["faces"])
            if kind == "wall":
                ub += _entry(f"bnd{b}", "type noSlip;")
            elif kind == "slip":
                ub += _entry(f"bnd{b}", "type slip;")
            elif kind in ("inlet_normal", "inlet_field"):
                if kind == "inlet_normal":
                    val = -s["values"]["U0"][:, None] * n
                else:
                    val = np.column_stack([s["values"][c] for c in "xyz"])
                if P.sdim == 2:
                    val[:, 2] = 0.0
                s["U"] = val
                ub += _entry(f"bnd{b}", f"type fixedValue;\nvalue {dicts.field_value(val, nf)};")
            elif kind == "pressure":
                ub += _entry(f"bnd{b}", "type zeroGradient;")
            elif kind == "pressure_normal":
                nn = n[0]
                vf = np.eye(3) - np.outer(nn, nn)
                st = [vf[0, 0], vf[0, 1], vf[0, 2], vf[1, 1], vf[1, 2], vf[2, 2]]
                ub += _entry(f"bnd{b}", "type directionMixed;\nrefValue uniform (0 0 0);\nrefGradient uniform (0 0 0);\n"
                                        f"valueFraction uniform {dicts.fmt(st)};\nvalue uniform (0 0 0);")
            elif kind == "symmetry":
                ub += _entry(f"bnd{b}", "type symmetryPlane;")
            if kind in ("pressure", "pressure_normal"):
                pv = s["values"]["p"] / P.rho
                s["p"] = pv
                pbody += _entry(f"bnd{b}", f"type fixedValue;\nvalue {dicts.field_value(pv, nf)};")
            elif kind == "symmetry":
                pbody += _entry(f"bnd{b}", "type symmetryPlane;")
            else:
                pbody += _entry(f"bnd{b}", "type zeroGradient;")
        dicts.write(os.path.join(tdir, "U"), dicts.header("volVectorField", "U", start_name)
                    + f"dimensions      [0 1 -1 0 0 0 0];\n\ninternalField   {dicts.field_value(U, len(U))};\n\n"
                      f"boundaryField\n{{\n{ub}{empty}}}\n")
        dicts.write(os.path.join(tdir, "p"), dicts.header("volScalarField", "p", start_name)
                    + "// kinematic pressure p/rho (m^2/s^2); Studio converts it back to Pa\n"
                      f"dimensions      [0 2 -2 0 0 0 0];\n\ninternalField   {dicts.field_value(pk, len(pk))};\n\n"
                      f"boundaryField\n{{\n{pbody}{empty}}}\n")
    else:
        T = internal("T")
        tb = ""
        for b, s in sorted(specs.items()):
            kind, nf = s["kind"], len(s["faces"])
            if kind == "temp":
                tb += _entry(f"bnd{b}", f"type fixedValue;\nvalue {dicts.field_value(s['values']['T'], nf)};")
            elif kind == "zero":
                tb += _entry(f"bnd{b}", "type zeroGradient;")
            elif kind == "flux":
                g = s["values"]["q"] / P.k
                tb += _entry(f"bnd{b}", f"type fixedGradient;\ngradient {dicts.field_value(g, nf)};")
            elif kind == "conv":
                h, Te = s["values"]["h"], s["values"]["Text"]
                f = h / (h + P.k / s["delta"])
                s["f"] = f
                tb += _entry(f"bnd{b}", f"type mixed;\nrefValue {dicts.field_value(Te, nf)};\n"
                                        f"refGradient uniform 0;\nvalueFraction {dicts.field_value(f, nf)};\n"
                                        f"value {dicts.field_value(Te, nf)};")
        dicts.write(os.path.join(tdir, "T"), dicts.header("volScalarField", "T", start_name)
                    + f"dimensions      [0 0 0 1 0 0 0];\n\ninternalField   {dicts.field_value(T, len(T))};\n\n"
                      f"boundaryField\n{{\n{tb}{empty}}}\n")
    return specs


def time_name(t: float) -> str:
    """OpenFOAM's directory name of time t (timeFormat general, timePrecision 12)."""
    return f"{t:.12g}"
