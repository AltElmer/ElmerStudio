"""Generate an Elmer Solver Input File (.sif) from the model tree.

Assembly rules (COMSOL semantics mapped to Elmer):
  * every geometry domain is one Elmer body (body id = domain number);
  * per domain, contributions of all active physics features are merged, later
    features in a physics interface override earlier ones on the same entity;
  * boundaries with identical merged keywords share one Boundary Condition section.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import units
from .builder import ensure_solver_configs, physics_nodes, study_steps
from .materials import MATPROPS
from .model import Model, Node, is_axisymmetric, space_dim
from .physics import COUPLINGS, FEATURES, physics_def

SIGMA_SB = 5.670374419e-08


class SifError(ValueError):
    pass


# --------------------------------------------------------------------------- value helpers
def parse_list(text: str, params=None) -> list[float]:
    """Parse 'range(a,step,b)', 'a b c', 'a,b,c' (values may carry units)."""
    s = str(text).strip()
    out: list[float] = []
    for part in re.split(r"\s+(?![^()]*\))", s):
        part = part.strip().strip(",")
        if not part:
            continue
        m = re.fullmatch(r"range\((.+)\)", part)
        if m:
            args = [float(np.real(units.evaluate(a, params or {}))) for a in _split_args(m.group(1))]
            if len(args) != 3:
                raise SifError("range() takes (start, step, stop)")
            a, st, b = args
            if st == 0:
                raise SifError("range() step must be non-zero")
            n = int(math.floor((b - a) / st + 1e-9)) + 1
            out.extend(a + i * st for i in range(max(n, 0)))
        else:
            for p in part.split(","):
                if p.strip():
                    out.append(float(np.real(units.evaluate(p, params or {}))))
    return out


def _split_args(s):
    depth, cur, args = 0, "", []
    for ch in s:
        if ch == "," and depth == 0:
            args.append(cur)
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    args.append(cur)
    return [a.strip() for a in args]


def fmt_num(v: float) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        return str(v)
    if v == 0:
        return "0.0"
    return f"{v:.12g}" if ("e" in f"{v:.12g}" or "." in f"{v:.12g}") else f"{v:.12g}.0"


FILE_KEYS = {"Solver Input File", "Output File Name", "Control Procedure"}


def kw_line(key: str, val, indent="  ") -> str:
    if isinstance(val, bool):
        return f"{indent}{key} = Logical {'True' if val else 'False'}"
    if isinstance(val, int):
        return f"{indent}{key} = Integer {val}"
    if isinstance(val, float):
        return f"{indent}{key} = Real {fmt_num(val)}"
    if isinstance(val, tuple):
        return f"{indent}{key} = " + " ".join(f'"{x}"' for x in val)
    if isinstance(val, SifValue):
        return f"{indent}{key} = {val.text}"
    if isinstance(val, list):
        return f"{indent}{key}({len(val)}) = " + " ".join(str(x) for x in val)
    s = str(val)
    if key in FILE_KEYS:
        return f'{indent}{key} = File "{s}"'
    if s.startswith(("Variable ", "Equals ", "Real", "Integer", "Logical", "String", "-dofs")) or "[" in s:
        return f"{indent}{key} = {s}"
    if key in ("Equation", "Variable", "Exec Solver", "Linear System Solver", "Linear System Direct Method",
               "Linear System Iterative Method", "Linear System Preconditioning", "Flux Variable", "Flux Coefficient",
               "Potential Variable", "Output File Name", "Output Format", "Radiation", "Convection",
               "Timestepping Method", "Simulation Type", "Coordinate System", "Eigen System Select"):
        return f'{indent}{key} = String "{s}"' if key not in ("Variable",) else f"{indent}{key} = {s}"
    return f'{indent}{key} = String "{s}"'


@dataclass(frozen=True)
class SifValue:
    text: str
    depends: tuple = ()

    def __str__(self):
        return self.text


def _freeze(d: dict):
    return tuple(sorted((k, v.text if isinstance(v, SifValue) else v) for k, v in d.items()))


# --------------------------------------------------------------------------- context
@dataclass
class SolverEntry:
    physics: Node | None
    keywords: dict
    post: bool = False
    sid: int = 0


class SifContext:
    def __init__(self, model: Model, study: Node, step: Node, geo, n_nodes: int = 0, overrides=None):
        self.model = model
        self.study = study
        self.step = step
        self.geo = geo                     # GeometryResult (numbering/adjacency)
        self.comp = model.component
        self.sdim = space_dim(self.comp)
        self.axi = is_axisymmetric(self.comp)
        self.params = model.parameters(overrides)
        self.n_nodes = n_nodes
        self.equation_extra: dict = {}
        self.nonlinear: set[str] = set()
        self.warnings: list[str] = []
        self._current_physics: Node | None = None
        self.active = self._active_physics()
        self.fields = self._fields()
        self.funcs, self.interp = self._functions()
        self.variables = self._variables()

    # -- active physics
    def _active_physics(self) -> list[Node]:
        sel = self.step.props.get("physics", {})
        out = []
        for ph in physics_nodes(self.model):
            if ph.enabled and sel.get(ph.tag, True):
                pd = physics_def(ph.kind)
                skind = self.step.kind.split(".", 1)[1]
                if skind not in pd.studies and not (skind == "time" and "time" in pd.studies):
                    raise SifError(f"{ph.label} does not support the {self.step.type.title} study step")
                out.append(ph)
        if not out:
            raise SifError("No physics interface is selected for solving in this study step")
        ids = [physics_def(p.kind).id for p in out]
        for p in out:
            for c in physics_def(p.kind).conflicts:
                if c in ids:
                    raise SifError(f"{p.label} cannot be solved together with {c} (both use the variable 'Potential')")
        return out

    def _fields(self):
        f = {"x": "Coordinate 1", "y": "Coordinate 2", "z": "Coordinate 3", "t": "Time"}
        if self.axi:
            f.update({"r": "Coordinate 1", "z": "Coordinate 2"})
        for ph in physics_nodes(self.model):
            pd = physics_def(ph.kind)
            for sym, (var, _lbl, _u) in pd.depvars.items():
                if pd.id == "pde":
                    var = ph.get("depvar", "u")
                    sym = var
                f.setdefault(sym, var)
        return f

    def _functions(self):
        funcs, interp = {}, {}
        g = self.model.global_defs
        for n in (g.children if g else []):
            if not n.enabled:
                continue
            if n.kind == "func.analytic":
                args = [a.strip() for a in str(n.get("args", "x")).split(",") if a.strip()]
                funcs[n.get("fname", n.tag)] = (args, str(n.get("expr", "0")))
            elif n.kind == "func.interp":
                pts = []
                for r in n.get("table", []):
                    try:
                        pts.append((float(units.evaluate(r.get("x", "0"), self.params)),
                                    float(units.evaluate(r.get("y", "0"), self.params))))
                    except Exception:
                        continue
                interp[n.get("fname", n.tag)] = sorted(pts)
        return funcs, interp

    def _variables(self):
        out = {}
        defs = self.comp.child("definitions") if self.comp else None
        for n in (defs.children if defs else []):
            if n.kind == "variables" and n.enabled:
                for r in n.get("table", []):
                    nm = str(r.get("name", "")).strip()
                    if nm:
                        out[nm] = str(r.get("expr", "0"))
        return out

    def _expand(self, expr: str) -> str:
        s = str(expr)
        for _ in range(8):
            changed = False
            for nm, ex in self.variables.items():
                new = re.sub(rf"(?<![\w.]){re.escape(nm)}(?![\w(])", f"({ex})", s)
                if new != s:
                    s, changed = new, True
            if not changed:
                break
        return s

    # -- values
    def value(self, expr) -> SifValue:
        if isinstance(expr, (int, float)):
            return SifValue(f"Real {fmt_num(float(expr))}")
        s = self._expand(str(expr).strip() or "0")
        m = re.fullmatch(r"\s*(\w+)\(\s*(\w+)\s*\)\s*", s)
        if m and m.group(1) in self.interp and m.group(2) in self.fields:
            var = self.fields[m.group(2)]
            rows = "\n".join(f"      {fmt_num(x)} {fmt_num(y)}" for x, y in self.interp[m.group(1)])
            self._mark_dep([var])
            return SifValue(f"Variable {var}\n    Real\n{rows}\n    End", (var,))
        # inline analytic functions
        s = self._inline_funcs(s)
        try:
            r = units.to_matc(s, self.params, self.fields)
        except units.ExprError as exc:
            raise SifError(f"{exc}") from exc
        if isinstance(r, float):
            return SifValue(f"Real {fmt_num(r)}")
        variables, code = r
        self._mark_dep(variables)
        return SifValue(f'Variable {", ".join(variables)}\n    Real MATC "{code}"', tuple(variables))

    def _inline_funcs(self, s: str) -> str:
        for _ in range(4):
            changed = False
            for name, (args, body) in self.funcs.items():
                pat = re.compile(rf"(?<![\w.]){re.escape(name)}\(")
                m = pat.search(s)
                while m:
                    start = m.end()
                    depth, i = 1, start
                    while i < len(s) and depth:
                        depth += s[i] == "("
                        depth -= s[i] == ")"
                        i += 1
                    actual = _split_args(s[start:i - 1])
                    b = body
                    for a, v in zip(args, actual):
                        b = re.sub(rf"(?<![\w.]){re.escape(a)}(?![\w(])", f"({v})", b)
                    s = s[:m.start()] + f"({b})" + s[i:]
                    changed = True
                    m = pat.search(s)
            if not changed:
                break
        return s

    def _mark_dep(self, variables):
        if self._current_physics is not None:
            for v in variables:
                if not v.startswith("Coordinate") and v != "Time":
                    self.nonlinear.add(v)

    def material_for(self, domain: int) -> Node | None:
        mats = self.comp.child("materials")
        found = None
        for m in (mats.children if mats else []):
            if m.enabled and m.kind == "material" and m.selection is not None:
                if domain in m.selection.resolve(self.geo.numbers["domain"]):
                    found = m
        return found

    def _mat_expr(self, node: Node, key: str, domain: int | None, src_key=None) -> str:
        label = MATPROPS[key][0]
        src = node.get(src_key or f"{key}_src", "mat")
        if src == "user":
            return str(node.get(key, "0"))
        if domain is None:
            raise SifError(f"{node.label}: cannot resolve material property '{label}'")
        mat = self.material_for(domain)
        if mat is None:
            raise SifError(f"Undefined material in domain {domain} (required by {node.label} for '{label}'). "
                           f"Add a material and assign it to domain {domain}.")
        expr = (mat.get("values") or {}).get(key, "")
        if str(expr).strip() == "":
            raise SifError(f"Undefined material property '{label}' required by {node.label} "
                           f"(material '{mat.label}', domain {domain}).")
        return str(expr)

    def matval(self, node: Node, key: str, domain: int) -> SifValue:
        return self.value(self._mat_expr(node, key, domain))

    def matnum(self, node: Node, key: str, domain, src_key=None, boundary=False) -> str:
        if boundary and domain is None and node.selection is not None:
            bnds = node.selection.resolve(self.geo.numbers["boundary"])
            for b in bnds:
                ups = self.geo.adj_up["boundary"].get(b, [])
                if ups:
                    domain = ups[0]
                    break
        return self._mat_expr(node, key, domain, src_key)

    def pde_var(self, node: Node) -> str:
        ph = node if node.kind.startswith("physics.") else node.ancestor("physics.")
        return str(ph.get("depvar", "u")) if ph else "u"

    def nl_iters(self, ph: Node, default: int) -> int:
        pd = physics_def(ph.kind)
        own = {v for (v, _l, _u) in pd.depvars.values()}
        if self.nonlinear & own:
            return default
        if pd.id == "spf":
            return default
        return 1

    def solver_settings(self, ph: Node, nl: dict) -> dict:
        sols = ensure_solver_configs(self.model, self.study)
        cfg = next((c for c in sols.children if c.props.get("physics") == ph.tag), None)
        pd = physics_def(ph.kind)
        dofs = {"solid": self.sdim, "spf": self.sdim + 1, "acpr": 2}.get(pd.id, 1) * max(self.n_nodes, 1)
        lin = (cfg.get("lin", "auto") if cfg else "auto")
        out = dict(nl)
        if lin == "auto":
            lin = "direct" if dofs < 150_000 or self.step.kind == "study.eigen" else "iterative"
        if lin == "direct":
            out.update({"Linear System Solver": "Direct",
                        "Linear System Direct Method": cfg.get("direct", "UMFPack") if cfg else "UMFPack"})
        else:
            method = cfg.get("iter", "BiCGStabl") if cfg else "BiCGStabl"
            prec = cfg.get("prec", "ILU1") if cfg else "ILU1"
            tol = float(units.evaluate(cfg.get("tol", "1e-10"))) if cfg else 1e-10
            maxit = int(cfg.get("maxit", 1000)) if cfg else 1000
            out.update({"Linear System Solver": "Iterative", "Linear System Iterative Method": method,
                        "Linear System Max Iterations": maxit, "Linear System Convergence Tolerance": tol,
                        "Linear System Preconditioning": prec, "Linear System Residual Output": 50,
                        "Linear System Abort Not Converged": False})
            if method == "BiCGStabl":
                out["BiCGstabl polynomial degree"] = 4
        if cfg:
            if int(cfg.get("nl_maxit", 0) or 0) > 0:
                out["Nonlinear System Max Iterations"] = int(cfg.get("nl_maxit"))
            try:
                out["Nonlinear System Convergence Tolerance"] = float(units.evaluate(cfg.get("nl_tol", "1e-7")))
                out["Nonlinear System Relaxation Factor"] = float(units.evaluate(cfg.get("relax", "1")))
            except Exception:
                pass
            if str(cfg.get("extra", "")).strip():
                out["__extra__"] = str(cfg.get("extra"))
        if str(ph.get("solver_extra", "")).strip():
            out["__extra2__"] = str(ph.get("solver_extra"))
        return out


# --------------------------------------------------------------------------- builder
def _features(ph: Node, level: str) -> list[Node]:
    out = []
    for f in ph.children:
        if not f.enabled:
            continue
        nt = f.type
        if nt.selection == level:
            out.append(f)
    return out


def _feature_fns(kind: str, section: str):
    return [f.fn for f in FEATURES.values() if f.kind == kind and f.section == section and f.fn is not None]


def _entities(ctx: SifContext, node: Node, level: str, within=None) -> list[int]:
    avail = ctx.geo.numbers[level]
    if node.selection is None:
        return list(avail)
    ents = node.selection.resolve(avail)
    if within is not None:
        ents = [e for e in ents if e in within]
    return ents


@dataclass
class SifResult:
    text: str
    meta: dict = field(default_factory=dict)


def build_sif(model: Model, study: Node, step: Node, geo, n_nodes: int = 0, overrides=None,
              output_name="case") -> SifResult:
    ctx = SifContext(model, study, step, geo, n_nodes, overrides)
    sd, axi = ctx.sdim, ctx.axi
    domains = list(geo.numbers["domain"])
    boundaries = [b for b in geo.numbers["boundary"] if geo.adj_up["boundary"].get(b)]

    # physics domain sets
    phys_domains = {ph.tag: _entities(ctx, ph, "domain") for ph in ctx.active}
    phys_bnds = {ph.tag: set(geo.boundaries_of_domains(phys_domains[ph.tag])) for ph in ctx.active}

    # ---- per-domain sections
    mat = {d: {} for d in domains}
    bf = {d: {} for d in domains}
    ic = {d: {} for d in domains}
    eqx = {d: {} for d in domains}
    for ph in ctx.active:
        ctx._current_physics = ph
        doms = set(phys_domains[ph.tag])
        for f in _features(ph, "domain"):
            for d in _entities(ctx, f, "domain", doms):
                for sec, target in (("material", mat), ("bodyforce", bf), ("ic", ic), ("equation", eqx)):
                    for fn in _feature_fns(f.kind, sec):
                        target[d].update(fn(f, ctx, d))
    # multiphysics couplings
    mp = ctx.comp.child("multiphysics")
    active_ids = {physics_def(p.kind).id for p in ctx.active}
    for c in (mp.children if mp and mp.enabled else []):
        if not c.enabled or c.kind not in COUPLINGS:
            continue
        cp = COUPLINGS[c.kind]
        if not set(cp["requires"]) <= active_ids:
            ctx.warnings.append(f"{c.label} skipped: requires {', '.join(cp['requires'])} in this study step")
            continue
        for d in _entities(ctx, c, "domain"):
            for sec, fn in cp["sections"].items():
                target = {"material": mat, "bodyforce": bf, "ic": ic, "equation": eqx}[sec]
                target[d].update(fn(c, ctx, d))

    # ---- boundaries (merged across physics)
    bc = {b: {} for b in boundaries}
    for ph in ctx.active:
        ctx._current_physics = ph
        pb = phys_bnds[ph.tag]
        pdom = set(phys_domains[ph.tag])
        exterior = {b for b in pb if len(set(geo.adj_up["boundary"].get(b, [])) & pdom) == 1}
        per: dict[int, dict] = {}
        for f in _features(ph, "boundary"):
            fns = _feature_fns(f.kind, "bc")
            kws = {}
            for fn in fns:
                kws.update(fn(f, ctx, None))
            # default features (insulation, wall, ...) act on the exterior boundaries only, as in COMSOL
            for b in _entities(ctx, f, "boundary", exterior if f.meta.get("default") else pb):
                per[b] = dict(kws)    # later feature overrides earlier (COMSOL semantics)
        for b, kws in per.items():
            if b in bc:
                bc[b].update(kws)
    ctx._current_physics = None

    # ---- solvers
    solvers: list[SolverEntry] = []
    for ph in ctx.active:
        ctx._current_physics = ph
        pd = physics_def(ph.kind)
        for s in pd.solvers(ph, ctx):
            post = bool(s.pop("__post__", False))
            solvers.append(SolverEntry(ph, s, post))
    ctx._current_physics = None
    skind = step.kind
    transient = skind == "study.time"
    eigen = skind == "study.eigen"
    freq = skind == "study.freq"
    meta = {"kind": skind, "times": [], "freqs": [], "neig": 0, "warnings": ctx.warnings, "physics": [p.tag for p in ctx.active]}
    if eigen:
        neig = int(step.get("neig", 6))
        meta["neig"] = neig
        for s in solvers:
            if not s.post and physics_def(s.physics.kind).id == "solid":
                s.keywords["Eigen Analysis"] = True
                s.keywords["Eigen System Values"] = neig
                s.keywords["Eigen System Select"] = "Smallest Magnitude"
                s.keywords["Eigen System Convergence Tolerance"] = 1e-9
                shift = float(np.real(units.evaluate(step.get("shift", "0") or "0", ctx.params)))
                if shift:
                    s.keywords["Eigen System Shift"] = float((2 * math.pi * shift) ** 2)
                s.keywords["Linear System Solver"] = "Direct"
                s.keywords["Linear System Direct Method"] = s.keywords.get("Linear System Direct Method", "UMFPack")
                for k in [k for k in s.keywords if k.startswith("Linear System Iterative") or k.startswith("Linear System Precond")]:
                    del s.keywords[k]
                s.keywords["Calculate Stresses"] = False

    out_solver = {"Equation": "Result Output", "Procedure": ("ResultOutputSolve", "ResultOutputSolver"),
                  "Output File Name": output_name, "Output Format": "vtu", "Vtu Format": True,
                  "Binary Output": True, "Single Precision": False, "Save Geometry Ids": True,
                  "Exec Solver": "After Saving"}
    if eigen:
        out_solver["Number of EigenModes"] = int(step.get("neig", 6))
    solvers.append(SolverEntry(None, out_solver, True))
    for i, s in enumerate(solvers, 1):
        s.sid = i

    # ---- equations per domain
    eq_sections: list[dict] = []
    eq_index: dict = {}
    body_eq = {}
    for d in domains:
        active_ids = [s.sid for s in solvers if s.physics is not None and d in phys_domains.get(s.physics.tag, [])]
        if not active_ids:
            body_eq[d] = None
            continue
        kw = {"Active Solvers": active_ids, **ctx.equation_extra, **eqx[d]}
        key = _freeze({k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()})
        if key not in eq_index:
            eq_sections.append(kw)
            eq_index[key] = len(eq_sections)
        body_eq[d] = eq_index[key]

    def dedupe(per: dict):
        secs, idx, assign = [], {}, {}
        for d, kw in per.items():
            if not kw:
                assign[d] = None
                continue
            key = _freeze(kw)
            if key not in idx:
                secs.append(kw)
                idx[key] = len(secs)
            assign[d] = idx[key]
        return secs, assign

    mat_secs, body_mat = dedupe(mat)
    bf_secs, body_bf = dedupe(bf)
    ic_secs, body_ic = dedupe(ic)

    # ---- simulation section
    sim: dict = {"Max Output Level": 5,
                 "Coordinate System": "Axi Symmetric" if axi else f"Cartesian {sd}D",
                 "Coordinate Mapping": [1, 2, 3]}
    n_primary = sum(1 for s in solvers if not s.post)
    coupling = int(step.get("coupling", 30) or 30)
    if transient:
        tfac = units.unit_info(step.get("tunit", "s"))[0]
        times = [t * tfac for t in parse_list(step.get("times", "range(0,0.1,1)"), ctx.params)]
        if len(times) < 2:
            raise SifError("Time Dependent: specify at least two output times, e.g. range(0,0.1,1)")
        dts = np.diff(times)
        if np.any(dts <= 0) or np.ptp(dts) > 1e-6 * max(abs(dts.max()), 1e-30):
            raise SifError("Time Dependent: output times must be equally spaced and increasing")
        sub = max(int(step.get("substeps", 1) or 1), 1)
        dt = float(dts[0]) / sub
        nsteps = (len(times) - 1) * sub
        sim.update({"Simulation Type": "Transient", "Timestepping Method": "BDF", "BDF Order": int(step.get("bdf", "2")),
                    "Timestep Sizes": [fmt_num(dt)], "Timestep Intervals": [nsteps], "Output Intervals": [sub],
                    "Steady State Max Iterations": coupling if n_primary > 1 else 1})
        meta["times"] = [times[0] + (i + 1) * dt * sub for i in range(len(times) - 1)]
        meta["t0"] = times[0]
    elif freq:
        ffac = units.unit_info(step.get("funit", "Hz"))[0]
        freqs = [f * ffac for f in parse_list(step.get("freqs", "500"), ctx.params)]
        if not freqs:
            raise SifError("Frequency Domain: no frequencies given")
        meta["freqs"] = freqs
        if len(freqs) == 1:
            sim.update({"Simulation Type": "Steady State", "Steady State Max Iterations": 1, "Output Intervals": [1],
                        "Frequency": float(freqs[0])})
        else:
            rows = "\n".join(f"      {i + 1} {fmt_num(f)}" for i, f in enumerate(freqs))
            sim.update({"Simulation Type": "Scanning", "Timestep Intervals": [len(freqs)], "Timestep Sizes": ["1"],
                        "Output Intervals": [1], "Steady State Max Iterations": 1,
                        "Frequency": SifValue(f"Variable Time\n    Real\n{rows}\n    End")})
    else:
        sim.update({"Simulation Type": "Steady State", "Steady State Max Iterations": coupling if n_primary > 1 else 1,
                    "Output Intervals": [1]})
    sim["Solver Input File"] = "case.sif"

    g = [0, 0, -1] if sd == 3 else [0, -1, 0]
    consts = {"Gravity": f"Real {g[0]} {g[1]} {g[2]} 9.80665", "Stefan Boltzmann": SIGMA_SB,
              "Permittivity of Vacuum": 8.8541878128e-12, "Permeability of Vacuum": 1.25663706212e-6,
              "Boltzmann Constant": 1.380649e-23, "Unit Charge": 1.602176634e-19}

    # ---- write
    L = []
    w = L.append
    from .. import __version__
    w(f"! Generated by Elmer Studio {__version__}  (model: {model.root.label}, {study.label} / {step.label})")
    w("Check Keywords \"Warn\"\n")
    w("Header\n  Mesh DB \".\" \"mesh\"\n  Include Path \"\"\n  Results Directory \"results\"\nEnd\n")
    w("Simulation")
    for k, v in sim.items():
        if k == "Coordinate Mapping":
            w(f"  Coordinate Mapping(3) = 1 2 3")
        elif isinstance(v, list):
            w(f"  {k}({len(v)}) = " + " ".join(str(x) for x in v))
        else:
            w(kw_line(k, v))
    w("End\n")
    w("Constants")
    for k, v in consts.items():
        if k == "Gravity":
            w(f"  Gravity(4) = {g[0]} {g[1]} {g[2]} 9.80665")
        else:
            w(kw_line(k, v))
    w("End\n")
    for d in domains:
        w(f"Body {d}")
        w(f"  Name = \"Domain {d}\"")
        w(f"  Target Bodies(1) = {d}")
        if body_eq.get(d):
            w(f"  Equation = {body_eq[d]}")
        if body_mat.get(d):
            w(f"  Material = {body_mat[d]}")
        if body_bf.get(d):
            w(f"  Body Force = {body_bf[d]}")
        if body_ic.get(d):
            w(f"  Initial Condition = {body_ic[d]}")
        w("End\n")
    for s in solvers:
        w(f"Solver {s.sid}")
        if s.physics is not None:
            w(f"  ! {s.physics.label} ({s.physics.tag})")
        kws = dict(s.keywords)
        extra = [kws.pop("__extra__", ""), kws.pop("__extra2__", "")]
        if "Exec Solver" not in kws:
            kws = {"Exec Solver": "Always", **kws}
        for k, v in kws.items():
            if k == "Variable":
                w(f"  Variable = {v}")
            else:
                w(kw_line(k, v))
        for e in extra:
            for line in str(e).splitlines():
                if line.strip():
                    w("  " + line.strip())
        w("End\n")
    for i, kw in enumerate(eq_sections, 1):
        w(f"Equation {i}")
        for k, v in kw.items():
            if k == "Active Solvers":
                w(f"  Active Solvers({len(v)}) = " + " ".join(map(str, v)))
            else:
                w(kw_line(k, v))
        w("End\n")
    for title, secs in (("Material", mat_secs), ("Body Force", bf_secs), ("Initial Condition", ic_secs)):
        for i, kw in enumerate(secs, 1):
            w(f"{title} {i}")
            for k, v in kw.items():
                w(kw_line(k, v))
            w("End\n")
    # boundary conditions grouped by identical keyword sets
    groups: dict = {}
    order = []
    for b in boundaries:
        if not bc[b]:
            continue
        key = _freeze(bc[b])
        if key not in groups:
            groups[key] = (bc[b], [])
            order.append(key)
        groups[key][1].append(b)
    for i, key in enumerate(order, 1):
        kw, bl = groups[key]
        w(f"Boundary Condition {i}")
        w(f"  Target Boundaries({len(bl)}) = " + " ".join(map(str, bl)))
        for k, v in kw.items():
            w(kw_line(k, v))
        w("End\n")
    meta["solvers"] = [{"sid": s.sid, "physics": s.physics.tag if s.physics else None,
                        "equation": s.keywords.get("Equation"), "post": s.post} for s in solvers]
    meta["nonlinear"] = sorted(ctx.nonlinear)
    return SifResult("\n".join(L) + "\n", meta)
