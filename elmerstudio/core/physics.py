"""Physics interfaces and their features, mapped onto Elmer solvers and SIF keywords.

Every interface registers:
  * a physics node type  (``physics.<id>``) with default child features,
  * feature node types   (``<id>.<feature>``) with a ``sif`` contribution function,
  * a :class:`PhysicsDef` describing studies, dependent variables, solvers and default plots.

Feature contribution functions return a dict of SIF keywords for one of the sections
``material``, ``bodyforce``, ``ic``, ``equation`` (per domain) or ``bc`` (per boundary).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .model import NodeType, Prop, register

SDIM3 = lambda p: p.get("__sdim", 3) == 3  # noqa: E731


@dataclass
class Feature:
    kind: str
    section: str                       # material | bodyforce | ic | bc | equation | none
    fn: Callable | None = None         # fn(node, ctx, domain_or_None) -> dict
    level: str = "domain"


@dataclass
class PhysicsDef:
    id: str
    name: str
    icon: str
    category: str
    subcategory: str = ""
    dims: tuple = ("3D", "2D", "2Daxi")
    studies: tuple = ("stationary", "time")
    depvars: dict = field(default_factory=dict)     # symbol -> (Elmer variable, label, unit)
    defaults: list = field(default_factory=list)    # [(kind, level)]
    domain: list = field(default_factory=list)      # addable domain features
    boundary: list = field(default_factory=list)    # addable boundary features
    edge: list = field(default_factory=list)
    point: list = field(default_factory=list)
    mat_props: list = field(default_factory=list)
    solvers: Callable | None = None                 # fn(phys_node, ctx) -> list[dict]
    plots: Callable | None = None                   # fn(phys_node, sdim, study_kind) -> list[plot spec]
    order: str = "2"
    description: str = ""
    equation: str = ""
    conflicts: tuple = ()

    @property
    def kind(self):
        return f"physics.{self.id}"


PHYSICS: dict[str, PhysicsDef] = {}
FEATURES: dict[str, Feature] = {}
COUPLINGS: dict[str, dict] = {}


def physics_def(kind_or_id: str) -> PhysicsDef:
    key = kind_or_id.split(".", 1)[1] if kind_or_id.startswith("physics.") else kind_or_id
    return PHYSICS[key]


def matprop_props(keys, section="Model Inputs"):
    from .materials import MATPROPS
    out = []
    for k in keys:
        label, sym, unit, _kw, _grp = MATPROPS[k]
        out.append(Prop(f"{k}_src", f"{label}:", "choice", "mat",
                        choices=[("mat", "From material"), ("user", "User defined")], section=section, symbol=sym))
        out.append(Prop(k, "", default="0", unit=unit, section=section, symbol=sym,
                        visible=(lambda kk: (lambda p: p.get(f"{kk}_src") == "user"))(k)))
    return out


def feature(kind, title, prefix, icon, level, section, fn=None, props=(), equation=None, default=False,
            dims=(1, 2, 3), description=""):
    register(NodeType(kind, title, prefix, icon, list(props), selection=level, selection_all=default,
                      selection_locked=default, deletable=not default, can_disable=not default,
                      equation=equation, dims=dims, description=description))
    FEATURES[kind] = Feature(kind, section, fn, level)


def _phys_children(pd: PhysicsDef):
    def kids(node):
        return pd.domain + pd.boundary + pd.edge + pd.point
    return kids


def define_physics(pd: PhysicsDef, extra_props=()):
    PHYSICS[pd.id] = pd
    props = [Prop("disc", "Element order:", "choice", pd.order,
                  choices=[("1", "Linear"), ("2", "Quadratic")], section="Discretization"),
             *extra_props,
             Prop("solver_extra", "Additional Elmer solver keywords:", "text", "", section="Elmer Solver",
                  tip="Lines appended verbatim to this interface's Solver section in the .sif file")]
    register(NodeType(pd.kind, pd.name, pd.id, pd.icon, props, selection="domain", selection_all=True,
                      deletable=True, children=_phys_children(pd), equation=pd.equation, numbered=False,
                      label=f"{pd.name}", description=pd.description))


# --------------------------------------------------------------------------- common solver keywords
def linsys(kind="iterative", method="BiCGStab", precond="ILU0", tol=1e-10, maxit=1000):
    if kind == "direct":
        return {"Linear System Solver": "Direct", "Linear System Direct Method": method if method else "UMFPack"}
    d = {"Linear System Solver": "Iterative", "Linear System Iterative Method": method,
         "Linear System Max Iterations": maxit, "Linear System Convergence Tolerance": tol,
         "Linear System Preconditioning": precond, "Linear System Residual Output": 20,
         "Linear System Abort Not Converged": False}
    if method.lower() == "bicgstabl":
        d["BiCGstabl polynomial degree"] = 4
    return d


def nonlin(maxit=1, tol=1e-7, newton_after=3, relax=1.0):
    return {"Nonlinear System Max Iterations": maxit, "Nonlinear System Convergence Tolerance": tol,
            "Nonlinear System Newton After Iterations": newton_after,
            "Nonlinear System Newton After Tolerance": 1e-3, "Nonlinear System Relaxation Factor": relax,
            "Steady State Convergence Tolerance": 1e-6}


def vec_props(prefix, label, unit, section, defaults=("0", "0", "0"), names=("x", "y", "z")):
    return [Prop(f"{prefix}{c}", f"{label} {c}:" if label else f"{c}:", default=d, unit=unit, section=section,
                 visible=SDIM3 if c == names[2] else None) for c, d in zip(names, defaults)]


# =========================================================================== Heat Transfer
def _ht_solid(n, ctx, d):
    return {"Heat Conductivity": ctx.matval(n, "k", d), "Density": ctx.matval(n, "rho", d),
            "Heat Capacity": ctx.matval(n, "Cp", d)}


def _ht_fluid(n, ctx, d):
    out = _ht_solid(n, ctx, d)
    if n.get("usrc", "user") == "user":
        for i, c in enumerate("xyz"[:ctx.sdim]):
            out[f"Convection Velocity {i + 1}"] = ctx.value(n.get(f"u{c}", "0"))
    return out


def _ht_fluid_eq(n, ctx, d):
    return {"Convection": "Computed" if n.get("usrc") == "spf" else "Constant"}


feature("ht.solid", "Solid", "solid", "feature_domain", "domain", "material", _ht_solid, default=True,
        props=matprop_props(["k", "rho", "Cp"], "Heat Conduction, Solid"),
        equation=r"$\rho C_p \frac{\partial T}{\partial t} + \nabla\cdot\mathbf{q} = Q,\quad \mathbf{q}=-k\nabla T$")
feature("ht.fluid", "Fluid", "fluid", "feature_domain", "domain", "material", _ht_fluid,
        props=[Prop("usrc", "Velocity field:", "choice", "user",
                    choices=[("user", "User defined"), ("spf", "Velocity field (spf)")], section="Model Inputs"),
               *[Prop(f"u{c}", f"u {c}:", default="0", unit="m/s", section="Model Inputs",
                      visible=(lambda cc: (lambda p: p.get("usrc", "user") == "user" and (cc != "z" or p.get("__sdim", 3) == 3)))(c))
                 for c in "xyz"],
               *matprop_props(["k", "rho", "Cp"], "Heat Convection")],
        equation=r"$\rho C_p \frac{\partial T}{\partial t} + \rho C_p\mathbf{u}\cdot\nabla T + \nabla\cdot\mathbf{q} = Q$")
FEATURES["ht.fluid.eq"] = Feature("ht.fluid", "equation", _ht_fluid_eq)
feature("ht.init", "Initial Values", "init", "initial_values", "domain", "ic",
        lambda n, ctx, d: {"Temperature": ctx.value(n.get("T0"))}, default=True,
        props=[Prop("T0", "Temperature:", default="293.15[K]", unit="K", section="Initial Values", symbol="T")])
feature("ht.source", "Heat Source", "hs", "feature_domain", "domain", "bodyforce",
        lambda n, ctx, d: {"Volumetric Heat Source": ctx.value(n.get("Q0"))},
        props=[Prop("Q0", "General source:", default="0", unit="W/m^3", section="Heat Source", symbol="Q₀")],
        equation=r"$Q = Q_0$")
feature("ht.insulation", "Thermal Insulation", "ins", "default_feature", "boundary", "bc", None, default=True,
        equation=r"$-\mathbf{n}\cdot\mathbf{q} = 0$")
feature("ht.temperature", "Temperature", "temp", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Temperature": ctx.value(n.get("T0"))},
        props=[Prop("T0", "Temperature:", default="293.15[K]", unit="K", section="Temperature", symbol="T₀")],
        equation=r"$T = T_0$")


def _ht_flux(n, ctx, d=None):
    if n.get("type", "general") == "general":
        return {"Heat Flux": ctx.value(n.get("q0"))}
    return {"Heat Transfer Coefficient": ctx.value(n.get("h")), "External Temperature": ctx.value(n.get("Text"))}


feature("ht.heatflux", "Heat Flux", "hf", "feature_boundary", "boundary", "bc", _ht_flux,
        props=[Prop("type", "Flux type:", "choice", "general",
                    choices=[("general", "General inward heat flux"), ("convective", "Convective heat flux")],
                    section="Heat Flux"),
               Prop("q0", "Heat flux:", default="0", unit="W/m^2", section="Heat Flux", symbol="q₀",
                    visible=lambda p: p.get("type", "general") == "general"),
               Prop("h", "Heat transfer coefficient:", default="10", unit="W/(m^2*K)", section="Heat Flux", symbol="h",
                    visible=lambda p: p.get("type") == "convective"),
               Prop("Text", "External temperature:", default="293.15[K]", unit="K", section="Heat Flux",
                    symbol="Tₑₓₜ", visible=lambda p: p.get("type") == "convective")],
        equation=r"$-\mathbf{n}\cdot\mathbf{q} = q_0 + h\,(T_{ext}-T)$")
feature("ht.radiation", "Surface-to-Ambient Radiation", "rad", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Radiation": "Idealized", "Emissivity": ctx.value(n.get("eps")),
                                "Radiation External Temperature": ctx.value(n.get("Tamb"))},
        props=[Prop("eps", "Surface emissivity:", default="0.8", unit="1", section="Surface Emissivity", symbol="ε"),
               Prop("Tamb", "Ambient temperature:", default="293.15[K]", unit="K", section="Ambient", symbol="T_amb")],
        equation=r"$-\mathbf{n}\cdot\mathbf{q} = \varepsilon\sigma(T_{amb}^4 - T^4)$")
feature("ht.symmetry", "Symmetry", "sym", "feature_boundary", "boundary", "bc", None,
        equation=r"$-\mathbf{n}\cdot\mathbf{q} = 0$")


def _ht_solvers(ph, ctx):
    s = {"Equation": "Heat Equation", "Procedure": ("HeatSolve", "HeatSolver"), "Variable": "Temperature",
         "Stabilize": True, "Optimize Bandwidth": True}
    s.update(ctx.solver_settings(ph, nonlin(maxit=ctx.nl_iters(ph, 20), tol=1e-7)))
    flux = {"Equation": "Heat Flux Solver", "Procedure": ("FluxSolver", "FluxSolver"),
            "Calculate Flux": True, "Flux Variable": "Temperature", "Flux Coefficient": "Heat Conductivity",
            **linsys("iterative", "CG", "Diagonal", 1e-10, 500), "__post__": True}
    return [s, flux]


def _ht_plots(ph, sdim, study):
    return [{"label": "Temperature (ht)", "type": "surface" if sdim == 3 else "surface", "expr": "T",
             "cmap": "ThermalLight", "dims": sdim},
            {"label": "Isothermal Contours (ht)", "type": "isosurface" if sdim == 3 else "contour", "expr": "T",
             "cmap": "ThermalLight", "dims": sdim}]


define_physics(PhysicsDef(
    "ht", "Heat Transfer in Solids", "physics_heat", "Heat Transfer",
    depvars={"T": ("Temperature", "Temperature", "K")},
    defaults=[("ht.solid", "domain"), ("ht.init", "domain"), ("ht.insulation", "boundary")],
    domain=["ht.solid", "ht.fluid", "ht.source", "ht.init"],
    boundary=["ht.temperature", "ht.heatflux", "ht.radiation", "ht.insulation", "ht.symmetry"],
    mat_props=["k", "rho", "Cp"], solvers=_ht_solvers, plots=_ht_plots,
    description="Heat transfer by conduction (and convection by a prescribed velocity). Elmer HeatSolver.",
    equation=r"$\rho C_p \frac{\partial T}{\partial t} + \rho C_p\mathbf{u}\cdot\nabla T + \nabla\cdot\mathbf{q} = Q,\quad \mathbf{q}=-k\nabla T$"))


# =========================================================================== Solid Mechanics
def _solid_lemm(n, ctx, d):
    return {"Youngs Modulus": ctx.matval(n, "E", d), "Poisson Ratio": ctx.matval(n, "nu", d),
            "Density": ctx.matval(n, "rho", d)}


feature("solid.lemm", "Linear Elastic Material", "lemm", "feature_domain", "domain", "material", _solid_lemm, default=True,
        props=matprop_props(["E", "nu", "rho"], "Linear Elastic Material"),
        equation=r"$\rho\frac{\partial^2\mathbf{u}}{\partial t^2} = \nabla\cdot\mathbf{S} + \mathbf{F}_V,\quad \mathbf{S}=\mathbf{C}:\boldsymbol{\epsilon}$")
feature("solid.init", "Initial Values", "init", "initial_values", "domain", "ic",
        lambda n, ctx, d: {f"Displacement {i + 1}": ctx.value(n.get(f"u{c}", "0")) for i, c in enumerate("xyz"[:ctx.sdim])},
        default=True, props=vec_props("u", "Displacement field", "m", "Initial Values"))
feature("solid.free", "Free", "free", "default_feature", "boundary", "bc", None, default=True,
        equation=r"$\mathbf{S}\cdot\mathbf{n} = \mathbf{0}$")
feature("solid.fixed", "Fixed Constraint", "fix", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {f"Displacement {i + 1}": 0.0 for i in range(ctx.sdim)},
        equation=r"$\mathbf{u} = \mathbf{0}$")


def _solid_disp(n, ctx, d=None):
    out = {}
    for i, c in enumerate("xyz"[:ctx.sdim]):
        if n.get(f"on{c}", False):
            out[f"Displacement {i + 1}"] = ctx.value(n.get(f"u{c}", "0"))
    return out


feature("solid.disp", "Prescribed Displacement", "disp", "feature_boundary", "boundary", "bc", _solid_disp,
        props=[p for c in "xyz" for p in (
            Prop(f"on{c}", f"Prescribed in {c} direction", "bool", c == "x", section="Prescribed Displacement",
                 visible=SDIM3 if c == "z" else None),
            Prop(f"u{c}", f"u0{c}:", default="0", unit="m", section="Prescribed Displacement",
                 visible=(lambda cc: (lambda p: p.get(f"on{cc}", False) and (cc != "z" or p.get("__sdim", 3) == 3)))(c)))],
        equation=r"$\mathbf{u} = \mathbf{u}_0$")
feature("solid.roller", "Roller", "rol", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Normal-Tangential Displacement": True, "Displacement 1": 0.0},
        equation=r"$\mathbf{u}\cdot\mathbf{n} = 0$")
feature("solid.symmetry", "Symmetry", "sym", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Normal-Tangential Displacement": True, "Displacement 1": 0.0},
        equation=r"$\mathbf{u}\cdot\mathbf{n} = 0$")


def _solid_bload(n, ctx, d=None):
    if n.get("type", "force") == "pressure":
        return {"Normal Force": ctx.value(f"-({n.get('p', '0')})")}
    return {f"Force {i + 1}": ctx.value(n.get(f"F{c}", "0")) for i, c in enumerate("xyz"[:ctx.sdim])}


feature("solid.bload", "Boundary Load", "bndl", "feature_boundary", "boundary", "bc", _solid_bload,
        props=[Prop("type", "Load type:", "choice", "force",
                    choices=[("force", "Force per unit area"), ("pressure", "Pressure")], section="Force"),
               *[Prop(f"F{c}", f"F_A {c}:", default="0", unit="N/m^2", section="Force",
                      visible=(lambda cc: (lambda p: p.get("type", "force") == "force" and (cc != "z" or p.get("__sdim", 3) == 3)))(c))
                 for c in "xyz"],
               Prop("p", "Pressure:", default="0", unit="Pa", section="Force", symbol="p",
                    visible=lambda p: p.get("type") == "pressure")],
        equation=r"$\mathbf{S}\cdot\mathbf{n} = \mathbf{F}_A$")
feature("solid.spring", "Spring Foundation", "spf", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Spring": ctx.value(n.get("k"))},
        props=[Prop("k", "Spring constant per unit area:", default="1e9", unit="N/m^3", section="Spring", symbol="k_A")],
        equation=r"$\mathbf{S}\cdot\mathbf{n} = -k_A\mathbf{u}$")
feature("solid.bodyload", "Body Load", "bl", "feature_domain", "domain", "bodyforce",
        lambda n, ctx, d: {f"Stress Bodyforce {i + 1}": ctx.value(n.get(f"F{c}", "0")) for i, c in enumerate("xyz"[:ctx.sdim])},
        props=vec_props("F", "F_V", "N/m^3", "Force"), equation=r"$\mathbf{F}_V$")


def _solid_gravity(n, ctx, d):
    rho = ctx.matnum(n, "rho", d, src_key="rho_src")
    return {f"Stress Bodyforce {i + 1}": ctx.value(f"({rho})*({n.get('g' + c, '0')})") for i, c in enumerate("xyz"[:ctx.sdim])}


feature("solid.gravity", "Gravity", "grav", "feature_domain", "domain", "bodyforce", _solid_gravity,
        props=[*vec_props("g", "g", "m/s^2", "Gravity", ("0", "0", "-g_const")), *matprop_props(["rho"], "Density")],
        equation=r"$\mathbf{F}_V = \rho\mathbf{g}$")


def _solid_solvers(ph, ctx):
    s = {"Equation": "Linear Elasticity", "Procedure": ("StressSolve", "StressSolver"),
         "Variable": f"-dofs {ctx.sdim} Displacement", "Calculate Stresses": True, "Displace Mesh": False}
    if ctx.sdim == 2 and not ctx.axi and ph.get("plane", "strain") == "stress":
        ctx.equation_extra["Plane Stress"] = True
    s.update(ctx.solver_settings(ph, nonlin(maxit=1)))
    return [s]


def _solid_plots(ph, sdim, study):
    if study == "eigen":
        return [{"label": "Mode Shape (solid)", "type": "volume" if sdim == 3 else "surface", "expr": "disp",
                 "cmap": "RainbowLight", "deform": True, "dims": sdim}]
    return [{"label": "Stress (solid)", "type": "volume" if sdim == 3 else "surface", "expr": "mises",
             "cmap": "RainbowLight", "deform": True, "dims": sdim}]


define_physics(PhysicsDef(
    "solid", "Solid Mechanics", "physics_solid", "Structural Mechanics",
    studies=("stationary", "eigen"),
    depvars={"u": ("Displacement 1", "Displacement field, x component", "m"),
             "v": ("Displacement 2", "Displacement field, y component", "m"),
             "w": ("Displacement 3", "Displacement field, z component", "m")},
    defaults=[("solid.lemm", "domain"), ("solid.init", "domain"), ("solid.free", "boundary")],
    domain=["solid.lemm", "solid.bodyload", "solid.gravity", "solid.init"],
    boundary=["solid.fixed", "solid.disp", "solid.roller", "solid.symmetry", "solid.bload", "solid.spring", "solid.free"],
    mat_props=["E", "nu", "rho"], solvers=_solid_solvers, plots=_solid_plots,
    description="Linear elastic stress analysis, eigenfrequencies. Elmer StressSolver.",
    equation=r"$-\nabla\cdot\mathbf{S} = \mathbf{F}_V,\quad \mathbf{S}=\mathbf{C}:\boldsymbol{\epsilon}(\mathbf{u})$"),
    extra_props=[Prop("plane", "2D approximation:", "choice", "strain",
                      choices=[("strain", "Plane strain"), ("stress", "Plane stress")], section="2D Approximation",
                      visible=lambda p: p.get("__sdim", 3) == 2 and not p.get("__axi", False))])


# =========================================================================== Electrostatics
feature("es.ccm", "Charge Conservation", "ccm", "feature_domain", "domain", "material",
        lambda n, ctx, d: {"Relative Permittivity": ctx.matval(n, "epsr", d)}, default=True,
        props=matprop_props(["epsr"], "Constitutive Relation D-E"),
        equation=r"$\nabla\cdot\mathbf{D} = \rho_V,\quad \mathbf{E}=-\nabla V,\quad \mathbf{D}=\varepsilon_0\varepsilon_r\mathbf{E}$")
feature("es.init", "Initial Values", "init", "initial_values", "domain", "ic",
        lambda n, ctx, d: {"Potential": ctx.value(n.get("V0"))}, default=True,
        props=[Prop("V0", "Electric potential:", default="0", unit="V", section="Initial Values", symbol="V")])
feature("es.zc", "Zero Charge", "zc", "default_feature", "boundary", "bc", None, default=True,
        equation=r"$\mathbf{n}\cdot\mathbf{D} = 0$")
feature("es.ground", "Ground", "gnd", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Potential": 0.0}, equation=r"$V = 0$")
feature("es.pot", "Electric Potential", "pot", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Potential": ctx.value(n.get("V0"))},
        props=[Prop("V0", "Electric potential:", default="1[V]", unit="V", section="Electric Potential", symbol="V₀")],
        equation=r"$V = V_0$")
feature("es.scd", "Surface Charge Density", "scd", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Surface Charge Density": ctx.value(n.get("rs"))},
        props=[Prop("rs", "Surface charge density:", default="0", unit="C/m^2", section="Surface Charge Density", symbol="ρₛ")],
        equation=r"$-\mathbf{n}\cdot(\mathbf{D}_1-\mathbf{D}_2) = \rho_s$")
feature("es.scharge", "Space Charge Density", "scd", "feature_domain", "domain", "bodyforce",
        lambda n, ctx, d: {"Charge Density": ctx.value(n.get("rv"))},
        props=[Prop("rv", "Space charge density:", default="0", unit="C/m^3", section="Space Charge Density", symbol="ρ_V")],
        equation=r"$\nabla\cdot\mathbf{D} = \rho_V$")


def _es_solvers(ph, ctx):
    s = {"Equation": "Electrostatics", "Procedure": ("StatElecSolve", "StatElecSolver"), "Variable": "Potential",
         "Calculate Electric Field": True, "Calculate Electric Flux": True, "Calculate Electric Energy": True}
    s.update(ctx.solver_settings(ph, nonlin(maxit=1)))
    return [s]


define_physics(PhysicsDef(
    "es", "Electrostatics", "physics_es", "AC/DC", "Electric Fields and Currents",
    depvars={"V": ("Potential", "Electric potential", "V")},
    defaults=[("es.ccm", "domain"), ("es.init", "domain"), ("es.zc", "boundary")],
    domain=["es.ccm", "es.scharge", "es.init"],
    boundary=["es.ground", "es.pot", "es.scd", "es.zc"],
    mat_props=["epsr"], solvers=_es_solvers, conflicts=("ec",),
    plots=lambda ph, sd, st: [
        {"label": "Electric Potential (es)", "type": "slice" if sd == 3 else "surface", "expr": "V", "cmap": "RainbowLight", "dims": sd},
        {"label": "Electric Field Norm (es)", "type": "slice" if sd == 3 else "surface", "expr": "normE", "cmap": "RainbowLight", "dims": sd}],
    description="Electric field and potential in dielectrics. Elmer StatElecSolver.",
    equation=r"$-\nabla\cdot(\varepsilon_0\varepsilon_r\nabla V) = \rho_V$"))


# =========================================================================== Electric Currents
feature("ec.cc", "Current Conservation", "cucn", "feature_domain", "domain", "material",
        lambda n, ctx, d: {"Electric Conductivity": ctx.matval(n, "sigma", d)}, default=True,
        props=matprop_props(["sigma"], "Conduction Current"),
        equation=r"$\nabla\cdot\mathbf{J} = Q_{j,v},\quad \mathbf{J}=\sigma\mathbf{E},\quad \mathbf{E}=-\nabla V$")
feature("ec.init", "Initial Values", "init", "initial_values", "domain", "ic",
        lambda n, ctx, d: {"Potential": ctx.value(n.get("V0"))}, default=True,
        props=[Prop("V0", "Electric potential:", default="0", unit="V", section="Initial Values", symbol="V")])
feature("ec.ins", "Electric Insulation", "ei", "default_feature", "boundary", "bc", None, default=True,
        equation=r"$\mathbf{n}\cdot\mathbf{J} = 0$")
feature("ec.ground", "Ground", "gnd", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Potential": 0.0}, equation=r"$V = 0$")
feature("ec.pot", "Electric Potential", "pot", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Potential": ctx.value(n.get("V0"))},
        props=[Prop("V0", "Electric potential:", default="1[V]", unit="V", section="Electric Potential", symbol="V₀")],
        equation=r"$V = V_0$")
feature("ec.ncd", "Normal Current Density", "ncd", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Current Density": ctx.value(n.get("Jn"))},
        props=[Prop("Jn", "Inward current density:", default="0", unit="A/m^2", section="Normal Current Density", symbol="Jₙ")],
        equation=r"$-\mathbf{n}\cdot\mathbf{J} = J_n$")


def _ec_solvers(ph, ctx):
    s = {"Equation": "Static Current", "Procedure": ("StatCurrentSolve", "StatCurrentSolver"), "Variable": "Potential",
         "Variable DOFs": 1, "Calculate Volume Current": True, "Calculate Joule Heating": True}
    s.update(ctx.solver_settings(ph, nonlin(maxit=1)))
    return [s]


define_physics(PhysicsDef(
    "ec", "Electric Currents", "physics_ec", "AC/DC", "Electric Fields and Currents",
    depvars={"V": ("Potential", "Electric potential", "V")},
    defaults=[("ec.cc", "domain"), ("ec.init", "domain"), ("ec.ins", "boundary")],
    domain=["ec.cc", "ec.init"], boundary=["ec.ground", "ec.pot", "ec.ncd", "ec.ins"],
    mat_props=["sigma"], solvers=_ec_solvers, conflicts=("es",),
    plots=lambda ph, sd, st: [
        {"label": "Electric Potential (ec)", "type": "surface", "expr": "V", "cmap": "RainbowLight", "dims": sd},
        {"label": "Current Density (ec)", "type": "slice" if sd == 3 else "surface", "expr": "normJ", "cmap": "RainbowLight", "dims": sd}],
    description="Stationary electric currents in conductors. Elmer StatCurrentSolver.",
    equation=r"$-\nabla\cdot(\sigma\nabla V) = Q_{j,v}$"))


# =========================================================================== Magnetic Fields (2D / axisymmetric)
feature("mf.al", "Ampere's Law", "al", "feature_domain", "domain", "material",
        lambda n, ctx, d: {"Relative Permeability": ctx.matval(n, "mur", d), "Electric Conductivity": ctx.matval(n, "sigma", d)},
        default=True, props=matprop_props(["mur", "sigma"], "Constitutive Relations"),
        equation=r"$\nabla\times\mathbf{H} = \mathbf{J},\quad \mathbf{B}=\nabla\times\mathbf{A},\quad \mathbf{B}=\mu_0\mu_r\mathbf{H}$")
feature("mf.init", "Initial Values", "init", "initial_values", "domain", "ic",
        lambda n, ctx, d: {"Az": 0.0}, default=True, props=[])
feature("mf.ecd", "External Current Density", "ecd", "feature_domain", "domain", "bodyforce",
        lambda n, ctx, d: {"Current Density": ctx.value(n.get("Jz"))},
        props=[Prop("Jz", "External current density, out-of-plane component:", default="1e6", unit="A/m^2",
                    section="External Current Density", symbol="J_e,z")],
        equation=r"$\mathbf{J} = \mathbf{J}_e$")
feature("mf.mi", "Magnetic Insulation", "mi", "default_feature", "boundary", "bc",
        lambda n, ctx, d=None: {"Az": 0.0}, default=True, equation=r"$\mathbf{n}\times\mathbf{A} = \mathbf{0}$")
feature("mf.pot", "Magnetic Potential", "mpot", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Az": ctx.value(n.get("A0"))},
        props=[Prop("A0", "Magnetic vector potential, z component:", default="0", unit="Wb/m", section="Magnetic Potential")],
        equation=r"$A_z = A_{0}$")
feature("mf.pmc", "Perfect Magnetic Conductor", "pmc", "feature_boundary", "boundary", "bc", None,
        equation=r"$\mathbf{n}\times\mathbf{H} = \mathbf{0}$")


def _mf_solvers(ph, ctx):
    s = {"Equation": "MgDyn2D", "Procedure": ("MagnetoDynamics2D", "MagnetoDynamics2D"), "Variable": "Az"}
    s.update(ctx.solver_settings(ph, nonlin(maxit=1)))
    calc = {"Equation": "MgDyn2D Fields", "Procedure": ("MagnetoDynamics", "MagnetoDynamicsCalcFields"),
            "Potential Variable": "Az", "Calculate Magnetic Field Strength": True,
            "Calculate Current Density": True, "Calculate Nodal Fields": True, "Calculate Elemental Fields": False,
            **linsys("iterative", "CG", "Diagonal", 1e-9, 500), "__post__": True}
    return [s, calc]


define_physics(PhysicsDef(
    "mf", "Magnetic Fields", "physics_mf", "AC/DC", "Electromagnetic Fields",
    dims=("2D", "2Daxi"),
    depvars={"Az": ("Az", "Magnetic vector potential, z component", "Wb/m")},
    defaults=[("mf.al", "domain"), ("mf.init", "domain"), ("mf.mi", "boundary")],
    domain=["mf.al", "mf.ecd", "mf.init"], boundary=["mf.mi", "mf.pot", "mf.pmc"],
    mat_props=["mur", "sigma"], solvers=_mf_solvers,
    plots=lambda ph, sd, st: [
        {"label": "Magnetic Flux Density Norm (mf)", "type": "surface", "expr": "normB", "cmap": "RainbowLight", "dims": sd,
         "contour": "Az"}],
    description="Magnetostatics with out-of-plane currents (2D, 2D axisymmetric). Elmer MagnetoDynamics2D.",
    equation=r"$\nabla\times(\mu_0^{-1}\mu_r^{-1}\nabla\times\mathbf{A}) = \mathbf{J}_e$"))


# =========================================================================== Laminar Flow
feature("spf.fp", "Fluid Properties", "fp", "feature_domain", "domain", "material",
        lambda n, ctx, d: {"Density": ctx.matval(n, "rho", d), "Viscosity": ctx.matval(n, "mu", d)}, default=True,
        props=matprop_props(["rho", "mu"], "Fluid Properties"),
        equation=r"$\rho(\mathbf{u}\cdot\nabla)\mathbf{u} = \nabla\cdot[-p\mathbf{I}+\mu(\nabla\mathbf{u}+\nabla\mathbf{u}^T)] + \mathbf{F},\ \nabla\cdot\mathbf{u}=0$")
feature("spf.init", "Initial Values", "init", "initial_values", "domain", "ic",
        lambda n, ctx, d: {**{f"Velocity {i + 1}": ctx.value(n.get(f"u{c}", "0")) for i, c in enumerate("xyz"[:ctx.sdim])},
                           "Pressure": ctx.value(n.get("p0", "0"))},
        default=True, props=[*vec_props("u", "Velocity field", "m/s", "Initial Values"),
                             Prop("p0", "Pressure:", default="0", unit="Pa", section="Initial Values", symbol="p")])
feature("spf.wall", "Wall", "wall", "default_feature", "boundary", "bc",
        lambda n, ctx, d=None: ({f"Velocity {i + 1}": 0.0 for i in range(ctx.sdim)} if n.get("cond", "noslip") == "noslip"
                                else {"Normal-Tangential Velocity": True, "Velocity 1": 0.0}),
        default=True, props=[Prop("cond", "Wall condition:", "choice", "noslip",
                                  choices=[("noslip", "No slip"), ("slip", "Slip")], section="Boundary Condition")],
        equation=r"$\mathbf{u} = \mathbf{0}$")


def _spf_inlet(n, ctx, d=None):
    if n.get("type", "velocity") == "pressure":
        return {"External Pressure": ctx.value(n.get("p0"))}
    if n.get("spec", "normal") == "normal":
        out = {"Normal-Tangential Velocity": True, "Velocity 1": ctx.value(f"-({n.get('U0', '0')})")}
        for i in range(1, ctx.sdim):
            out[f"Velocity {i + 1}"] = 0.0
        return out
    return {f"Velocity {i + 1}": ctx.value(n.get(f"u{c}", "0")) for i, c in enumerate("xyz"[:ctx.sdim])}


feature("spf.inlet", "Inlet", "inl", "feature_boundary", "boundary", "bc", _spf_inlet,
        props=[Prop("type", "Boundary condition:", "choice", "velocity",
                    choices=[("velocity", "Velocity"), ("pressure", "Pressure")], section="Boundary Condition"),
               Prop("spec", "Specification:", "choice", "normal",
                    choices=[("normal", "Normal inflow velocity"), ("field", "Velocity field")], section="Velocity",
                    visible=lambda p: p.get("type", "velocity") == "velocity"),
               Prop("U0", "Normal inflow velocity:", default="0.1", unit="m/s", section="Velocity", symbol="U₀",
                    visible=lambda p: p.get("type", "velocity") == "velocity" and p.get("spec", "normal") == "normal"),
               *[Prop(f"u{c}", f"u0 {c}:", default="0", unit="m/s", section="Velocity",
                      visible=(lambda cc: (lambda p: p.get("type", "velocity") == "velocity" and p.get("spec") == "field"
                                           and (cc != "z" or p.get("__sdim", 3) == 3)))(c)) for c in "xyz"],
               Prop("p0", "Pressure:", default="0", unit="Pa", section="Pressure Conditions", symbol="p₀",
                    visible=lambda p: p.get("type") == "pressure")],
        equation=r"$\mathbf{u} = -U_0\mathbf{n}$")
feature("spf.outlet", "Outlet", "out", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: ({"External Pressure": ctx.value(n.get("p0"))} if str(n.get("p0", "0")).strip() not in ("0", "0[Pa]") else {}),
        props=[Prop("p0", "Pressure:", default="0", unit="Pa", section="Pressure Conditions", symbol="p₀")],
        equation=r"$[-p\mathbf{I}+\mathbf{K}]\mathbf{n} = -p_0\mathbf{n}$")
feature("spf.symmetry", "Symmetry", "sym", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Normal-Tangential Velocity": True, "Velocity 1": 0.0},
        equation=r"$\mathbf{u}\cdot\mathbf{n} = 0$")
feature("spf.open", "Open Boundary", "open", "feature_boundary", "boundary", "bc", None,
        equation=r"$[-p\mathbf{I}+\mathbf{K}]\mathbf{n} = \mathbf{0}$")
feature("spf.vf", "Volume Force", "vf", "feature_domain", "domain", "bodyforce",
        lambda n, ctx, d: {f"Flow Bodyforce {i + 1}": ctx.value(n.get(f"F{c}", "0")) for i, c in enumerate("xyz"[:ctx.sdim])},
        props=vec_props("F", "F", "N/m^3", "Volume Force"), equation=r"$\mathbf{F}$")


def _spf_gravity(n, ctx, d):
    rho = ctx.matnum(n, "rho", d, src_key="rho_src")
    return {f"Flow Bodyforce {i + 1}": ctx.value(f"({rho})*({n.get('g' + c, '0')})") for i, c in enumerate("xyz"[:ctx.sdim])}


feature("spf.gravity", "Gravity", "grav", "feature_domain", "domain", "bodyforce", _spf_gravity,
        props=[*vec_props("g", "g", "m/s^2", "Gravity", ("0", "-g_const", "0")), *matprop_props(["rho"], "Density")],
        equation=r"$\mathbf{F} = \rho\mathbf{g}$")


def _spf_solvers(ph, ctx):
    d = ctx.sdim
    s = {"Equation": "Navier-Stokes", "Procedure": ("FlowSolve", "FlowSolver"),
         "Variable": f"Flow Solution[Velocity:{d} Pressure:1]", "Stabilize": True}
    if ph.get("stokes", False):
        ctx.equation_extra["NS Convect"] = False
    s.update(ctx.solver_settings(ph, nonlin(maxit=ctx.nl_iters(ph, 30), tol=1e-6, newton_after=5, relax=1.0)))
    return [s]


define_physics(PhysicsDef(
    "spf", "Laminar Flow", "physics_spf", "Fluid Flow", "Single-Phase Flow",
    depvars={"u": ("Velocity 1", "Velocity field, x component", "m/s"),
             "v": ("Velocity 2", "Velocity field, y component", "m/s"),
             "w": ("Velocity 3", "Velocity field, z component", "m/s"),
             "p": ("Pressure", "Pressure", "Pa")},
    defaults=[("spf.fp", "domain"), ("spf.init", "domain"), ("spf.wall", "boundary")],
    domain=["spf.fp", "spf.vf", "spf.gravity", "spf.init"],
    boundary=["spf.wall", "spf.inlet", "spf.outlet", "spf.symmetry", "spf.open"],
    mat_props=["rho", "mu"], solvers=_spf_solvers, order="1",
    plots=lambda ph, sd, st: [
        {"label": "Velocity (spf)", "type": "slice" if sd == 3 else "surface", "expr": "U", "cmap": "RainbowLight", "dims": sd},
        {"label": "Pressure (spf)", "type": "surface" if sd == 3 else "contour", "expr": "p", "cmap": "RainbowLight", "dims": sd}],
    description="Incompressible laminar Navier-Stokes flow (P1-P1 stabilized). Elmer FlowSolve.",
    equation=r"$\rho\frac{\partial\mathbf{u}}{\partial t}+\rho(\mathbf{u}\cdot\nabla)\mathbf{u} = \nabla\cdot[-p\mathbf{I}+\mu(\nabla\mathbf{u}+\nabla\mathbf{u}^T)]+\mathbf{F},\ \ \nabla\cdot\mathbf{u}=0$"),
    extra_props=[Prop("stokes", "Neglect inertial term (Stokes flow)", "bool", False, section="Physical Model")])


# =========================================================================== Pressure Acoustics, Frequency Domain
feature("acpr.pam", "Pressure Acoustics", "pam", "feature_domain", "domain", "material",
        lambda n, ctx, d: {"Sound Speed": ctx.matval(n, "c", d), "Density": ctx.matval(n, "rho", d)}, default=True,
        props=matprop_props(["c", "rho"], "Pressure Acoustics Model"),
        equation=r"$\nabla\cdot\left(-\frac{1}{\rho}\nabla p\right) - \frac{\omega^2 p}{\rho c^2} = 0$")
feature("acpr.init", "Initial Values", "init", "initial_values", "domain", "ic", lambda n, ctx, d: {}, default=True)
feature("acpr.hard", "Sound Hard Boundary (Wall)", "shb", "default_feature", "boundary", "bc", None, default=True,
        equation=r"$-\mathbf{n}\cdot\left(-\frac{1}{\rho}\nabla p\right) = 0$")
feature("acpr.soft", "Sound Soft Boundary", "ssb", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Pressure Wave 1": 0.0, "Pressure Wave 2": 0.0}, equation=r"$p = 0$")
feature("acpr.pressure", "Pressure", "pr", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Pressure Wave 1": ctx.value(n.get("p0")), "Pressure Wave 2": 0.0},
        props=[Prop("p0", "Pressure:", default="1[Pa]", unit="Pa", section="Pressure", symbol="p₀")],
        equation=r"$p = p_0$")
feature("acpr.nacc", "Normal Acceleration", "nacc", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Wave Flux 1": ctx.value(f"({ctx.matnum(n, 'rho', None, src_key='rho_src', boundary=True)})*({n.get('an', '0')})"),
                                "Wave Flux 2": 0.0},
        props=[Prop("an", "Inward normal acceleration:", default="1", unit="m/s^2", section="Normal Acceleration", symbol="aₙ"),
               *matprop_props(["rho"], "Density")],
        equation=r"$-\mathbf{n}\cdot\left(-\frac{1}{\rho}\nabla p\right) = a_n$")
feature("acpr.imp", "Impedance", "imp", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Wave Impedance 1": ctx.value(f"({n.get('Z', '1')})/({ctx.matnum(n, 'rho', None, src_key='rho_src', boundary=True)})"),
                                "Wave Impedance 2": 0.0},
        props=[Prop("Z", "Acoustic impedance:", default="415[Pa*s/m]", unit="Pa*s/m", section="Impedance", symbol="Zᵢ"),
               *matprop_props(["rho"], "Density")],
        equation=r"$-\mathbf{n}\cdot\left(-\frac{1}{\rho}\nabla p\right) = -\frac{i\omega p}{Z_i}$")
feature("acpr.pwr", "Plane Wave Radiation", "pwr", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Plane Wave BC": True},
        equation=r"$-\mathbf{n}\cdot\left(-\frac{1}{\rho}\nabla p\right) + \frac{i k}{\rho}p = 0$")


def _acpr_solvers(ph, ctx):
    s = {"Equation": "Helmholtz Equation", "Procedure": ("HelmholtzSolve", "HelmholtzSolver"),
         "Variable": "-dofs 2 Pressure Wave"}
    s.update(ctx.solver_settings(ph, nonlin(maxit=1)))
    return [s]


define_physics(PhysicsDef(
    "acpr", "Pressure Acoustics, Frequency Domain", "physics_acpr", "Acoustics", "Pressure Acoustics",
    studies=("freq",),
    depvars={"p": ("Pressure Wave 1", "Pressure (real part)", "Pa")},
    defaults=[("acpr.pam", "domain"), ("acpr.init", "domain"), ("acpr.hard", "boundary")],
    domain=["acpr.pam", "acpr.init"], boundary=["acpr.hard", "acpr.soft", "acpr.pressure", "acpr.nacc", "acpr.imp", "acpr.pwr"],
    mat_props=["c", "rho"], solvers=_acpr_solvers,
    plots=lambda ph, sd, st: [
        {"label": "Acoustic Pressure (acpr)", "type": "surface", "expr": "p_re", "cmap": "Wave", "dims": sd, "symmetric": True},
        {"label": "Sound Pressure Level (acpr)", "type": "surface", "expr": "Lp", "cmap": "RainbowLight", "dims": sd}],
    description="Time-harmonic acoustic pressure waves (Helmholtz equation). Elmer HelmholtzSolver.",
    equation=r"$\nabla\cdot\left(-\frac{1}{\rho}\nabla p\right) - \frac{k^2}{\rho}p = 0,\quad k=\frac{\omega}{c}$"))


# =========================================================================== Transport of Diluted Species (ModelPDE)
def _tds_tp(n, ctx, d):
    out = {"Diffusion Coefficient": ctx.value(n.get("D")), "Time Derivative Coefficient": 1.0}
    if n.get("usrc", "user") == "spf":
        out["Convection Coefficient"] = 1.0
        for i in range(ctx.sdim):
            out[f"Convection Velocity {i + 1}"] = f"Equals Velocity {i + 1}"
    else:
        vel = [str(n.get(f"u{c}", "0")).strip() for c in "xyz"[:ctx.sdim]]
        if any(v not in ("0", "") for v in vel):
            out["Convection Coefficient"] = 1.0
            for i, v in enumerate(vel):
                out[f"Convection Velocity {i + 1}"] = ctx.value(v or "0")
    return out


feature("tds.tp", "Transport Properties", "tp", "feature_domain", "domain", "material", _tds_tp, default=True,
        props=[Prop("usrc", "Velocity field:", "choice", "user",
                    choices=[("user", "User defined"), ("spf", "Velocity field (spf)")], section="Convection"),
               *[Prop(f"u{c}", f"u {c}:", default="0", unit="m/s", section="Convection",
                      visible=(lambda cc: (lambda p: p.get("usrc", "user") == "user" and (cc != "z" or p.get("__sdim", 3) == 3)))(c))
                 for c in "xyz"],
               Prop("D", "Diffusion coefficient:", default="1e-9", unit="m^2/s", section="Diffusion", symbol="D_c")],
        equation=r"$\frac{\partial c}{\partial t} + \nabla\cdot\mathbf{J} + \mathbf{u}\cdot\nabla c = R,\quad \mathbf{J}=-D\nabla c$")
feature("tds.init", "Initial Values", "init", "initial_values", "domain", "ic",
        lambda n, ctx, d: {"Concentration": ctx.value(n.get("c0"))}, default=True,
        props=[Prop("c0", "Concentration:", default="0", unit="mol/m^3", section="Initial Values", symbol="c")])
feature("tds.reac", "Reactions", "reac", "feature_domain", "domain", "bodyforce",
        lambda n, ctx, d: {"Field Source": ctx.value(n.get("R0"))},
        props=[Prop("k", "First-order rate constant:", default="0", unit="1/s", section="Reaction Rates", symbol="k"),
               Prop("R0", "Reaction rate (source):", default="0", unit="mol/(m^3*s)", section="Reaction Rates", symbol="R₀")],
        equation=r"$R = -k\,c + R_0$")
FEATURES["tds.reac.mat"] = Feature("tds.reac", "material", lambda n, ctx, d: {"Reaction Coefficient": ctx.value(n.get("k", "0"))}
                                   if str(n.get("k", "0")).strip() not in ("", "0") else {})
feature("tds.nf", "No Flux", "nflx", "default_feature", "boundary", "bc", None, default=True,
        equation=r"$-\mathbf{n}\cdot\mathbf{J} = 0$")
feature("tds.conc", "Concentration", "conc", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Concentration": ctx.value(n.get("c0"))},
        props=[Prop("c0", "Concentration:", default="1[mol/m^3]", unit="mol/m^3", section="Concentration", symbol="c₀")],
        equation=r"$c = c_0$")
feature("tds.flux", "Flux", "fl", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: ({"Field Flux": ctx.value(n.get("N0"))} if n.get("type", "general") == "general"
                                else {"Robin Coefficient": ctx.value(n.get("kc")), "External Field": ctx.value(n.get("cb"))}),
        props=[Prop("type", "Flux type:", "choice", "general",
                    choices=[("general", "General inward flux"), ("mt", "External convection (mass transfer)")], section="Inward Flux"),
               Prop("N0", "Inward flux:", default="0", unit="mol/(m^2*s)", section="Inward Flux", symbol="J₀",
                    visible=lambda p: p.get("type", "general") == "general"),
               Prop("kc", "Mass transfer coefficient:", default="1e-5", unit="m/s", section="Inward Flux", symbol="k_c",
                    visible=lambda p: p.get("type") == "mt"),
               Prop("cb", "Bulk concentration:", default="1", unit="mol/m^3", section="Inward Flux", symbol="c_b",
                    visible=lambda p: p.get("type") == "mt")],
        equation=r"$-\mathbf{n}\cdot\mathbf{J} = J_0 + k_c(c_b - c)$")
feature("tds.sym", "Symmetry", "sym", "feature_boundary", "boundary", "bc", None, equation=r"$-\mathbf{n}\cdot\mathbf{J} = 0$")
feature("tds.outflow", "Outflow", "out", "feature_boundary", "boundary", "bc", None,
        equation=r"$-\mathbf{n}\cdot D\nabla c = 0$")


def _tds_solvers(ph, ctx):
    s = {"Equation": "Diluted Species", "Procedure": ("ModelPDE", "AdvDiffSolver"), "Variable": "Concentration"}
    s.update(ctx.solver_settings(ph, nonlin(maxit=1)))
    return [s]


define_physics(PhysicsDef(
    "tds", "Transport of Diluted Species", "physics_tds", "Chemical Species Transport",
    depvars={"c": ("Concentration", "Concentration", "mol/m^3")},
    defaults=[("tds.tp", "domain"), ("tds.init", "domain"), ("tds.nf", "boundary")],
    domain=["tds.tp", "tds.reac", "tds.init"], boundary=["tds.conc", "tds.flux", "tds.nf", "tds.sym", "tds.outflow"],
    solvers=_tds_solvers,
    plots=lambda ph, sd, st: [{"label": "Concentration (tds)", "type": "surface", "expr": "c", "cmap": "RainbowLight", "dims": sd}],
    description="Diffusion, convection and first-order reactions of a dilute species. Elmer ModelPDE.",
    equation=r"$\frac{\partial c}{\partial t} + \nabla\cdot(-D\nabla c) + \mathbf{u}\cdot\nabla c = R$"))


# =========================================================================== Coefficient Form PDE
def _pde_eq(n, ctx, d):
    out = {"Diffusion Coefficient": ctx.value(n.get("c")), "Reaction Coefficient": ctx.value(n.get("a")),
           "Time Derivative Coefficient": ctx.value(n.get("da"))}
    beta = [str(n.get(f"b{c}", "0")).strip() for c in "xyz"[:ctx.sdim]]
    if any(b not in ("0", "") for b in beta):
        out["Convection Coefficient"] = 1.0
        for i, b in enumerate(beta):
            out[f"Convection Velocity {i + 1}"] = ctx.value(b or "0")
    return out


feature("pde.cfeq", "Coefficient Form PDE", "cfeq", "feature_domain", "domain", "material", _pde_eq, default=True,
        props=[Prop("c", "Diffusion coefficient c:", default="1", section="Diffusion Coefficient"),
               Prop("a", "Absorption coefficient a:", default="0", section="Absorption Coefficient"),
               Prop("f", "Source term f:", default="1", section="Source Term"),
               Prop("da", "Damping or mass coefficient dₐ:", default="1", section="Damping or Mass Coefficient"),
               *[Prop(f"b{c}", f"β {c}:", default="0", section="Convection Coefficient",
                      visible=SDIM3 if c == "z" else None) for c in "xyz"]],
        equation=r"$d_a\frac{\partial u}{\partial t} + \nabla\cdot(-c\nabla u) + \boldsymbol{\beta}\cdot\nabla u + a u = f$")
FEATURES["pde.cfeq.src"] = Feature("pde.cfeq", "bodyforce", lambda n, ctx, d: {"Field Source": ctx.value(n.get("f"))})
feature("pde.init", "Initial Values", "init", "initial_values", "domain", "ic",
        lambda n, ctx, d: {ctx.pde_var(n): ctx.value(n.get("u0"))}, default=True,
        props=[Prop("u0", "Initial value for u:", default="0", section="Initial Values")])
feature("pde.zf", "Zero Flux", "zflx", "default_feature", "boundary", "bc", None, default=True,
        equation=r"$-\mathbf{n}\cdot(-c\nabla u) = 0$")
feature("pde.dir", "Dirichlet Boundary Condition", "dir", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {ctx.pde_var(n): ctx.value(n.get("r"))},
        props=[Prop("r", "Prescribed value of u:", default="0", section="Dirichlet Boundary Condition", symbol="r")],
        equation=r"$u = r$")
feature("pde.flux", "Flux/Source", "flux", "feature_boundary", "boundary", "bc",
        lambda n, ctx, d=None: {"Field Flux": ctx.value(n.get("g")), "Robin Coefficient": ctx.value(n.get("q")),
                                "External Field": 0.0},
        props=[Prop("g", "Boundary flux/source g:", default="0", section="Boundary Flux/Source"),
               Prop("q", "Boundary absorption/impedance q:", default="0", section="Boundary Absorption/Impedance Term")],
        equation=r"$-\mathbf{n}\cdot(-c\nabla u) = g - q u$")


def _pde_solvers(ph, ctx):
    s = {"Equation": "Coefficient Form PDE", "Procedure": ("ModelPDE", "AdvDiffSolver"), "Variable": ctx.pde_var(ph)}
    s.update(ctx.solver_settings(ph, nonlin(maxit=1)))
    return [s]


define_physics(PhysicsDef(
    "pde", "Coefficient Form PDE", "physics_pde", "Mathematics", "PDE Interfaces",
    depvars={"u": ("u", "Dependent variable u", "1")},
    defaults=[("pde.cfeq", "domain"), ("pde.init", "domain"), ("pde.zf", "boundary")],
    domain=["pde.cfeq", "pde.init"], boundary=["pde.dir", "pde.flux", "pde.zf"],
    solvers=_pde_solvers,
    plots=lambda ph, sd, st: [{"label": "u", "type": "surface", "expr": "u", "cmap": "RainbowLight", "dims": sd}],
    description="Scalar second-order PDE in coefficient form. Elmer ModelPDE.",
    equation=r"$d_a\frac{\partial u}{\partial t} + \nabla\cdot(-c\nabla u) + \boldsymbol{\beta}\cdot\nabla u + a u = f$"),
    extra_props=[Prop("depvar", "Dependent variable name:", "string", "u", section="Dependent Variables")])


# =========================================================================== Multiphysics couplings
register(NodeType("multiphysics", "Multiphysics", "mp", "multiphysics", label="Multiphysics", numbered=False,
                  deletable=True, children=["mp.jh", "mp.te", "mp.nitf"]))


def coupling(kind, title, prefix, requires, fn_by_section, description, equation, props=()):
    register(NodeType(kind, title, prefix, "multiphysics", list(props), selection="domain", selection_all=True,
                      description=description, equation=equation))
    COUPLINGS[kind] = {"requires": requires, "sections": fn_by_section, "title": title}


coupling("mp.jh", "Electromagnetic Heating", "emh", ("ec", "ht"),
         {"bodyforce": lambda n, ctx, d: {"Joule Heat": True}},
         "Joule heating: resistive losses from Electric Currents act as a heat source in Heat Transfer.",
         r"$Q_e = \mathbf{J}\cdot\mathbf{E}$")
coupling("mp.te", "Thermal Expansion", "te", ("solid", "ht"),
         {"material": lambda n, ctx, d: {"Heat Expansion Coefficient": ctx.matval(n, "alpha", d),
                                         "Reference Temperature": ctx.value(n.get("Tref"))}},
         "Thermal strain from the Heat Transfer temperature field enters Solid Mechanics.",
         r"$\boldsymbol{\epsilon}_{th} = \alpha(T-T_{ref})\mathbf{I}$",
         props=[*matprop_props(["alpha"], "Model Input"),
                Prop("Tref", "Strain reference temperature:", default="293.15[K]", unit="K", section="Model Input", symbol="T_ref")])
coupling("mp.nitf", "Nonisothermal Flow", "nitf", ("spf", "ht"),
         {"equation": lambda n, ctx, d: {"Convection": "Computed"},
          "bodyforce": lambda n, ctx, d: ({"Boussinesq": True} if n.get("buoy", False) else {}),
          "material": lambda n, ctx, d: ({"Heat Expansion Coefficient": ctx.matval(n, "alpha", d),
                                          "Reference Temperature": ctx.value(n.get("Tref"))} if n.get("buoy", False) else {})},
         "Convective heat transport by the Laminar Flow velocity, optionally Boussinesq buoyancy.",
         r"$\rho C_p\mathbf{u}\cdot\nabla T,\quad \mathbf{F} = -\rho\alpha(T-T_{ref})\mathbf{g}$",
         props=[Prop("buoy", "Include buoyancy (Boussinesq)", "bool", False, section="Buoyancy"),
                *matprop_props(["alpha"], "Buoyancy"),
                Prop("Tref", "Reference temperature:", default="293.15[K]", unit="K", section="Buoyancy", symbol="T_ref")])

WIZARD_TREE = [
    ("AC/DC", [("Electric Fields and Currents", ["es", "ec"]), ("Electromagnetic Fields", ["mf"])]),
    ("Acoustics", [("Pressure Acoustics", ["acpr"])]),
    ("Chemical Species Transport", [("", ["tds"])]),
    ("Fluid Flow", [("Single-Phase Flow", ["spf"])]),
    ("Heat Transfer", [("", ["ht"]), ("Electromagnetic Heating", ["@jh"]), ("Thermal-Structure Interaction", ["@te"])]),
    ("Structural Mechanics", [("", ["solid"])]),
    ("Mathematics", [("PDE Interfaces", ["pde"])]),
]
# preset multiphysics bundles offered by the Model Wizard
BUNDLES = {
    "@jh": ("Joule Heating", "physics_heat", ["ec", "ht"], ["mp.jh"]),
    "@te": ("Thermal Stress", "physics_solid", ["ht", "solid"], ["mp.te"]),
}
STUDY_TYPES = {
    "stationary": ("Stationary", "stationary"),
    "time": ("Time Dependent", "time_dependent"),
    "eigen": ("Eigenfrequency", "eigenfrequency"),
    "freq": ("Frequency Domain", "frequency_domain"),
}
