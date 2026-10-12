"""Structural, definitions, study and results node types."""
from __future__ import annotations

from .model import NodeType, Prop, register

# --------------------------------------------------------------------------- root & definitions
register(NodeType("root", "Root", "root", "root", deletable=False, can_disable=False, numbered=False,
                  children=["component", "study"], props=[
                      Prop("unit_system", "Unit system:", "choice", "SI", choices=[("SI", "SI")], section="Unit System"),
                      Prop("author", "Author:", "string", "", section="Presentation"),
                      Prop("description", "Description:", "text", "", section="Presentation"),
                  ]))
register(NodeType("global", "Global Definitions", "global", "global_defs", label="Global Definitions", numbered=False,
                  deletable=False, can_disable=False, children=["params", "func.analytic", "func.interp"]))
register(NodeType("params", "Parameters", "param", "parameters", props=[
    Prop("table", "", "table", [], section="Parameters",
         columns=[("name", "Name"), ("expr", "Expression"), ("value", "Value"), ("descr", "Description")]),
]))
register(NodeType("func.analytic", "Analytic", "an", "functions", props=[
    Prop("fname", "Function name:", "string", "an1", section="Definition"),
    Prop("expr", "Expression:", "string", "x", section="Definition"),
    Prop("args", "Arguments:", "string", "x", section="Definition", tip="Comma-separated argument names"),
]))
register(NodeType("func.interp", "Interpolation", "int", "functions", props=[
    Prop("fname", "Function name:", "string", "int1", section="Definition"),
    Prop("table", "", "table", [{"x": "0", "y": "0"}, {"x": "1", "y": "1"}], section="Definition",
         columns=[("x", "t"), ("y", "f(t)")]),
]))
register(NodeType("component", "Component", "comp", "component", deletable=True, can_disable=False,
                  children=[], props=[
                      Prop("sdim", "Space dimension:", "choice", "3D",
                           choices=[("3D", "3D"), ("2Daxi", "2D Axisymmetric"), ("2D", "2D")], section="General",
                           readonly=True),
                  ]))
register(NodeType("definitions", "Definitions", "defs", "definitions", label="Definitions", numbered=False,
                  deletable=False, can_disable=False,
                  children=["variables", "cpl.average", "cpl.integration", "cpl.maximum", "cpl.minimum"]))
# Component couplings: nonlocal operators usable in results expressions, e.g. aveop1(T)
for _kind, _title, _prefix, _icon in (("cpl.average", "Average", "aveop", "average"),
                                      ("cpl.integration", "Integration", "intop", "integration"),
                                      ("cpl.maximum", "Maximum", "maxop", "maxmin"),
                                      ("cpl.minimum", "Minimum", "minop", "maxmin")):
    register(NodeType(_kind, _title, _prefix, _icon, selection="domain", selection_all=True,
                      description=f"{_title} coupling operator: use <name>(expression) in results, e.g. {_prefix}1(T).",
                      props=[Prop("opname", "Operator name:", "string", "", section="Operator Name"),
                             Prop("level", "Geometric entity level:", "choice", "domain",
                                  choices=[("domain", "Domain"), ("boundary", "Boundary"), ("point", "Point")],
                                  section="Source Selection")]))


def coupling_operators(model) -> dict:
    """{operator name: (kind, level, entity numbers or None for all)} from Definitions > Component Couplings."""
    out = {}
    comp = model.component
    defs = comp.child("definitions") if comp else None
    for n in (defs.children if defs else []):
        if n.kind.startswith("cpl.") and n.enabled:
            name = (n.get("opname") or n.tag).strip()
            sel = n.selection
            ents = None if (sel is None or sel.all) else list(sel.entities)
            out[name] = (n.kind.split(".", 1)[1], n.get("level", "domain"), ents)
    return out


def bind_results(model, sol, geo=None):
    """Make a Solution's result expressions read like COMSOL's: coupling operators, global parameters (with the
    sweep value it was computed for) and Definitions > Variables."""
    sol.operators = coupling_operators(model)
    try:
        params = dict(model.parameters())
    except Exception:
        params = {}
    if sol.meta.get("sweep_param") and sol.meta.get("sweep_value") is not None:
        params[sol.meta["sweep_param"]] = sol.meta["sweep_value"]
    params.update(sol.meta.get("sweep_params") or {})          # every swept parameter of this run
    sol.params = params
    comp = model.component
    defs = comp.child("definitions") if comp else None
    sol.variables = {str(r["name"]).strip(): r.get("expr", "0") for n in (defs.children if defs else [])
                     if n.kind == "variables" and n.enabled for r in (n.props.get("table") or [])
                     if str(r.get("name", "")).strip()}
    from .electrochem import electrode_results
    gl, var, flux = electrode_results(model, sol, params)
    sol.globals = gl
    if flux != sol.flux_boundaries:
        sol.flux_boundaries, sol._blump = flux, None
    sol.variables.update(var)
    if sol.kind == "study.freq":
        from .acoustics import efc_operators, incident_variables
        try:
            sol.efc = efc_operators(model, sol, geo)
        except Exception:
            sol.efc = {}
        sol.variables.update(incident_variables(model, sol))
    return sol
register(NodeType("variables", "Variables", "var", "variables", props=[
    Prop("table", "", "table", [], section="Variables",
         columns=[("name", "Name"), ("expr", "Expression"), ("descr", "Description")]),
]))

# --------------------------------------------------------------------------- study
STEP_KINDS = ["study.stationary", "study.time", "study.eigen", "study.freq"]
register(NodeType("study", "Study", "std", "study", deletable=True, can_disable=False,
                  children=STEP_KINDS + ["study.sweep"], props=[
                      Prop("plots", "Generate default plots", "bool", True, section="Study Settings"),
                      Prop("convplot", "Generate convergence plots", "bool", True, section="Study Settings"),
                      Prop("np", "Number of MPI processes:", "int", 1, section="Parallel Computing",
                           tip="Requires an MPI-enabled Elmer (ElmerSolver_mpi and mpiexec)."),
                  ]))
_phys_tbl = Prop("physics", "", "physics_table", {}, section="Physics and Variables Selection")
register(NodeType("study.stationary", "Stationary", "stat", "stationary", props=[
    Prop("coupling", "Max coupling iterations:", "int", 30, section="Study Settings",
         tip="Steady State Max Iterations when several physics are solved together"),
    _phys_tbl]))
register(NodeType("study.time", "Time Dependent", "time", "time_dependent", props=[
    Prop("tunit", "Time unit:", "choice", "s", choices=[("s", "s"), ("ms", "ms"), ("min", "min"), ("h", "h"), ("d", "d")],
         section="Study Settings"),
    Prop("times", "Output times:", "string", "range(0,0.1,1)", section="Study Settings",
         tip="range(start,step,stop) or a list of equally spaced times"),
    Prop("substeps", "Solver steps per output step:", "int", 1, section="Study Settings"),
    Prop("bdf", "BDF order:", "choice", "2", choices=[("1", "1 (backward Euler)"), ("2", "2")], section="Study Settings"),
    Prop("coupling", "Max coupling iterations per step:", "int", 10, section="Study Settings"),
    _phys_tbl]))
register(NodeType("study.eigen", "Eigenfrequency", "eig", "eigenfrequency", props=[
    Prop("neig", "Desired number of eigenfrequencies:", "int", 6, section="Study Settings"),
    Prop("shift", "Search for eigenfrequencies around:", default="0", unit="Hz", section="Study Settings"),
    _phys_tbl]))
register(NodeType("study.freq", "Frequency Domain", "freq", "frequency_domain", props=[
    Prop("funit", "Frequency unit:", "choice", "Hz", choices=[("Hz", "Hz"), ("kHz", "kHz"), ("MHz", "MHz")],
         section="Study Settings"),
    Prop("freqs", "Frequencies:", "string", "500", section="Study Settings",
         tip="A single value, a list, or range(start,step,stop)"),
    _phys_tbl]))
register(NodeType("study.sweep", "Parametric Sweep", "param", "parametric_sweep", props=[
    Prop("stype", "Sweep type:", "choice", "specified",
         choices=[("specified", "Specified combinations"), ("all", "All combinations")], section="Study Settings",
         tip="Specified: the i-th values of every parameter go together; All: every combination"),
    Prop("pname", "Parameter name:", "string", "", section="Study Settings"),
    Prop("pvalues", "Parameter value list:", "string", "range(1,1,3)", section="Study Settings"),
    Prop("punit", "Parameter unit:", "string", "", section="Study Settings"),
    Prop("more", "More parameters:", "table", [], section="Study Settings",
         columns=[("name", "Parameter name"), ("values", "Parameter value list"), ("unit", "Parameter unit")]),
]))
register(NodeType("study.solvers", "Solver Configurations", "sols", "solver_config", label="Solver Configurations",
                  numbered=False, deletable=True, can_disable=False, children=["study.solver"]))
register(NodeType("study.solver", "Solver", "s", "solver", props=[
    Prop("physics", "Physics interface:", "info", "", section="General"),
    Prop("lin", "Linear solver:", "choice", "auto",
         choices=[("auto", "Automatic (direct for small, iterative for large)"), ("direct", "Direct"),
                  ("iterative", "Iterative")], section="Linear Solver"),
    Prop("direct", "Direct solver:", "choice", "UMFPack",
         choices=[("UMFPack", "UMFPACK"), ("MUMPS", "MUMPS"), ("Banded", "Banded (LAPACK)")], section="Linear Solver",
         visible=lambda p: p.get("lin") == "direct"),
    Prop("iter", "Iterative method:", "choice", "BiCGStabl",
         choices=[("BiCGStab", "BiCGStab"), ("BiCGStabl", "BiCGStab(l)"), ("GCR", "GCR"), ("CG", "Conjugate gradients"),
                  ("GMRES", "GMRES"), ("TFQMR", "TFQMR"), ("Idrs", "IDR(s)")], section="Linear Solver",
         visible=lambda p: p.get("lin") == "iterative"),
    Prop("prec", "Preconditioner:", "choice", "ILU1",
         choices=[("ILU0", "ILU(0)"), ("ILU1", "ILU(1)"), ("ILU2", "ILU(2)"), ("ILUT", "ILUT"), ("Diagonal", "Jacobi"),
                  ("None", "None")], section="Linear Solver", visible=lambda p: p.get("lin") == "iterative"),
    Prop("tol", "Relative tolerance:", default="1e-10", section="Linear Solver", visible=lambda p: p.get("lin") != "direct"),
    Prop("maxit", "Maximum number of iterations:", "int", 1000, section="Linear Solver",
         visible=lambda p: p.get("lin") != "direct"),
    Prop("nl_maxit", "Maximum nonlinear iterations:", "int", 0, section="Nonlinear Solver",
         tip="0 = interface default"),
    Prop("nl_tol", "Nonlinear tolerance:", default="1e-7", section="Nonlinear Solver"),
    Prop("relax", "Damping (relaxation) factor:", default="1", section="Nonlinear Solver"),
    Prop("extra", "Additional SIF keywords:", "text", "", section="Advanced"),
]))

# --------------------------------------------------------------------------- results
CMAPS = [(c, c) for c in ("RainbowLight", "Rainbow", "ThermalLight", "Thermal", "Wave", "WaveLight", "Cyclic",
                          "GrayScale", "Traffic", "Viridis", "Cividis", "Plasma", "Magma", "Jupiter", "Prism")]
register(NodeType("results", "Results", "results", "results", label="Results", numbered=False, deletable=False,
                  can_disable=False, children=["pg3d", "pg2d", "pg1d", "pgpolar", "results.datasets", "results.derived",
                                               "results.tables", "results.export"]))
register(NodeType("results.datasets", "Datasets", "dsets", "dataset", label="Datasets", numbered=False, deletable=False,
                  can_disable=False, children=["dset.solution", "dset.cutline", "dset.cutpoint", "dset.cutplane",
                                               "dset.parext", "dset.revolution", "dset.mirror", "dset.grid3d",
                                               "dset.grid2d", "dset.paramcurve"]))
register(NodeType("results.derived", "Derived Values", "dvals", "derived_values", label="Derived Values", numbered=False,
                  deletable=False, can_disable=False,
                  children=["eval.point", "eval.volint", "eval.surfint", "eval.avg", "eval.max", "eval.min",
                            "eval.global"]))
register(NodeType("results.tables", "Tables", "tbls", "table", label="Tables", numbered=False, deletable=False,
                  can_disable=False, children=["table"]))
register(NodeType("results.export", "Export", "exps", "export", label="Export", numbered=False, deletable=False,
                  can_disable=False, children=["export.image", "export.data", "export.vtu", "export.report"]))

register(NodeType("dset.solution", "Solution", "dset", "solution", props=[
    Prop("study", "Study:", "study_ref", "", section="Solution"),
    Prop("sweepidx", "Parameter value index (sweep, -1 = last):", "string", "-1", section="Solution"),
]))
register(NodeType("dset.cutline", "Cut Line", "cln", "cut_line", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("x0", "Point 1, x:", default="0", unit="LEN", section="Line Data"),
    Prop("y0", "Point 1, y:", default="0", unit="LEN", section="Line Data"),
    Prop("z0", "Point 1, z:", default="0", unit="LEN", section="Line Data", visible=lambda p: p.get("__sdim", 3) == 3),
    Prop("x1", "Point 2, x:", default="1", unit="LEN", section="Line Data"),
    Prop("y1", "Point 2, y:", default="0", unit="LEN", section="Line Data"),
    Prop("z1", "Point 2, z:", default="0", unit="LEN", section="Line Data", visible=lambda p: p.get("__sdim", 3) == 3),
    Prop("res", "Resolution (points):", "int", 200, section="Line Data"),
]))
register(NodeType("dset.cutpoint", "Cut Point", "cpt", "cut_point", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("x", "x:", "string", "0", section="Point Data", tip="Space-separated list for several points"),
    Prop("y", "y:", "string", "0", section="Point Data"),
    Prop("z", "z:", "string", "0", section="Point Data", visible=lambda p: p.get("__sdim", 3) == 3),
    Prop("snap", "Snap to closest boundary", "bool", False, section="Point Data"),
]))
register(NodeType("dset.cutplane", "Cut Plane", "cpl", "cut_plane", dims=(3,), props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("ptype", "Plane type:", "choice", "general", choices=[("quick", "Quick"), ("general", "General")],
         section="Plane Data"),
    Prop("plane", "Plane:", "choice", "yz", choices=[("xy", "xy-planes"), ("yz", "yz-planes"), ("zx", "zx-planes")],
         section="Plane Data", visible=lambda p: p.get("ptype") == "quick"),
    Prop("coord", "Coordinate:", default="0", unit="LEN", section="Plane Data", visible=lambda p: p.get("ptype") == "quick"),
    Prop("px", "Point, x:", default="0", unit="LEN", section="Plane Data", visible=lambda p: p.get("ptype") != "quick"),
    Prop("py", "Point, y:", default="0", unit="LEN", section="Plane Data", visible=lambda p: p.get("ptype") != "quick"),
    Prop("pz", "Point, z:", default="0", unit="LEN", section="Plane Data", visible=lambda p: p.get("ptype") != "quick"),
    Prop("nx", "Normal, x:", default="1", section="Plane Data", visible=lambda p: p.get("ptype") != "quick"),
    Prop("ny", "Normal, y:", default="0", section="Plane Data", visible=lambda p: p.get("ptype") != "quick"),
    Prop("nz", "Normal, z:", default="0", section="Plane Data", visible=lambda p: p.get("ptype") != "quick"),
]))
register(NodeType("dset.parext", "Parametric Extrusion 2D", "pext", "dataset", props=[
    Prop("data", "Dataset (cut plane or 2D solution):", "dataset", "", section="Data"),
    Prop("values", "Parameter values (indices from 1, empty = all):", "string", "", section="Parameter Selection"),
    Prop("levelscale", "Level scale factor:", default="1", section="Settings",
         tip="Height of a layer = parameter value x this factor (in geometry units)"),
]))
register(NodeType("dset.revolution", "Revolution 2D", "rev", "revolve", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("start", "Start angle:", default="-90", unit="deg", section="Revolution Layers"),
    Prop("angle", "Revolution angle:", default="225", unit="deg", section="Revolution Layers"),
    Prop("layers", "Number of layers:", "int", 60, section="Revolution Layers"),
]))
for _k, _t, _d in (("dset.grid3d", "Grid 3D", 3), ("dset.grid2d", "Grid 2D", 2)):
    register(NodeType(_k, _t, "grid", "grid_dataset",
                      description="Regular grid of points: mesh values inside the mesh, exterior-field variables "
                                  "(acpr.efc1.pext, acpr.efc1.Lp_pext) everywhere.", props=[
        Prop("data", "Dataset:", "dataset", "", section="Data"),
        *[p for a in "xyz"[:_d] for p in (
            Prop(f"{a}min", f"{a} minimum (empty = dataset bounds):", "string", "", section="Parameter Bounds"),
            Prop(f"{a}max", f"{a} maximum:", "string", "", section="Parameter Bounds"),
            Prop(f"{a}res", f"{a} resolution (points):", "int", 40 if _d == 3 else 120, section="Grid"))]]))
register(NodeType("dset.paramcurve", "Parametric Curve 3D", "pc", "parametric_curve",
                  description="A curve x(s), y(s), z(s) in the geometry length unit, e.g. a spiral around a loudspeaker.",
                  props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("par", "Parameter name:", "string", "s", section="Parameter"),
    Prop("smin", "Minimum:", "string", "0", section="Parameter"),
    Prop("smax", "Maximum:", "string", "1", section="Parameter"),
    Prop("npts", "Resolution (points):", "int", 400, section="Parameter"),
    Prop("xexpr", "x:", "string", "cos(2*pi*s)", section="Expressions"),
    Prop("yexpr", "y:", "string", "sin(2*pi*s)", section="Expressions"),
    Prop("zexpr", "z:", "string", "s", section="Expressions"),
]))
register(NodeType("dset.mirror", "Mirror", "mir", "mirror", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("axis", "Plane/line of reflection:", "choice", "x",
         choices=[("x", "x = 0 (mirror in x)"), ("y", "y = 0 (mirror in y)"), ("z", "z = 0 (mirror in z)"),
                  ("xy", "x = 0 and y = 0 (four copies)")], section="Plane Data"),
]))

_expr_tbl = Prop("exprs", "", "expr_table", [{"expr": "", "unit": "", "descr": ""}], section="Expressions",
                 columns=[("expr", "Expression"), ("unit", "Unit"), ("descr", "Description")])
for kind, title, prefix, icon, level in [
        ("eval.volint", "Volume Integration", "int", "integration", "domain"),
        ("eval.surfint", "Surface Integration", "int", "integration", "boundary"),
        ("eval.avg", "Volume Average", "av", "average", "domain"),
        ("eval.max", "Volume Maximum", "max", "maxmin", "domain"),
        ("eval.min", "Volume Minimum", "min", "maxmin", "domain")]:
    register(NodeType(kind, title, prefix, icon, selection=level, selection_all=True, props=[
        Prop("data", "Dataset:", "dataset", "", section="Data"),
        Prop("solnum", "Time/parameter selection:", "choice", "last", choices=[("last", "Last"), ("all", "All")],
             section="Data"),
        Prop("table", "Table:", "string", "", section="Data", readonly=True),
        _expr_tbl]))
register(NodeType("eval.point", "Point Evaluation", "pev", "point_evaluation", selection="point",
                  description="Click geometry points in the Graphics window (or give coordinates).", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("solnum", "Time/parameter selection:", "choice", "last", choices=[("last", "Last"), ("all", "All")], section="Data"),
    Prop("x", "x (if no points selected):", "string", "0", section="Point"),
    Prop("y", "y:", "string", "0", section="Point"),
    Prop("z", "z:", "string", "0", section="Point", visible=lambda p: p.get("__sdim", 3) == 3),
    _expr_tbl]))
register(NodeType("eval.global", "Global Evaluation", "gev", "global_evaluation", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("solnum", "Time/parameter selection:", "choice", "all", choices=[("last", "Last"), ("all", "All")],
         section="Data"),
    Prop("exprs", "", "expr_table", [], section="Expressions",
         columns=[("expr", "Expression"), ("unit", "Unit"), ("descr", "Description")],
         tip="Scalar expressions such as aveop1(T) or intop1(ec.Qrh); empty = eigenfrequencies/times"),
]))
register(NodeType("table", "Table", "tbl", "table", props=[
    Prop("columns", "", "info", [], section="Data"), Prop("rows", "", "info", [], section="Data")]))

_dataset = Prop("data", "Dataset:", "dataset", "", section="Data")
_solsel = Prop("solnum", "Solution selection:", "solnum", "last", section="Data")
_TITLE = [Prop("title_type", "Title type:", "choice", "auto",
               choices=[("auto", "Automatic"), ("manual", "Manual"), ("none", "None")], section="Title"),
          Prop("title", "Title:", "string", "", section="Title", visible=lambda p: p.get("title_type") == "manual")]
_LEGEND = [Prop("legend", "Show legends", "bool", True, section="Color Legend"),
           Prop("maxmin", "Show maximum and minimum values", "bool", True, section="Color Legend"),
           Prop("legendpos", "Position:", "choice", "right", choices=[("right", "Right"), ("left", "Left")],
                section="Color Legend")]
_PG_SETTINGS = [Prop("edges", "Plot dataset edges", "bool", True, section="Plot Settings")]
_PLOTS_3D = ["plot.surface", "plot.volume", "plot.slice", "plot.multislice", "plot.isosurface", "plot.contour",
             "plot.arrow", "plot.streamline", "plot.mesh", "plot.line", "plot.annotation", "plot.radiation"]
_PLOTS_2D = ["plot.surface", "plot.contour", "plot.arrow", "plot.streamline", "plot.mesh", "plot.line", "plot.annotation"]
register(NodeType("pg3d", "3D Plot Group", "pg", "plot_group_3d", children=_PLOTS_3D, props=[
    _dataset, _solsel, *_TITLE, *_LEGEND, *_PG_SETTINGS,
    Prop("frame", "Frame:", "choice", "material", choices=[("material", "Material (undeformed)")], section="Plot Settings"),
]))
register(NodeType("pg2d", "2D Plot Group", "pg", "plot_group_2d", children=_PLOTS_2D, props=[
    _dataset, _solsel, *_TITLE, *_LEGEND, *_PG_SETTINGS,
    Prop("revolve", "Show 2D axisymmetric revolution (3D)", "bool", False, section="Plot Settings",
         visible=lambda p: p.get("__axi", False)),
]))
register(NodeType("pg1d", "1D Plot Group", "pg", "plot_group_1d",
                  children=["plot.linegraph", "plot.globalgraph", "plot.pointgraph", "plot.tablegraph",
                            "plot.directivity"], props=[
    _dataset, _solsel, *_TITLE,
    Prop("xlabel", "x-axis label:", "string", "", section="Axis"),
    Prop("ylabel", "y-axis label:", "string", "", section="Axis"),
    Prop("xlog", "x-axis log scale", "bool", False, section="Axis"),
    Prop("ylog", "y-axis log scale", "bool", False, section="Axis"),
    Prop("grid", "Show grid", "bool", True, section="Grid"),
    Prop("legend", "Show legends", "bool", True, section="Legend"),
    Prop("legendpos", "Position:", "choice", "upper right",
         choices=[(v, v.capitalize()) for v in ("upper right", "upper left", "lower right", "lower left", "center right",
                                                "center left", "upper center", "lower center", "best")], section="Legend"),
]))
register(NodeType("pgpolar", "Polar Plot Group", "pg", "plot_group_1d", children=["plot.polar", "plot.radiation"], props=[
    _dataset, *_TITLE,
    Prop("legend", "Show legends", "bool", True, section="Legend"),
    Prop("zero", "Zero angle:", "choice", "E", choices=[("E", "Right (0 deg = +x)"), ("N", "Top (0 deg = +y)")],
         section="Axis"),
]))


def _plot_props(extra=(), coloring=True, inherit=True):
    p = [Prop("data", "Dataset:", "dataset", "fromparent", section="Data"),
         Prop("expr", "Expression:", "rexpr", "", section="Expression"),
         Prop("unit", "Unit:", "runit", "", section="Expression"),
         Prop("descr", "Description:", "string", "", section="Expression"),
         *extra]
    if coloring:
        p += [Prop("coloring", "Coloring:", "choice", "table", choices=[("table", "Color table"), ("uniform", "Uniform")],
                   section="Coloring and Style"),
              Prop("cmap", "Color table:", "choice", "RainbowLight", choices=CMAPS, section="Coloring and Style",
                   visible=lambda q: q.get("coloring", "table") == "table"),
              Prop("ucolor", "Color:", "choice", "black",
                   choices=[(c, c.capitalize()) for c in ("black", "white", "gray", "red", "green", "blue", "cyan",
                                                         "magenta", "yellow")], section="Coloring and Style",
                   visible=lambda q: q.get("coloring") == "uniform"),
              Prop("colorleg", "Color legend", "bool", True, section="Coloring and Style"),
              Prop("reverse", "Reverse color table", "bool", False, section="Coloring and Style"),
              Prop("symmetric", "Symmetrize color range", "bool", False, section="Coloring and Style"),
              Prop("manual", "Manual color range", "bool", False, section="Range"),
              Prop("vmin", "Minimum:", "string", "0", section="Range", visible=lambda q: q.get("manual", False)),
              Prop("vmax", "Maximum:", "string", "1", section="Range", visible=lambda q: q.get("manual", False))]
    return p


_PLOT_CHILD = ["plot.deform", "plot.selection"]
_VEC_CHILD = ["plot.deform", "plot.selection", "plot.colorexpr"]
register(NodeType("plot.surface", "Surface", "surf", "surface_plot", children=_PLOT_CHILD, props=_plot_props(
    [Prop("edges", "Show element edges", "bool", False, section="Coloring and Style"),
     Prop("wireframe", "Wireframe", "bool", False, section="Coloring and Style")])))
register(NodeType("plot.volume", "Volume", "vol", "volume_plot", children=_PLOT_CHILD, props=_plot_props(
    [Prop("edges", "Show element edges", "bool", False, section="Coloring and Style"),
     Prop("wireframe", "Wireframe", "bool", False, section="Coloring and Style")])))
register(NodeType("plot.slice", "Slice", "slc", "slice_plot", children=_PLOT_CHILD, props=_plot_props([
    Prop("plane", "Plane type:", "choice", "yz", choices=[("yz", "yz-planes"), ("zx", "zx-planes"), ("xy", "xy-planes")],
         section="Plane Data"),
    Prop("entry", "Entry method:", "choice", "number", choices=[("number", "Number of planes"), ("coords", "Coordinates")],
         section="Plane Data"),
    Prop("nplanes", "Planes:", "int", 5, section="Plane Data", visible=lambda p: p.get("entry", "number") == "number"),
    Prop("coords", "Coordinates:", "string", "0", section="Plane Data", visible=lambda p: p.get("entry") == "coords",
         tip="Space-separated coordinates in the geometry length unit"),
    Prop("shift", "Shift:", default="0", unit="LEN", section="Plane Data",
         tip="Interactive positioning: moves all slices along their normal")])))
register(NodeType("plot.multislice", "Multislice", "mslc", "slice_plot", children=_PLOT_CHILD, props=_plot_props([
    Prop("nx", "x-planes:", "int", 1, section="Multiplane Data"),
    Prop("ny", "y-planes:", "int", 1, section="Multiplane Data"),
    Prop("nz", "z-planes:", "int", 1, section="Multiplane Data")])))
register(NodeType("plot.isosurface", "Isosurface", "iso", "isosurface", children=_PLOT_CHILD, props=_plot_props([
    Prop("levels", "Total levels:", "int", 5, section="Levels"),
    Prop("legendtype", "Legend type:", "choice", "filled", choices=[("filled", "Filled"), ("continuous", "Continuous")],
         section="Coloring and Style")])))
register(NodeType("plot.contour", "Contour", "con", "contour", children=_PLOT_CHILD, props=_plot_props([
    Prop("levels", "Total levels:", "int", 20, section="Levels"),
    Prop("ctype", "Contour type:", "choice", "lines", choices=[("lines", "Lines"), ("filled", "Filled"), ("tubes", "Tubes")],
         section="Coloring and Style"),
    Prop("labels", "Level labels", "bool", False, section="Coloring and Style"),
    Prop("legendtype", "Legend type:", "choice", "filled", choices=[("filled", "Filled"), ("continuous", "Continuous")],
         section="Coloring and Style"),
    Prop("filled", "Filled", "bool", False, section="Coloring and Style", visible=lambda p: False)])))
register(NodeType("plot.arrow", "Arrow Volume", "arwv", "arrow_plot", children=_VEC_CHILD, props=[
    Prop("data", "Dataset:", "dataset", "fromparent", section="Data"),
    Prop("expr", "Vector expression:", "rvexpr", "", section="Expression"),
    Prop("positioning", "Placement:", "choice", "grid", choices=[("grid", "Grid points"), ("random", "Random points")],
         section="Arrow Positioning"),
    Prop("xg", "x grid points:", "int", 15, section="Arrow Positioning", visible=lambda p: p.get("positioning", "grid") == "grid"),
    Prop("yg", "y grid points:", "int", 15, section="Arrow Positioning", visible=lambda p: p.get("positioning", "grid") == "grid"),
    Prop("zg", "z grid points:", "int", 15, section="Arrow Positioning",
         visible=lambda p: p.get("positioning", "grid") == "grid" and p.get("__sdim", 3) == 3),
    Prop("npts", "Number of points:", "int", 400, section="Arrow Positioning", visible=lambda p: p.get("positioning") == "random"),
    Prop("arrowlen", "Arrow length:", "choice", "proportional",
         choices=[("proportional", "Proportional"), ("normalized", "Normalized")], section="Coloring and Style"),
    Prop("scale_auto", "Automatic scale", "bool", True, section="Coloring and Style"),
    Prop("scale", "Scale factor:", "string", "1", section="Coloring and Style", visible=lambda p: not p.get("scale_auto", True)),
    Prop("color", "Color:", "choice", "red", choices=[("red", "Red"), ("black", "Black"), ("blue", "Blue"),
                                                     ("magnitude", "Color by magnitude")], section="Coloring and Style")]))
register(NodeType("plot.streamline", "Streamline", "str", "streamline", children=_VEC_CHILD, props=[
    Prop("data", "Dataset:", "dataset", "fromparent", section="Data"),
    Prop("expr", "Vector expression:", "rvexpr", "", section="Expression"),
    Prop("positioning", "Positioning:", "choice", "uniform",
         choices=[("uniform", "Uniform density"), ("magnitude", "Magnitude controlled"),
                  ("start", "Start point controlled (selected boundaries)")], section="Streamline Positioning"),
    Prop("nseeds", "Number of streamlines:", "int", 40, section="Streamline Positioning"),
    Prop("ltype", "Line type:", "choice", "line", choices=[("line", "Line"), ("tube", "Tube"), ("ribbon", "Ribbon")],
         section="Coloring and Style"),
    Prop("width", "Width / radius (fraction of model size):", "string", "0.004", section="Coloring and Style",
         visible=lambda p: p.get("ltype", "line") != "line"),
    Prop("cmap", "Color table:", "choice", "RainbowLight", choices=CMAPS, section="Coloring and Style"),
    Prop("tube", "Line type: tube", "bool", False, section="Coloring and Style", visible=lambda p: False)]))
register(NodeType("plot.mesh", "Mesh", "mesh", "mesh_plot", children=["plot.selection"], props=[
    Prop("data", "Dataset:", "dataset", "fromparent", section="Data"),
    Prop("quality", "Color by element quality", "bool", False, section="Coloring and Style")]))
register(NodeType("plot.line", "Line", "lin", "line_graph", children=["plot.selection", "plot.colorexpr", "plot.deform"], props=[
    Prop("data", "Dataset:", "dataset", "fromparent", section="Data"),
    Prop("expr", "Expression (empty = uniform color):", "rexpr", "", section="Expression"),
    Prop("ltype", "Line type:", "choice", "line", choices=[("line", "Line"), ("tube", "Tube")], section="Coloring and Style"),
    Prop("radius", "Tube radius (fraction of model size):", "string", "0.003", section="Coloring and Style",
         visible=lambda p: p.get("ltype") == "tube"),
    Prop("cmap", "Color table:", "choice", "RainbowLight", choices=CMAPS, section="Coloring and Style"),
    Prop("color", "Color:", "choice", "black", choices=[("black", "Black"), ("gray", "Gray"), ("blue", "Blue")],
         section="Coloring and Style")]))
register(NodeType("plot.annotation", "Annotation", "ann", "rename", props=[
    Prop("text", "Text:", "string", "T = eval(T, K)", section="Annotation",
         tip="eval(expression[, unit]) is replaced by the value at the position"),
    Prop("x", "x:", default="0", unit="LEN", section="Position"),
    Prop("y", "y:", default="0", unit="LEN", section="Position"),
    Prop("z", "z:", default="0", unit="LEN", section="Position", visible=lambda p: p.get("__sdim", 3) == 3),
    Prop("point", "Show point", "bool", True, section="Coloring and Style"),
    Prop("precision", "Precision:", "int", 4, section="Coloring and Style")]))
register(NodeType("plot.deform", "Deformation", "def", "deformation", props=[
    Prop("expr", "Vector expression (e.g. disp, or 0, -50[mm]):", "rvexpr", "disp", section="Expression"),
    Prop("scale_auto", "Automatic scale factor", "bool", True, section="Scale"),
    Prop("scale", "Scale factor:", "string", "1", section="Scale", visible=lambda p: not p.get("scale_auto", True))]))
register(NodeType("plot.selection", "Selection", "sel", "selection_node", selection="boundary", props=[
    Prop("level", "Geometric entity level:", "choice", "boundary",
         choices=[("domain", "Domain"), ("boundary", "Boundary"), ("edge", "Edge")], section="Selection")]))
register(NodeType("plot.colorexpr", "Color Expression", "col", "legend", props=[
    Prop("expr", "Expression:", "rexpr", "", section="Expression"),
    Prop("unit", "Unit:", "runit", "", section="Expression"),
    Prop("cmap", "Color table:", "choice", "RainbowLight", choices=CMAPS, section="Coloring and Style"),
    Prop("colorleg", "Color legend", "bool", True, section="Coloring and Style")]))
_LINESTYLE = [Prop("lstyle", "Line style:", "choice", "-", choices=[("-", "Solid"), ("--", "Dashed"), (":", "Dotted"),
                                                                    ("-.", "Dash-dot"), ("none", "None")], section="Coloring and Style"),
              Prop("lwidth", "Width:", "string", "1.6", section="Coloring and Style"),
              Prop("lcolor", "Color:", "choice", "cycle",
                   choices=[("cycle", "Cycle"), *[(c, c.capitalize()) for c in ("black", "blue", "red", "green", "magenta",
                                                                              "cyan", "orange", "gray")]],
                   section="Coloring and Style"),
              Prop("marker", "Marker:", "choice", "none",
                   choices=[("none", "None"), ("o", "Circle"), ("+", "Plus sign"), ("*", "Asterisk"), ("s", "Square"),
                            ("^", "Triangle"), ("x", "Cross"), ("d", "Diamond")], section="Coloring and Style"),
              Prop("nmarkers", "Number of markers:", "int", 10, section="Coloring and Style",
                   visible=lambda p: p.get("marker", "none") != "none"),
              Prop("legend_text", "Legend:", "string", "", section="Legends")]
register(NodeType("plot.linegraph", "Line Graph", "lngr", "line_graph", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("expr", "y-axis expression:", "rexpr", "", section="y-Axis Data"),
    Prop("unit", "Unit:", "runit", "", section="y-Axis Data"),
    Prop("xaxis", "x-axis parameter:", "choice", "arc", choices=[("arc", "Arc length"), ("x", "x"), ("y", "y"), ("z", "z")],
         section="x-Axis Data"),
    *_LINESTYLE]))
register(NodeType("plot.globalgraph", "Global", "glob", "global_evaluation", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("expr", "Expression (global, e.g. aveop1(T), or a field to reduce):", "rexpr", "", section="y-Axis Data"),
    Prop("reduce", "Reduction:", "choice", "max", choices=[("global", "Global (scalar expression)"),
                                                           ("max", "Maximum"), ("min", "Minimum"), ("avg", "Average"),
                                                           ("point", "At point")], section="y-Axis Data"),
    Prop("px", "x:", "string", "0", section="y-Axis Data", visible=lambda p: p.get("reduce") == "point"),
    Prop("py", "y:", "string", "0", section="y-Axis Data", visible=lambda p: p.get("reduce") == "point"),
    Prop("pz", "z:", "string", "0", section="y-Axis Data", visible=lambda p: p.get("reduce") == "point" and p.get("__sdim", 3) == 3),
    *_LINESTYLE]))
register(NodeType("plot.pointgraph", "Point Graph", "ptgr", "point_evaluation", props=[
    Prop("data", "Dataset (Cut Point):", "dataset", "", section="Data"),
    Prop("expr", "y-axis expression:", "rexpr", "", section="y-Axis Data"),
    Prop("unit", "Unit:", "runit", "", section="y-Axis Data"),
    *_LINESTYLE]))
register(NodeType("plot.tablegraph", "Table Graph", "tblg", "table", props=[
    Prop("table", "Table:", "table_ref", "", section="Data"),
    Prop("xcol", "x-axis data (column, 1-based; 0 = row index):", "int", 1, section="Data"),
    Prop("ycols", "Plot columns (comma-separated, empty = all others):", "string", "", section="Data"),
    *_LINESTYLE]))
_RADIATION_GEOM = [
    Prop("xc", "Center, x:", default="0", unit="LEN", section="Evaluation"),
    Prop("yc", "Center, y:", default="0", unit="LEN", section="Evaluation", visible=lambda p: not p.get("__axi", False)),
    Prop("zc", "Center, z:", default="0", unit="LEN", section="Evaluation",
         visible=lambda p: p.get("__sdim", 3) == 3 or p.get("__axi", False)),
    Prop("radius", "Evaluation distance:", default="1[m]", unit="LEN", section="Evaluation",
         tip="Distance from the center (COMSOL: Evaluation distance from origin)"),
    Prop("plane", "Plane:", "choice", "xz",
         choices=[("xz", "xz-plane (angle from x toward z)"), ("xy", "xy-plane (angle from x toward y)"),
                  ("yz", "yz-plane (angle from y toward z)")], section="Evaluation",
         visible=lambda p: p.get("__sdim", 3) == 3, tip="Axisymmetric models: the rz half plane"),
    Prop("npts", "Number of angles:", "int", 181, section="Evaluation"),
]
register(NodeType("plot.radiation", "Radiation Pattern", "rp", "far_field",
                  description="Exterior-field quantity on a circle (polar plot group) or a sphere (3D plot group).",
                  props=[
    Prop("data", "Dataset:", "dataset", "fromparent", section="Data"),
    Prop("expr", "Expression:", "rexpr", "acpr.efc1.Lp_pext", section="Expression"),
    Prop("unit", "Unit:", "runit", "", section="Expression"),
    *_RADIATION_GEOM,
    Prop("a0", "Start angle:", default="-90", unit="deg", section="Evaluation"),
    Prop("a1", "End angle:", default="90", unit="deg", section="Evaluation"),
    Prop("normalize", "Normalization:", "choice", "none",
         choices=[("none", "None"), ("max", "Maximum = 0 dB (or 1)")], section="Evaluation"),
    Prop("range", "Dynamic range (3D pattern):", "string", "40", section="Evaluation",
         tip="Values more than this below the maximum are drawn at the center"),
    Prop("params", "Parameter values (indices from 1, empty = current):", "string", "", section="Data"),
    *_LINESTYLE]))
register(NodeType("plot.directivity", "Directivity", "dir", "far_field",
                  description="Exterior-field level versus angle and frequency, normalized to the on-axis value.",
                  props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("expr", "Expression:", "rexpr", "acpr.efc1.Lp_pext", section="Expression"),
    *_RADIATION_GEOM,
    Prop("a0", "Start angle (from the axis):", default="-90", unit="deg", section="Evaluation"),
    Prop("a1", "End angle:", default="90", unit="deg", section="Evaluation"),
    Prop("axis", "Reference axis:", "choice", "z", choices=[("z", "+z"), ("x", "+x"), ("y", "+y")], section="Evaluation",
         tip="Angle 0 points along this axis"),
    Prop("normalize", "Normalization:", "choice", "axis",
         choices=[("axis", "On-axis value = 0 dB"), ("max", "Maximum = 0 dB"), ("none", "None")], section="Evaluation"),
    Prop("levels", "Contour levels:", "int", 13, section="Coloring and Style"),
    Prop("cmap", "Color table:", "choice", "RainbowLight", choices=CMAPS, section="Coloring and Style"),
    Prop("flog", "Logarithmic frequency axis", "bool", True, section="Coloring and Style")]))
register(NodeType("plot.polar", "Line Graph (Polar)", "plgr", "line_graph", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("expr", "Radius expression:", "rexpr", "", section="r-Axis Data"),
    Prop("unit", "Unit:", "runit", "", section="r-Axis Data"),
    Prop("xc", "Center, x:", default="0", unit="LEN", section="Evaluation Circle"),
    Prop("yc", "Center, y:", default="0", unit="LEN", section="Evaluation Circle"),
    Prop("radius", "Radius:", default="1", unit="LEN", section="Evaluation Circle"),
    Prop("a0", "Start angle:", default="0", unit="deg", section="Evaluation Circle"),
    Prop("a1", "End angle:", default="360", unit="deg", section="Evaluation Circle"),
    Prop("npts", "Points:", "int", 181, section="Evaluation Circle"),
    Prop("params", "Parameter values (indices from 1, empty = current):", "string", "", section="Data"),
    *_LINESTYLE]))

register(NodeType("export.image", "Image", "img", "image_export", props=[
    Prop("source", "Plot group:", "plotgroup", "", section="Image"),
    Prop("file", "Filename:", "savefile", "", section="Output"),
    Prop("w", "Width (px):", "int", 1600, section="Image"), Prop("h", "Height (px):", "int", 1000, section="Image")]))
register(NodeType("export.data", "Data", "data", "data_export", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("expr", "Expressions (comma-separated):", "string", "", section="Expressions"),
    Prop("file", "Filename:", "savefile", "", section="Output")]))
register(NodeType("export.vtu", "VTK Unstructured Grid", "vtu", "data_export", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("file", "Filename:", "savefile", "", section="Output")]))
register(NodeType("export.report", "Report", "rpt", "report", props=[
    Prop("file", "Filename:", "savefile", "", section="Output"),
    Prop("images", "Include plot images", "bool", True, section="Content")]))
