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
                  deletable=False, can_disable=False, children=["variables"]))
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
    Prop("pname", "Parameter name:", "string", "", section="Study Settings"),
    Prop("pvalues", "Parameter value list:", "string", "range(1,1,3)", section="Study Settings"),
    Prop("punit", "Parameter unit:", "string", "", section="Study Settings"),
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
                  can_disable=False, children=["pg3d", "pg2d", "pg1d", "results.datasets", "results.derived",
                                               "results.tables", "results.export"]))
register(NodeType("results.datasets", "Datasets", "dsets", "dataset", label="Datasets", numbered=False, deletable=False,
                  can_disable=False, children=["dset.solution", "dset.cutline", "dset.cutpoint"]))
register(NodeType("results.derived", "Derived Values", "dvals", "derived_values", label="Derived Values", numbered=False,
                  deletable=False, can_disable=False,
                  children=["eval.point", "eval.volint", "eval.surfint", "eval.avg", "eval.max", "eval.global"]))
register(NodeType("results.tables", "Tables", "tbls", "table", label="Tables", numbered=False, deletable=False,
                  can_disable=False, children=["table"]))
register(NodeType("results.export", "Export", "exps", "export", label="Export", numbered=False, deletable=False,
                  can_disable=False, children=["export.image", "export.data", "export.vtu", "export.report"]))

register(NodeType("dset.solution", "Solution", "dset", "solution", props=[
    Prop("study", "Study:", "study_ref", "", section="Solution"),
    Prop("sweepidx", "Parameter value (sweep):", "int", 0, section="Solution"),
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
    Prop("x", "x:", "string", "0", section="Point Data"), Prop("y", "y:", "string", "0", section="Point Data"),
    Prop("z", "z:", "string", "0", section="Point Data", visible=lambda p: p.get("__sdim", 3) == 3),
]))

_expr_tbl = Prop("exprs", "", "expr_table", [{"expr": "", "unit": "", "descr": ""}], section="Expressions",
                 columns=[("expr", "Expression"), ("unit", "Unit"), ("descr", "Description")])
for kind, title, prefix, icon, level in [
        ("eval.volint", "Volume Integration", "int", "integration", "domain"),
        ("eval.surfint", "Surface Integration", "int", "integration", "boundary"),
        ("eval.avg", "Volume Average", "av", "average", "domain"),
        ("eval.max", "Volume Maximum", "max", "maxmin", "domain")]:
    register(NodeType(kind, title, prefix, icon, selection=level, selection_all=True, props=[
        Prop("data", "Dataset:", "dataset", "", section="Data"),
        Prop("solnum", "Time/parameter selection:", "choice", "last", choices=[("last", "Last"), ("all", "All")],
             section="Data"),
        Prop("table", "Table:", "string", "", section="Data", readonly=True),
        _expr_tbl]))
register(NodeType("eval.point", "Point Evaluation", "pev", "point_evaluation", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("solnum", "Time/parameter selection:", "choice", "last", choices=[("last", "Last"), ("all", "All")], section="Data"),
    Prop("x", "x:", "string", "0", section="Point"), Prop("y", "y:", "string", "0", section="Point"),
    Prop("z", "z:", "string", "0", section="Point", visible=lambda p: p.get("__sdim", 3) == 3),
    _expr_tbl]))
register(NodeType("eval.global", "Global Evaluation", "gev", "global_evaluation", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("what", "Quantity:", "choice", "auto",
         choices=[("auto", "Eigenfrequencies / frequencies / times"), ("norms", "Solution norms")], section="Expressions"),
]))
register(NodeType("table", "Table", "tbl", "table", props=[
    Prop("columns", "", "info", [], section="Data"), Prop("rows", "", "info", [], section="Data")]))

_dataset = Prop("data", "Dataset:", "dataset", "", section="Data")
_solsel = Prop("solnum", "Solution selection:", "solnum", "last", section="Data")
register(NodeType("pg3d", "3D Plot Group", "pg", "plot_group_3d", children=[
    "plot.surface", "plot.volume", "plot.slice", "plot.isosurface", "plot.arrow", "plot.streamline", "plot.mesh",
    "plot.line"], props=[
    _dataset, _solsel,
    Prop("title_type", "Title type:", "choice", "auto", choices=[("auto", "Automatic"), ("manual", "Manual"), ("none", "None")],
         section="Title"),
    Prop("title", "Title:", "string", "", section="Title", visible=lambda p: p.get("title_type") == "manual"),
    Prop("legend", "Show legends", "bool", True, section="Color Legend"),
    Prop("frame", "Frame:", "choice", "material", choices=[("material", "Material (undeformed)")], section="Plot Settings"),
]))
register(NodeType("pg2d", "2D Plot Group", "pg", "plot_group_2d", children=[
    "plot.surface", "plot.contour", "plot.arrow", "plot.streamline", "plot.mesh", "plot.line"], props=[
    _dataset, _solsel,
    Prop("title_type", "Title type:", "choice", "auto", choices=[("auto", "Automatic"), ("manual", "Manual"), ("none", "None")],
         section="Title"),
    Prop("title", "Title:", "string", "", section="Title", visible=lambda p: p.get("title_type") == "manual"),
    Prop("legend", "Show legends", "bool", True, section="Color Legend"),
    Prop("revolve", "Show 2D axisymmetric revolution (3D)", "bool", False, section="Plot Settings",
         visible=lambda p: p.get("__axi", False)),
]))
register(NodeType("pg1d", "1D Plot Group", "pg", "plot_group_1d", children=["plot.linegraph", "plot.globalgraph"], props=[
    _dataset, _solsel,
    Prop("title_type", "Title type:", "choice", "auto", choices=[("auto", "Automatic"), ("manual", "Manual"), ("none", "None")],
         section="Title"),
    Prop("title", "Title:", "string", "", section="Title", visible=lambda p: p.get("title_type") == "manual"),
    Prop("xlabel", "x-axis label:", "string", "", section="Axis"),
    Prop("ylabel", "y-axis label:", "string", "", section="Axis"),
    Prop("grid", "Show grid", "bool", True, section="Grid"),
    Prop("legend", "Show legends", "bool", True, section="Legend"),
]))


def _plot_props(extra=(), coloring=True, inherit=True):
    p = [Prop("data", "Dataset:", "dataset", "fromparent", section="Data"),
         Prop("expr", "Expression:", "rexpr", "", section="Expression"),
         Prop("unit", "Unit:", "runit", "", section="Expression"),
         Prop("descr", "Description:", "string", "", section="Expression"),
         *extra]
    if coloring:
        p += [Prop("cmap", "Color table:", "choice", "RainbowLight", choices=CMAPS, section="Coloring and Style"),
              Prop("colorleg", "Color legend", "bool", True, section="Coloring and Style"),
              Prop("reverse", "Reverse color table", "bool", False, section="Coloring and Style"),
              Prop("manual", "Manual color range", "bool", False, section="Range"),
              Prop("vmin", "Minimum:", "string", "0", section="Range", visible=lambda q: q.get("manual", False)),
              Prop("vmax", "Maximum:", "string", "1", section="Range", visible=lambda q: q.get("manual", False))]
    return p


_PLOT_CHILD = ["plot.deform"]
register(NodeType("plot.surface", "Surface", "surf", "surface_plot", children=_PLOT_CHILD, props=_plot_props(
    [Prop("edges", "Show element edges", "bool", False, section="Coloring and Style")])))
register(NodeType("plot.volume", "Volume", "vol", "volume_plot", children=_PLOT_CHILD, props=_plot_props(
    [Prop("edges", "Show element edges", "bool", False, section="Coloring and Style")])))
register(NodeType("plot.slice", "Slice", "slc", "slice_plot", children=_PLOT_CHILD, props=_plot_props([
    Prop("plane", "Plane type:", "choice", "yz", choices=[("yz", "yz-planes"), ("zx", "zx-planes"), ("xy", "xy-planes")],
         section="Plane Data"),
    Prop("nplanes", "Planes:", "int", 5, section="Plane Data")])))
register(NodeType("plot.isosurface", "Isosurface", "iso", "isosurface", children=_PLOT_CHILD, props=_plot_props([
    Prop("levels", "Total levels:", "int", 5, section="Levels")])))
register(NodeType("plot.contour", "Contour", "con", "contour", children=_PLOT_CHILD, props=_plot_props([
    Prop("levels", "Total levels:", "int", 20, section="Levels"),
    Prop("filled", "Filled", "bool", False, section="Coloring and Style")])))
register(NodeType("plot.arrow", "Arrow Volume", "arwv", "arrow_plot", children=_PLOT_CHILD, props=[
    Prop("data", "Dataset:", "dataset", "fromparent", section="Data"),
    Prop("expr", "Vector expression:", "rvexpr", "", section="Expression"),
    Prop("npts", "Number of points:", "int", 400, section="Arrow Positioning"),
    Prop("scale_auto", "Automatic scale", "bool", True, section="Coloring and Style"),
    Prop("scale", "Scale factor:", "string", "1", section="Coloring and Style", visible=lambda p: not p.get("scale_auto", True)),
    Prop("color", "Color:", "choice", "red", choices=[("red", "Red"), ("black", "Black"), ("blue", "Blue"),
                                                     ("magnitude", "Color by magnitude")], section="Coloring and Style")]))
register(NodeType("plot.streamline", "Streamline", "str", "streamline", children=_PLOT_CHILD, props=[
    Prop("data", "Dataset:", "dataset", "fromparent", section="Data"),
    Prop("expr", "Vector expression:", "rvexpr", "", section="Expression"),
    Prop("nseeds", "Number of streamlines:", "int", 40, section="Streamline Positioning"),
    Prop("cmap", "Color table:", "choice", "RainbowLight", choices=CMAPS, section="Coloring and Style"),
    Prop("tube", "Line type: tube", "bool", False, section="Coloring and Style")]))
register(NodeType("plot.mesh", "Mesh", "mesh", "mesh_plot", props=[
    Prop("data", "Dataset:", "dataset", "fromparent", section="Data"),
    Prop("quality", "Color by element quality", "bool", False, section="Coloring and Style")]))
register(NodeType("plot.line", "Line", "lin", "line_graph", props=[
    Prop("data", "Dataset:", "dataset", "fromparent", section="Data"),
    Prop("color", "Color:", "choice", "black", choices=[("black", "Black"), ("gray", "Gray"), ("blue", "Blue")],
         section="Coloring and Style")]))
register(NodeType("plot.deform", "Deformation", "def", "deformation", props=[
    Prop("expr", "Vector expression:", "rvexpr", "disp", section="Expression"),
    Prop("scale_auto", "Automatic scale factor", "bool", True, section="Scale"),
    Prop("scale", "Scale factor:", "string", "1", section="Scale", visible=lambda p: not p.get("scale_auto", True))]))
register(NodeType("plot.linegraph", "Line Graph", "lngr", "line_graph", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("expr", "y-axis expression:", "rexpr", "", section="y-Axis Data"),
    Prop("unit", "Unit:", "runit", "", section="y-Axis Data"),
    Prop("xaxis", "x-axis parameter:", "choice", "arc", choices=[("arc", "Arc length"), ("x", "x"), ("y", "y"), ("z", "z")],
         section="x-Axis Data"),
    Prop("legend_text", "Legend:", "string", "", section="Legends")]))
register(NodeType("plot.globalgraph", "Global", "glob", "global_evaluation", props=[
    Prop("data", "Dataset:", "dataset", "", section="Data"),
    Prop("expr", "Expression (evaluated at a point or 'max'/'min'/'avg' of):", "rexpr", "", section="y-Axis Data"),
    Prop("reduce", "Reduction:", "choice", "max", choices=[("max", "Maximum"), ("min", "Minimum"), ("avg", "Average"),
                                                           ("point", "At point")], section="y-Axis Data"),
    Prop("px", "x:", "string", "0", section="y-Axis Data", visible=lambda p: p.get("reduce") == "point"),
    Prop("py", "y:", "string", "0", section="y-Axis Data", visible=lambda p: p.get("reduce") == "point"),
    Prop("pz", "z:", "string", "0", section="y-Axis Data", visible=lambda p: p.get("reduce") == "point" and p.get("__sdim", 3) == 3)]))

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
