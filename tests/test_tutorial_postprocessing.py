"""COMSOL 'Essentials of Postprocessing and Visualization' walkthroughs, reproduced with Elmer Studio.

The guide's models (pulley, heat sink, loudspeaker, absorber, horn) ship as COMSOL files, which are not
available; these are parametric stand-ins that exercise the same postprocessing workflow:

* Pulley: 2D plane-stress disk with a hub and six openings spinning in a Rotating Frame (Omega = n[rpm]),
  Parametric Sweep over n; Surface Maximum of displacement and von Mises stress over the sweep; Cut Point
  (19, 54) mm with a Point Graph; Point Evaluation on two selected geometry points; Integration couplings on
  the far-right and far-left points, a variable diam = intop1(R+u) - intop2(-R+u), Global Evaluation of
  diam - 2*R and a Table Graph of the result.
* Lame verification: a spinning quarter annulus against the closed-form plane-stress solution.
* Datasets: Cut Plane, Mirror and Parametric Extrusion views of a 3D solution, Revolution of a 2D one, and a
  polar plot of a rotationally symmetric field.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core.geometry import build_geometry  # noqa: E402

RPM = 2 * math.pi / 60


def pulley_model(sweep="range(2000,2000,8000)"):
    m = B.new_model("2D", ["solid"], "stationary")
    m.root.label = "pulley_postprocessing"
    m.global_defs.child("params").props["table"] = [
        {"name": "n", "expr": "4000", "descr": "Rotational speed (rpm)"},
        {"name": "R", "expr": "60[mm]", "descr": "Pulley radius"},
        {"name": "Rh", "expr": "10[mm]", "descr": "Hub radius"},
        {"name": "Ro", "expr": "12[mm]", "descr": "Opening radius"},
        {"name": "Rp", "expr": "35[mm]", "descr": "Opening pitch radius"}]
    g = m.component.child("geometry")
    g.props["unit"] = "mm"
    disk = m.create("geom.circle", g, r="R")
    tools = [m.create("geom.circle", g, r="Rh")]
    for k in range(6):
        tools.append(m.create("geom.circle", g, r="Ro", x=f"Rp*cos({k}*pi/3+pi/6)", y=f"Rp*sin({k}*pi/3+pi/6)"))
    m.create("geom.difference", g, input=[disk.tag], tools=[t.tag for t in tools])
    B.add_material(m, "Aluminum", all_domains=True)
    geo = build_geometry(m, display=False)
    solid = m.component.child("physics.solid")
    solid.props["plane"] = "stress"
    hub = [b for b in geo.numbers["boundary"] if max(abs(v) for v in geo.bbox_of["boundary"][b][:2] + geo.bbox_of["boundary"][b][3:5]) < 10.5]
    B.add_feature(m, solid, "solid.fixed", hub)
    B.add_feature(m, solid, "solid.rotframe", geo.numbers["domain"], omega="n[rpm]")
    right = [p for p in geo.numbers["point"] if abs(geo.bbox_of["point"][p][0] - 60) < 1e-3]
    left = [p for p in geo.numbers["point"] if abs(geo.bbox_of["point"][p][0] + 60) < 1e-3]
    defs = m.component.child("definitions")
    i1 = m.create("cpl.integration", defs, opname="intop1", level="point")
    i1.selection.level, i1.selection.all, i1.selection.entities = "point", False, right
    i2 = m.create("cpl.integration", defs, opname="intop2", level="point")
    i2.selection.level, i2.selection.all, i2.selection.entities = "point", False, left
    var = m.create("variables", defs)
    var.props["table"] = [{"name": "diam", "expr": "intop1(R+u)-intop2(-R+u)", "descr": "Deformed diameter"}]
    st = m.studies()[0]
    if sweep:
        m.create("study.sweep", st, index=0, pname="n", pvalues=sweep, punit="")
    return m, geo, hub, right, left


def _solutions(m, prefix):
    from elmerstudio.core.elmer import find_elmer
    from elmerstudio.core.nodes import bind_results
    from elmerstudio.core.results import Solution
    from elmerstudio.core.study_runner import StudyRunner
    folders = StudyRunner(m, m.studies()[0], tempfile.mkdtemp(prefix=prefix), find_elmer()).run()
    sols = [bind_results(m, Solution(f)) for f in folders]
    return sorted(sols, key=lambda s: s.meta.get("sweep_value") or 0)


def run_pulley():
    from elmerstudio.ui.plots import cut_points, evaluate_derived
    m, geo, hub, right, left = pulley_model()
    sols = _solutions(m, "es_pulley_")
    sweep = [(s.meta["sweep_value"], s) for s in sols]
    base = sols[-1]
    res = m.results
    derived = res.child("results.derived")
    # Surface Maximum of displacement and von Mises stress (MPa) over the sweep
    mx = m.create("eval.max", derived, solnum="all",
                  exprs=[{"expr": "solid.disp", "unit": "mm", "descr": ""}, {"expr": "solid.mises", "unit": "MPa", "descr": ""}])
    cols_max, rows_max = evaluate_derived(mx, m, base, [0], None, sweep, "n", "")
    # Point Evaluation on the far-right and far-left geometry points (mesh vertices)
    pts = [list(base.topo.nodes[base.topo.point_nodes[p]][:2] * base.scale) for p in right + left]
    pe = m.create("eval.point", derived, solnum="all", exprs=[{"expr": "u", "unit": "mm", "descr": ""}])
    cols_pt, rows_pt = evaluate_derived(pe, m, base, [0], None, sweep, "n", "", points=pts)
    # Global Evaluation of diam - 2*R with the integration couplings and the variable
    ge = m.create("eval.global", derived, solnum="all", exprs=[{"expr": "diam-2*R", "unit": "", "descr": "Diameter change"}])
    cols_g, rows_g = evaluate_derived(ge, m, base, [0], None, sweep, "n", "")
    # Cut Point (19, 54) mm: von Mises over the sweep (the Point Graph data)
    cp = m.create("dset.cutpoint", res.child("results.datasets"), x="19", y="54")
    P = [np.array(p[:2]) * base.scale for p in cut_points(cp, m, 2)]
    point_graph = [float(s.point_eval("solid.mises", s.frame_index("last"), P)[0]) for _v, s in sweep]
    return dict(m=m, sols=sols, cols_max=cols_max, rows_max=rows_max, cols_pt=cols_pt, rows_pt=rows_pt,
                cols_g=cols_g, rows_g=rows_g, point_graph=point_graph, right=right, left=left)


def test_pulley_walkthrough(elmer):
    r = run_pulley()
    n = np.array([row[0] for row in r["rows_max"]])
    assert n.tolist() == [2000, 4000, 6000, 8000], n
    assert r["cols_max"][1:] == ["Displacement magnitude (mm)", "Location (x, y, z) (m)", "von Mises stress (MPa)",
                                 "Location (x, y, z) (m)"], r["cols_max"]
    disp = np.array([row[1] for row in r["rows_max"]])
    mises = np.array([row[3] for row in r["rows_max"]])
    # linear elasticity with a centrifugal load rho*Omega^2*r: every result scales with n^2
    for q in (disp, mises, np.array(r["point_graph"])):
        assert np.allclose(q / q[-1], (n / n[-1]) ** 2, rtol=1e-6), q
    # magnitude: the free spinning annulus (hub radius a, rim b) is a lower bound for the stress of the perforated disk
    rho, nu, a, b = 2700.0, 0.33, 0.010, 0.060
    w = 8000 * RPM
    s_lame = (3 + nu) / 4 * rho * w ** 2 * (b ** 2 + (1 - nu) / (3 + nu) * a ** 2) / 1e6
    assert s_lame < mises[-1] < 6 * s_lame, (mises[-1], s_lame)
    assert 1e-4 < disp[-1] < 1e-2, disp                                          # mm
    # point evaluation: u at the far-right point is outward (+x), at the far-left point outward (-x)
    assert r["cols_pt"][1:] == ["Displacement field, x component, Point 1 (mm)",
                                "Displacement field, x component, Point 2 (mm)"], r["cols_pt"]
    for _n, ur, ul in r["rows_pt"]:
        assert ur > 0 > ul and abs(ur + ul) < 0.05 * ur          # symmetric openings (60 deg pattern)
    # diam - 2R from intop1/intop2 equals the difference of the point values (in metres)
    for (_n, ur, ul), (_n2, dd) in zip(r["rows_pt"], r["rows_g"]):
        assert abs(dd - (ur - ul) * 1e-3) <= 1e-9 * abs(dd), (dd, ur, ul)
    assert r["cols_g"] == ["n", "Diameter change"], r["cols_g"]


def test_pulley_table_graph(elmer):
    """Global Evaluation -> table -> Table Graph (1D plot group) draws one line per value column."""
    from matplotlib.figure import Figure

    from elmerstudio.ui.plots import draw_1d
    r = run_pulley()
    m = r["m"]
    tables = m.results.child("results.tables")
    tbl = m.create("table", tables)
    tbl.props["columns"], tbl.props["rows"] = r["cols_g"], r["rows_g"]
    pg = m.create("pg1d", m.results)
    m.create("plot.tablegraph", pg, table=tbl.tag, xcol=1, marker="o")
    fig = Figure()
    draw_1d(fig, pg, m, lambda d, s: (None, -1, 1.0), "last", get_table=lambda t: m.by_tag(t))
    ax = fig.axes[0]
    assert len(ax.lines) == 1
    assert np.allclose(ax.lines[0].get_xdata(), [2000, 4000, 6000, 8000])
    assert ax.get_title() == "Table Graph: Diameter change"


# --------------------------------------------------------------------------- Lame verification
def test_rotating_annulus_lame(elmer):
    """Quarter annulus with rollers on the cut edges, free inner and outer rims: Lame's plane-stress solution."""
    from elmerstudio.ui.plots import evaluate_derived
    a_mm, b_mm, rpm = 10.0, 60.0, 10000
    m = B.new_model("2D", ["solid"], "stationary")
    m.global_defs.child("params").props["table"] = [{"name": "n", "expr": str(rpm), "descr": ""}]
    g = m.component.child("geometry")
    g.props["unit"] = "mm"
    outer = m.create("geom.circle", g, r=str(b_mm), sector="90")
    hole = m.create("geom.circle", g, r=str(a_mm))
    m.create("geom.difference", g, input=[outer.tag], tools=[hole.tag])
    B.add_material(m, "Structural steel", all_domains=True)
    geo = build_geometry(m, display=False)
    solid = m.component.child("physics.solid")
    solid.props["plane"] = "stress"
    B.add_feature(m, solid, "solid.roller", geo.select("boundary", xmax=1e-6))
    B.add_feature(m, solid, "solid.roller", geo.select("boundary", ymax=1e-6))
    B.add_feature(m, solid, "solid.rotframe", geo.numbers["domain"], omega="n[rpm]")
    mesh = m.component.child("mesh")
    mesh.props["size"] = "2"           # finer
    sol = _solutions(m, "es_lame_")[-1]
    E, nu, rho = 200e9, 0.30, 7850.0
    w = rpm * RPM
    a, b = a_mm / 1e3, b_mm / 1e3
    s_a = (3 + nu) / 4 * rho * w ** 2 * (b ** 2 + (1 - nu) / (3 + nu) * a ** 2)
    s_b = (3 + nu) / 4 * rho * w ** 2 * (a ** 2 + (1 - nu) / (3 + nu) * b ** 2)
    u_b = b * s_b / E
    pn = sol.topo.point_nodes
    pa = next(p for p in geo.numbers["point"] if abs(geo.bbox_of["point"][p][0] - a_mm) < 1e-3 and abs(geo.bbox_of["point"][p][1]) < 1e-3)
    pb = next(p for p in geo.numbers["point"] if abs(geo.bbox_of["point"][p][0] - b_mm) < 1e-3 and abs(geo.bbox_of["point"][p][1]) < 1e-3)
    P = [list(sol.topo.nodes[pn[p]][:2] * sol.scale) for p in (pa, pb)]
    pe = m.create("eval.point", m.results.child("results.derived"),
                  exprs=[{"expr": "solid.sy", "unit": "", "descr": ""}, {"expr": "u", "unit": "", "descr": ""}])
    cols, rows = evaluate_derived(pe, m, sol, [sol.frame_index("last")], None, points=P)
    sy_a, sy_b, u_a, ub = rows[0]
    print(f"hoop stress at a: {sy_a / 1e6:.3f} MPa (Lame {s_a / 1e6:.3f}); u(b) = {ub * 1e6:.4f} um (Lame {u_b * 1e6:.4f})")
    assert abs(sy_a / s_a - 1) < 0.03, (sy_a, s_a)
    assert abs(sy_b / s_b - 1) < 0.03, (sy_b, s_b)
    assert abs(ub / u_b - 1) < 0.01, (ub, u_b)


# --------------------------------------------------------------------------- datasets
def _block_solution(prefix="es_block_"):
    """3D block 40 x 20 x 10 mm, T = 300 K at x = 0 and 400 K at x = 40: T is exactly linear in x."""
    m = B.new_model("3D", ["ht"], "stationary")
    g = m.component.child("geometry")
    g.props["unit"] = "mm"
    m.create("geom.block", g, w="40", d="20", h="10")
    B.add_material(m, "Aluminum", all_domains=True)
    geo = build_geometry(m, display=False)
    ht = m.component.child("physics.ht")
    B.add_feature(m, ht, "ht.temperature", geo.select("boundary", xmax=1e-6), T0="300[K]")
    B.add_feature(m, ht, "ht.temperature", geo.select("boundary", xmin=40 - 1e-6), T0="400[K]")
    return m, _solutions(m, prefix)[-1]


def test_cut_plane_mirror_extrusion(elmer):
    from elmerstudio.core.datasets import CutPlane, Mirror, ParametricExtrusion, resolve
    m, sol = _block_solution()
    fi = sol.frame_index("last")
    # quick yz-plane at x = 10 mm: T = 325 K everywhere on it, local coordinates are (y, z)
    cp = CutPlane(sol, [10, 0, 0], [1, 0, 0])
    T = cp.evaluate("T", fi)
    assert np.allclose(T, 325, atol=1e-6), (T.min(), T.max())
    loc = cp.topo.nodes
    assert np.isclose(loc[:, 0].min(), 0, atol=1e-6) and np.isclose(loc[:, 0].max(), 20, atol=1e-6)
    assert np.isclose(loc[:, 1].max(), 10, atol=1e-6)
    assert abs(cp.integrate("1", fi) - 20e-3 * 10e-3) < 1e-12                   # plane area (m^2)
    # an xy-plane: T varies along the first local axis (x) only
    cz = CutPlane(sol, [0, 0, 5], [0, 0, 1])
    Tz = cz.evaluate("T", fi)
    assert np.allclose(Tz, 300 + 2.5 * cz.topo.nodes[:, 0], atol=1e-6)
    v = cz.evaluate_vector("ht.tflux", fi)
    assert np.all(v[:, 0] < 0) and np.allclose(v[:, 2], 0)                       # heat flows towards -x
    # mirror in x = 0: twice the points, values copied, x mirrored
    mi = Mirror(sol, "x")
    assert len(mi.topo.nodes) == 2 * len(sol.topo.nodes)
    Tm = mi.evaluate("T", fi)
    assert np.allclose(Tm[len(sol.topo.nodes):], Tm[:len(sol.topo.nodes)])
    assert abs(mi.integrate("1", fi) - 2 * 40e-3 * 20e-3 * 10e-3) < 1e-12
    # parametric extrusion of the xy cut plane over three "parameter values" -> a 3D wedge volume
    pe = ParametricExtrusion([(cz, fi, 0.0), (cz, fi, 1.0), (cz, fi, 2.0)], levelscale=5.0)
    assert pe.sdim == 3 and pe.bulk_grid().n_cells == 2 * cz.bulk_grid().n_cells
    assert abs(pe.integrate("1", 0) - 40e-3 * 20e-3 * 10e-3) < 1e-12              # 40 x 20 mm x (2*5 mm)
    assert abs(pe.integrate("T", 0, average=True) - 350) < 1e-6
    # the same chain through dataset nodes (what the GUI resolves)
    dsets = m.results.child("results.datasets")
    sol_ds = B.solution_dataset(m, m.studies()[0], create=True)
    c = m.create("dset.cutplane", dsets, data=sol_ds.tag, ptype="quick", plane="yz", coord="30")
    mir = m.create("dset.mirror", dsets, data=c.tag, axis="x")
    view, rev = resolve(m, mir, lambda d: sol)
    assert rev is None and np.allclose(view.evaluate("T", fi), 375, atol=1e-6)
    rv = m.create("dset.revolution", dsets, data=c.tag, start="-90", angle="225", layers=30)
    view, rev = resolve(m, rv, lambda d: sol)
    assert rev == {"start": -90.0, "angle": 225.0, "layers": 30}


def test_plot_scenes_and_polar(elmer):
    """Every 2D/3D plot type renders on solution and derived datasets; polar plot of a symmetric field."""
    from matplotlib.figure import Figure

    from elmerstudio.core.datasets import CutPlane
    from elmerstudio.ui.plots import annotation_text, build_scene, draw_1d, polar_values
    m, sol = _block_solution("es_block2_")
    fi = sol.frame_index("last")
    pg = m.create("pg3d", m.results)
    cases = [("plot.surface", {"expr": "T", "wireframe": True}),
             ("plot.slice", {"expr": "T", "entry": "coords", "coords": "10 30", "plane": "yz"}),
             ("plot.isosurface", {"expr": "T", "levels": 4}), ("plot.contour", {"expr": "T", "ctype": "filled", "labels": True}),
             ("plot.arrow", {"expr": "ht.tflux", "xg": 5, "yg": 4, "zg": 3}),
             ("plot.streamline", {"expr": "ht.tflux", "positioning": "magnitude", "ltype": "tube"}),
             ("plot.line", {"expr": "T", "ltype": "tube"}), ("plot.mesh", {})]
    for kind, props in cases:
        for c in pg.children:
            c.enabled = False
        m.create(kind, pg, **props)
        sc = build_scene(pg, sol, fi, 3)
        assert sc.items, kind
        assert "error" not in sc.title, sc.title
    # arrows on a grid: 5 x 4 x 3 points, all inside the block
    arrow = next(c for c in pg.children if c.kind == "plot.arrow")
    for c in pg.children:
        c.enabled = c is arrow
    sc = build_scene(pg, sol, fi, 3)
    glyphs = sc.items[0][0]
    assert glyphs.n_points > 0
    # selection subnode restricts a surface plot to the two end faces
    surf = next(c for c in pg.children if c.kind == "plot.surface")
    for c in pg.children:
        c.enabled = c is surf
    surf.props["wireframe"] = False
    full = build_scene(pg, sol, fi, 3).items[0][0].n_cells
    sel = m.create("plot.selection", surf, level="boundary")
    sel.selection.level, sel.selection.all = "boundary", False
    sel.selection.entities = [1]
    part = build_scene(pg, sol, fi, 3).items[0][0].n_cells
    assert 0 < part < full
    # annotation with eval(): the temperature at x = 20 mm is 350 K = 76.85 degC
    txt = annotation_text("T = eval(T, degC)", sol, fi, [20, 10, 5])
    assert txt.startswith("T = 76.85"), txt
    # polar plot on a cut plane: centre (20, 10) on the yz-plane is constant T, on the xy-plane T ~ cos(theta)
    cz = CutPlane(sol, [0, 0, 5], [0, 0, 1])
    pol = m.create("plot.polar", m.create("pgpolar", m.results), expr="T", xc="20", yc="10", radius="8", npts=73)
    th, v = polar_values(cz, pol, m, fi)
    assert np.allclose(v, 350 + 2.5 * 8 * np.cos(th), atol=1e-6)
    fig = Figure()
    draw_1d(fig, pol.parent, m, lambda d, s: (cz, fi, cz.scale), "last")
    assert fig.axes[0].name == "polar" and len(fig.axes[0].lines) == 1


if __name__ == "__main__":
    r = run_pulley()
    for k in ("cols_max", "rows_max", "cols_pt", "rows_pt", "cols_g", "rows_g", "point_graph"):
        print(k, r[k])
