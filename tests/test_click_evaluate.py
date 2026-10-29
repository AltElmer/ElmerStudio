"""Clicking a 2D/3D result plot reports the plotted value there (issue #9), as COMSOL does: each click adds a row of
coordinates and plot values to the Evaluation 2D / Evaluation 3D table and a line to Messages.

Real mouse clicks (QTest) on the VTK Graphics window of the main window, on solved Electrostatics (2D, 3D) and
Magnetic Fields (2D) models.  The clicked value must equal a Point Evaluation at the reported coordinates; the unit
follows the plot's unit; a click off the geometry adds nothing; a deformed plot and a plot with its own dataset
report from that plot's own dataset."""
import math
import os
import tempfile
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPoint, QSettings, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core import project  # noqa: E402
from elmerstudio.core.geometry import build_geometry  # noqa: E402


@pytest.fixture(scope="module")
def win():
    app = QApplication.instance() or QApplication([])
    from elmerstudio.ui.main_window import MainWindow
    for name in ("warning", "critical", "information", "question", "about"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(prefix="es_qs_"), "s.ini"), QSettings.IniFormat)
    w.resize(1400, 860)
    w.show()
    w.activateWindow()
    yield w
    w.dirty = False
    w.stop_job()
    w.close()
    app.processEvents()


def _settle(w, n=15, limit=300):
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.005)
    t0 = time.time()
    while w.busy() and time.time() - t0 < limit:
        QApplication.processEvents()
        time.sleep(0.01)
    for _ in range(n):
        QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _solve(w, m, studies=None):
    w._set_model(m, None, project.new_workdir())
    w.build_geometry()
    _settle(w)
    for st in studies or m.studies():
        w.compute(st)
        _settle(w)


def _graphics_ok(w):
    return w.graphics._ok and w.graphics.plotter.interactor.isVisible() and \
        w.graphics.plotter.ren_win.GetSize()[0] > 50


def _show(w, pg):
    w.plot(pg)
    _settle(w)
    if not _graphics_ok(w):
        pytest.skip("no rendered Graphics window")
    assert w.graphics.mode == "results" and w.current_plot is pg


def _display(w, p):
    r = w.graphics.plotter.renderer
    r.SetWorldPoint(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0, 1.0)
    r.WorldToDisplay()
    return r.GetDisplayPoint()[:2]


def _click(w, p):
    """A real left click at world point *p* (the plotted coordinates)."""
    _click_display(w, _display(w, p))


def _click_display(w, xy):
    inter = w.graphics.plotter.interactor
    dpr = inter.devicePixelRatioF()
    h = w.graphics.plotter.ren_win.GetSize()[1]
    pos = QPoint(int(round(xy[0] / dpr)), int(round((h - 1 - xy[1]) / dpr)))
    QTest.mouseMove(inter, pos)
    _settle(w, 5, 1)
    QTest.mouseClick(inter, Qt.LeftButton, Qt.NoModifier, pos)
    _settle(w, 5, 1)


def _table(w, nd):
    tables = w.model.results.child("results.tables")
    return next((t for t in tables.children if t.meta.get("source") == f"click:{nd}D"), None)


def _rows(w, nd):
    t = _table(w, nd)
    return [] if t is None else [dict(zip(t.props["columns"], r)) for r in t.props["rows"]]


def _point_evaluation(w, data_tag, x0, expr, unit=""):
    """The value of a Point Evaluation node at *x0* (geometry units), evaluated by the window."""
    m = w.model
    keys = ("x", "y", "z")[:len(x0)]
    node = m.create("eval.point", m.results.child("results.derived"), data=data_tag,
                    exprs=[{"expr": expr, "unit": unit, "descr": ""}], **{k: repr(float(v)) for k, v in zip(keys, x0)})
    w.evaluate_node(node)
    _settle(w)        # selecting the new node shows its point selection (a deferred call): let it run now
    tbl = next(t for t in m.results.child("results.tables").children if t.meta.get("source") == node.tag)
    return float(tbl.props["rows"][-1][-1]), tbl.props["columns"][-1]


def _coords(row, n):
    return [v for k, v in row.items() if k.split(" ")[0] in ("x", "y", "z", "r")][:n]


def _value(row, start):
    return next(v for k, v in row.items() if k.startswith(start))


def _pg(w, label):
    return next(g for g in w.plot_groups() if g.label.startswith(label))


# ---------------------------------------------------------------- models
def _es2d():
    """1 x 1 square, 1 V on the top edge, ground on the right edge: a field with a corner, not uniform."""
    m = B.new_model("2D", ["es"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1", h="1")
    B.add_material(m, "Air", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.es")
    B.add_feature(m, ph, "es.pot", geo.select("boundary", ymin=1), V0="1[V]")
    B.add_feature(m, ph, "es.ground", geo.select("boundary", xmin=1))
    return m


def _mf2d():
    """A 2 x 1 air box with a 0.5 x 0.5 coil carrying an out-of-plane current: B varies everywhere."""
    m = B.new_model("2D", ["mf"], "stationary")
    g = m.component.child("geometry")
    a = m.create("geom.rectangle", g, index=0, w="2", h="1")
    b = m.create("geom.rectangle", g, index=1, w="0.5", h="0.5", x="0.5", y="0.25")
    m.create("geom.union", g, index=2, input=[a.tag, b.tag])
    B.add_material(m, "Air", all_domains=True)
    geo = build_geometry(m, display=False)
    coil = geo.select("domain", xmin=0.5, xmax=1.0, ymin=0.25, ymax=0.75)
    B.add_feature(m, m.component.child("physics.mf"), "mf.ecd", coil, Jz="1e6")
    return m


def _es3d():
    m = B.new_model("3D", ["es"], "stationary")
    m.create("geom.block", m.component.child("geometry"), index=0, w="1", d="0.5", h="0.5")
    B.add_material(m, "Air", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.es")
    B.add_feature(m, ph, "es.pot", geo.select("boundary", xmax=0), V0="10[V]")
    B.add_feature(m, ph, "es.ground", geo.select("boundary", xmin=1))
    return m


# ---------------------------------------------------------------- tests
def test_click_on_2d_electrostatics_reports_point_evaluation_values(win, elmer):
    w = win
    _settle(w)
    m = _es2d()
    _solve(w, m)
    pg = _pg(w, "Electric Potential")
    pg.children[0].props["expr"] = "V"
    # the flux the reporter wanted: a second plot of the same group, es.normD
    nd = m.create("plot.contour", pg, expr="es.normD", levels=10)
    _show(w, pg)
    assert _table(w, 2) is None
    targets = [(0.3, 0.6, 0.0), (0.7, 0.2, 0.0), (0.55, 0.85, 0.0)]
    for i, p in enumerate(targets):
        _click(w, p)
        rows = _rows(w, 2)
        assert len(rows) == i + 1, f"click {i + 1} at {p} added no row: {rows}"     # repeated clicks add rows
    assert _table(w, 2).label == "Evaluation 2D"
    data = pg.get("data")
    for p, row in zip(targets, _rows(w, 2)):
        x0 = _coords(row, 2)
        assert np.allclose(x0, p[:2], atol=0.01), (x0, p)                      # where the user clicked
        V, colV = _point_evaluation(w, data, x0, "V")
        D, colD = _point_evaluation(w, data, x0, "es.normD")
        assert math.isfinite(V) and 0.0 < V < 1.0
        assert _value(row, "Electric potential") == pytest.approx(V, rel=1e-12, abs=1e-15)
        assert _value(row, "Electric displacement field norm") == pytest.approx(D, rel=1e-12, abs=1e-30)
        assert colV in row and colD in row                                     # the same column names
    vals = [_value(r, "Electric potential") for r in _rows(w, 2)]
    assert len(set(np.round(vals, 9))) == 3                                    # the field is not uniform
    assert w.messages.view.toPlainText().count("Evaluation 2D: (x, y) = (") == 3
    nd.enabled = False


def test_click_unit_follows_the_plot_unit_and_off_geometry_adds_nothing(win, elmer):
    w = win
    pg = _pg(w, "Electric Potential")
    surf = pg.children[0]
    surf.props["unit"] = "mV"
    _show(w, pg)
    n = len(_rows(w, 2))
    _click(w, (0.4, 0.5, 0.0))
    row = _rows(w, 2)[-1]
    assert len(_rows(w, 2)) == n + 1
    key = "Electric potential (mV)"
    assert key in row, list(row)
    assert math.isnan(row["Electric potential (V)"])          # another unit: its own column; the V column is empty
    x0 = _coords(row, 2)
    mv, col = _point_evaluation(w, pg.get("data"), x0, "V", "mV")
    v, _c = _point_evaluation(w, pg.get("data"), x0, "V")
    assert row[key] == pytest.approx(mv, rel=1e-12) and row[key] == pytest.approx(1000 * v, rel=1e-9)
    # off the geometry: left of the square and above it; nothing is added
    _show(w, pg)
    for p in [(-0.3, 0.5, 0.0), (0.5, 1.3, 0.0)]:
        assert w.graphics.pick_result(_display(w, p)) is None
        _click(w, p)
        assert w.graphics.mode == "results"
    assert len(_rows(w, 2)) == n + 1
    _click(w, (0.6, 0.4, 0.0))                                 # the control: on the plot, a row
    assert len(_rows(w, 2)) == n + 2
    surf.props["unit"] = ""


def test_a_double_click_keeps_clicks_working(win, elmer):
    """pyvista treats two quick clicks at one spot as a double click and re-applies its own interactor style; the
    Graphics window's style (which turns clicks into picks) must survive it, in results and in geometry."""
    w = win
    pg = _pg(w, "Electric Potential")
    _show(w, pg)
    n = len(_rows(w, 2))
    _click(w, (0.25, 0.25, 0.0))
    _click(w, (0.25, 0.25, 0.0))                               # a double click (pyvista: same spot, < 0.5 s)
    iren = w.graphics.plotter.iren.interactor
    assert iren.GetInteractorStyle() is w.graphics.style
    _click(w, (0.8, 0.7, 0.0))
    assert len(_rows(w, 2)) == n + 3
    w.graphics.show_geometry(w.geo)                            # geometry picking uses the same style
    _settle(w)
    _click(w, (0.5, 0.5, 0.0))
    _click(w, (0.5, 0.5, 0.0))
    assert iren.GetInteractorStyle() is w.graphics.style


def test_click_on_2d_magnetic_flux_density(win, elmer):
    w = win
    _settle(w)
    m = _mf2d()
    _solve(w, m)
    pg = _pg(w, "Magnetic Flux Density")
    assert pg.children[0].get("expr") in ("normB", "mf.normB")
    _show(w, pg)
    targets = [(0.75, 0.5, 0.0), (1.5, 0.2, 0.0), (0.3, 0.8, 0.0)]
    for p in targets:
        _click(w, p)
    rows = _rows(w, 2)
    assert len(rows) == 3
    for p, row in zip(targets, rows):
        x0 = _coords(row, 2)
        assert np.allclose(x0, p[:2], atol=0.02)
        B_pe, col = _point_evaluation(w, pg.get("data"), x0, "mf.normB")
        assert math.isfinite(B_pe) and B_pe > 0
        key = next(k for k in row if k.startswith("Magnetic flux density norm"))
        assert key.endswith("(T)")
        assert row[key] == pytest.approx(B_pe, rel=1e-12)
    # the unit follows the plot: mT
    pg.children[0].props["unit"] = "mT"
    _show(w, pg)
    _click(w, targets[0])
    row = _rows(w, 2)[-1]
    key = next(k for k in row if k.startswith("Magnetic flux density norm") and k.endswith("(mT)"))
    assert row[key] == pytest.approx(1000 * _point_evaluation(w, pg.get("data"), _coords(row, 2), "mf.normB")[0],
                                     rel=1e-9)


def test_click_on_3d_electrostatics(win, elmer):
    w = win
    _settle(w)
    m = _es3d()
    _solve(w, m)
    pg = _pg(w, "Electric Potential")                 # the default 3D plot: a Multislice of V
    _show(w, pg)
    # a surface plot of the flux in its own plot group: a click lands on the outer boundary
    pg2 = m.create("pg3d", m.results, label="Flux")
    pg2.props["data"] = pg.get("data")
    m.create("plot.surface", pg2, expr="es.normD")
    assert _table(w, 3) is None
    # the centre of the middle x-slice (the multislice cuts at x = 0.5): seen from the default view it is in front
    g = w.graphics
    hit = None
    for p in [(0.5, 0.25, 0.25), (0.5, 0.1, 0.4), (0.5, 0.4, 0.1)]:
        if g.pick_result(_display(w, p)) is not None:
            hit = p
            break
    assert hit is not None, "nothing pickable on the multislice"
    _click(w, hit)
    rows = _rows(w, 3)
    assert len(rows) == 1 and _table(w, 3).label == "Evaluation 3D"
    x0 = _coords(rows[0], 3)
    V, _c = _point_evaluation(w, pg.get("data"), x0, "V")
    assert math.isfinite(V) and 0 < V < 10
    assert _value(rows[0], "Electric potential") == pytest.approx(V, rel=1e-12)
    _show(w, pg2)
    b = w.geo.bbox
    centre = np.array([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2])
    _click(w, centre)                                 # the ray hits the front face of the block first
    rows = _rows(w, 3)
    assert len(rows) == 2
    assert w.graphics.pick_result((12, 12)) is None
    _click_display(w, (12, 12))                       # the background in a corner of the window: nothing
    assert w.graphics.mode == "results" and len(_rows(w, 3)) == 2
    x0 = _coords(rows[1], 3)
    on_face = any(abs(x0[k] - lim) < 1e-6 for k, lim in [(0, 0), (0, 1), (1, 0), (1, 0.5), (2, 0), (2, 0.5)])
    assert on_face, x0
    D, _c = _point_evaluation(w, pg.get("data"), x0, "es.normD")
    assert math.isfinite(D) and _value(rows[1], "Electric displacement field norm") == pytest.approx(D, rel=1e-12)


def test_click_on_a_deformed_plot_reports_the_material_point(win, elmer):
    """A cantilever drawn deformed: the click lands where the plot shows the tip, which lies outside the undeformed
    beam.  The row gives the material point and the value there (a Point Evaluation at the clicked screen point would
    be outside the dataset)."""
    w = win
    _settle(w)
    m = B.new_model("2D", ["solid"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1", h="0.1")
    B.add_material(m, "Structural steel", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")
    B.add_feature(m, ph, "solid.fixed", geo.select("boundary", xmax=0))
    B.add_feature(m, ph, "solid.bload", geo.select("boundary", xmin=1), Fy="-1e6")
    _solve(w, m)
    pg = _pg(w, "Stress")
    surf = pg.children[0]
    surf.props["expr"] = "solid.mises"
    assert any(c.kind == "plot.deform" and c.enabled for c in surf.children)
    _show(w, pg)
    sc = w.graphics._scene
    i = next(j for j, ev in sc.evals.items() if ev["plot"] is surf)
    ds = sc.items[i][0]
    P, X = np.asarray(ds.points), np.asarray(ds.point_data["__x0"])
    sol = sc.evals[i]["sol"]
    # a point of the plotted (deformed) beam near the tip, not on its outline
    tip = np.where((X[:, 0] > 0.85) & (X[:, 0] < 0.95) & (np.abs(X[:, 1] - 0.05) < 0.03))[0]
    assert len(tip)
    k = tip[np.argmax(np.linalg.norm(P[tip] - X[tip], axis=1))]
    assert np.linalg.norm(P[k] - X[k]) > 0.02                          # visibly deformed
    outside = float(sol.point_eval("solid.mises", sc.evals[i]["fi"], [list(P[k][:2] * sol.scale)])[0])
    assert not math.isfinite(outside)                                   # the shown position is off the beam
    _click(w, P[k])
    rows = _rows(w, 2)
    assert rows, "the click on the deformed plot added nothing"
    x0 = _coords(rows[-1], 2)
    assert np.allclose(x0, X[k][:2], atol=0.02), (x0, X[k])
    S, _c = _point_evaluation(w, pg.get("data"), x0, "solid.mises")
    assert math.isfinite(S) and _value(rows[-1], "von Mises") == pytest.approx(S, rel=1e-12)


def test_a_plot_with_its_own_dataset_reports_from_that_dataset(win, elmer):
    """Two studies (1 V, then 3 V on the top edge); a plot of the second study's solution in a plot group of the
    first: the click reports the second dataset's value, three times the group's own."""
    w = win
    _settle(w)
    m = _es2d()
    pot = next(f for f in m.component.child("physics.es").children if f.kind == "es.pot")
    st1 = m.studies()[0]
    st2 = B.add_study(m, "stationary")
    w._set_model(m, None, project.new_workdir())
    w.build_geometry()
    _settle(w)
    w.compute(st1)
    _settle(w)
    pot.props["V0"] = "3[V]"
    w.compute(st2)
    _settle(w)
    d1, d2 = B.solution_dataset(m, st1, create=False), B.solution_dataset(m, st2, create=False)
    assert d1 is not None and d2 is not None and d1 is not d2
    pg = m.create("pg2d", m.results, label="Own Dataset")
    pg.props["data"] = d1.tag
    m.create("plot.surface", pg, expr="V", data=d2.tag)
    _show(w, pg)
    _click(w, (0.4, 0.6, 0.0))
    rows = _rows(w, 2)
    assert rows
    x0 = _coords(rows[-1], 2)
    v1, _c = _point_evaluation(w, d1.tag, x0, "V")
    v2, _c = _point_evaluation(w, d2.tag, x0, "V")
    assert v2 == pytest.approx(3 * v1, rel=1e-6)
    assert _value(rows[-1], "Electric potential") == pytest.approx(v2, rel=1e-12)
