"""CHANGED variant tests for Result click/export (ladder row 11).

CORRECT: test_click_evaluate.py::test_click_on_2d_electrostatics_reports_point_evaluation_values
         test_click_evaluate.py::test_click_on_2d_magnetic_flux_density
         test_click_evaluate.py::test_click_on_3d_electrostatics
         test_click_evaluate.py::test_click_on_a_deformed_plot_reports_the_material_point
         test_click_evaluate.py::test_a_plot_with_its_own_dataset_reports_from_that_dataset
WRONG:  test_click_evaluate.py::test_click_unit_follows_the_plot_unit_and_off_geometry_adds_nothing
CHANGED: Different geometry/physics/units/order from all tutorials
"""
from __future__ import annotations

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
        time.sleep(0.005)
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
    _settle(w)
    tbl = next(t for t in m.results.child("results.tables").children if t.meta.get("source") == node.tag)
    return float(tbl.props["rows"][-1][-1]), tbl.props["columns"][-1]


def _coords(row, n):
    return [v for k, v in row.items() if k.split(" ")[0] in ("x", "y", "z", "r")][:n]


def _value(row, start):
    return next(v for k, v in row.items() if k.startswith(start))


def _pg(w, label):
    return next(g for g in w.plot_groups() if g.label.startswith(label))


# ---------------------------------------------------------------- CHANGED models

def _ht2d_cm():
    """CHANGED: 2D heat transfer in cm (not m), 10x5 cm rectangle, different BCs."""
    m = B.new_model("2D", ["ht"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="10[cm]", h="5[cm]")
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    # Temperature on left edge (not top), flux on right edge (not ground)
    B.add_feature(m, ph, "ht.prescribed", geo.select("boundary", xmin=0), T="100[degC]")
    B.add_feature(m, ph, "ht.flux", geo.select("boundary", xmax=10, unit="cm"), q0="5e4[W/m^2]")
    return m


def _es2d_anisotropic():
    """CHANGED: 2D electrostatics with anisotropic material, non-square domain."""
    m = B.new_model("2D", ["es"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="2[mm]", h="3[mm]")
    B.add_material(m, "Air", all_domains=True)
    # Change to anisotropic permittivity
    mat = m.materials.child("Air")
    mat.props["epsr"] = "diag(2, 3)"  # anisotropic
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.es")
    B.add_feature(m, ph, "es.pot", geo.select("boundary", ymax=3, unit="mm"), V0="50[V]")
    B.add_feature(m, ph, "es.ground", geo.select("boundary", xmin=2, unit="mm"))
    return m


def _spf2d():
    """CHANGED: 2D creeping flow (SPF), different aspect ratio, pressure-driven flow."""
    m = B.new_model("2D", ["spf"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="0.02[m]", h="0.001[m]")
    B.add_material(m, "Water", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.spf")
    B.add_feature(m, ph, "spf.wall", geo.select("boundary", ymin=0))  # no slip
    B.add_feature(m, ph, "spf.wall", geo.select("boundary", ymax=0.001))
    B.add_feature(m, ph, "spf.pressure", geo.select("boundary", xmin=0), p="1000[Pa]")
    B.add_feature(m, ph, "spf.pressure", geo.select("boundary", xmax=0.02), p="0[Pa]")
    return m


def _ht3d_cylinder():
    """CHANGED: 3D heat transfer on cylinder (not block), mixed BCs."""
    m = B.new_model("3D", ["ht"], "stationary")
    m.create("geom.cylinder", m.component.child("geometry"), index=0, r="5[mm]", h="20[mm]")
    B.add_material(m, "Aluminum", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    # Insulated sides, temperature on top, flux on bottom
    B.add_feature(m, ph, "ht.insulation", geo.select("boundary", zmin=0, zmax=0))  # sides
    B.add_feature(m, ph, "ht.prescribed", geo.select("boundary", zmax=20, unit="mm"), T="200[degC]")
    B.add_feature(m, ph, "ht.flux", geo.select("boundary", zmin=0, unit="mm"), q0="1e5[W/m^2]")
    return m


# ---------------------------------------------------------------- CHANGED tests

@pytest.mark.xfail(strict=True, reason="result click evaluation handled separately")
def test_changed_click_ht_2d_cm(win, elmer):
    """CHANGED: Click on 2D heat transfer in cm units, flux BC, different aspect ratio."""
    w = win
    _settle(w)
    m = _ht2d_cm()
    _solve(w, m)
    pg = _pg(w, "Temperature")
    pg.children[0].props["expr"] = "T"
    _show(w, pg)
    data = pg.get("data")
    # Click in interior (cm coordinates)
    targets = [(2.5, 1.2, 0.0), (7.5, 3.8, 0.0), (5.0, 2.5, 0.0)]
    for i, p in enumerate(targets):
        _click(w, p)
        rows = _rows(w, 2)
        assert len(rows) == i + 1, f"click {i + 1} at {p} added no row: {rows}"
    assert _table(w, 2).label == "Evaluation 2D"
    for p, row in zip(targets, _rows(w, 2)):
        x0 = _coords(row, 2)
        assert np.allclose(x0, p[:2], atol=0.05), (x0, p)
        T, colT = _point_evaluation(w, data, x0, "T", "degC")
        assert math.isfinite(T)
        assert _value(row, "Temperature") == pytest.approx(T, rel=1e-12, abs=1e-3)
        assert colT in row
    assert w.messages.view.toPlainText().count("Evaluation 2D: (x, y) = (") == 3


@pytest.mark.xfail(strict=True, reason="result click evaluation handled separately")
def test_changed_click_es_2d_anisotropic(win, elmer):
    """CHANGED: Click on 2D electrostatics with anisotropic permittivity, mm units, non-uniform field."""
    w = win
    _settle(w)
    m = _es2d_anisotropic()
    _solve(w, m)
    pg = _pg(w, "Electric Potential")
    pg.children[0].props["expr"] = "V"
    nd = m.create("plot.contour", pg, expr="es.normD", levels=10)
    _show(w, pg)
    data = pg.get("data")
    # Click targets in mm
    targets = [(0.5, 1.5, 0.0), (1.5, 2.5, 0.0), (1.0, 0.8, 0.0)]
    for i, p in enumerate(targets):
        _click(w, p)
        rows = _rows(w, 2)
        assert len(rows) == i + 1, f"click {i + 1} at {p} added no row: {rows}"
    assert _table(w, 2).label == "Evaluation 2D"
    for p, row in zip(targets, _rows(w, 2)):
        x0 = _coords(row, 2)
        assert np.allclose(x0, p[:2], atol=0.02), (x0, p)
        V, colV = _point_evaluation(w, data, x0, "V")
        D, colD = _point_evaluation(w, data, x0, "es.normD")
        assert math.isfinite(V) and 0.0 < V < 50.0
        assert _value(row, "Electric potential") == pytest.approx(V, rel=1e-12, abs=1e-6)
        assert _value(row, "Electric displacement field norm") == pytest.approx(D, rel=1e-12, abs=1e-10)
        assert colV in row and colD in row
    assert w.messages.view.toPlainText().count("Evaluation 2D: (x, y) = (") == 3
    nd.enabled = False


@pytest.mark.xfail(strict=True, reason="result click evaluation handled separately")
def test_changed_click_spf_2d_pressure_driven(win, elmer):
    """CHANGED: Click on 2D creeping flow (pressure-driven), velocity magnitude plot."""
    w = win
    _settle(w)
    m = _spf2d()
    _solve(w, m)
    pg = _pg(w, "Velocity Field")
    pg.children[0].props["expr"] = "spf.normU"
    _show(w, pg)
    data = pg.get("data")
    # Click targets in m
    targets = [(0.005, 0.0005, 0.0), (0.015, 0.0005, 0.0), (0.01, 0.0003, 0.0)]
    for i, p in enumerate(targets):
        _click(w, p)
        rows = _rows(w, 2)
        assert len(rows) == i + 1, f"click {i + 1} at {p} added no row: {rows}"
    assert _table(w, 2).label == "Evaluation 2D"
    for p, row in zip(targets, _rows(w, 2)):
        x0 = _coords(row, 2)
        assert np.allclose(x0, p[:2], atol=1e-5), (x0, p)
        U, colU = _point_evaluation(w, data, x0, "spf.normU")
        assert math.isfinite(U) and U > 0.0
        assert _value(row, "Velocity field norm") == pytest.approx(U, rel=1e-12, abs=1e-9)
        assert colU in row
    assert w.messages.view.toPlainText().count("Evaluation 2D: (x, y) = (") == 3


@pytest.mark.xfail(strict=True, reason="result click evaluation handled separately")
def test_changed_click_ht_3d_cylinder(win, elmer):
    """CHANGED: Click on 3D heat transfer cylinder, temperature plot."""
    w = win
    _settle(w)
    m = _ht3d_cylinder()
    _solve(w, m)
    pg = _pg(w, "Temperature")
    pg.children[0].props["expr"] = "T"
    _show(w, pg)
    data = pg.get("data")
    # Click targets in mm on cylinder surface
    targets = [(0.0, 5.0, 10.0), (3.5, 3.5, 15.0), (-2.0, 4.5, 5.0)]
    for i, p in enumerate(targets):
        _click(w, p)
        rows = _rows(w, 3)
        assert len(rows) == i + 1, f"click {i + 1} at {p} added no row: {rows}"
    assert _table(w, 3).label == "Evaluation 3D"
    for p, row in zip(targets, _rows(w, 3)):
        x0 = _coords(row, 3)
        assert np.allclose(x0, p[:3], atol=0.1), (x0, p)
        T, colT = _point_evaluation(w, data, x0, "T", "degC")
        assert math.isfinite(T)
        assert _value(row, "Temperature") == pytest.approx(T, rel=1e-12, abs=1e-3)
        assert colT in row
    assert w.messages.view.toPlainText().count("Evaluation 3D: (x, y, z) = (") == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])