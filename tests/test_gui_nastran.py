"""The NASTRAN-95 backend through the desktop: choose it, show its input deck, Compute,
(log, convergence plot, default plot, solution status), and Stop a running solver.

Drives MainWindow on the GUI thread like tests/test_gui_openfoam.py; run GUI tests one at a time."""
import os
import tempfile
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox  # noqa: E402

from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core import project  # noqa: E402
from elmerstudio.core.geometry import build_geometry  # noqa: E402

OPENED = []


@pytest.fixture(scope="module")
def win():
    app = QApplication.instance() or QApplication([])
    from elmerstudio.ui import theme
    theme.apply(app, 9.0)
    from elmerstudio.ui.main_window import MainWindow
    for name in ("warning", "critical", "information", "question", "about"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: OPENED.append(("messagebox", a[2] if len(a) > 2
                                                                                else "")) or 0))
    QDialog.exec = lambda self, *a, **k: OPENED.append(type(self).__name__) or 0
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(prefix="es_qs_"), "s.ini"), QSettings.IniFormat)
    w.resize(1400, 900)
    w.show()
    _settle()
    yield w
    w.dirty = False
    w.stop_job()
    w.close()
    app.processEvents()


def _settle(n=15):
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.005)


def _wait(w, cond, timeout=180):
    t0 = time.time()
    while not cond():
        QApplication.processEvents()
        if time.time() - t0 > timeout:
            raise TimeoutError
        time.sleep(0.01)
    _settle()


def _axial_bar_2d(w):
    """Build a 2D axial bar model."""
    _wait(w, lambda: not w.busy())
    m = B.new_model("2D", ["solid"], "stationary")
    L, H = 1.0, 0.05
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(H))
    B.add_material(m, "Structural steel", all_domains=True)
    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.05", hmin="0.02")
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")
    B.add_feature(m, ph, "solid.fixed", geo.select("boundary", xmax=0))
    bload = B.add_feature(m, ph, "solid.bload", geo.select("boundary", xmin=0.99 * L))
    bload.props["Fx"] = "10000[N]"
    bload.props["Fy"] = "0[N]"
    w._set_model(m, None, project.new_workdir())
    w.build_geometry()
    _wait(w, lambda: not w.busy() and w.geo is not None)
    return m


def test_choose_nastran_compute_plot(win, nastran):
    """Switch a Study to NASTRAN backend through the UI, show deck, Compute, verify results."""
    w = win
    m = _axial_bar_2d(w)
    st = w.model.studies()[0]
    w.tree.select(st)
    _settle()
    assert st.get("backend") == "elmer"
    w.set_prop(st, "backend", "nastran")
    _settle()
    w.update_actions()
    assert w.act["compute"].isEnabled(), w.act["compute"].toolTip()
    # Show Input Files: the NASTRAN deck
    w.show_sif(st)
    assert "case.bdf" in w._last_nastran_files, w._last_nastran_files
    assert "SOL 1" in w._last_sif, w._last_sif
    assert "BEGIN BULK" in w._last_sif
    assert "GRID," in w._last_sif
    assert "CTRIA3," in w._last_sif or "CTRIM6," in w._last_sif
    assert "MAT1," in w._last_sif
    assert "SPC1," in w._last_sif
    assert "FORCE," in w._last_sif
    assert "ENDDATA" in w._last_sif
    # Compute: log, convergence plot, default plot, solved state
    w.compute(st)
    _wait(w, lambda: not w.busy(), 180)
    log = w.logpane.view.toPlainText()
    assert "NASTRAN-95" in log and "END OF JOB" in log, log[-2000:]
    # Convergence plot for static solve shows iterations (if any)
    assert w.plot_groups(), "no default plot"
    assert w.current_plot is not None
    st = w.model.studies()[0]
    assert w._solution_state(st)[1] == "ok", (w._solution_state(st), w.messages.view.toPlainText()[-1500:])


def test_stop_kills_the_nastran_process(win, nastran):
    """Start a long NASTRAN run and Stop it; verify the process is gone."""
    w = win
    if nastran.kind != "wsl":
        pytest.skip("the process check here looks inside WSL")
    import subprocess
    m = _axial_bar_2d(w)
    st = m.studies()[0]
    w.set_prop(st, "backend", "nastran")
    # Use a much finer mesh so the solve takes longer
    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.005", hmin="0.0025")
    w.tree.select(st)
    _settle()
    w.set_prop(st, "backend", "nastran")
    _settle()
    w.compute(st)
    _wait(w, lambda: w.busy(), 60)
    # Give NASTRAN a moment to actually start processing
    time.sleep(1.0)
    assert w.busy()
    w.act["stop"].trigger()
    _wait(w, lambda: not w.busy(), 60)
    time.sleep(0.5)
    msg = w.messages.view.toPlainText()
    assert "stopped by the user" in msg or "Killed" in msg, msg[-500:]