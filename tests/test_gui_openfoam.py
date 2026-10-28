"""The OpenFOAM backend through the desktop: choose it, see why it is unavailable, show its input files, Compute
(log, convergence plot, default plot, solution status), and Stop a running solver (its process is gone).

Drives MainWindow on the GUI thread like tests/test_desktop.py; run GUI tests one at a time."""
import os
import subprocess
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


def _channel(w, N=8, Lf=2.0):
    _wait(w, lambda: not w.busy())
    m = B.new_model("2D", ["spf"], "stationary")
    m.global_defs.child("params").props["table"] = [{"name": "h_el", "expr": f"0.01/{N}", "descr": ""}]
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=f"{Lf * 0.01}", h="0.01")
    B.add_material(m, "Water, liquid", all_domains=True)
    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="h_el", hmin="h_el/10")
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.spf")
    B.add_feature(m, ph, "spf.inlet", geo.select("boundary", xmax=0), U0="0.001")
    B.add_feature(m, ph, "spf.outlet", geo.select("boundary", xmin=Lf * 0.01))
    w._set_model(m, None, project.new_workdir())
    w.build_geometry()
    _wait(w, lambda: not w.busy() and w.geo is not None)
    return m


def test_choose_openfoam_compute_plot(win, openfoam):
    w = win
    _channel(w)
    st = w.model.studies()[0]
    w.tree.select(st)
    _settle()
    assert st.get("backend") == "elmer"
    w.set_prop(st, "backend", "openfoam")
    _settle()
    w.update_actions()
    assert w.act["compute"].isEnabled(), w.act["compute"].toolTip()
    # a feature OpenFOAM does not map: Compute says why, and so does the study's mark
    ph = w.model.component.child("physics.spf")
    vf = B.add_feature(w.model, ph, "spf.vf", [1])
    w.tree.select(st)
    _settle()
    w.update_actions()
    tip = w.act["compute"].toolTip()
    assert not w.act["compute"].isEnabled() and "Unavailable:" in tip and "Volume Force" in tip, tip
    assert any(lvl == "error" and "Volume Force" in txt for lvl, txt in w.node_issues(st))
    ph.children.remove(vf)
    w.tree.refresh(select=st)
    w.update_actions()
    assert w.act["compute"].isEnabled()
    # Show Input Files: the dictionaries of a dry run
    w.show_sif(st)
    assert {"system/blockMeshDict", "system/controlDict", "system/fvSchemes", "system/fvSolution",
            "constant/transportProperties"} <= set(w._last_foam_files), w._last_foam_files
    assert "simpleFoam" in w._last_sif and "bnd" in w._last_sif
    # Compute: log, convergence plot, default plot, solved state
    w.compute(st)
    _wait(w, lambda: not w.busy(), 300)
    log = w.logpane.view.toPlainText()
    assert "SIMPLE solution converged" in log and "blockMesh" in log, log[-2000:]
    assert any(k.startswith(("Ux", "p")) and "initial residual" in k for k in w.conv.series), list(w.conv.series)
    assert w.plot_groups(), "no default plot"
    assert w.current_plot is not None
    st = w.model.studies()[0]
    assert w._solution_state(st)[1] == "ok", (w._solution_state(st), w.messages.view.toPlainText()[-1500:])
    inlet = next(f for f in w.model.component.child("physics.spf").children if f.kind == "spf.inlet")
    w.set_prop(inlet, "U0", "0.002")
    _settle()
    assert w._solution_state(w.model.studies()[0])[1] == "stale"


def _solver_running() -> bool:
    r = subprocess.run(["wsl", "-e", "pgrep", "-x", "simpleFoam"], capture_output=True, text=True)
    return r.returncode == 0


def test_stop_kills_the_openfoam_process(win, openfoam):
    w = win
    if openfoam.kind != "wsl":
        pytest.skip("the process check here looks inside WSL")
    assert not _solver_running(), "a simpleFoam is already running"
    m = _channel(w, N=24, Lf=40.0)
    st = m.studies()[0]
    w.set_prop(st, "backend", "openfoam")
    sols = B.ensure_solver_configs(w.model, st)
    sols.children[0].props.update({"nl_tol": "1e-14", "nl_maxit": 100000})     # runs until stopped
    w.compute(st)
    _wait(w, lambda: len(next(iter(w.conv.series.values()), [])) > 20, 300)
    assert _solver_running()
    w.act["stop"].trigger()
    _wait(w, lambda: not w.busy(), 120)
    time.sleep(1.0)
    assert not _solver_running(), "simpleFoam still runs after Stop"
    assert "stopped by the user" in w.messages.view.toPlainText()
