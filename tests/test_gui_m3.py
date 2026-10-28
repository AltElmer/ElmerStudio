"""M3 in the window: probes fill a Probe Table and a Probe Plot Group when a study is computed, and an Animation
export writes the frames of a plot group (GIF or PNG files); an animation of a single solution is refused."""
import glob
import math
import os
import tempfile
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from elmerstudio.core import project  # noqa: E402
from test_time_steps import model as decay_model  # noqa: E402


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
    yield w
    w.dirty = False
    w.stop_job()
    w.close()
    app.processEvents()


def _settle(w, n=10, limit=300):
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.005)
    t0 = time.time()
    while w.busy() and time.time() - t0 < limit:
        QApplication.processEvents()
        time.sleep(0.01)
    for _ in range(n):
        QApplication.processEvents()


def test_probes_and_animation_in_the_window(win, elmer):
    w = win
    _settle(w)
    m, _geo, st, L = decay_model("range(0,0.05,0.5)")
    defs = m.component.child("definitions")
    probe = m.create("probe.domain", defs, expr="c", ptype="max")
    w._set_model(m, None, project.new_workdir())
    w.build_geometry()
    _settle(w)
    w.compute(st)
    _settle(w)
    tables = [t for t in m.results.child("results.tables").children if t.meta.get("source") == f"probes:{st.tag}"]
    assert len(tables) == 1 and tables[0].label == "Probe Table 1"
    rows = tables[0].props["rows"]
    t = np.array([r[0] for r in rows])
    assert len(rows) == 10 and np.allclose([r[1] for r in rows], np.exp(-t), rtol=3e-3)
    assert tables[0].props["columns"][1].startswith(probe.label)
    pg = next(g for g in w.plot_groups() if g.meta.get("source") == f"probes:{st.tag}")
    assert pg.kind == "pg1d" and pg.children[0].kind == "plot.tablegraph" and pg.children[0].get("table") == tables[0].tag
    w.compute(st)                                     # computing again updates the same table and plot group
    _settle(w)
    assert sum(1 for g in w.plot_groups() if g.meta.get("source") == f"probes:{st.tag}") == 1

    # Animation of the default 2D plot group: 4 frames spread over the 10 output times
    surf = next(g for g in w.plot_groups() if g.kind == "pg2d")
    keep = surf.props.get("solnum", "last")
    exps = m.results.child("results.export")
    out = tempfile.mkdtemp(prefix="es_anim_out_")
    an = m.create("export.anim", exps, source=surf.tag, frames=4, fps=5, w=320, h=240,
                  file=os.path.join(out, "decay.gif"))
    w.export_node(an)
    _settle(w)
    from PIL import Image
    with Image.open(os.path.join(out, "decay.gif")) as im:
        assert im.n_frames == 4 and im.size == (320, 240)
    assert surf.props.get("solnum", "last") == keep             # the plot group's own selection is restored
    an.props.update(format="png", frames=0, file=os.path.join(out, "seq.png"))
    w.export_node(an)
    _settle(w)
    assert len(glob.glob(os.path.join(out, "seq_*.png"))) == 10

    # control: a stationary study has one solution, nothing to animate
    from elmerstudio.core import builder as B
    st2 = B.add_study(m, "stationary")
    w.compute(st2)
    _settle(w)
    pg2 = next(g for g in w.plot_groups() if g.get("data") == B.solution_dataset(m, st2).tag)
    bad = m.create("export.anim", exps, source=pg2.tag, file=os.path.join(out, "one.gif"))
    w.messages.clear()
    w.export_node(bad)
    assert not os.path.exists(os.path.join(out, "one.gif"))
    assert "several solutions" in w.messages.view.toPlainText()
    assert math.isfinite(rows[-1][1])
