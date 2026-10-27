"""Main-window state: the modified flag follows the undo history, a failed Save As changes nothing, Stop drops the
queue and discards a build it cannot interrupt, a result built from an older model stays stale, Select All is one
edit, a syntax error in an expression is not applied, and a work plane in 3D offers the sketch primitives.

These drive the real MainWindow through its methods and signals (the Graphics window's VTK picking itself is not
exercised: it needs a rendered scene)."""
import os
import tempfile
import threading
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core import project  # noqa: E402
from elmerstudio.core.geometry import build_geometry  # noqa: E402


@pytest.fixture(scope="module")
def win():
    app = QApplication.instance() or QApplication([])
    from elmerstudio.ui.main_window import MainWindow
    for name in ("warning", "critical", "information"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    # never write the user's own settings (recent files, last folder)
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(prefix="es_qs_"), "s.ini"), QSettings.IniFormat)
    yield w
    w.dirty = False
    w.stop_job()
    w.close()
    app.processEvents()


def _wait(w, cond, timeout=60):
    t0 = time.time()
    while not cond():
        QApplication.processEvents()
        if time.time() - t0 > timeout:
            raise TimeoutError
        time.sleep(0.01)


def _model(w, sdim="2D"):
    _wait(w, lambda: not w.busy())                       # what the previous model started (a geometry build)
    m = B.new_model(sdim, ["ht"], "stationary")
    w._set_model(m, None, project.new_workdir())
    _wait(w, lambda: not w.busy())
    return m


def test_modified_flag_follows_the_undo_history(win):
    w = win
    m = _model(w)
    path = os.path.join(tempfile.mkdtemp(prefix="es_gui_"), "state.esm")
    assert w.file_save_as(path) and not w.dirty and w.file_path == path
    feat = m.component.child("physics.ht").children[0]
    w.set_prop(feat, "probe", "1")
    assert w.dirty
    w.undo()
    assert not w.dirty                                   # back at the saved state
    w.redo()
    assert w.dirty
    assert w.file_save() and not w.dirty
    w.undo()
    assert w.dirty                                       # the saved file has the edit; the model no longer does
    w.redo()
    assert not w.dirty                                   # forward to the saved state again


def test_failed_save_as_changes_nothing(win, monkeypatch):
    w = win
    _model(w)
    first = os.path.join(tempfile.mkdtemp(prefix="es_gui_"), "first.esm")
    assert w.file_save_as(first)
    label = w.model.root.label
    w.set_prop(w.model.component.child("physics.ht").children[0], "probe", "3")

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(project, "save", boom)
    assert not w.file_save_as(os.path.join(tempfile.mkdtemp(prefix="es_gui_"), "second.esm"))
    assert w.file_path == first and w.model.root.label == label and w.dirty


def test_stop_drops_the_queue_and_discards_an_uninterruptible_build(win):
    w = win
    _model(w)
    release, results = threading.Event(), []

    def slow(emit):
        release.wait(30)
        return "built"
    w._start_job(slow, "Slow build", results.append)
    w._start_job(lambda emit: "queued", "Queued build", results.append)
    assert w.busy() and len(w._queue) == 1
    w.stop_job()
    assert w._queue == [] and w.job.cancelled
    release.set()
    _wait(w, lambda: not w.busy())
    assert results == []                                 # neither the stopped nor the queued result was applied


def test_a_geometry_built_from_an_older_model_stays_stale(win):
    w = win
    m = _model(w)
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="2", h="1")
    said, add = [], w.messages.add
    w.messages.add = lambda text, kind="info": said.append(text) or add(text, kind)
    applied = []
    try:
        w.build_geometry(then=lambda: applied.append((w.geo_stale, w.geo.bbox[3] - w.geo.bbox[0])))
        w.set_prop(m.component.child("geometry").children[0], "w", "3")   # edited before it finishes
        _wait(w, lambda: not w.busy())
    finally:
        w.messages.add = add
    assert applied and applied[0][0] and applied[0][1] == pytest.approx(2, rel=1e-6)   # the 2 m result: stale
    assert any("changed while the geometry was built" in s for s in said), said
    w.build_geometry()
    _wait(w, lambda: not w.busy())
    assert not w.geo_stale and (w.geo.bbox[3] - w.geo.bbox[0]) == pytest.approx(3, rel=1e-6)


def test_select_all_and_clear_are_one_edit_each(win):
    w = win
    m = _model(w)
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="2", h="1")
    w.geo = build_geometry(m, display=False)
    feat = B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", [], T0="1")
    w.active_sel = feat
    w.graphics.geo, w.graphics.level = w.geo, "boundary"
    n = len(w.undo_stack)
    w.graphics.select_all()
    assert len(w.undo_stack) == n + 1 and feat.selection.entities == sorted(w.geo.numbers["boundary"])
    w.graphics.clear_selection()
    assert len(w.undo_stack) == n + 2 and feat.selection.entities == []
    w.undo()                                             # one step back: all selected again
    feat = w.model.by_tag(feat.tag)                      # (a restore rebuilds the nodes)
    assert feat.selection.entities == sorted(w.geo.numbers["boundary"])
    w.active_sel = feat
    w.graphics.entityClicked.emit("boundary", 1, False)  # a pick: one edit
    assert len(w.undo_stack) == n + 2 and 1 not in feat.selection.entities


def test_a_syntax_error_is_not_applied_an_unknown_name_is(win):
    w = win
    m = _model(w)
    feat = m.component.child("physics.ht").children[0]
    from elmerstudio.core.model import Prop
    from elmerstudio.ui.settings import ExprEdit
    p = Prop("Tx", "Test value:", default="1", unit="K")
    applied = []
    for text, ok in (("2*(3", False), ("4[furlongs]", False), ("later_param*2", True), ("5[K]", True)):
        e = ExprEdit(text, w.settings._validator(feat, p))
        w.settings._commit_expr(e, p, lambda v: applied.append(v))
        assert (text in applied) is ok, (text, e.toolTip())


def test_a_work_plane_offers_the_sketch_primitives(win):
    w = win
    m = _model(w, "3D")
    wp = m.create("geom.workplane", m.component.child("geometry"), index=0)
    rect = next(a for a in w.geo2d_actions if a.data() == "geom.rectangle")
    w._update_geom_actions(m.component.child("geometry"))
    assert not rect.isEnabled()
    w._update_geom_actions(wp)
    assert rect.isEnabled()
    w.tree.refresh(select=wp)                            # selecting it in the tree updates the ribbon
    QApplication.processEvents()
    assert rect.isEnabled()


def test_adding_a_second_heat_transfer_is_refused_in_the_window(win):
    w = win
    _model(w)
    n = len(w.undo_stack)
    w.add_physics("ht")
    assert len([c for c in w.model.component.children if c.kind == "physics.ht"]) == 1
    assert len(w.undo_stack) == n                        # nothing to undo
    w.add_physics("tds")
    w.add_physics("tds")
    assert [c.get("species") for c in w.model.component.children if c.kind == "physics.tds"] == ["c", "c2"]
