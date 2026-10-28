"""M2 in the window: several components (the active one follows the selection, each keeps its own geometry), unit
warnings in fields and in the Model Builder, a selection that lost entities, identity pair nodes, "Used By", the
warning when deleting something in use, and the study's initial-values choice."""
import os
import tempfile
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core import project  # noqa: E402


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


def _settle(w, n=10):
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.005)
    t0 = time.time()
    while w.busy() and time.time() - t0 < 120:
        QApplication.processEvents()
        time.sleep(0.01)
    for _ in range(n):
        QApplication.processEvents()


def _model(w, sdim="2D"):
    _settle(w)
    m = B.new_model(sdim, ["ht"], "stationary")
    m.create("geom.rectangle" if sdim == "2D" else "geom.block", m.component.child("geometry"), index=0)
    w._set_model(m, None, project.new_workdir())
    w.build_geometry()
    _settle(w)
    return m


def test_components_have_their_own_geometry_and_the_active_one_follows_the_selection(win):
    w = win
    m = _model(w)
    c1 = m.component
    assert w.act["add_comp"].isEnabled()
    w.add_component("3D")
    _settle(w)
    c2 = m.components()[1]
    assert m.component is c2 and w.tree.current() is c2.child("geometry")
    w.add_geom("geom.block")                                # the ribbon acts on the active component
    assert c2.child("geometry").children[0].kind == "geom.block"
    assert not any(n.kind == "geom.block" for n in c1.walk())
    w.build_geometry()
    _settle(w)
    assert w.geo.sdim == 3 and "comp2" in w.st_geo.text()
    w.tree.select(c1.child("geometry"))
    _settle(w)
    assert m.component is c1 and w.geo is not None and w.geo.sdim == 2 and "comp1" in w.st_geo.text()
    assert w._slot(c2)["geo"].sdim == 3                    # comp2's geometry is still there
    # a parameter edit makes every component's geometry stale; a feature edit only its own
    for sl in w._slots.values():
        sl["geo_stale"] = False
    w.set_prop(c2.child("geometry").children[0], "w", "2")
    assert w._slot(c2)["geo_stale"] and not w._slot(c1)["geo_stale"]
    w.set_prop(m.global_defs.child("params"), "table", [{"name": "a", "expr": "1"}])
    assert w._slot(c1)["geo_stale"] and w._slot(c2)["geo_stale"]
    tags = [n.tag for n in m.root.walk()]
    assert len(tags) == len(set(tags))


def test_unit_warnings_show_in_the_field_and_the_tree(win):
    w = win
    m = _model(w)
    feat = B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", [1], T0="300[K]")
    w.tree.refresh(select=feat)
    _settle(w)
    e = w.settings.editor("T0")
    e.setText("2[m]")
    assert e.property("warn") is True and not e.blocking
    assert e.note.isVisibleTo(w.settings) and "[m] deduced" in e.note.text() and "[K] expected" in e.note.text()
    w.settings._commit_expr(e, next(p for p in __import__("elmerstudio.core.model", fromlist=["R"]).REGISTRY[
        feat.kind].props if p.key == "T0"), lambda v: w.set_prop(feat, "T0", v))
    assert feat.props["T0"] == "2[m]"                      # used, as COMSOL does
    _settle(w)
    assert any(lv == "warning" and "Unit:" in t for lv, t in w.node_issues(feat))
    e = w.settings.editor("T0")
    e.setText("310[K]")
    assert not e.property("warn") and not e.note.isVisibleTo(w.settings)       # control


def test_a_lost_entity_is_marked(win):
    w = win
    m = _model(w, "3D")
    g = m.component.child("geometry")
    m.create("geom.cylinder", g, index=1, r="0.2", h="1", x="3")
    w.build_geometry()
    _settle(w)
    geo = w.geo
    far = [b for b in geo.numbers["boundary"] if geo.bbox_of["boundary"][b][0] > 2.5]
    feat = B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", [], T0="1")
    w.tree.refresh(select=feat)
    w.set_selection(feat, entities=far)
    w.delete_node(g.children[1])
    w.build_geometry()
    _settle(w)
    feat = m.by_tag(feat.tag) or w.model.by_tag(feat.tag)
    assert feat.selection.entities == [] and any(lv == "error" and "no longer exist" in t
                                                 for lv, t in w.node_issues(feat))
    w.set_selection(feat, entities=[1])                    # picking again clears it
    assert not any("no longer exist" in t for _lv, t in w.node_issues(feat))


def test_pairs_used_by_and_initial_values(win):
    w = win
    m = _model(w, "3D")
    g = m.component.child("geometry")
    m.create("geom.block", g, index=1, x="1")
    w.set_prop(g.child("geom.finalize"), "action", "none")
    w.build_geometry()
    _settle(w)
    pairs = [n for n in m.component.child("definitions").children if n.kind == "pair.identity"]
    assert len(pairs) == 1 and pairs[0].props["objects"] == "blk1 and blk2"
    # Used By and the warning when deleting a node in use
    params = m.global_defs.child("params")
    w.set_prop(params, "table", [{"name": "Wd", "expr": "1"}])
    w.set_prop(g.children[0], "w", "Wd")
    titles = [t for t, _wid in w.extra_sections(params)]
    assert "Used By" in titles
    fn = w.create_node("func.analytic", m.global_defs)     # (Parameters 1 itself cannot be deleted)
    w.set_prop(fn, "fname", "f1")
    w.set_prop(g.children[0], "d", "f1(1)")
    w.delete_node(fn)
    assert "which 'Block 1' used" in w.messages.view.toPlainText()
    # the study's initial values: physics controlled, or another study (not itself)
    st2 = B.add_study(m, "time")
    w.tree.refresh(select=st2.children[0])
    _settle(w)
    cb = w.settings.editor("initsol")
    items = [cb.itemData(i) for i in range(cb.count())]
    assert items[0] == "" and m.studies()[0].tag in items and st2.tag not in items
