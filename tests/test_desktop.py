"""Desktop familiarity (M1): real keyboard and mouse events through Qt (QTest), not handler calls.

Ribbon collapse and minimize, the two layouts, disabled commands that say why, no enabled command that does nothing,
transactional Settings fields (Enter applies, Escape restores, a paste is one undo step), completion, Paste
Selection, the Model Builder filter and its state marks, links from Messages to nodes, F1 help per node, focus
traversal, picking in the VTK Graphics window with the mouse and stepping to hidden faces with the keyboard.
Every detector here also has a negative control that must fire."""
import os
import tempfile
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QSettings, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QInputDialog, QLabel,  # noqa: E402
                               QMessageBox)

from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core import project  # noqa: E402
from elmerstudio.core.geometry import build_geometry  # noqa: E402
from elmerstudio.ui import acceptance  # noqa: E402

OPENED = []
MODALS = []


@pytest.fixture(scope="module")
def win():
    app = QApplication.instance() or QApplication([])
    from elmerstudio.ui import theme
    theme.apply(app, 9.0)
    from elmerstudio.ui.main_window import MainWindow
    for name in ("warning", "critical", "information", "question", "about"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: OPENED.append("messagebox") or 0))
    # a modal that slipped past the stubs would block the test forever: close it and fail instead
    from PySide6.QtCore import QTimer
    dog = QTimer()
    dog.timeout.connect(lambda: (MODALS.append(type(m).__name__), m.close())
                        if (m := QApplication.activeModalWidget()) is not None else None)
    dog.start(1000)
    QDialog.exec = lambda self, *a, **k: OPENED.append(type(self).__name__) or 0
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: OPENED.append("file") or ("", ""))
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: OPENED.append("file") or ("", ""))
    QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: OPENED.append("dir") or "")
    QInputDialog.getText = staticmethod(lambda *a, **k: OPENED.append("input") or ("", False))
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(prefix="es_qs_"), "s.ini"), QSettings.IniFormat)
    w.resize(1500, 900)
    w.show()
    w.activateWindow()
    _settle()
    yield w
    dog.stop()
    w.dirty = False
    w.stop_job()
    w.close()
    app.processEvents()
    assert not MODALS, f"modal dialogs opened during the tests: {MODALS}"


def _settle(n=15):
    for _ in range(n):
        QApplication.processEvents()
        time.sleep(0.005)


def _wait(w, cond, timeout=90):
    t0 = time.time()
    while not cond():
        QApplication.processEvents()
        if time.time() - t0 > timeout:
            raise TimeoutError
        time.sleep(0.01)
    _settle()


def _model(w, sdim="2D", geo=True):
    _wait(w, lambda: not w.busy())
    m = B.new_model(sdim, ["ht"], "stationary")
    if geo:
        g = m.component.child("geometry")
        if sdim == "3D":
            m.create("geom.block", g, index=0, w="2", d="1", h="1")
        else:
            m.create("geom.rectangle", g, index=0, w="2", h="1")
    w._set_model(m, None, project.new_workdir())
    w.build_geometry()
    _wait(w, lambda: not w.busy() and w.geo is not None)
    return m


# ---------------------------------------------------------------- ribbon
def test_every_ribbon_command_stays_reachable_as_the_window_narrows(win):
    w = win
    _model(w, "3D")
    pg = w.model.results.children[-1] if w.model.results.children else None
    w.add_plot_group("pg3d")                             # the contextual tab is shown too
    _settle()
    all_actions = {a for n, t in w.ribbon.tabs.items() for g in t.groups for a in g.actions_}
    collapsed_any = False
    for width in (1900, 1366, 1024, 800, 640):
        w.resize(width, 820)
        _settle()
        reach, problems = acceptance.ribbon_reachability(w.ribbon)
        visible = {a for n, t in w.ribbon.tabs.items() if w.ribbon.tab_visible(n) for g in t.groups for a in g.actions_}
        assert not problems, (width, problems)
        assert visible <= reach, (width, [a.text() for a in visible - reach])
        collapsed_any |= any(g.collapsed for t in w.ribbon.tabs.values() for g in t.groups)
    assert collapsed_any                                 # the narrow widths did collapse groups
    assert w.minimumSizeHint().width() <= 700            # the ribbon does not hold the window wide
    assert all_actions
    # negative control: a command missing from a collapsed group's menu is reported
    w.resize(640, 820)
    _settle()
    g = next(g for t in w.ribbon.tabs.values() for g in t.groups if g.collapsed and len(g.actions_) > 1)
    fill = g._fill_menu
    g._fill_menu = lambda: (fill(), g.menu.removeAction(g.menu.actions()[-1]))
    try:
        _reach, problems = acceptance.ribbon_reachability(w.ribbon)
    finally:
        g._fill_menu = fill
    assert any("not in the collapsed group's menu" in p for p in problems)
    w.resize(1500, 900)
    del pg


def test_a_collapsed_group_runs_the_same_command(win):
    w = win
    _model(w)
    w.resize(700, 820)
    w.ribbon.show_tab("Home")
    _settle()
    g = next(g for g in w.ribbon.tabs["Home"].groups if g.title == "Study")
    assert g.collapsed
    g._fill_menu()
    acts = [a for a in g.menu.actions() if a.menu() is None]
    assert w.act["compute"] in acts                      # the same QAction: same shortcut, availability, handler
    w.resize(1500, 900)
    _settle()


def test_minimized_ribbon_shows_a_tab_until_a_command_runs(win):
    w = win
    _model(w)
    r = w.ribbon
    bar = r.tabbar
    i = list(r.tabs).index("Home")
    QTest.mouseDClick(bar, Qt.LeftButton, pos=bar.tabRect(i).center())
    _settle()
    assert r.minimized and r.stack.isHidden() and w.act["minimize_ribbon"].isChecked()
    time.sleep(0.5)                                      # a later click, not the end of the double-click
    QTest.mouseClick(bar, Qt.LeftButton, pos=bar.tabRect(i).center())
    _settle()
    assert r.stack.isVisible()                           # shown on a click
    w.act["undo"].trigger() if w.act["undo"].isEnabled() else w.act["reset_desktop"].trigger()
    _settle()
    assert r.stack.isHidden()                            # folded away after the command
    r.set_minimized(False)
    _settle()
    assert r.stack.isVisible() and not w.act["minimize_ribbon"].isChecked()


def test_tabs_follow_the_model_and_the_plot_group_tab_is_named_after_it(win):
    w = win
    w._set_model(B.new_model(None), None, project.new_workdir())
    _settle()
    assert not w.ribbon.tab_visible("Geometry") and not w.ribbon.tab_visible("Physics")
    _model(w)
    assert w.ribbon.tab_visible("Geometry") and w.ribbon.tab_visible("Mesh")
    assert not w.ribbon.tab_visible("Plot Group")
    w.add_plot_group("pg2d")
    _settle()
    pg = w.tree.current()
    i = list(w.ribbon.tabs).index("Plot Group")
    assert w.ribbon.tab_visible("Plot Group") and w.ribbon.tabbar.tabText(i) == pg.label


# ---------------------------------------------------------------- layouts
def _dock_rects(w):
    return w.dock_tree.geometry(), w.dock_settings.geometry()


def test_widescreen_and_regular_layouts_keep_their_own_arrangement(win):
    w = win
    _model(w)
    w.apply_layout("widescreen", restore=False)
    _settle()
    t, s = _dock_rects(w)
    assert s.left() >= t.right() and abs(s.top() - t.top()) < 5            # side by side
    w.dock_msg.hide()                                                      # a change in this layout
    w.apply_layout("regular")
    _settle()
    t, s = _dock_rects(w)
    assert s.top() >= t.bottom() and abs(s.left() - t.left()) < 5         # stacked
    assert w.dock_msg.isVisible()                                          # its own arrangement
    w.apply_layout("widescreen")
    _settle()
    assert not w.dock_msg.isVisible()                                      # the widescreen change came back
    w.reset_desktop()
    _settle()
    assert w.dock_msg.isVisible() and w.act["layout_widescreen"].isChecked()


def test_maximize_graphics_and_back(win):
    w = win
    _model(w)
    w.apply_layout("widescreen", restore=False)
    _settle()
    bar = w.central.tabBar()
    QTest.mouseDClick(bar, Qt.LeftButton, pos=bar.tabRect(0).center())
    _settle()
    assert not w.dock_tree.isVisible() and not w.dock_settings.isVisible() and w.act["maximize"].isChecked()
    w.act["maximize"].trigger()
    _settle()
    assert w.dock_tree.isVisible() and w.dock_settings.isVisible() and not w.act["maximize"].isChecked()


# ---------------------------------------------------------------- availability
def test_unavailable_commands_say_why(win):
    w = win
    w._set_model(B.new_model(None), None, project.new_workdir())
    _settle()
    for name, why in (("compute", "Add a study first"), ("add_phys", "Add a component first"),
                      ("undo", "Nothing to undo"), ("delete", "fixed part of the model"),
                      ("stop", "Nothing is running"), ("geom.block", "Add a component first")):
        a = w.act[name]
        assert not a.isEnabled(), name
        assert "Unavailable: " in a.toolTip() and why in a.toolTip(), (name, a.toolTip())
    _model(w)
    for name in ("add_phys", "geom_build", "mesh_build", "add_study"):
        assert w.act[name].isEnabled(), (name, w.act[name].toolTip())
    assert "Unavailable" not in w.act["add_phys"].toolTip()
    assert not w.act["geom.block"].isEnabled() and "3D component" in w.act["geom.block"].toolTip()
    # the context menu uses the same actions, so the same rule and reason
    feat = w.model.component.child("physics.ht").children[0]
    w.tree.select(feat)
    _settle()
    assert feat.meta.get("default")
    assert not w.act["delete"].isEnabled() and "default feature" in w.act["delete"].toolTip()
    own = B.add_feature(w.model, w.model.component.child("physics.ht"), "ht.temperature", [], T0="1")
    w.tree.refresh(select=own)
    _settle()
    m = w.build_context_menu(own)
    assert w.act["delete"] in m.actions() and w.act["delete"].isEnabled()


def _state(w):
    sb = w.statusBar().currentMessage()
    return (w.rev, id(w.tree.current()), id(w.settings.node), tuple(d.isVisible() for d in (
        w.dock_tree, w.dock_settings, w.dock_msg, w.dock_prog, w.dock_log, w.dock_table, w.dock_mat)),
        w.ribbon.current_tab(), w.ribbon.minimized, len(w.messages.view.toPlainText()), sb, w.busy(),
        w.graphics.mode, w.gstack.currentIndex(), w.central.currentIndex(), getattr(w, "last_url", None),
        len(OPENED), str(w.graphics.plotter.camera_position) if w.graphics.plotter is not None else "",
        tuple(sorted(n for n, a in w.act.items() if a.isCheckable() and a.isChecked())),
        id(w.focusWidget()), w.layout_mode, w.graphics.draws, w.tree.tree.state().name)


# commands that only start a modal flow (file dialogs, the wizard) are covered by OPENED; these are left out:
# Exit-like and process-level commands
SKIP = {"next_window", "prev_window"}


def test_no_enabled_command_does_nothing(win):
    """Every enabled QAction changes something visible (the model, the selection, a window, a message, the status
    bar, a dialog, the camera, the focus...) or opens a menu with something enabled in it."""
    w = win
    dead = []
    for setup in ("blank", "2D", "3D-plot"):
        if setup == "blank":
            w._set_model(B.new_model(None), None, project.new_workdir())
            _settle()
            nodes = [w.model.root]
        elif setup == "2D":
            m = _model(w)
            feat = B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", [], T0="1")
            w.tree.refresh(select=feat)
            nodes = [feat, m.component.child("geometry")]
        else:
            m = _model(w, "3D")
            w.add_plot_group("pg3d")
            nodes = [w.tree.current().children[0]]
        for tag in [n.tag for n in nodes]:
            for name, a in sorted(w.act.items()):
                if name in SKIP:
                    continue
                node = w.model.by_tag(tag) if tag else w.model.root
                if node is None:                         # an earlier command removed it (Delete, Undo)
                    break
                w.tree.select(node)
                w.messages.view.setFocus()               # elsewhere: commands that move the focus are seen
                w.dock_mat.hide()                        # undo the previous command's window changes, so that a
                w.dock_table.hide()                      # command showing them again is seen to act
                _settle(3)
                w.update_actions()
                if not a.isEnabled() or not a.isVisible():
                    continue
                if a.menu() is not None:
                    a.menu().aboutToShow.emit()
                    if not any(x.isEnabled() for x in a.menu().actions() if not x.isSeparator()):
                        dead.append(f"{setup}/{node.kind}: {name} (menu with nothing enabled)")
                    continue
                before = _state(w)
                kind = node.kind
                a.trigger()
                _settle(3)
                _wait(w, lambda: not w.busy(), 120)
                if _state(w) == before:
                    dead.append(f"{setup}/{kind}: {name}")
                if w.tree.tree.state() == w.tree.tree.State.EditingState:
                    w.tree.tree.setFocus()
                    QTest.keyClick(w.tree.tree.focusWidget() or w.tree.tree, Qt.Key_Escape)
                if w.ribbon.minimized:
                    w.ribbon.set_minimized(False)
                if w._maximized_state is not None:
                    w.toggle_maximize()
                w.tree.filter.clear()
                if w.model is not None and setup != "blank" and w.model.component is None:
                    break
    assert not dead, "enabled commands that did nothing:\n" + "\n".join(dead)


# ---------------------------------------------------------------- settings: transactional fields
def _feature_field(w, key="T0"):
    m = _model(w)
    feat = B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", [], T0="300[K]")
    w.tree.refresh(select=feat)
    _settle()
    e = w.settings.editor(key)
    assert e is not None
    return feat, e


def test_enter_applies_escape_restores(win):
    w = win
    feat, e = _feature_field(w)
    n = len(w.undo_stack)
    e.setFocus()
    QTest.keyClick(e, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(e, "350[K]")
    assert e.property("draft") is True and feat.props["T0"] == "300[K]"     # a draft, not applied yet
    QTest.keyClick(e, Qt.Key_Escape)
    assert e.text() == "300[K]" and not e.property("draft")                # restored
    assert feat.props["T0"] == "300[K]" and len(w.undo_stack) == n
    QTest.keyClick(e, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(e, "350[K]")
    QTest.keyClick(e, Qt.Key_Return)
    _settle()
    assert feat.props["T0"] == "350[K]" and len(w.undo_stack) == n + 1     # applied: one edit
    assert not e.property("draft")


def test_an_invalid_expression_stays_a_draft_and_says_why(win):
    w = win
    feat, e = _feature_field(w)
    e.setFocus()
    QTest.keyClick(e, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(e, "2*(3")
    QTest.keyClick(e, Qt.Key_Return)
    _settle()
    assert feat.props["T0"] == "300[K]"
    assert e.note.isVisibleTo(w.settings) and e.note.text().startswith("Not applied")
    assert e.text() == "2*(3"                            # the typed text is kept for fixing
    QTest.keyClick(e, Qt.Key_Escape)
    assert e.text() == "300[K]" and not e.note.isVisibleTo(w.settings)


def test_completion_lists_parameters_with_values(win):
    w = win
    m = _model(w)
    params = m.global_defs.child("params")
    w.set_prop(params, "table", [{"name": "Tamb", "expr": "293.15[K]", "descr": "ambient"}], rebuild=True)
    feat = B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", [], T0="1")
    w.tree.refresh(select=feat)
    _settle()
    e = w.settings.editor("T0")
    e.setFocus()
    QTest.keyClick(e, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(e, "Tam")
    c = e.completer
    assert c is not None and c.completionCount() >= 1 and c.currentCompletion() == "Tamb"
    model = c.completionModel()
    assert "293.15" in model.index(0, 1).data() and "ambient" in model.index(0, 1).data()
    c.activated[str].emit("Tamb")
    assert e.text() == "Tamb"
    QTest.keyClick(e, Qt.Key_Escape)                     # closes the list (or restores the field)


def test_pasting_parameter_rows_is_one_undo_step(win, monkeypatch):
    w = win

    class Clip:                      # the desktop's clipboard may be held by another program: a stand-in
        text_ = ""

        def setText(self, t):
            self.text_ = t

        def text(self):
            return self.text_
    clip = Clip()
    monkeypatch.setattr(QApplication, "clipboard", staticmethod(lambda: clip))
    m = _model(w)
    params = m.global_defs.child("params")
    w.tree.select(params)
    _settle()
    t = w.settings.editor("table")
    n = len(w.undo_stack)
    before = list(params.props.get("table") or [])
    t.setCurrentCell(len(before), 0)
    QApplication.clipboard().setText("a\t1[mm]\nb\t2*a\nc\t3[s]\n")
    t.setFocus()
    QTest.keyClick(t, Qt.Key_V, Qt.ControlModifier)
    _settle()
    rows = params.props["table"]
    assert [r["name"] for r in rows[len(before):]] == ["a", "b", "c"] and len(w.undo_stack) == n + 1
    w.undo()
    assert (w.model.global_defs.child("params").props.get("table") or []) == before


def test_paste_selection(win):
    from elmerstudio.ui.settings import parse_entities
    assert parse_entities("1, 3-5 and 8;9") == [1, 3, 4, 5, 8, 9]
    assert parse_entities("2:4\t6") == [2, 4, 6]
    assert parse_entities("7\t2") == [2, 7]
    with pytest.raises(ValueError, match="'x'"):
        parse_entities("1, x")
    with pytest.raises(ValueError, match="not in the geometry: 9"):
        parse_entities("1 9", available=[1, 2, 3])
    w = win
    m = _model(w)
    feat = B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", [], T0="1")
    w.tree.refresh(select=feat)
    _settle()
    n = len(w.undo_stack)
    assert w.settings.paste_selection(feat, w.geo.numbers["boundary"], "1, 3-4")
    feat = w.model.by_tag(feat.tag)
    assert feat.selection.entities == [1, 3, 4] and len(w.undo_stack) == n + 1
    assert not w.settings.paste_selection(feat, w.geo.numbers["boundary"], "99")     # refused, said why
    assert "not in the geometry" in w.messages.view.toPlainText()


def test_a_refresh_keeps_the_field_you_are_in(win):
    w = win
    feat, e = _feature_field(w)
    e.setFocus()
    _settle()
    if w.focusWidget() is not e:
        pytest.skip("the test window does not get keyboard focus here")
    w.settings.refresh()
    _settle()
    fw = w.focusWidget()
    assert fw is not None and fw.property("prop_key") == "T0"


# ---------------------------------------------------------------- Model Builder
def test_filter_hides_non_matching_nodes_and_escape_clears_it(win):
    w = win
    m = _model(w)
    cur = m.component.child("geometry")
    w.tree.select(cur)
    w.tree.focus_filter()
    QTest.keyClicks(w.tree.filter, "insulation")
    _settle()
    shown = [w.tree.node_of(it) for it in w.tree._items.values() if not it.isHidden()]
    assert any("Insulation" in n.label for n in shown if n is not None)
    assert all(n.kind in ("root", "component", "physics.ht") or "Insulation" in n.label
               for n in shown), [n.label for n in shown]
    assert w.tree.current() is cur                       # filtering keeps the selection
    QTest.keyClick(w.tree.filter, Qt.Key_Escape)
    _settle()
    assert w.tree.filter.text() == "" and all(not it.isHidden() for it in w.tree._items.values())


def test_node_states_are_explained(win):
    w = win
    m = _model(w)
    ht = m.component.child("physics.ht")
    feat = B.add_feature(m, ht, "ht.temperature", [], T0="1")
    w.tree.refresh(select=feat)
    _settle()
    tip = w.tree._items[id(feat)].toolTip(0)
    assert "No boundaries selected" in tip
    w.set_enabled(feat, False)
    _settle()
    feat = w.model.by_tag(feat.tag)
    assert "Disabled" in w.tree._items[id(feat)].toolTip(0)
    mat = w.model.create("material", w.model.component.child("materials"))
    mat.selection.all = True
    w.tree.refresh(select=mat)
    _settle()
    assert any(lv == "error" and "Missing material properties" in t for lv, t in w.node_issues(mat))
    # the geometry is marked once edited after the build
    g = w.model.component.child("geometry")
    w.set_prop(g.children[0], "w", "3")
    _settle()
    assert ("stale", "Edited since it was built: build it again (F8).") in w.node_issues(g)
    assert "needs update" in w.st_geo.text() or w.st_geo.text().endswith("*")


def test_message_links_select_the_node(win):
    w = win
    m = _model(w)
    feat = m.component.child("physics.ht").children[0]
    w.tree.select(m.root)
    w.messages.add("about this feature", "warning", node=feat)
    assert "Warning: about this feature" in w.messages.view.toPlainText()     # said in words, not only color
    from PySide6.QtCore import QUrl
    w.messages.view.anchorClicked.emit(QUrl(f"node:{feat.tag}"))
    _settle()
    assert w.tree.current() is feat


# ---------------------------------------------------------------- keyboard
def test_f1_opens_the_help_of_the_selected_node(win):
    w = win
    m = _model(w)
    feat = m.component.child("physics.ht").children[0]
    w.tree.select(feat)
    w.activateWindow()
    w.tree.tree.setFocus()
    _settle()
    if not w.isActiveWindow():
        pytest.skip("the test window is not the active window: shortcuts do not reach it")
    w.last_url = None
    QTest.keyClick(w.tree.tree, Qt.Key_F1)
    _settle()
    assert w.last_url.endswith(f"/reference/physics-ht.html#{feat.kind.replace('.', '-')}"), w.last_url
    QTest.keyClick(w.tree.tree, Qt.Key_F1, Qt.ControlModifier)
    assert w.last_url.rstrip("/").endswith("ElmerStudio")


def test_every_node_kind_has_a_help_section():
    """F1 targets exist: every registered node type's page and anchor are in the generated Reference."""
    import elmerstudio.core.geometry  # noqa: F401
    import elmerstudio.core.meshing  # noqa: F401
    import elmerstudio.core.nodes  # noqa: F401
    from elmerstudio.core.helpmap import help_page
    from elmerstudio.core.model import REGISTRY
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "modules")
    missing = []
    for kind in REGISTRY:
        page, anc = help_page(kind)
        mod, name = page.split("/", 1) if "/" in page else ("ROOT", page)
        path = os.path.join(root, mod, "pages", name + ".adoc")
        if not os.path.isfile(path):
            missing.append(f"{kind}: no page {page}")
            continue
        if anc and f"[#{anc}]" not in open(path, encoding="utf-8").read():
            missing.append(f"{kind}: no section #{anc} in {page}")
    assert not missing, missing
    assert help_page("no.such.kind")[0] == "user-interface"          # control: unknown kinds fall back


def test_keyboard_moves_between_windows(win):
    w = win
    _model(w)
    w.activateWindow()
    _settle()
    QTest.keyClick(w.tree.tree, Qt.Key_Right, Qt.ControlModifier | Qt.AltModifier)
    _settle()
    fw = QApplication.focusWidget()
    if fw is None:
        pytest.skip("the test window does not get keyboard focus here")
    assert w.settings.isAncestorOf(fw)
    QTest.keyClick(fw, Qt.Key_Left, Qt.ControlModifier | Qt.AltModifier)
    _settle()
    assert QApplication.focusWidget() is w.tree.tree
    QTest.keyClick(w.tree.tree, Qt.Key_Tab, Qt.ControlModifier)
    _settle()
    assert w.settings.isAncestorOf(QApplication.focusWidget())
    QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab, Qt.ControlModifier)
    _settle()
    assert QApplication.focusWidget() is w.graphics.plotter.interactor


def test_tree_shortcuts_rename_delete_undo(win):
    w = win
    m = _model(w)
    feat = B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", [], T0="1")
    w.tree.refresh(select=feat)
    w.tree.tree.setFocus()
    _settle()
    QTest.keyClick(w.tree.tree, Qt.Key_F2)
    _settle()
    ed = QApplication.focusWidget()
    if ed is None or ed is w.tree.tree:
        pytest.skip("the inline editor did not get the focus")
    QTest.keyClick(ed, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(ed, "Wall")
    QTest.keyClick(ed, Qt.Key_Return)
    _settle()
    assert w.model.by_tag(feat.tag).label == "Wall"
    QTest.keyClick(w.tree.tree, Qt.Key_Delete)
    _settle()
    assert w.model.by_tag(feat.tag) is None
    QTest.keyClick(w.tree.tree, Qt.Key_Z, Qt.ControlModifier)
    _settle()
    assert w.model.by_tag(feat.tag) is not None and w.model.by_tag(feat.tag).label == "Wall"


# ---------------------------------------------------------------- Graphics: real mouse picks
def _vtk_to_qt(w, xy):
    inter = w.graphics.plotter.interactor
    dpr = inter.devicePixelRatioF()
    h = w.graphics.plotter.ren_win.GetSize()[1]
    return QPoint(int(round(xy[0] / dpr)), int(round((h - 1 - xy[1]) / dpr)))


def _world_to_display(w, p):
    r = w.graphics.plotter.renderer
    r.SetWorldPoint(float(p[0]), float(p[1]), float(p[2]), 1.0)
    r.WorldToDisplay()
    return r.GetDisplayPoint()[:2]


def _graphics_ok(w):
    return w.graphics._ok and w.graphics.plotter.interactor.isVisible() and \
        w.graphics.plotter.ren_win.GetSize()[0] > 50


def test_clicking_a_boundary_in_the_graphics_selects_it(win):
    w = win
    m = _model(w)                                        # 2 x 1 rectangle
    feat = B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", [], T0="1")
    w.tree.refresh(select=feat)
    _settle(30)
    if not _graphics_ok(w):
        pytest.skip("no rendered Graphics window")
    assert w.active_sel is feat and w.graphics.level == "boundary"
    disp = w.geo.display["boundary"]
    target = next(b for b, (P, _S) in disp.items() if abs(P[:, 1].max() - P[:, 1].min()) < 1e-9
                  and P[:, 1].min() < 1e-9)                # the bottom edge y = 0
    P, _S = disp[target]
    mid = P.mean(axis=0)
    pos = _vtk_to_qt(w, _world_to_display(w, mid))
    inter = w.graphics.plotter.interactor
    QTest.mouseMove(inter, pos)
    _settle()
    QTest.mouseClick(inter, Qt.LeftButton, Qt.NoModifier, pos)
    _settle()
    feat = w.model.by_tag(feat.tag)
    assert target in feat.selection.entities, (target, feat.selection.entities)
    # a small drag does not select (it pans)
    other = [b for b in disp if b != target][0]
    P2, _S = disp[other]
    pos2 = _vtk_to_qt(w, _world_to_display(w, P2.mean(axis=0)))
    QTest.mousePress(inter, Qt.LeftButton, Qt.NoModifier, pos2)
    QTest.mouseMove(inter, pos2 + QPoint(30, 0))
    QTest.mouseRelease(inter, Qt.LeftButton, Qt.NoModifier, pos2 + QPoint(30, 0))
    _settle()
    assert other not in w.model.by_tag(feat.tag).selection.entities


def test_up_down_reach_a_hidden_face(win):
    """A 3D block seen from the default view: the ray under the cursor crosses a front and a back face; Down steps
    to the back one and a click selects it (the control: a plain click selects the front face)."""
    w = win
    m = _model(w, "3D")
    feat = B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", [], T0="1")
    w.tree.refresh(select=feat)
    _settle(30)
    if not _graphics_ok(w):
        pytest.skip("no rendered Graphics window")
    import numpy as np
    b = w.geo.bbox
    centre = np.array([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2])
    xy = _world_to_display(w, centre)
    cands = w.graphics.candidates_at(xy)
    assert len(cands) >= 2, cands
    inter = w.graphics.plotter.interactor
    pos = _vtk_to_qt(w, xy)
    QTest.mouseMove(inter, pos)
    _settle()
    w.graphics._mouse = xy                               # the exact display point (rounding of the Qt position)
    w.graphics.hover = w.graphics._pick(xy)
    assert w.graphics.hover == cands[0]                  # the picker and the ray agree on the front face
    inter.setFocus()
    QTest.keyClick(inter, Qt.Key_Down)
    _settle()
    assert w.graphics.hover == cands[1]
    w.graphics._cand_pos = w.graphics._mouse
    QTest.mouseClick(inter, Qt.LeftButton, Qt.NoModifier, pos)
    _settle()
    assert w.model.by_tag(feat.tag).selection.entities == [cands[1]]


# ---------------------------------------------------------------- appearance detectors and their controls
def test_clipping_and_naming_detectors_fire_on_a_planted_defect(win):
    w = win
    _model(w)
    w.ribbon.set_minimized(False)
    w.resize(1500, 900)
    w.ribbon.show_tab("Home")
    _settle()
    g = next(g for g in w.ribbon.tabs["Home"].groups if not g.collapsed)
    lab = QLabel("A deliberately long label that cannot fit", g.full)
    lab.setFixedWidth(20)
    lab.show()
    from PySide6.QtWidgets import QToolButton
    nameless = QToolButton(g.full)
    nameless.show()
    w.ribbon.show_tab("Home")
    _settle()
    try:
        assert any("deliberately long" in p for p in acceptance.clipped_widgets(w.ribbon))
        assert any(p.startswith("QToolButton") for p in acceptance.unnamed_controls(w.ribbon))
    finally:
        for x in (lab, nameless):
            x.hide()
            x.setParent(None)
        _settle()
    assert not [p for p in acceptance.clipped_widgets(w.ribbon) if "deliberately" in p]
    assert not acceptance.unnamed_controls(w)


# ---------------------------------------------------------------- results feedback
def test_a_constant_field_is_labelled_with_its_value():
    import numpy as np
    from elmerstudio.core.model import Node
    from elmerstudio.ui.plots import _clim
    c = _clim(Node("plot.surface"), np.full(7, 0.0))
    assert c[0] < 0 < c[1] and c.data == (0.0, 0.0)          # a band around 0 for the colors, 0 on the markers
    c = _clim(Node("plot.surface"), np.full(7, 350.0))
    assert c[0] < 350 < c[1] and c.data == (350.0, 350.0)
    c = _clim(Node("plot.surface"), np.array([1.0, 2.0]))
    assert tuple(c) == (1.0, 2.0) == c.data                    # control: an ordinary range is unchanged


def test_solution_status_follows_the_inputs(win, elmer):
    """Solved right after Compute (the default solver settings the run adds are not an edit); out of date after a
    physics edit; solved again after Undo; a plot edit (Results) does not make it out of date."""
    w = win
    w.inst = elmer
    m = _model(w)
    ht = m.component.child("physics.ht")
    B.add_feature(m, ht, "ht.temperature", [1], T0="300[K]")
    B.add_feature(m, ht, "ht.temperature", [3], T0="350[K]")
    w.add_library_material("Copper")
    st = w.model.studies()[0]
    w.compute(st)
    _wait(w, lambda: not w.busy(), 300)
    st = w.model.studies()[0]
    w.tree.select(st)
    _settle()
    assert w._solution_state(st)[1] == "ok", (w._solution_state(st), w.messages.view.toPlainText()[-1500:])
    pg = w.plot_groups()[0]
    w.set_prop(pg, "title", "edited title")
    assert w._solution_state(st)[1] == "ok"
    feat = next(f for f in w.model.component.child("physics.ht").children if f.get("T0") == "350[K]")
    w.set_prop(feat, "T0", "360[K]")
    _settle()
    assert w._solution_state(st)[1] == "stale" and ("stale", w._solution_state(st)[2]) in w.node_issues(st)
    assert "out of date" in w.st_sol.text() or w.st_sol.text().endswith("*")
    w.undo()
    _settle()
    assert w._solution_state(w.model.studies()[0])[1] == "ok"
