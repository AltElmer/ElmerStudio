"""Desktop acceptance checks, shared by the test suite and the packaged application's self-test
(``ElmerStudio --selftest-ui``), so the frozen build is checked by the same code as a development checkout.

Each check returns a list of problems (empty: passed). They read the widgets as laid out, so run them on a shown
window after the event loop has settled.
"""
from __future__ import annotations

from PySide6.QtWidgets import QAbstractButton, QLabel, QLineEdit, QToolButton, QWidget


def _visible(w: QWidget) -> bool:
    return w.isVisibleTo(w.window()) and w.isVisible()


def ribbon_reachability(ribbon) -> tuple[set, list[str]]:
    """Every command of every visible ribbon tab is reachable at the current width: as a visible button, or in the
    menu of its collapsed group. Returns (reachable actions, problems)."""
    reach, problems = set(), []
    for name, tab in ribbon.tabs.items():
        if not ribbon.tab_visible(name):
            continue
        ribbon.show_tab(name)
        tab.fit(ribbon.stack.width())
        for g in tab.groups:
            if g.collapsed:
                if not g.compact.isVisibleTo(tab):
                    problems.append(f"{name} > {g.title}: collapsed but its button is hidden")
                    continue
                g._fill_menu()
                listed = set()
                for a in g.menu.actions():
                    listed.add(a)
                    if a.menu() is not None:
                        listed |= set(a.menu().actions())
                for a in g.actions_:
                    if a in listed or (a.menu() is not None and any(x.text() == a.text().replace("\n", " ")
                                                                    for x in g.menu.actions())):
                        reach.add(a)
                    else:
                        problems.append(f"{name} > {g.title} > {a.text()!r}: not in the collapsed group's menu")
            else:
                for b in g.full.findChildren(QToolButton):
                    if b.defaultAction() is not None and b.isVisibleTo(tab):
                        reach.add(b.defaultAction())
                for a in g.actions_:
                    if a not in reach:
                        problems.append(f"{name} > {g.title} > {a.text()!r}: no visible button")
        # a group whose full width does not fit must be collapsed (else its buttons are cut off)
        right = max((g.geometry().right() for g in tab.groups if g.isVisible()), default=0)
        if right > tab.width() + 1:
            problems.append(f"{name}: groups extend {right - tab.width()} px past the window")
    return reach, problems


def clipped_widgets(root: QWidget) -> list[str]:
    """Visible text widgets smaller than they need to show their text (cut-off labels and buttons)."""
    out = []
    for w in root.findChildren(QWidget):
        if not isinstance(w, (QLabel, QAbstractButton)) or not _visible(w):
            continue
        text = w.text() if hasattr(w, "text") else ""
        if not text or (isinstance(w, QLabel) and (w.wordWrap() or w.pixmap() is not None and not w.pixmap().isNull())):
            continue
        if isinstance(w, QLabel) and w.textFormat() != 0 and "<" in text:     # rich text: no reliable metric
            continue
        need = w.minimumSizeHint()
        if isinstance(w, QLabel):
            fm = w.fontMetrics()
            lines = text.split("\n")
            m = w.contentsMargins()
            need_w = max(fm.horizontalAdvance(ln) for ln in lines) + m.left() + m.right() + 2 * w.margin()
            need_h = fm.height() * len(lines) + m.top() + m.bottom()
        else:
            need_w, need_h = need.width(), need.height()
        if isinstance(w, QToolButton) and w.toolButtonStyle() == 0:          # icon only: no text to clip
            continue
        if w.width() + 1 < need_w or w.height() + 1 < need_h:
            out.append(f"{type(w).__name__} {text!r}: {w.width()}x{w.height()} < needed {need_w}x{need_h}")
    return out


def unnamed_controls(root: QWidget) -> list[str]:
    """Visible controls a screen reader cannot name: no text, no accessible name, no tooltip."""
    out = []
    for w in root.findChildren(QWidget):
        if not _visible(w):
            continue
        if isinstance(w, (QAbstractButton,)):
            name = (w.text() or "").strip() or w.accessibleName() or w.toolTip()
            if isinstance(w, QToolButton) and w.defaultAction() is not None:
                name = name or w.defaultAction().text() or w.defaultAction().toolTip()
            if not name:
                out.append(f"{type(w).__name__} at {w.mapTo(root, w.rect().topLeft()).toTuple()} "
                           f"in {type(w.parentWidget()).__name__}")
        elif isinstance(w, QLineEdit) and not w.isReadOnly():
            if not (w.accessibleName() or w.placeholderText() or w.toolTip()):
                parent = w.parentWidget()
                # a field in a form row is named by its label
                labelled = parent is not None and any(isinstance(c, QLabel) and c.buddy() is w
                                                      for c in parent.findChildren(QLabel))
                if not labelled and not w.property("prop_key"):
                    out.append(f"QLineEdit {w.text()!r} in {type(parent).__name__}")
    return out


def run_ui_selftest(w, widths=(1600, 1024, 760)) -> list[str]:
    """The packaged app's desktop self-test on the open model: ribbon reachability at several widths, both layouts,
    clipping and names, the recovery checkpoint. Returns the problems found."""
    from PySide6.QtWidgets import QApplication
    problems = []

    def settle():
        for _ in range(10):
            QApplication.processEvents()

    all_actions = {a for t in w.ribbon.tabs.values() for g in t.groups for a in g.actions_}
    for mode in ("widescreen", "regular"):
        w.apply_layout(mode, restore=False)
        for width in widths:
            w.resize(width, 820)
            settle()
            reach, p = ribbon_reachability(w.ribbon)
            problems += [f"{mode} {width}px: {x}" for x in p]
            visible_tabs = {a for n, t in w.ribbon.tabs.items() if w.ribbon.tab_visible(n)
                            for g in t.groups for a in g.actions_}
            missing = (all_actions & visible_tabs) - reach
            problems += [f"{mode} {width}px: unreachable {a.text()!r}" for a in missing]
            problems += [f"{mode} {width}px: clipped {x}" for x in clipped_widgets(w.ribbon)]
            problems += [f"{mode} {width}px: clipped {x}" for x in clipped_widgets(w.statusBar())]
        problems += [f"{mode}: unnamed {x}" for x in unnamed_controls(w)]
    w.ribbon.show_tab("Home")
    w.apply_layout("widescreen", restore=False)
    w.resize(1500, 900)
    settle()
    # recovery: an unsaved edit is checkpointed, a save or close removes it
    was = w.dirty
    w.dirty = True
    w.checkpoint(force=True)
    if not w.recovery.has_checkpoint():
        problems.append("recovery: no checkpoint written")
    w.recovery.discard()
    w._checkpoint_rev = None
    if not was:
        w.dirty = False
    return problems
