"""Ribbon toolbar: quick-access icons, a File menu button, tabs of grouped commands.

The ribbon adapts to the window width: when a tab's groups do not fit, groups collapse, from the right, into one
button with the group's title that opens a menu of all its commands (as in COMSOL and Office), so every command stays
reachable at any width. Double-clicking a tab (or Ctrl+F1) minimizes the ribbon to its tab row; a tab clicked while
minimized shows the panel until a command is chosen.
"""
from __future__ import annotations

import time

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLayout, QMenu, QSizePolicy, QStackedWidget, QTabBar,
                               QToolButton, QVBoxLayout, QWidget)

from . import theme


class RibbonGroup(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("RibbonGroup")
        self.title = title
        self.actions_: list[QAction] = []          # in display order, for the collapsed menu
        m = theme.metrics()
        self._m = m
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.full = QWidget(self)
        v = QVBoxLayout(self.full)
        v.setContentsMargins(3, 2, 3, 1)
        v.setSpacing(0)
        self.row = QHBoxLayout()
        self.row.setSpacing(1)
        self.row.setContentsMargins(0, 0, 0, 0)
        v.addLayout(self.row, 1)
        lbl = QLabel(title)
        lbl.setObjectName("RibbonGroupTitle")
        lbl.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        v.addWidget(lbl)
        outer.addWidget(self.full)
        # collapsed form: one large button, the group title, a menu of every command
        self.compact = QToolButton(self)
        self.compact.setObjectName("RibbonGroupCollapsed")
        self.compact.setText(title)
        self.compact.setToolTip(f"{title} (the window is too narrow to show this group's commands)")
        self.compact.setAccessibleName(f"{title} group")
        self.compact.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.compact.setIconSize(QSize(m.large_icon, m.large_icon))
        self.compact.setPopupMode(QToolButton.InstantPopup)
        self.compact.setAutoRaise(True)
        self.compact.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.compact.setMinimumWidth(m.large_button_w)
        self.menu = QMenu(self.compact)
        self.menu.aboutToShow.connect(self._fill_menu)
        self.compact.setMenu(self.menu)
        outer.addWidget(self.compact)
        self.compact.hide()
        self.collapsed = False

    def _btn(self, action: QAction, large: bool) -> QToolButton:
        m = self._m
        b = QToolButton(self)
        b.setDefaultAction(action)
        b.setAutoRaise(True)
        b.setAccessibleName(action.text().replace("\n", " "))
        if large:
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.setIconSize(QSize(m.large_icon, m.large_icon))
            b.setMinimumWidth(m.large_button_w)
            b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        else:
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.setIconSize(QSize(m.small_icon, m.small_icon))
            b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            b.setFixedHeight(m.small_button)
        if action.menu() is not None:
            b.setPopupMode(QToolButton.InstantPopup if action.property("instant") is not False
                           else QToolButton.MenuButtonPopup)
        if not self.actions_ or self.compact.icon().isNull():
            self.compact.setIcon(action.icon())
        self.actions_.append(action)
        return b

    def add_large(self, action: QAction) -> QToolButton:
        b = self._btn(action, True)
        self.row.addWidget(b)
        return b

    def add_small(self, actions: list[QAction]) -> list[QToolButton]:
        col = QVBoxLayout()
        col.setSpacing(0)
        col.setContentsMargins(0, 0, 0, 0)
        out = []
        for a in actions:
            b = self._btn(a, False)
            col.addWidget(b, 0, Qt.AlignLeft)
            out.append(b)
        col.addStretch(1)
        self.row.addLayout(col)
        return out

    def _fill_menu(self):
        """The collapsed group's menu: the same QActions (same shortcut, availability and handler)."""
        self.menu.clear()
        for a in self.actions_:
            if a.menu() is not None:
                sub = self.menu.addMenu(a.icon(), a.text().replace("\n", " "))
                sub.setEnabled(a.isEnabled())
                sub.setToolTip(a.toolTip())
                src = a.menu()
                src.aboutToShow.emit()          # menus filled on demand (Add Physics, Domains, ...)
                for sa in src.actions():
                    sub.addAction(sa)
            else:
                self.menu.addAction(a)

    def full_width(self) -> int:
        return self.full.sizeHint().width()

    def compact_width(self) -> int:
        return self.compact.sizeHint().width()

    def set_collapsed(self, on: bool):
        if on == self.collapsed:
            return
        self.collapsed = on
        self.full.setVisible(not on)
        self.compact.setVisible(on)


class RibbonTab(QWidget):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.h = QHBoxLayout(self)
        self.h.setContentsMargins(2, 2, 2, 0)
        self.h.setSpacing(0)
        self.h.addStretch(1)
        # the groups fold instead: the full groups' width must not become the window's minimum width
        self.h.setSizeConstraint(QLayout.SetNoConstraint)
        self.groups: list[RibbonGroup] = []
        self.seps: list[QFrame] = []

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

    def add_group(self, title: str) -> RibbonGroup:
        if self.groups:
            sep = QFrame(self)
            sep.setObjectName("RibbonSep")
            sep.setFixedWidth(1)
            sep.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            self.h.insertWidget(self.h.count() - 1, sep)
            self.seps.append(sep)
        g = RibbonGroup(title, self)
        self.h.insertWidget(self.h.count() - 1, g)
        self.groups.append(g)
        return g

    def fit(self, width: int | None = None):
        """Collapse groups from the right until the tab fits *width* (default: its own width); expand when there is
        room again."""
        width = self.width() if width is None else width
        m = self.h.contentsMargins()
        avail = width - m.left() - m.right() - len(self.seps) - 2
        full = [g.full_width() for g in self.groups]
        comp = [g.compact_width() for g in self.groups]
        state = [False] * len(self.groups)
        total = sum(full)
        for i in range(len(self.groups) - 1, -1, -1):
            if total <= avail:
                break
            state[i] = True
            total += comp[i] - full[i]
        for g, c in zip(self.groups, state):
            g.set_collapsed(c)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.fit(ev.size().width())

    def showEvent(self, ev):
        super().showEvent(ev)
        self.fit()


class Ribbon(QWidget):
    tabChanged = Signal(str)
    minimizedChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("RibbonBar")
        m = theme.metrics()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        top = QWidget(self)
        top.setObjectName("RibbonTabs")
        th = QHBoxLayout(top)
        th.setContentsMargins(4, 3, 4, 0)
        th.setSpacing(2)
        self.qat = QWidget(top)
        self.qat.setObjectName("QuickAccess")
        self.qat_layout = QHBoxLayout(self.qat)
        self.qat_layout.setContentsMargins(0, 0, 8, 0)
        self.qat_layout.setSpacing(1)
        th.addWidget(self.qat)
        self.file_btn = QToolButton(top)
        self.file_btn.setObjectName("FileButton")
        self.file_btn.setText("File")
        self.file_btn.setAccessibleName("File menu")
        self.file_btn.setPopupMode(QToolButton.InstantPopup)
        th.addWidget(self.file_btn)
        self.tabbar = QTabBar(top)
        self.tabbar.setDrawBase(False)
        self.tabbar.setExpanding(False)
        self.tabbar.setUsesScrollButtons(True)
        self.tabbar.setAccessibleName("Ribbon tabs")
        self.tabbar.currentChanged.connect(self._on_tab)
        self.tabbar.tabBarDoubleClicked.connect(lambda _i: self.set_minimized(not self.minimized))
        self.tabbar.tabBarClicked.connect(self._tab_clicked)
        th.addWidget(self.tabbar, 1)
        self.right = QHBoxLayout()
        self.right.setSpacing(1)
        th.addLayout(self.right)
        v.addWidget(top)
        self.stack = QStackedWidget(self)
        self.stack.setObjectName("RibbonPanel")
        self.stack.setFixedHeight(m.ribbon_panel)
        v.addWidget(self.stack)
        self.tabs: dict[str, RibbonTab] = {}
        self.minimized = False
        self._peek = False                      # minimized, panel shown after a tab click

    def add_quick(self, action: QAction):
        m = theme.metrics()
        b = QToolButton(self.qat)
        b.setDefaultAction(action)
        b.setAutoRaise(True)
        b.setIconSize(QSize(m.small_icon, m.small_icon))
        b.setToolButtonStyle(Qt.ToolButtonIconOnly)
        b.setAccessibleName(action.text().replace("\n", " "))
        self.qat_layout.addWidget(b)
        return b

    def add_quick_sep(self):
        s = QFrame(self.qat)
        s.setFrameShape(QFrame.VLine)
        s.setObjectName("RibbonSep")
        self.qat_layout.addWidget(s)

    def add_right(self, action: QAction):
        m = theme.metrics()
        b = QToolButton(self)
        b.setDefaultAction(action)
        b.setAutoRaise(True)
        b.setIconSize(QSize(m.small_icon, m.small_icon))
        b.setAccessibleName(action.text().replace("\n", " "))
        self.right.addWidget(b)
        return b

    def set_file_menu(self, menu: QMenu):
        self.file_btn.setMenu(menu)

    def add_tab(self, name: str) -> RibbonTab:
        t = RibbonTab(name)
        t.setObjectName("RibbonTabPage")
        self.tabs[name] = t
        self.stack.addWidget(t)
        self.tabbar.addTab(name)
        return t

    def _on_tab(self, i):
        if 0 <= i < self.stack.count():
            self.stack.setCurrentIndex(i)
            self.stack.currentWidget().fit(self.stack.width())
            self.tabChanged.emit(self.tabbar.tabText(i))

    def _index(self, name: str) -> int:
        return list(self.tabs).index(name) if name in self.tabs else -1

    def show_tab(self, name: str):
        i = self._index(name)
        if i < 0:                                    # by its shown text (a contextual tab is named after its node)
            i = next((k for k in range(self.tabbar.count()) if self.tabbar.tabText(k) == name), -1)
        if i >= 0 and self.tabbar.isTabVisible(i):
            self.tabbar.setCurrentIndex(i)

    def current_tab(self) -> str:
        i = self.tabbar.currentIndex()
        return list(self.tabs)[i] if 0 <= i < len(self.tabs) else ""

    def set_tab_visible(self, name: str, on: bool, text: str | None = None):
        """Show or hide a tab (tabs that do not apply yet, contextual tabs); a hidden current tab gives way to Home."""
        i = self._index(name)
        if i < 0:
            return
        if text is not None:
            self.tabbar.setTabText(i, text)
        if self.tabbar.isTabVisible(i) != on:
            self.tabbar.setTabVisible(i, on)
        if not on and self.tabbar.currentIndex() == i:
            self.tabbar.setCurrentIndex(0)

    def tab_visible(self, name: str) -> bool:
        i = self._index(name)
        return i >= 0 and self.tabbar.isTabVisible(i)

    # -- minimize
    def set_minimized(self, on: bool):
        self.minimized = on
        self._min_t = time.monotonic()
        self._peek = False
        self.stack.setVisible(not on)
        self.minimizedChanged.emit(on)

    def _tab_clicked(self, _i):
        # the press that ends a double-click (which minimized the ribbon) is not a click on a tab
        if self.minimized and time.monotonic() - getattr(self, "_min_t", 0.0) > 0.4:
            self._peek = not self._peek or self.stack.isHidden()
            self.stack.setVisible(self._peek)

    def command_triggered(self):
        """A command was chosen: a minimized ribbon folds its panel away again."""
        if self.minimized and self._peek:
            self._peek = False
            QTimer.singleShot(0, self.stack.hide)
