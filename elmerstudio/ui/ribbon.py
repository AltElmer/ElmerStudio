"""Ribbon toolbar: quick-access icons, a File menu button, tabs of grouped commands."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMenu, QSizePolicy, QStackedWidget, QTabBar,
                               QToolButton, QVBoxLayout, QWidget)

LARGE = QSize(32, 32)
SMALL = QSize(16, 16)


class RibbonGroup(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("RibbonGroup")
        v = QVBoxLayout(self)
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

    def _btn(self, action: QAction, large: bool) -> QToolButton:
        b = QToolButton(self)
        b.setDefaultAction(action)
        b.setAutoRaise(True)
        if large:
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            b.setIconSize(LARGE)
            b.setMinimumWidth(46)
            b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        else:
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.setIconSize(SMALL)
            b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            b.setFixedHeight(22)
        if action.menu() is not None:
            b.setPopupMode(QToolButton.InstantPopup if action.property("instant") is not False
                           else QToolButton.MenuButtonPopup)
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


class RibbonTab(QWidget):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.h = QHBoxLayout(self)
        self.h.setContentsMargins(2, 2, 2, 0)
        self.h.setSpacing(0)
        self.h.addStretch(1)
        self.groups: list[RibbonGroup] = []

    def add_group(self, title: str) -> RibbonGroup:
        if self.groups:
            sep = QFrame(self)
            sep.setObjectName("RibbonSep")
            sep.setFixedWidth(1)
            sep.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            self.h.insertWidget(self.h.count() - 1, sep)
        g = RibbonGroup(title, self)
        self.h.insertWidget(self.h.count() - 1, g)
        self.groups.append(g)
        return g


class Ribbon(QWidget):
    tabChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("RibbonBar")
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
        self.file_btn.setPopupMode(QToolButton.InstantPopup)
        th.addWidget(self.file_btn)
        self.tabbar = QTabBar(top)
        self.tabbar.setDrawBase(False)
        self.tabbar.setExpanding(False)
        self.tabbar.currentChanged.connect(self._on_tab)
        th.addWidget(self.tabbar)
        th.addStretch(1)
        self.right = QHBoxLayout()
        self.right.setSpacing(1)
        th.addLayout(self.right)
        v.addWidget(top)
        self.stack = QStackedWidget(self)
        self.stack.setObjectName("RibbonPanel")
        self.stack.setFixedHeight(88)
        v.addWidget(self.stack)
        self.tabs: dict[str, RibbonTab] = {}

    def add_quick(self, action: QAction):
        b = QToolButton(self.qat)
        b.setDefaultAction(action)
        b.setAutoRaise(True)
        b.setIconSize(SMALL)
        b.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.qat_layout.addWidget(b)
        return b

    def add_quick_sep(self):
        s = QFrame(self.qat)
        s.setFrameShape(QFrame.VLine)
        s.setStyleSheet("color:#c9cdd2")
        self.qat_layout.addWidget(s)

    def add_right(self, action: QAction):
        b = QToolButton(self)
        b.setDefaultAction(action)
        b.setAutoRaise(True)
        b.setIconSize(SMALL)
        self.right.addWidget(b)

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
            self.tabChanged.emit(self.tabbar.tabText(i))

    def show_tab(self, name: str):
        for i in range(self.tabbar.count()):
            if self.tabbar.tabText(i) == name:
                self.tabbar.setCurrentIndex(i)
                return
