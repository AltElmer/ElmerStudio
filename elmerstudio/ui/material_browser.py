"""Add Material window: browse the material library and add materials to the component."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QTableWidget,
                               QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core.materials import LIBRARY, MATPROPS, library_categories
from . import icons
from .settings import pretty_unit


class MaterialBrowser(QWidget):
    addRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search")
        self.search.textChanged.connect(self._filter)
        v.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(QSize(16, 16))
        for cat, names in library_categories().items():
            ci = QTreeWidgetItem([cat])
            ci.setIcon(0, icons.icon("materials"))
            for n in names:
                it = QTreeWidgetItem([n])
                it.setIcon(0, icons.icon("material"))
                it.setData(0, Qt.UserRole, n)
                ci.addChild(it)
            self.tree.addTopLevelItem(ci)
        self.tree.expandAll()
        self.tree.currentItemChanged.connect(self._show)
        self.tree.itemDoubleClicked.connect(lambda it, c: it.data(0, Qt.UserRole) and self.addRequested.emit(it.data(0, Qt.UserRole)))
        v.addWidget(self.tree, 3)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        v.addWidget(self.note)
        self.props = QTableWidget(0, 3)
        self.props.setHorizontalHeaderLabels(["Property", "Value", "Unit"])
        self.props.verticalHeader().setVisible(False)
        self.props.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        v.addWidget(self.props, 2)
        row = QHBoxLayout()
        row.addStretch(1)
        b = QPushButton(icons.icon("add"), "Add to Component")
        b.clicked.connect(self._add)
        row.addWidget(b)
        v.addLayout(row)

    def _filter(self, text):
        t = text.lower()
        for i in range(self.tree.topLevelItemCount()):
            ci = self.tree.topLevelItem(i)
            vis = False
            for j in range(ci.childCount()):
                it = ci.child(j)
                h = bool(t) and t not in it.text(0).lower()
                it.setHidden(h)
                vis |= not h
            ci.setHidden(not vis)

    def _show(self, cur, prev):
        name = cur.data(0, Qt.UserRole) if cur else None
        self.props.setRowCount(0)
        if not name:
            self.note.setText("")
            return
        cat, vals, _rgb, note = LIBRARY[name]
        self.note.setText(f"<b>{name}</b> — {note}")
        for k, v in vals.items():
            r = self.props.rowCount()
            self.props.insertRow(r)
            label, _sym, unit, _kw, _g = MATPROPS[k]
            self.props.setItem(r, 0, QTableWidgetItem(label))
            self.props.setItem(r, 1, QTableWidgetItem(v))
            self.props.setItem(r, 2, QTableWidgetItem(pretty_unit(unit)))

    def _add(self):
        it = self.tree.currentItem()
        name = it.data(0, Qt.UserRole) if it else None
        if name:
            self.addRequested.emit(name)
