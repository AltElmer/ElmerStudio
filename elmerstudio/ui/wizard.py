"""New window and Model Wizard (space dimension -> physics -> study)."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
                               QStackedWidget, QTextBrowser, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from ..core.physics import BUNDLES, PHYSICS, STUDY_TYPES, WIZARD_TREE
from . import icons

STUDY_DESCR = {
    "stationary": "Used when field variables do not change over time. Examples: stationary heat transfer, "
                  "linear elastic stress, electrostatics, steady laminar flow.",
    "time": "Used when field variables change over time. Examples: transient heat transfer, diffusion, "
            "start-up of a flow.",
    "eigen": "Used to compute eigenmodes and eigenfrequencies of a structure.",
    "freq": "Used to compute the response of a linear model subjected to harmonic excitation at one or "
            "several frequencies (frequency sweep).",
}


def _tile(text, icon_name, sub=""):
    b = QToolButton()
    b.setObjectName("NewTile")
    b.setIcon(icons.icon(icon_name))
    b.setIconSize(QSize(64, 64))
    b.setText(text + (f"\n{sub}" if sub else ""))
    b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
    b.setMinimumSize(190, 150)
    f = b.font()
    f.setPointSize(11)
    b.setFont(f)
    b.setCursor(Qt.PointingHandCursor)
    return b


class NewDialog(QDialog):
    """COMSOL-like 'New' window: Model Wizard, Blank Model, or an example."""

    def __init__(self, examples: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("New")
        self.choice = None
        self.example = None
        self.resize(820, 520)
        v = QVBoxLayout(self)
        v.setContentsMargins(28, 22, 28, 18)
        t = QLabel("New")
        t.setObjectName("WizardTitle")
        v.addWidget(t)
        v.addSpacing(10)
        h = QHBoxLayout()
        h.setSpacing(18)
        w = _tile("Model Wizard", "wizard")
        b = _tile("Blank Model", "new")
        w.clicked.connect(lambda: self._pick("wizard"))
        b.clicked.connect(lambda: self._pick("blank"))
        h.addWidget(w)
        h.addWidget(b)
        h.addStretch(1)
        v.addLayout(h)
        v.addSpacing(14)
        lab = QLabel("Application Libraries (examples)")
        f = lab.font()
        f.setBold(True)
        lab.setFont(f)
        v.addWidget(lab)
        self.lst = QListWidget()
        self.lst.setIconSize(QSize(20, 20))
        for key, title in examples:
            it = QListWidgetItem(icons.icon("app"), title)
            it.setData(Qt.UserRole, key)
            self.lst.addItem(it)
        self.lst.itemDoubleClicked.connect(lambda it: self._pick("example", it.data(Qt.UserRole)))
        v.addWidget(self.lst, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        ob = QPushButton("Open Example")
        ob.clicked.connect(lambda: self.lst.currentItem() and self._pick("example", self.lst.currentItem().data(Qt.UserRole)))
        cb = QPushButton("Cancel")
        cb.clicked.connect(self.reject)
        row.addWidget(ob)
        row.addWidget(cb)
        v.addLayout(row)

    def _pick(self, what, ex=None):
        self.choice = what
        self.example = ex
        self.accept()


class ModelWizard(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Wizard")
        self.resize(900, 600)
        self.sdim = None
        self.added: list[str] = []
        self.study = None
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 16, 20, 14)
        self.stack = QStackedWidget()
        v.addWidget(self.stack, 1)
        self.stack.addWidget(self._page_dim())
        self.stack.addWidget(self._page_physics())
        self.stack.addWidget(self._page_study())

    # -- page 1
    def _page_dim(self):
        w = QWidget()
        v = QVBoxLayout(w)
        t = QLabel("Select Space Dimension")
        t.setObjectName("WizardTitle")
        v.addWidget(t)
        v.addSpacing(16)
        h = QHBoxLayout()
        h.setSpacing(18)
        for key, title, icon_name in (("3D", "3D", "component"), ("2Daxi", "2D Axisymmetric", "revolve"),
                                      ("2D", "2D", "rectangle")):
            b = _tile(title, icon_name)
            b.clicked.connect(lambda _=False, k=key: self._dim(k))
            h.addWidget(b)
        h.addStretch(1)
        v.addLayout(h)
        v.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        c = QPushButton("Cancel")
        c.clicked.connect(self.reject)
        row.addWidget(c)
        v.addLayout(row)
        return w

    def _dim(self, k):
        self.sdim = k
        self._fill_tree()
        self.stack.setCurrentIndex(1)

    # -- page 2
    def _page_physics(self):
        w = QWidget()
        v = QVBoxLayout(w)
        t = QLabel("Select Physics")
        t.setObjectName("WizardTitle")
        v.addWidget(t)
        h = QHBoxLayout()
        left = QVBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search")
        self.search.textChanged.connect(self._filter)
        left.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(QSize(16, 16))
        self.tree.itemDoubleClicked.connect(lambda it, c: self._add())
        self.tree.currentItemChanged.connect(self._describe)
        left.addWidget(self.tree, 1)
        brow = QHBoxLayout()
        add = QPushButton(icons.icon("add"), "Add")
        add.clicked.connect(self._add)
        brow.addWidget(add)
        brow.addStretch(1)
        left.addLayout(brow)
        h.addLayout(left, 3)
        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Added physics interfaces:</b>"))
        self.added_list = QListWidget()
        self.added_list.setIconSize(QSize(16, 16))
        right.addWidget(self.added_list, 1)
        rem = QPushButton(icons.icon("remove"), "Remove")
        rem.clicked.connect(self._remove)
        right.addWidget(rem, 0, Qt.AlignLeft)
        self.descr = QTextBrowser()
        self.descr.setMaximumHeight(110)
        right.addWidget(self.descr)
        h.addLayout(right, 2)
        v.addLayout(h, 1)
        row = QHBoxLayout()
        back = QPushButton("Back")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        row.addWidget(back)
        row.addStretch(1)
        self.btn_study = QPushButton(icons.icon("forward"), "Study")
        self.btn_study.clicked.connect(self._to_study)
        self.btn_study.setEnabled(False)
        done = QPushButton(icons.icon("check"), "Done")
        done.setToolTip("Finish without a study")
        done.clicked.connect(self._finish_no_study)
        c = QPushButton("Cancel")
        c.clicked.connect(self.reject)
        row.addWidget(self.btn_study)
        row.addWidget(done)
        row.addWidget(c)
        v.addLayout(row)
        return w

    def _fill_tree(self):
        self.tree.clear()
        for cat, subs in WIZARD_TREE:
            ci = QTreeWidgetItem([cat])
            ci.setIcon(0, icons.icon("folder"))
            any_child = False
            for sub, pids in subs:
                parent = ci
                if sub:
                    parent = QTreeWidgetItem([sub])
                    parent.setIcon(0, icons.icon("folder"))
                    ci.addChild(parent)
                for pid in pids:
                    if pid.startswith("@"):
                        title, icon_name, phys, _c = BUNDLES[pid]
                        if not all(self.sdim in PHYSICS[p].dims for p in phys):
                            continue
                        it = QTreeWidgetItem([title])
                        it.setIcon(0, icons.icon(icon_name))
                    else:
                        pd = PHYSICS[pid]
                        if self.sdim not in pd.dims:
                            continue
                        it = QTreeWidgetItem([f"{pd.name} ({pd.id})"])
                        it.setIcon(0, icons.icon(pd.icon))
                    it.setData(0, Qt.UserRole, pid)
                    parent.addChild(it)
                    any_child = True
                if sub and parent.childCount() == 0:
                    ci.removeChild(parent)
            if any_child:
                self.tree.addTopLevelItem(ci)
        self.tree.expandAll()

    def _filter(self, text):
        text = text.lower().strip()

        def visit(it):
            pid = it.data(0, Qt.UserRole)
            match = not text or text in it.text(0).lower()
            child_vis = False
            for i in range(it.childCount()):
                child_vis |= visit(it.child(i))
            vis = match if pid else (child_vis or (match and not text))
            it.setHidden(not (vis or child_vis))
            return vis or child_vis
        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i))

    def _describe(self, cur, prev):
        if cur is None:
            return
        pid = cur.data(0, Qt.UserRole)
        if not pid:
            self.descr.setHtml("")
        elif pid.startswith("@"):
            title, _i, phys, _c = BUNDLES[pid]
            self.descr.setHtml(f"<b>{title}</b><br>Adds {' and '.join(PHYSICS[p].name for p in phys)} with a "
                               f"multiphysics coupling.")
        else:
            pd = PHYSICS[pid]
            self.descr.setHtml(f"<b>{pd.name}</b><br>{pd.description}")

    def _add(self):
        it = self.tree.currentItem()
        pid = it.data(0, Qt.UserRole) if it else None
        if not pid:
            return
        if pid.startswith("@"):
            title, icon_name, phys, _c = BUNDLES[pid]
            label = title
        else:
            icon_name, label = PHYSICS[pid].icon, f"{PHYSICS[pid].name} ({pid})"
        self.added.append(pid)
        li = QListWidgetItem(icons.icon(icon_name), label)
        li.setData(Qt.UserRole, pid)
        self.added_list.addItem(li)
        self.btn_study.setEnabled(True)

    def _remove(self):
        row = self.added_list.currentRow()
        if row >= 0:
            self.added_list.takeItem(row)
            del self.added[row]
        self.btn_study.setEnabled(bool(self.added))

    def _finish_no_study(self):
        self.study = None
        self.accept()

    # -- page 3
    def _page_study(self):
        w = QWidget()
        v = QVBoxLayout(w)
        t = QLabel("Select Study")
        t.setObjectName("WizardTitle")
        v.addWidget(t)
        h = QHBoxLayout()
        self.stree = QTreeWidget()
        self.stree.setHeaderHidden(True)
        self.stree.setIconSize(QSize(16, 16))
        self.stree.currentItemChanged.connect(self._study_descr)
        self.stree.itemDoubleClicked.connect(lambda it, c: self._done())
        h.addWidget(self.stree, 3)
        self.sdescr = QTextBrowser()
        h.addWidget(self.sdescr, 2)
        v.addLayout(h, 1)
        row = QHBoxLayout()
        back = QPushButton("Back")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        row.addWidget(back)
        row.addStretch(1)
        done = QPushButton(icons.icon("check"), "Done")
        done.setDefault(True)
        done.clicked.connect(self._done)
        c = QPushButton("Cancel")
        c.clicked.connect(self.reject)
        row.addWidget(done)
        row.addWidget(c)
        v.addLayout(row)
        return w

    def _phys_ids(self):
        out = []
        for pid in self.added:
            out += BUNDLES[pid][2] if pid.startswith("@") else [pid]
        return out

    def _to_study(self):
        self.stree.clear()
        ids = self._phys_ids()
        common = set.intersection(*[set(PHYSICS[p].studies) for p in ids]) if ids else set()
        gen = QTreeWidgetItem(["General Studies"])
        gen.setIcon(0, icons.icon("folder"))
        preset = QTreeWidgetItem(["Preset Studies for Selected Physics Interfaces"])
        preset.setIcon(0, icons.icon("folder"))
        first = None
        for key in ("stationary", "time", "eigen", "freq"):
            title, icon_name = STUDY_TYPES[key]
            it = QTreeWidgetItem([title])
            it.setIcon(0, icons.icon(icon_name))
            it.setData(0, Qt.UserRole, key)
            if key in ("stationary", "time"):
                gen.addChild(it)
                if key not in common:
                    it.setDisabled(True)
            elif key in common:
                preset.addChild(it)
            if key in common and first is None:
                first = it
        self.stree.addTopLevelItem(preset)
        self.stree.addTopLevelItem(gen)
        if preset.childCount() == 0:
            self.stree.takeTopLevelItem(0)
        self.stree.expandAll()
        if first is not None:
            self.stree.setCurrentItem(first)
        self.stack.setCurrentIndex(2)

    def _study_descr(self, cur, prev):
        key = cur.data(0, Qt.UserRole) if cur else None
        self.sdescr.setHtml(f"<b>{STUDY_TYPES[key][0]}</b><br>{STUDY_DESCR[key]}" if key else "")

    def _done(self):
        it = self.stree.currentItem()
        key = it.data(0, Qt.UserRole) if it else None
        if not key or it.isDisabled():
            return
        self.study = key
        self.accept()
