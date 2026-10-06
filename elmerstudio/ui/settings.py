"""Settings window: schema-driven forms for the selected model node."""
from __future__ import annotations

import io
import re
from functools import lru_cache

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
                               QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTableWidget,
                               QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from ..core import units
from ..core.materials import MATPROPS
from ..core.model import REGISTRY, Node, Prop, level_plural
from ..core.physics import PHYSICS, physics_def
from . import icons

FIELD_SYMBOLS = {"T", "x", "y", "z", "t", "r", "V", "u", "v", "w", "p", "c", "Az"}


def pretty_unit(u: str, length_unit="m") -> str:
    if u == "LEN":
        u = length_unit
    if not u or u == "1":
        return "1" if u == "1" else ""
    s = u.replace("um", "µm").replace("^-1", "⁻¹").replace("^2", "²").replace("^3", "³") \
        .replace("^4", "⁴").replace("*", "·").replace("deg", "°") if u != "degC" else "°C"
    return s


@lru_cache(maxsize=256)
def equation_pixmap(tex: str, color="#1d1f22") -> QPixmap | None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        from matplotlib import mathtext
        from matplotlib.font_manager import FontProperties
        buf = io.BytesIO()
        mathtext.math_to_image(tex, buf, prop=FontProperties(size=11), dpi=110, format="png", color=color)
        pm = QPixmap()
        pm.loadFromData(buf.getvalue(), "PNG")
        return pm
    except Exception:
        return None


# --------------------------------------------------------------------------- section widget
class Section(QWidget):
    def __init__(self, title: str, collapsed=False, on_toggle=None, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.btn = QToolButton(self)
        self.btn.setObjectName("SectionHeader")
        self.btn.setText(title)
        self.btn.setCheckable(True)
        self.btn.setChecked(not collapsed)
        self.btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self.btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn.toggled.connect(self._toggle)
        v.addWidget(self.btn)
        self.body = QWidget(self)
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(14, 6, 10, 8)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(5)
        self.grid.setColumnStretch(1, 1)
        v.addWidget(self.body)
        self.body.setVisible(not collapsed)
        self.title = title
        self.on_toggle = on_toggle
        self.row = 0

    def _toggle(self, on):
        self.btn.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(on)
        if self.on_toggle:
            self.on_toggle(self.title, not on)

    def add_row(self, label: str | None, widget: QWidget, unit: str = "", symbol: str = ""):
        r = self.row
        if label:
            lab = QLabel(label)
            lab.setWordWrap(True)
            self.grid.addWidget(lab, r, 0, 1, 3)
            r += 1
        if symbol:
            s = QLabel(symbol)
            s.setObjectName("SymbolLabel")
            s.setMinimumWidth(22)
            self.grid.addWidget(s, r, 0)
        self.grid.addWidget(widget, r, 1)
        if unit:
            u = QLabel(unit)
            u.setObjectName("UnitLabel")
            u.setMinimumWidth(34)
            self.grid.addWidget(u, r, 2)
        self.row = r + 1

    def add_inline(self, label: str, widget: QWidget, unit: str = ""):
        r = self.row
        lab = QLabel(label)
        self.grid.addWidget(lab, r, 0)
        self.grid.addWidget(widget, r, 1)
        if unit:
            u = QLabel(unit)
            u.setObjectName("UnitLabel")
            self.grid.addWidget(u, r, 2)
        self.row = r + 1

    def add_full(self, widget: QWidget):
        self.grid.addWidget(widget, self.row, 0, 1, 3)
        self.row += 1


# --------------------------------------------------------------------------- editors
class ExprEdit(QLineEdit):
    def __init__(self, text, validate, parent=None):
        super().__init__(str(text), parent)
        self.validate = validate
        self.textChanged.connect(self._check)
        self._check(self.text())

    def _check(self, t):
        ok, tip = self.validate(t)
        self.setProperty("error", not ok)
        self.setToolTip(tip)
        self.style().unpolish(self)
        self.style().polish(self)


class SettingsWindow(QWidget):
    actionRequested = Signal(str, object)

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.node: Node | None = None
        self.collapsed: dict = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.header = QWidget(self)
        self.header.setObjectName("SettingsBody")
        hv = QVBoxLayout(self.header)
        hv.setContentsMargins(8, 6, 8, 4)
        hv.setSpacing(4)
        top = QHBoxLayout()
        self.h_icon = QLabel()
        self.h_title = QLabel()
        self.h_title.setObjectName("SettingsTitle")
        top.addWidget(self.h_icon)
        top.addWidget(self.h_title, 1)
        hv.addLayout(top)
        self.h_buttons = QHBoxLayout()
        self.h_buttons.setSpacing(4)
        hv.addLayout(self.h_buttons)
        v.addWidget(self.header)
        line = QFrame(self)
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#d9dce1")
        v.addWidget(line)
        self.scroll = QScrollArea(self)
        self.scroll.setObjectName("SettingsArea")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        v.addWidget(self.scroll, 1)
        self.body = None
        self._sel_widgets = []

    # ------------------------------------------------------------------ public
    def show_node(self, node: Node | None, keep_scroll=False):
        pos = self.scroll.verticalScrollBar().value() if keep_scroll else 0
        self.node = node
        self._sel_widgets = []
        while self.h_buttons.count():
            it = self.h_buttons.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        body = QWidget()
        body.setObjectName("SettingsBody")
        self.form = QVBoxLayout(body)
        self.form.setContentsMargins(0, 0, 0, 10)
        self.form.setSpacing(0)
        if node is None:
            self.h_title.setText("")
            self.h_icon.clear()
            self.scroll.setWidget(body)
            return
        nt = REGISTRY.get(node.kind)
        self.h_icon.setPixmap(icons.pixmap(nt.icon if nt else "info", 20))
        self.h_title.setText(nt.title if nt else node.kind)
        for name, icon_name, tip in self.ctx.header_actions(node):
            b = QPushButton(icons.icon(icon_name), name)
            b.setIconSize(QSize(16, 16))
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, n=name: self.actionRequested.emit(n, self.node))
            self.h_buttons.addWidget(b)
        self.h_buttons.addStretch(1)
        self._build(node, nt)
        self.form.addStretch(1)
        self.scroll.setWidget(body)
        if keep_scroll:
            QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(pos))

    def refresh(self):
        self.show_node(self.node, keep_scroll=True)

    # ------------------------------------------------------------------ build
    def _section(self, title, default_collapsed=False):
        key = (self.node.kind, title)
        s = Section(title, self.collapsed.get(key, default_collapsed),
                    on_toggle=lambda t, c, k=self.node.kind: self.collapsed.__setitem__((k, t), c))
        self.form.addWidget(s)
        return s

    def _vis_props(self, node):
        d = dict(node.props)
        d["__sdim"] = self.ctx.sdim()
        d["__axi"] = self.ctx.is_axi()
        return d

    def _build(self, node: Node, nt):
        # label / name
        s = self._section("Label")
        le = QLineEdit(node.label)
        le.setReadOnly(node.kind == "root" or (nt is not None and not nt.renamable))
        le.editingFinished.connect(lambda: self.ctx.rename(node, le.text()))
        s.add_inline("Label:", le)
        if node.kind != "root":
            nm = QLineEdit(node.tag)
            nm.setReadOnly(True)
            s.add_inline("Name:", nm)
        if node.meta.get("note"):
            s.add_full(QLabel(f"<i>{node.meta['note']}</i>"))
        if nt is None:
            return
        if nt.description and (node.kind.startswith("physics.") or node.kind.startswith("mp.")):
            d = QLabel(nt.description)
            d.setWordWrap(True)
            d.setObjectName("SettingsSubtitle")
            s.add_full(d)
        # selection
        if node.selection is not None:
            self._selection_section(node, nt)
        # equation
        if nt.equation:
            es = self._section("Equation", default_collapsed=not node.kind.startswith("physics."))
            pm = equation_pixmap(nt.equation)
            lab = QLabel()
            if pm is not None:
                lab.setPixmap(pm)
            else:
                lab.setText(nt.equation)
            es.add_full(lab)
        # property sections in schema order
        vis = self._vis_props(node)
        sections: dict[str, list[Prop]] = {}
        for p in nt.props:
            if p.visible is not None:
                try:
                    if not p.visible(vis):
                        continue
                except Exception:
                    pass
            sections.setdefault(p.section, []).append(p)
        for title, props in sections.items():
            sec = self._section(title)
            for p in props:
                self._editor(sec, node, p)
        # extra custom sections
        extra = getattr(self.ctx, "extra_sections", None)
        if extra:
            for title, widget in extra(node):
                sec = self._section(title)
                sec.add_full(widget)

    # ------------------------------------------------------------------ selection section
    def _selection_section(self, node: Node, nt):
        sel = node.selection
        level = sel.level
        sec = self._section(f"{level.capitalize() if level != 'boundary' else 'Boundary'} Selection")
        geo = self.ctx.geometry
        avail = geo.numbers.get(level, []) if geo is not None else []
        row = QHBoxLayout()
        lab = QLabel("Selection:")
        combo = QComboBox()
        combo.addItems(["Manual", f"All {level_plural(level)}"])
        combo.setCurrentIndex(1 if sel.all else 0)
        locked = nt.selection_locked or node.meta.get("default")
        combo.setEnabled(not locked)
        row.addWidget(lab)
        row.addWidget(combo, 1)
        w = QWidget()
        w.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        sec.add_full(w)
        body = QHBoxLayout()
        lst = QListWidget()
        lst.setMaximumHeight(110)
        lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        ents = sel.resolve(avail) if geo is not None else (sel.entities if not sel.all else [])
        overridden = self.ctx.overridden_entities(node) if hasattr(self.ctx, "overridden_entities") else set()
        for e in ents:
            txt = str(e)
            if e in overridden:
                txt += " (overridden)"
            it = QListWidgetItem(txt)
            it.setData(Qt.UserRole, e)
            if e in overridden:
                it.setForeground(QColor("#9aa0a6"))
            lst.addItem(it)
        if geo is None:
            lst.addItem(QListWidgetItem("(build the geometry to select entities)"))
        body.addWidget(lst, 1)
        btns = QVBoxLayout()
        btns.setSpacing(2)
        active = QToolButton()
        active.setIcon(icons.icon("active"))
        active.setCheckable(True)
        active.setChecked(not locked and not sel.all)
        active.setToolTip("Activate selection: click entities in the Graphics window to add/remove them")
        active.setEnabled(not locked and not sel.all)
        add = QToolButton()
        add.setIcon(icons.icon("add"))
        add.setToolTip("Add entities selected in the Graphics window")
        rem = QToolButton()
        rem.setIcon(icons.icon("remove"))
        rem.setToolTip("Remove selected entities from the list")
        clr = QToolButton()
        clr.setIcon(icons.icon("clear"))
        clr.setToolTip("Clear selection")
        for b in (active, add, rem, clr):
            b.setAutoRaise(True)
            b.setIconSize(QSize(16, 16))
            b.setEnabled(b.isEnabled() and not locked and not sel.all)
            btns.addWidget(b)
        btns.addStretch(1)
        body.addLayout(btns)
        w2 = QWidget()
        w2.setLayout(body)
        body.setContentsMargins(0, 0, 0, 0)
        sec.add_full(w2)
        if locked:
            note = QLabel("<i>Default feature: applies to all entities not overridden by other features.</i>")
            note.setWordWrap(True)
            sec.add_full(note)

        def on_combo(i):
            self.ctx.set_selection(node, all_=(i == 1))
            self.refresh()
        combo.currentIndexChanged.connect(on_combo)
        active.toggled.connect(lambda on: self.ctx.activate_selection(node if on else None))
        add.clicked.connect(lambda: self.ctx.add_graphics_selection(node))
        rem.clicked.connect(lambda: self.ctx.remove_entities(node, [i.data(Qt.UserRole) for i in lst.selectedItems()
                                                                     if i.data(Qt.UserRole) is not None]))
        clr.clicked.connect(lambda: self.ctx.set_selection(node, entities=[]))
        lst.itemSelectionChanged.connect(lambda: self.ctx.highlight_entities(
            level, [i.data(Qt.UserRole) for i in lst.selectedItems() if i.data(Qt.UserRole) is not None]))
        self._sel_widgets.append((node, lst, active))
        if not locked and not sel.all:
            QTimer.singleShot(0, lambda: self.ctx.activate_selection(node))
        else:
            QTimer.singleShot(0, lambda: self.ctx.activate_selection(None, show=node))

    # ------------------------------------------------------------------ editors
    def _validator(self, node, p):
        def check(text):
            t = str(text).strip()
            if p.kind != "expr":
                return True, ""
            if t == "":
                return False, "Empty expression"
            names = dict(self.ctx.parameters())
            for s in FIELD_SYMBOLS | set(self.ctx.variable_names()):
                names.setdefault(s, 1.0)
            try:
                units.evaluate(t, names)
                try:
                    pure = complex(units.evaluate(t, dict(self.ctx.parameters()))).real
                    unit = "m" if p.unit == "LEN" else pretty_unit(p.unit)
                    return True, f"{t}\n= {pure:.6g} {unit} (SI)"
                except Exception:
                    return True, f"{t}\n(depends on field variables)"
            except Exception as exc:
                fn = re.findall(r"(\w+)\(", t)
                if fn and all(f in self.ctx.function_names() or f in units.MATH_FUNCS for f in fn):
                    return True, t
                return False, str(exc)
        return check

    def _editor(self, sec: Section, node: Node, p: Prop):
        k = p.kind
        val = node.props.get(p.key, p.default)
        lu = self.ctx.length_unit()
        unit = pretty_unit(p.unit, lu)
        commit = lambda v, key=p.key, rebuild=False: self.ctx.set_prop(node, key, v, rebuild)  # noqa: E731
        if k == "expr":
            e = ExprEdit(val, self._validator(node, p))
            e.editingFinished.connect(lambda: commit(e.text()))
            sec.add_row(p.label, e, unit, p.symbol)
        elif k in ("string",):
            e = QLineEdit(str(val))
            e.setReadOnly(p.readonly)
            if p.tip:
                e.setToolTip(p.tip)
            e.editingFinished.connect(lambda: commit(e.text()))
            sec.add_row(p.label, e, unit, p.symbol)
        elif k == "int":
            sb = QSpinBox()
            sb.setRange(0, 10_000_000)
            try:
                sb.setValue(int(val))
            except (TypeError, ValueError):
                sb.setValue(0)
            if p.tip:
                sb.setToolTip(p.tip)
            sb.editingFinished.connect(lambda: commit(sb.value()))
            sec.add_row(p.label, sb, unit)
        elif k == "bool":
            cb = QCheckBox(p.label)
            cb.setChecked(bool(val))
            cb.toggled.connect(lambda on: commit(on, rebuild=True))
            sec.add_full(cb)
        elif k == "choice":
            cb = QComboBox()
            for value, text in p.choices:
                cb.addItem(text, value)
            idx = max(0, cb.findData(val))
            cb.setCurrentIndex(idx)
            cb.currentIndexChanged.connect(lambda i: commit(cb.itemData(i), rebuild=True))
            sec.add_row(p.label, cb, unit, p.symbol)
        elif k == "text":
            te = QPlainTextEdit(str(val))
            te.setMaximumHeight(90)
            if p.tip:
                te.setToolTip(p.tip)
            te.focusOutEvent = (lambda ev, te=te, f=te.focusOutEvent: (commit(te.toPlainText()), f(ev)))
            sec.add_row(p.label, te)
        elif k in ("file", "savefile"):
            row = QHBoxLayout()
            e = QLineEdit(str(val))
            b = QPushButton("Browse…")

            def browse(_=False, e=e, save=(k == "savefile")):
                if save:
                    fn, _f = QFileDialog.getSaveFileName(self, p.label, e.text())
                else:
                    fn, _f = QFileDialog.getOpenFileName(self, p.label, e.text(),
                                                         "CAD files (*.step *.stp *.iges *.igs *.brep);;All files (*)")
                if fn:
                    e.setText(fn)
                    commit(fn)
            b.clicked.connect(browse)
            e.editingFinished.connect(lambda: commit(e.text()))
            row.addWidget(e, 1)
            row.addWidget(b)
            w = QWidget()
            row.setContentsMargins(0, 0, 0, 0)
            w.setLayout(row)
            sec.add_row(p.label, w)
        elif k == "color":
            b = QPushButton()
            rgb = val if isinstance(val, (list, tuple)) and len(val) == 3 else [190, 190, 190]
            b.setStyleSheet(f"background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); min-width: 60px;")

            def pick(_=False):
                c = QColorDialog.getColor(QColor(*rgb), self)
                if c.isValid():
                    commit([c.red(), c.green(), c.blue()], rebuild=True)
            b.clicked.connect(pick)
            sec.add_inline(p.label, b)
        elif k == "info":
            if isinstance(val, str) and val:
                sec.add_inline(p.label, QLabel(val))
        elif k == "table":
            sec.add_full(self._table(node, p))
        elif k == "objects":
            sec.add_row(p.label, self._objects(node, p))
        elif k == "matprops":
            sec.add_full(self._matprops(node))
        elif k == "physics_table":
            sec.add_full(self._physics_table(node))
        elif k == "study_ref":
            cb = QComboBox()
            for st in self.ctx.model.studies():
                cb.addItem(st.label, st.tag)
            cb.setCurrentIndex(max(0, cb.findData(val)))
            cb.currentIndexChanged.connect(lambda i: commit(cb.itemData(i), rebuild=True))
            sec.add_row(p.label, cb)
        elif k == "dataset":
            cb = QComboBox()
            if val == "fromparent" or p.default == "fromparent":
                cb.addItem("From parent", "fromparent")
            for d in self.ctx.datasets():
                cb.addItem(icons.icon("solution"), d.label, d.tag)
            if cb.count() == 0:
                cb.addItem("None", "")
            cb.setCurrentIndex(max(0, cb.findData(val)))
            cb.currentIndexChanged.connect(lambda i: commit(cb.itemData(i), rebuild=True))
            sec.add_row(p.label, cb)
        elif k == "solnum":
            cb = QComboBox()
            labels = self.ctx.frame_labels(node)
            cb.addItem("Last", "last")
            for i, lbl in enumerate(labels):
                cb.addItem(lbl or f"Solution {i + 1}", str(i))
            cb.setCurrentIndex(max(0, cb.findData(str(val))))
            cb.currentIndexChanged.connect(lambda i: commit(cb.itemData(i), rebuild=False))
            sec.add_row(self.ctx.frame_kind_label(node) or p.label, cb)
        elif k in ("rexpr", "rvexpr"):
            row = QHBoxLayout()
            e = QLineEdit(str(val))
            btn = QToolButton()
            btn.setIcon(icons.icon("variables"))
            btn.setToolTip("Replace Expression")
            btn.setPopupMode(QToolButton.InstantPopup)
            menu = QMenu(btn)
            vars_ = self.ctx.result_variables(node, vector=(k == "rvexpr"))
            groups: dict = {}
            for alias, label, u, kind in vars_:
                grp = alias.split(".")[0] if "." in alias else "General"
                groups.setdefault(grp, []).append((alias, label, u))
            names_ = {pd.id: pd.name for pd in PHYSICS.values()}
            for grp, items in groups.items():
                sub = menu.addMenu(icons.icon("physics_list"), f"{names_.get(grp, grp)}" if grp != "General" else "General")
                for alias, label, u in items:
                    a = sub.addAction(f"{label} - {alias}" + (f" ({pretty_unit(u)})" if u else ""))
                    a.triggered.connect(lambda _=False, al=alias, e=e: (e.setText(al), commit(al, rebuild=True)))
            if not vars_:
                menu.addAction("(compute the study to list variables)").setEnabled(False)
            btn.setMenu(menu)
            e.editingFinished.connect(lambda: commit(e.text(), rebuild=True))
            row.addWidget(e, 1)
            row.addWidget(btn)
            w = QWidget()
            row.setContentsMargins(0, 0, 0, 0)
            w.setLayout(row)
            sec.add_row(p.label, w)
        elif k == "runit":
            cb = QComboBox()
            cb.setEditable(True)
            for u in self.ctx.result_units(node):
                cb.addItem(pretty_unit(u), u)
            i = cb.findData(val)
            cb.setCurrentIndex(i if i >= 0 else 0)
            cb.currentIndexChanged.connect(lambda i: commit(cb.itemData(i) or cb.currentText(), rebuild=False))
            sec.add_row(p.label, cb)
        elif k == "plotgroup":
            cb = QComboBox()
            for pg in self.ctx.plot_groups():
                cb.addItem(pg.label, pg.tag)
            cb.setCurrentIndex(max(0, cb.findData(val)))
            cb.currentIndexChanged.connect(lambda i: commit(cb.itemData(i)))
            sec.add_row(p.label, cb)
        elif k == "expr_table":
            sec.add_full(self._table(node, p, result=True))

    # ------------------------------------------------------------------ table editors
    def _table(self, node: Node, p: Prop, result=False):
        cols = p.columns
        rows = list(node.props.get(p.key) or [])
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        t = QTableWidget(len(rows) + 1, len(cols))
        t.setHorizontalHeaderLabels([c[1] for c in cols])
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        t.horizontalHeader().setStretchLastSection(True)
        t.setMinimumHeight(140)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        is_params = node.kind == "params"
        values = {}
        if is_params:
            values, errs = units.evaluate_parameters([(r.get("name", ""), r.get("expr", "")) for r in rows if r.get("name")])
        for i, r in enumerate(rows):
            for j, (key, _title) in enumerate(cols):
                if key == "value" and is_params:
                    nm = r.get("name", "")
                    txt = f"{values[nm]:.6g}" if nm in values else "(error)"
                    unit = _unit_of(r.get("expr", ""))
                    it = QTableWidgetItem(txt + (f" {unit}" if unit and nm in values else ""))
                    it.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                    it.setForeground(QColor("#555") if nm in values else QColor("#c0392b"))
                else:
                    it = QTableWidgetItem(str(r.get(key, "")))
                t.setItem(i, j, it)
        t.resizeColumnsToContents()
        for j in range(len(cols)):
            t.setColumnWidth(j, max(t.columnWidth(j), 80))

        def commit_table(*_):
            out = []
            for i in range(t.rowCount()):
                rec = {}
                empty = True
                for j, (key, _title) in enumerate(cols):
                    if key == "value" and is_params:
                        continue
                    it = t.item(i, j)
                    txt = it.text().strip() if it else ""
                    rec[key] = txt
                    if txt:
                        empty = False
                if not empty:
                    out.append(rec)
            if out != rows:
                self.ctx.set_prop(node, p.key, out, rebuild=True)
        t.itemChanged.connect(lambda it: QTimer.singleShot(0, commit_table) if it.column() != (
            [c[0] for c in cols].index("value") if is_params and "value" in [c[0] for c in cols] else -1) else None)
        v.addWidget(t)
        bar = QHBoxLayout()
        for icon_name, tip, fn in (("up", "Move Up", lambda: _move(-1)), ("down", "Move Down", lambda: _move(1)),
                                   ("delete", "Delete", lambda: _delete()), ("clear", "Clear Table", lambda: _clear())):
            b = QToolButton()
            b.setIcon(icons.icon(icon_name))
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.clicked.connect(fn)
            bar.addWidget(b)
        if is_params or node.kind == "variables":
            b = QToolButton()
            b.setIcon(icons.icon("open"))
            b.setToolTip("Load from File")
            b.setAutoRaise(True)
            b.clicked.connect(lambda: self._load_table(node, p))
            bar.addWidget(b)
            b2 = QToolButton()
            b2.setIcon(icons.icon("save"))
            b2.setToolTip("Save to File")
            b2.setAutoRaise(True)
            b2.clicked.connect(lambda: self._save_table(node, p))
            bar.addWidget(b2)
        bar.addStretch(1)
        v.addLayout(bar)

        def _move(d):
            i = t.currentRow()
            j = i + d
            if 0 <= i < len(rows) and 0 <= j < len(rows):
                rows[i], rows[j] = rows[j], rows[i]
                self.ctx.set_prop(node, p.key, list(rows), rebuild=True)

        def _delete():
            i = t.currentRow()
            if 0 <= i < len(rows):
                del rows[i]
                self.ctx.set_prop(node, p.key, list(rows), rebuild=True)

        def _clear():
            self.ctx.set_prop(node, p.key, [], rebuild=True)
        return w

    def _load_table(self, node, p):
        fn, _ = QFileDialog.getOpenFileName(self, "Load from File", "", "Text files (*.txt *.csv);;All files (*)")
        if not fn:
            return
        rows = []
        with open(fn, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "%")):
                    continue
                parts = re.split(r"\s+|\t|,", line, maxsplit=2)
                if len(parts) >= 2:
                    rows.append({"name": parts[0], "expr": parts[1], "descr": parts[2] if len(parts) > 2 else ""})
        self.ctx.set_prop(node, p.key, rows, rebuild=True)

    def _save_table(self, node, p):
        fn, _ = QFileDialog.getSaveFileName(self, "Save to File", "parameters.txt", "Text files (*.txt)")
        if not fn:
            return
        with open(fn, "w", encoding="utf-8") as f:
            for r in node.props.get(p.key, []):
                f.write(f"{r.get('name', '')} {r.get('expr', '')} \"{r.get('descr', '')}\"\n")

    def _objects(self, node: Node, p: Prop):
        names = self.ctx.object_names_before(node)
        chosen = list(node.props.get(p.key) or [])
        lst = QListWidget()
        lst.setMaximumHeight(110)
        for nm in names + [c for c in chosen if c not in names]:
            it = QListWidgetItem(nm)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if nm in chosen else Qt.Unchecked)
            if nm not in names:
                it.setForeground(QColor("#c0392b"))
                it.setToolTip("Object no longer exists at this point of the sequence")
            lst.addItem(it)

        def changed(_it):
            out = [lst.item(i).text() for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked]
            self.ctx.set_prop(node, p.key, out, rebuild=False)
        lst.itemChanged.connect(changed)
        if not names:
            lst.addItem(QListWidgetItem("(no objects before this feature)"))
        return lst

    def _matprops(self, node: Node):
        req = self.ctx.required_matprops(node)
        vals = dict(node.props.get("values") or {})
        keys = [k for k in MATPROPS if k in req] + [k for k in MATPROPS if k not in req]
        t = QTableWidget(len(keys), 6)
        t.setHorizontalHeaderLabels(["", "Property", "Variable", "Value", "Unit", "Property group"])
        t.verticalHeader().setVisible(False)
        t.setMinimumHeight(min(340, 26 * len(keys) + 30))
        ok_icon, bad_icon = icons.icon("check"), icons.icon("cross")
        for i, k in enumerate(keys):
            label, sym, unit, _kw, grp = MATPROPS[k]
            st = QTableWidgetItem()
            if k in req:
                st.setIcon(ok_icon if str(vals.get(k, "")).strip() else bad_icon)
                st.setToolTip("Required by a physics interface" + ("" if str(vals.get(k, "")).strip() else " - missing!"))
            st.setFlags(Qt.ItemIsEnabled)
            t.setItem(i, 0, st)
            for j, txt in ((1, label), (2, sym), (4, pretty_unit(unit)), (5, grp)):
                it = QTableWidgetItem(txt)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if k not in req and not str(vals.get(k, "")).strip():
                    it.setForeground(QColor("#9aa0a6"))
                t.setItem(i, j, it)
            vi = QTableWidgetItem(str(vals.get(k, "")))
            vi.setData(Qt.UserRole, k)
            t.setItem(i, 3, vi)
        t.resizeColumnsToContents()
        t.setColumnWidth(0, 24)
        t.horizontalHeader().setStretchLastSection(True)

        def changed(it):
            if it.column() != 3:
                return
            k = it.data(Qt.UserRole)
            new = dict(node.props.get("values") or {})
            txt = it.text().strip()
            if txt:
                new[k] = txt
            else:
                new.pop(k, None)
            if new != node.props.get("values"):
                QTimer.singleShot(0, lambda: self.ctx.set_prop(node, "values", new, rebuild=True))
        t.itemChanged.connect(changed)
        return t

    def _physics_table(self, node: Node):
        phys = self.ctx.physics_nodes()
        sel = dict(node.props.get("physics") or {})
        t = QTableWidget(len(phys), 3)
        t.setHorizontalHeaderLabels(["Physics interface", "Solve for", "Equation form"])
        t.verticalHeader().setVisible(False)
        t.setMinimumHeight(min(220, 28 * len(phys) + 30))
        for i, ph in enumerate(phys):
            it = QTableWidgetItem(icons.icon(REGISTRY[ph.kind].icon), f"{ph.label} ({ph.tag})")
            it.setFlags(Qt.ItemIsEnabled)
            t.setItem(i, 0, it)
            cb = QTableWidgetItem()
            cb.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            cb.setCheckState(Qt.Checked if sel.get(ph.tag, True) else Qt.Unchecked)
            cb.setData(Qt.UserRole, ph.tag)
            t.setItem(i, 1, cb)
            ef = QTableWidgetItem("Automatic (" + self.ctx.step_kind_name(node) + ")")
            ef.setFlags(Qt.ItemIsEnabled)
            t.setItem(i, 2, ef)
        t.resizeColumnsToContents()
        t.horizontalHeader().setStretchLastSection(True)

        def changed(it):
            if it.column() == 1:
                new = dict(node.props.get("physics") or {})
                new[it.data(Qt.UserRole)] = it.checkState() == Qt.Checked
                self.ctx.set_prop(node, "physics", new)
        t.itemChanged.connect(changed)
        return t


def _unit_of(expr: str) -> str:
    m = re.search(r"\[([^\]]+)\]\s*$", str(expr))
    return pretty_unit(m.group(1)) if m else ""
