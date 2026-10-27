"""Model Builder window: the model tree."""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPixmap, QPolygonF
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLineEdit, QToolButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from ..core.model import REGISTRY, Model, Node
from . import icons

TAG_SUFFIX_KINDS = ("component", "physics.", "material", "mp.")


def display_text(node: Node, file_name="Untitled.esm") -> str:
    if node.kind == "root":
        return f"{file_name} (root)"
    lbl = node.label or node.type.title
    if node.kind.startswith("geom.") or node.kind.startswith(TAG_SUFFIX_KINDS):
        return f"{lbl} ({node.tag})"
    return lbl


@lru_cache(maxsize=512)
def badge_icon(name: str, badge: str = "", disabled: bool = False, mark: str = "") -> QIcon:
    """The node's icon with its decorations (shapes, not only colors): a letter badge (D: default feature), a red
    stroke (disabled), and a *mark*: "error" (red cross, lower left), "warning" (orange triangle, lower left) or
    "stale" (asterisk, upper right: edited since it was built or computed)."""
    base = icons.icon(name)
    if not badge and not disabled and not mark:
        return base
    out = QIcon()
    for size in (16, 20, 24, 32):
        pm = base.pixmap(size, size, QIcon.Disabled if disabled else QIcon.Normal)
        canvas = QPixmap(pm.size())
        canvas.setDevicePixelRatio(pm.devicePixelRatio())
        canvas.fill(Qt.transparent)
        p = QPainter(canvas)
        p.setRenderHint(QPainter.Antialiasing)
        p.drawPixmap(0, 0, pm)
        if badge:
            s = size * 0.5
            r = QRectF(size - s, size - s, s, s)
            p.setBrush(QBrush(QColor("#4a5560")))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(r, 1.5, 1.5)
            f = QFont()
            f.setBold(True)
            f.setPixelSize(max(6, int(s * 0.8)))
            p.setFont(f)
            p.setPen(QColor("white"))
            p.drawText(r, Qt.AlignCenter, badge)
        if disabled:
            p.setPen(QColor("#c0392b"))
            p.drawLine(int(size * 0.15), int(size * 0.85), int(size * 0.85), int(size * 0.15))
        s = size * 0.55
        if mark == "error":
            r = QRectF(0, size - s, s, s)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#c0392b"))
            p.drawEllipse(r)
            pen = p.pen()
            pen.setColor(QColor("white"))
            pen.setWidthF(max(1.2, size / 12))
            pen.setStyle(Qt.SolidLine)
            p.setPen(pen)
            k = s * 0.3
            p.drawLine(QPointF(r.left() + k, r.top() + k), QPointF(r.right() - k, r.bottom() - k))
            p.drawLine(QPointF(r.left() + k, r.bottom() - k), QPointF(r.right() - k, r.top() + k))
        elif mark == "warning":
            p.setPen(QColor("#7a4500"))
            p.setBrush(QColor("#f0a020"))
            p.drawPolygon(QPolygonF([QPointF(s / 2, size - s), QPointF(0, size - 0.5), QPointF(s, size - 0.5)]))
        elif mark == "stale":
            f = QFont()
            f.setBold(True)
            f.setPixelSize(max(7, int(size * 0.7)))
            p.setFont(f)
            p.setPen(QColor("#a05a00"))
            p.drawText(QRectF(size * 0.5, -size * 0.12, size * 0.5, size * 0.6), Qt.AlignCenter, "*")
        p.end()
        out.addPixmap(canvas)
    return out


class ModelBuilder(QWidget):
    nodeSelected = Signal(object)
    contextRequested = Signal(object, QPoint)
    renameRequested = Signal(object, str)
    deleteRequested = Signal(object)
    activated = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model: Model | None = None
        self.file_name = "Untitled.esm"
        self._items: dict[int, QTreeWidgetItem] = {}
        self._nodes: dict[int, Node] = {}
        self._history: list[int] = []
        self._hist_pos = -1
        self._navigating = False
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QWidget(self)
        bar.setObjectName("PaneToolbar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(1)
        self.btn_back = self._tb(bar, "back", "Go Back", self.go_back)
        self.btn_fwd = self._tb(bar, "forward", "Go Forward", self.go_forward)
        h.addWidget(self.btn_back)
        h.addWidget(self.btn_fwd)
        h.addSpacing(6)
        h.addWidget(self._tb(bar, "up", "Move Up (Ctrl+Up)", lambda: self.moveRequested(-1)))
        h.addWidget(self._tb(bar, "down", "Move Down (Ctrl+Down)", lambda: self.moveRequested(1)))
        h.addStretch(1)
        h.addWidget(self._tb(bar, "expand_all", "Expand All", lambda: self.tree.expandAll()))
        h.addWidget(self._tb(bar, "collapse_all", "Collapse All", self.collapse_to_default))
        v.addWidget(bar)
        self.filter = QLineEdit(self)
        self.filter.setObjectName("TreeFilter")
        self.filter.setPlaceholderText("Filter (Ctrl+L)")
        self.filter.setAccessibleName("Model Builder filter")
        self.filter.setToolTip("Show only nodes whose label or name contains this text (and the branches leading to "
                               "them); Esc clears it")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self.apply_filter)
        self.filter.installEventFilter(self)
        v.addWidget(self.filter)
        self.tree = QTreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(QSize(16, 16))
        self.tree.setIndentation(16)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.setEditTriggers(QAbstractItemView.EditKeyPressed)
        self.tree.customContextMenuRequested.connect(self._ctx)
        self.tree.currentItemChanged.connect(self._sel)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemDoubleClicked.connect(lambda it, c: self.activated.emit(self.node_of(it)))
        self.tree.setAccessibleName("Model Builder tree")
        v.addWidget(self.tree, 1)
        self._suspend = False
        self.move_cb = None
        self.issues = None              # node -> [("error"|"warning"|"stale", explanation)], set by the window

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if obj is self.filter and ev.type() == QEvent.KeyPress and ev.key() == Qt.Key_Escape:
            self.filter.clear()
            self.tree.setFocus()
            return True
        return super().eventFilter(obj, ev)

    def focus_filter(self):
        self.filter.setFocus(Qt.ShortcutFocusReason)
        self.filter.selectAll()

    def apply_filter(self, text: str | None = None):
        """Hide nodes that do not match (case-insensitive, label or name) unless a descendant does; the current
        node and the expanded branches are kept."""
        text = (self.filter.text() if text is None else text).strip().lower()

        def visit(it) -> bool:
            n = self.node_of(it)
            hit = not text or (n is not None and (text in (n.label or "").lower() or text in n.tag.lower()
                                                  or text in it.text(0).lower()))
            child_hit = False
            for i in range(it.childCount()):
                child_hit = visit(it.child(i)) or child_hit
            show = hit or child_hit or it.parent() is None
            it.setHidden(not show)
            if text and child_hit:
                it.setExpanded(True)
            return hit or child_hit
        root = self.tree.topLevelItem(0)
        if root is not None:
            visit(root)
        cur = self.tree.currentItem()
        if cur is not None and not cur.isHidden():
            self.tree.scrollToItem(cur)

    def can_go(self, d: int) -> bool:
        return 0 <= self._hist_pos + d < len(self._history)

    def _tb(self, parent, icon_name, tip, cb):
        b = QToolButton(parent)
        b.setIcon(icons.icon(icon_name))
        b.setIconSize(QSize(16, 16))
        b.setToolTip(tip)
        b.setAccessibleName(tip.split(" (")[0])
        b.setAutoRaise(True)
        b.clicked.connect(cb)
        return b

    def moveRequested(self, d):
        if self.move_cb:
            self.move_cb(self.current(), d)

    # -- population
    def set_model(self, model: Model, file_name="Untitled.esm"):
        self.model = model
        self.file_name = file_name
        self._history.clear()
        self._hist_pos = -1
        self.refresh(select=model.root)
        self.collapse_to_default()

    def collapse_to_default(self):
        self.tree.collapseAll()
        root = self.tree.topLevelItem(0)
        if root is None:
            return
        root.setExpanded(True)
        for i in range(root.childCount()):
            it = root.child(i)
            n = self.node_of(it)
            if n is not None and n.kind in ("global", "component", "study", "results"):
                it.setExpanded(True)
                if n.kind == "component":
                    for j in range(it.childCount()):
                        c = self.node_of(it.child(j))
                        if c is not None and (c.kind.startswith("physics.") or c.kind in ("geometry", "materials", "mesh")):
                            it.child(j).setExpanded(True)

    def refresh(self, select: Node | None = None):
        if self.model is None:
            return
        expanded = {k for k, it in self._items.items() if it.isExpanded()}
        cur = select or self.current()
        self._suspend = True
        self.tree.clear()
        self._items.clear()
        self._nodes.clear()
        root_item = self._make_item(self.model.root, None)
        self.tree.addTopLevelItem(root_item)
        self._fill(root_item, self.model.root)
        for k, it in self._items.items():
            if k in expanded:
                it.setExpanded(True)
        root_item.setExpanded(True)
        self._suspend = False
        if self.filter.text().strip():
            self.apply_filter()
        if cur is not None and id(cur) in self._items:
            it = self._items[id(cur)]
            p = it.parent()
            while p is not None:
                p.setExpanded(True)
                p = p.parent()
            self.tree.setCurrentItem(it)

    def _fill(self, item, node):
        for c in node.children:
            ci = self._make_item(c, item)
            item.addChild(ci)
            self._fill(ci, c)

    def _make_item(self, node: Node, parent) -> QTreeWidgetItem:
        it = QTreeWidgetItem()
        it.setText(0, display_text(node, self.file_name))
        nt = REGISTRY.get(node.kind)
        it.setData(0, Qt.UserRole, id(node))
        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if nt and nt.renamable and node.kind != "root":
            flags |= Qt.ItemIsEditable
        it.setFlags(flags)
        self._decorate(it, node)
        self._items[id(node)] = it
        self._nodes[id(node)] = node
        return it

    def _decorate(self, it: QTreeWidgetItem, node: Node):
        """Icon marks, color and tooltip: what state the node is in and why (disabled, inactive, default, stale,
        invalid), in words and shapes as well as colors."""
        nt = REGISTRY.get(node.kind)
        icon_name = nt.icon if nt else "info"
        is_default = bool(node.meta.get("default"))
        issues = list(self.issues(node)) if self.issues is not None else []
        mark = next((lv for lv in ("error", "warning", "stale") if any(i[0] == lv for i in issues)), "")
        it.setIcon(0, badge_icon(icon_name, "D" if is_default else "", not node.enabled, mark))
        notes = []
        if not node.enabled:
            notes.append("Disabled: ignored when building and solving (F4 enables it).")
        elif not node.is_active():
            off = next((a for a in node.path()[1:] if not a.enabled), None)
            notes.append(f"Inactive: '{off.label}' above it is disabled." if off is not None else "Inactive.")
        if is_default:
            notes.append("Default feature: applies wherever no later feature overrides it; it cannot be deleted.")
        if node.meta.get("needs_update"):
            notes.append("Needs update: compute the study again to see the current settings.")
        notes += [text for _lv, text in issues]
        f = it.font(0)
        f.setItalic(not node.is_active())
        it.setFont(0, f)
        color = "#9aa0a6" if not node.is_active() else "#a05a00" if node.meta.get("needs_update") else None
        it.setForeground(0, QBrush(QColor(color)) if color else QBrush())
        tip = nt.description if nt and nt.description else (nt.title if nt else node.kind)
        it.setToolTip(0, tip + ("\n\n" + "\n".join(notes) if notes else ""))
        it.setData(0, Qt.AccessibleDescriptionRole, " ".join(notes))

    def update_marks(self):
        """Re-decorate every node (after an edit that changes no structure)."""
        self._suspend = True
        try:
            for k, it in self._items.items():
                n = self._nodes.get(k)
                if n is not None:
                    self._decorate(it, n)
        finally:
            self._suspend = False

    def node_of(self, item) -> Node | None:
        if item is None:
            return None
        return self._nodes.get(item.data(0, Qt.UserRole))

    def current(self) -> Node | None:
        return self.node_of(self.tree.currentItem())

    def select(self, node: Node):
        it = self._items.get(id(node))
        if it is not None:
            self.tree.setCurrentItem(it)
            self.tree.scrollToItem(it)

    def update_node(self, node: Node):
        it = self._items.get(id(node))
        if it is not None:
            self._suspend = True
            it.setText(0, display_text(node, self.file_name))
            self._suspend = False

    def edit_current(self):
        it = self.tree.currentItem()
        if it is not None and it.flags() & Qt.ItemIsEditable:
            node = self.node_of(it)
            self._suspend = True
            it.setText(0, node.label)
            self._suspend = False
            self.tree.editItem(it)

    # -- history
    def go_back(self):
        if self._hist_pos > 0:
            self._hist_pos -= 1
            self._goto_hist()

    def go_forward(self):
        if self._hist_pos < len(self._history) - 1:
            self._hist_pos += 1
            self._goto_hist()

    def _goto_hist(self):
        k = self._history[self._hist_pos]
        it = self._items.get(k)
        if it is not None:
            self._navigating = True
            self.tree.setCurrentItem(it)
            self._navigating = False

    # -- signals
    def _sel(self, cur, prev):
        if self._suspend:
            return
        n = self.node_of(cur)
        if n is not None:
            if not self._navigating:
                self._history = self._history[: self._hist_pos + 1] + [id(n)]
                self._hist_pos = len(self._history) - 1
            self.nodeSelected.emit(n)

    def _ctx(self, pos):
        it = self.tree.itemAt(pos)
        if it is None and self.tree.currentItem() is not None and self.tree.hasFocus():   # Shift+F10, Menu key
            it = self.tree.currentItem()
            pos = self.tree.visualItemRect(it).center()
        if it is not None:
            self.tree.setCurrentItem(it)
            self.contextRequested.emit(self.node_of(it), self.tree.viewport().mapToGlobal(pos))

    def _item_changed(self, item, col):
        if self._suspend:
            return
        n = self.node_of(item)
        if n is None:
            return
        text = item.text(0).strip()
        if text and text != display_text(n, self.file_name) and text != n.label:
            self.renameRequested.emit(n, text)
        else:
            self._suspend = True
            item.setText(0, display_text(n, self.file_name))
            self._suspend = False
