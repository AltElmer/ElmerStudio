"""Messages, Progress, Log, Table and Convergence Plot windows; 1D plot canvas."""
from __future__ import annotations

import csv
import datetime as _dt

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit,
                               QProgressBar, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from . import icons


def _toolbar(parent, buttons):
    bar = QWidget(parent)
    bar.setObjectName("PaneToolbar")
    h = QHBoxLayout(bar)
    h.setContentsMargins(4, 1, 4, 1)
    h.setSpacing(1)
    for icon_name, tip, cb in buttons:
        b = QToolButton(bar)
        b.setIcon(icons.icon(icon_name))
        b.setIconSize(QSize(16, 16))
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.clicked.connect(cb)
        h.addWidget(b)
    h.addStretch(1)
    return bar, h


class MessagesPane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar, _ = _toolbar(self, [("clear", "Clear", self.clear), ("copy", "Copy All", self.copy_all)])
        v.addWidget(bar)
        self.view = QPlainTextEdit(self)
        self.view.setObjectName("MessagesPlain")
        self.view.setReadOnly(True)
        v.addWidget(self.view, 1)

    def add(self, text: str, kind="info"):
        stamp = _dt.datetime.now().strftime("[%b %d, %Y, %I:%M %p]").replace(" 0", " ")
        color = {"error": "#c0392b", "warning": "#b36b00"}.get(kind)
        line = f"{stamp} {text}"
        if color:
            self.view.appendHtml(f'<span style="color:{color}">{_esc(line)}</span>')
        else:
            self.view.appendHtml(_esc(line))
        self.view.moveCursor(QTextCursor.End)

    def clear(self):
        self.view.clear()

    def copy_all(self):
        QApplication.clipboard().setText(self.view.toPlainText())


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class LogPane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar, _ = _toolbar(self, [("clear", "Clear", lambda: self.view.clear()),
                                 ("copy", "Copy All", lambda: QApplication.clipboard().setText(self.view.toPlainText()))])
        v.addWidget(bar)
        self.view = QPlainTextEdit(self)
        self.view.setObjectName("MessagesView")
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(20000)
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        f = QFont("Consolas")
        f.setStyleHint(QFont.Monospace)
        f.setPointSize(8)
        self.view.setFont(f)
        v.addWidget(self.view, 1)
        self._buf: list[str] = []

    def add(self, line):
        self._buf.append(line)

    def flush(self):
        if self._buf:
            self.view.appendPlainText("\n".join(self._buf))
            self._buf.clear()
            self.view.moveCursor(QTextCursor.End)


class ProgressPane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Description", "Progress", "Convergence"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(2, 120)
        v.addWidget(self.table)

    def start(self, description):
        self.table.setRowCount(0)
        self._rows: dict[str, int] = {}
        self.set(description, 0.0)

    def set(self, description, frac=None, conv=None):
        """One row per description; frac None keeps the bar, conv None keeps the convergence text."""
        rows = getattr(self, "_rows", {})
        self._rows = rows
        if description not in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(description))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            self.table.setCellWidget(r, 1, bar)
            self.table.setItem(r, 2, QTableWidgetItem(""))
            rows[description] = r
        r = rows[description]
        if frac is not None:
            self.table.cellWidget(r, 1).setValue(int(round(100 * max(0.0, min(1.0, frac)))))
        if conv is not None:
            self.table.item(r, 2).setText(conv)

    def finish_all(self):
        for r in range(self.table.rowCount()):
            self.table.cellWidget(r, 1).setValue(100)


class TablePane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar, h = _toolbar(self, [("copy", "Copy Table to Clipboard", self.copy), ("data_export", "Export (CSV)", self.export),
                                 ("clear", "Clear", self.clear)])
        self.title = QLabel("")
        self.title.setStyleSheet("color:#555; padding-left:8px;")
        h.insertWidget(3, self.title)
        v.addWidget(bar)
        self.table = QTableWidget(0, 0, self)
        self.table.verticalHeader().setVisible(False)
        v.addWidget(self.table, 1)

    def show_table(self, title, cols, rows):
        self.title.setText(title)
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(cols)
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                txt = f"{v:.6g}" if isinstance(v, float) else str(v)
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()

    def copy(self):
        out = ["\t".join(self.table.horizontalHeaderItem(j).text() for j in range(self.table.columnCount()))]
        for i in range(self.table.rowCount()):
            out.append("\t".join(self.table.item(i, j).text() if self.table.item(i, j) else ""
                                 for j in range(self.table.columnCount())))
        QApplication.clipboard().setText("\n".join(out))

    def export(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Export Table", "table.csv", "CSV (*.csv)")
        if not fn:
            return
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([self.table.horizontalHeaderItem(j).text() for j in range(self.table.columnCount())])
            for i in range(self.table.rowCount()):
                w.writerow([self.table.item(i, j).text() if self.table.item(i, j) else "" for j in range(self.table.columnCount())])

    def clear(self):
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.title.setText("")


class MplCanvas(QWidget):
    """A matplotlib figure in a Qt widget (1D plot groups, convergence plots)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.fig)
        v.addWidget(self.canvas)

    def draw(self):
        self.canvas.draw_idle()


class ConvergencePlot(MplCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.series: dict[str, list] = {}
        self.reset()

    def reset(self):
        self.series = {}
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self._style()
        self.draw()

    def _style(self):
        ax = self.ax
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Relative change")
        ax.set_title("Convergence Plot", fontsize=10)
        ax.grid(True, which="both", color="#e1e4e8", lw=0.7)
        for sp in ax.spines.values():
            sp.set_color("#8a9099")

    def add(self, point):
        idx, eq, kind, relc, it = point
        key = f"{eq} ({'nonlinear' if kind == 'NS' else 'coupling'})"
        self.series.setdefault(key, []).append((idx, max(relc, 1e-16)))

    def redraw(self):
        self.ax.clear()
        self._style()
        colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
        for i, (k, pts) in enumerate(self.series.items()):
            x, y = zip(*pts)
            self.ax.plot(x, y, "-o", ms=3, lw=1.3, color=colors[i % len(colors)], label=k)
        if self.series:
            self.ax.legend(fontsize=7, loc="upper right")
        self.fig.tight_layout()
        self.draw()
