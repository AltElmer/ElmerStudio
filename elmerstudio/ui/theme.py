"""Light engineering-desktop theme (Fusion based, identical on Windows/macOS/Linux).

Design tokens: colors, text roles and control metrics live here; widgets ask for a role (``pt("caption")``,
``metrics().small_button``) instead of hard-coding sizes, so one base font size (Preferences > Font size) scales the
whole desktop and the plots alike, and Qt's display scaling (100-200 %) applies on top of it.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QFont, QFontMetrics, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

# ---------------------------------------------------------------- color tokens
ACCENT = "#2b6cc4"
ACCENT_LIGHT = "#d4e6fa"
SEL_BORDER = "#7fb3ea"
FILE_TAB = "#2559a4"
BORDER = "#c9cdd2"
PANEL = "#ffffff"
WINDOW = "#eef0f2"
TEXT = "#1d1f22"
MUTED = "#6b7178"
HEADER_BG = "#eceef1"
DISABLED_TEXT = "#9aa0a6"
ERROR = "#c0392b"
ERROR_BG = "#fff1f0"
WARNING = "#a05a00"
STALE = "#a05a00"
OK = "#2e7d32"

# ---------------------------------------------------------------- text roles (points relative to the base font)
BASE_PT = 9.0
ROLES = {
    "ui": 0.0,              # labels, tree, fields, menus
    "caption": -1.0,        # ribbon group titles, status hints
    "mono": -1.0,           # log, Elmer input file
    "heading": 2.0,         # Settings window title
    "tile": 2.0,            # New window tiles
    "display": 5.0,         # wizard titles
    # plots (VTK and matplotlib): the same scale, so a larger desktop font gives larger legends too
    "plot_axis": 0.0,
    "plot_legend": 1.0,
    "plot_small": -1.0,
    "plot_annotation": 2.0,
    "plot_title": 3.0,
}
MIN_PT = 6.0
_base = BASE_PT


def pt(role: str = "ui") -> float:
    """Point size of a text role at the current base font size."""
    return max(MIN_PT, _base + ROLES.get(role, 0.0) * _base / BASE_PT)


def plot_pt(role: str) -> int:
    return int(round(pt(role)))


@dataclass
class Metrics:
    """Control sizes derived from the font (logical pixels), so text never clips at larger fonts."""
    line: int               # one line of ui text
    small_icon: int
    large_icon: int
    small_button: int       # height of a small ribbon button (icon or one text line)
    large_button_w: int     # minimum width of a large ribbon button
    ribbon_panel: int       # height of the ribbon's command panel (three small rows + the group title)


def metrics() -> Metrics:
    f = QFont(QApplication.font()) if QApplication.instance() else QFont()
    f.setPointSizeF(pt("ui"))
    line = QFontMetrics(f).height()
    cf = QFont(f)
    cf.setPointSizeF(pt("caption"))
    cap = QFontMetrics(cf).height()
    small_icon = 16 if line <= 20 else 20
    large_icon = 32 if line <= 20 else 40
    small = max(small_icon, line) + 6
    # three small rows or a large icon over two text lines (plus the button's frame and padding, 12 px measured
    # with the Fusion style and this style sheet), the group title and the margins
    large = large_icon + 2 * line + 12
    panel = max(3 * small, large) + cap + 8
    return Metrics(line, small_icon, large_icon, small, max(46, 4 * line), panel)


def qss() -> str:
    return f"""
QMainWindow, QDialog {{ background: {WINDOW}; }}
QWidget {{ color: {TEXT}; }}
QToolTip {{ background: #ffffe8; color: #222; border: 1px solid #a8a8a8; padding: 3px; }}

/* ---------- ribbon ---------- */
#RibbonBar {{ background: {WINDOW}; border-bottom: 1px solid {BORDER}; }}
#QuickAccess {{ background: {WINDOW}; }}
#QuickAccess QToolButton {{ border: 1px solid transparent; border-radius: 2px; padding: 1px; }}
#QuickAccess QToolButton:hover {{ background: #dfe7f1; border-color: #b7c9de; }}
#RibbonTabs QTabBar::tab {{ background: transparent; border: 1px solid transparent; border-bottom: none;
    padding: 4px 13px 5px 13px; margin-right: 1px; color: #2c2f33; }}
#RibbonTabs QTabBar::tab:hover {{ background: #e3e8ee; }}
#RibbonTabs QTabBar::tab:selected {{ background: #fbfbfc; border-color: {BORDER}; color: #000; }}
#FileButton {{ background: {FILE_TAB}; color: white; border: none; padding: 4px 16px 5px 16px; font-weight: 600; }}
#FileButton:hover {{ background: #1d4b8d; }}
#FileButton::menu-indicator {{ image: none; width: 0; }}
#RibbonPanel {{ background: #fbfbfc; border-top: 1px solid {BORDER}; }}
#RibbonGroup {{ background: transparent; }}
#RibbonGroupTitle {{ color: {MUTED}; font-size: {pt("caption"):.1f}pt; }}
#RibbonSep {{ background: #dcdfe3; }}
#RibbonPanel QToolButton {{ border: 1px solid transparent; border-radius: 2px; padding: 1px 3px; background: transparent; }}
#RibbonPanel QToolButton:hover {{ background: #e2ebf6; border-color: #a9c6e8; }}
#RibbonPanel QToolButton:pressed, #RibbonPanel QToolButton:checked {{ background: #cfe0f4; border-color: #7fa9dc; }}
#RibbonPanel QToolButton:disabled {{ color: {DISABLED_TEXT}; }}
#RibbonPanel QToolButton::menu-indicator {{ subcontrol-position: bottom center; subcontrol-origin: padding; bottom: -1px; }}

/* ---------- docks / pane headers ---------- */
QDockWidget {{ titlebar-close-icon: none; font-weight: 600; }}
QDockWidget::title {{ background: {HEADER_BG}; padding: 4px 6px; border-bottom: 1px solid {BORDER}; text-align: left; }}
QMainWindow::separator {{ background: {WINDOW}; width: 5px; height: 5px; }}
QMainWindow::separator:hover {{ background: #c6d8ee; }}
QTabBar::tab {{ background: #e4e7eb; border: 1px solid {BORDER}; padding: 4px 12px; margin-right: -1px; color: #333; }}
QTabBar::tab:selected {{ background: {PANEL}; border-bottom-color: {PANEL}; color: #000; border-top: 2px solid {ACCENT}; }}
QTabBar::tab:hover:!selected {{ background: #edf1f5; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; background: {PANEL}; top: -1px; }}
#PaneHeader {{ background: {HEADER_BG}; border-bottom: 1px solid {BORDER}; }}
#PaneTitle {{ font-weight: 600; color: #222; }}
#PaneToolbar QToolButton {{ border: 1px solid transparent; border-radius: 2px; padding: 1px; }}
#PaneToolbar QToolButton:hover {{ background: #dfe7f1; border-color: #b7c9de; }}
#PaneToolbar QToolButton:checked {{ background: #cfe0f4; border-color: #7fa9dc; }}
#TreeFilter {{ border: 1px solid #c9cdd2; padding: 1px 4px; }}

/* ---------- tree ---------- */
QTreeView {{ background: {PANEL}; border: none; show-decoration-selected: 1; outline: 0; }}
QTreeView::item {{ padding: 2px 2px; border: 1px solid transparent; }}
QTreeView::item:hover {{ background: #eef5fd; border-color: #e0ecf9; }}
QTreeView::item:selected {{ background: {ACCENT_LIGHT}; border: 1px solid {SEL_BORDER}; color: #000; }}
QTreeView:focus {{ border: 1px solid #9cc2ee; }}

/* ---------- settings forms ---------- */
#SettingsArea, #SettingsBody {{ background: {PANEL}; }}
#SettingsTitle {{ font-size: {pt("heading"):.1f}pt; font-weight: 600; color: #111; }}
#SettingsSubtitle {{ color: {MUTED}; }}
#SectionHeader {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #f6f7f9, stop:1 #e8eaee);
    border-top: 1px solid #d9dce1; border-bottom: 1px solid #d9dce1; text-align: left; padding: 3px 4px;
    font-weight: 600; color: #1f2328; }}
#SectionHeader:hover {{ background: #e6edf6; }}
#UnitLabel {{ color: #444; }}
#SymbolLabel {{ color: #222; font-style: italic; }}
#FieldError {{ color: {ERROR}; font-size: {pt("caption"):.1f}pt; }}
#FieldNote {{ color: {WARNING}; font-size: {pt("caption"):.1f}pt; }}
#SelectionCount {{ color: {MUTED}; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: white; border: 1px solid #b9bec5; border-radius: 1px; padding: 2px 4px; selection-background-color: {ACCENT}; }}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QComboBox:focus {{ border-color: #3d8ee6; }}
QLineEdit[error="true"] {{ background: {ERROR_BG}; border-color: #e0605a; }}
QLineEdit[draft="true"] {{ border-style: dashed; border-color: #3d8ee6; }}
QLineEdit[warn="true"] {{ border-color: #e09a2a; }}
QLineEdit:read-only {{ background: #f3f4f6; color: #555; }}
QComboBox::drop-down {{ border-left: 1px solid #d0d4d9; width: 16px; }}
QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #ffffff, stop:1 #eceef1); border: 1px solid #b4b9c0;
    border-radius: 2px; padding: 3px 10px; }}
QPushButton:hover {{ border-color: #7fa9dc; background: #eef4fb; }}
QPushButton:pressed {{ background: #d6e4f4; }}
QPushButton:default {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {DISABLED_TEXT}; border-color: #d0d3d8; }}
QHeaderView::section {{ background: #eef0f3; border: none; border-right: 1px solid #d5d8dc; border-bottom: 1px solid #d5d8dc;
    padding: 3px 5px; font-weight: 600; }}
QTableView, QTableWidget, QListWidget {{ background: white; border: 1px solid #c9cdd2; gridline-color: #e3e5e8;
    selection-background-color: {ACCENT_LIGHT}; selection-color: #000; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 13px; height: 13px; }}
QScrollArea {{ border: none; background: {PANEL}; }}
QScrollBar:vertical {{ background: #f0f1f3; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #c5c9ce; min-height: 24px; border-radius: 5px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: #a9aeb5; }}
QScrollBar:horizontal {{ background: #f0f1f3; height: 12px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #c5c9ce; min-width: 24px; border-radius: 5px; margin: 2px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QMenu {{ background: white; border: 1px solid #b9bec5; padding: 3px 0; }}
QMenu::item {{ padding: 4px 26px 4px 26px; }}
QMenu::item:selected {{ background: {ACCENT_LIGHT}; color: #000; }}
QMenu::item:disabled {{ color: {DISABLED_TEXT}; }}
QMenu::separator {{ height: 1px; background: #e1e3e6; margin: 3px 6px; }}
QMenu::icon {{ padding-left: 6px; }}
QStatusBar {{ background: {WINDOW}; border-top: 1px solid {BORDER}; color: #333; }}
QStatusBar QLabel {{ padding: 0 6px; }}
#StatusState {{ color: {MUTED}; }}
#StatusState[state="stale"] {{ color: {STALE}; }}
#StatusState[state="error"] {{ color: {ERROR}; font-weight: 600; }}
#StatusState[state="ok"] {{ color: {OK}; }}
QProgressBar {{ border: 1px solid #b9bec5; border-radius: 1px; background: white; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #5aa65a, stop:1 #3e8a3e); }}
#GraphicsToolbar {{ background: #f6f7f8; border-bottom: 1px solid {BORDER}; }}
#GraphicsToolbar QToolButton {{ border: 1px solid transparent; border-radius: 2px; padding: 1px; }}
#GraphicsToolbar QToolButton:hover {{ background: #dfe7f1; border-color: #b7c9de; }}
#GraphicsToolbar QToolButton:checked {{ background: #cfe0f4; border-color: #7fa9dc; }}
#GraphicsSep {{ color: {BORDER}; padding: 0 3px; }}
#GraphicsInfo, #TableTitle {{ color: #555; padding: 0 6px; }}
#MessagesView {{ background: white; border: none; font-family: "Consolas", "Menlo", "DejaVu Sans Mono", monospace;
    font-size: {pt("mono"):.1f}pt; }}
#MessagesPlain {{ background: white; border: none; }}
#WizardTitle {{ font-size: {pt("display"):.1f}pt; font-weight: 600; color: #1d1f22; }}
#WizardCard {{ background: white; border: 1px solid #c9cdd2; border-radius: 3px; }}
#WizardCard:hover {{ border-color: {ACCENT}; background: #f3f8fe; }}
#NewTile {{ background: white; border: 1px solid #c9cdd2; border-radius: 4px; padding: 12px; font-size: {pt("tile"):.1f}pt; }}
#NewTile:hover {{ border: 2px solid {ACCENT}; background: #f3f8fe; }}
"""


QSS = None       # kept for scripts that imported the old constant; apply() fills it


def apply(app: QApplication, font_pt: float | None = None):
    """Apply the theme. *font_pt*: base font size in points (default: the saved preference, else 9 pt)."""
    global _base, QSS
    import os
    if font_pt is None and os.environ.get("ELMERSTUDIO_FONT_PT"):
        font_pt = float(os.environ["ELMERSTUDIO_FONT_PT"])          # tests and screenshots at other sizes
    if font_pt is None:
        try:
            from PySide6.QtCore import QSettings
            font_pt = float(QSettings("ElmerStudio", "ElmerStudio").value("font_pt", BASE_PT))
        except (TypeError, ValueError):
            font_pt = BASE_PT
    _base = min(16.0, max(7.0, float(font_pt)))
    app.setStyle(QStyleFactory.create("Fusion"))
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(WINDOW))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase, QColor("#f6f8fa"))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor("#f2f3f5"))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor("#ffffe8"))
    pal.setColor(QPalette.ToolTipText, QColor("#222222"))
    pal.setColor(QPalette.Link, QColor(ACCENT))
    pal.setColor(QPalette.PlaceholderText, QColor(DISABLED_TEXT))
    for group in (QPalette.Disabled,):
        pal.setColor(group, QPalette.WindowText, QColor(DISABLED_TEXT))
        pal.setColor(group, QPalette.Text, QColor(DISABLED_TEXT))
        pal.setColor(group, QPalette.ButtonText, QColor(DISABLED_TEXT))
    app.setPalette(pal)
    f = QFont()
    for fam in ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Ubuntu", "Noto Sans", "DejaVu Sans", "Arial"):
        f.setFamily(fam)
        if QFont(fam).exactMatch() or fam == "Arial":
            break
    f.setPointSizeF(pt("ui"))
    app.setFont(f)
    QSS = qss()
    app.setStyleSheet(QSS)


def set_base_pt(value: float):
    """For tests and the Preferences dialog: change the base size without re-styling (apply() re-styles)."""
    global _base
    _base = float(value)
