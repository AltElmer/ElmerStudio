"""Elmer Studio icon set.

Original flat engineering icons in the spirit of a modern CAE desktop
(model tree, ribbon, graphics toolbar).  Every icon is a complete SVG
document drawn on a 32x32 grid (``viewBox="0 0 32 32"``) and designed to stay
legible at 16 px: few details, 1.5-2 px strokes, a shared palette.  Letter
glyphs (a=, f(x), sif, Py, ...) are drawn as stroked paths rather than
``<text>`` so they render identically whatever fonts the machine has.

Public API::

    SVG: dict[str, str]                      # name -> SVG document
    icon(name) -> QIcon                      # cached, 16/20/24/32/48 px
    pixmap(name, size) -> QPixmap            # cached, DPR aware
    names() -> list[str]

``icon()`` / ``pixmap()`` need a ``QGuiApplication`` (or ``QApplication``);
nothing Qt-GUI related is created at import time.  Unknown names resolve to a
neutral gray fallback icon instead of raising.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

__all__ = ["SVG", "icon", "names", "pixmap"]

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
WHITE = "#ffffff"
TB = "#495057"  # toolbar glyph gray
TB_L = "#adb5bd"
INK = "#343a40"
NAVY = "#1f3d66"
BLUE = "#2f6fde"  # selection / highlight
BLUE_D = "#1f4fa8"
BLUE_L = "#dbe7fb"
GREEN = "#2b8a3e"
GREEN_M = "#2f9e44"
GREEN_L = "#40c057"
RED = "#e03131"
RED_D = "#c92a2a"
AMBER = "#f59f00"
YELLOW = "#fcc419"
GEO_F = "#8fa8c8"  # geometry fill
GEO_S = "#3d5a80"  # geometry outline
GEO_T = "#c9d6e8"  # geometry top face
GEO_D = "#6f8cb4"  # geometry shaded face
MESH_E = "#1f3d66"
MESH_F = "#d0e2f5"
HEAT = "#e8590c"
HEAT_D = "#c92a2a"
SOLID = "#0b7285"
ELEC = "#f08c00"
ES = "#7048e8"
FLOW = "#1098ad"
ACOU = "#9c36b5"
SPECIES = "#2b8a3e"
PDE = "#343a40"
FOLD_F = "#f7c948"
FOLD_L = "#fcdc7b"
FOLD_S = "#b07d10"
F_TOP = "#f8f9fa"  # neutral (feature) cube faces
F_LEFT = "#e9ecef"
F_RIGHT = "#dee2e6"
F_S = "#868e96"
X_C = "#e03131"
Y_C = "#2f9e44"
Z_C = "#1971c2"
RAINBOW = ((0.0, "#2f6fde"), (0.35, "#37b24d"), (0.68, "#fcc419"), (1.0, "#e03131"))
DASH = "2 1.5"


# ---------------------------------------------------------------------------
# SVG building helpers
# ---------------------------------------------------------------------------
def _n(v: float) -> str:
    s = f"{float(v):.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _pts(points) -> str:
    return " ".join(f"{_n(x)},{_n(y)}" for x, y in points)


def _attrs(fill, stroke=None, sw=1.5, extra="", dash=None) -> str:
    a = f' fill="{fill}"'
    if stroke:
        cap = "butt" if dash else "round"
        a += (f' stroke="{stroke}" stroke-width="{_n(sw)}"'
              f' stroke-linejoin="round" stroke-linecap="{cap}"')
        if dash:
            a += f' stroke-dasharray="{dash}"'
    if extra:
        a += " " + extra
    return a


def _doc(body: str, defs: str = "") -> str:
    d = f"<defs>{defs}</defs>" if defs else ""
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" '
            f'viewBox="0 0 32 32">{d}{body}</svg>')


def _rect(x, y, w, h, fill="none", stroke=None, sw=1.5, rx=0.0, extra="", dash=None):
    r = f' rx="{_n(rx)}" ry="{_n(rx)}"' if rx else ""
    return (f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}"{r}'
            f'{_attrs(fill, stroke, sw, extra, dash)}/>')


def _circle(cx, cy, r, fill="none", stroke=None, sw=1.5, extra="", dash=None):
    return f'<circle cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(r)}"{_attrs(fill, stroke, sw, extra, dash)}/>'


def _ellipse(cx, cy, rx, ry, fill="none", stroke=None, sw=1.5, extra="", dash=None):
    return (f'<ellipse cx="{_n(cx)}" cy="{_n(cy)}" rx="{_n(rx)}" ry="{_n(ry)}"'
            f'{_attrs(fill, stroke, sw, extra, dash)}/>')


def _poly(points, fill="none", stroke=None, sw=1.5, extra="", dash=None):
    return f'<polygon points="{_pts(points)}"{_attrs(fill, stroke, sw, extra, dash)}/>'


def _polyline(points, stroke, sw=2.0, extra="", dash=None):
    return f'<polyline points="{_pts(points)}"{_attrs("none", stroke, sw, extra, dash)}/>'


def _path(d, fill="none", stroke=None, sw=1.5, extra="", dash=None):
    return f'<path d="{d}"{_attrs(fill, stroke, sw, extra, dash)}/>'


def _line(x1, y1, x2, y2, stroke, sw=2.0, extra="", dash=None):
    return _path(f"M{_n(x1)} {_n(y1)}L{_n(x2)} {_n(y2)}", "none", stroke, sw, extra, dash)


def _g(content: str, transform: str) -> str:
    return f'<g transform="{transform}">{content}</g>'


def _stops(stops) -> str:
    return "".join(f'<stop offset="{_n(o)}" stop-color="{c}"/>' for o, c in stops)


def _lin(gid, stops, x1=0.0, y1=0.0, x2=1.0, y2=0.0) -> str:
    return (f'<linearGradient id="{gid}" x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" '
            f'y2="{_n(y2)}">{_stops(stops)}</linearGradient>')


def _rad(gid, stops, cx=0.5, cy=0.5, r=0.5, fx=None, fy=None) -> str:
    f = ""
    if fx is not None:
        f = f' fx="{_n(fx)}" fy="{_n(fy)}"'
    return (f'<radialGradient id="{gid}" cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(r)}"{f}>'
            f"{_stops(stops)}</radialGradient>")


def _circle_d(cx, cy, r) -> str:
    return (f"M{_n(cx - r)} {_n(cy)}A{_n(r)} {_n(r)} 0 1 0 {_n(cx + r)} {_n(cy)}"
            f"A{_n(r)} {_n(r)} 0 1 0 {_n(cx - r)} {_n(cy)}Z")


def _star_d(cx, cy, big, small, n=5, rot=-90.0) -> str:
    pts = []
    for i in range(2 * n):
        a = math.radians(rot + i * 180.0 / n)
        rad = big if i % 2 == 0 else small
        pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    return "M" + "L".join(f"{_n(x)} {_n(y)}" for x, y in pts) + "Z"


def _gear_d(cx, cy, ro, ri, n, hole=0.0) -> str:
    step = 2 * math.pi / n
    pts = []
    for i in range(n):
        a = i * step - math.pi / 2
        for f, rad in ((-0.30, ri), (-0.16, ro), (0.16, ro), (0.30, ri)):
            ang = a + f * step
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    d = "M" + "L".join(f"{_n(x)} {_n(y)}" for x, y in pts) + "Z"
    if hole:
        d += _circle_d(cx, cy, hole)
    return d


def _unit(dx, dy):
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d


def _head(bx, by, ux, uy, color, length=6.0, width=7.0):
    """Arrow head whose base centre is (bx, by), pointing along (ux, uy)."""
    nx, ny = -uy * width / 2, ux * width / 2
    tip = (bx + ux * length, by + uy * length)
    return _poly([tip, (bx + nx, by + ny), (bx - nx, by - ny)], color, color, 1.0)


def _arrow(x1, y1, x2, y2, color, sw=2.2, length=6.0, width=7.0):
    """Straight arrow from (x1, y1) with its tip exactly at (x2, y2)."""
    ux, uy = _unit(x2 - x1, y2 - y1)
    bx, by = x2 - ux * length, y2 - uy * length
    return _line(x1, y1, bx, by, color, sw) + _head(bx, by, ux, uy, color, length, width)


def _arc_arrow(cx, cy, r, a0, a1, color, sw=2.2, length=5.5, width=7.0):
    """Circular arc (degrees, screen orientation) with an arrow head at a1."""
    def p(a):
        return cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))

    x0, y0 = p(a0)
    x1, y1 = p(a1)
    large = 1 if abs(a1 - a0) > 180 else 0
    sweep = 1 if a1 > a0 else 0
    d = f"M{_n(x0)} {_n(y0)}A{_n(r)} {_n(r)} 0 {large} {sweep} {_n(x1)} {_n(y1)}"
    sgn = 1 if a1 > a0 else -1
    rad = math.radians(a1)
    ux, uy = -math.sin(rad) * sgn, math.cos(rad) * sgn
    return _path(d, "none", color, sw) + _head(x1, y1, ux, uy, color, length, width)


def _iso(cx, cy, r):
    h = r * 0.8660254
    return {
        "T": (cx, cy - r), "UR": (cx + h, cy - r / 2), "LR": (cx + h, cy + r / 2),
        "B": (cx, cy + r), "LL": (cx - h, cy + r / 2), "UL": (cx - h, cy - r / 2),
        "C": (cx, cy),
    }


def _cube(cx=16.0, cy=16.0, r=13.0, top=GEO_T, left=GEO_F, right=GEO_D, stroke=GEO_S, sw=1.5):
    p = _iso(cx, cy, r)
    return (_poly([p["T"], p["UR"], p["C"], p["UL"]], top, stroke, sw)
            + _poly([p["UL"], p["C"], p["B"], p["LL"]], left, stroke, sw)
            + _poly([p["C"], p["UR"], p["LR"], p["B"]], right, stroke, sw))


def _fcube(cx=16.0, cy=16.0, r=13.0, top=F_TOP, left=F_LEFT, right=F_RIGHT, stroke=F_S, sw=1.5):
    return _cube(cx, cy, r, top, left, right, stroke, sw)


def _rainbow_cube(cx, cy, r, stroke=INK, sw=1.2):
    defs = (_lin("rt", [(0, "#fcc419"), (1, "#e03131")], 0, 1, 1, 0)
            + _lin("rl", [(0, "#37b24d"), (1, "#fcc419")], 0, 1, 0, 0)
            + _lin("rr", [(0, "#2f6fde"), (1, "#22b8cf")], 0, 1, 1, 0))
    return defs, _cube(cx, cy, r, "url(#rt)", "url(#rl)", "url(#rr)", stroke, sw)


def _page(x, y, w, h, fold=6.0, fill=WHITE, stroke=TB, sw=1.5):
    d = (f"M{_n(x)} {_n(y)}H{_n(x + w - fold)}L{_n(x + w)} {_n(y + fold)}"
         f"V{_n(y + h)}H{_n(x)}Z")
    f = f"M{_n(x + w - fold)} {_n(y)}V{_n(y + fold)}H{_n(x + w)}Z"
    return _path(d, fill, stroke, sw) + _path(f, "#dee2e6", stroke, sw)


def _eye(cx, cy, w, iris=BLUE, stroke=TB, sw=1.8):
    hw = w / 2
    d = (f"M{_n(cx - hw)} {_n(cy)}Q{_n(cx)} {_n(cy - hw)} {_n(cx + hw)} {_n(cy)}"
         f"Q{_n(cx)} {_n(cy + hw)} {_n(cx - hw)} {_n(cy)}Z")
    return _path(d, WHITE, stroke, sw) + _circle(cx, cy, w * 0.21, iris) + _circle(cx, cy, w * 0.09, NAVY)


def _cursor(x, y, s=1.0):
    pts = [(0, 0), (0, 12.5), (3, 9.6), (5.2, 14.2), (7.5, 13.2), (5.4, 8.8), (9.2, 8.8)]
    return _poly([(x + px * s, y + py * s) for px, py in pts], WHITE, "#212529", 1.3)


def _globe(cx, cy, r, sw=1.6):
    h = r * 0.866
    return (_circle(cx, cy, r, "#e7f0fb", BLUE, sw)
            + _ellipse(cx, cy, r * 0.42, r, "none", BLUE, sw * 0.8)
            + _line(cx - r, cy, cx + r, cy, BLUE, sw * 0.8)
            + _line(cx - h, cy - r / 2, cx + h, cy - r / 2, BLUE, sw * 0.7)
            + _line(cx - h, cy + r / 2, cx + h, cy + r / 2, BLUE, sw * 0.7))


def _equals(x, y, w, color=GREEN, h=3.0, gap=3.0, rx=0.8):
    return _rect(x, y, w, h, color, rx=rx) + _rect(x, y + h + gap, w, h, color, rx=rx)


def _window(x=2.5, y=4.5, w=27.0, h=23.0, bar=BLUE):
    return (_rect(x, y, w, h, WHITE, TB, 1.5, rx=2)
            + _rect(x + 0.75, y + 0.75, w - 1.5, 4.5, bar, rx=1.2))


def _plot_frame():
    defs = _lin("leg", [(0, "#e03131"), (0.33, "#fcc419"), (0.66, "#37b24d"), (1, "#2f6fde")],
                0, 0, 0, 1)
    body = _rect(2.5, 3.5, 27, 25, WHITE, TB, 1.5, rx=2) + _rect(23, 7, 4, 18, "url(#leg)", INK, 0.8)
    return defs, body


def _mesh(verts, tris, fills=None, stroke=MESH_E, sw=1.2):
    out = []
    for k, t in enumerate(tris):
        f = fills[k % len(fills)] if fills else MESH_F
        out.append(_poly([verts[i] for i in t], f, stroke, sw))
    return "".join(out)


_SQ_V = [(4, 4), (16, 4), (28, 4), (28, 16), (28, 28), (16, 28), (4, 28), (4, 16),
         (12, 12.5), (20, 19.5)]
_SQ_T = [(0, 1, 8), (7, 0, 8), (1, 2, 3), (1, 3, 9), (1, 9, 8), (3, 4, 9), (4, 5, 9),
         (5, 8, 9), (5, 6, 7), (5, 7, 8)]


def _square_mesh(fills=None, stroke=MESH_E, sw=1.2, scale=1.0, dx=0.0, dy=0.0):
    verts = [(dx + x * scale, dy + y * scale) for x, y in _SQ_V]
    return _mesh(verts, _SQ_T, fills, stroke, sw)


def _pencil(tx, ty, ex, ey, w=5.5):
    ux, uy = _unit(ex - tx, ey - ty)
    nx, ny = -uy * w / 2, ux * w / 2
    cone = 5.5
    bx, by = tx + ux * cone, ty + uy * cone
    shaft = [(bx + nx, by + ny), (ex + nx, ey + ny), (ex - nx, ey - ny), (bx - nx, by - ny)]
    band = [(ex + nx, ey + ny), (ex - nx, ey - ny),
            (ex - ux * 3 - nx, ey - uy * 3 - ny), (ex - ux * 3 + nx, ey - uy * 3 + ny)]
    tip = [(tx, ty), (bx + nx, by + ny), (bx - nx, by - ny)]
    lead = [(tx, ty), (tx + ux * 2.2 + nx * 0.4, ty + uy * 2.2 + ny * 0.4),
            (tx + ux * 2.2 - nx * 0.4, ty + uy * 2.2 - ny * 0.4)]
    return (_poly(shaft, AMBER, TB, 1.2) + _poly(band, "#f06595", TB, 1.2)
            + _poly(tip, "#ffe8cc", TB, 1.2) + _poly(lead, TB))


def _floppy():
    return (_path("M5 6.5Q5 5 6.5 5H22.5L27 9.5V25.5Q27 27 25.5 27H6.5Q5 27 5 25.5Z", BLUE, BLUE_D, 1.5)
            + _rect(10, 5, 11, 7.5, "#e9ecef", BLUE_D, 1.2)
            + _rect(16.8, 6.6, 2.6, 4.4, BLUE_D)
            + _rect(8.5, 16, 15, 11, WHITE, BLUE_D, 1.2)
            + _line(11, 20, 21, 20, TB_L, 1.4) + _line(11, 23.5, 21, 23.5, TB_L, 1.4))


def _folder_closed():
    return (_path("M3 25V7.5Q3 6 4.5 6H11L13.5 8.5H27.5Q29 8.5 29 10V25Z", FOLD_F, FOLD_S, 1.5)
            + _path("M3 12.5H29V25.5Q29 27 27.5 27H4.5Q3 27 3 25.5Z", FOLD_L, FOLD_S, 1.5))


def _tree_toggle(plus: bool) -> str:
    out = ""
    for y in (3.5, 18.5):
        out += _rect(3.5, y, 10, 10, WHITE, TB, 1.5, rx=1.5) + _line(6, y + 5, 11, y + 5, BLUE, 2)
        if plus:
            out += _line(8.5, y + 2.5, 8.5, y + 7.5, BLUE, 2)
        out += _line(16.5, y + 5, 28.5, y + 5, TB, 2.4)
    return out + _line(8.5, 13.5, 8.5, 18.5, TB, 1.2, dash="1.5 1.2")


def _pi(color=NAVY, sw=3.2):
    return (_path("M6 11Q6.5 8 9.5 8H26.5", "none", color, sw)
            + _path("M12.5 8.5Q12.5 19 8 25.5", "none", color, sw)
            + _path("M20.5 8.5V21.5Q20.5 25 24.5 24.5", "none", color, sw))


def _db(cx, top, rx, ry, h, gid="db"):
    defs = _lin(gid, [(0, "#5b8fe8"), (0.45, "#a5c3f5"), (1, BLUE)])
    x0, x1, bot = cx - rx, cx + rx, top + h
    body = _path(f"M{_n(x0)} {_n(top)}V{_n(bot)}A{_n(rx)} {_n(ry)} 0 0 0 {_n(x1)} {_n(bot)}V{_n(top)}",
                 f"url(#{gid})", BLUE_D, 1.4)
    bands = "".join(
        _path(f"M{_n(x0)} {_n(top + h * k)}A{_n(rx)} {_n(ry)} 0 0 0 {_n(x1)} {_n(top + h * k)}",
              "none", BLUE_D, 1.1)
        for k in (1 / 3, 2 / 3))
    return defs, body + bands + _ellipse(cx, top, rx, ry, "#e7f0fb", BLUE_D, 1.4)


def _eq_badge(x=17.5, y=18.5):
    return (_rect(x, y, 12, 10.5, "#ebfbee", GREEN, 1.2, rx=2)
            + _equals(x + 2.5, y + 2.6, 7, GREEN, 2, 1.8, 0.5))


# Boolean operation shapes: circles r=8.5 centred at (12,12) and (20,20); they
# intersect at ~(20.5,11.5) and ~(11.5,20.5).  The diagonal layout keeps the
# waist of the union visible at 16 px.
_BOOL_UNION = "M20.5 11.5A8.5 8.5 0 1 0 11.5 20.5A8.5 8.5 0 1 0 20.5 11.5Z"
_BOOL_LENS = "M20.5 11.5A8.5 8.5 0 0 1 11.5 20.5A8.5 8.5 0 0 1 20.5 11.5Z"
_BOOL_DIFF = "M20.5 11.5A8.5 8.5 0 1 0 11.5 20.5A8.5 8.5 0 0 1 20.5 11.5Z"


# ---------------------------------------------------------------------------
# Icon definitions
# ---------------------------------------------------------------------------
def _build() -> dict[str, str]:  # noqa: C901 - a flat catalogue of drawings
    S: dict[str, str] = {}

    # ===================== application / quick access =====================
    S["new"] = _doc(_page(5, 3, 18, 24) + _path(_star_d(23.5, 23.5, 6.5, 2.0, 4), YELLOW, "#e67700", 1.1))
    S["open"] = _doc(
        _path("M3 26V7.5Q3 6 4.5 6H11L13.5 8.5H24.5Q26 8.5 26 10V26Z", FOLD_F, FOLD_S, 1.5)
        + _path("M7 13.5H30L25.5 26H3Z", FOLD_L, FOLD_S, 1.5))
    S["save"] = _doc(_floppy())
    S["save_as"] = _doc(_g(_floppy(), "translate(0.2,0.2) scale(0.8)") + _pencil(16.5, 28.5, 28, 17))
    S["undo"] = _doc(_path("M26 25Q26 11 12 11", "none", BLUE, 3.2)
                     + _poly([(4, 11), (12.5, 5), (12.5, 17)], BLUE, BLUE, 1))
    S["redo"] = _doc(_path("M6 25Q6 11 20 11", "none", BLUE, 3.2)
                     + _poly([(28, 11), (19.5, 5), (19.5, 17)], BLUE, BLUE, 1))
    S["delete"] = _doc(
        _path("M12.5 7V5Q12.5 3.8 13.7 3.8H18.3Q19.5 3.8 19.5 5V7", "none", TB, 1.6)
        + _path("M8 11H24L22.6 27Q22.5 28.3 21.3 28.3H10.7Q9.5 28.3 9.4 27Z", "#f1f3f5", TB, 1.6)
        + _rect(5.5, 7, 21, 3.5, TB, rx=1.2)
        + _line(12.6, 14.5, 12.9, 25, RED_D, 1.8) + _line(16, 14.5, 16, 25, RED_D, 1.8)
        + _line(19.4, 14.5, 19.1, 25, RED_D, 1.8))
    S["copy"] = _doc(
        _page(4, 3, 15, 19, 5) + _page(12, 10, 16, 20, 5)
        + _line(15, 17.5, 24.5, 17.5, BLUE, 1.6) + _line(15, 21.5, 24.5, 21.5, BLUE, 1.6)
        + _line(15, 25.5, 21, 25.5, BLUE, 1.6))
    S["paste"] = _doc(
        _rect(4.5, 6, 18, 23, "#ced4da", TB, 1.5, rx=2)
        + _rect(9.5, 3.5, 8, 5, TB, rx=1.5)
        + _page(13, 12, 15, 18, 5)
        + _line(16, 19.5, 24.5, 19.5, BLUE, 1.6) + _line(16, 23, 24.5, 23, BLUE, 1.6)
        + _line(16, 26.5, 22, 26.5, BLUE, 1.6))
    S["duplicate"] = _doc(
        _rect(3, 3, 16, 13, WHITE, TB, 1.5, rx=2) + _rect(8.5, 8.5, 16, 13, WHITE, TB, 1.5, rx=2)
        + _line(11.5, 13, 21, 13, TB_L, 1.6) + _line(11.5, 17, 18, 17, TB_L, 1.6)
        + _circle(24, 24, 6.5, WHITE)
        + _line(24, 19.5, 24, 28.5, GREEN_M, 3) + _line(19.5, 24, 28.5, 24, GREEN_M, 3))
    S["rename"] = _doc(
        _rect(2.5, 8.5, 27, 15, WHITE, TB, 1.5, rx=2)
        + _circle(8.6, 17.6, 3.3, "none", TB, 2.1) + _line(11.9, 14.2, 11.9, 21, TB, 2.1)
        + _line(14.8, 11, 14.8, 21, TB, 2.1) + _circle(18.1, 17.6, 3.3, "none", TB, 2.1)
        + _line(25, 11.5, 25, 20.5, BLUE, 2)
        + _line(23, 11.5, 27, 11.5, BLUE, 1.5) + _line(23, 20.5, 27, 20.5, BLUE, 1.5))
    S["preferences"] = _doc(_path(_gear_d(16, 16, 13, 10, 8, 4.3), TB, extra='fill-rule="evenodd"'))
    S["help"] = _doc(
        _circle(16, 16, 13, BLUE, BLUE_D, 1.5)
        + _path("M11.8 12.3Q11.8 7.8 16 7.8Q20.2 7.8 20.2 11.8Q20.2 14.3 17.6 15.5Q16 16.3 16 18.8",
                "none", WHITE, 3.2)
        + _circle(16, 23.6, 2, WHITE))
    S["info"] = _doc(
        _circle(16, 16, 13, BLUE, BLUE_D, 1.5) + _circle(16, 9.6, 2.2, WHITE)
        + _rect(14, 13.5, 4, 11, WHITE, rx=1))
    S["warning"] = _doc(
        _path("M16 3.5L29.5 27.5H2.5Z", "#fab005", "#e67700", 1.6)
        + _rect(14.5, 11, 3, 9.5, INK, rx=1.4) + _circle(16, 23.6, 1.8, INK))
    S["error"] = _doc(
        _circle(16, 16, 13, RED, RED_D, 1.5)
        + _line(11, 11, 21, 21, WHITE, 3.4) + _line(21, 11, 11, 21, WHITE, 3.4))
    S["check"] = _doc(_path("M5 17L12.5 24.5L27 8", "none", GREEN_M, 4.2))
    S["cross"] = _doc(_line(7, 7, 25, 25, RED, 4.2) + _line(25, 7, 7, 25, RED, 4.2))
    S["add"] = _doc(_line(16, 5, 16, 27, GREEN_M, 4.6) + _line(5, 16, 27, 16, GREEN_M, 4.6))
    S["remove"] = _doc(_line(5, 16, 27, 16, RED, 4.6))
    S["clear"] = _doc(
        _poly([(14, 4.5), (21.43, 11.93), (13.43, 19.93), (6, 12.5)], WHITE, TB, 1.5)
        + _poly([(21.43, 11.93), (27.5, 18), (19.5, 26), (13.43, 19.93)], "#f783ac", TB, 1.5)
        + _line(12, 28.5, 28.5, 28.5, TB, 1.6))
    S["active"] = _doc(_circle(16, 16, 11.5, WHITE, TB, 1.6) + _circle(16, 16, 6.5, GREEN_L, GREEN, 1.2))
    up = [(16, 3), (28, 15), (20.5, 15), (20.5, 28.5), (11.5, 28.5), (11.5, 15), (4, 15)]
    S["up"] = _doc(_poly(up, BLUE, BLUE_D, 1.2))
    S["down"] = _doc(_poly([(x, 31.5 - y) for x, y in up], BLUE, BLUE_D, 1.2))
    S["collapse_all"] = _doc(_tree_toggle(False))
    S["expand_all"] = _doc(_tree_toggle(True))
    S["build_selected"] = _doc(_cube(12, 12.5, 9.5) + _poly([(17.5, 16), (29.5, 23), (17.5, 30)], GREEN_L, GREEN, 1.3))
    S["build_all"] = _doc(
        _cube(12, 12.5, 9.5)
        + _poly([(12.5, 17), (21, 23.5), (12.5, 30)], GREEN_L, GREEN, 1.3)
        + _poly([(20.5, 17), (29, 23.5), (20.5, 30)], GREEN_L, GREEN, 1.3))
    S["python"] = _doc(
        _rect(3, 3, 26, 26, "#3776ab", "#24527a", 1.5, rx=5)
        + _path("M9 23.5V8.5H13.5Q18.2 8.5 18.2 13Q18.2 17.5 13.5 17.5H9", "none", WHITE, 2.8)
        + _line(19.6, 12.5, 22.8, 21.2, "#ffd43b", 2.6) + _line(26, 12.5, 20.6, 27, "#ffd43b", 2.6))
    S["sif"] = _doc(
        _page(5, 3, 22, 26, 6) + _line(8.5, 9, 17, 9, TB_L, 1.8)
        + _path("M13.9 18.4Q13.3 16.6 11 16.6Q8.3 16.6 8.3 18.7Q8.3 20.5 11 20.8Q13.9 21.2 13.9 23.1"
                "Q13.9 25.3 11 25.3Q8.5 25.3 7.9 23.5", "none", BLUE_D, 2)
        + _line(17, 17, 17, 25.3, BLUE_D, 2.1) + _circle(17, 13.4, 1.4, BLUE_D)
        + _path("M21 25.3V14.6Q21 11.4 24.2 11.4", "none", BLUE_D, 2.1) + _line(19, 17.4, 24, 17.4, BLUE_D, 2))
    S["back"] = _doc(_line(27, 16, 8, 16, BLUE, 3.4) + _path("M15 8L7 16L15 24", "none", BLUE, 3.4))
    S["forward"] = _doc(_line(5, 16, 24, 16, BLUE, 3.4) + _path("M17 8L25 16L17 24", "none", BLUE, 3.4))
    S["show"] = _doc(_eye(16, 16, 27))
    S["hide"] = _doc(_eye(16, 16, 27, TB_L, F_S) + _line(6, 27, 26, 5, WHITE, 5.5) + _line(6, 27, 26, 5, TB, 2.6))
    S["enable"] = _doc(_rect(4, 4, 24, 24, WHITE, TB, 1.6, rx=3)
                       + _path("M9 16.5L14 21.5L23.5 10.5", "none", GREEN_M, 3.6))
    S["disable"] = _doc(_circle(16, 16, 11.5, "none", RED, 3.2) + _line(8, 24, 24, 8, RED, 3.2))
    S["app"] = _doc(
        _rect(2, 2, 28, 28, "url(#ag)", rx=6.5)
        + _poly([(11.5, 16.75), (20.5, 16.75), (16, 25)], HEAT)
        + _poly([(7, 25), (25, 25), (16, 8.5)], "none", WHITE, 1.9)
        + _poly([(11.5, 16.75), (20.5, 16.75), (16, 25)], "none", WHITE, 1.9),
        _lin("ag", [(0, "#4d8ef0"), (1, NAVY)], 0, 0, 1, 1))
    S["close"] = _doc(_line(8, 8, 24, 24, TB, 2.8) + _line(24, 8, 8, 24, TB, 2.8))
    S["folder"] = _doc(_folder_closed())
    S["run"] = _doc(_path("M9 5.5L27 16L9 26.5Z", GREEN_L, GREEN, 1.6))
    S["stop"] = _doc(_rect(6.5, 6.5, 19, 19, RED, RED_D, 1.5, rx=2.5))
    S["refresh"] = _doc(_arc_arrow(16, 16, 10.5, -165, -35, BLUE, 3, 6, 8)
                        + _arc_arrow(16, 16, 10.5, 15, 145, BLUE, 3, 6, 8))

    # =============================== model tree ===============================
    S["root"] = _doc(_page(5, 2.5, 22, 27, 6) + _cube(16, 18.5, 7.5))
    S["global_defs"] = _doc(_globe(16, 16, 12.5))
    S["parameters"] = _doc(_pi())
    S["variables"] = _doc(_circle(9.6, 18.5, 5, "none", NAVY, 3) + _line(14.6, 12.5, 14.6, 23.8, NAVY, 3)
                          + _equals(18.5, 13.8, 10, BLUE, 3, 3))
    S["functions"] = _doc(
        _path("M8 24V11.6Q8 7.4 12.1 7.4", "none", NAVY, 2.7) + _line(5, 13, 11.5, 13, NAVY, 2.5)
        + _path("M16 7.5Q12.4 15.75 16 24", "none", NAVY, 2.1)
        + _line(17.6, 12.8, 22.4, 22.5, NAVY, 2.4) + _line(22.4, 12.8, 17.6, 22.5, NAVY, 2.4)
        + _path("M24.2 7.5Q27.8 15.75 24.2 24", "none", NAVY, 2.1))
    S["component"] = _doc(_cube(16, 16, 13, "#eef4fd", WHITE, BLUE_L, BLUE, 1.9))
    S["definitions"] = _doc(
        _path("M4 9Q4 6.5 6.5 6.5H20L28 16L20 25.5H6.5Q4 25.5 4 23Z", "#e7f0fb", BLUE_D, 1.5)
        + _circle(21.5, 16, 1.8, WHITE, BLUE_D, 1.2)
        + _path("M8.5 10.5H11.3Q16.8 10.5 16.8 16Q16.8 21.5 11.3 21.5H8.5Z", "none", BLUE_D, 2.5))
    S["selection_node"] = _doc(_cube(16, 16, 13, BLUE, GEO_F, GEO_D))
    S["geometry"] = _doc(
        _rect(3.5, 15, 13, 13, GEO_F, GEO_S) + _circle(22, 21.5, 7, GEO_T, GEO_S)
        + _poly([(16, 3), (24.5, 15.5), (7.5, 15.5)], GEO_D, GEO_S))
    S["block"] = _doc(_cube())
    S["sphere"] = _doc(
        _circle(16, 16, 12.5, "url(#sg)", GEO_S)
        + _path("M3.5 16A12.5 4.5 0 0 0 28.5 16", "none", GEO_S, 1, 'opacity="0.55"'),
        _rad("sg", [(0, "#f4f7fb"), (0.45, GEO_F), (1, "#4a678f")], 0.5, 0.5, 0.5, 0.35, 0.3))
    S["cylinder"] = _doc(
        _path("M5 8V24A11 4 0 0 0 27 24V8", "url(#cg)", GEO_S)
        + _ellipse(16, 8, 11, 4, GEO_T, GEO_S),
        _lin("cg", [(0, GEO_D), (0.4, GEO_T), (1, "#56739c")]))
    S["cone"] = _doc(
        _path("M16 3.5L28 24A12 4.5 0 0 1 4 24Z", "url(#cg)", GEO_S)
        + _path("M4 24A12 4.5 0 0 1 28 24", "none", GEO_S, 1, 'opacity="0.6"', dash=DASH),
        _lin("cg", [(0, GEO_D), (0.45, GEO_T), (1, "#56739c")]))
    S["torus"] = _doc(
        _path("M2.5 16.5A13.5 9 0 1 0 29.5 16.5A13.5 9 0 1 0 2.5 16.5Z"
              "M10.5 15A5.5 2.6 0 1 0 21.5 15A5.5 2.6 0 1 0 10.5 15Z",
              "url(#tg)", GEO_S, 1.5, 'fill-rule="evenodd"')
        + _path("M7.5 15.5Q16 24 24.5 15.5", "none", GEO_S, 1.1, 'opacity="0.5"'),
        _lin("tg", [(0, "#dfe7f2"), (0.55, GEO_F), (1, "#4f6c94")], 0, 0, 0, 1))
    S["rectangle"] = _doc(_rect(3.5, 8, 25, 16, GEO_F, GEO_S, 1.6))
    S["circle"] = _doc(_circle(16, 16, 12.5, GEO_F, GEO_S, 1.6))
    S["ellipse"] = _doc(_ellipse(16, 16, 13, 8.5, GEO_F, GEO_S, 1.6))
    poly = [(4, 20), (8.5, 6), (21, 4), (28, 14.5), (22, 27.5), (9, 26)]
    S["polygon"] = _doc(_poly(poly, GEO_F, GEO_S, 1.6) + "".join(_circle(x, y, 2, GEO_S) for x, y in poly))
    S["point"] = _doc(
        _line(16, 3.5, 16, 10, GEO_F, 2) + _line(16, 22, 16, 28.5, GEO_F, 2)
        + _line(3.5, 16, 10, 16, GEO_F, 2) + _line(22, 16, 28.5, 16, GEO_F, 2)
        + _circle(16, 16, 4.6, GEO_S))
    S["interval"] = _doc(_line(6, 16, 26, 16, GEO_S, 2.6)
                         + _circle(6, 16, 3.4, GEO_F, GEO_S, 1.5) + _circle(26, 16, 3.4, GEO_F, GEO_S, 1.5))
    S["union"] = _doc(_path(_BOOL_UNION, GEO_F, GEO_S, 1.6))
    S["difference"] = _doc(_circle(20, 20, 8.5, "none", GEO_S, 1.3, dash=DASH)
                           + _path(_BOOL_DIFF, GEO_F, GEO_S, 1.6))
    S["intersection"] = _doc(_circle(12, 12, 8.5, "none", GEO_S, 1.3, dash=DASH)
                             + _circle(20, 20, 8.5, "none", GEO_S, 1.3, dash=DASH)
                             + _path(_BOOL_LENS, GEO_F, GEO_S, 1.6))
    S["form_union"] = _doc(
        _rect(3.5, 7.5, 15, 15, GEO_F) + _circle(20.5, 18, 8.5, GEO_F)
        + _rect(3.5, 7.5, 15, 15, "none", GEO_S, 1.5) + _circle(20.5, 18, 8.5, "none", GEO_S, 1.5))
    S["form_assembly"] = _doc(
        _rect(3, 8, 10.5, 16, GEO_F, GEO_S) + _rect(18.5, 8, 10.5, 16, GEO_T, GEO_S)
        + _line(13.5, 8, 13.5, 24, BLUE, 2.6) + _line(18.5, 8, 18.5, 24, BLUE, 2.6))
    S["import"] = _doc(_cube(21.5, 16, 9.5) + _arrow(2.5, 16, 14.5, 16, GREEN_M, 3, 6, 9))
    S["export"] = _doc(_cube(10.5, 16, 9.5) + _arrow(17, 16, 30, 16, BLUE, 3, 6, 9))
    S["move"] = _doc(
        _rect(3.5, 16.5, 11, 11, "none", GEO_S, 1.3, dash=DASH) + _rect(17.5, 4.5, 11, 11, GEO_F, GEO_S)
        + _arrow(9, 22, 19, 13.5, BLUE, 2.2, 5.5, 6.5))
    S["rotate"] = _doc(
        _rect(9, 9, 14, 14, "none", GEO_S, 1.2, dash=DASH)
        + _rect(9, 9, 14, 14, GEO_F, GEO_S, 1.5, extra='transform="rotate(30 16 16)"')
        + _arc_arrow(16, 16, 13, -115, -32, BLUE, 2.2, 5, 6.5))
    S["scale"] = _doc(
        _rect(3.5, 3.5, 25, 25, "none", GEO_S, 1.2, dash=DASH) + _rect(3.5, 17, 11.5, 11.5, GEO_F, GEO_S)
        + _arrow(15.5, 16.5, 27, 5, BLUE, 2.2, 5.5, 6.5))
    S["mirror"] = _doc(
        _poly([(13.5, 6), (13.5, 26), (3, 26)], GEO_F, GEO_S)
        + _poly([(18.5, 6), (18.5, 26), (29, 26)], GEO_T, GEO_S, 1.3, dash=DASH)
        + _line(16, 3, 16, 29, BLUE, 1.6, dash="3 2"))
    S["array"] = _doc("".join(
        _rect(4 + i * 9, 4 + j * 9, 6.5, 6.5, GEO_F if i == j == 0 else GEO_T, GEO_S, 1.3)
        for i in range(3) for j in range(3)))
    S["transform_copy"] = _doc(
        _rect(3.5, 16.5, 11, 11, GEO_F, GEO_S) + _rect(17.5, 4.5, 11, 11, GEO_F, GEO_S)
        + _arrow(9, 22, 19, 13.5, BLUE, 2.2, 5.5, 6.5)
        + _line(8, 3.5, 8, 11.5, GREEN_M, 2.6) + _line(4, 7.5, 12, 7.5, GREEN_M, 2.6))
    S["fillet"] = _doc(
        _path("M17 5H27V15", "none", GEO_S, 1.1, dash=DASH)
        + _path("M5 27V5H17A10 10 0 0 1 27 15V27Z", GEO_F, GEO_S)
        + _path("M17 5A10 10 0 0 1 27 15", "none", BLUE, 3))
    S["chamfer"] = _doc(
        _path("M18 5H27V14", "none", GEO_S, 1.1, dash=DASH)
        + _path("M5 27V5H18L27 14V27Z", GEO_F, GEO_S) + _line(18, 5, 27, 14, BLUE, 3))
    S["extrude"] = _doc(
        _poly([(3, 12), (13, 17), (13, 27), (3, 22)], GEO_F, GEO_S)
        + _poly([(13, 17), (23, 12), (23, 22), (13, 27)], GEO_D, GEO_S)
        + _poly([(3, 12), (13, 7), (23, 12), (13, 17)], GEO_T, GEO_S)
        + _arrow(27.5, 25, 27.5, 3.5, BLUE, 2.6, 6, 6.8))
    S["revolve"] = _doc(
        _line(16, 2.5, 16, 29.5, TB, 1.4, dash="3 2")
        + _rect(18.5, 3.5, 8.5, 14, GEO_F, GEO_S)
        + _path("M4.72 21.96A12 4.5 0 1 0 27.28 21.96", "none", BLUE, 2.7)
        + _head(27.28, 21.96, -0.696, -0.718, BLUE, 5.5, 7.5))
    S["work_plane"] = _doc(
        _poly([(3, 23), (11, 9), (29, 9), (21, 23)], "#dbe7fb", BLUE, 1.6)
        + _ellipse(16, 16, 5.5, 3.2, GEO_F, GEO_S, 1.3))

    # ------------------------------ materials ------------------------------
    copper = [(0, "#ffe8cc"), (0.45, "#e8833a"), (1, "#8f3f0c")]
    steel = [(0, "#f8f9fa"), (0.45, "#adb5bd"), (1, "#495057")]
    glass = [(0, "#e7f5ff"), (0.45, "#4dabf7"), (1, "#1864ab")]
    S["materials"] = _doc(
        _circle(11, 11.5, 7.5, "url(#m1)", "#495057", 1.1)
        + _circle(21, 11.5, 7.5, "url(#m2)", "#1864ab", 1.1)
        + _circle(16, 20.5, 7.5, "url(#m3)", "#8f3f0c", 1.1),
        _rad("m1", steel, 0.5, 0.5, 0.5, 0.35, 0.3) + _rad("m2", glass, 0.5, 0.5, 0.5, 0.35, 0.3)
        + _rad("m3", copper, 0.5, 0.5, 0.5, 0.35, 0.3))
    S["material"] = _doc(_circle(16, 16, 12.5, "url(#m3)", "#8f3f0c", 1.4),
                         _rad("m3", copper, 0.5, 0.5, 0.5, 0.35, 0.3))

    # ------------------------------- physics -------------------------------
    S["physics"] = _doc(
        _ellipse(16, 16, 12.8, 4.8, "none", BLUE, 1.7)
        + _ellipse(16, 16, 12.8, 4.8, "none", BLUE, 1.7, 'transform="rotate(60 16 16)"')
        + _ellipse(16, 16, 12.8, 4.8, "none", BLUE, 1.7, 'transform="rotate(-60 16 16)"')
        + _circle(16, 16, 3, BLUE_D))
    S["physics_heat"] = _doc(
        _rect(4, 22.5, 24, 6, "url(#hg)", HEAT_D, 1.2, rx=1.5)
        + "".join(_path(f"M{x} 19.5Q{x - 3} 17 {x} 14.5T{x} 9.5T{x} 4.5", "none", HEAT, 2.4)
                  for x in (9, 16, 23)),
        _lin("hg", [(0, "#fd7e14"), (1, HEAT_D)], 0, 0, 0, 1))
    S["physics_solid"] = _doc(
        _rect(2.5, 4, 3.5, 24, F_S, TB, 1)
        + _path("M6 8.5Q18 8.5 28 15.5L26.8 21.5Q17 15.5 6 15.5Z", SOLID, "#085a69", 1.3)
        + _arrow(25.5, 3, 25.5, 12.5, INK, 2, 4.5, 5.5))
    S["physics_es"] = _doc(
        _path("M10.5 10Q16 3 21.5 10", "none", ES, 1.6) + _path("M10.5 22Q16 29 21.5 22", "none", ES, 1.6)
        + _circle(8.5, 16, 6, ES, "#5235b8", 1.2) + _circle(23.5, 16, 6, "#b197fc", "#5235b8", 1.2)
        + _line(5.5, 16, 11.5, 16, WHITE, 2) + _line(8.5, 13, 8.5, 19, WHITE, 2)
        + _line(20.5, 16, 26.5, 16, WHITE, 2))
    S["physics_ec"] = _doc(
        _poly([(18, 2.5), (6.5, 18), (14.5, 18), (11.5, 29.5), (25.5, 12.5), (17.5, 12.5), (21, 2.5)],
              "url(#eg)", "#c25e00", 1.3),
        _lin("eg", [(0, "#ffd43b"), (1, ELEC)], 0, 0, 0, 1))
    S["physics_mf"] = _doc(
        _path("M4.5 4V17A11.5 11.5 0 0 0 16 28.5V21.5A4.5 4.5 0 0 1 11.5 17V4Z", RED_D, INK, 1.2)
        + _path("M27.5 4V17A11.5 11.5 0 0 1 16 28.5V21.5A4.5 4.5 0 0 0 20.5 17V4Z", Z_C, INK, 1.2)
        + _rect(4.5, 4, 7, 4.5, "#dee2e6", INK, 1.2) + _rect(20.5, 4, 7, 4.5, "#dee2e6", INK, 1.2))
    S["physics_spf"] = _doc("".join(
        _path(f"M3 {y}Q7 {y - 4} 11 {y}T19 {y}H23", "none", FLOW, 2.3) + _head(23, y, 1, 0, FLOW, 6, 7)
        for y in (8, 16, 24)))
    S["physics_acpr"] = _doc(
        _poly([(3.5, 12), (8.5, 12), (15, 6), (15, 26), (8.5, 20), (3.5, 20)], ACOU, "#6f2483", 1.2)
        + _path("M19.24 11.76A6 6 0 0 1 19.24 20.24", "none", ACOU, 2.2)
        + _path("M22.07 8.93A10 10 0 0 1 22.07 23.07", "none", ACOU, 2.2)
        + _path("M24.9 6.1A14 14 0 0 1 24.9 25.9", "none", ACOU, 2.2))
    dots = [(6.5, 9, 3.2, 1), (5.5, 18, 3.2, 1), (11, 13.5, 3.2, 1), (7.5, 25.5, 2.8, 0.95),
            (13, 22, 2.8, 0.95), (16, 8, 2.6, 0.85), (18.5, 16.5, 2.4, 0.8), (22.5, 10.5, 2, 0.7),
            (21.5, 23.5, 2, 0.7), (26.5, 17, 1.8, 0.6), (27, 7.5, 1.5, 0.5), (27, 26, 1.5, 0.5)]
    S["physics_tds"] = _doc("".join(
        _circle(x, y, r, SPECIES, extra=f'fill-opacity="{_n(o)}"') for x, y, r, o in dots))
    S["physics_pde"] = _doc(_path("M4 6.5H28L16 28Z" "M10.8 9.6H23.2L16.6 22Z", PDE,
                                  extra='fill-rule="evenodd"'))
    S["multiphysics"] = _doc(
        _circle(12, 12, 8.5, HEAT, HEAT_D, 1.2, 'fill-opacity="0.75"')
        + _circle(20, 12, 8.5, SOLID, "#085a69", 1.2, 'fill-opacity="0.75"')
        + _circle(16, 20, 8.5, ELEC, "#c25e00", 1.2, 'fill-opacity="0.75"'))

    # ------------------------------- features -------------------------------
    iso = _iso(16, 16, 13)
    S["feature_domain"] = _doc(_cube(16, 16, 13, "#a5c3f5", "#6d9bea", BLUE, BLUE_D))
    S["feature_boundary"] = _doc(_fcube(right=BLUE))
    S["feature_edge"] = _doc(_fcube() + _line(*iso["C"], *iso["B"], BLUE, 3.6)
                             + _line(*iso["C"], *iso["UR"], BLUE, 3.6))
    S["feature_point"] = _doc(_fcube() + _circle(16, 16, 4, BLUE, WHITE, 1.3))
    S["initial_values"] = _doc(
        _fcube(15, 14.5, 12, "#e7f0fb", "#cfe0fb", "#b6cff7")
        + _circle(24, 24, 6, BLUE_D, WHITE, 1.2) + _ellipse(24, 24, 1.9, 2.9, "none", WHITE, 1.7))
    S["default_feature"] = _doc(
        _fcube(15, 14.5, 12, F_TOP, F_LEFT, "#ced4da", TB_L)
        + _rect(19.5, 19.5, 11, 11, F_S, WHITE, 1, rx=2)
        + _path("M22.3 21.8H24.3Q27.7 21.8 27.7 25Q27.7 28.2 24.3 28.2H22.3Z", "none", WHITE, 1.7))

    # --------------------------------- mesh ---------------------------------
    S["mesh"] = _doc(_square_mesh())
    tri_a, tri_b, tri_c = (3, 27.5), (29, 27.5), (16, 4.5)

    def bary(i, j, n=3):
        x = tri_a[0] + i / n * (tri_b[0] - tri_a[0]) + j / n * (tri_c[0] - tri_a[0])
        y = tri_a[1] + i / n * (tri_b[1] - tri_a[1]) + j / n * (tri_c[1] - tri_a[1])
        if (i, j) == (1, 1):
            x, y = x + 1.2, y - 1.5
        return x, y

    ft = []
    for i in range(3):
        for j in range(3 - i):
            ft.append(_poly([bary(i, j), bary(i + 1, j), bary(i, j + 1)], MESH_F, MESH_E, 1.2))
            if i + j <= 1:
                ft.append(_poly([bary(i + 1, j), bary(i + 1, j + 1), bary(i, j + 1)], MESH_F, MESH_E, 1.2))
    S["free_tri"] = _doc("".join(ft))
    a, b, c, d = (16, 3), (3.5, 23), (28.5, 21), (17, 29)

    def mid(p, q):
        return ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)

    S["free_tet"] = _doc(
        _poly([a, b, d], MESH_F, MESH_E, 1.3) + _poly([a, d, c], "#a9c8ea", MESH_E, 1.3)
        + _poly([mid(a, b), mid(b, d), mid(a, d)], "none", MESH_E, 1.1)
        + _poly([mid(a, d), mid(d, c), mid(a, c)], "none", MESH_E, 1.1))
    sw_ = []
    p = _iso(16, 16, 13)
    sw_.append(_poly([p["T"], p["UR"], p["C"], p["UL"]], "#e7f1fb", MESH_E, 1.2))
    sw_.append(_line(*p["UL"], *p["UR"], MESH_E, 1.1))
    sw_.append(_poly([p["UL"], p["C"], p["B"], p["LL"]], MESH_F, MESH_E, 1.2))
    sw_.append(_poly([p["C"], p["UR"], p["LR"], p["B"]], "#a9c8ea", MESH_E, 1.2))
    for k in (1 / 3, 2 / 3):
        for (s0, s1), (e0, e1) in (((p["UL"], p["LL"]), (p["C"], p["B"])), ((p["C"], p["B"]), (p["UR"], p["LR"]))):
            sx, sy = s0[0] + k * (s1[0] - s0[0]), s0[1] + k * (s1[1] - s0[1])
            ex, ey = e0[0] + k * (e1[0] - e0[0]), e0[1] + k * (e1[1] - e0[1])
            sw_.append(_line(sx, sy, ex, ey, MESH_E, 1.1))
    S["swept"] = _doc("".join(sw_))
    bl = [_rect(3, 4, 26, 24, MESH_F)]
    for y in (26, 23.8, 21):
        bl.append(_line(3, y, 29, y, MESH_E, 1))
    for x in (9.5, 16, 22.5):
        bl.append(_line(x, 21, x, 28, MESH_E, 1))
    bl.append(_polyline([(3, 21), (9.5, 4), (16, 21), (22.5, 4), (29, 21)], MESH_E, 1.2))
    bl.append(_rect(3, 4, 26, 24, "none", MESH_E, 1.3))
    bl.append(_line(2, 28.5, 30, 28.5, INK, 2.6))
    S["boundary_layer"] = _doc("".join(bl))
    gx = [4, 5.6, 7.6, 10.2, 13.6, 18, 23.2, 28]
    S["distribution"] = _doc(
        _rect(4, 9, 24, 13, MESH_F, MESH_E, 1.2)
        + "".join(_line(x, 9, x, 22, MESH_E, 1.2) for x in gx[1:-1])
        + _line(4, 22, 28, 22, BLUE, 2.2)
        + "".join(_circle(x, 22, 1.9, BLUE, WHITE, 0.8) for x in gx))
    S["mesh_size"] = _doc(
        _mesh([(4, 21), (28, 21), (16, 4), (10, 12.5), (22, 12.5), (16, 21)],
              [(0, 5, 3), (5, 1, 4), (3, 4, 2), (3, 5, 4)])
        + _line(4, 24, 4, 30, BLUE, 1.3) + _line(28, 24, 28, 30, BLUE, 1.3)
        + _line(9.5, 27, 22.5, 27, BLUE, 1.8)
        + _head(9.5, 27, -1, 0, BLUE, 5, 5.5) + _head(22.5, 27, 1, 0, BLUE, 5, 5.5))
    S["mesh_stats"] = _doc(
        "".join(_rect(x, 28 - h, 4.5, h, MESH_F, MESH_E, 1.2)
                for x, h in ((6.5, 7), (12, 15), (17.5, 21), (23, 11)))
        + _path("M4 3.5V28H29.5", "none", TB, 1.7))

    # -------------------------------- study ---------------------------------
    S["study"] = _doc(_rect(3, 5, 26, 22, "#ebfbee", GREEN, 1.6, rx=4) + _equals(9, 10.5, 14, GREEN, 3.5, 4))
    S["compute"] = _doc(_rect(4.5, 8, 23, 5.5, GREEN_L, GREEN, 1.3, rx=1)
                        + _rect(4.5, 18.5, 23, 5.5, GREEN_L, GREEN, 1.3, rx=1))
    S["stationary"] = _doc(
        _path("M5 3.5V27.5H29", "none", TB, 1.7) + _line(5.5, 13, 28, 13, GREEN, 3)
        + "".join(_circle(x, 13, 2.2, WHITE, GREEN, 1.5) for x in (11, 18, 25)))
    S["time_dependent"] = _doc(
        _circle(16, 16, 12.5, WHITE, GREEN, 2.2)
        + "".join(_line(16 + 9.5 * math.cos(math.radians(a)), 16 + 9.5 * math.sin(math.radians(a)),
                        16 + 11 * math.cos(math.radians(a)), 16 + 11 * math.sin(math.radians(a)), TB, 1.5)
                  for a in range(0, 360, 90))
        + _line(16, 16, 16, 8, INK, 2.3) + _line(16, 16, 21.5, 19.5, INK, 2.3) + _circle(16, 16, 1.7, INK))
    S["eigenfrequency"] = _doc(
        _path("M4 16C9 29 13 29 16 16S23 3 28 16", "none", TB, 1.1, 'opacity="0.7"', dash=DASH)
        + _path("M4 16C9 3 13 3 16 16S23 29 28 16", "none", GREEN, 2.6)
        + _poly([(4, 17), (1.5, 21.5), (6.5, 21.5)], TB) + _poly([(28, 17), (25.5, 21.5), (30.5, 21.5)], TB))
    S["frequency_domain"] = _doc(
        _line(3, 16, 29, 16, TB_L, 1.3)
        + _path("M3 16Q6.25 1 9.5 16T16 16T22.5 16T29 16", "none", GREEN, 2.6))
    S["parametric_sweep"] = _doc(
        _arc_arrow(16, 16, 12, -55, 235, GREEN, 2.4, 5.5, 7)
        + _g(_pi(NAVY, 4.6), "translate(8.5,8.3) scale(0.47)"))
    S["solver"] = _doc(
        _path(_gear_d(16, 16, 13, 10, 8, 6.8), TB, extra='fill-rule="evenodd"')
        + _equals(12, 12.4, 8, GREEN_M, 2.6, 2.2, 0.6))
    S["solver_config"] = _doc(
        _line(4, 6, 20, 6, TB, 2.4) + _line(4, 11.5, 17, 11.5, TB, 2.4) + _line(4, 17, 13, 17, TB, 2.4)
        + _path(_gear_d(21.5, 21.5, 8.5, 6.4, 7, 2.8), GREEN_M, extra='fill-rule="evenodd"'))

    # ------------------------------- results --------------------------------
    S["results"] = _doc(
        _rect(6, 20, 4.5, 8, "#2f6fde") + _rect(11.5, 14, 4.5, 14, "#37b24d")
        + _rect(17, 9, 4.5, 19, "#fcc419") + _rect(22.5, 4, 4.5, 24, "#e03131")
        + _path("M4 3.5V28H29.5", "none", TB, 1.7))
    ddefs, dbody = _db(16, 7, 10, 3.5, 18)
    S["dataset"] = _doc(dbody, ddefs)
    sdefs, sbody = _db(12.5, 7, 9, 3.2, 17)
    S["solution"] = _doc(sbody + _eq_badge(17.5, 18.5), sdefs)
    fdefs, fbody = _plot_frame()
    cdefs, cbody = _rainbow_cube(13, 16.5, 7.5)
    S["plot_group_3d"] = _doc(fbody + cbody, fdefs + cdefs)
    S["plot_group_2d"] = _doc(fbody + _rect(5.5, 8, 15, 16, "url(#p2)", INK, 1),
                              fdefs + _lin("p2", RAINBOW, 0, 1, 1, 0))
    S["plot_group_1d"] = _doc(
        fbody + _path("M5.5 6.5V25H21", "none", TB, 1.2)
        + _path("M6.5 22C10 7 13.5 24 20.5 9", "none", "url(#p1)", 2.2),
        fdefs + _lin("p1", RAINBOW))
    S["surface_plot"] = _doc(
        _path("M3 12C8 4 13 16 18 9S26 6 29 8V21C25 18 21 25 16 22S7 22 3 25Z", "url(#sp)", INK, 1.2)
        + _path("M3 18.5C8 11 13 21 17 15.5S25 12 29 14.5", "none", WHITE, 0.9, 'opacity="0.7"'),
        _lin("sp", RAINBOW, 0, 1, 1, 0))
    vdefs, vbody = _rainbow_cube(16, 16, 13)
    S["volume_plot"] = _doc(vbody, vdefs)
    S["slice_plot"] = _doc("".join(
        _poly([(x0, 11), (x0 + 6, 5), (x0 + 6, 21), (x0, 27)], "url(#sl)", INK, 1.1) for x0 in (4, 12.5, 21)),
        _lin("sl", list(reversed([(1 - o, c) for o, c in RAINBOW])), 0, 0, 0, 1))
    S["isosurface"] = _doc(
        _path("M6 18C3 10 10 3 17 5S30 9 28 17S22 29 14 28S7.5 24 6 18Z", "url(#ib)", "#1864ab", 1.2)
        + _path("M10.5 17C10 12 14 9 18 10S24 15 22 19S12 23 10.5 17Z", "url(#io)", "#d9480f", 1.2),
        _rad("ib", [(0, "#d0ebff"), (0.6, "#4dabf7"), (1, "#1971c2")], 0.5, 0.5, 0.6, 0.35, 0.3)
        + _rad("io", [(0, "#fff3bf"), (0.5, "#fcc419"), (1, "#e8590c")], 0.5, 0.5, 0.6, 0.35, 0.3))
    S["arrow_plot"] = _doc(
        _arrow(4, 27.5, 13.5, 18, "#2f6fde", 2.4, 5.5, 6.5) + _arrow(16, 27.5, 27, 16.5, "#37b24d", 2.4, 5.5, 6.5)
        + _arrow(4, 15.5, 14.5, 5, "#f59f00", 2.4, 5.5, 6.5) + _arrow(16, 15.5, 28.5, 3, "#e03131", 2.4, 5.5, 6.5))
    S["streamline"] = _doc(
        _path("M3 26C12 26 14 12 29 12", "none", "#2f6fde", 2.4)
        + _path("M3 19C11 19 15 6 29 5", "none", "#37b24d", 2.4)
        + _path("M3 12C9 12 14 3 22 3", "none", "#e03131", 2.4)
        + _path("M10 29C17 29 21 19 29 19", "none", "#fab005", 2.4))
    S["contour"] = _doc(
        _ellipse(16, 16, 13, 9.5, "none", "#2f6fde", 2, 'transform="rotate(-25 16 16)"')
        + _ellipse(17, 15.5, 9.5, 6.8, "none", "#37b24d", 2, 'transform="rotate(-25 17 15.5)"')
        + _ellipse(17.8, 15, 6, 4.3, "none", "#f59f00", 2, 'transform="rotate(-25 17.8 15)"')
        + _ellipse(18.5, 14.6, 2.6, 1.9, "#e03131", "#e03131", 1.2, 'transform="rotate(-25 18.5 14.6)"'))
    S["line_graph"] = _doc(
        _path("M4.5 3.5V27.5H29", "none", TB, 1.7)
        + _path("M6 22C11 6 17 26 28 9", "none", BLUE, 2.3)
        + _path("M6 13C12 20 19 4 28 19", "none", RED, 2.3))
    S["deformation"] = _doc(
        _rect(2.5, 4, 3.5, 24, F_S, TB, 1)
        + _rect(6, 8, 22, 6, "none", TB, 1, dash=DASH)
        + _path("M6 8Q18 8 28 15L27 20.5Q17 14 6 14Z", "url(#dg)", INK, 1.2),
        _lin("dg", RAINBOW))
    S["mesh_plot"] = _doc(_square_mesh(["#74c0fc", "#8ce99a", "#ffe066", "#ffa8a8", "#a5d8ff", "#b2f2bb", "#ffd8a8"],
                                       INK, 1.1))
    S["derived_values"] = _doc(
        _rect(2.5, 7, 27, 18, "#f1f3f5", TB, 1.5, rx=3)
        + _path("M7.8 12.6L10.8 10.2V21.6", "none", NAVY, 2.6) + _circle(14.4, 20.6, 1.6, NAVY)
        + _path("M17.6 12.8Q18.1 10.2 21 10.2Q23.9 10.2 23.9 12.8Q23.9 14.9 21.5 16.9L17.6 21.4H24.4",
                "none", NAVY, 2.6))
    S["global_evaluation"] = _doc(_globe(12, 12, 9, 1.4) + _equals(18.5, 19.5, 10.5, GREEN, 3, 3))
    S["point_evaluation"] = _doc(
        _circle(11, 11, 7.5, WHITE, BLUE, 1.6)
        + _line(11, 1.8, 11, 5.5, BLUE, 1.6) + _line(11, 16.5, 11, 20.2, BLUE, 1.6)
        + _line(1.8, 11, 5.5, 11, BLUE, 1.6) + _line(16.5, 11, 20.2, 11, BLUE, 1.6)
        + _circle(11, 11, 2.6, BLUE) + _equals(18.5, 19.5, 10.5, GREEN, 3, 3))
    S["integration"] = _doc(_path("M22 6.2Q21.5 3.5 19 3.8Q16.2 4.2 16 9V23Q15.8 27.8 13 28.2Q10.5 28.5 10 25.8",
                                  "none", NAVY, 2.9))
    S["average"] = _doc(
        _path("M3 21C8 5 12 5 16 16S24 27 29 11", "none", BLUE, 2.3)
        + _line(3, 16, 29, 16, HEAT, 2.2, dash="3.5 2"))
    S["maxmin"] = _doc(
        _path("M3 18Q9 2 15 16T27 14", "none", TB, 2)
        + _poly([(9, 5), (13, 11.5), (5, 11.5)], RED, RED_D, 1)
        + _poly([(21, 26), (25, 19.5), (17, 19.5)], BLUE, BLUE_D, 1))
    S["table"] = _doc(
        _rect(3, 5, 26, 22, WHITE, TB, 1.5, rx=1.5) + _rect(3.75, 5.75, 24.5, 5.5, BLUE)
        + _line(3, 16.5, 29, 16.5, TB, 1.2) + _line(3, 21.75, 29, 21.75, TB, 1.2)
        + _line(12, 11.25, 12, 27, TB, 1.2) + _line(20.5, 11.25, 20.5, 27, TB, 1.2))
    S["image_export"] = _doc(
        _rect(2.5, 3.5, 21, 17, "#e7f5ff", TB, 1.5, rx=1.5)
        + _poly([(4, 19), (10, 11), (14, 16), (17, 13), (22, 19)], GREEN_L, GREEN, 1)
        + _circle(18, 8.5, 2.2, "#fab005")
        + _arrow(12, 26.5, 30, 26.5, BLUE, 2.8, 6, 8))
    S["data_export"] = _doc(
        _rect(2.5, 3.5, 21, 17, WHITE, TB, 1.5, rx=1.5) + _rect(3.2, 4.2, 19.6, 4.3, BLUE)
        + _line(2.5, 14.5, 23.5, 14.5, TB, 1.1) + _line(9.5, 8.5, 9.5, 20.5, TB, 1.1)
        + _line(16.5, 8.5, 16.5, 20.5, TB, 1.1)
        + _arrow(12, 26.5, 30, 26.5, BLUE, 2.8, 6, 8))
    S["animation"] = _doc(
        _rect(4, 3, 24, 26, INK, rx=1.5)
        + "".join(_rect(x, y, 2.2, 2.4, WHITE, rx=0.4) for x in (5.4, 24.4) for y in (5, 10, 15, 20, 25))
        + _rect(9.5, 5, 13, 10, "url(#an)") + _rect(9.5, 17, 13, 10, "url(#an)"),
        _lin("an", RAINBOW, 0, 1, 1, 0))
    S["report"] = _doc(
        _page(5, 3, 22, 26, 6) + _line(9, 9, 18, 9, BLUE, 2.4)
        + _line(9, 13.5, 23, 13.5, TB_L, 1.6) + _line(9, 17, 23, 17, TB_L, 1.6)
        + _rect(9.5, 22, 3, 4, "#2f6fde") + _rect(14, 20, 3, 6, "#37b24d") + _rect(18.5, 18.5, 3, 7.5, "#e03131"))
    S["cut_line"] = _doc(
        _rect(4, 4, 24, 24, "#f1f3f5", F_S, 1.4, rx=1)
        + _line(8.5, 23.5, 23.5, 8.5, BLUE, 2.6)
        + _circle(8.5, 23.5, 2.6, BLUE, WHITE, 1) + _circle(23.5, 8.5, 2.6, BLUE, WHITE, 1))
    pc = _iso(16, 16, 12.5)
    half = 6.25
    S["cut_plane"] = _doc(
        _fcube(16, 16, 12.5)
        + _poly([(pc["T"][0], pc["T"][1] + half), (pc["UR"][0], pc["UR"][1] + half),
                 (pc["C"][0], pc["C"][1] + half), (pc["UL"][0], pc["UL"][1] + half)],
                BLUE, BLUE_D, 1.2, 'fill-opacity="0.8"'))
    S["cut_point"] = _doc(_fcube(16, 16, 12.5) + _circle(16, 21, 3.6, BLUE, WHITE, 1.3))
    S["plot"] = _doc(
        _path("M5 27.5V18C10 8 15 22 20 12S26.5 8 28.5 8V27.5Z", "url(#pl)", extra='fill-opacity="0.9"')
        + _path("M5 18C10 8 15 22 20 12S26.5 8 28.5 8", "none", INK, 1.4)
        + _path("M4 3.5V27.5H29.5", "none", TB, 1.7),
        _lin("pl", RAINBOW))

    # =========================== graphics toolbar ============================
    def magnifier(inner=""):
        return (_circle(13, 13, 9.5, "#e7f5ff", TB, 2.3) + inner
                + _line(20, 20, 28, 28, TB, 4.2))

    S["zoom_in"] = _doc(magnifier(_line(8.5, 13, 17.5, 13, BLUE, 2.6) + _line(13, 8.5, 13, 17.5, BLUE, 2.6)))
    S["zoom_out"] = _doc(magnifier(_line(8.5, 13, 17.5, 13, BLUE, 2.6)))
    S["zoom_box"] = _doc(
        _rect(3, 3.5, 21, 16, BLUE_L, BLUE, 1.5, dash="3 2")
        + _circle(20, 20, 6, WHITE, TB, 2.1) + _line(24.3, 24.3, 29, 29, TB, 3.4))
    S["zoom_extents"] = _doc(
        _cube(16, 16, 7.5)
        + _path("M3.5 10V3.5H10", "none", BLUE, 2.4) + _path("M22 3.5H28.5V10", "none", BLUE, 2.4)
        + _path("M28.5 22V28.5H22", "none", BLUE, 2.4) + _path("M10 28.5H3.5V22", "none", BLUE, 2.4))
    S["view_default"] = _doc(
        _cube(12.5, 19, 9.5)
        + _poly([(23.5, 2), (30.5, 9), (16.5, 9)], BLUE, WHITE, 1.2)
        + _rect(18.6, 8.6, 9.8, 7.4, BLUE, WHITE, 1.2) + _rect(22.2, 11.2, 2.6, 4.8, WHITE))

    def view(h_col, v_col, dot_col):
        return (_rect(7, 8, 18, 17, "#e7f0fb")
                + _arrow(7, 25, 29, 25, h_col, 2.6, 6, 7) + _arrow(7, 25, 7, 3, v_col, 2.6, 6, 7)
                + _circle(7, 25, 3.2, dot_col, WHITE, 1.2))

    S["view_xy"] = _doc(view(X_C, Y_C, Z_C))
    S["view_yz"] = _doc(view(Y_C, Z_C, X_C))
    S["view_zx"] = _doc(view(Z_C, X_C, Y_C))
    S["view_iso"] = _doc(
        _fcube(16, 16, 12, "#f8f9fa", "#f1f3f5", "#e9ecef", "#ced4da", 1)
        + _arrow(16, 16, 16, 2.5, Z_C, 2.4, 5.5, 6.5)
        + _arrow(16, 16, 4.3, 22.8, X_C, 2.4, 5.5, 6.5)
        + _arrow(16, 16, 27.7, 22.8, Y_C, 2.4, 5.5, 6.5))
    S["perspective"] = _doc(
        _line(4.5, 16, 28, 4.5, TB, 1.4) + _line(4.5, 16, 28, 27.5, TB, 1.4)
        + _line(28, 4.5, 28, 27.5, BLUE, 2.6)
        + _rect(13, 12.5, 5, 7, GEO_F, GEO_S, 1.3) + _circle(4.5, 16, 2.4, INK))
    S["orthographic"] = _doc(
        "".join(_arrow(3, y, 26.5, y, TB, 1.4, 3.8, 4.6) for y in (6.5, 16, 25.5))
        + _line(28, 4.5, 28, 27.5, BLUE, 2.6)
        + _rect(12, 12, 5, 8, GEO_F, GEO_S, 1.3))
    S["select_box"] = _doc(_rect(3, 3, 21, 17, BLUE_L, BLUE, 1.5, dash="3 2") + _cursor(17, 14, 1.05))
    S["select_all"] = _doc(
        _rect(3, 3, 26, 26, "none", TB, 1.3, dash="3 2")
        + "".join(_rect(x, y, 8, 8, BLUE, BLUE_D, 1.2, rx=1) for x in (7, 17) for y in (7, 17)))
    S["clear_selection"] = _doc(
        _rect(3, 3, 26, 26, "none", TB, 1.3, dash="3 2")
        + "".join(_rect(x, y, 8, 8, "#e9ecef", F_S, 1.2, rx=1) for x in (7, 17) for y in (7, 17))
        + _circle(23.5, 23.5, 7, WHITE) + _line(20, 20, 27, 27, RED, 3) + _line(27, 20, 20, 27, RED, 3))
    checks = "".join(_rect(3 + i * 6, 3 + j * 6, 6, 6, TB_L)
                     for i in range(3) for j in range(3) if (i + j) % 2 == 0)
    S["transparency"] = _doc(
        _rect(3, 3, 18, 18, WHITE) + checks + _rect(3, 3, 18, 18, "none", TB, 1.2)
        + _rect(11, 11, 18, 18, BLUE, BLUE_D, 1.4, extra='fill-opacity="0.55"'))
    S["wireframe"] = _doc(_path(
        "M4 11H21V28H4ZM11 4H28V21H11ZM4 11L11 4M21 11L28 4M21 28L28 21M4 28L11 21",
        "none", TB, 1.6))
    S["scene_light"] = _doc(
        "".join(_line(16 + 9.3 * math.cos(math.radians(a)), 13.5 + 9.3 * math.sin(math.radians(a)),
                      16 + 11.5 * math.cos(math.radians(a)), 13.5 + 11.5 * math.sin(math.radians(a)),
                      AMBER, 1.8) for a in (-90, -45, -135, 0, 180))
        + _circle(16, 13.5, 7, "#ffe066", "#e67700", 1.5)
        + _rect(12.5, 20, 7, 6, F_S, TB, 1.3, rx=1)
        + _line(12.8, 23, 19.2, 23, TB, 1.1)
        + _rect(14, 26, 4, 2.5, TB, rx=1))
    S["axes"] = _doc(
        _arrow(13, 19, 13, 2.5, Z_C, 2.5, 6, 7) + _arrow(13, 19, 29.5, 19, Y_C, 2.5, 6, 7)
        + _arrow(13, 19, 3, 29, X_C, 2.5, 6, 7) + _circle(13, 19, 2, INK))
    S["grid"] = _doc(
        _rect(3, 3, 26, 26, WHITE, TB, 1.5)
        + "".join(_line(v, 3, v, 29, F_S, 1.1) + _line(3, v, 29, v, F_S, 1.1) for v in (9.5, 16, 22.5)))
    S["snapshot"] = _doc(
        _path("M10 9L12 5.5H20L22 9Z", TB, TB, 1.3)
        + _rect(3, 9, 26, 18, TB, INK, 1.2, rx=3)
        + _circle(16, 18, 6.5, "#dbe7fb", WHITE, 2) + _circle(16, 18, 2.6, BLUE)
        + _rect(23, 11.5, 3.5, 2, WHITE, rx=0.5))
    S["print"] = _doc(
        _rect(9, 3, 14, 9, WHITE, TB, 1.4)
        + _rect(3, 10, 26, 12, TB, INK, 1.2, rx=2)
        + _rect(8, 17.5, 16, 11.5, WHITE, TB, 1.4)
        + _line(11, 21.5, 21, 21.5, TB_L, 1.4) + _line(11, 25, 18, 25, TB_L, 1.4)
        + _circle(25, 13.8, 1.4, GREEN_L))
    S["show_edges"] = _doc(_cube(16, 16, 13, "#f1f3f5", "#e9ecef", "#dee2e6", BLUE, 2.4))

    def sel(cube):
        return _doc(cube + _cursor(18.5, 16.5, 1.0))

    iso2 = _iso(13.5, 14, 11)
    S["select_domain"] = sel(_cube(13.5, 14, 11, "#a5c3f5", "#6d9bea", BLUE, BLUE_D))
    S["select_boundary"] = sel(_fcube(13.5, 14, 11, right=BLUE))
    S["select_edge"] = sel(_fcube(13.5, 14, 11) + _line(*iso2["UL"], *iso2["C"], BLUE, 3.4)
                           + _line(*iso2["C"], *iso2["UR"], BLUE, 3.4))
    S["select_point"] = sel(_fcube(13.5, 14, 11) + _circle(*iso2["C"], 3.6, BLUE, WHITE, 1.2))
    S["show_geometry"] = _doc(_cube(12, 12.5, 10) + _eye(22.5, 24, 15, BLUE, TB, 1.5))
    S["show_mesh"] = _doc(_square_mesh(scale=0.75, dx=-1, dy=-1, sw=1.1) + _eye(22.5, 24, 15, BLUE, TB, 1.5))
    S["legend"] = _doc(
        _rect(19, 3.5, 8, 25, "url(#lg)", INK, 1.2)
        + "".join(_line(14, y, 18, y, TB, 1.6) + _line(4.5, y, 11.5, y, TB_L, 2) for y in (6, 16, 26)),
        _lin("lg", list(reversed([(1 - o, c) for o, c in RAINBOW])), 0, 0, 0, 1))

    # ================================ windows ================================
    S["model_builder"] = _doc(
        _rect(3.5, 3.5, 7, 7, BLUE, BLUE_D, 1.2, rx=1)
        + _path("M7 10.5V24.5H13M7 17.5H13", "none", TB, 1.5)
        + _rect(13, 14.5, 6, 6, GEO_F, GEO_S, 1.2, rx=1)
        + _rect(13, 21.5, 6, 6, GREEN_L, GREEN, 1.2, rx=1)
        + _line(14, 7, 28, 7, TB, 2.2) + _line(22, 17.5, 28.5, 17.5, TB, 2.2)
        + _line(22, 24.5, 28.5, 24.5, TB, 2.2))
    S["settings_window"] = _doc(
        _window()
        + "".join(_line(6, y + 2.25, 11, y + 2.25, TB, 2) + _rect(13.5, y, 12.5, 4.5, WHITE, F_S, 1.2, rx=1)
                  for y in (13, 20)))
    S["graphics_window"] = _doc(_window() + _cube(16, 18.7, 7.2))
    S["messages"] = _doc(
        _path("M6 5H26Q29 5 29 8V19Q29 22 26 22H14L8 27.5V22H6Q3 22 3 19V8Q3 5 6 5Z", WHITE, TB, 1.6)
        + _line(8, 10.5, 24, 10.5, BLUE, 2.2) + _line(8, 16, 20, 16, TB_L, 2.2))
    S["progress"] = _doc(_window() + _rect(6, 15.5, 20, 7, WHITE, TB, 1.3, rx=1)
                         + _rect(6.7, 16.2, 12, 5.6, GREEN_L))
    S["log"] = _doc(
        _page(5, 3, 22, 26, 6)
        + "".join(_line(8.5, y, 11, y, BLUE, 2) + _line(13, y, 20 if y < 11 else 23.5, y, TB_L, 2)
                  for y in (10, 14.5, 19, 23.5)))
    S["convergence"] = _doc(
        _path("M5 4V27.5H29", "none", TB, 1.7)
        + _line(6, 24, 28.5, 24, GREEN_M, 1.3, dash=DASH)
        + _polyline([(6.5, 6), (9, 10.5), (11, 8.5), (13.5, 14), (15.5, 12.5), (18, 18), (20, 16.5),
                     (23, 22.5), (28, 25)], RED, 2.1))
    S["wizard"] = _doc(
        _line(4.5, 27.5, 16.5, 15.5, INK, 3.8) + _line(14, 18, 16.5, 15.5, WHITE, 1.8)
        + _path(_star_d(22, 10, 7, 3), YELLOW, "#e67700", 1.2)
        + _path(_star_d(27, 22, 3, 0.9, 4), AMBER) + _path(_star_d(10.5, 6, 3, 0.9, 4), AMBER))
    S["physics_list"] = _doc("".join(
        _circle(7, y, 3.4, col) + _line(13, y, 28, y, TB, 2.4)
        for y, col in ((8, HEAT), (16, SOLID), (24, ELEC))))

    def layout_body():
        return (_rect(3, 4.5, 26, 23, WHITE, TB, 1.5, rx=1.5)
                + _rect(3.75, 5.25, 7, 21.5, BLUE_L)
                + _rect(11.75, 21.25, 16.5, 5.5, "#e9ecef")
                + _line(11, 4.5, 11, 27.5, TB, 1.5) + _line(11, 20.5, 29, 20.5, TB, 1.5)
                + _line(5.5, 9, 9, 9, TB, 1.4) + _line(5.5, 12.5, 9, 12.5, TB, 1.4)
                + _line(5.5, 16, 9, 16, TB, 1.4))

    S["layout"] = _doc(layout_body())
    S["reset_desktop"] = _doc(
        _g(layout_body(), "translate(-1,-1.5) scale(0.82)")
        + _circle(23.5, 23.5, 7, WHITE)
        + _arc_arrow(23.5, 23.5, 4.5, -250, 20, BLUE, 2.2, 3.5, 5))
    return S


SVG: dict[str, str] = _build()

_FALLBACK_KEY = "\x00fallback"
_FALLBACK_SVG = _doc(_rect(5, 5, 22, 22, "#ced4da", F_S, 1.5, rx=4))

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_SIZES = (16, 20, 24, 32, 48)
_renderers: dict[str, QSvgRenderer] = {}
_pixmaps: dict[tuple[str, int, float], QPixmap] = {}
_icons: dict[str, QIcon] = {}


def names() -> list[str]:
    """All icon names, in catalogue order."""
    return list(SVG)


def _gui_app() -> QGuiApplication | None:
    app = QGuiApplication.instance()
    return app if isinstance(app, QGuiApplication) else None


def _dpr() -> float:
    app = _gui_app()
    if app is None:
        return 1.0
    try:
        return max(1.0, float(app.devicePixelRatio()))
    except Exception:  # pragma: no cover - defensive
        return 1.0


def _key(name: str) -> str:
    return name if isinstance(name, str) and name in SVG else _FALLBACK_KEY


def _renderer(key: str) -> QSvgRenderer:
    r = _renderers.get(key)
    if r is None:
        r = QSvgRenderer(QByteArray(SVG.get(key, _FALLBACK_SVG).encode("utf-8")))
        if not r.isValid():
            r = QSvgRenderer(QByteArray(_FALLBACK_SVG.encode("utf-8")))
        _renderers[key] = r
    return r


def _render(key: str, size: int, dpr: float) -> QPixmap:
    px = max(1, int(round(size * dpr)))
    pm = QPixmap(px, px)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        _renderer(key).render(painter, QRectF(0, 0, px, px))
    finally:
        painter.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def _cached_pixmap(key: str, size: int, dpr: float) -> QPixmap:
    ck = (key, size, dpr)
    pm = _pixmaps.get(ck)
    if pm is None:
        pm = _render(key, size, dpr)
        _pixmaps[ck] = pm
    return pm


def pixmap(name: str, size: int) -> QPixmap:
    """Return a cached pixmap of *name* at *size* logical pixels.

    Rendered at ``size * devicePixelRatio`` device pixels with the ratio set,
    so it stays crisp on high-DPI screens.  Unknown names give the fallback
    icon.  Requires a QGuiApplication (Qt aborts on QPixmap creation without
    one, so a RuntimeError is raised instead).
    """
    if _gui_app() is None:
        raise RuntimeError("elmerstudio.ui.icons.pixmap() needs a QGuiApplication/QApplication instance")
    try:
        size = max(1, int(size))
    except (TypeError, ValueError):
        size = 16
    return _cached_pixmap(_key(name), size, _dpr())


def icon(name: str) -> QIcon:
    """Return a cached QIcon for *name* (fallback icon for unknown names).

    Pixmaps at 16, 20, 24, 32 and 48 px are added for the current device
    pixel ratio (plus 1.0x copies on high-DPI screens).  Never raises; before
    a QGuiApplication exists an empty, uncached QIcon is returned.
    """
    key = _key(name)
    ic = _icons.get(key)
    if ic is not None:
        return ic
    if _gui_app() is None:
        return QIcon()
    try:
        dpr = _dpr()
        ic = QIcon()
        for s in _SIZES:
            ic.addPixmap(_cached_pixmap(key, s, dpr))
            if dpr != 1.0:
                ic.addPixmap(_cached_pixmap(key, s, 1.0))
    except Exception:  # pragma: no cover - never let an icon break the UI
        ic = QIcon()
    _icons[key] = ic
    return ic
