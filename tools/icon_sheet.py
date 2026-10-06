"""Render the Elmer Studio icon set into a labeled contact sheet and validate it.

Usage::

    python tools/icon_sheet.py                 # writes tools/icon_sheet.png
    python tools/icon_sheet.py --zoom 3 --range 0:40 --out some.png   # close-up page

Checks (exit status 1 on failure):
  * every required icon name exists in ``icons.SVG``
  * every SVG is a 32x32-viewBox document that QSvgRenderer accepts
  * no icon renders blank at 16 px
  * unknown names fall back without raising; ``icon()`` carries 16..48 px sizes
Warnings (do not fail): icons whose 32 px rendering touches the outer pixel ring
(possible clipping).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# The offscreen platform ships no font database on Windows; point it at the
# system fonts so the sheet labels are not drawn as empty boxes.
if os.name == "nt" and os.environ["QT_QPA_PLATFORM"] == "offscreen":
    _fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    if _fonts.is_dir():
        os.environ.setdefault("QT_QPA_FONTDIR", str(_fonts))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QByteArray, QRect, QSize, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

REQUIRED = """
new open save save_as undo redo delete copy paste duplicate rename preferences help app close
folder run stop refresh info warning error check cross add remove clear active up down
collapse_all expand_all build_selected build_all python sif back forward show hide enable disable
root global_defs parameters variables functions component definitions selection_node geometry
block sphere cylinder cone torus rectangle circle polygon point interval ellipse union difference
intersection form_union form_assembly import export move rotate scale mirror array transform_copy
fillet chamfer extrude revolve work_plane materials material physics physics_heat physics_solid
physics_es physics_ec physics_mf physics_spf physics_acpr physics_tds physics_pde multiphysics
feature_domain feature_boundary feature_edge feature_point initial_values default_feature mesh
mesh_size free_tet free_tri swept boundary_layer distribution mesh_stats study compute stationary
time_dependent eigenfrequency frequency_domain parametric_sweep solver solver_config results
dataset solution plot_group_3d plot_group_2d plot_group_1d surface_plot volume_plot slice_plot
isosurface arrow_plot streamline contour line_graph deformation mesh_plot derived_values
global_evaluation point_evaluation integration average maxmin table image_export data_export
animation report cut_line cut_plane cut_point plot
zoom_in zoom_out zoom_box zoom_extents view_default view_xy view_yz view_zx view_iso perspective
orthographic select_box select_all clear_selection transparency wireframe scene_light axes grid
snapshot print show_edges select_domain select_boundary select_edge select_point show_geometry
show_mesh legend
model_builder settings_window graphics_window messages progress log convergence wizard
physics_list layout reset_desktop
""".split()

CELL_W, CELL_H, HEADER = 196, 46, 34


def alpha_stats(img: QImage):
    """Return (opaque pixel count, touches-outer-ring flag) for a square image."""
    img = img.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = img.width(), img.height()
    count, edge = 0, False
    for y in range(h):
        for x in range(w):
            a = img.pixelColor(x, y).alpha()
            if a > 24:
                count += 1
                if a > 64 and (x == 0 or y == 0 or x == w - 1 or y == h - 1):
                    edge = True
    return count, edge


def validate(icons) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = [n for n in REQUIRED if n not in icons.SVG]
    if missing:
        errors.append(f"missing required icons: {', '.join(missing)}")
    if len(set(REQUIRED)) != len(REQUIRED):
        errors.append("REQUIRED list has duplicates")
    for name, svg in icons.SVG.items():
        if not svg.lstrip().startswith("<svg") or 'viewBox="0 0 32 32"' not in svg:
            errors.append(f"{name}: not a 32x32 viewBox SVG document")
        if "<text" in svg:
            warnings.append(f"{name}: uses <text>; rendering depends on installed fonts")
        if not QSvgRenderer(QByteArray(svg.encode("utf-8"))).isValid():
            errors.append(f"{name}: QSvgRenderer.isValid() is False")
            continue
        n16, _ = alpha_stats(icons.pixmap(name, 16).toImage())
        if n16 < 12:
            errors.append(f"{name}: blank or nearly blank at 16 px ({n16} px)")
        _, edge = alpha_stats(icons.pixmap(name, 32).toImage())
        if edge:
            warnings.append(f"{name}: touches the outer pixel ring at 32 px (possible clipping)")
    try:
        fb = icons.icon("__definitely_not_an_icon__")
        if fb.isNull():
            errors.append("fallback icon is null")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"icon() raised for an unknown name: {exc!r}")
    first = icons.names()[0]
    ic = icons.icon(first)
    sizes = {(s.width(), s.height()) for s in ic.availableSizes()}
    for s in (16, 20, 24, 32, 48):
        if (s, s) not in sizes:
            errors.append(f"icon('{first}') lacks a {s}px pixmap (has {sorted(sizes)})")
    if icons.icon(first) is not ic:
        errors.append("icon() is not cached (returned a different object)")
    if icons.pixmap(first, 16) is not icons.pixmap(first, 16):
        errors.append("pixmap() is not cached (returned a different object)")
    return errors, warnings


def draw_sheet(icons, names: list[str], cols: int, zoom: int, out: Path) -> None:
    rows = max(1, math.ceil(len(names) / cols))
    w, h = cols * CELL_W + 12, HEADER + rows * CELL_H + 8
    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(QColor("#ffffff"))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    title = QFont("Segoe UI")
    title.setPixelSize(15)
    title.setBold(True)
    p.setFont(title)
    p.setPen(QColor("#212529"))
    p.drawText(QRect(8, 4, w - 16, HEADER - 8), Qt.AlignmentFlag.AlignVCenter,
               f"Elmer Studio icons - {len(names)} shown of {len(icons.SVG)}  (16 px | 32 px)")
    label = QFont("Segoe UI")
    label.setPixelSize(11)
    p.setFont(label)
    for k, name in enumerate(names):
        r, c = divmod(k, cols)
        x, y = 6 + c * CELL_W, HEADER + r * CELL_H
        if r % 2 == 0:
            p.fillRect(x, y, CELL_W, CELL_H, QColor("#f8f9fa"))
        p.drawPixmap(x + 6, y + 15, icons.pixmap(name, 16))
        p.drawPixmap(x + 28, y + 7, icons.pixmap(name, 32))
        p.setPen(QColor("#343a40"))
        p.drawText(QRect(x + 66, y, CELL_W - 70, CELL_H), Qt.AlignmentFlag.AlignVCenter, name)
    p.end()
    if zoom > 1:
        img = img.scaled(QSize(w * zoom, h * zoom), Qt.AspectRatioMode.IgnoreAspectRatio,
                         Qt.TransformationMode.FastTransformation)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not img.save(str(out)):
        raise SystemExit(f"could not write {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("icon_sheet.png"))
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--zoom", type=int, default=1, help="nearest-neighbour upscale for close review")
    ap.add_argument("--range", default="", help="slice of names() to draw, e.g. 0:40")
    ap.add_argument("--only", default="", help="comma-separated names to draw")
    args = ap.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])  # noqa: F841 - keep alive
    from elmerstudio.ui import icons

    errors, warnings = validate(icons)
    names = icons.names()
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
    elif args.range:
        a, _, b = args.range.partition(":")
        names = names[int(a or 0):int(b) if b else None]
    draw_sheet(icons, names, args.cols, args.zoom, args.out)

    extra = sorted(set(icons.SVG) - set(REQUIRED))
    print(f"icons: {len(icons.SVG)} (required {len(REQUIRED)}, extra {len(extra)}{': ' + ', '.join(extra) if extra else ''})")
    print(f"sheet: {args.out.resolve()}")
    for wmsg in warnings:
        print("WARN ", wmsg)
    for emsg in errors:
        print("ERROR", emsg)
    print("OK" if not errors else f"FAILED ({len(errors)} errors)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
