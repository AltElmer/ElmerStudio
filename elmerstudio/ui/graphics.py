"""Graphics window: VTK rendering of geometry, mesh and result scenes with entity picking."""
from __future__ import annotations

import os
import time

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QMenu, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from . import icons

FACE = np.array([205, 207, 212], np.uint8)
FACE_2D = np.array([214, 217, 222], np.uint8)
SELECTED = np.array([38, 92, 225], np.uint8)
HOVER = np.array([226, 46, 46], np.uint8)
LISTED = np.array([90, 190, 245], np.uint8)
EDGE = np.array([20, 20, 24], np.uint8)
MESH_FACE = (0.78, 0.86, 0.94)
BG_BOTTOM = (0.78, 0.82, 0.88)
BG_TOP = (1.0, 1.0, 1.0)


class Scene:
    """Result scene description produced by the plot engine."""

    def __init__(self):
        self.items: list[tuple] = []        # (dataset, add_mesh kwargs)
        self.legends: list[dict] = []       # {cmap, clim, title}
        self.title = ""
        self.sdim = 3
        self.texts: list[tuple] = []


class GraphicsView(QWidget):
    entityClicked = Signal(str, int, bool)    # level, number, ctrl
    status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ok = True
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.toolbar = QWidget(self)
        self.toolbar.setObjectName("GraphicsToolbar")
        self.tb = QHBoxLayout(self.toolbar)
        self.tb.setContentsMargins(4, 2, 4, 2)
        self.tb.setSpacing(1)
        v.addWidget(self.toolbar)
        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor
            pv.global_theme.allow_empty_mesh = True
            self.plotter = QtInteractor(self, auto_update=False)
            self.plotter.enable_anti_aliasing("ssaa") if hasattr(self.plotter, "enable_anti_aliasing") else None
            v.addWidget(self.plotter.interactor, 1)
        except Exception as exc:  # no usable OpenGL
            self._ok = False
            self.plotter = None
            lab = QLabel(f"3D graphics unavailable ({exc}).\nSee README: software OpenGL (Mesa) fallback.")
            lab.setAlignment(Qt.AlignCenter)
            v.addWidget(lab, 1)
        self.mode = "empty"
        self.sdim = 3
        self.geo = None
        self.level = None           # active selection level
        self.selected: set[int] = set()
        self.listed: set[int] = set()
        self.hover: int | None = None
        self.material_colors: dict | None = None
        self.transparent = False
        self.wireframe = False
        self._actors: dict = {}
        self._face_cells = None
        self._press = None
        self._last_hover = 0.0
        self._cam_key = None
        self._build_toolbar()
        if self._ok:
            self._setup_vtk()

    # ------------------------------------------------------------------ toolbar
    def _tool(self, icon_name, tip, cb, checkable=False):
        b = QToolButton(self.toolbar)
        b.setIcon(icons.icon(icon_name))
        b.setIconSize(QSize(16, 16))
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.setCheckable(checkable)
        if checkable:
            b.toggled.connect(cb)
        else:
            b.clicked.connect(cb)
        self.tb.addWidget(b)
        return b

    def _sep(self):
        s = QLabel("|")
        s.setStyleSheet("color:#c9cdd2; padding: 0 3px;")
        self.tb.addWidget(s)

    def _build_toolbar(self):
        self._tool("zoom_in", "Zoom In", lambda: self.zoom(1.25))
        self._tool("zoom_out", "Zoom Out", lambda: self.zoom(0.8))
        self._tool("zoom_box", "Zoom Box", self.zoom_box)
        self._tool("zoom_extents", "Zoom Extents (F5)", self.zoom_extents)
        self._sep()
        self.b_default = self._tool("view_default", "Go to Default View", self.default_view)
        self.b_xy = self._tool("view_xy", "Go to XY View", lambda: self.view("xy"))
        self.b_yz = self._tool("view_yz", "Go to YZ View", lambda: self.view("yz"))
        self.b_zx = self._tool("view_zx", "Go to ZX View", lambda: self.view("zx"))
        self.b_persp = self._tool("perspective", "Orthographic/Perspective Projection", self.toggle_projection, True)
        self._sep()
        self._tool("select_all", "Select All", self.select_all)
        self._tool("clear_selection", "Clear Selection", self.clear_selection)
        self._sep()
        self.b_transp = self._tool("transparency", "Transparency", self.set_transparency, True)
        self.b_wire = self._tool("wireframe", "Wireframe Rendering", self.set_wireframe, True)
        self.b_light = self._tool("scene_light", "Scene Light", self.set_scene_light, True)
        self.b_light.setChecked(True)
        self.b_axes = self._tool("axes", "Show Axis Orientation", self.set_axes, True)
        self.b_axes.setChecked(True)
        self.b_grid = self._tool("grid", "Show Grid", self.set_grid, True)
        self._sep()
        self._tool("snapshot", "Image Snapshot", self.snapshot_dialog)
        self.tb.addStretch(1)
        self.info = QLabel("")
        self.info.setStyleSheet("color:#555; padding-right:6px;")
        self.tb.addWidget(self.info)

    # ------------------------------------------------------------------ vtk setup
    def _setup_vtk(self):
        import vtk
        p = self.plotter
        p.set_background(BG_BOTTOM, top=BG_TOP)
        self._axes_on = True
        p.add_axes(interactive=False, line_width=2, labels_off=False, color="black",
                   x_color=(0.85, 0.15, 0.15), y_color=(0.15, 0.6, 0.15), z_color=(0.15, 0.3, 0.85))
        style = vtk.vtkInteractorStyleTrackballCamera()
        style.AddObserver("LeftButtonPressEvent", self._lpress)
        style.AddObserver("LeftButtonReleaseEvent", self._lrelease)
        style.AddObserver("RightButtonPressEvent", self._rpress)
        style.AddObserver("RightButtonReleaseEvent", self._rrelease)
        style.AddObserver("MouseMoveEvent", self._mmove)
        self.style = style
        p.iren.interactor.SetInteractorStyle(style)
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.004)
        self.grid_on = False

    # ------------------------------------------------------------------ mouse
    def _pos(self):
        return self.plotter.iren.interactor.GetEventPosition()

    def _lpress(self, obj, ev):
        self._press = self._pos()
        if self.sdim == 2:
            obj.StartPan()
        else:
            obj.OnLeftButtonDown()

    def _lrelease(self, obj, ev):
        if self.sdim == 2:
            obj.EndPan()
        else:
            obj.OnLeftButtonUp()
        pos = self._pos()
        if self._press is not None and abs(pos[0] - self._press[0]) + abs(pos[1] - self._press[1]) < 4:
            ctrl = bool(self.plotter.iren.interactor.GetControlKey())
            self._click(pos, ctrl)
        self._press = None

    def _rpress(self, obj, ev):
        self._rpress_pos = self._pos()
        obj.StartPan()

    def _rrelease(self, obj, ev):
        obj.EndPan()

    def _mmove(self, obj, ev):
        obj.OnMouseMove()
        if self._press is None and self.level and self.mode == "geometry":
            now = time.time()
            if now - self._last_hover > 0.04:
                self._last_hover = now
                h = self._pick(self._pos())
                if h != self.hover:
                    self.hover = h
                    self._recolor()
                    if h is not None:
                        self.status.emit(f"{self.level.capitalize()} {h}")

    # ------------------------------------------------------------------ picking
    def _pick(self, pos) -> int | None:
        if self.geo is None or not self.level:
            return None
        lvl = self.level
        sd = self.sdim
        if lvl == "point":
            return self._pick_point(pos)
        if (sd == 3 and lvl in ("domain", "boundary")) or (sd == 2 and lvl == "domain"):
            act = self._actors.get("faces")
            arr = self._face_ids
        else:
            act = self._actors.get("curves")
            arr = self._curve_ids
        if act is None:
            return None
        self.picker.InitializePickList()
        self.picker.AddPickList(act)
        self.picker.PickFromListOn()
        if not self.picker.Pick(pos[0], pos[1], 0, self.plotter.renderer):
            return None
        cid = self.picker.GetCellId()
        if cid < 0 or cid >= len(arr):
            return None
        ent = int(arr[cid])
        if sd == 3 and lvl == "domain":
            doms = self.geo.adj_up["boundary"].get(ent, [])
            if not doms:
                return None
            if len(doms) > 1 and self.hover in doms:
                return self.hover
            return doms[0]
        return ent

    def _pick_point(self, pos):
        pts = self.geo.display.get("point", {})
        if not pts:
            return None
        r = self.plotter.renderer
        best, bd = None, 12.0
        for num, xyz in pts.items():
            r.SetWorldPoint(float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0)
            r.WorldToDisplay()
            d = r.GetDisplayPoint()
            dist = abs(d[0] - pos[0]) + abs(d[1] - pos[1])
            if dist < bd:
                best, bd = num, dist
        return best

    def _click(self, pos, ctrl):
        if self.mode != "geometry" or not self.level:
            return
        ent = self._pick(pos)
        if ent is not None:
            self.entityClicked.emit(self.level, ent, ctrl)

    # ------------------------------------------------------------------ geometry scene
    def show_geometry(self, geo, level=None, selected=(), material_colors=None, reset_camera=None):
        if not self._ok:
            return
        import pyvista as pv
        self.mode = "geometry"
        self.geo = geo
        self.sdim = geo.sdim
        self.level = level
        self.selected = set(selected)
        self.material_colors = material_colors
        self.hover = None
        self.listed = set()
        p = self.plotter
        self._clear()
        disp = geo.display
        sd = geo.sdim
        face_level = "boundary" if sd == 3 else "domain"
        curve_level = "edge" if sd == 3 else "boundary"
        # faces
        pts, tris, ids = [], [], []
        off = 0
        for num, (P, T) in sorted(disp.get(face_level, {}).items()):
            pts.append(P)
            tris.append(T + off)
            ids.append(np.full(len(T), num))
            off += len(P)
        if pts:
            P = np.vstack(pts)
            T = np.vstack(tris)
            faces = np.hstack([np.full((len(T), 1), 3), T]).ravel()
            mesh = pv.PolyData(P, faces)
            self._face_ids = np.concatenate(ids)
            mesh.cell_data["rgb"] = np.tile(FACE if sd == 3 else FACE_2D, (mesh.n_cells, 1))
            mesh = mesh.compute_normals(cell_normals=False, split_vertices=True, feature_angle=35) if sd == 3 else mesh
            self._face_mesh = mesh
            a = p.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=sd == 3, show_scalar_bar=False,
                           ambient=0.25, diffuse=0.8, specular=0.25, specular_power=30, pickable=True,
                           opacity=0.35 if self.transparent else 1.0, name="faces")
            if sd == 2:
                a.GetProperty().LightingOff()
            self._actors["faces"] = a
        else:
            self._face_ids = np.zeros(0, int)
        # curves
        pts, segs, ids = [], [], []
        off = 0
        for num, (P, S) in sorted(disp.get(curve_level, {}).items()):
            pts.append(P)
            segs.append(S + off)
            ids.append(np.full(len(S), num))
            off += len(P)
        if pts:
            P = np.vstack(pts)
            S = np.vstack(segs)
            lines = np.hstack([np.full((len(S), 1), 2), S]).ravel()
            cm = pv.PolyData(P, lines=lines)
            self._curve_ids = np.concatenate(ids)
            cm.cell_data["rgb"] = np.tile(EDGE, (cm.n_cells, 1))
            self._curve_mesh = cm
            a = p.add_mesh(cm, scalars="rgb", rgb=True, line_width=1.6 if sd == 3 else 2.0, show_scalar_bar=False,
                           pickable=True, name="curves", lighting=False)
            self._actors["curves"] = a
        else:
            self._curve_ids = np.zeros(0, int)
        # points
        pd = disp.get("point", {})
        if pd:
            P = np.array([pd[k] for k in sorted(pd)], float)
            self._point_ids = np.array(sorted(pd))
            pm = pv.PolyData(P)
            pm.point_data["rgb"] = np.tile(EDGE, (len(P), 1))
            self._point_mesh = pm
            a = p.add_mesh(pm, scalars="rgb", rgb=True, point_size=5, render_points_as_spheres=True,
                           show_scalar_bar=False, pickable=False, name="points", lighting=False)
            self._actors["points"] = a
        self._recolor(render=False)
        self._apply_wire()
        self._camera_for(geo.bbox, sd, reset_camera)
        self._update_grid()
        p.render()

    def set_selection_state(self, level, selected, listed=()):
        self.level = level
        self.selected = set(selected)
        self.listed = set(listed)
        self.hover = None
        self._recolor()

    def set_material_colors(self, colors):
        self.material_colors = colors
        self._recolor()

    def _recolor(self, render=True):
        if self.mode != "geometry" or self.geo is None:
            return
        sd = self.sdim
        lvl = self.level
        sel, hov, lst = self.selected, self.hover, self.listed
        # faces
        fm = getattr(self, "_face_mesh", None)
        if fm is not None and len(self._face_ids):
            base = FACE if sd == 3 else FACE_2D
            rgb = np.tile(base, (len(self._face_ids), 1))
            ids = self._face_ids
            if self.material_colors:
                for dom, col in self.material_colors.items():
                    if sd == 3:
                        bnds = set(self.geo.adj_down["domain"].get(dom, []))
                        m = np.isin(ids, list(bnds))
                    else:
                        m = ids == dom
                    rgb[m] = np.array(col, np.uint8)
            if sd == 3 and lvl == "domain":
                def faces_of(doms):
                    b = set()
                    for d in doms:
                        b.update(self.geo.adj_down["domain"].get(d, []))
                    return list(b)
                rgb[np.isin(ids, faces_of(lst))] = LISTED
                rgb[np.isin(ids, faces_of(sel))] = SELECTED
                if hov is not None:
                    rgb[np.isin(ids, faces_of([hov]))] = HOVER
            elif (sd == 3 and lvl == "boundary") or (sd == 2 and lvl == "domain"):
                rgb[np.isin(ids, list(lst))] = LISTED
                rgb[np.isin(ids, list(sel))] = SELECTED
                if hov is not None:
                    rgb[ids == hov] = HOVER
            fm.cell_data["rgb"] = rgb
        cm = getattr(self, "_curve_mesh", None)
        if cm is not None and len(self._curve_ids):
            rgb = np.tile(EDGE, (len(self._curve_ids), 1))
            ids = self._curve_ids
            if (sd == 3 and lvl == "edge") or (sd == 2 and lvl == "boundary"):
                rgb[np.isin(ids, list(lst))] = LISTED
                rgb[np.isin(ids, list(sel))] = SELECTED
                if hov is not None:
                    rgb[ids == hov] = HOVER
            cm.cell_data["rgb"] = rgb
            a = self._actors.get("curves")
            if a is not None:
                wide = (sd == 3 and lvl == "edge") or (sd == 2 and lvl == "boundary")
                a.GetProperty().SetLineWidth(3.0 if wide else (1.6 if sd == 3 else 2.0))
        pm = getattr(self, "_point_mesh", None)
        if pm is not None:
            rgb = np.tile(EDGE, (pm.n_points, 1))
            if lvl == "point":
                rgb[np.isin(self._point_ids, list(sel))] = SELECTED
                if hov is not None:
                    rgb[self._point_ids == hov] = HOVER
            pm.point_data["rgb"] = rgb
            a = self._actors.get("points")
            if a is not None:
                a.GetProperty().SetPointSize(9 if lvl == "point" else 5)
        if render:
            self.plotter.render()

    # ------------------------------------------------------------------ mesh scene
    def show_mesh(self, mesh_result, quality=False, reset_camera=None):
        if not self._ok:
            return
        import pyvista as pv
        from ..core.results import Topology
        self.mode = "mesh"
        self.level = None
        self.sdim = mesh_result.sdim
        p = self.plotter
        self._clear()
        topo = Topology(mesh_result=mesh_result)
        if self.sdim == 3:
            g = topo.boundary_grid()
            surf = g.extract_surface().triangulate() if g.n_cells else g
            if quality:
                bulk = topo.bulk_grid()
                sh = bulk.extract_surface()
                p.add_mesh(sh, scalars="quality", cmap=_cmap("Traffic", True), clim=(0, 1), show_edges=True,
                           edge_color=(0.1, 0.1, 0.12), line_width=0.5, show_scalar_bar=False, name="mesh")
                self._legend(_cmap("Traffic", True), (0, 1), "Element quality")
            else:
                p.add_mesh(g.extract_surface(nonlinear_subdivision=1) if g.n_cells else g, color=MESH_FACE,
                           show_edges=True, edge_color=(0.12, 0.12, 0.15), line_width=0.6, ambient=0.3,
                           diffuse=0.75, name="mesh")
        else:
            g = topo.bulk_grid()
            if quality:
                p.add_mesh(g, scalars="quality", cmap=_cmap("Traffic", True), clim=(0, 1), show_edges=True,
                           edge_color=(0.1, 0.1, 0.12), line_width=0.6, show_scalar_bar=False, lighting=False, name="mesh")
                self._legend(_cmap("Traffic", True), (0, 1), "Element quality")
            else:
                p.add_mesh(g.extract_surface(nonlinear_subdivision=1), color=MESH_FACE, show_edges=True,
                           edge_color=(0.12, 0.12, 0.15), line_width=0.7, lighting=False, name="mesh")
            b = topo.boundary_grid()
            if b.n_cells:
                p.add_mesh(b, color="black", line_width=2, lighting=False, name="mesh_bnd")
        n = sum(len(c) for blocks in mesh_result.bulk.values() for _, c in blocks)
        self.info.setText(f"{n} elements")
        P = mesh_result.nodes
        bb = (*P.min(axis=0), *P.max(axis=0))
        self._camera_for(bb, self.sdim, reset_camera)
        self._update_grid()
        p.render()

    # ------------------------------------------------------------------ results scene
    def show_scene(self, scene: Scene, reset_camera=None):
        if not self._ok:
            return
        self.mode = "results"
        self.level = None
        self.sdim = scene.sdim
        p = self.plotter
        self._clear()
        bounds = None
        for ds, kw in scene.items:
            kw = dict(kw)
            kw.setdefault("show_scalar_bar", False)
            try:
                p.add_mesh(ds, **kw)
            except Exception as exc:
                self.status.emit(f"Plot error: {exc}")
                continue
            if ds is not None and getattr(ds, "n_points", 1):
                b = ds.bounds
                bounds = b if bounds is None else (min(bounds[0], b[0]), max(bounds[1], b[1]), min(bounds[2], b[2]),
                                                   max(bounds[3], b[3]), min(bounds[4], b[4]), max(bounds[5], b[5]))
        for i, lg in enumerate(scene.legends[:3]):
            self._legend(lg["cmap"], lg["clim"], lg.get("title", ""), index=i, n_labels=lg.get("n_labels", 6))
        if scene.title:
            import vtk
            t = vtk.vtkTextActor()
            t.SetInput(scene.title)
            _font(t.GetTextProperty(), 12)
            t.GetTextProperty().SetVerticalJustificationToTop()
            t.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            t.SetPosition(0.008, 0.985)
            p.add_actor(t, name="title")
        self.info.setText("")
        if bounds is not None:
            bb = (bounds[0], bounds[2], bounds[4], bounds[1], bounds[3], bounds[5])
            self._camera_for(bb, self.sdim, reset_camera)
        self._update_grid()
        p.render()

    def _legend(self, cmap, clim, title, index=0, n_labels=6):
        """COMSOL-style color legend: bar with nice ticks, x10^n multiplier, max/min markers."""
        import math

        import vtk
        p = self.plotter
        lo, hi = float(clim[0]), float(clim[1])
        if hi <= lo:
            hi = lo + max(abs(lo) * 1e-6, 1e-30)
        big = max(abs(lo), abs(hi))
        exp = int(math.floor(math.log10(big))) if big > 0 else 0
        exp = exp if (exp >= 4 or exp <= -3) else 0
        sc = 10.0 ** (-exp)
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetRange(lo * sc, hi * sc)
        for i in range(256):
            r, g, b, _a = cmap(i / 255.0)
            lut.SetTableValue(i, r, g, b, 1.0)
        lut.Build()
        sb = vtk.vtkScalarBarActor()
        sb.SetLookupTable(lut)
        sb.SetOrientationToVertical()
        ticks = _nice_ticks(lo * sc, hi * sc, n_labels)
        if hasattr(sb, "SetUseCustomLabels") and len(ticks) >= 2:
            arr = vtk.vtkDoubleArray()
            for t in ticks:
                arr.InsertNextValue(t)
            sb.SetUseCustomLabels(True)
            sb.SetCustomLabels(arr)
            sb.SetAnnotationTextScaling(False)
        else:
            sb.SetNumberOfLabels(n_labels)
        step = (ticks[1] - ticks[0]) if len(ticks) > 1 else (hi - lo) * sc
        dec = max(0, -int(math.floor(math.log10(abs(step)))) if step else 0)
        sb.SetLabelFormat(f"%.{min(dec, 6)}f" if dec < 7 else "%.3g")
        x0 = 0.905 - 0.09 * index
        sb.SetBarRatio(0.3)
        sb.SetPosition(x0, 0.12)
        sb.SetWidth(0.085)
        sb.SetHeight(0.70)
        sb.SetTextPositionToSucceedScalarBar()
        sb.SetMaximumWidthInPixels(105)
        sb.DrawTickLabelsOn()
        sb.DrawFrameOff()
        sb.UnconstrainedFontSizeOn()
        for tp in (sb.GetLabelTextProperty(), sb.GetTitleTextProperty()):
            _font(tp, 10)
        sb.SetTitle("")
        p.add_actor(sb, name=f"legend{index}")
        texts = [(f"▲ {_fmt(hi)}", 0.855), (f"▼ {_fmt(lo)}", 0.07)]
        if exp:
            texts.append((f"×10{_sup(exp)}", 0.825))
        for k, (s, y) in enumerate(texts):
            t = vtk.vtkTextActor()
            t.SetInput(s)
            _font(t.GetTextProperty(), 10)
            t.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            t.SetPosition(x0 + 0.004, y)
            p.add_actor(t, name=f"legend{index}_{k}")

    # ------------------------------------------------------------------ helpers
    def _clear(self):
        p = self.plotter
        p.clear_actors()
        for nm in list(getattr(p, "_scalar_bars", {}).keys() if hasattr(p, "_scalar_bars") else []):
            try:
                p.remove_scalar_bar(nm)
            except Exception:
                pass
        try:
            p.remove_actor("title")
        except Exception:
            pass
        self._actors.clear()
        self._face_mesh = None
        self._curve_mesh = None
        self._point_mesh = None
        self._face_ids = np.zeros(0, int)
        self._curve_ids = np.zeros(0, int)

    def clear(self):
        if self._ok:
            self.mode = "empty"
            self._clear()
            self.plotter.render()

    def _camera_for(self, bb, sdim, reset):
        key = (sdim, tuple(round(float(v), 6) for v in bb))
        p = self.plotter
        if sdim == 2:
            p.enable_parallel_projection()
            self.b_persp.setEnabled(False)
            for b in (self.b_yz, self.b_zx):
                b.setEnabled(False)
        else:
            self.b_persp.setEnabled(True)
            for b in (self.b_yz, self.b_zx):
                b.setEnabled(True)
        if reset or (reset is None and key != self._cam_key and (self._cam_key is None or self._cam_key[0] != sdim
                                                                  or _bbox_changed(self._cam_key[1], key[1]))):
            if sdim == 2:
                p.view_xy()
            else:
                self._iso()
            p.reset_camera()
            p.camera.zoom(0.92)
        self._cam_key = key

    def _iso(self):
        p = self.plotter
        p.view_isometric()
        cam = p.camera
        # COMSOL-like default: looking from (+x?, -y, +z) slightly from the front
        fp = np.array(cam.focal_point)
        d = np.linalg.norm(np.array(cam.position) - fp)
        direction = np.array([-0.62, -0.95, 0.75])
        direction /= np.linalg.norm(direction)
        cam.position = tuple(fp + direction * d)
        cam.up = (0, 0, 1)

    # ------------------------------------------------------------------ toolbar actions
    def zoom(self, f):
        if self._ok:
            self.plotter.camera.zoom(f)
            self.plotter.render()

    def zoom_box(self):
        if not self._ok:
            return
        import vtk
        st = vtk.vtkInteractorStyleRubberBandZoom()
        iren = self.plotter.iren.interactor

        def done(obj, ev):
            QTimer.singleShot(0, lambda: iren.SetInteractorStyle(self.style))
        st.AddObserver("LeftButtonReleaseEvent", lambda o, e: (o.OnLeftButtonUp(), done(o, e)))
        iren.SetInteractorStyle(st)

    def zoom_extents(self):
        if self._ok:
            self.plotter.reset_camera()
            self.plotter.camera.zoom(0.92)
            self.plotter.render()

    def default_view(self):
        if not self._ok:
            return
        if self.sdim == 2:
            self.plotter.view_xy()
        else:
            self._iso()
        self.plotter.reset_camera()
        self.plotter.camera.zoom(0.92)
        self.plotter.render()

    def view(self, which):
        if not self._ok:
            return
        getattr(self.plotter, f"view_{which}")()
        self.plotter.reset_camera()
        self.plotter.render()

    def toggle_projection(self, persp):
        if not self._ok or self.sdim == 2:
            return
        if persp:
            self.plotter.disable_parallel_projection()
        else:
            self.plotter.enable_parallel_projection()
        self.plotter.render()

    def select_all(self):
        if self.geo is not None and self.level:
            for e in self.geo.numbers.get(self.level, []):
                if e not in self.selected:
                    self.entityClicked.emit(self.level, e, True)

    def clear_selection(self):
        if self.level:
            for e in list(self.selected):
                self.entityClicked.emit(self.level, e, True)

    def set_transparency(self, on):
        self.transparent = on
        a = self._actors.get("faces")
        if a is not None:
            a.GetProperty().SetOpacity(0.35 if on else 1.0)
            self.plotter.render()

    def set_wireframe(self, on):
        self.wireframe = on
        self._apply_wire()
        if self._ok:
            self.plotter.render()

    def _apply_wire(self):
        a = self._actors.get("faces")
        if a is not None:
            a.SetVisibility(not self.wireframe)

    def set_scene_light(self, on):
        if not self._ok:
            return
        self.lighting = on
        acts = self.plotter.renderer.GetActors()
        acts.InitTraversal()
        for _ in range(acts.GetNumberOfItems()):
            a = acts.GetNextActor()
            if a is not None and a.GetProperty().GetRepresentation() == 2:
                a.GetProperty().SetLighting(on)
        self.plotter.render()

    def set_axes(self, on):
        if not self._ok:
            return
        if on:
            self.plotter.show_axes()
        else:
            self.plotter.hide_axes()
        self.plotter.render()

    def set_grid(self, on):
        self.grid_on = on
        self._update_grid()
        if self._ok:
            self.plotter.render()

    def _update_grid(self):
        if not self._ok:
            return
        p = self.plotter
        try:
            p.remove_bounds_axes()
        except Exception:
            pass
        if getattr(self, "grid_on", False) or (self.sdim == 2 and self.mode in ("geometry", "results", "mesh")):
            try:
                p.show_bounds(grid=getattr(self, "grid_on", False), location="outer", ticks="outside",
                              xtitle=" ", ytitle=" ", ztitle=" ", color=(0.25, 0.25, 0.28), font_size=9,
                              show_zaxis=self.sdim == 3, show_zlabels=self.sdim == 3, n_xlabels=6, n_ylabels=5,
                              fmt="%.3g", use_2d=self.sdim == 2)
            except Exception:
                pass

    def snapshot(self, path, w=None, h=None):
        if not self._ok:
            return
        size = (w, h) if w and h else None
        self.plotter.screenshot(path, window_size=size)

    def snapshot_dialog(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Image Snapshot", "snapshot.png", "PNG image (*.png)")
        if fn:
            self.snapshot(fn)
            self.status.emit(f"Image saved: {fn}")


_FONT_FILE = None


def _font(tp, size):
    """DejaVu Sans (shipped with matplotlib on every platform) so glyphs like ▲▼×² render everywhere."""
    global _FONT_FILE
    import vtk
    if _FONT_FILE is None:
        try:
            import matplotlib
            f = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
            _FONT_FILE = f if os.path.isfile(f) else ""
        except Exception:
            _FONT_FILE = ""
    tp.SetColor(0.1, 0.1, 0.12)
    tp.ShadowOff()
    tp.BoldOff()
    tp.ItalicOff()
    tp.SetFontSize(size)
    if _FONT_FILE:
        tp.SetFontFamily(vtk.VTK_FONT_FILE)
        tp.SetFontFile(_FONT_FILE)
    else:
        tp.SetFontFamilyToArial()


def _nice_ticks(lo, hi, n=6):
    import math
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step - 1e-9) * step
    out = []
    v = start
    while v <= hi + 1e-9 * step:
        out.append(round(v, 12))
        v += step
    return out


def _fmt(v):
    if v == 0:
        return "0"
    return f"{v:.4g}" if 1e-3 <= abs(v) < 1e5 else f"{v:.3e}"


def _sup(n):
    return str(n).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def _bbox_changed(a, b):
    """True when the model extent changed enough that the camera should be re-fitted."""
    a, b = np.array(a), np.array(b)
    return np.max(np.abs(a - b)) > 0.25 * np.max(np.abs(a[3:] - a[:3]) + 1e-30)


def _cmap(name, reverse=False):
    from ..core.results import colormap
    return colormap(name, reverse)
