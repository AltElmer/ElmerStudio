"""Plot engine: plot-group nodes -> VTK scenes (2D/3D) or matplotlib figures (1D, polar); derived values."""
from __future__ import annotations

import math
import re

import numpy as np

from ..core import units
from ..core.model import Node
from ..core.results import Solution, auto_scale_deformation, colormap
from .graphics import Scene
from .settings import pretty_unit

PLOT_NAMES = {"plot.surface": "Surface", "plot.volume": "Volume", "plot.slice": "Slice", "plot.multislice": "Multislice",
              "plot.isosurface": "Isosurface", "plot.contour": "Contour", "plot.arrow": "Arrow Volume",
              "plot.streamline": "Streamline", "plot.mesh": "Mesh", "plot.line": "Line", "plot.annotation": "Annotation"}
COLORS = {"black": (0, 0, 0), "white": (1, 1, 1), "gray": (0.5, 0.5, 0.5), "red": (0.85, 0.1, 0.1),
          "green": (0.1, 0.6, 0.15), "blue": (0.1, 0.25, 0.85), "cyan": (0.0, 0.7, 0.8), "magenta": (0.8, 0.1, 0.7),
          "yellow": (0.95, 0.85, 0.1), "orange": (1.0, 0.55, 0.0)}


def default_expr(sol: Solution) -> str:
    av = sol.available()
    pref = ["T", "solid.mises", "V", "spf.U", "acpr.p_t", "c", "mf.normB", "es.normE", "u"]
    names = [a for a, *_ in av]
    for p in pref:
        if p in names:
            return p
    for a, _l, _u, kind in av:
        if kind == "scalar":
            return a
    return "x"


def default_vector(sol: Solution) -> str:
    for a, _l, _u, kind in sol.available():
        if kind == "vector":
            return a
    return ""


def _values(sol, expr, fi, unit):
    v = sol.evaluate(expr, fi)
    label, si = sol.describe(expr)
    if unit and unit != si:
        try:
            v = units.from_si(v, unit) if si else v
        except Exception:
            unit = si
    return v, label, (unit or si)


def _num(text, default=0.0):
    try:
        return float(units.evaluate(str(text)))
    except Exception:
        return default


def _clim(node, v):
    if node.get("manual", False):
        try:
            return float(units.evaluate(node.get("vmin", "0"))), float(units.evaluate(node.get("vmax", "1")))
        except Exception:
            pass
    finite = v[np.isfinite(v)]
    if not finite.size:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    if node.get("__symmetric") or node.get("symmetric", False):
        m = max(abs(lo), abs(hi))
        return -m, m
    if hi - lo < 1e-14 * max(abs(hi), 1e-300):
        hi = lo + max(abs(lo) * 1e-6, 1e-12)
    return lo, hi


def _deform(sol, feat, fi):
    for c in feat.children:
        if c.kind == "plot.deform" and c.enabled:
            expr = c.get("expr", "disp") or "disp"
            try:
                vec = sol.evaluate_vector(expr, fi)
            except Exception:
                return None, None, None
            if c.get("scale_auto", True):
                s = auto_scale_deformation(sol, vec)
            else:
                try:
                    s = float(units.evaluate(c.get("scale", "1")))
                except Exception:
                    s = 1.0
            return vec / sol.scale * s, s, expr
    return None, None, None


def _sub(feat, kind):
    return next((c for c in feat.children if c.kind == kind and c.enabled), None)


def _restrict(g, feat):
    """Apply a Selection subnode: keep cells whose domain/boundary number is selected."""
    sel = _sub(feat, "plot.selection")
    if sel is None or sel.selection is None or g is None or not g.n_cells:
        return g
    idname = "dom" if sel.get("level", "boundary") == "domain" else "bnd"
    if idname not in g.cell_data:
        return g
    ids = np.asarray(g.cell_data[idname])
    keep = set(sel.selection.resolve(sorted(set(int(x) for x in ids))))
    cells = np.where(np.isin(ids, list(keep)))[0]
    return g.extract_cells(cells) if len(cells) else g.extract_cells([])


def _color_values(sol, feat, fi, n):
    """Values of a Color Expression subnode (or None)."""
    ce = _sub(feat, "plot.colorexpr")
    if ce is None or not str(ce.get("expr", "")).strip():
        return None, None
    v, label, unit = _values(sol, ce.get("expr"), fi, ce.get("unit", ""))
    return np.asarray(v, float), (ce, label, unit)


def _model_size(nodes):
    return float(np.linalg.norm(np.ptp(nodes, axis=0))) or 1.0


def _legend(sc, pg, f, cmap, clim, unit):
    if f.get("colorleg", True) and pg.get("legend", True):
        sc.legends.append({"cmap": cmap, "clim": clim, "title": pretty_unit(unit), "maxmin": pg.get("maxmin", True)})


def _surface_kw(f, cmap, clim, sdim, edges):
    if f.get("coloring", "table") == "uniform":
        kw = dict(color=COLORS.get(f.get("ucolor", "black"), (0, 0, 0)))
    else:
        kw = dict(scalars="f", cmap=cmap, clim=clim, interpolate_before_map=True)
    kw.update(show_edges=edges, edge_color=(0.15, 0.15, 0.18), line_width=0.5, smooth_shading=sdim == 3,
              lighting=sdim == 3, ambient=0.3, diffuse=0.75, specular=0.15)
    if f.get("wireframe", False):
        kw.update(style="wireframe", line_width=1.0, show_edges=False)
    return kw


def build_scene(pg: Node, sol, fi: int, sdim: int, axi=False, revolution=None) -> Scene:
    import pyvista as pv
    sc = Scene()
    sc.sdim = sdim
    fr_label = sol.frames[fi].label if 0 <= fi < len(sol.frames) else ""
    titles = []
    edge_source = None
    for f in pg.children:
        if not f.enabled or f.kind not in PLOT_NAMES:
            continue
        k = f.kind
        disp, dscale, dexpr = _deform(sol, f, fi)
        nodes = np.asarray(sol.topo.nodes, float).copy()
        if disp is not None:
            nodes = nodes + disp
        size = _model_size(nodes)
        if k in ("plot.surface", "plot.volume", "plot.slice", "plot.multislice", "plot.isosurface", "plot.contour"):
            expr = (f.get("expr") or "").strip() or default_expr(sol)
            try:
                v, label, unit = _values(sol, expr, fi, f.get("unit", ""))
            except Exception as exc:
                titles.append(f"{PLOT_NAMES[k]}: error in '{expr}' ({exc})")
                continue
            cmap = colormap(f.get("cmap", "RainbowLight"), f.get("reverse", False))
            if expr in ("acpr.p_t", "p_re") or f.get("cmap") in ("Wave", "WaveLight"):
                f.props["__symmetric"] = True
            clim = _clim(f, v)
            f.props.pop("__symmetric", None)
            show_edges = bool(f.get("edges", False))
            uniform = f.get("coloring", "table") == "uniform"
            if k in ("plot.surface", "plot.volume"):
                g = (sol.boundary_grid() if sdim == 3 and k == "plot.surface" else sol.bulk_grid()).copy()
                g.points = nodes
                g.point_data["f"] = v
                g = _restrict(g, f)
                ds = g.extract_surface(nonlinear_subdivision=1)
                sc.items.append((ds, _surface_kw(f, cmap, clim, sdim, show_edges)))
                edge_source = nodes if edge_source is None else edge_source
            elif k in ("plot.slice", "plot.multislice"):
                g = sol.bulk_grid().copy()
                g.points = nodes
                g.point_data["f"] = v
                g = _restrict(g, f)
                planes = []                       # (normal axis index, coordinate)
                lo, hi = np.min(nodes, axis=0), np.max(nodes, axis=0)
                if k == "plot.slice":
                    ax = {"yz": 0, "zx": 1, "xy": 2}[f.get("plane", "yz")]
                    shift = _num(f.get("shift", "0"))
                    if f.get("entry", "number") == "coords":
                        cs = [_num(t) for t in re.split(r"[\s,;]+", str(f.get("coords", "0")).strip()) if t]
                    else:
                        n = max(1, int(f.get("nplanes", 5)))
                        cs = list(np.linspace(lo[ax], hi[ax], n + 2)[1:-1])
                    planes = [(ax, c + shift) for c in cs]
                else:
                    for ax, a in enumerate("xyz"):
                        n = int(f.get(f"n{a}", 1) or 0)
                        if n > 0:
                            planes += [(ax, c) for c in np.linspace(lo[ax], hi[ax], n + 2)[1:-1]]
                for ax, c in planes:
                    o = (lo + hi) / 2
                    o[ax] = c
                    try:
                        sl = g.slice(normal=np.eye(3)[ax], origin=o)
                        if sl.n_points:
                            sc.items.append((sl, dict(scalars="f", cmap=cmap, clim=clim, interpolate_before_map=True,
                                                      lighting=False) if not uniform else
                                             dict(color=COLORS.get(f.get("ucolor", "black")), lighting=False)))
                    except Exception:
                        pass
                edge_source = nodes if edge_source is None else edge_source
            elif k == "plot.isosurface":
                g = sol.bulk_grid().copy()
                g.points = nodes
                g.point_data["f"] = v
                g = _restrict(g, f)
                lv = max(1, int(f.get("levels", 5)))
                iso_vals = np.linspace(clim[0], clim[1], lv + 2)[1:-1]
                try:
                    iso = g.contour(isosurfaces=list(iso_vals), scalars="f")
                    kw = dict(scalars="f", cmap=cmap, clim=clim, smooth_shading=True, opacity=0.85)
                    if uniform:
                        kw = dict(color=COLORS.get(f.get("ucolor", "black")), smooth_shading=True, opacity=0.85)
                    if iso.n_points:
                        sc.items.append((iso, kw))
                except Exception:
                    pass
                edge_source = nodes if edge_source is None else edge_source
            elif k == "plot.contour":
                g = (sol.boundary_grid() if sdim == 3 else sol.bulk_grid()).copy()
                g.points = nodes
                g.point_data["f"] = v
                g = _restrict(g, f)
                lv = max(1, int(f.get("levels", 20)))
                iso_vals = list(np.linspace(clim[0], clim[1], lv + 2)[1:-1])
                ctype = f.props.get("ctype") or ("filled" if f.props.get("filled") else "lines")
                surf = g.extract_surface(nonlinear_subdivision=1)
                try:
                    if ctype == "filled":
                        sc.items.append((surf, dict(scalars="f", cmap=cmap, clim=clim, n_colors=lv + 1,
                                                    lighting=sdim == 3)))
                    lines = surf.contour(isosurfaces=iso_vals, scalars="f")
                    if lines.n_points:
                        if ctype == "tubes":
                            lines = lines.tube(radius=0.003 * size)
                        if uniform or (f.get("cmap") == "GrayScale" and not f.get("colorleg", True)) or ctype == "filled":
                            kw = dict(color=COLORS.get(f.get("ucolor", "black"), (0, 0, 0)), line_width=1.2)
                        else:
                            kw = dict(scalars="f", cmap=cmap, clim=clim, line_width=2)
                        sc.items.append((lines, dict(**kw, lighting=ctype == "tubes")))
                        if f.get("labels", False):
                            vals = np.asarray(lines.point_data["f"])
                            P = np.asarray(lines.points)
                            for lvl in iso_vals:
                                j = int(np.argmin(np.abs(vals - lvl)))
                                if abs(vals[j] - lvl) <= 1e-6 * max(abs(lvl), 1e-30) + 1e-12 * (clim[1] - clim[0] + 1):
                                    sc.texts.append((P[j], f"{lvl:.3g}", False))
                except Exception:
                    pass
                edge_source = nodes if edge_source is None else edge_source
            if not uniform:
                _legend(sc, pg, f, cmap, clim, unit)
            titles.append(f"{PLOT_NAMES[k]}: {label} ({pretty_unit(unit)})" if unit else f"{PLOT_NAMES[k]}: {label}")
        elif k == "plot.arrow":
            expr = (f.get("expr") or "").strip() or default_vector(sol)
            try:
                vec = sol.evaluate_vector(expr, fi)
            except Exception as exc:
                titles.append(f"Arrow Volume: error ({exc})")
                continue
            g = sol.bulk_grid().copy()
            g.points = nodes
            g.point_data["vec"] = vec
            cvals, cinfo = _color_values(sol, f, fi, len(nodes))
            if cvals is not None:
                g.point_data["c"] = cvals
            g = _restrict(g, f)
            if f.get("positioning", "grid") == "grid":
                lo, hi = np.min(nodes, axis=0), np.max(nodes, axis=0)
                ns = [max(1, int(f.get(key, 15))) for key in ("xg", "yg", "zg")]
                if sdim == 2:
                    ns[2] = 1
                axes = [np.linspace(lo[d], hi[d], ns[d] + 2)[1:-1] if ns[d] > 1 or hi[d] > lo[d] else np.array([lo[d]])
                        for d in range(3)]
                if sdim == 2:
                    axes[2] = np.array([0.0])
                X, Y, Z = np.meshgrid(*axes, indexing="ij")
                probe = pv.PolyData(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])).sample(g)
                ok = np.asarray(probe.point_data.get("vtkValidPointMask", np.ones(probe.n_points)), bool)
                pdata = pv.PolyData(np.asarray(probe.points)[ok])
                for nm in ("vec", "c"):
                    if nm in probe.point_data:
                        pdata.point_data[nm] = np.asarray(probe.point_data[nm])[ok]
            else:
                npts = max(10, int(f.get("npts", 400)))
                rng = np.random.default_rng(0)
                idx = rng.choice(g.n_points, size=min(npts, g.n_points), replace=False)
                pdata = pv.PolyData(np.asarray(g.points)[idx])
                for nm in ("vec", "c"):
                    if nm in g.point_data:
                        pdata.point_data[nm] = np.asarray(g.point_data[nm])[idx]
            if not pdata.n_points:
                titles.append("Arrow Volume: no points inside the domain")
                continue
            V = np.asarray(pdata.point_data["vec"])
            mag = np.linalg.norm(V, axis=1)
            normalized = f.get("arrowlen", "proportional") == "normalized"
            if normalized:
                pdata.point_data["vec"] = V / np.where(mag > 0, mag, 1)[:, None]
                mag = (mag > 0).astype(float)
            pdata.point_data["mag"] = mag
            if f.get("scale_auto", True):
                spacing = size / max(8.0, float(np.cbrt(pdata.n_points) if sdim == 3 else np.sqrt(pdata.n_points)))
                fac = 0.9 * spacing / (np.nanmax(mag) or 1.0)
            else:
                fac = _num(f.get("scale", "1"), 1.0)
            arrows = pdata.glyph(orient="vec", scale="mag", factor=fac, geom=pv.Arrow(shaft_radius=0.04, tip_radius=0.11))
            col = f.get("color", "red")
            if cinfo is not None:
                ce, clabel, cunit = cinfo
                cm = colormap(ce.get("cmap", "RainbowLight"))
                cl = _clim(ce, np.asarray(pdata.point_data["c"]))
                sc.items.append((arrows, dict(scalars="c", cmap=cm, clim=cl)))
                _legend(sc, pg, ce, cm, cl, cunit)
            elif col == "magnitude":
                cm = colormap("RainbowLight")
                sc.items.append((arrows, dict(scalars="mag", cmap=cm)))
                sc.legends.append({"cmap": cm, "clim": (float(np.nanmin(mag)), float(np.nanmax(mag))), "title": "",
                                   "maxmin": pg.get("maxmin", True)})
            else:
                sc.items.append((arrows, dict(color=COLORS.get(col, (0.85, 0.1, 0.1)))))
            label, _unit = sol.describe(expr)
            titles.append(f"Arrow Volume: {label}")
            edge_source = nodes if edge_source is None else edge_source
        elif k == "plot.streamline":
            expr = (f.get("expr") or "").strip() or default_vector(sol)
            try:
                vec = sol.evaluate_vector(expr, fi)
            except Exception as exc:
                titles.append(f"Streamline: error ({exc})")
                continue
            g = sol.bulk_grid().copy()
            g.points = nodes
            g.point_data["vec"] = vec
            g.point_data["mag"] = np.linalg.norm(vec, axis=1)
            cvals, cinfo = _color_values(sol, f, fi, len(nodes))
            if cvals is not None:
                g.point_data["c"] = cvals
            n = max(2, int(f.get("nseeds", 40)))
            rng = np.random.default_rng(1)
            pos = f.get("positioning", "uniform")
            seeds = None
            if pos == "start":
                b = sol.boundary_grid().copy()
                b.points = nodes
                b = _restrict(b, f)
                if b.n_cells:
                    bpts = np.asarray(b.extract_cells(np.arange(b.n_cells)).points)    # only points on those cells
                    pick = np.linspace(0, len(bpts) - 1, min(n, len(bpts))).astype(int)
                    order = np.lexsort(bpts.T[::-1])
                    centre = bpts.mean(axis=0)
                    seeds_p = bpts[order][pick]
                    seeds = pv.PolyData(seeds_p + (centre - seeds_p) * 1e-4)    # nudge inside the domain
            if seeds is None:
                if pos == "magnitude":
                    w = np.asarray(g.point_data["mag"], float)
                    w = w / w.sum() if w.sum() > 0 else None
                    idx = rng.choice(len(nodes), size=min(n, len(nodes)), replace=False, p=w)
                else:
                    idx = rng.choice(len(nodes), size=min(n, len(nodes)), replace=False)
                seeds = pv.PolyData(nodes[idx])
            try:
                sl = g.streamlines_from_source(seeds, vectors="vec", integration_direction="both" if pos != "start" else "both",
                                               max_length=10 * size)
                ltype = f.get("ltype") or ("tube" if f.get("tube", False) else "line")
                w = _num(f.get("width", "0.004"), 0.004) * size
                if sl.n_points and ltype == "tube":
                    sl = sl.tube(radius=w)
                elif sl.n_points and ltype == "ribbon":
                    try:
                        sl = sl.ribbon(width=w, normal=(0, 0, 1) if sdim == 2 else None)
                    except Exception:
                        sl = sl.tube(radius=w / 2)
                if sl.n_points:
                    if cinfo is not None:
                        ce, _cl, cunit = cinfo
                        cm = colormap(ce.get("cmap", "RainbowLight"))
                        clim = _clim(ce, np.asarray(sl.point_data["c"]))
                        sc.items.append((sl, dict(scalars="c", cmap=cm, clim=clim, line_width=2, lighting=ltype != "line")))
                        _legend(sc, pg, ce, cm, clim, cunit)
                    else:
                        cm = colormap(f.get("cmap", "RainbowLight"))
                        mags = sl.point_data["mag"]
                        clim = (float(np.nanmin(mags)), float(np.nanmax(mags)))
                        sc.items.append((sl, dict(scalars="mag", cmap=cm, clim=clim, line_width=2, lighting=ltype != "line")))
                        sc.legends.append({"cmap": cm, "clim": clim, "title": "", "maxmin": pg.get("maxmin", True)})
            except Exception as exc:
                titles.append(f"Streamline: error ({exc})")
                continue
            titles.append(f"Streamline: {sol.describe(expr)[0]}")
            edge_source = nodes if edge_source is None else edge_source
        elif k == "plot.mesh":
            g = sol.bulk_grid().copy()
            g.points = nodes
            g = _restrict(g, f)
            if f.get("quality", False) and "quality" in g.cell_data:
                cm = colormap("Traffic", True)
                sc.items.append((g.extract_surface(), dict(scalars="quality", cmap=cm, clim=(0, 1), show_edges=True,
                                                           edge_color=(0.12, 0.12, 0.15), line_width=0.6)))
                sc.legends.append({"cmap": cm, "clim": (0.0, 1.0), "title": "", "maxmin": pg.get("maxmin", True)})
            else:
                sc.items.append((g.extract_surface(), dict(color=(0.78, 0.86, 0.94), show_edges=True,
                                                           edge_color=(0.12, 0.12, 0.15), line_width=0.6)))
            titles.append("Mesh")
        elif k == "plot.line":
            g = sol.boundary_grid().copy()
            g.points = nodes
            expr = str(f.get("expr", "") or "").strip()
            unit = label = ""
            if expr:
                try:
                    v, label, unit = _values(sol, expr, fi, f.get("unit", ""))
                    g.point_data["f"] = v
                except Exception as exc:
                    titles.append(f"Line: error in '{expr}' ({exc})")
                    expr = ""
            g = _restrict(g, f)
            ed = g if sdim == 2 else g.extract_surface().extract_feature_edges(30, boundary_edges=True,
                                                                               non_manifold_edges=True, manifold_edges=False)
            if f.get("ltype", "line") == "tube" and ed.n_points:
                ed = ed.extract_surface() if not isinstance(ed, pv.PolyData) else ed
                ed = ed.tube(radius=_num(f.get("radius", "0.003"), 0.003) * size)
            if expr and "f" in ed.point_data:
                cm = colormap(f.get("cmap", "RainbowLight"))
                clim = _clim(f, np.asarray(ed.point_data["f"]))
                sc.items.append((ed, dict(scalars="f", cmap=cm, clim=clim, line_width=3, lighting=False)))
                _legend(sc, pg, f, cm, clim, unit)
                titles.append(f"Line: {label} ({pretty_unit(unit)})" if unit else f"Line: {label}")
            else:
                sc.items.append((ed, dict(color=COLORS.get(f.get("color", "black"), (0, 0, 0)), line_width=2,
                                          lighting=False)))
        elif k == "plot.annotation":
            p = np.array([_num(f.get("x", "0")), _num(f.get("y", "0")), _num(f.get("z", "0")) if sdim == 3 else 0.0])
            sc.texts.append((p, annotation_text(f.get("text", ""), sol, fi, p, int(f.get("precision", 4))),
                             bool(f.get("point", True))))
        if dscale is not None and titles:
            titles[-1] = titles[-1] + f"   Deformation: {sol.describe(dexpr)[0] if dexpr not in ('disp', 'u') else 'Displacement field'}"
    # dataset edges (COMSOL plots draw the geometry edges)
    if edge_source is not None and pg.get("edges", True):
        try:
            b = sol.boundary_grid().copy()
            if len(edge_source) == b.n_points:         # follow the (deformed) plotted nodes
                b.points = edge_source
            if sdim == 3:
                ed = b.extract_surface().extract_feature_edges(30, boundary_edges=True, non_manifold_edges=True,
                                                               manifold_edges=False)
            else:
                ed = b
            if ed.n_cells:
                sc.items.append((ed, dict(color="black", line_width=1.0, lighting=False)))
        except Exception:
            pass
    if revolution is not None:
        _revolve_scene(sc, revolution.get("angle", 225.0), int(revolution.get("layers", 60)), revolution.get("start", -90.0))
    elif axi and pg.kind == "pg2d" and pg.get("revolve", False):
        _revolve_scene(sc)
    tt = pg.get("title_type", "auto")
    if tt == "manual":
        sc.title = pg.get("title", "")
    elif tt == "auto":
        head = fr_label + "    " if fr_label else ""
        sc.title = head + "   ".join(titles)
    return sc


_EVAL = re.compile(r"eval\(\s*([^,()]+(?:\([^()]*\))?[^,()]*)\s*(?:,\s*([^()]*?))?\s*\)")


def annotation_text(text, sol, fi, p, precision=4):
    """Replace eval(expr[, unit]) in an annotation by the expression value at point *p* (geometry units)."""
    def rep(m):
        expr, unit = m.group(1).strip(), (m.group(2) or "").strip()
        try:
            val = float(sol.point_eval(expr, fi, [np.asarray(p, float)[:max(2, sol.sdim)] * sol.scale])[0])
            _l, si = sol.describe(expr)
            if unit and si and unit != si:
                val = float(units.from_si(val, unit))
            return f"{val:.{precision}g}" + (f" {pretty_unit(unit)}" if unit else "")
        except Exception:
            return "?"
    return _EVAL.sub(rep, str(text))


def _revolve_scene(sc: Scene, angle=225.0, resolution=60, start=0.0):
    """Revolution 2D dataset: sweep the (r, z) plot around the z (= y) axis, like COMSOL (default -90..135 deg)."""
    import pyvista as pv
    items = []
    for ds, kw in sc.items:
        if "scalars" not in kw:          # dataset edges/lines would sweep into black skirts; drop them
            continue
        surf = ds if isinstance(ds, pv.PolyData) else ds.extract_surface()
        if surf.n_cells == 0:
            continue
        if start:
            surf = surf.rotate_y(-start, inplace=False)
        try:
            rev = surf.extrude_rotate(resolution=resolution, angle=angle, rotation_axis=(0, 1, 0), capping=True)
        except TypeError:
            rev = surf.extrude_rotate(resolution=resolution, angle=angle, capping=True)
        kw = dict(kw)
        kw["lighting"] = True
        kw["smooth_shading"] = True
        items.append((rev, kw))
    sc.items = items
    sc.texts = []
    sc.sdim = 3


# --------------------------------------------------------------------------- 1D
_CYCLE = ["#1f5fbf", "#2e9e3e", "#d62728", "#00a0b0", "#b5179e", "#e08a00", "#333333", "#7f3fbf"]


def _style(f, i):
    """matplotlib kwargs from a graph's Coloring and Style section; *i* = index in the color cycle."""
    ls = f.get("lstyle", "-")
    col = f.get("lcolor", "cycle")
    kw = dict(lw=_num(f.get("lwidth", "1.6"), 1.6), linestyle="none" if ls == "none" else ls,
              color=_CYCLE[i % len(_CYCLE)] if col == "cycle" else col)
    mk = f.get("marker", "none")
    if mk != "none":
        kw.update(marker=mk, ms=5)
    return kw


def _markevery(f, n):
    if f.get("marker", "none") == "none" or n <= 0:
        return None
    return max(1, n // max(1, int(f.get("nmarkers", 10))))


def _plot(ax, f, xs, ys, i, label):
    kw = _style(f, i)
    me = _markevery(f, len(xs))
    if me:
        kw["markevery"] = me
    ax.plot(xs, ys, label=f.get("legend_text") or label, **kw)


def _reduce(f: Node, s, expr, fi, model):
    """Scalar for a Global graph: an operator expression (aveop1(T)) or a reduction of a field."""
    how = f.get("reduce", "max")
    if how == "point":
        ev = lambda key: float(units.evaluate(f.get(key, "0"), model.parameters())) * s.scale  # noqa: E731
        pt = [ev("px"), ev("py")] + ([ev("pz")] if s.sdim == 3 else [])
        return float(s.point_eval(expr, fi, [pt])[0])
    v = s.evaluate(expr, fi)
    if how == "global" or np.ptp(v) == 0:
        return float(np.nanmean(v))
    return float({"max": np.nanmax, "min": np.nanmin, "avg": np.nanmean}[how](v))


def cut_points(ds, model, sdim):
    """Coordinates (geometry units) of a Cut Point dataset: space-separated lists, broadcast."""
    def lst(key):
        return [float(units.evaluate(t, model.parameters())) for t in re.split(r"[\s,;]+", str(ds.get(key, "0")).strip()) if t]
    xs, ys = lst("x"), lst("y")
    zs = lst("z") if sdim == 3 else [0.0]
    n = max(len(xs), len(ys), len(zs))
    pick = lambda a, i: a[i] if i < len(a) else a[-1]  # noqa: E731
    return [[pick(xs, i), pick(ys, i), pick(zs, i)] for i in range(n)]


def _xlabel_for(sol):
    return {"study.time": f"Time ({sol.meta.get('tunit', 's')})", "study.freq": "freq (Hz)",
            "study.eigen": "eigfreq (Hz)"}.get(sol.kind, "Solution")


def draw_1d(fig, pg: Node, model, get_solution, frame_sel, get_sweep=None, get_table=None):
    """Draw a 1D (or polar) plot group into a matplotlib Figure."""
    fig.clear()
    if pg.kind == "pgpolar":
        return draw_polar(fig, pg, model, get_solution)
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    lines = 0
    ylabels = []
    xlabel = None
    for f in pg.children:
        if not f.enabled:
            continue
        if f.kind == "plot.linegraph":
            ds = model.by_tag(f.get("data") or pg.get("data") or "")
            if ds is None or ds.kind != "dset.cutline":
                ax.text(0.5, 0.5, "Select a Cut Line dataset for the Line Graph", ha="center", transform=ax.transAxes)
                continue
            sol, fi, scale = get_solution(ds, frame_sel)
            if sol is None:
                continue
            ev = lambda key: float(units.evaluate(ds.get(key, "0"), model.parameters()))  # noqa: E731
            sdim = sol.sdim
            p0 = [ev("x0"), ev("y0")] + ([ev("z0")] if sdim == 3 else [0.0])
            p1 = [ev("x1"), ev("y1")] + ([ev("z1")] if sdim == 3 else [0.0])
            p0 = np.array(p0) * sol.scale
            p1 = np.array(p1) * sol.scale
            expr = (f.get("expr") or "").strip() or default_expr(sol)
            s, pts, val = sol.line_eval(expr, fi, p0, p1, int(ds.get("res", 200)))
            label, si = sol.describe(expr)
            unit = f.get("unit") or si
            if unit and unit != si and si:
                val = units.from_si(val, unit)
            xax = f.get("xaxis", "arc")
            x = s if xax == "arc" else pts[:, "xyz".index(xax)]
            lu = sol.unit
            x = x / sol.scale
            _plot(ax, f, x, val, lines, label)
            xlabel = xlabel or (f"Arc length ({lu})" if xax == "arc" else f"{xax} ({lu})")
            ylabels.append(f"{label} ({pretty_unit(unit)})" if unit else label)
            lines += 1
        elif f.kind == "plot.globalgraph":
            ds = model.by_tag(f.get("data") or pg.get("data") or "")
            if ds is None:
                continue
            sol, _fi, _ = get_solution(ds, "last")
            if sol is None:
                continue
            expr = (f.get("expr") or "").strip() or default_expr(sol)
            sweep, sw_label, sw_unit = get_sweep(ds) if get_sweep else ([], None, "")
            label, si = sol.describe(expr)
            if sweep:
                fac = units.unit_info(sw_unit)[0] if sw_unit else 1.0
                xs, ys = [], []
                for val, s in sweep:
                    try:
                        y = _reduce(f, s, expr, s.frame_index("last"), model)
                    except Exception:
                        y = np.nan
                    xs.append(val / fac)
                    ys.append(y)
                _plot(ax, f, xs, ys, lines, label)
                xlabel = xlabel or (f"{sw_label} ({pretty_unit(sw_unit)})" if sw_unit else str(sw_label))
            else:
                xs, ys = [], []
                for i, fr in enumerate(sol.frames):
                    try:
                        y = _reduce(f, sol, expr, i, model)
                    except Exception:
                        y = np.nan
                    xs.append(fr.value)
                    ys.append(y)
                _plot(ax, f, xs, ys, lines, label)
                xlabel = xlabel or _xlabel_for(sol)
            ylabels.append(f"{label} ({pretty_unit(si)})" if si else label)
            lines += 1
        elif f.kind == "plot.pointgraph":
            ds = model.by_tag(f.get("data") or pg.get("data") or "")
            if ds is None or ds.kind != "dset.cutpoint":
                ax.text(0.5, 0.5, "Select a Cut Point dataset for the Point Graph", ha="center", transform=ax.transAxes)
                continue
            sol, _fi, _ = get_solution(ds, "last")
            if sol is None:
                continue
            expr = (f.get("expr") or "").strip() or default_expr(sol)
            label, si = sol.describe(expr)
            unit = f.get("unit") or si
            pts = cut_points(ds, model, sol.sdim)
            P = [np.array(p[:max(2, sol.sdim)]) * sol.scale for p in pts]
            sweep, sw_label, sw_unit = get_sweep(ds) if get_sweep else ([], None, "")
            if sweep:
                fac = units.unit_info(sw_unit)[0] if sw_unit else 1.0
                xs = [val / fac for val, _s in sweep]
                Y = np.array([s.point_eval(expr, s.frame_index("last"), P) for _v, s in sweep])
                xlabel = xlabel or (f"{sw_label} ({pretty_unit(sw_unit)})" if sw_unit else str(sw_label))
            else:
                xs = [fr.value for fr in sol.frames]
                Y = np.array([sol.point_eval(expr, i, P) for i in range(len(sol.frames))])
                xlabel = xlabel or _xlabel_for(sol)
            if unit and si and unit != si:
                Y = units.from_si(Y, unit)
            for j in range(Y.shape[1]):
                lbl = label if Y.shape[1] == 1 else f"{label}, Point {j + 1}"
                _plot(ax, f, xs, Y[:, j], lines, lbl)
                lines += 1
            ylabels.append(f"{label} ({pretty_unit(unit)})" if unit else label)
        elif f.kind == "plot.tablegraph":
            tbl = get_table(f.get("table", "")) if get_table else model.by_tag(f.get("table", "") or "")
            if tbl is None:
                ax.text(0.5, 0.5, "Select a table (evaluate a Derived Value first)", ha="center", transform=ax.transAxes)
                continue
            cols = list(tbl.props.get("columns") or [])
            rows = [r for r in (tbl.props.get("rows") or []) if r]
            if not rows:
                continue
            num = np.array([[float(c) if isinstance(c, (int, float)) else np.nan for c in r] for r in rows], float)
            xc = int(f.get("xcol", 1) or 0)
            xs = num[:, xc - 1] if 1 <= xc <= num.shape[1] else np.arange(1, len(num) + 1)
            want = [int(t) for t in re.split(r"[\s,;]+", str(f.get("ycols", "")).strip()) if t.strip().isdigit()]
            ycols = [c - 1 for c in want if 1 <= c <= num.shape[1]] or \
                [c for c in range(num.shape[1]) if c != xc - 1 and np.isfinite(num[:, c]).any()]
            for c in ycols:
                _plot(ax, f, xs, num[:, c], lines, cols[c] if c < len(cols) else f"Column {c + 1}")
                lines += 1
            xlabel = xlabel or (cols[xc - 1] if 1 <= xc <= len(cols) else "Row")
            ylabels.append(cols[ycols[0]] if len(ycols) == 1 and ycols[0] < len(cols) else "")
    ax.set_xlabel(pg.get("xlabel") or xlabel or "")
    if ylabels:
        ax.set_ylabel(pg.get("ylabel") or ylabels[0])
    if pg.get("xlog", False):
        ax.set_xscale("log")
    if pg.get("ylog", False):
        ax.set_yscale("log")
    if pg.get("grid", True):
        ax.grid(True, color="#d0d4d9", lw=0.8)
    for sp in ax.spines.values():
        sp.set_color("#8a9099")
    if pg.get("legend", True) and lines > 1:
        ax.legend(frameon=True, fontsize=8, loc=pg.get("legendpos", "upper right"))
    tt = pg.get("title_type", "auto")
    if tt == "manual":
        ax.set_title(pg.get("title", ""), fontsize=10)
    elif tt == "auto":
        kinds = {c.kind for c in pg.children if c.enabled}
        head = {frozenset({"plot.globalgraph"}): "Global: ", frozenset({"plot.pointgraph"}): "Point Graph: ",
                frozenset({"plot.tablegraph"}): "Table Graph: "}.get(frozenset(kinds), "Line Graph: ")
        ax.set_title(head + (ylabels[0] if ylabels else ""), fontsize=10)
    fig.tight_layout()


def polar_values(view, f, model, fi):
    """(angles rad, values) of a polar graph's evaluation circle on a 2D dataset."""
    ev = lambda key, d="0": float(units.evaluate(str(f.get(key, d)), model.parameters()))  # noqa: E731
    xc, yc, r = ev("xc"), ev("yc"), ev("radius", "1")
    a0, a1 = ev("a0", "0"), ev("a1", "360")
    n = max(3, int(f.get("npts", 181)))
    th = np.radians(np.linspace(a0, a1, n))
    P = np.column_stack([xc + r * np.cos(th), yc + r * np.sin(th)]) * view.scale
    expr = (f.get("expr") or "").strip() or default_expr(view)
    v = np.asarray(view.point_eval(expr, fi, P), float)
    _l, si = view.describe(expr)
    unit = f.get("unit") or si
    if unit and si and unit != si:
        v = units.from_si(v, unit)
    return th, v


def draw_polar(fig, pg: Node, model, get_solution):
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location(pg.get("zero", "E"))
    ylabels, lines = [], 0
    for f in pg.children:
        if not f.enabled or f.kind != "plot.polar":
            continue
        ds = model.by_tag(f.get("data") or pg.get("data") or "")
        sol, fi, _ = get_solution(ds, "last")
        if sol is None:
            continue
        expr = (f.get("expr") or "").strip() or default_expr(sol)
        label, si = sol.describe(expr)
        unit = f.get("unit") or si
        idx = [int(t) - 1 for t in re.split(r"[\s,;]+", str(f.get("params", "")).strip()) if t.strip().isdigit()]
        idx = [i for i in idx if 0 <= i < len(sol.frames)] or [fi]
        for i in idx:
            th, v = polar_values(sol, f, model, i)
            lbl = sol.frames[i].label or label
            _plot(ax, f, th, v, lines, lbl)
            lines += 1
        ylabels.append(f"{label} ({pretty_unit(unit)})" if unit else label)
    if pg.get("legend", True) and lines > 1:
        ax.legend(frameon=True, fontsize=8, loc="upper right", bbox_to_anchor=(1.25, 1.1))
    tt = pg.get("title_type", "auto")
    if tt == "manual":
        ax.set_title(pg.get("title", ""), fontsize=10)
    elif tt == "auto":
        ax.set_title("Line Graph: " + (ylabels[0] if ylabels else ""), fontsize=10)
    fig.tight_layout()


# --------------------------------------------------------------------------- derived values
def evaluate_derived(node: Node, model, sol: Solution, sel_frames, ctx_entities, sweep=None, sweep_label=None,
                     sweep_unit="", points=None) -> tuple[list[str], list[list]]:
    """Table for a Derived Values node. *sweep*: [(value_SI, Solution)] to evaluate over a parametric sweep.

    *points*: point coordinates (SI) for a Point Evaluation (one column per point and expression).
    """
    kind = node.kind
    exprs = [r for r in (node.get("exprs") or []) if str(r.get("expr", "")).strip()]
    if kind == "eval.global" and not exprs:
        if sol.kind == "study.eigen":
            cols = ["Mode", "Eigenfrequency (Hz)", "Angular frequency (rad/s)"]
            rows = [[i + 1, fr.value, 2 * math.pi * fr.value] for i, fr in enumerate(sol.frames)]
            return cols, rows
        cols = ["Solution", "Value"]
        return cols, [[fr.label or f"{i + 1}", fr.value] for i, fr in enumerate(sol.frames)]
    if not exprs:
        exprs = [{"expr": default_expr(sol), "unit": "", "descr": ""}]
    if kind == "eval.point" and not points:
        ev = lambda key: float(units.evaluate(node.get(key, "0"), model.parameters())) * sol.scale  # noqa: E731
        points = [[ev("x"), ev("y")] + ([ev("z")] if sol.sdim == 3 else [])]
    npt = len(points) if kind == "eval.point" else 1
    # rows: (leading cells, solution, frame index)
    head = {"study.time": f"Time ({sol.meta.get('tunit', 's')})", "study.freq": "freq (Hz)",
            "study.eigen": "eigfreq (Hz)"}.get(sol.kind)
    entries = []
    if sweep and node.get("solnum", "last") == "all":
        fac = units.unit_info(sweep_unit)[0] if sweep_unit else 1.0
        for val, s in sweep:
            entries.append(([val / fac if val is not None else float("nan")], s, s.frame_index("last")))
        cols = [f"{sweep_label} ({pretty_unit(sweep_unit)})" if sweep_unit else str(sweep_label)]
    else:
        frames = sel_frames if kind != "eval.global" or node.get("solnum", "all") != "all" else range(len(sol.frames))
        for fi in frames:
            entries.append(([sol.frames[fi].value] if head else [], sol, fi))
        cols = [head] if head else []
    for r in exprs:
        label, si = sol.describe(r["expr"])
        u = r.get("unit") or si
        if kind == "eval.volint":
            u = _int_unit(si, sol.sdim, "domain")
        elif kind == "eval.surfint":
            u = _int_unit(si, sol.sdim, "boundary")
        name = r.get("descr") or label
        for j in range(npt):
            suffix = f", Point {j + 1}" if npt > 1 else ""
            cols.append(f"{name}{suffix} ({pretty_unit(u)})" if u else f"{name}{suffix}")
        if kind in ("eval.max", "eval.min"):
            cols.append("Location (x, y, z) (m)")
    rows = []
    for lead, s, fi in entries:
        row = list(lead)
        for r in exprs:
            e = r["expr"]
            loc = None
            vals = []
            try:
                if kind == "eval.volint":
                    vals = [s.integrate(e, fi, "domain", ctx_entities)]
                elif kind == "eval.surfint":
                    vals = [s.integrate(e, fi, "boundary", ctx_entities)]
                elif kind == "eval.avg":
                    vals = [s.integrate(e, fi, "domain", ctx_entities, average=True)]
                elif kind in ("eval.max", "eval.min"):
                    mm = s.maxmin_locations(e, fi, "domain", ctx_entities)
                    key = "max" if kind == "eval.max" else "min"
                    vals, loc = [mm[key]], mm["x" + key]
                elif kind == "eval.point":
                    vals = [float(x) for x in s.point_eval(e, fi, points)]
                elif kind == "eval.global":
                    v = s.evaluate(e, fi)
                    vals = [float(np.nanmean(v))]          # operators like aveop1(T) are constant fields
                else:
                    vals = [float("nan")]
                u = r.get("unit")
                _l, si = s.describe(e)
                if u and si and u != si and kind not in ("eval.volint", "eval.surfint"):
                    vals = [float(units.from_si(x, u)) for x in vals]
            except Exception:
                vals = [float("nan")] * npt
            row.extend(vals + [float("nan")] * (npt - len(vals)))
            if kind in ("eval.max", "eval.min"):
                row.append("(" + ", ".join(f"{c:.4g}" for c in (loc or [])[:s.sdim]) + ")" if loc else "")
        rows.append(row)
    return cols, rows


def _int_unit(si, sdim, level):
    d = sdim if level == "domain" else sdim - 1
    m = {3: "m^3", 2: "m^2", 1: "m"}.get(d, "")
    if not si or si == "1":
        return m
    return f"{si}*{m}"
