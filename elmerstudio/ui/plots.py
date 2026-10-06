"""Plot engine: plot-group nodes -> VTK scenes (2D/3D) or matplotlib figures (1D); derived values."""
from __future__ import annotations

import math

import numpy as np

from ..core import units
from ..core.model import Node
from ..core.results import Solution, auto_scale_deformation, colormap
from .graphics import Scene
from .settings import pretty_unit

PLOT_NAMES = {"plot.surface": "Surface", "plot.volume": "Volume", "plot.slice": "Slice", "plot.isosurface": "Isosurface",
              "plot.contour": "Contour", "plot.arrow": "Arrow Volume", "plot.streamline": "Streamline",
              "plot.mesh": "Mesh", "plot.line": "Line"}


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
    if node.get("__symmetric"):
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


def build_scene(pg: Node, sol: Solution, fi: int, sdim: int, axi=False) -> Scene:
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
        nodes = sol.topo.nodes.copy()
        if disp is not None:
            nodes = nodes + disp
        if k in ("plot.surface", "plot.volume", "plot.slice", "plot.isosurface", "plot.contour"):
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
            if k == "plot.surface" or k == "plot.volume":
                if sdim == 3 and k == "plot.surface":
                    g = sol.boundary_grid().copy()
                else:
                    g = sol.bulk_grid().copy()
                g.points = nodes
                g.point_data["f"] = v
                ds = g.extract_surface(nonlinear_subdivision=1) if sdim == 3 or k == "plot.volume" else g.extract_surface(nonlinear_subdivision=1)
                sc.items.append((ds, dict(scalars="f", cmap=cmap, clim=clim, show_edges=show_edges,
                                          edge_color=(0.15, 0.15, 0.18), line_width=0.5, smooth_shading=sdim == 3,
                                          interpolate_before_map=True, lighting=sdim == 3, ambient=0.3, diffuse=0.75,
                                          specular=0.15)))
                edge_source = g
            elif k == "plot.slice":
                g = sol.bulk_grid().copy()
                g.points = nodes
                g.point_data["f"] = v
                axis = {"yz": "x", "zx": "y", "xy": "z"}[f.get("plane", "yz")]
                n = max(1, int(f.get("nplanes", 5)))
                try:
                    sl = g.slice_along_axis(n=n, axis=axis)
                    if isinstance(sl, pv.MultiBlock):
                        sl = sl.combine(merge_points=False).extract_surface()
                    sc.items.append((sl, dict(scalars="f", cmap=cmap, clim=clim, interpolate_before_map=True,
                                              lighting=False)))
                except Exception:
                    pass
                edge_source = g
            elif k == "plot.isosurface":
                g = sol.bulk_grid().copy()
                g.points = nodes
                g.point_data["f"] = v
                lv = max(1, int(f.get("levels", 5)))
                iso_vals = np.linspace(clim[0], clim[1], lv + 2)[1:-1]
                try:
                    iso = g.contour(isosurfaces=list(iso_vals), scalars="f")
                    sc.items.append((iso, dict(scalars="f", cmap=cmap, clim=clim, smooth_shading=True,
                                               opacity=0.85)))
                except Exception:
                    pass
                edge_source = g
            elif k == "plot.contour":
                g = sol.bulk_grid().copy()
                g.points = nodes
                g.point_data["f"] = v
                lv = max(1, int(f.get("levels", 20)))
                iso_vals = list(np.linspace(clim[0], clim[1], lv + 2)[1:-1])
                try:
                    if f.get("filled", False):
                        sc.items.append((g.extract_surface(nonlinear_subdivision=1),
                                         dict(scalars="f", cmap=cmap, clim=clim, n_colors=lv, lighting=False)))
                    else:
                        lines = g.extract_surface(nonlinear_subdivision=1).contour(isosurfaces=iso_vals, scalars="f")
                        if lines.n_points:
                            gray = f.get("cmap") == "GrayScale" and not f.get("colorleg", True)
                            kw = dict(color="black", line_width=1.2) if gray else dict(scalars="f", cmap=cmap, clim=clim, line_width=2)
                            sc.items.append((lines, dict(**kw, lighting=False)))
                except Exception:
                    pass
                edge_source = g
            if f.get("colorleg", True):
                sc.legends.append({"cmap": cmap, "clim": clim, "title": pretty_unit(unit)})
            titles.append(f"{PLOT_NAMES[k]}: {label} ({pretty_unit(unit)})" if unit else f"{PLOT_NAMES[k]}: {label}")
        elif k == "plot.arrow":
            expr = (f.get("expr") or "").strip() or default_vector(sol)
            try:
                vec = sol.evaluate_vector(expr, fi)
            except Exception as exc:
                titles.append(f"Arrow Volume: error ({exc})")
                continue
            npts = max(10, int(f.get("npts", 400)))
            rng = np.random.default_rng(0)
            idx = rng.choice(len(nodes), size=min(npts, len(nodes)), replace=False)
            P = nodes[idx]
            V = vec[idx]
            mag = np.linalg.norm(V, axis=1)
            size = float(np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0))) or 1.0
            if f.get("scale_auto", True):
                fac = 0.06 * size / (np.nanmax(mag) or 1.0)
            else:
                fac = float(units.evaluate(f.get("scale", "1")))
            pdata = pv.PolyData(P)
            pdata.point_data["vec"] = V
            pdata.point_data["mag"] = mag
            arrows = pdata.glyph(orient="vec", scale="mag", factor=fac, geom=pv.Arrow(shaft_radius=0.04, tip_radius=0.11))
            col = f.get("color", "red")
            if col == "magnitude":
                cm = colormap("RainbowLight")
                sc.items.append((arrows, dict(scalars="mag", cmap=cm)))
                sc.legends.append({"cmap": cm, "clim": (float(np.nanmin(mag)), float(np.nanmax(mag))), "title": ""})
            else:
                sc.items.append((arrows, dict(color={"red": (0.85, 0.1, 0.1), "black": "black", "blue": (0.1, 0.25, 0.85)}[col])))
            label, unit = sol.describe(expr.replace("_vec", "")) if "_vec" in expr else sol.describe(expr)
            titles.append(f"Arrow Volume: {label}")
            if edge_source is None:
                edge_source = sol.boundary_grid() if sdim == 3 else sol.bulk_grid()
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
            n = max(2, int(f.get("nseeds", 40)))
            rng = np.random.default_rng(1)
            seeds = pv.PolyData(nodes[rng.choice(len(nodes), size=min(n, len(nodes)), replace=False)])
            try:
                sl = g.streamlines_from_source(seeds, vectors="vec", integration_direction="both",
                                               max_length=10 * float(np.linalg.norm(np.ptp(nodes, axis=0)) or 1.0))
                cm = colormap(f.get("cmap", "RainbowLight"))
                if f.get("tube", False) and sl.n_points:
                    sl = sl.tube(radius=0.004 * float(np.linalg.norm(np.ptp(nodes, axis=0))))
                if sl.n_points:
                    mags = sl.point_data["mag"]
                    clim = (float(np.nanmin(mags)), float(np.nanmax(mags)))
                    sc.items.append((sl, dict(scalars="mag", cmap=cm, clim=clim, line_width=2, lighting=False)))
                    sc.legends.append({"cmap": cm, "clim": clim, "title": ""})
            except Exception as exc:
                titles.append(f"Streamline: error ({exc})")
                continue
            titles.append("Streamline")
            edge_source = g
        elif k == "plot.mesh":
            g = sol.bulk_grid().copy()
            g.points = nodes
            sc.items.append((g.extract_surface(), dict(color=(0.78, 0.86, 0.94), show_edges=True,
                                                       edge_color=(0.12, 0.12, 0.15), line_width=0.6)))
            titles.append("Mesh")
        elif k == "plot.line":
            g = sol.boundary_grid().copy() if sdim == 2 else sol.boundary_grid().copy()
            g.points = nodes
            ed = g if sdim == 2 else g.extract_surface().extract_feature_edges(30, boundary_edges=True,
                                                                               non_manifold_edges=True, manifold_edges=False)
            sc.items.append((ed, dict(color={"black": "black", "gray": "gray", "blue": "blue"}[f.get("color", "black")],
                                      line_width=2, lighting=False)))
        if dscale is not None:
            titles[-1] = titles[-1] + f"   Deformation: {sol.describe(dexpr)[0] if dexpr not in ('disp', 'u') else 'Displacement field'}"
    # dataset edges (COMSOL plots draw the geometry edges)
    if edge_source is not None:
        try:
            if sdim == 3:
                b = sol.boundary_grid().copy()
                b.points = edge_source.points if edge_source.n_points == b.n_points else b.points
                ed = b.extract_surface().extract_feature_edges(30, boundary_edges=True, non_manifold_edges=True,
                                                               manifold_edges=False)
            else:
                b = sol.boundary_grid().copy()
                b.points = edge_source.points if edge_source.n_points == b.n_points else b.points
                ed = b
            if ed.n_cells:
                sc.items.append((ed, dict(color="black", line_width=1.0, lighting=False)))
        except Exception:
            pass
    if axi and pg.kind == "pg2d" and pg.get("revolve", False):
        _revolve_scene(sc)
    tt = pg.get("title_type", "auto")
    if tt == "manual":
        sc.title = pg.get("title", "")
    elif tt == "auto":
        head = fr_label + "    " if fr_label else ""
        sc.title = head + "   ".join(titles)
    return sc


def _revolve_scene(sc: Scene, angle=225.0, resolution=60):
    """Revolution 2D dataset: sweep the (r, z) plot 225 degrees around the z (= y) axis, like COMSOL."""
    import pyvista as pv
    items = []
    for ds, kw in sc.items:
        if "scalars" not in kw:          # dataset edges/lines would sweep into black skirts; drop them
            continue
        surf = ds if isinstance(ds, pv.PolyData) else ds.extract_surface()
        if surf.n_cells == 0:
            continue
        try:
            rev = surf.extrude_rotate(resolution=resolution, angle=angle, rotation_axis=(0, 1, 0), capping=True)
        except TypeError:
            rev = surf.extrude_rotate(resolution=resolution, angle=angle, capping=True)
        kw = dict(kw)
        kw["lighting"] = True
        kw["smooth_shading"] = True
        items.append((rev, kw))
    sc.items = items
    sc.sdim = 3


# --------------------------------------------------------------------------- 1D
def draw_1d(fig, pg: Node, model, get_solution, frame_sel):
    """Draw a 1D plot group into a matplotlib Figure."""
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    lines = 0
    ylabels = []
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
            ax.plot(x, val, lw=1.6, label=f.get("legend_text") or label)
            ax.set_xlabel(pg.get("xlabel") or (f"Arc length ({lu})" if xax == "arc" else f"{xax} ({lu})"))
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
            xs, ys = [], []
            for i, fr in enumerate(sol.frames):
                try:
                    if f.get("reduce", "max") == "point":
                        ev = lambda key: float(units.evaluate(f.get(key, "0"), model.parameters())) * sol.scale  # noqa: E731
                        pt = [ev("px"), ev("py")] + ([ev("pz")] if sol.sdim == 3 else [])
                        y = float(sol.point_eval(expr, i, [pt])[0])
                    else:
                        v = sol.evaluate(expr, i)
                        y = {"max": np.nanmax, "min": np.nanmin, "avg": np.nanmean}[f.get("reduce", "max")](v)
                except Exception:
                    y = np.nan
                xs.append(fr.value)
                ys.append(y)
            ax.plot(xs, ys, "o-", lw=1.6, ms=4, label=sol.describe(expr)[0])
            kind = sol.kind
            ax.set_xlabel(pg.get("xlabel") or {"study.time": f"Time ({sol.meta.get('tunit', 's')})",
                                               "study.freq": "freq (Hz)", "study.eigen": "eigfreq (Hz)"}.get(kind, "Solution"))
            label, si = sol.describe(expr)
            ylabels.append(f"{label} ({pretty_unit(si)})" if si else label)
            lines += 1
    if ylabels:
        ax.set_ylabel(pg.get("ylabel") or ylabels[0])
    if pg.get("grid", True):
        ax.grid(True, color="#d0d4d9", lw=0.8)
    for sp in ax.spines.values():
        sp.set_color("#8a9099")
    if pg.get("legend", True) and lines > 1:
        ax.legend(frameon=True, fontsize=8)
    tt = pg.get("title_type", "auto")
    if tt == "manual":
        ax.set_title(pg.get("title", ""), fontsize=10)
    elif tt == "auto":
        ax.set_title("Line Graph: " + (ylabels[0] if ylabels else ""), fontsize=10)
    fig.tight_layout()


# --------------------------------------------------------------------------- derived values
def evaluate_derived(node: Node, model, sol: Solution, sel_frames, ctx_entities) -> tuple[list[str], list[list]]:
    kind = node.kind
    exprs = [r for r in (node.get("exprs") or []) if str(r.get("expr", "")).strip()]
    if kind == "eval.global":
        if sol.kind == "study.eigen":
            cols = ["Mode", "Eigenfrequency (Hz)", "Angular frequency (rad/s)"]
            rows = [[i + 1, fr.value, 2 * math.pi * fr.value] for i, fr in enumerate(sol.frames)]
            return cols, rows
        cols = ["Solution", "Value"]
        return cols, [[fr.label or f"{i + 1}", fr.value] for i, fr in enumerate(sol.frames)]
    if not exprs:
        exprs = [{"expr": default_expr(sol), "unit": "", "descr": ""}]
    head = {"study.time": "Time", "study.freq": "freq (Hz)", "study.eigen": "eigfreq (Hz)"}.get(sol.kind)
    cols = ([head] if head else [])
    for r in exprs:
        label, si = sol.describe(r["expr"])
        u = r.get("unit") or si
        if kind == "eval.volint":
            u = _int_unit(si, sol.sdim, "domain")
        elif kind == "eval.surfint":
            u = _int_unit(si, sol.sdim, "boundary")
        cols.append(f"{r.get('descr') or label} ({pretty_unit(u)})" if u else (r.get("descr") or label))
    rows = []
    for fi in sel_frames:
        row = [sol.frames[fi].value] if head else []
        for r in exprs:
            e = r["expr"]
            try:
                if kind == "eval.volint":
                    val = sol.integrate(e, fi, "domain", ctx_entities)
                elif kind == "eval.surfint":
                    val = sol.integrate(e, fi, "boundary", ctx_entities)
                elif kind == "eval.avg":
                    val = sol.integrate(e, fi, "domain", ctx_entities, average=True)
                elif kind == "eval.max":
                    val = sol.maxmin(e, fi, "domain", ctx_entities)[0]
                elif kind == "eval.point":
                    ev = lambda key: float(units.evaluate(node.get(key, "0"), model.parameters())) * sol.scale  # noqa: E731
                    pt = [ev("x"), ev("y")] + ([ev("z")] if sol.sdim == 3 else [])
                    val = float(sol.point_eval(e, fi, [pt])[0])
                else:
                    val = float("nan")
                u = r.get("unit")
                _l, si = sol.describe(e)
                if u and si and u != si and kind not in ("eval.volint", "eval.surfint"):
                    val = float(units.from_si(val, u))
            except Exception:
                val = float("nan")
            row.append(val)
        rows.append(row)
    return cols, rows


def _int_unit(si, sdim, level):
    d = sdim if level == "domain" else sdim - 1
    m = {3: "m^3", 2: "m^2", 1: "m"}.get(d, "")
    if not si or si == "1":
        return m
    return f"{si}*{m}"
