"""Derived datasets: Solution-like views of a cut plane, a mirror, or a parametric extrusion.

Each view exposes the subset of the Solution API the plot engine and the derived-value
evaluators use (topo.nodes, bulk_grid, boundary_grid, evaluate, evaluate_vector, point_eval,
line_eval, integrate, maxmin_locations, describe, frames, ...).  Anything not overridden is
delegated to the underlying solution, so expression parsing, units and aliases stay identical.
"""
from __future__ import annotations

import numpy as np

from .results import Frame, Solution, Topology


class _Topo:
    def __init__(self, nodes):
        self.nodes = nodes
        self.point_nodes: dict = {}


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("The plane normal must not be zero")
    return v / n


def plane_frame(normal):
    """In-plane basis (e1, e2, n): the remaining global axes for coordinate planes (yz -> (y, z), zx -> (x, z),
    xy -> (x, y)), otherwise an orthonormal pair perpendicular to the normal."""
    n = _unit(normal)
    for k in range(3):
        if abs(abs(n[k]) - 1) < 1e-12:
            a, b = [i for i in range(3) if i != k]
            return np.eye(3)[a], np.eye(3)[b], n
    a = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = _unit(a - a.dot(n) * n)
    e2 = np.cross(n, e1)
    return e1, e2, n


def _linear_triangles(grid):
    """Triangles of a 2D (possibly quadratic) grid, indexed into the grid's original points."""
    surf = _surface(grid, nonlinear_subdivision=0).triangulate()
    ids = np.asarray(surf.point_data["vtkOriginalPointIds"])
    tri = surf.faces.reshape(-1, 4)[:, 1:]
    return ids[tri]


def _surface(grid, **kw):
    """extract_surface with original point ids (quiet on recent pyvista, compatible with 0.43)."""
    try:
        return grid.extract_surface(pass_pointid=True, algorithm="dataset_surface", **kw)
    except TypeError:
        return grid.extract_surface(pass_pointid=True, **kw)


def _boundary_lines(tri, n_points):
    """Boundary edges (edges used by one triangle) of a triangulation, as a line PolyData index array."""
    e = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return uniq[cnt == 1]


class _View:
    """Common machinery: delegation to the base solution and generic sampling helpers."""

    def __init__(self, base):
        self.base = base
        self.unit, self.scale, self.kind, self.meta = base.unit, base.scale, base.kind, base.meta
        self.frames = base.frames
        self._cache: dict = {}

    def __getattr__(self, name):                 # describe, available, names, frame_index, operators, ...
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    # Solution's sampling/integration code only uses the public view API, so reuse it verbatim.
    point_eval = Solution.point_eval
    line_eval = Solution.line_eval
    integrate = Solution.integrate
    _integrate_values = Solution._integrate_values

    def maxmin_locations(self, expr, i, level="domain", entities=None):
        vals = self.evaluate(expr, i)
        ok = np.where(np.isfinite(vals))[0]
        if not len(ok):
            nan = float("nan")
            return {"max": nan, "min": nan, "xmax": [nan] * 3, "xmin": [nan] * 3}
        imax, imin = ok[np.argmax(vals[ok])], ok[np.argmin(vals[ok])]
        P = self.topo.nodes * self.scale
        return {"max": float(vals[imax]), "min": float(vals[imin]), "xmax": P[imax].tolist(), "xmin": P[imin].tolist()}

    def _node_mask(self, level, ents):
        return None


class CutPlane(_View):
    """Intersection of a 3D solution with a plane, presented in in-plane coordinates (a 2D dataset)."""

    def __init__(self, base: Solution, origin, normal):
        super().__init__(base)
        self.origin = np.asarray(origin, float)       # geometry units
        self.e1, self.e2, self.n = plane_frame(normal)
        g = base.bulk_grid().copy()
        s = g.slice(normal=self.n, origin=self.origin)
        if s.n_points == 0:
            raise ValueError("The cut plane does not intersect the geometry")
        self._slice3d = s
        P = np.asarray(s.points)                       # in-plane coordinates of the global position
        loc = np.column_stack([P @ self.e1, P @ self.e2, np.zeros(len(P))])
        self.topo = _Topo(loc)
        self.sdim = 2
        ts = s.triangulate()
        tri = ts.faces.reshape(-1, 4)[:, 1:]
        import pyvista as pv
        self._bulk = pv.PolyData(loc.copy(), np.hstack([np.full((len(tri), 1), 3), tri]).ravel())
        self._bulk.cell_data["dom"] = np.asarray(ts.cell_data["dom"]) if "dom" in ts.cell_data else np.zeros(len(tri), int)
        bl = _boundary_lines(tri, len(loc))
        self._bnd = pv.PolyData(loc.copy(), lines=np.hstack([np.full((len(bl), 1), 2), bl]).ravel())
        self._bnd.cell_data["bnd"] = np.zeros(len(bl), int)

    def bulk_grid(self):
        return self._bulk

    def boundary_grid(self):
        return self._bnd

    def to_global(self, local):
        """Map in-plane (u, v) coordinates (geometry units) back to 3D."""
        loc = np.atleast_2d(local)
        return self.origin.dot(self.n) * self.n + loc[:, :1] * self.e1 + loc[:, 1:2] * self.e2

    def _on_plane(self, vals):
        g = self.base.bulk_grid().copy()
        g.point_data["f"] = vals
        s = g.slice(normal=self.n, origin=self.origin)
        return np.asarray(s.point_data["f"], float)

    def evaluate(self, expr, i=None):
        key = ("s", expr, i)
        if key not in self._cache:
            v = np.asarray(self.base.evaluate(expr, i), float)
            if v.ndim == 0 or v.size == 1:
                v = np.full(len(self.base.topo.nodes), float(np.ravel(v)[0]))
            self._cache[key] = self._on_plane(v)
        return self._cache[key]

    def evaluate_vector(self, expr, i=None):
        vec = np.asarray(self.base.evaluate_vector(expr, i), float)
        V = np.column_stack([self._on_plane(vec[:, k]) for k in range(3)])
        return np.column_stack([V @ self.e1, V @ self.e2, np.zeros(len(V))])


class Mirror(_View):
    """The solution plus its reflection(s) in the coordinate plane(s) through the origin."""

    SIGNS = {"x": [(1, 1, 1), (-1, 1, 1)], "y": [(1, 1, 1), (1, -1, 1)], "z": [(1, 1, 1), (1, 1, -1)],
             "xy": [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1)]}

    def __init__(self, base: Solution, axis="x"):
        super().__init__(base)
        self.signs = np.array(self.SIGNS.get(axis, self.SIGNS["x"]), float)
        self.sdim = base.sdim
        bt = base.topo
        N = len(bt.nodes)
        self._bulk = None
        self._bnd = None
        if not isinstance(bt, Topology):          # a derived view (e.g. a cut plane): reflect its grids
            def copies(g):
                parts = []
                for s in self.signs:
                    c = g.copy()
                    c.points = np.asarray(g.points) * s
                    parts.append(c)
                out = parts[0].merge(parts[1:], merge_points=False)
                if out.n_points != N * len(self.signs):
                    raise ValueError("Mirror: unexpected point count after merging")
                return out
            self.topo = _Topo(np.vstack([bt.nodes * s for s in self.signs]))
            self._bulk, self._bnd = copies(base.bulk_grid()), copies(base.boundary_grid())
            return
        t = Topology.__new__(Topology)
        t.nodes = np.vstack([bt.nodes * s for s in self.signs])
        t.sdim, t.scale, t.unit, t.order = bt.sdim, bt.scale, bt.unit, bt.order
        t.quality, t.point_nodes = {}, {}
        t.bulk = {num: [(gt, c + k * N) for k in range(len(self.signs)) for gt, c in blocks] for num, blocks in bt.bulk.items()}
        t.bnd = {num: [(gt, c + k * N) for k in range(len(self.signs)) for gt, c in blocks] for num, blocks in bt.bnd.items()}
        self.topo = t

    def bulk_grid(self):
        if self._bulk is None:
            self._bulk = self.topo.bulk_grid()
        return self._bulk

    def boundary_grid(self):
        if self._bnd is None:
            self._bnd = self.topo.boundary_grid()
        return self._bnd

    def evaluate(self, expr, i=None):
        v = np.asarray(self.base.evaluate(expr, i), float)
        if v.ndim == 0 or v.size == 1:
            v = np.full(len(self.base.topo.nodes), float(np.ravel(v)[0]))
        return np.tile(v, len(self.signs))

    def evaluate_vector(self, expr, i=None):
        vec = np.asarray(self.base.evaluate_vector(expr, i), float)
        return np.vstack([vec * s for s in self.signs])

    def _node_mask(self, level, ents):
        m = self.base._node_mask(level, ents)
        return None if m is None else np.tile(m, len(self.signs))

    def integrate(self, expr, i, level="domain", entities=None, average=False):
        # reflected cells are inside-out (negative measure), and values are copies: integrate the original
        v = self.base.integrate(expr, i, level, entities, average)
        return v if average else v * len(self.signs)


class ParametricExtrusion(_View):
    """Stack a 2D dataset over parameter values (frames or sweep solutions): z = value * level scale.

    *layers* is a list of (view2d, frame index, parameter value); all views must share one mesh.
    """

    def __init__(self, layers, levelscale=1.0, label="", unit=""):
        if not layers:
            raise ValueError("Parametric extrusion needs at least one parameter value")
        super().__init__(layers[0][0])
        self.layers = layers
        self.levelscale = float(levelscale)
        self.param_label, self.param_unit = label, unit
        self.sdim = 3
        import pyvista as pv
        v0 = layers[0][0]
        P2 = np.asarray(v0.topo.nodes, float)
        N = len(P2)
        self._n2 = N
        zs = [float(val) * self.levelscale for _v, _fi, val in layers]
        self.topo = _Topo(np.vstack([np.column_stack([P2[:, 0], P2[:, 1], np.full(N, z)]) for z in zs]))
        tri = _linear_triangles(v0.bulk_grid())
        ccw = ((P2[tri[:, 1], 0] - P2[tri[:, 0], 0]) * (P2[tri[:, 2], 1] - P2[tri[:, 0], 1])
               - (P2[tri[:, 1], 1] - P2[tri[:, 0], 1]) * (P2[tri[:, 2], 0] - P2[tri[:, 0], 0])) > 0
        cells = []
        for k in range(len(layers) - 1):
            # positive-volume wedges: base triangle counter-clockwise seen from the top layer
            flip = ccw != (zs[k + 1] > zs[k])
            t = np.where(flip[:, None], tri[:, [0, 2, 1]], tri)
            a, b = t + k * N, t + (k + 1) * N
            cells.append(np.hstack([np.full((len(tri), 1), 6), a, b]))
        if cells:
            conn = np.vstack(cells).ravel()
            types = np.full(len(tri) * (len(layers) - 1), pv.CellType.WEDGE, dtype=np.uint8)
            self._bulk = pv.UnstructuredGrid(conn, types, self.topo.nodes.copy())
        else:                                         # a single layer: just the 2D dataset at its level
            self._bulk = pv.PolyData(self.topo.nodes.copy(), np.hstack([np.full((len(tri), 1), 3), tri]).ravel())
        surf = _surface(self._bulk)
        ids = np.asarray(surf.point_data["vtkOriginalPointIds"])
        faces = surf.faces
        out, j = [], 0
        while j < len(faces):
            n = faces[j]
            out.append(n)
            out.extend(ids[faces[j + 1:j + 1 + n]])
            j += n + 1
        self._bnd = pv.PolyData(self.topo.nodes.copy(), np.asarray(out, dtype=np.int64))
        self._bnd.cell_data["bnd"] = np.zeros(self._bnd.n_cells, int)
        self.frames = [Frame("", "", 0.0)]

    def bulk_grid(self):
        return self._bulk

    def boundary_grid(self):
        return self._bnd

    def evaluate(self, expr, i=None):
        out = []
        for v, fi, _val in self.layers:
            a = np.asarray(v.evaluate(expr, fi), float)
            if a.ndim == 0 or a.size == 1:
                a = np.full(self._n2, float(np.ravel(a)[0]))
            if len(a) != self._n2:
                raise ValueError("Parametric extrusion needs the same mesh for every parameter value")
            out.append(a)
        return np.concatenate(out)

    def evaluate_vector(self, expr, i=None):
        return np.vstack([np.asarray(v.evaluate_vector(expr, fi), float) for v, fi, _val in self.layers])


def parse_indices(text, n):
    """'1 3 5', '1,2', '2-4' (1-based) -> sorted 0-based indices; empty -> all."""
    text = str(text or "").replace(",", " ").strip()
    if not text:
        return list(range(n))
    out = set()
    for tok in text.split():
        if "-" in tok.strip("-"):
            a, b = tok.split("-", 1)
            out.update(range(int(a) - 1, int(b)))
        else:
            out.add(int(tok) - 1)
    return sorted(i for i in out if 0 <= i < n)


# --------------------------------------------------------------------------- dataset chains
def root_dataset(model, dset):
    """The Solution dataset at the bottom of a dataset chain (cut plane -> mirror -> solution ...)."""
    seen = set()
    while dset is not None and dset.kind != "dset.solution" and dset.tag not in seen:
        seen.add(dset.tag)
        dset = _parent(model, dset)
    return dset


def _parent(model, dset):
    p = model.by_tag(dset.get("data", "") or "") if dset.get("data") not in (None, "", "fromparent") else None
    if p is None or p is dset:
        dsets = model.results.child("results.datasets") if model.results else None
        p = next((d for d in (dsets.children if dsets else []) if d.kind == "dset.solution"), None)
    return p


def resolve(model, dset, load, sweep=None, base=None, _depth=0):
    """Solution-like view for a dataset node -> (view, revolution settings or None).

    *load(solution_dataset)* returns its Solution; *sweep(solution_dataset)* returns [(value, Solution)].
    *base* substitutes the Solution at the bottom of the chain (used to rebuild a chain per sweep value).
    """
    from . import units
    if dset is None or _depth > 12:
        return None, None
    num = lambda key, d="0": float(units.evaluate(str(dset.get(key, d)), model.parameters()))  # noqa: E731
    k = dset.kind
    if k == "dset.solution":
        return (base if base is not None else load(dset)), None
    parent = _parent(model, dset)
    if k in ("dset.cutline", "dset.cutpoint"):
        return resolve(model, parent, load, sweep, base, _depth + 1)
    if k == "dset.parext":
        view, post = resolve(model, parent, load, sweep, base, _depth + 1)
        if view is None:
            return None, None
        if len(view.frames) > 1:
            idx = parse_indices(dset.get("values", ""), len(view.frames))
            layers = [(view, i, view.frames[i].value) for i in idx]
        else:
            sw = sweep(root_dataset(model, dset)) if sweep else []
            idx = parse_indices(dset.get("values", ""), len(sw))
            layers = []
            for i in idx:
                val, s = sw[i]
                v, _ = resolve(model, parent, load, sweep, s, _depth + 1)
                layers.append((v, v.frame_index("last"), float(val)))
        return ParametricExtrusion(layers, num("levelscale", "1")), post
    view, post = resolve(model, parent, load, sweep, base, _depth + 1)
    if view is None:
        return None, None
    if k == "dset.cutplane":
        if view.sdim != 3:
            raise ValueError("A Cut Plane needs a 3D dataset")
        if dset.get("ptype", "general") == "quick":
            ax = {"yz": 0, "zx": 1, "xy": 2}[dset.get("plane", "yz")]
            o = np.zeros(3)
            o[ax] = num("coord")
            nrm = np.eye(3)[ax]
        else:
            o = np.array([num("px"), num("py"), num("pz")])
            nrm = np.array([num("nx", "1"), num("ny"), num("nz")])
        return CutPlane(view, o, nrm), post
    if k == "dset.mirror":
        return Mirror(view, dset.get("axis", "x")), post
    if k == "dset.revolution":
        return view, {"start": num("start", "-90"), "angle": num("angle", "225"), "layers": int(dset.get("layers", 60))}
    return view, post
