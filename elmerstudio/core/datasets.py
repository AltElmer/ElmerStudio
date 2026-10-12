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


def _surface(grid, nonlinear_subdivision=1):
    """Linear surface of *grid* whose point data 'vtkOriginalPointIds' really indexes *grid*'s points.

    VTK's own ids are unreliable for 2D quadratic cells, so points are matched by coordinates
    (subdivision level 1 of quadratic cells reuses the existing nodes); unmatched points get -1.
    """
    try:
        s = grid.extract_surface(nonlinear_subdivision=nonlinear_subdivision, algorithm="dataset_surface")
    except TypeError:
        s = grid.extract_surface(nonlinear_subdivision=nonlinear_subdivision)
    s.point_data["vtkOriginalPointIds"] = match_points(grid, np.asarray(s.points, float))
    return s


def match_points(grid, Q, rtol=1e-5):
    """Index of the *grid* point at each query point (-1 when none lies within rtol x model size)."""
    import vtk
    X = np.asarray(grid.points, float)
    tol = rtol * (float(np.linalg.norm(np.ptp(X, axis=0))) or 1.0)
    loc = vtk.vtkStaticPointLocator()
    loc.SetDataSet(grid)
    loc.BuildLocator()
    ids = np.fromiter((loc.FindClosestPoint(q) for q in Q), dtype=np.int64, count=len(Q))
    bad = np.linalg.norm(X[ids] - Q, axis=1) > tol
    ids[bad] = -1
    return ids


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
    # coordinates in the geometry unit; values with units are converted (COMSOL semantics)
    glen = lambda key, d="0": units.geometry_value(dset.get(key, d), model.parameters(),  # noqa: E731
                                                   model.parameter_rows(), view.scale)
    if k == "dset.cutplane":
        if view.sdim != 3:
            raise ValueError("A Cut Plane needs a 3D dataset")
        if dset.get("ptype", "general") == "quick":
            ax = {"yz": 0, "zx": 1, "xy": 2}[dset.get("plane", "yz")]
            o = np.zeros(3)
            o[ax] = glen("coord")
            nrm = np.eye(3)[ax]
        else:
            o = np.array([glen("px"), glen("py"), glen("pz")])
            nrm = np.array([num("nx", "1"), num("ny"), num("nz")])
        return CutPlane(view, o, nrm), post
    if k == "dset.mirror":
        return Mirror(view, dset.get("axis", "x")), post
    if k in ("dset.grid3d", "dset.grid2d"):
        root = _root(view)
        X = np.asarray(root.topo.nodes, float)
        lo, hi = X.min(axis=0), X.max(axis=0)
        d = 3 if k == "dset.grid3d" else 2
        if d == 3 and root.sdim == 2:          # 3D grid around an axisymmetric model: revolve its extent
            rmax = float(np.max(np.abs(X[:, 0])))
            lo, hi = np.array([-rmax, -rmax, lo[1]]), np.array([rmax, rmax, hi[1]])
        lo_, hi_ = [], []
        for j, a in enumerate("xyz"[:d]):
            lo_.append(glen(f"{a}min") if str(dset.get(f"{a}min", "")).strip() else float(lo[j]))
            hi_.append(glen(f"{a}max") if str(dset.get(f"{a}max", "")).strip() else float(hi[j]))
        res = [int(dset.get(f"{a}res", 40) or 40) for a in "xyz"[:d]]
        return GridView(view, lo_, hi_, res), post
    if k == "dset.paramcurve":
        P, s = curve_points(model, dset)
        return CurveView(view, P, s), post
    if k == "dset.revolution":
        return view, {"start": num("start", "-90"), "angle": num("angle", "225"), "layers": int(dset.get("layers", 60))}
    return view, post


# --------------------------------------------------------------------------- grids and curves (no mesh needed)
def _root(view):
    while not isinstance(view, Solution) and hasattr(view, "base"):
        view = view.base
    return view


class _PointView(_View):
    """A dataset of free points: values are interpolated from the mesh and, for exterior-field variables
    (acpr.efc1.pext, ...), taken from the Kirchhoff-Helmholtz integral where there is no mesh."""

    def _points_for_eval(self):
        """Evaluation points (SI) in the root solution's coordinates."""
        P = np.asarray(self.topo.nodes, float) * self.scale
        root = _root(self.base)
        if root.sdim == 2:
            if self.sdim == 3:                                # a 3D curve on an axisymmetric model: (r, z)
                return np.column_stack([np.hypot(P[:, 0], P[:, 1]), P[:, 2]])
            return P[:, :2]
        return P

    def evaluate(self, expr, i=None):
        from .acoustics import point_values
        root = _root(self.base)
        i = root.frame_index(i)
        key = ("e", expr, i)
        if key not in self._cache:
            v = point_values(root, expr, i, self._points_for_eval())
            self._cache[key] = np.asarray(v, float)
        return self._cache[key]

    def evaluate_vector(self, expr, i=None):
        root = _root(self.base)
        i = root.frame_index(i)
        vec = np.asarray(root.evaluate_vector(expr, i), float)
        g = root.bulk_grid().copy()
        P = self._points_for_eval()
        out = []
        for k in range(3):
            g.point_data["f"] = vec[:, k]
            from .results import _probe
            Q = P / root.scale
            if Q.shape[1] == 2:
                Q = np.column_stack([Q, np.zeros(len(Q))])
            out.append(_probe(g, Q))
        V = np.column_stack(out)
        if root.sdim == 2 and self.sdim == 3:            # (r, z) components back to (x, y, z)
            Pc = np.asarray(self.topo.nodes, float)
            phi = np.arctan2(Pc[:, 1], Pc[:, 0])
            V = np.column_stack([V[:, 0] * np.cos(phi), V[:, 0] * np.sin(phi), V[:, 1]])
        return V


class GridView(_PointView):
    """Grid 2D / Grid 3D dataset: a regular grid of points over a box (geometry units)."""

    def __init__(self, base, lo, hi, res):
        super().__init__(base)
        import pyvista as pv
        root = _root(base)
        self.sdim = 3 if len(lo) == 3 else 2
        dims = [max(2, int(r)) for r in res] + ([1] if self.sdim == 2 else [])
        lo3 = list(lo) + ([0.0] if self.sdim == 2 else [])
        hi3 = list(hi) + ([0.0] if self.sdim == 2 else [])
        sp = [(h - l) / (d - 1) if d > 1 else 1.0 for l, h, d in zip(lo3, hi3, dims)]
        img = pv.ImageData(dimensions=dims, spacing=sp, origin=lo3)
        self._bulk = img.cast_to_unstructured_grid()
        self.topo = _Topo(np.asarray(self._bulk.points, float).copy())
        if self.sdim == 3:
            surf = _surface(self._bulk)
            ids = np.asarray(surf.point_data["vtkOriginalPointIds"])
            faces = surf.faces.reshape(-1, 5)[:, 1:]          # quads
            self._bnd = pv.PolyData(self.topo.nodes.copy(), np.hstack([np.full((len(faces), 1), 4), ids[faces]]).ravel())
        else:
            tri = self._bulk.extract_surface().triangulate()
            t = tri.faces.reshape(-1, 4)[:, 1:]
            bl = _boundary_lines(t, len(self.topo.nodes))
            self._bnd = pv.PolyData(self.topo.nodes.copy(), lines=np.hstack([np.full((len(bl), 1), 2), bl]).ravel())
        self._bnd.cell_data["bnd"] = np.zeros(self._bnd.n_cells, int)
        self._bulk.cell_data["dom"] = np.zeros(self._bulk.n_cells, int)
        self.frames = root.frames

    def bulk_grid(self):
        return self._bulk

    def boundary_grid(self):
        return self._bnd


class CurveView(_PointView):
    """Parametric Curve 3D/2D dataset: points x(s), y(s)[, z(s)] (geometry units)."""

    def __init__(self, base, points, s):
        super().__init__(base)
        import pyvista as pv
        P = np.atleast_2d(np.asarray(points, float))
        if P.shape[1] == 2:
            P = np.column_stack([P, np.zeros(len(P))])
        self.sdim = 3
        self.s = np.asarray(s, float)
        self.topo = _Topo(P)
        line = pv.PolyData(P.copy(), lines=np.hstack([[len(P)], np.arange(len(P))]))
        self._bulk = line
        self._bnd = line.copy()
        self._bnd.cell_data["bnd"] = np.zeros(1, int)
        self.frames = _root(base).frames

    def bulk_grid(self):
        return self._bulk

    def boundary_grid(self):
        return self._bnd


def curve_points(model, dset):
    """(points (N,3) geometry units, parameter values) of a Parametric Curve dataset."""
    from . import units
    params = dict(model.parameters())
    name = str(dset.get("par", "s")).strip() or "s"
    s0 = float(units.evaluate(str(dset.get("smin", "0")), params))
    s1 = float(units.evaluate(str(dset.get("smax", "1")), params))
    s = np.linspace(s0, s1, max(2, int(dset.get("npts", 400))))
    names = {**params, name: s}
    cols = [np.broadcast_to(np.asarray(units.evaluate(str(dset.get(k, "0")), names), float), s.shape)
            for k in ("xexpr", "yexpr", "zexpr")]
    return np.column_stack(cols), s
