"""Moving meshes for deforming electrodes (COMSOL: Deforming Domain + Deforming Electrode Surface).

The electrode boundary moves along its normal; the rest of the mesh follows by a harmonic extension of the
boundary displacement (each component solves a Laplace equation on the undeformed-step mesh, stiffened by the
inverse element size so that small elements near the electrode move rigidly instead of collapsing).  Other
boundaries of the deforming domains are fixed, except straight axis-aligned walls, symmetry planes and the
axis of a 2D axisymmetric model, along which the mesh slides.

Everything works on the Topology arrays (gmsh element types, node coordinates in geometry units); quadratic
elements are split into linear sub-elements for the smoothing, so their mid-side nodes move consistently.
"""
from __future__ import annotations

import math

import numpy as np

# gmsh element type -> linear sub-simplices (node indices into the element)
_SUB = {
    1: [(0, 1)], 8: [(0, 2), (2, 1)],                                            # lines
    2: [(0, 1, 2)], 9: [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)],              # triangles
    3: [(0, 1, 2), (0, 2, 3)],                                                    # quadrangles
    16: [(0, 4, 7), (4, 1, 5), (5, 2, 6), (6, 3, 7), (4, 5, 6), (4, 6, 7)],
    10: [(0, 4, 8), (0, 8, 7), (4, 1, 5), (4, 5, 8), (8, 5, 2), (8, 2, 6), (7, 8, 6), (7, 6, 3)],
    4: [(0, 1, 2, 3)],                                                            # tetrahedra
    11: [(0, 4, 6, 7), (4, 1, 5, 9), (6, 5, 2, 8), (7, 9, 8, 3),
         (4, 8, 5, 9), (4, 8, 9, 7), (4, 8, 7, 6), (4, 8, 6, 5)],
}
# corner nodes of boundary elements (2D: lines, 3D: faces) and their faces/edges as seen from the parent element
_CORNERS = {1: 2, 8: 2, 2: 3, 9: 3, 3: 4, 16: 4, 10: 4, 4: 4, 11: 4}
_FACETS = {   # bulk type -> corner-index tuples of its boundary facets
    2: [(0, 1), (1, 2), (2, 0)], 9: [(0, 1), (1, 2), (2, 0)],
    3: [(0, 1), (1, 2), (2, 3), (3, 0)], 16: [(0, 1), (1, 2), (2, 3), (3, 0)], 10: [(0, 1), (1, 2), (2, 3), (3, 0)],
    4: [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], 11: [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
}


class MeshMotionError(RuntimeError):
    pass


def _simplices(blocks, sdim):
    out = []
    for gt, conn in blocks:
        sub = _SUB.get(int(gt))
        if sub is None:
            raise MeshMotionError("Deforming domains need triangular, quadrilateral or tetrahedral elements "
                                  f"(found gmsh element type {gt})")
        if len(sub[0]) != sdim + 1:
            continue
        conn = np.asarray(conn, np.int64)
        out.append(np.vstack([conn[:, list(s)] for s in sub]))
    return np.vstack(out) if out else np.zeros((0, sdim + 1), np.int64)


def signed_measures(nodes, simp, sdim):
    P = np.asarray(nodes, float)[:, :sdim][simp]
    E = P[:, 1:] - P[:, :1]
    if sdim == 2:
        return 0.5 * (E[:, 0, 0] * E[:, 1, 1] - E[:, 0, 1] * E[:, 1, 0])
    return np.einsum("ij,ij->i", E[:, 0], np.cross(E[:, 1], E[:, 2])) / 6.0


def shape_quality(nodes, simp, sdim):
    """Normalised shape quality of each simplex: 1 for an equilateral one, 0 when degenerate (negative if folded)."""
    P = np.asarray(nodes, float)[:, :sdim][simp]
    vol = signed_measures(nodes, simp, sdim)
    k = sdim + 1
    l2 = sum(np.sum((P[:, a] - P[:, b]) ** 2, axis=1) for a in range(k) for b in range(a + 1, k))
    if sdim == 2:
        return 4 * math.sqrt(3) * vol / np.where(l2 > 0, l2, 1.0)
    return 6 * math.sqrt(2) * vol / np.where(l2 > 0, (l2 / 6) ** 1.5, 1.0)


def _stiffness(nodes, simp, sdim):
    """COO rows, cols, values of the inverse-size-stiffened P1 Laplacian on the simplices."""
    P = np.asarray(nodes, float)[:, :sdim][simp]
    E = P[:, 1:] - P[:, :1]                                    # (n, d, d) edge vectors
    vol = np.abs(signed_measures(nodes, simp, sdim))
    ok = vol > 0
    E, simp, vol = E[ok], simp[ok], vol[ok]
    Einv = np.linalg.inv(E)                                    # rows of E^-T are gradients of barycentrics 1..d
    G = np.concatenate([-Einv.sum(axis=2, keepdims=True), Einv], axis=2).transpose(0, 2, 1)   # (n, d+1, d)
    w = (vol.mean() / vol)                                     # small elements are stiffer
    K = np.einsum("nid,njd->nij", G, G) * (vol * w)[:, None, None]
    k = sdim + 1
    rows = np.repeat(simp, k, axis=1).ravel()
    cols = np.tile(simp, (1, k)).ravel()
    return rows, cols, K.ravel()


def _coo_sum(rows, cols, vals, n):
    key = rows * n + cols
    order = np.argsort(key, kind="stable")
    key, vals = key[order], vals[order]
    uniq, start = np.unique(key, return_index=True)
    return uniq // n, uniq % n, np.add.reduceat(vals, start)


def _cg(matvec, b, diag, tol=1e-7, maxit=20000, x0=None):
    """Jacobi-preconditioned conjugate gradients; stops when the residual has dropped by *tol* from its start
    (a warm start is already close: what is left to resolve is the step's increment)."""
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - matvec(x)
    nb = np.linalg.norm(r)
    if nb <= 1e-14 * (np.linalg.norm(b) or 1.0):
        return x
    dinv = np.where(diag > 0, 1.0 / np.where(diag > 0, diag, 1.0), 1.0)
    z = dinv * r
    p = z.copy()
    rz = r @ z
    for _ in range(maxit):
        Ap = matvec(p)
        a = rz / (p @ Ap)
        x += a * p
        r -= a * Ap
        if np.linalg.norm(r) <= tol * nb:
            break
        z = dinv * r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


class MeshMover:
    """Moves the nodes of the deforming domains of one Topology, step after step (the connectivity never changes)."""

    def __init__(self, topo, domains, moving_boundaries, slide_straight=True, sliding_boundaries=(), axi=False):
        self.topo = topo
        self.sdim = int(topo.sdim)
        self.domains = set(int(d) for d in domains)
        self.moving = set(int(b) for b in moving_boundaries)
        n = len(topo.nodes)
        sd = self.sdim
        self.simp = np.vstack([_simplices(topo.bulk[d], sd) for d in sorted(self.domains) if d in topo.bulk]
                              or [np.zeros((0, sd + 1), np.int64)])
        if not len(self.simp):
            raise MeshMotionError("The deforming domains have no mesh elements")
        in_dom = np.zeros(n, bool)
        in_dom[np.unique(self.simp)] = True
        # facets of deforming-domain elements -> number of such parents, and an inside point for orientation
        self._parent_centroid: dict = {}
        count: dict = {}
        for d in self.domains:
            for gt, conn in topo.bulk.get(d, []):
                fac = _FACETS.get(int(gt))
                if fac is None:
                    continue
                nc = _CORNERS[int(gt)]
                corners = np.asarray(conn)[:, :nc]
                for f in fac:
                    keys = np.sort(np.asarray(conn)[:, list(f)], axis=1)
                    for key, cn in zip(map(tuple, keys), corners):
                        count[key] = count.get(key, 0) + 1
                        # the parent's corner nodes, not its centroid: the inside point must follow the moving mesh,
                        # or a boundary that moves past its parent's initial centroid flips its normal
                        self._parent_centroid[key] = cn
        # boundary nodes: moving, sliding (per component), fixed
        self.fix = np.zeros((n, 3), bool)                     # component constrained
        self.fix[~in_dom] = True
        if sd == 2:
            self.fix[:, 2] = True
        self.moving_nodes = np.zeros(n, bool)
        sliding_boundaries = set(int(b) for b in sliding_boundaries)
        nodes = np.asarray(topo.nodes, float)
        span = float(np.ptp(nodes[:, :sd], axis=0).max()) or 1.0
        self.slide_axis: dict = {}
        for b, blocks in topo.bnd.items():
            b = int(b)
            ids = np.unique(np.concatenate([np.asarray(c).ravel() for _gt, c in blocks]))
            ids = ids[in_dom[ids]]
            if not len(ids):
                continue
            interior = all(count.get(tuple(sorted(row[:_CORNERS[int(gt)]])), 0) > 1
                           for gt, c in blocks for row in np.asarray(c)[:1])
            if b in self.moving:
                self.moving_nodes[ids] = True
                continue
            if interior and b not in sliding_boundaries:
                continue                                       # between two deforming domains: free
            axis = None
            if slide_straight or b in sliding_boundaries:
                for a in range(sd):
                    if np.ptp(nodes[ids, a]) <= 1e-9 * span:
                        axis = a
                        break
            if axis is None:
                self.fix[ids] = True
            else:
                self.slide_axis[b] = axis
                self.fix[ids, axis] = True
        if axi:                                                # the symmetry axis r = 0: slide along z
            on_axis = in_dom & (np.abs(nodes[:, 0]) <= 1e-9 * span)
            self.fix[on_axis, 0] = True
        self.moving_bnd_blocks = [(gt, np.asarray(c)) for b in sorted(self.moving) for gt, c in topo.bnd.get(b, [])]
        if not self.moving_bnd_blocks:
            raise MeshMotionError("No moving electrode boundary lies on a deforming domain")

    # ------------------------------------------------------------------ normals
    def normals(self, nodes):
        """Unit normals at the moving boundary nodes, pointing out of the deforming domains (into the metal)."""
        nodes = np.asarray(nodes, float)
        sd = self.sdim
        acc = np.zeros((len(nodes), 3))
        for gt, conn in self.moving_bnd_blocks:
            gt = int(gt)
            nc = _CORNERS[gt]
            for row in conn:
                key = tuple(sorted(row[:nc]))
                parent = self._parent_centroid.get(key)
                cin = nodes[parent].mean(axis=0) if parent is not None else None
                P = nodes[row]
                if sd == 2:
                    a, b = P[0, :2], P[1, :2]
                    if gt == 8:                           # quadratic edge: tangents at its three nodes
                        m = P[2, :2]
                        tans = [(-1.5 * a - 0.5 * b + 2 * m), (0.5 * a + 1.5 * b - 2 * m), (b - a) / 2]
                    else:
                        tans = [b - a, b - a]
                    length = np.linalg.norm(b - a)
                    mid = (a + b) / 2
                    for j, t in enumerate(tans):
                        nv = np.array([t[1], -t[0], 0.0])
                        if cin is not None and np.dot(nv[:2], mid - cin[:2]) < 0:
                            nv = -nv
                        nn = np.linalg.norm(nv)
                        if nn > 0:
                            acc[row[j]] += nv / nn * length
                else:
                    c0, c1, c2 = P[0], P[1], P[2]
                    nv = np.cross(c1 - c0, c2 - c0)
                    cen = P[:nc].mean(axis=0)
                    if cin is not None and np.dot(nv, cen - cin) < 0:
                        nv = -nv
                    for j in range(len(row)):
                        acc[row[j]] += nv            # area-weighted (|nv| = 2 x area)
        nrm = np.linalg.norm(acc, axis=1)
        out = np.zeros_like(acc)
        on = nrm > 0
        out[on] = acc[on] / nrm[on, None]
        return out

    # ------------------------------------------------------------------ motion
    def boundary_displacement(self, nodes, speed, dt):
        """Normal displacement speed*dt*n at the moving nodes (speed per node, SI m/s; dt s), geometry units."""
        n = self.normals(nodes)
        d = (np.asarray(speed, float) * dt)[:, None] * n / float(self.topo.scale)
        d[~self.moving_nodes] = 0.0
        return d

    def _operator(self):
        """The smoothing operator, assembled once on the reference (initial or last remeshed) shape."""
        if getattr(self, "_K", None) is None:
            ref = np.asarray(self.topo.nodes, float)
            n = len(ref)
            rows, cols, vals = _coo_sum(*_stiffness(ref, self.simp, self.sdim), n)
            self._K = []
            for c in range(self.sdim):
                fixed = self.fix[:, c] | self.moving_nodes
                free = ~fixed
                fr = free[rows] & free[cols]
                r2, c2, v2 = rows[fr], cols[fr], vals[fr]
                diag = np.bincount(r2[r2 == c2], v2[r2 == c2], minlength=n)
                self._K.append((fixed, free, (rows, cols, vals), (r2, c2, v2), diag))
            self._ref = ref
        return self._K

    def extend(self, nodes, dbnd):
        """Displacement of every node (geometry units) for the prescribed displacement *dbnd* of the moving nodes.

        Solved for the TOTAL displacement from the reference shape, with the reference shape's operator (assembled
        once) and the present total as the starting guess: one step changes the total by a small increment, so a few
        conjugate-gradient iterations are enough."""
        nodes = np.asarray(nodes, float)
        n = len(nodes)
        K = self._operator()
        ref = self._ref
        target = nodes + dbnd - ref                          # total displacement wanted at the moving nodes
        now = nodes - ref
        disp = np.zeros((n, 3))
        for c in range(self.sdim):
            fixed, free, (rows, cols, vals), (r2, c2, v2), diag = K[c]
            val = np.where(self.moving_nodes, target[:, c], now[:, c])
            val[self.fix[:, c]] = 0.0                       # sliding walls: the constrained component stays at 0
            if not free.any():
                disp[:, c] = val
                continue
            bfix = np.where(fixed, val, 0.0)
            b = -np.bincount(rows, vals * bfix[cols], minlength=n)
            b[fixed] = 0.0

            def mv(x, r2=r2, c2=c2, v2=v2):
                return np.bincount(r2, v2 * x[c2], minlength=n)

            x = _cg(mv, b, diag, x0=np.where(free, now[:, c], 0.0))
            disp[:, c] = np.where(free, x, val)
        return disp + ref - nodes

    def check(self, nodes):
        """Raise if a sub-element turned over; returns the smallest measure ratio (current / initial)."""
        return signed_measures(nodes, self.simp, self.sdim)
