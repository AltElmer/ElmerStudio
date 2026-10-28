"""The blockMesh mesh read back from constant/polyMesh, and Studio's topology (mesh.npz) built from it."""
from __future__ import annotations

import os

import numpy as np

from . import dicts
from .blocks import BlockLayout

# hexahedron corner order (OpenFOAM/VTK/Gmsh) from the low/high bits (x, y, z) of a corner in an axis-aligned box
_HEX_OF_BITS = [0, 1, 3, 2, 4, 5, 7, 6]


class PolyMesh:
    def __init__(self, case: str):
        pm = os.path.join(case, "constant", "polyMesh")
        self.points = dicts.read_list(os.path.join(pm, "points"), 3)
        self.faces = dicts.read_faces(os.path.join(pm, "faces"))
        self.owner = dicts.read_list(os.path.join(pm, "owner"), 1, np.int64)
        self.neighbour = dicts.read_list(os.path.join(pm, "neighbour"), 1, np.int64)
        self.patches = dicts.read_boundary(os.path.join(pm, "boundary"))
        self.zones = dicts.read_cell_zones(os.path.join(pm, "cellZones"))
        self.n_cells = int(self.owner.max()) + 1 if len(self.owner) else 0
        self.n_internal = len(self.neighbour)
        self.face_centres = self.points[self.faces].mean(axis=1)
        self.hexes = self._hexes()
        self.cell_centres = self.points[self.hexes].mean(axis=1)

    def patch(self, name: str) -> dict | None:
        return next((p for p in self.patches if p["name"] == name), None)

    def patch_faces(self, name: str) -> np.ndarray:
        p = self.patch(name)
        return np.arange(p["startFace"], p["startFace"] + p["nFaces"]) if p else np.zeros(0, np.int64)

    def _hexes(self) -> np.ndarray:
        """Corner points of every (axis-aligned box) cell in hexahedron order."""
        nf = len(self.faces)
        cell_of = np.concatenate([self.owner, self.neighbour])
        face_of = np.concatenate([np.arange(nf), np.arange(self.n_internal)])
        order = np.argsort(cell_of, kind="stable")
        counts = np.bincount(cell_of, minlength=self.n_cells)
        if np.any(counts != 6):
            raise ValueError("blockMesh produced cells that are not hexahedra")
        pts = self.faces[face_of[order]].reshape(self.n_cells, 24)
        P = self.points[pts]
        mid = 0.5 * (P.min(axis=1) + P.max(axis=1))
        bits = ((P[:, :, 0] > mid[:, None, 0]).astype(int) + 2 * (P[:, :, 1] > mid[:, None, 1])
                + 4 * (P[:, :, 2] > mid[:, None, 2]))
        hexes = np.zeros((self.n_cells, 8), np.int64)
        for b in range(8):
            j = np.argmax(bits == b, axis=1)
            if not np.all(bits[np.arange(self.n_cells), j] == b):
                raise ValueError("a blockMesh cell is not an axis-aligned box")
            hexes[:, _HEX_OF_BITS[b]] = pts[np.arange(self.n_cells), j]
        return hexes

    def cell_zone(self) -> np.ndarray:
        """Studio domain number per cell, from the dom<N> cell zones."""
        dom = np.zeros(self.n_cells, np.int64)
        for name, cells in self.zones.items():
            if name.startswith("dom"):
                dom[cells] = int(name[3:])
        if np.any(dom == 0):
            raise ValueError("blockMesh left cells outside every domain zone")
        return dom


def topology(pm: PolyMesh, L: BlockLayout):
    """A MeshResult on the blockMesh mesh, numbered like the geometry.

    2D: the front plane (z = 0) as quadrilaterals and its boundary edges; 3D: hexahedra and boundary quadrilaterals.
    Also returns the node index -> polyMesh point index map."""
    from ...meshing import MeshResult
    geo, sd = L.geo, L.sdim
    pts = pm.points / geo.scale
    dom = pm.cell_zone()
    if sd == 2:
        t = L.coords[2][1]
        front = np.nonzero(np.abs(pts[:, 2]) < 0.5 * t)[0]
        new = np.full(len(pts), -1, np.int64)
        new[front] = np.arange(len(front))
        nodes = pts[front].copy()
        nodes[:, 2] = 0.0
        quads = new[pm.hexes[:, :4]]                 # corners 0-3 are z-low (counter-clockwise in x, y)
        bulk = {int(d): [(3, quads[dom == d])] for d in np.unique(dom)}
        point_map = front
    else:
        nodes = pts.copy()
        bulk = {int(d): [(5, pm.hexes[dom == d])] for d in np.unique(dom)}
        new = np.arange(len(pts))
        point_map = np.arange(len(pts))
    bnd = {}
    centres = pm.face_centres / geo.scale
    for b, faces in L.faces.items():
        if b in L.interior:
            fl = _faces_at(pm, centres, L, faces, internal=True)
        else:
            fl = pm.patch_faces(f"bnd{b}")
        conn = pm.faces[fl]
        if sd == 2:                                  # the two front-plane points of each side face
            keep = np.abs(pts[conn, 2]) < 0.5 * L.coords[2][1]
            conn = np.array([new[row[k]] for row, k in zip(conn, keep)], np.int64).reshape(-1, 2)
            bnd[b] = [(1, conn)]
        else:
            bnd[b] = [(3, conn)]
    point_nodes = {}
    for num, bb in geo.bbox_of["point"].items():
        p = np.array(bb[:3])
        j = int(np.argmin(np.linalg.norm(nodes - p, axis=1)))
        point_nodes[num] = j
    m = MeshResult(sd, geo.unit, geo.scale, nodes, bulk=bulk, bnd=bnd,
                   quality={d: np.ones(sum(len(c) for _t, c in blocks)) for d, blocks in bulk.items()},
                   point_nodes=point_nodes, geometry=geo, order=1)
    m.stats.update({"nodes": len(nodes), "elements": pm.n_cells,
                    "boundary_elements": int(sum(len(c) for v in bnd.values() for _t, c in v)),
                    "types": {"Quadrilateral" if sd == 2 else "Hexahedron": pm.n_cells}, "order": 1,
                    "min_quality": 1.0, "avg_quality": 1.0, "mesher": "blockMesh"})
    return m, point_map


def _faces_at(pm: PolyMesh, centres, L: BlockLayout, faces, internal=False) -> np.ndarray:
    """polyMesh faces lying on the grid faces *faces* (an interior boundary, which has no patch)."""
    out = []
    lim = range(pm.n_internal) if internal else range(len(pm.faces))
    cand = np.asarray(list(lim))
    C = centres[cand]
    for a, p in faces:
        lo = [L.coords[b][p[b]] for b in range(3)]
        hi = [L.coords[b][p[b] + 1] if b != a else L.coords[b][p[b]] for b in range(3)]
        tol = 1e-9 * (max(L.coords[0][-1] - L.coords[0][0], 1e-300))
        sel = np.abs(C[:, a] - lo[a]) <= tol
        for b in range(3):
            if b != a:
                sel &= (C[:, b] > lo[b] - tol) & (C[:, b] < hi[b] + tol)
        out.append(cand[sel])
    return np.concatenate(out) if out else np.zeros(0, np.int64)
