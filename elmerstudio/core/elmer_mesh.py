"""Writer for the native Elmer mesh format (mesh.header/nodes/elements/boundary).

Boundary elements carry their two parent bulk elements, computed by face matching,
so Elmer features that need parents (fluxes, discontinuities, flux BCs) work.
"""
from __future__ import annotations

import os

import numpy as np

# gmsh element type -> (elmer type, node permutation or None)
GMSH_TO_ELMER = {
    15: (101, None),
    1: (202, None), 8: (203, None),
    2: (303, None), 9: (306, None),
    3: (404, None), 16: (408, None), 10: (409, None),
    4: (504, None), 11: (510, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]),
    7: (605, None),
    6: (706, None),
    5: (808, None),
}
CORNERS = {101: 1, 202: 2, 203: 2, 303: 3, 306: 3, 404: 4, 408: 4, 409: 4, 504: 4, 510: 4, 605: 5,
           706: 6, 808: 8}

# faces (in corner indices) of bulk element types, grouped by face size
FACES = {
    303: {2: [(0, 1), (1, 2), (2, 0)]},
    306: {2: [(0, 1), (1, 2), (2, 0)]},
    404: {2: [(0, 1), (1, 2), (2, 3), (3, 0)]},
    408: {2: [(0, 1), (1, 2), (2, 3), (3, 0)]},
    409: {2: [(0, 1), (1, 2), (2, 3), (3, 0)]},
    504: {3: [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]},
    510: {3: [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]},
    605: {4: [(0, 1, 2, 3)], 3: [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]},
    706: {3: [(0, 1, 2), (3, 4, 5)], 4: [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]},
    808: {4: [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]},
    202: {1: [(0,), (1,)]},
    203: {1: [(0,), (1,)]},
}


class ElmerMesh:
    """Container: nodes (N,3) in meters, bulk and boundary element blocks (1-based node ids)."""

    def __init__(self):
        self.nodes = np.zeros((0, 3))
        self.bulk: list[tuple[int, int, np.ndarray]] = []       # (body id, elmer type, conn (M,k))
        self.boundary: list[tuple[int, int, np.ndarray]] = []   # (bc id, elmer type, conn)

    @property
    def n_bulk(self):
        return sum(len(c) for _, _, c in self.bulk)

    @property
    def n_boundary(self):
        return sum(len(c) for _, _, c in self.boundary)

    def compute_parents(self):
        """Return list of (M,2) parent arrays aligned with self.boundary blocks."""
        bulk_faces: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        offset = 0
        for _, etype, conn in self.bulk:
            ids = np.arange(offset + 1, offset + len(conn) + 1)
            offset += len(conn)
            for k, flist in FACES.get(etype, {}).items():
                for f in flist:
                    keys = np.sort(conn[:, list(f)], axis=1)
                    bulk_faces.setdefault(k, []).append((keys, ids))
        parents = []
        lookup: dict[int, tuple] = {}
        for k, chunks in bulk_faces.items():
            keys = np.vstack([c[0] for c in chunks])
            owners = np.concatenate([c[1] for c in chunks])
            lookup[k] = (keys, owners)
        for _, etype, conn in self.boundary:
            k = CORNERS[etype] if etype != 101 else 1
            par = np.zeros((len(conn), 2), dtype=np.int64)
            if k in lookup and len(conn):
                keys, owners = lookup[k]
                bkeys = np.sort(conn[:, :k], axis=1)
                allk = np.vstack([keys, bkeys])
                _, inv = np.unique(allk, axis=0, return_inverse=True)
                inv = inv.ravel()
                fid, bid = inv[:len(keys)], inv[len(keys):]
                order = np.argsort(fid, kind="stable")
                fs, os_ = fid[order], owners[order]
                start = np.searchsorted(fs, bid, side="left")
                end = np.searchsorted(fs, bid, side="right")
                has1 = end > start
                par[has1, 0] = os_[start[has1]]
                has2 = end - start >= 2
                par[has2, 1] = os_[start[has2] + 1]
            parents.append(par)
        return parents

    def write(self, directory: str, names: dict | None = None):
        os.makedirs(directory, exist_ok=True)
        nn = len(self.nodes)
        nb, nbd = self.n_bulk, self.n_boundary
        counts: dict[int, int] = {}
        for _, t, c in self.bulk + self.boundary:
            counts[t] = counts.get(t, 0) + len(c)
        with open(os.path.join(directory, "mesh.header"), "w") as f:
            f.write(f"{nn} {nb} {nbd}\n{len(counts)}\n")
            for t in sorted(counts):
                f.write(f"{t} {counts[t]}\n")
        ids = np.arange(1, nn + 1)
        np.savetxt(os.path.join(directory, "mesh.nodes"),
                   np.column_stack([ids, -np.ones(nn, int), self.nodes]),
                   fmt=["%d", "%d", "%.15g", "%.15g", "%.15g"])
        with open(os.path.join(directory, "mesh.elements"), "w") as f:
            eid = 1
            for body, t, conn in self.bulk:
                n = len(conn)
                arr = np.column_stack([np.arange(eid, eid + n), np.full(n, body), np.full(n, t), conn])
                np.savetxt(f, arr, fmt="%d")
                eid += n
        parents = self.compute_parents()
        with open(os.path.join(directory, "mesh.boundary"), "w") as f:
            eid = nb + 1
            for (bc, t, conn), par in zip(self.boundary, parents):
                n = len(conn)
                arr = np.column_stack([np.arange(eid, eid + n), np.full(n, bc), par, np.full(n, t), conn])
                np.savetxt(f, arr, fmt="%d")
                eid += n
        if names:
            with open(os.path.join(directory, "mesh.names"), "w") as f:
                f.write("! ----- names for bodies -----\n")
                for nm, i in names.get("bodies", {}).items():
                    f.write(f"$ {nm} = {i}\n")
                f.write("! ----- names for boundaries -----\n")
                for nm, i in names.get("boundaries", {}).items():
                    f.write(f"$ {nm} = {i}\n")
