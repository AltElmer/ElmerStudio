"""OpenFOAM cell values -> Studio node values, written as the VTU files a Solution reads.

A node on a boundary with a fixed value of the field (wall, inlet, fixed temperature, fixed pressure) takes that
boundary value; where a no-slip wall meets another fixed-velocity boundary the wall wins (as at Elmer's corner
nodes).  Every other node gets a linear least-squares fit to the cell centres around it (at a boundary also the
next layer of cells inward) and to the boundary faces whose value is a physical condition; the fit is exact for a
linear field and second order otherwise.  Pressure goes back from kinematic (p/rho) to Pa.
"""
from __future__ import annotations

import os

import numpy as np

from . import dicts
from .case import FoamPlan

# boundary kinds whose face value is fixed, per field
DIRICHLET = {"U": {"wall", "inlet_normal", "inlet_field"}, "p": {"pressure", "pressure_normal"}, "T": {"temp"}}
# boundary kinds whose face value is a physical condition and enters the fit (flux, Robin, symmetry, slip)
INFORMATIVE = {"U": {"slip", "symmetry"}, "p": {"symmetry"}, "T": {"flux", "conv"}}


class Converter:
    def __init__(self, P: FoamPlan, pm, specs: dict, topo, point_map: np.ndarray):
        self.P, self.pm, self.specs, self.topo = P, pm, specs, topo
        self.point_map = point_map
        self.sd = P.sdim
        nc = pm.n_cells
        # boundary faces in sample order (after the cells)
        self.bfaces = []
        self.bkind = []
        for b, s in sorted(specs.items()):
            self.bfaces.append(s["faces"])
            self.bkind += [s["kind"]] * len(s["faces"])
        self.bfaces = np.concatenate(self.bfaces) if self.bfaces else np.zeros(0, np.int64)
        self.bkind = np.array(self.bkind)
        node_of = np.full(len(pm.points), -1, np.int64)
        node_of[point_map] = np.arange(len(point_map))
        self._node_of = node_of
        # point -> cells (all points: the wider boundary stencils reach the back plane in 2D), node -> boundary slots
        n = len(point_map)
        self.pcells = [[] for _ in range(len(pm.points))]
        for c, row in enumerate(pm.hexes):
            for p in row:
                self.pcells[p].append(c)
        cells_of = [self.pcells[p] for p in point_map]
        # neighbour cells across internal faces, per axis (the grid is axis-aligned)
        self.nbrs = [dict() for _ in range(3)]
        own, nb = pm.owner[:pm.n_internal], pm.neighbour
        ax = np.argmax(np.abs(pm.face_centres[:pm.n_internal] - pm.cell_centres[own]), axis=1)
        for o, q, a in zip(own, nb, ax):
            self.nbrs[a].setdefault(int(o), []).append(int(q))
            self.nbrs[a].setdefault(int(q), []).append(int(o))
        faces_of = [[] for _ in range(n)]
        for k, f in enumerate(self.bfaces):
            for p in pm.faces[f]:
                j = node_of[p]
                if j >= 0:
                    faces_of[j].append(k)
        self.cells_of, self.faces_of = cells_of, faces_of
        self.ncells = nc
        self._ops: dict = {}

    def _operator(self, field: str):
        """(rows, cols, weights): node value = sum of weights * samples (cell values, then boundary face values)."""
        if field in self._ops:
            return self._ops[field]
        pm, sd = self.pm, self.sd
        X = np.vstack([pm.cell_centres, pm.face_centres[self.bfaces]])[:, :sd]
        nodes = self.pm.points[self.point_map][:, :sd]
        tol = 1e-9 * float(np.ptp(pm.points, axis=0).max())
        dirich = DIRICHLET[field]
        rows, cols, wts = [], [], []
        for j in range(len(nodes)):
            fs = self.faces_of[j]
            kinds = self.bkind[fs] if fs else np.array([])
            fixed = [k for k, kd in zip(fs, kinds) if kd in dirich]
            if field == "U":
                walls = [k for k, kd in zip(fs, kinds) if kd == "wall"]
                fixed = walls or fixed
            if fixed:
                idx = np.array(fixed) + self.ncells
                w = np.full(len(idx), 1.0 / len(idx))
            else:
                # the cells around the node, and the boundary faces whose value carries the physics (a zero-gradient
                # face value is a first-order closure, e.g. the pressure at a velocity inlet, and is left out)
                faces = [k + self.ncells for k, kd in zip(fs, kinds) if kd in INFORMATIVE[field]]
                cells = set(self.cells_of[j])
                for _layer in range(2):
                    # a boundary node: its cells lie in one layer along the boundary normal; add the next layer
                    # inward along exactly those axes (never sideways, which would smear the profile)
                    D = X[list(cells) + faces] - nodes[j]
                    flat = [a for a in range(sd) if np.ptp(D[:, a]) <= tol]
                    add = {q for c in cells for a in flat for q in self.nbrs[a].get(c, [])} - cells
                    if not add:
                        break
                    cells |= add
                idx = np.array(sorted(cells) + faces, np.int64)
                D = X[idx] - nodes[j]
                fit = [a for a in range(sd) if np.ptp(D[:, a]) > tol]     # a single cell layer: no slope there
                A = np.hstack([np.ones((len(idx), 1)), D[:, fit] / (np.max(np.abs(D)) or 1.0)])
                w = np.linalg.pinv(A)[0]
            rows.append(np.full(len(idx), j))
            cols.append(idx)
            wts.append(w)
        op = (np.concatenate(rows), np.concatenate(cols), np.concatenate(wts))
        self._ops[field] = op
        return op

    def _apply(self, field, samples):
        r, c, w = self._operator(field)
        n = len(self.point_map)
        if samples.ndim == 1:
            return np.bincount(r, weights=w * samples[c], minlength=n)
        return np.column_stack([np.bincount(r, weights=w * samples[c, i], minlength=n) for i in range(samples.shape[1])])

    def face_values(self, field, cell):
        """Boundary face values implied by each patch's condition (sample order)."""
        out = []
        for b, s in sorted(self.specs.items()):
            kind, cv = s["kind"], cell[s["cells"]]
            if field == "U":
                n = s["normals"]
                un = np.einsum("ij,ij->i", cv, n)[:, None] * n
                v = {"wall": np.zeros_like(cv), "slip": cv - un, "symmetry": cv - un, "pressure": cv,
                     "pressure_normal": un}.get(kind)
                if kind in ("inlet_normal", "inlet_field"):
                    v = s["U"]
            elif field == "p":
                v = s["p"] if kind in ("pressure", "pressure_normal") else cv
            else:
                if kind == "temp":
                    v = s["values"]["T"]
                elif kind == "flux":
                    v = cv + s["values"]["q"] / self.P.k * s["delta"]
                elif kind == "conv":
                    v = s["f"] * s["values"]["Text"] + (1 - s["f"]) * cv
                else:
                    v = cv
            out.append(np.asarray(v, float))
        if not out:
            return np.zeros((0,) + cell.shape[1:])
        return np.concatenate(out)

    def nodal(self, field, cell):
        samples = np.concatenate([cell, self.face_values(field, cell)])
        out = self._apply(field, samples)
        if field == "U":
            self._constrain(out)
        return out

    def _constrain(self, vel):
        """Velocity constraints that hold exactly on the boundary: no normal component on symmetry and slip
        boundaries, no tangential component on a normal-flow outlet (fixed-velocity nodes keep their value)."""
        for j, fs in enumerate(self.faces_of):
            if not fs:
                continue
            kinds = self.bkind[fs]
            if any(k in DIRICHLET["U"] for k in kinds):
                continue
            for k, kd in zip(fs, kinds):
                n = self._face_normal(k)
                if kd in ("slip", "symmetry"):
                    vel[j] -= np.dot(vel[j], n) * n
            for k, kd in zip(fs, kinds):
                if kd == "pressure_normal":
                    n = self._face_normal(k)
                    vel[j] = np.dot(vel[j], n) * n

    def _face_normal(self, k):
        if not hasattr(self, "_bnormals"):
            self._bnormals = np.vstack([s["normals"] for _b, s in sorted(self.specs.items())]) if self.specs \
                else np.zeros((0, 3))
        return self._bnormals[k]

    def convert_time(self, tdir: str) -> dict:
        """Node arrays (SI, Elmer's names) from one time directory."""
        P, nc = self.P, self.pm.n_cells
        out = {}
        if P.physics == "spf":
            U = _read_field(os.path.join(tdir, "U"), 3, nc)
            p = _read_field(os.path.join(tdir, "p"), 1, nc)
            vel = self.nodal("U", U)
            if self.sd == 2:
                vel[:, 2] = 0.0
            out["velocity"] = vel
            out["pressure"] = self.nodal("p", p) * P.rho          # kinematic -> Pa
        else:
            T = _read_field(os.path.join(tdir, "T"), 1, nc)
            out["temperature"] = self.nodal("T", T)
        return out

    def write_vtu(self, arrays: dict, path: str):
        from ...results import Topology
        g = Topology(mesh_result=self.topo).bulk_grid()
        for k, v in arrays.items():
            g.point_data[k] = np.asarray(v, float)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        g.save(path, binary=True)


def _read_field(path: str, width: int, n: int) -> np.ndarray:
    arr, uni = dicts.read_internal_field(path, width)
    if arr is None:
        return np.tile(np.asarray(uni, float), (n, 1)) if width > 1 else np.full(n, float(uni))
    if len(arr) != n:
        raise ValueError(f"{path}: {len(arr)} values for {n} cells")
    return arr
