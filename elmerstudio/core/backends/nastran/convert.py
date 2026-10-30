"""Convert NASTRAN parsed results into ElmerStudio VTU format and mesh.npz."""
from __future__ import annotations

import os
import numpy as np

from ...results import Topology, topology_to_npz
from .deck import BULK_CELLS, NastranPlan
from .parser import NastranResult


class NastranConverter:
    """Converts NastranResult into ElmerStudio solution format (VTU, mesh.npz)."""

    def __init__(self, mesh, result: NastranResult, plan: NastranPlan | None = None):
        self.mesh = mesh
        self.res = result
        self.plan = plan
        self.n_nodes = len(mesh.nodes)

    def _vectors(self, table: dict[int, list[float]]) -> np.ndarray:
        """Grid-point vectors (1-based IDs) as an (N, 3) array in mesh node order."""
        out = np.zeros((self.n_nodes, 3), dtype=float)
        sdim = getattr(self.mesh, "sdim", 3)
        for nid, vec in table.items():
            if 0 < nid <= self.n_nodes:
                out[nid - 1, :sdim] = vec[:sdim]
        return out

    def _nodal_mises(self) -> np.ndarray:
        """Element von Mises stress averaged to the nodes of each element that reported one.  Element IDs follow
        the deck's split of every mesh element into linear cells (BULK_CELLS)."""
        n = self.n_nodes
        plan = self.plan
        mises = np.zeros(n, dtype=float)
        counts = np.zeros(n, dtype=int)
        eid = 1
        for dom, blocks in self.mesh.bulk.items():
            nu = plan.domain_nu.get(dom, 0.3) if plan is not None else 0.3
            for gt, conn in blocks:
                for row in conn:
                    for _card, pos in BULK_CELLS.get(gt, ()):
                        vm = self.res.stresses.get(eid)
                        if vm is not None and plan is not None and plan.plane_strain and eid in self.res.principal:
                            # plane strain: the out-of-plane stress nu*(s1 + s2) enters the equivalent stress
                            s1, s2 = self.res.principal[eid]
                            s3 = nu * (s1 + s2)
                            vm = (((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2) / 2) ** 0.5
                        if vm is not None:
                            for p in pos:
                                mises[row[p]] += vm
                                counts[row[p]] += 1
                        eid += 1
        mask = counts > 0
        mises[mask] /= counts[mask]
        return mises

    def extract_point_data(self) -> dict[str, np.ndarray]:
        """Convert result dictionaries into nodal numpy arrays matching mesh.nodes order."""
        out: dict[str, np.ndarray] = {}
        if self.res.temperatures:
            temp = np.zeros(self.n_nodes, dtype=float)
            for nid, val in self.res.temperatures.items():
                if 0 < nid <= self.n_nodes:
                    temp[nid - 1] = val
            out["temperature"] = temp
            out["t"] = temp
            out["T"] = temp
            return out

        if self.res.eigenvectors:
            # Elmer's convention for eigenmodes: all modes in one file, "<variable> EigenMode<k>"
            for k, table in sorted(self.res.eigenvectors.items()):
                disp = self._vectors(table)
                for name, arr in self._displacement_arrays(disp).items():
                    out[f"{name} EigenMode{k}"] = arr
                    if k == 1:
                        out[name] = arr
            return out

        out.update(self._displacement_arrays(self._vectors(self.res.displacements)))
        out["vonmises"] = self._nodal_mises()
        return out

    @staticmethod
    def _displacement_arrays(disp: np.ndarray) -> dict[str, np.ndarray]:
        return {"displacement": disp, "u": disp[:, 0], "v": disp[:, 1], "w": disp[:, 2],
                "disp_mag": np.linalg.norm(disp, axis=1)}

    def write_vtu(self, arrays: dict[str, np.ndarray], path: str):
        """Write VTU file using ElmerStudio's Topology and PyVista."""
        g = Topology(mesh_result=self.mesh).bulk_grid()
        for k, v in arrays.items():
            g.point_data[k] = np.asarray(v, float)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        g.save(path, binary=True)

    def save_mesh_npz(self, path: str):
        """Write mesh.npz."""
        topology_to_npz(self.mesh, path)
