"""Convert NASTRAN parsed results into ElmerStudio VTU format and mesh.npz."""
from __future__ import annotations

import os
import numpy as np

from ...results import Topology, topology_to_npz
from .parser import NastranResult


class NastranConverter:
    """Converts NastranResult into ElmerStudio solution format (VTU, mesh.npz)."""

    def __init__(self, mesh, result: NastranResult):
        self.mesh = mesh
        self.res = result
        self.n_nodes = len(mesh.nodes)

    def extract_point_data(self) -> dict[str, np.ndarray]:
        """Convert result dictionaries into nodal numpy arrays matching mesh.nodes order."""
        n = self.n_nodes
        sdim = getattr(self.mesh, "sdim", 3)

        # 1. Displacement (N, 3)
        disp = np.zeros((n, 3), dtype=float)
        for nid, vec in self.res.displacements.items():
            idx = nid - 1  # 1-based to 0-based
            if 0 <= idx < n:
                disp[idx, 0] = vec[0]
                disp[idx, 1] = vec[1]
                disp[idx, 2] = vec[2] if len(vec) > 2 and sdim == 3 else 0.0

        out = {
            "displacement": disp,
            "u": disp[:, 0],
            "v": disp[:, 1],
            "w": disp[:, 2],
            "disp_mag": np.linalg.norm(disp, axis=1),
        }

        # 2. Temperature (N,)
        if self.res.temperatures:
            temp = np.zeros(n, dtype=float)
            for nid, val in self.res.temperatures.items():
                idx = nid - 1
                if 0 <= idx < n:
                    temp[idx] = val
            out["temperature"] = temp
            out["t"] = temp
            out["T"] = temp

        # 3. von Mises Stress (N,) - element to node averaging
        # Only average over elements that actually have stress data to avoid
        # boundary nodes getting pulled down by missing/zero stress elements
        mises = np.zeros(n, dtype=float)
        node_counts = np.zeros(n, dtype=int)

        eid = 1
        for _dom_id, blocks in self.mesh.bulk.items():
            for _gt, conn in blocks:
                for row in conn:
                    st = self.res.stresses.get(eid)
                    if st is not None:
                        for nid in row:
                            if 0 <= nid < n:
                                mises[nid] += st
                                node_counts[nid] += 1
                    eid += 1

        mask = node_counts > 0
        mises[mask] /= node_counts[mask]
        out["vonmises"] = mises

        return out

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
