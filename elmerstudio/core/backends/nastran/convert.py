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

    def _nodal_stress_components(self) -> dict[str, np.ndarray]:
        """Element stress components averaged to nodes. For 2D membranes (CTRIA3), rotate from element system
        (x from G1 to G2) to global x-y. For 3D solids (CTETRA), components are already in basic system.
        In 2D plane strain, also compute szz = nu*(sxx+syy)."""
        n = self.n_nodes
        plan = self.plan
        sdim = getattr(self.mesh, "sdim", 3)
        
        # Nodal accumulators
        sxx = np.zeros(n, dtype=float)
        syy = np.zeros(n, dtype=float)
        sxy = np.zeros(n, dtype=float)
        szz = np.zeros(n, dtype=float)
        syz = np.zeros(n, dtype=float)
        sxz = np.zeros(n, dtype=float)
        counts = np.zeros(n, dtype=int)
        
        eid = 1
        for dom, blocks in self.mesh.bulk.items():
            nu = plan.domain_nu.get(dom, 0.3) if plan is not None else 0.3
            for gt, conn in blocks:
                for row in conn:
                    for card, pos in BULK_CELLS.get(gt, ()):
                        if sdim == 2 and card == "CTRIA3":
                            # 2D membrane: stress components in element system, need to rotate to global
                            comp = self.res.membrane_stress_comp.get(eid)
                            if comp is not None:
                                nx, ny, sxy_elem = comp
                                # Element x-axis is from G1 to G2
                                g1, g2, g3 = row[pos[0]], row[pos[1]], row[pos[2]]
                                p1 = self.mesh.nodes[g1]
                                p2 = self.mesh.nodes[g2]
                                ex = p2 - p1
                                ex_len = np.linalg.norm(ex)
                                if ex_len > 1e-12:
                                    ex = ex / ex_len
                                    # Rotation matrix from element to global
                                    c, s = ex[0], ex[1]
                                    # Stress transformation: sigma_global = R^T * sigma_elem * R
                                    # where element x-axis is at angle theta from global x-axis
                                    # sxx = nx*c^2 + ny*s^2 - 2*sxy*c*s
                                    # syy = nx*s^2 + ny*c^2 + 2*sxy*c*s
                                    # sxy = (nx - ny)*c*s + sxy*(c^2 - s^2)
                                    c2, s2 = c*c, s*s
                                    cs = c*s
                                    sxx_g = nx*c2 + ny*s2 - 2*sxy_elem*cs
                                    syy_g = nx*s2 + ny*c2 + 2*sxy_elem*cs
                                    sxy_g = (nx - ny)*cs + sxy_elem*(c2 - s2)
                                else:
                                    sxx_g, syy_g, sxy_g = nx, ny, sxy_elem
                                
                                if plan is not None and plan.plane_strain:
                                    szz_g = nu * (sxx_g + syy_g)
                                else:
                                    szz_g = 0.0
                                
                                for p in pos:
                                    sxx[p] += sxx_g
                                    syy[p] += syy_g
                                    sxy[p] += sxy_g
                                    szz[p] += szz_g
                                    counts[p] += 1
                        
                        elif sdim == 3 and card == "CTETRA":
                            # 3D solid: stress components already in basic system
                            comp = self.res.solid_stress_comp.get(eid)
                            if comp is not None:
                                sx, sy, sz, tyz, txz, txy = comp
                                for p in pos:
                                    sxx[p] += sx
                                    syy[p] += sy
                                    szz[p] += sz
                                    syz[p] += tyz
                                    sxz[p] += txz
                                    sxy[p] += txy
                                    counts[p] += 1
                        
                        eid += 1
        
        mask = counts > 0
        out = {}
        if mask.any():
            out["sx"] = np.zeros(n)
            out["sy"] = np.zeros(n)
            out["sxy"] = np.zeros(n)
            out["sx"][mask] = sxx[mask] / counts[mask]
            out["sy"][mask] = syy[mask] / counts[mask]
            out["sxy"][mask] = sxy[mask] / counts[mask]
            if plan is not None and plan.plane_strain:
                out["szz"] = np.zeros(n)
                out["szz"][mask] = szz[mask] / counts[mask]
            if sdim == 3:
                out["sz"] = np.zeros(n)
                out["syz"] = np.zeros(n)
                out["sxz"] = np.zeros(n)
                out["sz"][mask] = szz[mask] / counts[mask]
                out["syz"][mask] = syz[mask] / counts[mask]
                out["sxz"][mask] = sxz[mask] / counts[mask]
        return out

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
        
        # Reaction forces (SPC forces)
        if self.res.reactions:
            react = self._vectors(self.res.reactions)
            out["reaction_force"] = react
            out["rx"] = react[:, 0]
            out["ry"] = react[:, 1]
            out["rz"] = react[:, 2]
        
        out["vonmises"] = self._nodal_mises()
        
        # Stress components
        stress_comp = self._nodal_stress_components()
        out.update(stress_comp)
        
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