"""Solutions, result variables (COMSOL-style names), expression evaluation and derived values.

A solved study lives in a folder:  meta.json, mesh.npz (topology with domain/boundary
numbers) and results/*.vtu written by ElmerSolver. Field values from the VTU files are
mapped onto our own topology, so plots and integrals can address domains/boundaries
by the same numbers the geometry uses.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np

from . import units

# --------------------------------------------------------------------------- variable catalog
# alias, Elmer array, component, label, SI unit, kind
CATALOG = [
    ("T", "temperature", None, "Temperature", "K", "scalar"),
    ("ht.tflux", "temperature flux", "vec", "Conductive heat flux", "W/m^2", "vector"),
    ("ht.tfluxMag", "temperature flux", "mag", "Conductive heat flux magnitude", "W/m^2", "scalar"),
    ("ht.tfluxx", "temperature flux", 0, "Conductive heat flux, x component", "W/m^2", "scalar"),
    ("ht.tfluxy", "temperature flux", 1, "Conductive heat flux, y component", "W/m^2", "scalar"),
    ("ht.tfluxz", "temperature flux", 2, "Conductive heat flux, z component", "W/m^2", "scalar"),
    ("solid.u_vec", "displacement", "vec", "Displacement field", "m", "vector"),
    ("solid.disp", "displacement", "mag", "Displacement magnitude", "m", "scalar"),
    ("u", "displacement", 0, "Displacement field, x component", "m", "scalar"),
    ("v", "displacement", 1, "Displacement field, y component", "m", "scalar"),
    ("w", "displacement", 2, "Displacement field, z component", "m", "scalar"),
    ("solid.mises", "vonmises", None, "von Mises stress", "Pa", "scalar"),
    ("solid.sx", "stress", 0, "Stress tensor, xx component", "Pa", "scalar"),
    ("solid.sy", "stress", 1, "Stress tensor, yy component", "Pa", "scalar"),
    ("solid.sz", "stress", 2, "Stress tensor, zz component", "Pa", "scalar"),
    ("solid.sxy", "stress", 3, "Stress tensor, xy component", "Pa", "scalar"),
    ("solid.syz", "stress", 4, "Stress tensor, yz component", "Pa", "scalar"),
    ("solid.sxz", "stress", 5, "Stress tensor, xz component", "Pa", "scalar"),
    ("V", "potential", None, "Electric potential", "V", "scalar"),
    ("es.E", "electric field", "vec", "Electric field", "V/m", "vector"),
    ("es.normE", "electric field", "mag", "Electric field norm", "V/m", "scalar"),
    ("es.Ex", "electric field", 0, "Electric field, x component", "V/m", "scalar"),
    ("es.Ey", "electric field", 1, "Electric field, y component", "V/m", "scalar"),
    ("es.Ez", "electric field", 2, "Electric field, z component", "V/m", "scalar"),
    ("es.normD", "electric flux", "mag", "Electric displacement field norm", "C/m^2", "scalar"),
    ("es.We", "electric energy density", None, "Electric energy density", "J/m^3", "scalar"),
    ("ec.J", "volume current", "vec", "Current density", "A/m^2", "vector"),
    ("ec.normJ", "volume current", "mag", "Current density norm", "A/m^2", "scalar"),
    ("ec.Qrh", "joule heating", None, "Volumetric loss density, electric", "W/m^3", "scalar"),
    ("Az", "az", None, "Magnetic vector potential, z component", "Wb/m", "scalar"),
    ("mf.B", "magnetic flux density", "vec", "Magnetic flux density", "T", "vector"),
    ("mf.normB", "magnetic flux density", "mag", "Magnetic flux density norm", "T", "scalar"),
    ("mf.normH", "magnetic field strength", "mag", "Magnetic field norm", "A/m", "scalar"),
    ("spf.u_vec", "velocity", "vec", "Velocity field", "m/s", "vector"),
    ("spf.U", "velocity", "mag", "Velocity magnitude", "m/s", "scalar"),
    ("spf.u", "velocity", 0, "Velocity field, x component", "m/s", "scalar"),
    ("spf.v", "velocity", 1, "Velocity field, y component", "m/s", "scalar"),
    ("spf.w", "velocity", 2, "Velocity field, z component", "m/s", "scalar"),
    ("p", "pressure", None, "Pressure", "Pa", "scalar"),
    ("acpr.p_t", "pressure wave", 0, "Total acoustic pressure (real part)", "Pa", "scalar"),
    ("acpr.p_im", "pressure wave", 1, "Total acoustic pressure (imaginary part)", "Pa", "scalar"),
    ("acpr.absp", "pressure wave", "mag", "Absolute pressure amplitude", "Pa", "scalar"),
    ("acpr.Lp", "pressure wave", "spl", "Sound pressure level", "dB", "scalar"),
    ("c", "concentration", None, "Concentration", "mol/m^3", "scalar"),
    ("cd.phil", "phil", None, "Electrolyte potential", "V", "scalar"),
    ("cd.Il", "phil flux", "vec", "Electrolyte current density vector", "A/m^2", "vector"),
    ("cd.IlMag", "phil flux", "mag", "Electrolyte current density magnitude", "A/m^2", "scalar"),
    ("cd.Ilx", "phil flux", 0, "Electrolyte current density, x component", "A/m^2", "scalar"),
    ("cd.Ily", "phil flux", 1, "Electrolyte current density, y component", "A/m^2", "scalar"),
    ("cd.Ilz", "phil flux", 2, "Electrolyte current density, z component", "A/m^2", "scalar"),
    ("cd.iloc", "phil loads", "bflux", "Local current density into the electrolyte (from the reaction)", "A/m^2", "scalar"),
    ("cd.nIl", "phil loads", "-bflux", "Normal electrolyte current density (outward)", "A/m^2", "scalar"),
    ("tds.tflux", "concentration flux", "vec", "Diffusive flux", "mol/(m^2*s)", "vector"),
    ("tds.tfluxMag", "concentration flux", "mag", "Diffusive flux magnitude", "mol/(m^2*s)", "scalar"),
    # at fixed-concentration boundaries, from the reaction loads
    ("tds.ntflux", "concentration loads", "-bflux", "Normal total flux (outward)", "mol/(m^2*s)", "scalar"),
    ("cd.disp", "mesh displacement", "vec", "Mesh displacement (deforming electrodes)", "m", "vector"),
    ("cd.dispMag", "mesh displacement", "mag", "Mesh displacement magnitude", "m", "scalar"),
    ("cd.dx", "mesh displacement", 0, "Mesh displacement, x component", "m", "scalar"),
    ("cd.dy", "mesh displacement", 1, "Mesh displacement, y component", "m", "scalar"),
    ("cd.dz", "mesh displacement", 2, "Mesh displacement, z component", "m", "scalar"),
]
# result quantities reconstructed from nodal reaction loads (integrated with the lumped weights)
_BFLUX = re.compile(r"\b(?:" + "|".join(re.escape(a).replace(r"\.", r"[._]") for a, _s, c, *_r in CATALOG
                                       if c in ("bflux", "-bflux")) + r")\b")
# short aliases accepted in expressions (and used by default plots)
SHORT = {"mises": "solid.mises", "disp": "solid.disp", "normE": "es.normE", "normJ": "ec.normJ",
         "normB": "mf.normB", "U": "spf.U", "p_re": "acpr.p_t", "Lp": "acpr.Lp", "abs_p": "acpr.absp",
         "Qrh": "ec.Qrh", "tflux": "ht.tfluxMag", "phil": "cd.phil"}
UNIT_FAMILIES = [
    ["K", "degC", "degF"], ["Pa", "kPa", "MPa", "GPa", "bar", "psi"], ["m", "cm", "mm", "um", "nm"],
    ["V", "mV", "kV"], ["V/m", "kV/m", "V/mm", "kV/mm"], ["m/s", "mm/s", "cm/s", "km/h"],
    ["W/m^2", "kW/m^2", "W/cm^2", "mW/m^2"], ["A/m^2", "A/mm^2", "kA/m^2", "MA/m^2"], ["T", "mT", "uT", "G"],
    ["W/m^3", "kW/m^3", "MW/m^3", "W/cm^3", "W/mm^3"], ["mol/m^3", "mmol/m^3", "mol/L", "mmol/L"],
    ["J/m^3", "kJ/m^3", "mJ/m^3"], ["C/m^2", "uC/m^2", "nC/m^2"], ["A/m", "kA/m"], ["Wb/m", "mWb/m"],
]


def compatible_units(unit: str) -> list[str]:
    for fam in UNIT_FAMILIES:
        if unit in fam:
            return fam
    return [unit] if unit else ["1"]


# --------------------------------------------------------------------------- color tables
COLOR_TABLES = {
    "Rainbow": [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    "RainbowLight": [(0.20, 0.28, 0.72), (0.27, 0.56, 0.90), (0.42, 0.80, 0.86), (0.66, 0.90, 0.52),
                     (0.98, 0.88, 0.38), (0.96, 0.56, 0.25), (0.80, 0.18, 0.15)],
    "Thermal": [(0, 0, 0), (0.6, 0.0, 0.0), (1, 0.3, 0), (1, 0.8, 0.0), (1, 1, 1)],
    "ThermalLight": [(0.38, 0.02, 0.02), (0.72, 0.10, 0.04), (0.93, 0.40, 0.10), (0.99, 0.72, 0.30),
                     (1.0, 0.95, 0.75)],
    "Wave": [(0.0, 0.0, 0.55), (0.25, 0.45, 1.0), (1, 1, 1), (1.0, 0.35, 0.25), (0.55, 0.0, 0.0)],
    "WaveLight": [(0.25, 0.35, 0.75), (0.55, 0.7, 1.0), (1, 1, 1), (1.0, 0.65, 0.55), (0.75, 0.25, 0.2)],
    "Cyclic": [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 0, 0)],
    "GrayScale": [(0.05, 0.05, 0.05), (0.97, 0.97, 0.97)],
    "Traffic": [(0.1, 0.65, 0.2), (1.0, 0.9, 0.1), (0.85, 0.1, 0.1)],
    "Jupiter": [(0.36, 0.20, 0.10), (0.75, 0.50, 0.30), (0.98, 0.92, 0.78), (0.70, 0.45, 0.30), (0.40, 0.25, 0.15)],
    "Prism": [(1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (0.5, 0, 1)],
}


def colormap(name: str, reverse=False):
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    if name in COLOR_TABLES:
        cm = LinearSegmentedColormap.from_list(name, COLOR_TABLES[name], N=256)
    else:
        try:
            cm = matplotlib.colormaps[name.lower()]
        except KeyError:
            cm = LinearSegmentedColormap.from_list("RainbowLight", COLOR_TABLES["RainbowLight"], N=256)
    return cm.reversed() if reverse else cm


# --------------------------------------------------------------------------- VTK cell types
VTK_OF_GMSH = {1: 3, 8: 21, 2: 5, 9: 22, 3: 9, 16: 23, 10: 28, 4: 10, 11: 24, 7: 14, 6: 13, 5: 12}
PERM_GMSH_TO_VTK = {11: [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]}


def topology_to_npz(mesh_result, path):
    arrays = {"nodes": mesh_result.nodes, "sdim": mesh_result.sdim, "scale": mesh_result.scale,
              "unit": mesh_result.unit, "order": mesh_result.order}
    for kind, src in (("bulk", mesh_result.bulk), ("bnd", mesh_result.bnd)):
        for num, blocks in src.items():
            for i, (gt, conn) in enumerate(blocks):
                arrays[f"{kind}_{num}_{gt}_{i}"] = conn
    for num, q in mesh_result.quality.items():
        arrays[f"qual_{num}"] = q
    for num, node in getattr(mesh_result, "point_nodes", {}).items():
        arrays[f"ptnode_{num}"] = np.array(node)
    np.savez_compressed(path, **arrays)


class Topology:
    """Bulk and boundary connectivity with domain/boundary numbers (pure numpy)."""

    def __init__(self, path=None, mesh_result=None):
        self.bulk: dict = {}
        self.bnd: dict = {}
        self.quality: dict = {}
        self.point_nodes: dict = {}
        if mesh_result is not None:
            self.nodes = mesh_result.nodes
            self.sdim, self.scale, self.unit, self.order = mesh_result.sdim, mesh_result.scale, mesh_result.unit, mesh_result.order
            self.bulk, self.bnd, self.quality = mesh_result.bulk, mesh_result.bnd, mesh_result.quality
            self.point_nodes = dict(getattr(mesh_result, "point_nodes", {}))
            return
        z = np.load(path, allow_pickle=False)
        self.nodes = z["nodes"]
        self.sdim = int(z["sdim"])
        self.scale = float(z["scale"])
        self.unit = str(z["unit"])
        self.order = int(z["order"])
        for k in z.files:
            m = re.match(r"(bulk|bnd)_(\d+)_(\d+)_(\d+)$", k)
            if m:
                getattr(self, m.group(1)).setdefault(int(m.group(2)), []).append((int(m.group(3)), z[k]))
            elif k.startswith("qual_"):
                self.quality[int(k[5:])] = z[k]
            elif k.startswith("ptnode_"):
                self.point_nodes[int(k[7:])] = int(z[k])

    def _grid(self, src, idname):
        import pyvista as pv
        cells, types, ids = [], [], []
        for num, blocks in sorted(src.items()):
            for gt, conn in blocks:
                vt = VTK_OF_GMSH.get(gt)
                if vt is None:
                    continue
                c = conn[:, PERM_GMSH_TO_VTK[gt]] if gt in PERM_GMSH_TO_VTK else conn
                n = c.shape[1]
                cells.append(np.hstack([np.full((len(c), 1), n), c]).ravel())
                types.append(np.full(len(c), vt, dtype=np.uint8))
                ids.append(np.full(len(c), num))
        if not cells:
            g = pv.UnstructuredGrid()
            return g
        g = pv.UnstructuredGrid(np.concatenate(cells), np.concatenate(types), self.nodes.copy())
        g.cell_data[idname] = np.concatenate(ids)
        return g

    def bulk_grid(self):
        g = self._grid(self.bulk, "dom")
        if self.quality and g.n_cells:
            q = []
            for num in sorted(self.bulk):
                qq = self.quality.get(num)
                n = sum(len(c) for _, c in self.bulk[num])
                q.append(qq if qq is not None and len(qq) == n else np.ones(n))
            g.cell_data["quality"] = np.concatenate(q)
        return g

    def boundary_grid(self):
        return self._grid(self.bnd, "bnd")


# --------------------------------------------------------------------------- solution
@dataclass
class Frame:
    file: str
    label: str
    value: float
    mode: int = 0          # eigenmode index (1-based) when the file holds several modes


class Solution:
    def __init__(self, folder: str):
        self.folder = folder
        with open(os.path.join(folder, "meta.json")) as f:
            self.meta = json.load(f)
        self.topo = Topology(os.path.join(folder, "mesh.npz"))
        self.sdim = self.topo.sdim
        self.unit = self.topo.unit
        self.scale = self.topo.scale
        self.kind = self.meta.get("kind", "study.stationary")
        self.frames = self._frames()
        self._bulk = None
        self._bnd = None
        self._cache: dict = {}
        self.moving = bool(self.meta.get("moving_mesh"))   # deforming electrodes: node coordinates per frame
        self._geom_frame = 0
        self._frame_nodes = None

    @property
    def catalog(self) -> list[tuple]:
        """CATALOG plus this solution's own entries (species named per interface, from the SIF build)."""
        return CATALOG + [tuple(e) for e in (self.meta.get("catalog") or [])]

    def _bflux(self, expr) -> bool:
        """True if *expr* reads a flux density reconstructed from nodal reaction loads."""
        extra = [a for a, _s, c, *_r in (self.meta.get("catalog") or []) if c in ("bflux", "-bflux")]
        return bool(_BFLUX.search(str(expr)) or extra and re.search(
            r"\b(?:" + "|".join(re.escape(a).replace(r"\.", r"[._]") for a in extra) + r")\b", str(expr)))

    def _set_geometry(self, i: int):
        """Moving meshes: put frame *i*'s node coordinates into the topology (grids and weights follow)."""
        if not self.moving or i < 0 or i == self._geom_frame:
            return
        if self._frame_nodes is None:
            with np.load(os.path.join(self.folder, "frames.npz")) as z:
                self._frame_nodes = {k: z[k] for k in z.files}
        nodes = self._frame_nodes.get(f"f{i}")
        if nodes is None:
            return
        fm = self.meta.get("frame_mesh") or []
        r = int(fm[i]) if i < len(fm) else 0
        if r != getattr(self, "_mesh_index", 0):            # remeshed: this frame has its own connectivity
            self.topo = Topology(os.path.join(self.folder, "mesh.npz" if r == 0 else f"mesh_{r}.npz"))
            self._mesh_index = r
        self.topo.nodes = nodes
        self._geom_frame = i
        self._bulk = self._bnd = None
        self._blump = None
        self._lscale = {}

    def boundary_paths(self, entities) -> list[np.ndarray]:
        """2D: node indices along the selected boundaries, one ordered chain per connected piece (mid-side nodes
        of quadratic edges included, in order)."""
        edges = []
        for b in entities:
            for gt, conn in self.topo.bnd.get(int(b), []):
                for row in np.asarray(conn):
                    edges.append((int(row[0]), int(row[1]), int(row[2]) if int(gt) == 8 and len(row) > 2 else None))
        adj: dict = {}
        for a, b, m in edges:
            adj.setdefault(a, []).append((b, m))
            adj.setdefault(b, []).append((a, m))
        used, paths = set(), []
        starts = [n for n, nb in adj.items() if len(nb) != 2] + list(adj)
        for s in starts:
            for nxt, mid in adj[s]:
                if frozenset((s, nxt)) in used:
                    continue
                path, prev, cur = [s], s, nxt
                used.add(frozenset((s, nxt)))
                if mid is not None:
                    path.append(mid)
                while True:
                    path.append(cur)
                    if cur == s or len(adj[cur]) != 2:
                        break
                    (n1, m1), (n2, m2) = adj[cur]
                    nn, mm = (n2, m2) if n1 == prev else (n1, m1)
                    if frozenset((cur, nn)) in used:
                        break
                    used.add(frozenset((cur, nn)))
                    if mm is not None:
                        path.append(mm)
                    prev, cur = cur, nn
                paths.append(np.array(path, np.int64))
        return paths

    def initial_nodes(self):
        if not self.moving:
            return self.topo.nodes
        if self._frame_nodes is None:
            with np.load(os.path.join(self.folder, "frames.npz")) as z:
                self._frame_nodes = {k: z[k] for k in z.files}
        return self._frame_nodes["f0"]

    def _frames(self) -> list[Frame]:
        files = sorted(glob.glob(os.path.join(self.folder, "results", "case_t*.vtu")))
        if not files:
            files = sorted(glob.glob(os.path.join(self.folder, "results", "*.pvtu")))
        out = []
        if self.kind == "study.eigen":
            eig = self.meta.get("eigen", [])
            for i, lam in enumerate(eig, 1):
                f = math.sqrt(abs(lam)) / (2 * math.pi)
                out.append(Frame(files[-1] if files else "", f"eigfreq={f:.6g} Hz", f, mode=i))
            if not eig and files:
                out.append(Frame(files[-1], "Mode 1", 0.0, mode=1))
        elif self.kind == "study.time":
            times = self.meta.get("times", [])
            for i, fn in enumerate(files):
                t = times[i] if i < len(times) else float(i + 1)
                out.append(Frame(fn, f"Time={t:.6g} {self.meta.get('tunit', 's')}", t))
        elif self.kind == "study.freq":
            freqs = self.meta.get("freqs", [])
            for i, fn in enumerate(files):
                f = freqs[i] if i < len(freqs) else float(i + 1)
                out.append(Frame(fn, f"freq={f:.6g} Hz", f))
        else:
            for fn in files[-1:]:
                out.append(Frame(fn, "", 0.0))
        return out

    def frame_labels(self):
        return [fr.label for fr in self.frames]

    def frame_index(self, sel) -> int:
        if not self.frames:
            return -1
        if sel in (None, "", "last"):
            i = len(self.frames) - 1
        else:
            try:
                i = max(0, min(int(sel), len(self.frames) - 1))
            except (TypeError, ValueError):
                i = len(self.frames) - 1
        self._set_geometry(i)
        return i

    # -- raw arrays mapped onto our topology
    def arrays(self, i: int) -> dict:
        self._set_geometry(i)
        if i in self._cache:
            return self._cache[i]
        import pyvista as pv
        fr = self.frames[i]
        g = pv.read(fr.file)
        n = len(self.topo.nodes)
        out = {}
        if g.n_points == n:
            for k in g.point_data.keys():
                out[k.lower()] = np.asarray(g.point_data[k])
        else:  # parallel output or different numbering: map by coordinates
            pts = np.asarray(g.points) / self.scale
            key = lambda P: [tuple(r) for r in np.round(P / (np.ptp(self.topo.nodes) + 1e-30) * 1e9).astype(np.int64)]  # noqa: E731
            idx = {k: j for j, k in enumerate(key(pts))}
            mapping = np.array([idx.get(k, 0) for k in key(self.topo.nodes)])
            for k in g.point_data.keys():
                out[k.lower()] = np.asarray(g.point_data[k])[mapping]
        if fr.mode:
            suffix = f" eigenmode{fr.mode}"
            for k in list(out.keys()):
                if k.endswith(suffix):
                    out[k[: -len(suffix)]] = out[k]
        for k in [k for k in out if k.endswith(" 1")]:      # split components "name 1", "name 2" -> "name"
            base = k[:-2]
            if base not in out and f"{base} 2" in out:
                cols = [out[f"{base} {j}"] for j in (1, 2, 3) if f"{base} {j}" in out]
                if all(np.ndim(c) == 1 for c in cols):
                    out[base] = np.column_stack(cols + [np.zeros(n)] * (3 - len(cols)))
        comps = ("xx", "yy", "zz", "xy", "yz", "xz")
        if "stress" not in out and "stress_xx" in out:       # ResultOutputSolve writes tensor components apart
            out["stress"] = np.column_stack([out.get(f"stress_{c}", np.zeros(n)) for c in comps])
        for var, doms in (self.meta.get("var_domains") or {}).items():
            # a physics on some domains only: its fields (and their loads, fluxes, gradients) are undefined outside,
            # where Elmer writes zeros; leave them blank there, as COMSOL does
            mask = self._node_mask("domain", doms)
            if mask is None:
                continue
            for k in [k for k in out if k == var or k.startswith(var + " ")]:
                a = np.array(out[k], dtype=float)
                a[~mask] = np.nan
                out[k] = a
        if self.moving and getattr(self, "_mesh_index", 0) == 0:   # deforming electrodes: displacement from the start
            out["mesh displacement"] = (np.asarray(self.topo.nodes, float) - self.initial_nodes()) * self.scale
        self._cache = {i: out}
        return out

    def available(self, i: int | None = None) -> list[tuple]:
        """Catalog entries whose source array exists, plus raw arrays."""
        i = self.frame_index(i)
        if i < 0:
            return []
        arr = self.arrays(i)
        out = []
        for alias, src, comp, label, unit, kind in self.catalog:
            if src in arr:
                a = arr[src]
                if isinstance(comp, int) and (a.ndim < 2 or comp >= a.shape[1]):
                    continue
                if isinstance(comp, int) and comp == 2 and self.sdim == 2 and src not in ("stress",):
                    continue
                if src == "stress" and comp in (4, 5) and self.sdim == 2:
                    continue
                out.append((alias, label, unit, kind))
        known = {c[1] for c in self.catalog}
        for k, a in arr.items():
            if k not in known and "eigenmode" not in k:
                out.append((k.replace(" ", "_"), k, "", "vector" if a.ndim == 2 and a.shape[1] in (2, 3) else "scalar"))
        out += [("x", "x-coordinate", "m", "scalar"), ("y", "y-coordinate", "m", "scalar")]
        if self.sdim == 3:
            out.append(("z", "z-coordinate", "m", "scalar"))
        return out

    def names(self, i: int) -> dict:
        arr = self.arrays(i)
        names: dict = {}
        for alias, src, comp, label, unit, kind in self.catalog:
            if src not in arr:
                continue
            a = arr[src]
            if comp is None:
                v = a if a.ndim == 1 else a[:, 0]
            elif comp == "vec":
                v = a if a.ndim == 2 else a[:, None]
                if v.shape[1] == 2:
                    v = np.column_stack([v, np.zeros(len(v))])
            elif comp == "mag":
                v = np.linalg.norm(a, axis=1) if a.ndim == 2 else np.abs(a)
            elif comp in ("bflux", "-bflux"):       # reaction (Loads) -> boundary flux density
                loads = (a if a.ndim == 1 else a[:, 0]) * self.load_scale(src, arr)
                v = self.boundary_projection(loads) if self.sdim == 3 else None
                if v is None:
                    A = self.boundary_lumped()
                    with np.errstate(divide="ignore", invalid="ignore"):
                        v = np.where(A > 0, loads / np.where(A > 0, A, 1.0), np.nan)
                if comp == "-bflux":
                    v = -v
            elif comp == "spl":
                pabs = np.linalg.norm(a, axis=1) if a.ndim == 2 else np.abs(a)
                v = 20 * np.log10(np.maximum(pabs / math.sqrt(2), 1e-30) / 20e-6)
            else:
                if a.ndim < 2 or comp >= a.shape[1]:
                    continue
                v = a[:, comp]
            names[alias.replace(".", "_")] = v
        for k, a in arr.items():
            names.setdefault(k.replace(" ", "_"), a if a.ndim == 1 else a)
        if "u" not in names and "spf_u" in names:      # flow models: u, v, w are velocity components
            names["u"], names["v"] = names["spf_u"], names["spf_v"]
            names["w"] = names.get("spf_w", np.zeros_like(names["spf_u"]))
        P = self.topo.nodes * self.scale
        names["x"], names["y"], names["z"] = P[:, 0], P[:, 1], P[:, 2]
        if self.sdim == 2:
            names["r"] = P[:, 0]
            if self.meta.get("axi"):              # 2D axisymmetric: (r, z) coordinates
                names["z"] = P[:, 1]
        if self.kind == "study.freq" and 0 <= i < len(self.frames):
            names["freq"] = self.frames[i].value
        n = len(self.topo.nodes)
        for gname, (src, ents, factor) in self.globals.items():     # e.g. cd.es1.Itot = sum of reaction loads
            a = arr.get(src)
            if a is None:
                continue
            mask = self._node_mask("boundary", ents)
            tot = float(np.sum((a if a.ndim == 1 else a[:, 0])[mask])) * factor * self.load_scale(src, arr) \
                if mask is not None else float("nan")
            names[gname.replace(".", "_")] = np.full(n, tot)
        for s, full in SHORT.items():
            fk = full.replace(".", "_")
            if fk in names:
                names[s] = names[fk]
        return names

    # component coupling operators (aveop1(T), intop1(...), maxop1, minop1), global parameters and
    # Definitions variables, all set by the model so result expressions read like COMSOL's
    operators: dict = {}
    efc: dict = {}             # exterior field calculations {efc1: acoustics.ExteriorField}
    globals: dict = {}         # {name: (array, boundary numbers, factor)}: sums of nodal reactions (electrode currents)
    flux_boundaries: set = set()   # boundaries where a reaction flux exists (others are insulating)

    def boundary_projection(self, loads):
        """Flux density j on the reaction boundaries of a 3D model from nodal loads L: solve M j = L with the
        consistent boundary mass matrix (matrix-free conjugate gradients).

        Dividing loads by a lumped measure is exact in total but, on 6-node triangles, a uniform flux has zero
        load at the corner nodes, so the corners read zero and a face looks spotted.  The consistent projection
        is smooth and still integrates to the sum of the loads.  None when the boundary is not triangulated."""
        tm = self._tri_mass(self.flux_boundaries)
        if tm is None:
            return None
        nodes, conns, mats = tm
        n = len(nodes)

        def mv(x):
            y = np.zeros(n)
            for c, Me in zip(conns, mats):
                np.add.at(y, c.ravel(), np.einsum("eij,ej->ei", Me, x[c]).ravel())
            return y
        diag = np.zeros(n)
        for c, Me in zip(conns, mats):
            np.add.at(diag, c.ravel(), np.einsum("eii->ei", Me).ravel())
        rhs = np.nan_to_num(np.asarray(loads, float)[nodes])
        x = rhs / diag
        r = rhs - mv(x)
        z = r / diag
        p = z.copy()
        rz = r @ z
        tol = 1e-12 * max(np.linalg.norm(rhs), 1e-300)
        for _ in range(400):
            if np.linalg.norm(r) <= tol:
                break
            q = mv(p)
            alpha = rz / (p @ q)
            x += alpha * p
            r -= alpha * q
            z = r / diag
            rz, rz0 = r @ z, rz
            p = z + (rz / rz0) * p
        out = np.full(len(self.topo.nodes), np.nan)
        out[nodes] = x
        return out

    def boundary_weights(self, boundaries):
        """Consistent integration weights W_i = integral of N_i over *boundaries* (3D, triangulated), so that the
        integral of a nodal field is sum W_i v_i; None when the boundary is not triangulated."""
        tm = self._tri_mass(boundaries)
        if tm is None:
            return None
        nodes, conns, mats = tm
        W = np.zeros(len(nodes))
        for c, Me in zip(conns, mats):
            np.add.at(W, c.ravel(), Me.sum(axis=2).ravel())
        out = np.zeros(len(self.topo.nodes))
        out[nodes] = W
        return out

    def _tri_mass(self, boundaries):
        """(global node ids, element connectivities in local ids, element mass matrices) of the triangulated
        boundary elements on *boundaries*; None for other element shapes."""
        if not boundaries:
            return None
        g = self.boundary_grid()
        if not g.n_cells or "bnd" not in g.cell_data:
            return None
        keep = np.where(np.isin(np.asarray(g.cell_data["bnd"]), sorted(int(b) for b in boundaries)))[0]
        if not len(keep):
            return None
        g = g.extract_cells(keep)
        ids = np.asarray(g.point_data["vtkOriginalPointIds"]) if "vtkOriginalPointIds" in g.point_data else None
        cd = g.cells_dict
        if ids is None or set(int(t) for t in cd) - {5, 22}:
            return None                                  # quadrilaterals etc.: keep the lumped form
        P = np.asarray(self.topo.nodes, float)[ids] * self.scale
        a, b = 0.445948490915965, 0.091576213509771     # 6-point degree-4 rule on the reference triangle
        gp = np.array([[a, a], [1 - 2 * a, a], [a, 1 - 2 * a], [b, b], [1 - 2 * b, b], [b, 1 - 2 * b]])
        gw = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3) / 2
        xi, et = gp[:, 0], gp[:, 1]
        l1, l2, l3 = 1 - xi - et, xi, et
        conns, mats = [], []
        for ct, conn in cd.items():
            if int(ct) == 22:
                N = np.array([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1), 4 * l1 * l2, 4 * l2 * l3,
                              4 * l3 * l1])
                dxi = np.array([1 - 4 * l1, 4 * l2 - 1, 0 * xi, 4 * (l1 - l2), 4 * l3, -4 * l3])
                det = np.array([1 - 4 * l1, 0 * xi, 4 * l3 - 1, -4 * l2, 4 * l2, 4 * (l1 - l3)])
            else:
                N = np.array([l1, l2, l3])
                dxi = np.array([-np.ones_like(xi), np.ones_like(xi), 0 * xi])
                det = np.array([-np.ones_like(xi), 0 * xi, np.ones_like(xi)])
            X = P[conn]                                                    # (ne, nn, 3)
            t1, t2 = np.einsum("enk,ng->egk", X, dxi), np.einsum("enk,ng->egk", X, det)
            jac = np.linalg.norm(np.cross(t1, t2), axis=2)                 # (ne, ng)
            mats.append(np.einsum("ig,jg,eg->eij", N, N, jac * gw))
            conns.append(conn)
        used = np.unique(np.concatenate([c.ravel() for c in conns]))
        loc = np.full(len(P), -1)
        loc[used] = np.arange(len(used))
        conns = [loc[c] for c in conns]
        return ids[used], conns, mats

    def load_scale(self, src: str, arr: dict) -> float:
        """Factor that makes Elmer's axisymmetric nodal loads per radian.

        Elmer 9.0 and the classic assembly integrate with r dr dz (loads per radian); the vectorized axisymmetric
        assembly of later builds returns loads for the full revolution.  The two differ by 2 pi, so compare the
        loads on the reaction boundaries with the solver's own flux through them, per radian (HRZ measure x r);
        an element-gradient flux is within ~10 % even at singular edges, far from a factor 6.3."""
        if not (self.sdim == 2 and self.meta.get("axi")) or not src.endswith(" loads"):
            return 1.0
        memo = self.__dict__.setdefault("_lscale", {})
        if src in memo:
            return memo[src]
        scale = 1.0
        F, L = arr.get(src[: -len(" loads")] + " flux"), arr.get(src)
        if F is not None and L is not None and self.flux_boundaries:
            A = self.boundary_lumped()
            on = A > 0
            est = float(np.sum(np.linalg.norm(np.atleast_2d(F.T).T, axis=1)[on] * A[on]))
            got = float(np.sum(np.abs(L if L.ndim == 1 else L[:, 0])[on]))
            if est > 0 and got / est > math.sqrt(2 * math.pi):
                scale = 1 / (2 * math.pi)
        memo[src] = scale
        return scale

    def boundary_lumped(self, boundaries=None, r_weight=True):
        """Lumped boundary measure per node (HRZ), with the r weight on axisymmetric models -- the measure that
        turns Elmer's nodal reaction loads into a boundary flux density.  Zero on interior nodes.

        *boundaries*: only these boundaries (the quadrature for integrating a reaction flux over a selection);
        default: the boundaries carrying a reaction."""
        if boundaries is None and r_weight and getattr(self, "_blump", None) is not None:
            return self._blump
        g = self.boundary_grid()
        P = np.asarray(self.topo.nodes, float) * self.scale
        A = np.zeros(len(P))
        axi = self.sdim == 2 and self.meta.get("axi") and r_weight
        only = self.flux_boundaries if boundaries is None else set(int(b) for b in boundaries)
        if only and g.n_cells and "bnd" in g.cell_data:                       # only boundaries carrying a reaction,
            keep = np.where(np.isin(np.asarray(g.cell_data["bnd"]), sorted(only)))[0]
            g = g.extract_cells(keep) if len(keep) else g                     # so corners are not diluted
            if "vtkOriginalPointIds" in g.point_data:
                P = P[np.asarray(g.point_data["vtkOriginalPointIds"])]
        ids = np.asarray(g.point_data["vtkOriginalPointIds"]) if "vtkOriginalPointIds" in g.point_data else None
        for ct, conn in (g.cells_dict.items() if g.n_cells else []):
            fr = HRZ.get(int(ct))
            if fr is None or conn.shape[1] != len(fr):
                continue
            X = P[conn]
            if ct in (3, 21):                       # line (end, end[, mid])
                meas = np.linalg.norm(X[:, 1] - X[:, 0], axis=1) if ct == 3 else \
                    np.linalg.norm(X[:, 2] - X[:, 0], axis=1) + np.linalg.norm(X[:, 1] - X[:, 2], axis=1)
            elif ct in (5, 22):                     # triangle: area from the corners
                meas = 0.5 * np.linalg.norm(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), axis=1)
            else:                                   # quadrilateral: two triangles over the corners
                meas = 0.5 * (np.linalg.norm(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), axis=1)
                              + np.linalg.norm(np.cross(X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]), axis=1))
            if axi and ct in (3, 21):               # HRZ of the r-weighted mass: positive on the axis too
                w = _axi_line_hrz(X)
            else:
                w = fr[None, :] * meas[:, None]
                if axi:
                    w = w * np.abs(P[conn, 0])
            np.add.at(A, (ids[conn] if ids is not None else conn).ravel(), w.ravel())
        if boundaries is None and r_weight:
            self._blump = A
        return A
    params: dict = {}
    variables: dict = {}

    def _expand(self, expr: str) -> str:
        s = str(expr)
        for _ in range(8):
            changed = False
            for nm, ex in self.variables.items():
                new = re.sub(rf"(?<![\w.]){re.escape(nm)}(?![\w(])", f"({ex})", s)
                if new != s:
                    s, changed = new, True
            if not changed:
                break
        return s

    def _operator_funcs(self):
        def make(kind, level, ents):
            def op(arr):
                arr = np.asarray(np.real(arr), float)
                if arr.ndim == 0:
                    arr = np.full(len(self.topo.nodes), float(arr))
                if level == "point":       # COMSOL: integrating over points sums the point values
                    idx = [self.topo.point_nodes[e] for e in (ents or self.topo.point_nodes) if e in self.topo.point_nodes]
                    if not idx:
                        return float("nan")
                    v = arr[idx]
                    return float({"integration": np.sum, "average": np.mean, "maximum": np.max,
                                  "minimum": np.min}[kind](v))
                if kind in ("maximum", "minimum"):
                    sel = self._node_mask(level, ents)
                    v = arr[sel] if sel is not None else arr
                    return float(np.nanmax(v) if kind == "maximum" else np.nanmin(v))
                return self._integrate_values(arr, level, ents, average=(kind == "average"))
            return op
        return {nm: make(k, lv, ents) for nm, (k, lv, ents) in self.operators.items()}

    def _node_mask(self, level, ents):
        if ents is None:
            return None
        g = self.bulk_grid() if level == "domain" else self.boundary_grid()
        idname = "dom" if level == "domain" else "bnd"
        cells = np.where(np.isin(g.cell_data[idname], list(ents)))[0]
        if not len(cells):
            return np.zeros(len(self.topo.nodes), bool)
        sub = g.extract_cells(cells)
        mask = np.zeros(len(self.topo.nodes), bool)
        mask[np.asarray(sub.point_data["vtkOriginalPointIds"])] = True
        return mask

    comp_tag: str = ""
    comp_tags: tuple = ()

    def _own(self, expr) -> str:
        """Results of this component may carry its prefix (comp1.T); another component's are refused by name."""
        s = str(expr)
        if self.comp_tag:
            s = re.sub(rf"(?<![\w.]){re.escape(self.comp_tag)}\.(?=[A-Za-z_])", "", s)
        for m in re.finditer(r"(?<![\w.])(comp\w*)\.(?=[A-Za-z_])", s):
            if m.group(1) in self.comp_tags:
                raise units.ExprError(f"'{expr}' refers to {m.group(1)}, another component: use a dataset of "
                                      f"{m.group(1)} (its Solution dataset)")
        return s

    def evaluate(self, expr: str, i: int | None = None) -> np.ndarray:
        i = self.frame_index(i)
        names = {**self.params, **self.names(i)}
        e = _dotted(self._expand(self._own(expr)))
        v = units.evaluate(e, names, self._operator_funcs())
        v = np.asarray(np.real(v), dtype=float)
        if v.ndim == 0:
            v = np.full(len(self.topo.nodes), float(v))
        return v

    def evaluate_vector(self, expr: str, i: int | None = None) -> np.ndarray:
        i = self.frame_index(i)
        names = self.names(i)
        e = _dotted(self._own(expr).strip())
        # allow "u,v,w" component lists or a vector alias
        parts = [p for p in re.split(r"\s*,\s*", e) if p]
        if len(parts) >= 2:
            comps = [self.evaluate(p, i) for p in parts[:3]]          # scalars broadcast to nodes
            while len(comps) < 3:
                comps.append(np.zeros_like(comps[0]))
            return np.column_stack(comps)
        aliases = {"disp": "solid_u_vec", "u": "solid_u_vec", "spf_U": "spf_u_vec", "U": "spf_u_vec",
                   "E": "es_E", "J": "ec_J", "B": "mf_B", "tflux": "ht_tflux"}
        key = aliases.get(e, e)
        if key in names and np.ndim(names[key]) == 2:
            v = names[key]
            if v.shape[1] == 2:
                v = np.column_stack([v, np.zeros(len(v))])
            return np.asarray(v, float)
        for cand in (f"{e}_vec", e.replace(" ", "_")):
            if cand in names and np.ndim(names[cand]) == 2:
                return np.asarray(names[cand], float)
        raise units.ExprError(f"'{expr}' is not a vector quantity")

    def describe(self, expr: str):
        """(label, SI unit) for an expression if it is a catalog variable."""
        e = SHORT.get(expr.strip(), expr.strip())
        if e in ("x", "y", "z", "r"):
            return f"{e}-coordinate", "m"
        for alias, src, comp, label, unit, kind in self.catalog:
            if alias == e or alias.replace(".", "_") == e:
                return label, unit
        from .acoustics import EFC_LABELS, EFC_NAME, INCIDENT_LABELS
        m = EFC_NAME.fullmatch(e)
        if m:
            for name, label, unit in EFC_LABELS:
                if m.group(2) in (name, name.replace("pext", "p_ext")):
                    return label, unit
        for alias, label, unit in INCIDENT_LABELS:
            if alias == e:
                return label, unit
        return expr, ""

    # -- grids
    def bulk_grid(self):
        if self._bulk is None:
            self._bulk = self.topo.bulk_grid()
        return self._bulk

    def boundary_grid(self):
        if self._bnd is None:
            self._bnd = self.topo.boundary_grid()
        return self._bnd

    # -- derived values
    def integrate(self, expr, i, level="domain", entities=None, average=False, revolved=False):
        """Integral (or average) of *expr* over domains or boundaries.

        *revolved*: on a 2D axisymmetric model, integrate in the revolved geometry (weight 2 pi r), as COMSOL's
        "Compute integral in revolved geometry"; ignored on other models.  Off: the r-z plane measure."""
        vals = self.evaluate(expr, i)
        rev = bool(revolved) and self.sdim == 2 and bool(self.meta.get("axi"))
        if level == "boundary" and self._bflux(expr):
            # a flux density reconstructed from nodal reaction loads: integrate with the lumped (HRZ) weights that
            # reconstructed it, so integrals over boundary subsets add up to the electrode current exactly (the
            # quadratic shape functions would drop the corner loads -- a corner of a 6-node triangle integrates
            # to zero)
            ents = entities if entities is not None else sorted(int(b) for b in np.unique(self.topo.bnd))
            W = self.boundary_weights(ents) if self.sdim == 3 else None
            if W is not None:                      # 3D: j is the consistent projection, so integrate it consistently
                tot = float(np.sum(W * np.nan_to_num(vals)))
                return tot / float(W.sum()) if average else tot
            A = self.boundary_lumped(ents, r_weight=rev)
            on = (A > 0) & np.isfinite(vals)
            tot = float(np.sum(vals[on] * A[on]))
            if average:
                return tot / float(A.sum())
            return tot * 2 * math.pi if rev else tot
        return self._integrate_values(vals, level, entities, average, revolved=rev)

    def _integrate_values(self, vals, level="domain", entities=None, average=False, revolved=False):
        g = self.bulk_grid() if level == "domain" else self.boundary_grid()
        idname = "dom" if level == "domain" else "bnd"
        if g.n_cells == 0:
            return float("nan")
        g = g.copy()
        g.points = self.topo.nodes * self.scale
        if revolved:                                  # 2D axisymmetric, revolved geometry: weight 2 pi r
            w = 2 * math.pi * np.abs(np.asarray(g.points, float)[:, 0])
            g.point_data["f"] = vals * w
            g.point_data["w"] = w
        else:
            g.point_data["f"] = vals
        if entities is not None:
            mask = np.isin(g.cell_data[idname], list(entities))
            if not mask.any():
                return float("nan")
            g = g.extract_cells(np.where(mask)[0])
        r = _split_quadratic_edges(g).integrate_data()
        tot = float(r.point_data["f"][0])
        if revolved:                                  # the revolved volume / surface area
            size = float(r.point_data["w"][0])
        else:
            dim = self.sdim if level == "domain" else self.sdim - 1
            size_key = {3: "Volume", 2: "Area", 1: "Length"}.get(dim, "Volume")
            size = float(r.cell_data[size_key][0]) if size_key in r.cell_data else float("nan")
        if average:
            return tot / size if size else float("nan")
        return tot

    def maxmin(self, expr, i, level="domain", entities=None):
        r = self.maxmin_locations(expr, i, level, entities)
        return r["max"], r["min"]

    def maxmin_locations(self, expr, i, level="domain", entities=None):
        """Max/min values and the coordinates (SI) where they occur, like COMSOL's Max/Min evaluation."""
        vals = self.evaluate(expr, i)
        mask = self._node_mask(level, entities)
        if mask is None:
            mask = np.zeros(len(vals), bool)
            g = self.bulk_grid() if level == "domain" else self.boundary_grid()
            if g.n_cells:
                mask[np.unique(np.asarray(g.extract_cells(np.arange(g.n_cells)).point_data["vtkOriginalPointIds"]))] = True
            else:
                mask[:] = True
        idx = np.where(mask & np.isfinite(vals))[0]
        if not len(idx):
            nan = float("nan")
            return {"max": nan, "min": nan, "xmax": [nan] * 3, "xmin": [nan] * 3}
        imax = idx[np.argmax(vals[idx])]
        imin = idx[np.argmin(vals[idx])]
        P = self.topo.nodes * self.scale
        return {"max": float(vals[imax]), "min": float(vals[imin]), "xmax": P[imax].tolist(), "xmin": P[imin].tolist()}

    def point_eval(self, expr, i, points):
        vals = self.evaluate(expr, i)
        g = self.bulk_grid().copy()
        g.point_data["f"] = vals
        P = np.atleast_2d(np.asarray(points, float)) / self.scale
        if P.shape[1] == 2:
            P = np.column_stack([P, np.zeros(len(P))])
        return _probe(g, P)

    def line_eval(self, expr, i, p0, p1, n=200):
        vals = self.evaluate(expr, i)
        g = self.bulk_grid().copy()
        g.point_data["f"] = vals
        a = np.array(p0, float) / self.scale        # SI in, geometry units internally
        b = np.array(p1, float) / self.scale
        if len(a) == 2:
            a, b = np.append(a, 0), np.append(b, 0)
        pts = a + np.linspace(0, 1, n + 1)[:, None] * (b - a)
        f = _probe(g, pts)
        s = np.linalg.norm(pts - pts[0], axis=1) * self.scale
        return s, pts * self.scale, f


def _probe(g, P):
    """Interpolate point data 'f' of grid *g* at points *P* (NaN outside).

    VTK's probe can miss points lying on edges of curved quadratic cells; those are retried on the
    linearised grid (same nodes, corner-node interpolation), which always agrees on the containing cell.
    """
    import pyvista as pv
    s = pv.PolyData(P).sample(g)
    ok = np.asarray(s.point_data.get("vtkValidPointMask", np.ones(len(P))), bool)
    out = np.asarray(s.point_data["f"], float).copy()
    if not ok.all() and hasattr(g, "linear_copy"):
        bad = np.where(~ok)[0]
        s2 = pv.PolyData(P[bad]).sample(g.linear_copy())
        ok2 = np.asarray(s2.point_data.get("vtkValidPointMask", np.ones(len(bad))), bool)
        out[bad] = np.where(ok2, np.asarray(s2.point_data["f"], float), np.nan)
    elif not ok.all():
        out[~ok] = np.nan
    bad = np.where(np.isnan(out))[0]
    if len(bad) and "f" in g.point_data and g.n_points:     # a point exactly on a mesh vertex (corner, rim)
        X = np.asarray(g.points)
        tol = 1e-9 * (float(np.linalg.norm(np.ptp(X, axis=0))) or 1.0)
        for j in bad:
            d = np.linalg.norm(X - P[j], axis=1)
            k = int(np.argmin(d))
            if d[k] <= tol:
                out[j] = float(np.asarray(g.point_data["f"])[k])
    return out


def _split_quadratic_edges(g):
    """*g* with each quadratic edge (end, end, mid) split into two linear lines through its mid node.

    VTK 9.7's integrate_data skips 3-node 1D cells ("Odd number of points"), so line integrals on 2D models came
    out zero; the halves integrate like VTK's own subdivision of quadratic triangles and quads."""
    cd = getattr(g, "cells_dict", None)
    if not cd or 21 not in cd:
        return g
    import pyvista as pv
    e = np.asarray(cd[21])
    cells = {int(ct): np.asarray(c) for ct, c in cd.items() if int(ct) not in (3, 21)}
    lines = np.vstack([e[:, [0, 2]], e[:, [2, 1]]])
    cells[3] = np.vstack([np.asarray(cd[3]), lines]) if 3 in cd else lines
    out = pv.UnstructuredGrid(cells, np.asarray(g.points))
    for k in g.point_data.keys():
        out.point_data[k] = np.asarray(g.point_data[k])
    return out


# interface prefixes: tds.tflux, and a second instance's tds1.tflux
_DOTTED = re.compile(r"\b((?:ht|solid|es|ec|mf|spf|acpr|tds|pde|cd|tcd)\d*)((?:\.\w+)+)")


def _dotted(expr: str) -> str:
    return _DOTTED.sub(lambda m: m.group(1) + m.group(2).replace(".", "_"), expr)


def _hrz_fractions():
    """Lumped (HRZ) boundary mass fractions per VTK cell type: the share of a cell's measure each node carries.
    Dividing a nodal reaction (Loads) by this lumped measure gives the boundary flux density."""
    g = np.array([-0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053])
    w = np.array([0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454])
    X, Y = np.meshgrid(g, g, indexing="ij")
    W = np.outer(w, w)
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    mids = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    q8 = []
    for xi, yi in corners:
        q8.append(0.25 * (1 + X * xi) * (1 + Y * yi) * (X * xi + Y * yi - 1))
    for xi, yi in mids:
        q8.append(0.5 * (1 - X ** 2) * (1 + Y * yi) if xi == 0 else 0.5 * (1 + X * xi) * (1 - Y ** 2))
    d8 = np.array([np.sum(W * n * n) for n in q8])
    l3 = np.array([1 / 6, 1 / 6, 2 / 3])
    q9 = np.array([l3[a] * l3[b] for a, b in ((0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (1, 2), (2, 1), (0, 2))] + [4 / 9])
    tri6 = np.array([6, 6, 6, 32, 32, 32], float)
    return {3: np.array([0.5, 0.5]), 21: l3, 5: np.full(3, 1 / 3), 22: tri6 / tri6.sum(),
            9: np.full(4, 0.25), 23: d8 / d8.sum(), 28: q9 / q9.sum()}


HRZ = _hrz_fractions()


def _axi_line_hrz(X):
    """Lumped r-weighted measure of 2-node or 3-node (end, end, mid) boundary lines, per cell and node: HRZ
    lumping of the boundary mass matrix with the axisymmetric weight r, w_k = M_kk / sum_j M_jj * int r dl with
    M_kk = int N_k^2 r dl.  Equals the HRZ fraction x length x r on lines of constant r; unlike that, it is positive
    on the axis node, whose reaction load int N_k r dl i is not zero when the flux varies along the line."""
    g, gw = np.polynomial.legendre.leggauss(4)
    xi, gw = 0.5 * (g + 1), 0.5 * gw
    if X.shape[1] == 2:
        N = np.array([1 - xi, xi])
        dN = np.array([np.full_like(xi, -1.0), np.ones_like(xi)])
    else:
        N = np.array([(1 - xi) * (1 - 2 * xi), xi * (2 * xi - 1), 4 * xi * (1 - xi)])
        dN = np.array([4 * xi - 3, 4 * xi - 1, 4 - 8 * xi])
    r = np.abs(X[:, :, 0] @ N)                                          # (cells, Gauss points)
    jac = np.linalg.norm(np.einsum("ckd,kg->cgd", X, dN), axis=2)
    rw = r * jac * gw
    M = rw @ (N ** 2).T                                                 # (cells, nodes)
    tot = M.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot > 0, M / np.where(tot > 0, tot, 1.0) * rw.sum(axis=1, keepdims=True), 0.0)


def write_solution_meta(folder: str, meta: dict):
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def auto_scale_deformation(sol: Solution, vec: np.ndarray) -> float:
    """COMSOL-like automatic deformation scale: max displacement ~ 10 % of the geometry size."""
    P = sol.topo.nodes
    size = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) or 1.0
    mx = float(np.nanmax(np.linalg.norm(vec, axis=1))) / sol.scale if len(vec) else 0.0
    if mx <= 0:
        return 1.0
    raw = 0.1 * size / mx
    exp = 10 ** math.floor(math.log10(raw))
    nice = max(m * exp for m in (1, 2, 5) if m * exp <= raw * 1.0001)
    return float(nice)
