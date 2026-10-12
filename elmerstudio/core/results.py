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
]
# short aliases accepted in expressions (and used by default plots)
SHORT = {"mises": "solid.mises", "disp": "solid.disp", "normE": "es.normE", "normJ": "ec.normJ",
         "normB": "mf.normB", "U": "spf.U", "p_re": "acpr.p_t", "Lp": "acpr.Lp", "abs_p": "acpr.absp",
         "Qrh": "ec.Qrh", "tflux": "ht.tfluxMag"}
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
            return len(self.frames) - 1
        try:
            i = int(sel)
            return max(0, min(i, len(self.frames) - 1))
        except (TypeError, ValueError):
            return len(self.frames) - 1

    # -- raw arrays mapped onto our topology
    def arrays(self, i: int) -> dict:
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
        comps = ("xx", "yy", "zz", "xy", "yz", "xz")
        if "stress" not in out and "stress_xx" in out:       # ResultOutputSolve writes tensor components apart
            out["stress"] = np.column_stack([out.get(f"stress_{c}", np.zeros(n)) for c in comps])
        self._cache = {i: out}
        return out

    def available(self, i: int | None = None) -> list[tuple]:
        """Catalog entries whose source array exists, plus raw arrays."""
        i = self.frame_index(i)
        if i < 0:
            return []
        arr = self.arrays(i)
        out = []
        for alias, src, comp, label, unit, kind in CATALOG:
            if src in arr:
                a = arr[src]
                if isinstance(comp, int) and (a.ndim < 2 or comp >= a.shape[1]):
                    continue
                if isinstance(comp, int) and comp == 2 and self.sdim == 2 and src not in ("stress",):
                    continue
                if src == "stress" and comp in (4, 5) and self.sdim == 2:
                    continue
                out.append((alias, label, unit, kind))
        known = {c[1] for c in CATALOG}
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
        for alias, src, comp, label, unit, kind in CATALOG:
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
        for s, full in SHORT.items():
            fk = full.replace(".", "_")
            if fk in names:
                names[s] = names[fk]
        return names

    # component coupling operators (aveop1(T), intop1(...), maxop1, minop1), global parameters and
    # Definitions variables, all set by the model so result expressions read like COMSOL's
    operators: dict = {}
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

    def evaluate(self, expr: str, i: int | None = None) -> np.ndarray:
        i = self.frame_index(i)
        names = {**self.params, **self.names(i)}
        e = _dotted(self._expand(str(expr)))
        v = units.evaluate(e, names, self._operator_funcs())
        v = np.asarray(np.real(v), dtype=float)
        if v.ndim == 0:
            v = np.full(len(self.topo.nodes), float(v))
        return v

    def evaluate_vector(self, expr: str, i: int | None = None) -> np.ndarray:
        i = self.frame_index(i)
        names = self.names(i)
        e = _dotted(str(expr).strip())
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
        for alias, src, comp, label, unit, kind in CATALOG:
            if alias == e or alias.replace(".", "_") == e:
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
    def integrate(self, expr, i, level="domain", entities=None, average=False):
        return self._integrate_values(self.evaluate(expr, i), level, entities, average)

    def _integrate_values(self, vals, level="domain", entities=None, average=False):
        g = self.bulk_grid() if level == "domain" else self.boundary_grid()
        idname = "dom" if level == "domain" else "bnd"
        if g.n_cells == 0:
            return float("nan")
        g = g.copy()
        g.points = self.topo.nodes * self.scale
        g.point_data["f"] = vals
        if entities is not None:
            mask = np.isin(g.cell_data[idname], list(entities))
            if not mask.any():
                return float("nan")
            g = g.extract_cells(np.where(mask)[0])
        r = g.integrate_data()
        tot = float(r.point_data["f"][0])
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


_DOTTED = re.compile(r"\b(ht|solid|es|ec|mf|spf|acpr|tds|pde)\.(\w+)")


def _dotted(expr: str) -> str:
    return _DOTTED.sub(lambda m: f"{m.group(1)}_{m.group(2)}", expr)


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
