"""Mesh sequence (physics- or user-controlled) generated with gmsh and exported to Elmer."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .elmer_mesh import CORNERS, GMSH_TO_ELMER, ElmerMesh
from .geometry import GeometryBuilder, GeometryError, GeometryResult, levels_for
from .gmsh_session import session
from .model import Model, Node, NodeType, Prop, register, space_dim

SIZE_PRESETS = [  # label, hmax, hmin, hgrad, hcurve, hnarrow  (fractions of the largest geometry side)
    ("Extremely fine", 0.02, 0.0002, 1.3, 0.2, 1.0),
    ("Extra fine", 0.035, 0.0015, 1.35, 0.3, 0.85),
    ("Finer", 0.055, 0.004, 1.4, 0.4, 0.7),
    ("Fine", 0.08, 0.01, 1.45, 0.5, 0.6),
    ("Normal", 0.1, 0.018, 1.5, 0.6, 0.5),
    ("Coarse", 0.15, 0.03, 1.6, 0.7, 0.4),
    ("Coarser", 0.2, 0.04, 1.7, 0.8, 0.3),
    ("Extra coarse", 0.3, 0.056, 1.8, 0.9, 0.2),
    ("Extremely coarse", 0.5, 0.1, 2.0, 1.0, 0.1),
]
SIZE_CHOICES = [(str(i + 1), p[0]) for i, p in enumerate(SIZE_PRESETS)]
NORMAL = "5"

register(NodeType("mesh", "Mesh", "mesh", "mesh", deletable=False, can_disable=False,
                  children=lambda n: (["mesh.size", "mesh.ftet", "mesh.distribution"] if space_dim(n.ancestor("component")) == 3
                                      else ["mesh.size", "mesh.ftri", "mesh.fquad", "mesh.distribution"])
                  if n.get("sequence") == "user" else [],
                  props=[
                      Prop("sequence", "Sequence type:", "choice", "physics",
                           choices=[("physics", "Physics-controlled mesh"), ("user", "User-controlled mesh")],
                           section="Physics-Controlled Mesh"),
                      Prop("size", "Element size:", "choice", NORMAL, choices=SIZE_CHOICES, section="Physics-Controlled Mesh",
                           visible=lambda p: p.get("sequence") != "user"),
                      Prop("order", "Geometry shape order / element order:", "choice", "auto",
                           choices=[("auto", "Automatic (from physics discretization)"), ("1", "Linear"), ("2", "Quadratic")],
                           section="Element Order"),
                      Prop("algo3", "3D algorithm:", "choice", "1",
                           choices=[("1", "Delaunay"), ("4", "Frontal (Netgen)"), ("10", "HXT (parallel Delaunay)")],
                           section="Advanced"),
                      Prop("algo2", "2D algorithm:", "choice", "6",
                           choices=[("6", "Frontal-Delaunay"), ("5", "Delaunay"), ("8", "Frontal-Delaunay for quads"),
                                    ("1", "MeshAdapt")], section="Advanced"),
                      Prop("optimize", "Optimize element quality", "bool", True, section="Advanced"),
                  ]))
register(NodeType("mesh.size", "Size", "size", "mesh_size", selection="domain", selection_all=True, props=[
    Prop("scope", "Geometric entity level:", "choice", "entire",
         choices=[("entire", "Entire geometry"), ("domain", "Domain"), ("boundary", "Boundary"), ("edge", "Edge"),
                  ("point", "Point")], section="Geometric Entity Selection"),
    Prop("mode", "Element size:", "choice", "predefined", choices=[("predefined", "Predefined"), ("custom", "Custom")],
         section="Element Size"),
    Prop("preset", "Size:", "choice", NORMAL, choices=SIZE_CHOICES, section="Element Size",
         visible=lambda p: p.get("mode") != "custom"),
    Prop("hmax", "Maximum element size:", default="0.1", unit="LEN", section="Element Size Parameters",
         visible=lambda p: p.get("mode") == "custom"),
    Prop("hmin", "Minimum element size:", default="0.018", unit="LEN", section="Element Size Parameters",
         visible=lambda p: p.get("mode") == "custom"),
    Prop("hcurve", "Curvature factor:", default="0.6", section="Element Size Parameters",
         visible=lambda p: p.get("mode") == "custom"),
]))
register(NodeType("mesh.ftet", "Free Tetrahedral", "ftet", "free_tet", selection="domain", selection_all=True, props=[]))
register(NodeType("mesh.ftri", "Free Triangular", "ftri", "free_tri", selection="domain", selection_all=True, props=[]))
register(NodeType("mesh.fquad", "Free Quad", "fq", "free_tri", selection="domain", selection_all=True, props=[]))
register(NodeType("mesh.distribution", "Distribution", "dis", "distribution", selection="boundary", props=[
    Prop("n", "Number of elements:", "int", 10, section="Distribution")]))


@dataclass
class MeshResult:
    sdim: int
    unit: str
    scale: float
    nodes: np.ndarray
    bulk: dict = field(default_factory=dict)       # domain -> list[(gmsh type, conn (M,k) 0-based)]
    bnd: dict = field(default_factory=dict)        # boundary -> list[(gmsh type, conn)]
    quality: dict = field(default_factory=dict)    # domain -> array
    stats: dict = field(default_factory=dict)
    geometry: GeometryResult | None = None
    order: int = 1
    messages: list = field(default_factory=list)

    def to_elmer(self) -> ElmerMesh:
        em = ElmerMesh()
        em.nodes = self.nodes * self.scale
        if self.sdim == 2:
            em.nodes[:, 2] = 0.0
        for dom, blocks in sorted(self.bulk.items()):
            for gt, conn in blocks:
                et, perm = GMSH_TO_ELMER[gt]
                c = conn[:, perm] if perm else conn
                em.bulk.append((dom, et, c + 1))
        for b, blocks in sorted(self.bnd.items()):
            for gt, conn in blocks:
                et, perm = GMSH_TO_ELMER[gt]
                c = conn[:, perm] if perm else conn
                em.boundary.append((b, et, c + 1))
        return em

    def surface_triangles(self):
        """Linear display triangles for boundaries (3D) or domains (2D): returns (tris, ids)."""
        src = self.bnd if self.sdim == 3 else self.bulk
        tris, ids = [], []
        for num, blocks in src.items():
            for gt, conn in blocks:
                if gt in (2, 9):
                    t = conn[:, :3]
                elif gt in (3, 16, 10):
                    q = conn[:, :4]
                    t = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
                else:
                    continue
                tris.append(t)
                ids.append(np.full(len(t), num))
        if not tris:
            return np.zeros((0, 3), int), np.zeros(0, int)
        return np.vstack(tris), np.concatenate(ids)


def physics_order(model: Model) -> int:
    comp = model.component
    if comp is None:
        return 1
    orders = []
    for n in comp.children:
        if n.kind.startswith("physics.") and n.enabled:
            disc = n.get("disc", "2")
            orders.append(int(disc) if str(disc).isdigit() else 1)
    return min(orders) if orders else 1


def _preset(i) -> tuple:
    try:
        return SIZE_PRESETS[int(i) - 1]
    except (ValueError, IndexError):
        return SIZE_PRESETS[4]


class Mesher:
    def __init__(self, model: Model, overrides=None):
        self.model = model
        self.comp = model.component
        self.mesh_node = self.comp.child("mesh")
        self.geom_node = self.comp.child("geometry")
        self.overrides = overrides

    def _has_flow(self) -> bool:
        return any(c.kind == "physics.spf" and c.enabled for c in self.comp.children)

    def run(self, progress=None) -> MeshResult:
        import gmsh
        t0 = time.time()
        with session("mesh"):
            gb = GeometryBuilder(self.model, self.geom_node, self.overrides)
            geo = gb.build(display=False)
            sd = geo.sdim
            L = max(geo.bbox[3] - geo.bbox[0], geo.bbox[4] - geo.bbox[1],
                    geo.bbox[5] - geo.bbox[2] if sd == 3 else 0.0) or 1.0
            mn = self.mesh_node
            order = mn.get("order", "auto")
            order = physics_order(self.model) if order == "auto" else int(order)
            preset = _preset(mn.get("size", NORMAL))
            hmax, hmin, hcurve = preset[1] * L, preset[2] * L, preset[4]
            hnarrow = preset[5]
            if mn.get("sequence") != "user" and self._has_flow():
                # "Fluid dynamics" calibration of the physics-controlled mesh: about twice as fine
                hmax, hmin, hcurve, hnarrow = hmax * 0.5, hmin * 0.5, hcurve * 0.6, min(1.0, hnarrow * 1.4)
            local = []
            recombine = False
            dists = []
            if mn.get("sequence") == "user":
                for ch in mn.children:
                    if not ch.enabled:
                        continue
                    if ch.kind == "mesh.size":
                        if ch.get("mode") == "custom":
                            h1 = gb.L(ch, "hmax")
                            h0 = gb.L(ch, "hmin")
                            hc = gb.N(ch, "hcurve")
                        else:
                            p = _preset(ch.get("preset", NORMAL))
                            h1, h0, hc = p[1] * L, p[2] * L, p[4]
                        if ch.get("scope", "entire") == "entire":
                            hmax, hmin, hcurve = h1, h0, hc
                            if ch.get("mode") != "custom":
                                hnarrow = _preset(ch.get("preset", NORMAL))[5]
                        else:
                            local.append((ch.get("scope"), ch.selection, h1))
                    elif ch.kind == "mesh.fquad":
                        recombine = True
                    elif ch.kind == "mesh.distribution":
                        dists.append((ch.selection, int(ch.get("n", 10))))
            gmsh.option.setNumber("Mesh.MeshSizeMax", hmax)
            gmsh.option.setNumber("Mesh.MeshSizeMin", hmin)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", max(4.0, 2 * math.pi / max(hcurve, 0.05)))
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
            gmsh.option.setNumber("Mesh.Algorithm", int(mn.get("algo2", "6")))
            gmsh.option.setNumber("Mesh.Algorithm3D", int(mn.get("algo3", "1")))
            gmsh.option.setNumber("Mesh.Optimize", 1 if mn.get("optimize", True) else 0)
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
            lv = levels_for(sd)
            # resolution of narrow regions: at least `layers` elements across each domain's thinnest side
            layers = max(1, int(round(hnarrow * 6)))
            point_h: dict[int, float] = {}
            for num, tag in geo.tag_of["domain"].items():
                b = geo.bbox_of["domain"][num]
                sides = [b[3] - b[0], b[4] - b[1]] + ([b[5] - b[2]] if sd == 3 else [])
                sides = [s for s in sides if s > 0]
                if not sides:
                    continue
                hd = min(hmax, min(sides) / layers)
                if hd < hmax * 0.999:
                    pts = gmsh.model.getBoundary([(sd, tag)], combined=False, oriented=False, recursive=True)
                    for pdim, ptag in pts:
                        if pdim == 0:
                            point_h[ptag] = min(point_h.get(ptag, hd), hd)
            for ptag, hp in point_h.items():
                gmsh.model.mesh.setSize([(0, ptag)], hp)
            for scope, sel, h in local:
                if sel is None:
                    continue
                level = scope
                dim = lv.get(level)
                if dim is None:
                    continue
                nums = sel.resolve(geo.numbers.get(level, []))
                dts = [(dim, geo.tag_of[level][n]) for n in nums]
                pts = gmsh.model.getBoundary(dts, combined=False, oriented=False, recursive=True) if dim > 0 else dts
                gmsh.model.mesh.setSize(list({p for p in pts if p[0] == 0}), h)
            for sel, n in dists:
                if sel is None:
                    continue
                level = "edge" if sd == 3 else "boundary"
                nums = sel.resolve(geo.numbers.get(level, []))
                for num in nums:
                    gmsh.model.mesh.setTransfiniteCurve(geo.tag_of[level][num], n + 1)
            if recombine and sd == 2:
                for num, tag in geo.tag_of["domain"].items():
                    gmsh.model.mesh.setRecombine(2, tag)
            if progress:
                progress(0.2, "Generating mesh")
            try:
                gmsh.model.mesh.generate(sd)
            except Exception as exc:
                raise GeometryError(f"Meshing failed: {exc}") from exc
            if order == 2:
                if progress:
                    progress(0.7, "Creating second-order elements")
                gmsh.model.mesh.setOrder(2)
            if progress:
                progress(0.8, "Extracting mesh")
            res = self._extract(geo, order)
        res.stats["time"] = time.time() - t0
        return res

    def _extract(self, geo: GeometryResult, order: int) -> MeshResult:
        import gmsh
        sd = geo.sdim
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        ntags = ntags.astype(np.int64)
        idx = np.full(int(ntags.max()) + 1 if len(ntags) else 1, -1, dtype=np.int64)
        idx[ntags] = np.arange(len(ntags))
        xyz = ncoords.reshape(-1, 3).copy()
        res = MeshResult(sd, geo.unit, geo.scale, xyz, geometry=geo, order=order)
        qual_all = []
        n_bulk = 0
        type_counts: dict[str, int] = {}
        for num, tag in geo.tag_of["domain"].items():
            blocks = []
            qs = []
            types, etags, conns = gmsh.model.mesh.getElements(sd, tag)
            for t, et, c in zip(types, etags, conns):
                t = int(t)
                if t not in GMSH_TO_ELMER:
                    continue
                nper = gmsh.model.mesh.getElementProperties(t)[3]
                conn = idx[c.astype(np.int64)].reshape(-1, nper)
                blocks.append((t, conn))
                n_bulk += len(conn)
                name = gmsh.model.mesh.getElementProperties(t)[0]
                type_counts[name] = type_counts.get(name, 0) + len(conn)
                try:
                    q = np.asarray(gmsh.model.mesh.getElementQualities(et, "minSICN"))
                    qs.append(q)
                except Exception:
                    pass
            res.bulk[num] = blocks
            if qs:
                res.quality[num] = np.concatenate(qs)
                qual_all.append(res.quality[num])
        bdim = sd - 1
        n_bnd = 0
        for num, tag in geo.tag_of["boundary"].items():
            if not geo.adj_up["boundary"].get(num):
                continue
            blocks = []
            types, _, conns = gmsh.model.mesh.getElements(bdim, tag)
            for t, c in zip(types, conns):
                t = int(t)
                if t not in GMSH_TO_ELMER:
                    continue
                nper = gmsh.model.mesh.getElementProperties(t)[3]
                conn = idx[c.astype(np.int64)].reshape(-1, nper)
                blocks.append((t, conn))
                n_bnd += len(conn)
            res.bnd[num] = blocks
        q = np.concatenate(qual_all) if qual_all else np.zeros(0)
        res.stats.update({
            "nodes": len(xyz), "elements": n_bulk, "boundary_elements": n_bnd, "types": type_counts,
            "min_quality": float(q.min()) if q.size else float("nan"),
            "avg_quality": float(q.mean()) if q.size else float("nan"),
            "order": order,
        })
        if n_bulk == 0:
            raise GeometryError("The mesh has no domain elements (is the geometry empty?)")
        return res


def build_mesh(model: Model, overrides=None, progress=None) -> MeshResult:
    return Mesher(model, overrides).run(progress)
