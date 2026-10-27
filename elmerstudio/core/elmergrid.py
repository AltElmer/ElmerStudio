"""ElmerGrid's structured mesher for axis-aligned rectangle (2D) and block (3D) layouts.

ElmerGrid (ElmerGrid Manual, chapters 2-3) meshes a tensor grid of subcells: "Subcell Limits" in each direction,
a "Material Structure" naming the material of every subcell of the xy-plane and, in 3D, "Extruded Material
Mappings" giving each z-layer's material for every plane material (0: no elements). Studio writes that grid from the
geometry: its limits are the coordinates of the geometry's points and a subcell's material is the domain that holds
its centre. That expresses exactly the geometries whose every boundary (and, in 3D, every edge) is straight and
parallel to a coordinate axis; anything else is refused (`problems`).

The numbering contract holds: ElmerGrid keeps material numbers as body ids, so bodies are the domain numbers. Its
boundary ids are not the geometry's (in 3D they come from its own extrusion heuristic), so every boundary element
is given the number of the geometry boundary it lies on, and a boundary ElmerGrid left without elements is an error.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile

import numpy as np

from .elmer_mesh import CORNERS, GMSH_TO_ELMER
from .geometry import GeometryError

# Elmer type -> (gmsh type, node permutation from Elmer to gmsh order or None)
ELMER_TO_GMSH = {}
for _gt, (_et, _perm) in GMSH_TO_ELMER.items():
    if _et not in ELMER_TO_GMSH:
        ELMER_TO_GMSH[_et] = (_gt, list(np.argsort(_perm)) if _perm else None)

AXES = "xyz"


def _tol(geo) -> float:
    return 1e-6 * geo.diag


def _flat_axes(bb, tol) -> int:
    """Number of coordinate directions in which a box has no extent."""
    return sum(1 for i in range(3) if bb[i + 3] - bb[i] <= tol)


def problems(geo) -> list[str]:
    """Why ElmerGrid cannot mesh this geometry (empty when it can). Uses only the built geometry's boxes: a curve
    with no extent in all but one direction is a straight segment along that axis, and a face flat in one direction
    lies in a plane normal to it."""
    if geo is None:
        return []
    sd = geo.sdim
    if sd not in (2, 3):
        return [f"ElmerGrid meshes 2D and 3D components here, not {sd}D ones."]
    if not geo.numbers.get("domain"):
        return ["The geometry has no domains."]
    tol = _tol(geo)
    out = []
    # in 2D a boundary is a curve, flat in z and one more axis; in 3D faces are flat in one axis, edges in two
    checks = [("boundary", 2 if sd == 2 else 1, "boundaries")] + ([("edge", 2, "edges")] if sd == 3 else [])
    for level, need, plural in checks:
        bad = [n for n, bb in sorted(geo.bbox_of.get(level, {}).items()) if _flat_axes(bb, tol) < need]
        if bad:
            what = "straight and parallel to the x or y axis" if sd == 2 else (
                "plane and normal to the x, y or z axis" if level == "boundary" else
                "straight and parallel to the x, y or z axis")
            shown = ", ".join(str(n) for n in bad[:8]) + (" ..." if len(bad) > 8 else "")
            out.append(f"ElmerGrid meshes only layouts of axis-aligned {'rectangles' if sd == 2 else 'blocks'}: "
                       f"{plural} {shown} {'are' if len(bad) > 1 else 'is'} not {what}.")
    return out


def sequence_problems(mesh_node, sdim, order) -> list[str]:
    """Mesh settings ElmerGrid cannot honour (it has no local sizes, distributions or tetrahedra)."""
    out = []
    if mesh_node.get("sequence") == "user":
        for ch in mesh_node.children:
            if not ch.enabled:
                continue
            if ch.kind == "mesh.size" and ch.get("scope", "entire") != "entire":
                out.append(f"{ch.label}: ElmerGrid uses one element size for the entire geometry; "
                           "set the Size's geometric entity level to Entire geometry.")
            elif ch.kind == "mesh.distribution":
                out.append(f"{ch.label}: Distribution is not available with ElmerGrid; "
                           "it divides every subcell into equal elements no larger than the element size.")
            elif ch.kind == "mesh.ftet":
                out.append(f"{ch.label}: ElmerGrid makes hexahedra or prisms, not tetrahedra.")
    if sdim == 3 and order == 2 and element_shape(mesh_node) == "tri":
        out.append("ElmerGrid extrudes only linear triangles into prisms: use quadrilaterals (hexahedra) "
                   "or linear elements.")
    return out


def element_shape(mesh_node) -> str:
    """'quad' or 'tri': the Mesh node's ElmerGrid element shape, or the user-controlled sequence's operation."""
    shape = mesh_node.get("eg_shape", "quad")
    if mesh_node.get("sequence") == "user":
        for ch in mesh_node.children:
            if ch.enabled and ch.kind == "mesh.ftri":
                shape = "tri"
            elif ch.enabled and ch.kind in ("mesh.fquad", "mesh.mapped"):
                shape = "quad"
    return shape


def _point_xyz(geo) -> dict:
    """Geometry point coordinates: their box centres (a point's box is padded, or shifted by the pad removal)."""
    return {n: 0.5 * (np.array(bb[:3]) + np.array(bb[3:])) for n, bb in geo.bbox_of.get("point", {}).items()}


def _merge(values, tol):
    out = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return np.array(out)


class Layout:
    """The subcell grid of a geometry: limits per axis and the domain of every subcell (0: outside)."""

    def __init__(self, geo):
        import gmsh
        sd = geo.sdim
        tol = _tol(geo)
        pts = np.array(list(_point_xyz(geo).values()))
        self.limits = [_merge(pts[:, i], tol) for i in range(sd)]
        if any(len(lim) < 2 for lim in self.limits):
            raise GeometryError("ElmerGrid: the geometry has no extent in some direction")
        centres = np.meshgrid(*[0.5 * (lim[1:] + lim[:-1]) for lim in self.limits], indexing="ij")
        shape = tuple(len(lim) - 1 for lim in self.limits)
        self.cells = np.zeros(shape, dtype=int)
        flat = np.column_stack([c.ravel() for c in centres] + ([np.zeros(centres[0].size)] if sd == 2 else []))
        owner = np.zeros(len(flat), dtype=int)
        for num, tag in geo.tag_of["domain"].items():
            b = geo.bbox_of["domain"][num]
            near = np.all((flat >= np.array(b[:3]) - tol) & (flat <= np.array(b[3:]) + tol), axis=1)
            for i in np.nonzero(near)[0]:
                if gmsh.model.isInside(sd, tag, list(flat[i])):
                    if owner[i]:
                        raise GeometryError(f"ElmerGrid: domains {owner[i]} and {num} overlap")
                    owner[i] = num
        self.cells = owner.reshape(shape)
        # every domain must be exactly its subcells (the axis-aligned check makes it so; this proves it)
        widths = np.meshgrid(*[np.diff(lim) for lim in self.limits], indexing="ij")
        vol = np.prod(widths, axis=0)
        for num in geo.numbers["domain"]:
            got = float(vol[self.cells == num].sum()) * geo.scale ** sd
            want = geo.measure["domain"].get(num, float("nan"))
            if not math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-12 * geo.scale ** sd):
                raise GeometryError(f"ElmerGrid: the subcell grid does not reproduce domain {num} "
                                    f"({got:.6g} vs {want:.6g} m^{sd})")


def divisions(lengths, h) -> list[int]:
    """Equal elements no larger than h in every subcell."""
    return [max(1, math.ceil(length / h - 1e-9)) for length in lengths]


def grd_text(lay: Layout, h: float, order: int, shape: str) -> str:
    """The ElmerGrid input (.grd) for a layout (ElmerGrid Manual, section 2.1)."""
    sd = len(lay.limits)
    cells = lay.cells
    lines = ["***** Elmer Studio: ElmerGrid structured mesh *****", "Version = 210903",
             f"Coordinate System = Cartesian {sd}D",
             f"Subcell Divisions in {sd}D = " + " ".join(str(len(lim) - 1) for lim in lay.limits)]
    for i, lim in enumerate(lay.limits):
        lines.append(f"Subcell Limits {i + 1} = " + " ".join(f"{v:.17g}" for v in lim))
    if sd == 2:
        plane = cells
        mapping = None
    else:
        # one plane material per distinct column of z-layer domains; the extrusion maps it per layer
        cols = {}
        plane = np.zeros(cells.shape[:2], dtype=int)
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                prof = tuple(int(v) for v in cells[i, j, :])
                if any(prof):
                    plane[i, j] = cols.setdefault(prof, len(cols) + 1)
        mapping = [[prof[k] for prof in cols] for k in range(cells.shape[2])]
    lines.append("Material Structure in 2D")
    for j in reversed(range(plane.shape[1])):        # the first row is the top (largest y)
        lines.append("  " + " ".join(str(int(plane[i, j])) for i in range(plane.shape[0])))
    lines.append("End")
    nmat = int(plane.max())
    lines.append(f"Materials Interval = 1 {nmat}")
    if mapping is not None:
        lines.append("Extruded Material Mappings")
        lines += ["  " + " ".join(str(v) for v in row) for row in mapping]
        lines.append("End")
    # boundaries between every pair of neighbouring plane materials; Studio renumbers them from the geometry
    pairs = set()
    nx, ny = plane.shape
    pad = np.zeros((nx + 2, ny + 2), dtype=int)
    pad[1:-1, 1:-1] = plane
    for a, b in ((pad[1:, :], pad[:-1, :]), (pad[:, 1:], pad[:, :-1])):
        for u, v in zip(a.ravel(), b.ravel()):
            if u != v:
                pairs.add((min(u, v), max(u, v)))
    lines.append("Boundary Definitions")
    lines.append("! type out int double")
    for k, (u, v) in enumerate(sorted(pairs)):
        lines.append(f"  {k + 1} {u} {v} 1")
    lines.append("End")
    et = {("quad", 1): 404, ("quad", 2): 409 if sd == 2 else 408, ("tri", 1): 303, ("tri", 2): 306}[(shape, order)]
    lines.append(f"Elementtype = {et}")
    lines.append("Numbering = Horizontal")
    for i, lim in enumerate(lay.limits):
        lines.append(f"Element Divisions {i + 1} = " + " ".join(str(n) for n in divisions(np.diff(lim), h)))
    return "\n".join(lines) + "\n"


def run(inst, grd: str, timeout=600):
    """Run ElmerGrid on a .grd text; returns (nodes (N,3), elements, boundary, log). Elements are
    (body, elmer type, conn 0-based) rows grouped per type; boundary rows also carry the two parent element ids."""
    if inst is None or not inst.grid:
        where = f" next to ElmerSolver in {inst.bin}" if inst is not None else " (Elmer was not found)"
        raise GeometryError(f"ElmerGrid was not found{where}. Install Elmer with ElmerGrid or choose Gmsh.")
    with tempfile.TemporaryDirectory(prefix="es_elmergrid_") as d:
        with open(os.path.join(d, "studio.grd"), "w") as f:
            f.write(grd)
        cmd = [inst.grid, "1", "2", "studio.grd", "-out", "mesh", "-decimals", "15"]
        try:
            p = subprocess.run(cmd, cwd=d, env=inst.env(), capture_output=True, text=True, errors="replace",
                               timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as exc:
            raise GeometryError(f"ElmerGrid did not finish in {timeout} s") from exc
        log = p.stdout + p.stderr
        md = os.path.join(d, "mesh")
        if p.returncode != 0 or not os.path.isfile(os.path.join(md, "mesh.boundary")):
            tail = " / ".join(ln.strip() for ln in log.strip().splitlines()[-4:])
            raise GeometryError(f"ElmerGrid failed (exit code {p.returncode}): {tail}")
        nodes_txt = np.loadtxt(os.path.join(md, "mesh.nodes"), ndmin=2)
        ids = nodes_txt[:, 0].astype(np.int64)
        index = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
        index[ids] = np.arange(len(ids))
        nodes = nodes_txt[:, 2:5].copy()

        def rows(path, head):
            groups: dict[int, list] = {}
            with open(path) as f:
                for line in f:
                    v = line.split()
                    if v:
                        groups.setdefault(int(v[head - 1]), []).append([int(x) for x in v])
            return {t: np.array(r, dtype=np.int64) for t, r in groups.items()}

        elements = rows(os.path.join(md, "mesh.elements"), 3)      # id body type nodes...
        boundary = rows(os.path.join(md, "mesh.boundary"), 5)      # id bc parent1 parent2 type nodes...
    elems = {t: (r[:, 0], r[:, 1], index[r[:, 3:]]) for t, r in elements.items()}
    bnds = {t: (r[:, 2], r[:, 3], index[r[:, 5:]]) for t, r in boundary.items()}
    return nodes, elems, bnds, log


def _gmsh_block(et, conn):
    if et not in ELMER_TO_GMSH:
        raise GeometryError(f"ElmerGrid produced Elmer element type {et}, which Studio does not handle")
    gt, perm = ELMER_TO_GMSH[et]
    return gt, (conn[:, perm] if perm is not None else conn)


def mesh(mesher, gb, geo, h: float, order: int, progress=None):
    """Mesh the built geometry (current gmsh model) with ElmerGrid and return a MeshResult in Studio's numbering."""
    import gmsh

    from .meshing import MeshResult
    msgs = problems(geo) + sequence_problems(mesher.mesh_node, geo.sdim, order)
    if msgs:
        raise GeometryError(" ".join(msgs))
    sd = geo.sdim
    tol = _tol(geo)
    shape = element_shape(mesher.mesh_node)
    if progress:
        progress(0.2, "Laying out the ElmerGrid subcells")
    lay = Layout(geo)
    grd = grd_text(lay, h, order, shape)
    if progress:
        progress(0.4, "Running ElmerGrid")
    nodes, elems, bnds, _log = run(mesher.inst, grd)
    res = MeshResult(sd, geo.unit, geo.scale, nodes, geometry=geo, order=order)
    res.stats["mesher"] = "ElmerGrid"
    res.stats["grd"] = grd
    # bulk: ElmerGrid's body ids are the material numbers Studio wrote, i.e. the domain numbers
    elem_body: dict[int, int] = {}
    type_counts: dict[str, int] = {}
    n_bulk = 0
    for et, (eids, body, conn) in sorted(elems.items()):
        gt, c = _gmsh_block(et, conn)
        name = gmsh.model.mesh.getElementProperties(gt)[0]
        type_counts[name] = type_counts.get(name, 0) + len(c)
        n_bulk += len(c)
        elem_body.update(zip(eids.tolist(), body.tolist()))
        for num in np.unique(body):
            res.bulk.setdefault(int(num), []).append((gt, c[body == num]))
    if set(res.bulk) != set(geo.numbers["domain"]):
        raise GeometryError(f"ElmerGrid bodies {sorted(res.bulk)} are not the geometry's domains "
                            f"{geo.numbers['domain']}")
    # boundaries: each element gets the number of the geometry boundary it lies on
    if progress:
        progress(0.7, "Numbering boundaries from the geometry")
    bboxes = geo.bbox_of["boundary"]
    up = {b: set(geo.adj_up["boundary"].get(b, [])) for b in geo.numbers["boundary"]}
    n_bnd = 0
    for et, (p1, p2, conn) in sorted(bnds.items()):
        gt, c = _gmsh_block(et, conn)
        k = CORNERS[et]
        cen = nodes[c[:, :k]].mean(axis=1)
        par = [frozenset(x for x in (elem_body.get(int(a), 0), elem_body.get(int(b), 0)) if x)
               for a, b in zip(p1, p2)]
        target = np.zeros(len(c), dtype=int)
        for i, (x, ps) in enumerate(zip(cen, par)):
            if elem_body.get(int(p1[i])) == elem_body.get(int(p2[i]), -1):
                continue                               # a face inside one domain: not a geometry boundary
            cand = [b for b, bb in bboxes.items() if up[b] == ps and
                    all(bb[j] - tol <= x[j] <= bb[j + 3] + tol for j in range(3))]
            if len(cand) > 1:                          # coplanar boundaries with overlapping boxes: the nearest
                dist = [np.linalg.norm(np.array(gmsh.model.getClosestPoint(sd - 1, geo.tag_of["boundary"][b],
                                                                          list(x))[0]) - x) for b in cand]
                cand = [cand[int(np.argmin(dist))]]
            if not cand:
                raise GeometryError(f"ElmerGrid boundary element at {np.round(x, 9).tolist()} (between domains "
                                    f"{sorted(ps) or 'none'}) lies on no geometry boundary")
            target[i] = cand[0]
        for b in np.unique(target):
            if b:
                res.bnd.setdefault(int(b), []).append((gt, c[target == b]))
                n_bnd += int((target == b).sum())
    missing = [b for b in geo.numbers["boundary"] if up[b] and b not in res.bnd]
    if missing:
        raise GeometryError(f"ElmerGrid created no elements on boundaries {missing} (a boundary inside a single "
                            "domain cannot be expressed in its subcell grid); choose Gmsh")
    for num, xyz in _point_xyz(geo).items():                  # geometry vertices -> mesh node index
        d = np.linalg.norm(nodes - xyz, axis=1)
        i = int(np.argmin(d))
        if d[i] <= tol:
            res.point_nodes[num] = i
    q = _quality(res)
    res.stats.update({
        "nodes": len(nodes), "elements": n_bulk, "boundary_elements": n_bnd, "types": type_counts,
        "min_quality": float(q.min()) if q.size else float("nan"),
        "avg_quality": float(q.mean()) if q.size else float("nan"),
        "order": order,
        "subcells": "x".join(str(len(lim) - 1) for lim in lay.limits),
    })
    res.messages.append(
        f"Meshed with ElmerGrid: {res.stats['subcells']} subcells, elements no larger than {h:.4g} {geo.unit}. "
        "ElmerGrid uses only the maximum element size; the minimum size, growth rate, curvature factor and "
        "narrow-region resolution apply to Gmsh.")
    return res


def _quality(res) -> np.ndarray:
    """gmsh's minSICN quality of the ElmerGrid elements (the measure the Gmsh path reports), per domain."""
    import gmsh
    current = gmsh.model.getCurrent()
    gmsh.model.add("es_elmergrid_quality")
    try:
        sd = res.sdim
        tags = np.arange(1, len(res.nodes) + 1)
        ent = gmsh.model.addDiscreteEntity(sd)
        gmsh.model.mesh.addNodes(sd, ent, tags.tolist(), res.nodes.ravel().tolist())
        nxt = 1
        allq = []
        for num, blocks in sorted(res.bulk.items()):
            qs = []
            for gt, conn in blocks:
                et = np.arange(nxt, nxt + len(conn))
                nxt += len(conn)
                gmsh.model.mesh.addElementsByType(ent, gt, et.tolist(), (conn + 1).ravel().tolist())
                qs.append(np.asarray(gmsh.model.mesh.getElementQualities(et.tolist(), "minSICN")))
            res.quality[num] = np.concatenate(qs) if qs else np.zeros(0)
            allq.append(res.quality[num])
        return np.concatenate(allq) if allq else np.zeros(0)
    except Exception:  # noqa: BLE001 - quality is informative; the mesh stands without it
        res.quality.clear()
        return np.zeros(0)
    finally:
        gmsh.model.remove()
        gmsh.model.setCurrent(current)
