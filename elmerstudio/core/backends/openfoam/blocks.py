"""blockMesh from Studio geometry: unions of axis-aligned rectangles (2D) or blocks (3D), and nothing else.

The geometry is cut into a tensor grid by every vertex coordinate.  Each grid cell inside a domain becomes a
blockMesh block in cell zone ``dom<N>``; each grid face on a boundary goes into patch ``bnd<N>``, so domains and
boundaries keep the geometry's numbers.  A shape that is not exactly such a union (a circle, an arc, a rotated
rectangle, a curved face) is refused: the grid cells inside a domain must add up to its area or volume.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...geometry import GeometryBuilder, GeometryError, GeometryResult, apply_named_selections
from ...gmsh_session import session
from ...model import Model, is_axisymmetric

# OpenFOAM hex faces (cellModel "hex"), outward normals by the right-hand rule: x-, x+, y-, y+, z-, z+
HEX_FACES = [(0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7)]


class BlockRefusal(GeometryError):
    """The geometry cannot be meshed exactly by blockMesh (the message says why)."""


@dataclass
class BlockLayout:
    geo: GeometryResult
    sdim: int
    coords: list                    # grid coordinates per axis (geometry units); 2D: z = [0, thickness]
    cells: np.ndarray               # domain number per grid cell (0 = outside), shape (nx, ny, nz)
    faces: dict = field(default_factory=dict)     # boundary -> [(axis, (i, j, k) of the grid point at the face's low corner)]
    interior: set = field(default_factory=set)    # boundaries between two domains (internal faces, no patch)
    counts: list = field(default_factory=list)    # cells per grid interval, per axis
    grading: list = field(default_factory=list)   # grading per grid interval, per axis: a ratio or [(len, cells, ratio)]

    @property
    def scale(self) -> float:
        return self.geo.scale

    def block_list(self):
        """(i, j, k, domain) of every block."""
        return [(i, j, k, int(d)) for (i, j, k), d in np.ndenumerate(self.cells) if d > 0]


def _gtype(dim, tag) -> str:
    import gmsh
    try:
        return gmsh.model.getType(dim, tag)
    except Exception:
        return "?"


def _axis_aligned(bb, tol) -> list[int]:
    """Axes along which a bounding box has zero extent."""
    return [i for i in range(3) if bb[i + 3] - bb[i] <= tol]


def analyse(model: Model, overrides=None) -> BlockLayout:
    """Build the geometry and cut it into blocks; raises BlockRefusal with the reason when that is not exact."""
    import gmsh
    comp = model.component
    if is_axisymmetric(comp):
        raise BlockRefusal("2D axisymmetric components are not supported by the OpenFOAM backend (a wedge mesh is "
                           "not verified yet); use Elmer.")
    with session("blockmesh"):
        gb = GeometryBuilder(model, comp.child("geometry"), overrides)
        geo = gb.build(display=False)
        apply_named_selections(model, geo)
        sd = geo.sdim
        if sd not in (2, 3):
            raise BlockRefusal("The OpenFOAM backend needs a 2D or 3D component.")
        if not geo.numbers.get("domain"):
            raise BlockRefusal("The geometry has no domains.")
        tol = max(geo.diag * 1e-9, 1e-14)
        bnd_level, curve_level = "boundary", ("edge" if sd == 3 else "boundary")
        for num, tag in geo.tag_of[curve_level].items():          # straight, axis-aligned edges only
            t = _gtype(1, tag)
            flat = _axis_aligned(geo.bbox_of[curve_level][num], tol)
            if t != "Line" or len(flat) != 2:
                what = "Boundary" if sd == 2 else "Edge"
                shape = "curved" if t != "Line" else "not parallel to an axis"
                raise BlockRefusal(f"{what} {num} is {shape} ({t}): blockMesh meshes only unions of axis-aligned "
                                   f"{'rectangles' if sd == 2 else 'blocks'}. Use Elmer for this geometry.")
        if sd == 3:
            for num, tag in geo.tag_of[bnd_level].items():
                t = _gtype(2, tag)
                if t != "Plane" or len(_axis_aligned(geo.bbox_of[bnd_level][num], tol)) != 1:
                    raise BlockRefusal(f"Boundary {num} is not a plane normal to an axis ({t}): blockMesh meshes only "
                                       "unions of axis-aligned blocks. Use Elmer for this geometry.")
        # exact vertex coordinates (bounding boxes carry OpenCASCADE's tolerance padding)
        vxyz = {num: np.array(gmsh.model.getValue(0, tag, []), float) for num, tag in geo.tag_of["point"].items()}
        exact = {}
        for num, tag in geo.tag_of["boundary"].items():
            pts = [vxyz[geo.num_of["point"][t]] for d, t in
                   gmsh.model.getBoundary([(sd - 1, tag)], combined=False, oriented=False, recursive=True) if d == 0]
            P = np.array(pts) if pts else np.zeros((1, 3))
            exact[num] = tuple(P.min(axis=0)) + tuple(P.max(axis=0))
        geo.bbox_of = dict(geo.bbox_of, boundary=exact, point={n: tuple(v) + tuple(v) for n, v in vxyz.items()})
        coords = []
        for a in range(sd):
            vals = sorted(v[a] for v in vxyz.values())
            grid = [vals[0]]
            for v in vals[1:]:
                if v - grid[-1] > tol:
                    grid.append(v)
            coords.append(np.array(grid, float))
        if sd == 2:
            t = min(float(np.min(np.diff(coords[0]))), float(np.min(np.diff(coords[1]))))
            coords.append(np.array([0.0, t]))
        shape = tuple(len(c) - 1 for c in coords)
        cells = np.zeros(shape, dtype=np.int64)
        dom_tags = geo.tag_of["domain"]
        for idx in np.ndindex(*shape):
            c = [0.5 * (coords[a][idx[a]] + coords[a][idx[a] + 1]) for a in range(3)]
            if sd == 2:
                c[2] = 0.0
            inside = [num for num, tag in dom_tags.items() if gmsh.model.isInside(sd, tag, c) > 0]
            if len(inside) > 1:
                raise BlockRefusal(f"Domains {inside} overlap at {c}.")
            cells[idx] = inside[0] if inside else 0
        # exactness: the blocks of each domain fill it
        fac = geo.scale
        for num in dom_tags:
            vol = 0.0
            for idx in zip(*np.nonzero(cells == num)):
                ext = [coords[a][idx[a] + 1] - coords[a][idx[a]] for a in range(sd)]
                vol += math.prod(ext) * fac ** sd
            meas = geo.measure["domain"].get(num, float("nan"))
            if not (abs(vol - meas) <= 1e-7 * max(abs(meas), 1e-300)):
                raise BlockRefusal(f"Domain {num} is not a union of axis-aligned {'rectangles' if sd == 2 else 'blocks'} "
                                   f"(the blocks cover {vol:.6g} of its {'area' if sd == 2 else 'volume'} {meas:.6g}). "
                                   "Use Elmer for this geometry.")
        layout = BlockLayout(geo, sd, coords, cells)
        _classify_faces(layout, tol)
        _sizes(model, layout, gb)
    return layout


def _face_centre(coords, a, p):
    return [coords[b][p[b]] if b == a else 0.5 * (coords[b][p[b]] + coords[b][p[b] + 1]) for b in range(3)]


def _classify_faces(L: BlockLayout, tol):
    """Every grid face between two different domains (or a domain and the outside) onto its Studio boundary."""
    import gmsh
    geo, sd, coords, cells = L.geo, L.sdim, L.coords, L.cells
    shape = cells.shape
    bnds = {num: geo.bbox_of["boundary"][num] for num in geo.numbers["boundary"]}
    area = {num: 0.0 for num in bnds}
    for a in range(sd):
        for p in np.ndindex(*[shape[b] + (1 if b == a else 0) for b in range(3)]):
            lo = list(p)
            lo[a] -= 1
            d0 = cells[tuple(lo)] if lo[a] >= 0 else 0
            d1 = cells[p] if p[a] < shape[a] else 0
            if d0 == d1:
                continue
            c = _face_centre(coords, a, p)
            if sd == 2:
                c[2] = 0.0
            hits = []
            for num, bb in bnds.items():
                if abs(bb[a] - c[a]) > tol or abs(bb[a + 3] - c[a]) > tol:
                    continue
                if any(c[b] < bb[b] - tol or c[b] > bb[b + 3] + tol for b in range(sd) if b != a):
                    continue
                if sd == 3 and gmsh.model.isInside(2, geo.tag_of["boundary"][num], c) <= 0:
                    continue
                hits.append(num)
            if len(hits) != 1:
                raise BlockRefusal(f"The block face at {c} lies on {len(hits)} boundaries; the geometry cannot be "
                                   "split into blocks exactly.")
            b = hits[0]
            L.faces.setdefault(b, []).append((a, tuple(p)))
            if d0 and d1:
                L.interior.add(b)
            ext = [coords[x][p[x] + 1] - coords[x][p[x]] for x in range(sd) if x != a]
            area[b] += math.prod(ext) * geo.scale ** (sd - 1)
    for num in bnds:
        if not geo.adj_up["boundary"].get(num):
            continue
        meas = geo.measure["boundary"].get(num, float("nan"))
        if not (abs(area[num] - meas) <= 1e-7 * max(abs(meas), 1e-300)):
            raise BlockRefusal(f"Boundary {num} is not covered by block faces ({area[num]:.6g} of {meas:.6g}): "
                               "an embedded boundary inside a domain cannot be meshed by blockMesh.")


def _sizes(model: Model, L: BlockLayout, gb: GeometryBuilder):
    """Cells per grid interval from the Mesh node's element sizes, Size nodes and Distributions (as Gmsh reads them)."""
    import gmsh
    from ...meshing import NORMAL, _preset
    geo, sd = L.geo, L.sdim
    comp = model.component
    mn = comp.child("mesh")
    b = geo.bbox
    Lmax = max(b[3] - b[0], b[4] - b[1], b[5] - b[2] if sd == 3 else 0.0) or 1.0
    preset = _preset(mn.get("size", NORMAL))
    hmax, hnarrow = preset[1] * Lmax, preset[5]
    has_flow = any(c.kind == "physics.spf" and c.enabled for c in comp.children)
    if mn.get("sequence") != "user" and has_flow:
        hmax, hnarrow = hmax * 0.5, min(1.0, hnarrow * 1.4)
    local: dict = {}
    dists = []
    if mn.get("sequence") == "user":
        for ch in mn.children:
            if not ch.enabled:
                continue
            if ch.kind == "mesh.size":
                if ch.get("mode") == "custom":
                    h1 = gb.L(ch, "hmax")
                    if ch.get("scope", "entire") == "entire" and "hnarrow" in ch.props:
                        hnarrow = gb.N(ch, "hnarrow")
                else:
                    p = _preset(ch.get("preset", NORMAL))
                    h1 = p[1] * Lmax
                    if ch.get("scope", "entire") == "entire":
                        hnarrow = p[5]
                scope = ch.get("scope", "entire")
                if scope == "entire":
                    hmax = h1
                elif scope == "domain":
                    for d in (ch.selection.resolve(geo.numbers["domain"]) if ch.selection else []):
                        local[d] = min(local.get(d, h1), h1)
                else:
                    raise BlockRefusal(f"{ch.label}: a Size on {scope}s has no blockMesh equivalent; use a Size on "
                                       "domains or a Distribution.")
            elif ch.kind == "mesh.distribution":
                dists.append(ch)
    hdom = {}
    for d in geo.numbers["domain"]:
        h = min(hmax, local.get(d, hmax))
        bb = geo.bbox_of["domain"][d]
        sides = [s for s in (bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2] if sd == 3 else 0.0) if s > 0]
        if sides and hnarrow > 0:
            h = min(h, min(sides) / float(hnarrow))
        hdom[d] = h
    L.counts, L.grading = [], []
    for a in range(3):
        n_int = len(L.coords[a]) - 1
        if a >= sd:
            L.counts.append([1] * n_int)
            L.grading.append([1.0] * n_int)
            continue
        cnt = []
        for q in range(n_int):
            slab = np.take(L.cells, q, axis=a)
            doms = [int(d) for d in np.unique(slab) if d > 0]
            h = min((hdom[d] for d in doms), default=hmax)
            ln = L.coords[a][q + 1] - L.coords[a][q]
            cnt.append(max(1, int(math.ceil(ln / h - 1e-9))))
        L.counts.append(cnt)
        L.grading.append([1.0] * n_int)
    level = "edge" if sd == 3 else "boundary"
    tol = max(geo.diag * 1e-9, 1e-14)
    fixed: dict = {}
    for dn in dists:
        n = int(dn.get("n", 10))
        try:
            ratio = gb.N(dn, "ratio") if str(dn.get("ratio", "1")).strip() else 1.0
        except GeometryError:
            ratio = 1.0
        ratio = ratio if ratio > 0 else 1.0
        if dn.get("reverse", False):
            ratio = 1.0 / ratio
        for num in (dn.selection.resolve(geo.numbers.get(level, [])) if dn.selection else []):
            bb = geo.bbox_of[level][num]
            axes = [i for i in range(sd) if bb[i + 3] - bb[i] > tol]
            a = axes[0]
            q0 = int(np.argmin(np.abs(L.coords[a] - bb[a])))
            q1 = int(np.argmin(np.abs(L.coords[a] - bb[a + 3])))
            if q1 != q0 + 1:
                raise BlockRefusal(f"{dn.label}: {level} {num} spans {q1 - q0} block intervals; a Distribution must "
                                   "cover exactly one (split it at the vertices that cut it, or remove it).")
            tag = geo.tag_of[level][num]
            u0 = gmsh.model.getParametrizationBounds(1, tag)[0][0]
            start = gmsh.model.getValue(1, tag, [u0])
            r = ratio if start[a] <= 0.5 * (bb[a] + bb[a + 3]) else 1.0 / ratio
            if dn.get("symmetric", False) and n >= 2 and abs(r - 1) > 1e-12:
                r = ratio
                g = [(0.5, 0.5, r), (0.5, 0.5, 1.0 / r)]
            else:
                g = r
            key = (n, repr(g))
            if fixed.get((a, q0), key) != key:
                raise BlockRefusal(f"{dn.label}: block interval {q0 + 1} along {'xyz'[a]} has two different "
                                   "Distributions; blockMesh needs one cell count and grading per interval.")
            fixed[(a, q0)] = key
            L.counts[a][q0] = n
            L.grading[a][q0] = g


# --------------------------------------------------------------------------- blockMeshDict
def _grading_text(g) -> str:
    if isinstance(g, (list, tuple)):
        return "(" + " ".join(f"({a:g} {b:g} {r:.12g})" for a, b, r in g) + ")"
    return f"{float(g):.12g}"


def block_mesh_dict(L: BlockLayout, patch_types: dict) -> str:
    """The blockMeshDict text.  *patch_types*: boundary -> OpenFOAM patch type (wall, patch, symmetryPlane)."""
    from .dicts import header
    coords = L.coords
    used: dict = {}
    vlist = []

    def vid(i, j, k):
        key = (i, j, k)
        if key not in used:
            used[key] = len(vlist)
            vlist.append(key)
        return used[key]

    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    blocks = []
    for i, j, k, d in L.block_list():
        v = [vid(i + c[0], j + c[1], k + c[2]) for c in corners]
        n = (L.counts[0][i], L.counts[1][j], L.counts[2][k])
        g = (L.grading[0][i], L.grading[1][j], L.grading[2][k])
        blocks.append(f"    hex ({' '.join(map(str, v))}) dom{d} ({n[0]} {n[1]} {n[2]}) "
                      f"simpleGrading ({' '.join(_grading_text(x) for x in g)})")
    patches = []

    def face_vertices(a, p, side):
        """The 4 vertex labels of the grid face at point p normal to axis a, ordered for the block on *side*."""
        i, j, k = p
        cell = list(p)
        if side == 0:                     # the block below the face: its + face along a
            cell[a] -= 1
        hexf = HEX_FACES[2 * a + (1 if side == 0 else 0)]
        return [vid(cell[0] + corners[c][0], cell[1] + corners[c][1], cell[2] + corners[c][2]) for c in hexf]

    for b in sorted(L.faces):
        if b in L.interior:
            continue
        fl = []
        for a, p in L.faces[b]:
            lo = list(p)
            lo[a] -= 1
            inside_low = lo[a] >= 0 and L.cells[tuple(lo)] > 0
            fl.append(face_vertices(a, p, 0 if inside_low else 1))
        body = "\n".join(f"            ({' '.join(map(str, f))})" for f in fl)
        patches.append(f"    bnd{b}\n    {{\n        type {patch_types.get(b, 'patch')};\n        faces\n        (\n"
                       f"{body}\n        );\n    }}")
    if L.sdim == 2:
        fl = []
        for i, j, k, _d in L.block_list():
            for side in (4, 5):
                fl.append([vid(i + corners[c][0], j + corners[c][1], k + corners[c][2]) for c in HEX_FACES[side]])
        body = "\n".join(f"            ({' '.join(map(str, f))})" for f in fl)
        patches.append(f"    frontAndBack\n    {{\n        type empty;\n        faces\n        (\n{body}\n        );\n    }}")
    verts = "\n".join(f"    ({coords[0][i]:.15g} {coords[1][j]:.15g} {coords[2][k]:.15g})" for i, j, k in vlist)
    return (header("dictionary", "blockMeshDict", "system")
            + f"scale {L.scale:.15g};\n\nvertices\n(\n{verts}\n);\n\nblocks\n(\n" + "\n".join(blocks)
            + "\n);\n\nedges\n(\n);\n\ndefaultPatch\n{\n    name unassigned;\n    type patch;\n}\n\nboundary\n(\n"
            + "\n".join(patches) + "\n);\n\nmergePatchPairs\n(\n);\n")
