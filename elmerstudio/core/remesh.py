"""Remeshing a deformed 2D mesh for deforming electrodes (COMSOL: automatic remeshing).

The new mesh is built from the deformed mesh's own boundaries: every boundary number becomes a chain of straight
segments between its present nodes, so domain and boundary numbers -- and with them every feature's selection --
stay as they are.  The chains of moving boundaries are re-spaced along their arc length with the grading they had in
the first mesh (otherwise nodes bunch where the metal recedes fastest and the boundary folds), fixed boundaries keep
their nodes exactly, and gmsh fills the domains with element sizes taken from the boundary spacing.
"""
from __future__ import annotations

import numpy as np

from .gmsh_session import session
from .meshing import GMSH_TO_ELMER, MeshResult


class RemeshError(RuntimeError):
    pass


def boundary_chains(bnd_blocks):
    """Ordered corner-node chains (lists of node ids) of one boundary's line elements."""
    edges = [tuple(int(v) for v in row[:2]) for _gt, conn in bnd_blocks for row in np.asarray(conn)]
    adj: dict = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    used = set()
    chains = []
    ends = [n for n, nb in adj.items() if len(nb) != 2]
    for start in ends + [n for n in adj if n not in ends]:
        for nxt in adj[start]:
            e = frozenset((start, nxt))
            if e in used:
                continue
            chain = [start]
            prev, cur = start, nxt
            used.add(e)
            while True:
                chain.append(cur)
                if cur == start or len(adj[cur]) != 2:
                    break
                a, b = adj[cur]
                nn = b if a == prev else a
                e = frozenset((cur, nn))
                if e in used:
                    break
                used.add(e)
                prev, cur = cur, nn
            chains.append(chain)
    return chains


def _arclength(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def chain_profile(P):
    """Normalised arc-length positions of a chain's nodes (its grading)."""
    s = _arclength(P)
    return s / s[-1] if s[-1] > 0 else np.linspace(0, 1, len(P))


def respace(P, profile):
    """Points along polyline P at the normalised arc-length positions *profile* (end points kept)."""
    s = _arclength(P)
    if s[-1] <= 0:
        return P.copy()
    t = np.asarray(profile, float) * s[-1]
    out = np.column_stack([np.interp(t, s, P[:, k]) for k in range(P.shape[1])])
    out[0], out[-1] = P[0], P[-1]
    return out


def remesh_2d(mesh: MeshResult, nodes, moving, profiles, order=None) -> MeshResult:
    """New MeshResult for the shape *nodes* (geometry units) of *mesh*.

    *moving*: boundary numbers whose chains are re-spaced with *profiles* {(boundary, chain index): fractions}."""
    import gmsh
    nodes = np.asarray(nodes, float)
    order = int(order or mesh.order or 2)
    chains = {}                                            # (b, k) -> node-id chain
    for b, blocks in sorted(mesh.bnd.items()):
        for k, ch in enumerate(boundary_chains(blocks)):
            chains[(int(b), k)] = ch
    # boundaries around each domain (from the parent elements)
    dom_of_edge: dict = {}
    for d, blocks in mesh.bulk.items():
        for gt, conn in blocks:
            nc = 3 if int(gt) in (2, 9) else 4
            conn = np.asarray(conn)[:, :nc]
            for j in range(nc):
                e = np.sort(np.column_stack([conn[:, j], conn[:, (j + 1) % nc]]), axis=1)
                for a, bb in map(tuple, e):
                    dom_of_edge.setdefault((int(a), int(bb)), set()).add(int(d))
    chain_doms = {}
    for key, ch in chains.items():
        e = tuple(sorted(ch[:2]))
        chain_doms[key] = dom_of_edge.get(e, set())
    spacing = []
    for ch in chains.values():
        spacing += list(np.linalg.norm(np.diff(nodes[ch, :2], axis=0), axis=1))
    hmax = float(np.max(spacing)) * 4 if spacing else 1.0
    with session("remesh"):
        geo = gmsh.model.geo
        pt_of_node: dict = {}
        lc_of_node: dict = {}
        polylines = {}
        for key, ch in chains.items():
            P = nodes[ch, :2]
            if key[0] in moving and len(ch) > 2:
                prof = profiles.get(key)
                if prof is not None and len(prof) == len(ch):
                    P = respace(P, prof)
            polylines[key] = P
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            if np.any(seg <= 0):
                raise RemeshError(f"boundary {key[0]}: the moving boundary collapsed (zero-length segment)")
            for j, nid in enumerate((ch[0], ch[-1])):
                h = seg[0] if j == 0 else seg[-1]
                lc_of_node[nid] = min(lc_of_node.get(nid, np.inf), h)
        lines_of = {}
        for key, ch in chains.items():
            P = polylines[key]
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            tags = []
            for j in range(len(P)):
                if j in (0, len(P) - 1):
                    nid = ch[0] if j == 0 else ch[-1]
                    if nid not in pt_of_node:
                        pt_of_node[nid] = geo.addPoint(P[j, 0], P[j, 1], 0.0, float(lc_of_node[nid]))
                    tags.append(pt_of_node[nid])
                else:
                    h = 0.5 * (seg[j - 1] + seg[j])
                    tags.append(geo.addPoint(P[j, 0], P[j, 1], 0.0, float(h)))
            lines = [geo.addLine(tags[j], tags[j + 1]) for j in range(len(tags) - 1)]
            lines_of[key] = (lines, ch[0], ch[-1])
        surf_of = {}
        for d in sorted(mesh.bulk):
            mine = [key for key, ds in chain_doms.items() if d in ds]
            loops = _loops(mine, lines_of)
            if not loops:
                raise RemeshError(f"domain {d}: could not trace its boundary")
            areas = []
            loop_tags = []
            for seq in loops:
                curves = []
                for key, fwd in seq:
                    ls = lines_of[key][0]
                    curves += ls if fwd else [-t for t in reversed(ls)]
                loop_tags.append(geo.addCurveLoop(curves, reorient=True))
                pts = np.vstack([polylines[key] if fwd else polylines[key][::-1] for key, fwd in seq])
                areas.append(0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))))
            o = int(np.argmax(areas))
            surf_of[d] = geo.addPlaneSurface([loop_tags[o]] + [t for i, t in enumerate(loop_tags) if i != o])
        for lines, _a, _b in lines_of.values():
            for t in lines:
                geo.mesh.setTransfiniteCurve(t, 2)
        geo.synchronize()
        for d, s in surf_of.items():
            gmsh.model.addPhysicalGroup(2, [s], d)
        for b in sorted({key[0] for key in chains}):
            gmsh.model.addPhysicalGroup(1, [t for key, (ls, _a, _c) in lines_of.items() if key[0] == b for t in ls], b)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", hmax)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        try:
            gmsh.model.mesh.generate(2)
        except Exception as exc:
            raise RemeshError(f"remeshing failed: {exc}") from exc
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        ntags = ntags.astype(np.int64)
        idx = np.full(int(ntags.max()) + 1, -1, np.int64)
        idx[ntags] = np.arange(len(ntags))
        xyz = coords.reshape(-1, 3).copy()
        res = MeshResult(mesh.sdim, mesh.unit, mesh.scale, xyz, geometry=mesh.geometry, order=order)
        nb = 0
        for dim, groups, store in ((2, surf_of, res.bulk), (1, None, res.bnd)):
            for _dd, tag in gmsh.model.getPhysicalGroups(dim):
                blocks = []
                for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                    types, _e, conns = gmsh.model.mesh.getElements(dim, ent)
                    for t, c in zip(types, conns):
                        t = int(t)
                        if t not in GMSH_TO_ELMER:
                            continue
                        nper = gmsh.model.mesh.getElementProperties(t)[3]
                        blocks.append((t, idx[c.astype(np.int64)].reshape(-1, nper)))
                merged = {}
                for t, c in blocks:
                    merged.setdefault(t, []).append(c)
                store[int(tag)] = [(t, np.vstack(cs)) for t, cs in merged.items()]
                if dim == 2:
                    nb += sum(len(c) for _t, c in store[int(tag)])
        for num, nid in getattr(mesh, "point_nodes", {}).items():
            ptag = pt_of_node.get(int(nid))
            if ptag is not None:
                pn, _c, _p = gmsh.model.mesh.getNodes(0, ptag)
                if len(pn):
                    res.point_nodes[num] = int(idx[int(pn[0])])
        res.stats.update({"nodes": len(xyz), "elements": nb, "order": order,
                          "boundary_elements": sum(len(c) for bl in res.bnd.values() for _t, c in bl)})
    return res


def _loops(keys, lines_of):
    """Closed loops [(chain key, forward?)...] from chains given by their end nodes."""
    left = list(keys)
    loops = []
    while left:
        k0 = left.pop(0)
        _l, a, b = lines_of[k0]
        seq = [(k0, True)]
        start, cur = a, b
        guard = 0
        while cur != start:
            guard += 1
            if guard > 100000:
                raise RemeshError("boundary chains do not close")
            for i, k in enumerate(left):
                _l, s, e = lines_of[k]
                if s == cur:
                    seq.append((k, True))
                    cur = e
                    left.pop(i)
                    break
                if e == cur:
                    seq.append((k, False))
                    cur = s
                    left.pop(i)
                    break
            else:
                raise RemeshError("boundary chains do not close")
        loops.append(seq)
    return loops
