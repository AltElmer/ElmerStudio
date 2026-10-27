"""The ElmerGrid mesher: closed-form solutions, agreement with Gmsh, the numbering contract and its refusals.

Run:  python -m pytest tests/test_elmergrid.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core import elmergrid  # noqa: E402
from elmerstudio.core.elmer import ElmerInstall, find_elmer, result_files, run_case_blocking, write_case  # noqa: E402
from elmerstudio.core.elmer_mesh import GMSH_TO_ELMER  # noqa: E402
from elmerstudio.core.geometry import GeometryError, build_geometry  # noqa: E402
from elmerstudio.core.meshing import build_mesh  # noqa: E402
from elmerstudio.core.results import Topology  # noqa: E402
from elmerstudio.core.sif import build_sif  # noqa: E402

INST = find_elmer()
needs_grid = pytest.mark.skipif(INST is None or not INST.grid, reason="ElmerGrid not installed")


def use_elmergrid(m, **props):
    mn = m.component.child("mesh")
    mn.props["mesher"] = "elmergrid"
    mn.props.update(props)
    return mn


def solve(model, mesh=None):
    st = model.studies()[0]
    step = B.study_steps(st)[0]
    mesh = mesh or build_mesh(model)
    sif = build_sif(model, st, step, mesh.geometry, len(mesh.nodes))
    case = tempfile.mkdtemp(prefix="es_eg_")
    write_case(case, mesh.to_elmer(), sif.text)
    rs = run_case_blocking(case, INST)
    files = result_files(case)
    assert files, f"no results; errors={rs.errors[-5:]} case={case}"
    import pyvista as pv
    return mesh, pv.read(files[-1]), case


def heat_block():
    m = B.new_model("3D", ["ht"], "stationary")
    m.create("geom.block", m.component.child("geometry"), index=0, w="1", d="0.2", h="0.1")
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0="300[K]")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmin=1), T0="400[K]")
    return m


def series_2d():
    """Copper (0<x<0.6) then steel (0.6<x<1), T = 300 K and 400 K at the ends: a piecewise linear temperature."""
    m = B.new_model("2D", ["ht"], "stationary")
    g = m.component.child("geometry")
    m.create("geom.rectangle", g, index=0, w="0.6", h="0.2")
    m.create("geom.rectangle", g, index=1, w="0.4", h="0.2", x="0.6")
    geo = build_geometry(m, display=False)
    left = geo.select("domain", xmax=0.6)
    right = [d for d in geo.numbers["domain"] if d not in left]
    B.add_material(m, "Copper", domains=left)
    B.add_material(m, "Structural steel", domains=right)
    ph = m.component.child("physics.ht")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0="300[K]")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmin=1), T0="400[K]")
    return m


def at_vertices(mesh, g, name):
    """A result at the geometry's points (nodes of every mesh of it), by position in the result file."""
    out = {}
    for num, xyz in elmergrid._point_xyz(mesh.geometry).items():
        out[num] = float(np.asarray(g.point_data[name])[np.argmin(np.linalg.norm(g.points - xyz * mesh.scale,
                                                                                  axis=1))])
    return out


def series_exact(x, k1, k2):
    # equal flux q through both: T(0.6) = Ti with k1 (Ti - 300)/0.6 = k2 (400 - Ti)/0.4
    ti = (k1 * 300 / 0.6 + k2 * 400 / 0.4) / (k1 / 0.6 + k2 / 0.4)
    return np.where(x <= 0.6, 300 + (ti - 300) * x / 0.6, ti + (400 - ti) * (x - 0.6) / 0.4)


@needs_grid
def test_heat_block_elmergrid_matches_exact_and_gmsh():
    m = heat_block()
    use_elmergrid(m)
    mesh, g, case = solve(m)
    assert mesh.stats["mesher"] == "ElmerGrid" and mesh.stats["types"] == {"Hexahedron 20": 40}
    T = np.asarray(g.point_data["temperature"])
    err = np.max(np.abs(T - (300 + 100 * g.points[:, 0])))
    m.component.child("mesh").props["mesher"] = "gmsh"
    gm, gg, _c = solve(m)
    eg = np.max(np.abs(np.asarray(gg.point_data["temperature"]) - (300 + 100 * gg.points[:, 0])))
    a, b = at_vertices(mesh, g, "temperature"), at_vertices(gm, gg, "temperature")
    diff = max(abs(a[n] - b[n]) for n in a)
    print(f"ElmerGrid heat block: max |T - Texact| = {err:.2e} K (Gmsh {eg:.2e} K), max |T_elmergrid - T_gmsh| "
          f"at the {len(a)} vertices = {diff:.2e} K ({len(mesh.nodes)} vs {len(gm.nodes)} nodes)  {case}")
    assert err < 1e-6 and eg < 1e-6 and diff < 1e-6 and len(a) == 8


@needs_grid
@pytest.mark.parametrize("shape,order", [("quad", "1"), ("quad", "2"), ("tri", "1"), ("tri", "2")])
def test_two_materials_2d_elmergrid_vs_gmsh(shape, order):
    m = series_2d()
    k1 = 400.0
    k2 = 44.5
    use_elmergrid(m, eg_shape=shape, order=order)
    mesh, g, case = solve(m)
    T = np.asarray(g.point_data["temperature"])
    err = np.max(np.abs(T - series_exact(g.points[:, 0], k1, k2)))
    m.component.child("mesh").props["mesher"] = "gmsh"
    gm, gg, _c = solve(m)
    eg = np.max(np.abs(np.asarray(gg.point_data["temperature"]) - series_exact(gg.points[:, 0], k1, k2)))
    a, b = at_vertices(mesh, g, "temperature"), at_vertices(gm, gg, "temperature")
    diff = max(abs(a[n] - b[n]) for n in a)
    print(f"two materials {shape} order {order}: ElmerGrid {mesh.stats['types']} err {err:.2e} K, "
          f"Gmsh err {eg:.2e} K, max difference at the {len(a)} vertices {diff:.2e} K  {case}")
    assert err < 1e-6 and eg < 1e-6 and diff < 1e-6


def check_numbering(mesh):
    """Bodies are the domains and boundary ids the boundaries: every element lies inside the box of the entity it
    is numbered with, and the elements of each entity add up to its length, area or volume."""
    geo = mesh.geometry
    tol = 1e-6 * geo.diag
    assert sorted(mesh.bulk) == geo.numbers["domain"]
    assert sorted(mesh.bnd) == [b for b in geo.numbers["boundary"] if geo.adj_up["boundary"].get(b)]
    topo = Topology(mesh_result=mesh)
    for src, level, grid in ((mesh.bulk, "domain", topo.bulk_grid()), (mesh.bnd, "boundary", topo.boundary_grid())):
        ids = np.asarray(grid.cell_data["dom" if level == "domain" else "bnd"])
        sizes = np.asarray(grid.compute_cell_sizes(length=True, area=True, volume=True).cell_data[
            {1: "Length", 2: "Area", 3: "Volume"}[geo.sdim - (0 if level == "domain" else 1)]])
        assert (sizes > 0).all(), f"inverted {level} elements"
        for num, blocks in src.items():
            bb = np.array(geo.bbox_of[level][num])
            for _gt, conn in blocks:
                P = mesh.nodes[conn]
                assert (P >= bb[:3] - tol).all() and (P <= bb[3:] + tol).all(), f"{level} {num} element outside"
            got = sizes[ids == num].sum() * geo.scale ** (geo.sdim - (0 if level == "domain" else 1))
            assert np.isclose(got, geo.measure[level][num], rtol=1e-9), (level, num, got, geo.measure[level][num])


@needs_grid
def test_boundary_numbers_match_geometry_2d():
    m = B.new_model("2D", ["ht"], "stationary")
    g = m.component.child("geometry")
    m.create("geom.rectangle", g, index=0, w="1", h="0.5")
    m.create("geom.rectangle", g, index=1, w="0.5", h="0.25", x="1")
    m.create("geom.rectangle", g, index=2, w="0.25", h="0.25", x="0.25", y="0.5")
    use_elmergrid(m, order="2")
    mesh = build_mesh(m)
    check_numbering(mesh)
    assert len(mesh.bulk) == 3 and len(mesh.point_nodes) == len(mesh.geometry.numbers["point"])
    print("2D boundaries:", {b: sum(len(c) for _t, c in bl) for b, bl in sorted(mesh.bnd.items())})


@needs_grid
def test_boundary_numbers_match_geometry_3d_layers():
    """An L of two blocks with a third block on one leg: extruded layers with a missing cell and an interface."""
    m = B.new_model("3D", ["ht"], "stationary")
    g = m.component.child("geometry")
    m.create("geom.block", g, index=0, w="1", d="0.4", h="0.2")
    m.create("geom.block", g, index=1, w="0.4", d="0.6", h="0.2", y="0.4")
    m.create("geom.block", g, index=2, w="0.4", d="0.4", h="0.3", x="0.6", z="0.2")
    for order in ("1", "2"):
        use_elmergrid(m, order=order)
        mesh = build_mesh(m)
        check_numbering(mesh)
        em = mesh.to_elmer()
        parents = em.compute_parents()
        assert all((p[:, 0] > 0).all() for p in parents), "boundary element without a parent"
        print(f"3D order {order}: {mesh.stats['types']}, {len(mesh.bnd)} boundaries, "
              f"{mesh.stats['boundary_elements']} boundary elements, subcells {mesh.stats['subcells']}")
    # 20-node hexahedra: every edge node is the midpoint of its corners in gmsh order and in Elmer order
    gm_edges = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)]
    el_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 4)]
    perm = GMSH_TO_ELMER[17][1]
    for blocks in mesh.bulk.values():
        for gt, conn in blocks:
            assert gt == 17
            P = mesh.nodes[conn]
            Pe = mesh.nodes[conn[:, perm]]
            for k, (a, b) in enumerate(gm_edges):
                assert np.allclose(P[:, 8 + k], 0.5 * (P[:, a] + P[:, b]), atol=1e-12)
            for k, (a, b) in enumerate(el_edges):
                assert np.allclose(Pe[:, 8 + k], 0.5 * (Pe[:, a] + Pe[:, b]), atol=1e-12)


@needs_grid
def test_three_layer_block_solves():
    """The 3D layered L above conducts heat from its x=0 face to the top of the third block: Elmer runs on the
    ElmerGrid mesh and the temperature stays between the two fixed values (maximum principle)."""
    m = B.new_model("3D", ["ht"], "stationary")
    g = m.component.child("geometry")
    m.create("geom.block", g, index=0, w="1", d="0.4", h="0.2")
    m.create("geom.block", g, index=1, w="0.4", d="0.6", h="0.2", y="0.4")
    m.create("geom.block", g, index=2, w="0.4", d="0.4", h="0.3", x="0.6", z="0.2")
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0="300[K]")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", zmin=0.5), T0="400[K]")
    use_elmergrid(m)
    mesh, g, case = solve(m)
    T = np.asarray(g.point_data["temperature"])
    print(f"3-block L: T in [{T.min():.6f}, {T.max():.6f}] K  {case}")
    assert T.min() > 300 - 1e-6 and T.max() < 400 + 1e-6 and T.max() - T.min() > 99


def test_refuses_curved_geometry():
    m = B.new_model("2D", ["ht"], "stationary")
    m.create("geom.circle", m.component.child("geometry"), index=0, r="1")
    use_elmergrid(m)
    msgs = elmergrid.problems(build_geometry(m, display=False))
    assert msgs and "not straight and parallel to the x or y axis" in msgs[0], msgs
    with pytest.raises(GeometryError, match="ElmerGrid meshes only layouts of axis-aligned rectangles"):
        build_mesh(m)


def test_refuses_rotated_block_and_tetrahedra():
    m = B.new_model("3D", ["ht"], "stationary")
    m.create("geom.block", m.component.child("geometry"), index=0, rot="30")
    use_elmergrid(m)
    with pytest.raises(GeometryError, match="not plane and normal to the x, y or z axis"):
        build_mesh(m)
    m = heat_block()
    mn = use_elmergrid(m, sequence="user")
    m.create("mesh.ftet", mn)
    with pytest.raises(GeometryError, match="not tetrahedra"):
        build_mesh(m)


def test_refuses_without_elmergrid():
    m = heat_block()
    use_elmergrid(m)
    inst = ElmerInstall("C:/nowhere", "C:/nowhere/bin", "C:/nowhere/bin/ElmerSolver", grid=None)
    with pytest.raises(GeometryError, match="ElmerGrid was not found next to ElmerSolver"):
        build_mesh(m, inst=inst)


def test_supported_geometry_has_no_problems():
    m = heat_block()
    assert elmergrid.problems(build_geometry(m, display=False)) == []
