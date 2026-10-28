"""Boundary Layers mesh node: layers of the requested thickness, number and stretching, and what they buy.

A reaction-diffusion layer, D c'' = k c with c = 1 on the wall x = 0, has c = exp(-x/delta), delta = sqrt(D/k),
and the wall flux D/delta exactly.  With delta = 1 mm in a 1 m domain a mesh of 5 cm elements cannot see the layer;
twelve boundary layers on the wall resolve it.
"""
import math
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import GeometryError, build_geometry
from elmerstudio.core.meshing import boundary_layer_sizes, build_mesh
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.project import load, save
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner

W, H, D, K = 1.0, 0.2, 1e-6, 1.0
DELTA = math.sqrt(D / K)


def model(bl=True, order="1", first="manual"):
    m = B.new_model("2D", ["tds"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(W), h=str(H))
    geo = build_geometry(m, display=False)
    tds = m.component.child("physics.tds")
    tds.child("tds.tp").props["D"] = f"{D!r}[m^2/s]"
    B.add_feature(m, tds, "tds.reac", geo.select("domain"), k=f"{K!r}[1/s]")
    wall = geo.select("boundary", xmax=0)
    B.add_feature(m, tds, "tds.conc", wall, c0="1[mol/m^3]")
    mesh = m.component.child("mesh")
    mesh.props.update(sequence="user", order=order)
    m.create("mesh.size", mesh).props.update(mode="custom", hmax="0.05", hmin="0.01")
    m.create("mesh.ftri", mesh)
    node = None
    if bl:
        node = B.add_feature(m, mesh, "mesh.bl", wall, n=12, stretch="1.3", first=first, h1=f"{DELTA / 5!r}")
    return m, geo, wall, node


def test_layers_have_the_requested_thickness_number_and_stretching():
    m, geo, wall, node = model(order="1")
    mesh = build_mesh(m)
    info = mesh.stats["boundary_layers"][0]
    assert info["layers"] == 12 and info["first"] == pytest.approx(DELTA / 5)
    want = np.cumsum(boundary_layer_sizes(12, 1.3, DELTA / 5))
    # node x positions along the middle of the wall: the layer interfaces
    P = mesh.nodes * mesh.scale
    mid = np.abs(P[:, 1] - H / 2) < 0.02
    xs = np.unique(np.round(P[mid & (P[:, 0] < want[-1] * 1.0001), 0], 12))
    xs = xs[xs > 0]
    assert len(xs) == 12, xs
    assert np.allclose(xs, want, rtol=1e-6)
    assert "Quadrilateral 4" in mesh.stats["types"] and "Triangle 3" in mesh.stats["types"]
    # every element is local (gmsh joins layer nodes to far boundaries unless it knows where the layers end)
    for blocks in mesh.bulk.values():
        for _t, conn in blocks:
            span = np.ptp(P[conn][:, :, :2], axis=1).max(axis=1)
            assert span.max() < 2 * 0.05, span.max()
    # the automatic first layer: 1/20 of the maximum element size times the adjustment factor
    m2, *_r, n2 = model(order="1", first="auto")
    n2.props["adjust"] = "0.5"
    assert build_mesh(m2).stats["boundary_layers"][0]["first"] == pytest.approx(0.05 / 20 * 0.5)


def wall_flux(m, geo, wall, elmer):
    folders = StudyRunner(m, m.studies()[0], tempfile.mkdtemp(prefix="es_bl_"), elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    i = sol.frame_index("last")
    return abs(sol._integrate_values(sol.evaluate("tds.ntflux", i), "boundary", wall)) / H


def test_boundary_layers_resolve_a_thin_layer(elmer):
    exact = D / DELTA
    m, geo, wall, _n = model(order="2")
    assert wall_flux(m, geo, wall, elmer) == pytest.approx(exact, rel=0.01)
    # control: the same mesh without the layers misses the wall flux badly
    m0, geo0, wall0, _ = model(bl=False, order="2")
    assert abs(wall_flux(m0, geo0, wall0, elmer) / exact - 1) > 0.3


def test_boundary_layers_persist_and_are_refused_in_3d(tmp_path):
    m, geo, wall, node = model()
    p = tmp_path / "bl.esm"
    save(m, str(p), None)
    m2, _w = load(str(p))
    n2 = m2.by_tag(node.tag)
    assert n2.kind == "mesh.bl" and n2.get("n") == 12 and n2.get("stretch") == "1.3"
    assert n2.selection.entities == wall
    assert "mesh.bl" in m2.component.child("mesh").type.children(m2.component.child("mesh"))
    m3 = B.new_model("3D", ["ht"], "stationary")
    m3.create("geom.block", m3.component.child("geometry"), index=0)
    g3 = build_geometry(m3, display=False)
    mesh3 = m3.component.child("mesh")
    mesh3.props["sequence"] = "user"
    assert "mesh.bl" not in mesh3.type.children(mesh3)
    bl3 = m3.create("mesh.bl", mesh3)
    bl3.selection.entities = g3.select("boundary", zmax=0)
    with pytest.raises(GeometryError, match="2D"):
        build_mesh(m3)
