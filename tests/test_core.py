"""Engine tests that need neither Elmer nor a display (run on every CI platform)."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core import project, units
from elmerstudio.core.examples import EXAMPLES
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.meshing import build_mesh
from elmerstudio.core.sif import build_sif, parse_list


# ------------------------------------------------------------------ units / expressions
@pytest.mark.parametrize("expr,si", [("10[mm]", 0.01), ("20[degC]", 293.15), ("1[kW]/2[m^2]", 500.0),
                                     ("2[bar]", 2e5), ("sqrt(16)*pi", 4 * math.pi), ("3^2", 9.0),
                                     ("1[MPa]", 1e6), ("30[deg]", math.pi / 6)])
def test_unit_expressions(expr, si):
    assert units.evaluate(expr) == pytest.approx(si)


def test_compound_units_and_parameters():
    assert units.unit_info("W/(m*K)")[0] == pytest.approx(1.0)
    assert units.unit_info("mm^2")[0] == pytest.approx(1e-6)
    vals, errs = units.evaluate_parameters([("b", "2*a"), ("a", "5[cm]")])   # forward reference
    assert not errs and vals["b"] == pytest.approx(0.1)


def test_matc_translation():
    r = units.to_matc("k0*(1+0.01*(T-293.15[K]))", {"k0": 2.0}, {"T": "Temperature"})
    assert r[0] == ["Temperature"] and "tx" in r[1]
    assert units.to_matc("2[mm]", {}, {}) == pytest.approx(0.002)
    with pytest.raises(units.ExprError):
        units.to_matc("q*2", {}, {})


def test_parse_list():
    assert parse_list("range(0,0.25,1)") == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
    assert parse_list("1 2,3") == [1, 2, 3]


# ------------------------------------------------------------------ geometry / mesh
def _block_with_hole():
    m = B.new_model("3D", ["ht"], "stationary")
    g = m.component.child("geometry")
    blk = m.create("geom.block", g, index=0, w="0.1", d="0.04", h="0.02")
    cyl = m.create("geom.cylinder", g, index=1, r="0.008", h="0.02", x="0.05", y="0.02")
    m.create("geom.difference", g, index=2, input=[blk.tag], tools=[cyl.tag])
    return m


def test_geometry_numbering_and_selection():
    geo = build_geometry(_block_with_hole(), display=False)
    assert geo.count("domain") == 1 and geo.count("boundary") == 7
    assert len(geo.select("boundary", xmax=0)) == 1          # x = 0 face (bbox padding removed)
    assert geo.adj_up["boundary"][geo.select("boundary", xmax=0)[0]] == [1]


def test_mesh_and_elmer_writer(tmp_path):
    m = _block_with_hole()
    mesh = build_mesh(m)
    assert mesh.order == 2 and mesh.stats["elements"] > 50 and 0 < mesh.stats["min_quality"] <= 1
    em = mesh.to_elmer()
    em.write(str(tmp_path))
    head = (tmp_path / "mesh.header").read_text().split()
    assert int(head[0]) == len(mesh.nodes) and int(head[1]) == mesh.stats["elements"]
    bnd = np.loadtxt(tmp_path / "mesh.boundary", dtype=int)
    assert (bnd[:, 2] > 0).all(), "every boundary element must have a parent"


def test_sif_generation_rules():
    m = _block_with_hole()
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    ht = m.component.child("physics.ht")
    B.add_feature(m, ht, "ht.source", [1], Q0="1e5[W/m^3]")
    B.add_feature(m, ht, "ht.temperature", geo.select("boundary", xmax=0), T0="20[degC]")
    st = m.studies()[0]
    sif = build_sif(m, st, B.study_steps(st)[0], geo, 1000).text
    assert "Volumetric Heat Source = Real 100000.0" in sif
    assert "Temperature = Real 293.15" in sif
    assert 'Solver Input File = File "case.sif"' in sif
    assert "Heat Conductivity = Real 400.0" in sif


def test_default_boundary_features_skip_interior():
    """COMSOL semantics: default features (e.g. Magnetic Insulation) act on exterior boundaries only."""
    m = B.new_model("2D", ["mf"], "stationary")
    g = m.component.child("geometry")
    a = m.create("geom.rectangle", g, index=0, w="2", h="1")
    b = m.create("geom.rectangle", g, index=1, w="0.5", h="0.5", x="0.5", y="0.25")
    m.create("geom.union", g, index=2, input=[a.tag, b.tag])
    B.add_material(m, "Air", all_domains=True)
    geo = build_geometry(m, display=False)
    interior = [n for n in geo.numbers["boundary"] if len(geo.adj_up["boundary"][n]) > 1]
    st = m.studies()[0]
    sif = build_sif(m, st, B.study_steps(st)[0], geo, 100).text
    targets = [int(x) for line in sif.splitlines() if "Target Boundaries" in line for x in line.split("=")[1].split()]
    assert interior and not set(interior) & set(targets)
    assert set(targets) == set(geo.exterior_boundaries())


def test_missing_material_is_reported():
    m = _block_with_hole()
    geo = build_geometry(m, display=False)
    st = m.studies()[0]
    with pytest.raises(Exception, match="material"):
        build_sif(m, st, B.study_steps(st)[0], geo, 100)


# ------------------------------------------------------------------ project files, examples
def test_project_roundtrip(tmp_path):
    m = _block_with_hole()
    p = os.path.join(tmp_path, "m.esm")
    project.save(m, p, None)
    m2, _ = project.load(p)
    assert [n.kind for n in m.root.walk()] == [n.kind for n in m2.root.walk()]


@pytest.mark.parametrize("key", list(EXAMPLES))
def test_examples_build_and_generate_sif(key):
    m = EXAMPLES[key][1]()
    geo = build_geometry(m, display=False)
    st = m.studies()[0]
    sif = build_sif(m, st, B.study_steps(st)[0], geo, 1000).text
    assert "Solver 1" in sif and "Boundary Condition 1" in sif
