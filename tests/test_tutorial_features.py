"""Features exercised by the COMSOL introduction tutorials (no Elmer needed except where marked)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.sif import build_sif


def _wp_model(**wp):
    m = B.new_model("3D", ["ht"], "stationary")
    g = m.component.child("geometry")
    blk = m.create("geom.block", g, index=0, w="0.1", d="0.05", h="0.02")
    plane = m.create("geom.workplane", g, index=1, **wp)
    m.create("geom.circle", plane, r="0.005")
    m.create("geom.extrude", g, index=2, input=[plane.tag], dist="-0.04")
    return m, blk


def test_face_parallel_workplane_outward_normal_and_overlap_domains():
    """Face-parallel plane on the top face (z=0.02): origin = face centre, normal outward (+z);
    Extrude -0.04 goes down through the 0.02-thick block and protrudes 0.02 below -> 3 domains (COMSOL rule:
    overlaps become separate domains)."""
    # find the top face number of the block alone (as a user would by clicking it)
    m2 = B.new_model("3D", ["ht"], "stationary")
    m2.create("geom.block", m2.component.child("geometry"), index=0, w="0.1", d="0.05", h="0.02")
    top = build_geometry(m2, display=False).select("boundary", zmin=0.02)[0]
    m, _ = _wp_model(ptype="face", face=top)
    geo = build_geometry(m, display=False)
    assert geo.count("domain") == 3
    zmin = min(geo.bbox_of["domain"][d][2] for d in geo.numbers["domain"])
    assert zmin == pytest.approx(-0.02, abs=1e-5)         # OCC boxes of curved solids are loose (~5 um)
    cyl = [d for d in geo.numbers["domain"] if geo.bbox_of["domain"][d][3] - geo.bbox_of["domain"][d][0] < 0.011]
    xs = [0.5 * (geo.bbox_of["domain"][d][0] + geo.bbox_of["domain"][d][3]) for d in cyl]
    assert cyl and all(abs(x - 0.05) < 1e-5 for x in xs), xs   # circle centred on the face centre


@pytest.mark.parametrize("plane,normal_axis", [("xy", 2), ("xz", 1), ("yz", 0), ("zx", 1)])
def test_quick_workplanes(plane, normal_axis):
    m = B.new_model("3D", ["ht"], "stationary")
    g = m.component.child("geometry")
    wp = m.create("geom.workplane", g, index=0, ptype="quick", plane=plane)
    m.create("geom.rectangle", wp, w="0.03", h="0.02")
    m.create("geom.extrude", g, index=1, input=[wp.tag], dist="0.01")
    geo = build_geometry(m, display=False)
    bb = np.array(geo.bbox_of["domain"][1]).reshape(2, 3)
    assert np.ptp(bb, axis=0)[normal_axis] == pytest.approx(0.01)


def test_fillet2d_area():
    """Square 1x1 with one corner filleted r=0.2: area = 1 - (1 - pi/4) r^2."""
    m = B.new_model("2D", ["ht"], "stationary")
    g = m.component.child("geometry")
    sq = m.create("geom.rectangle", g, index=0, w="1", h="1")
    m.create("geom.fillet2d", g, index=1, points="4", r="0.2")    # point 4 = (1, 1) in position order
    geo = build_geometry(m, display=False)
    assert geo.measure["domain"][1] == pytest.approx(1 - (1 - math.pi / 4) * 0.04, rel=1e-9)
    m.create("geom.chamfer2d", g, index=2, points="1", d="0.1")     # (0, 0)
    geo = build_geometry(m, display=False)
    assert geo.measure["domain"][1] == pytest.approx(1 - (1 - math.pi / 4) * 0.04 - 0.005, rel=1e-9)


def test_total_force_load_divides_by_area():
    m = B.new_model("3D", ["solid"], "stationary")
    m.create("geom.block", m.component.child("geometry"), index=0, w="0.1", d="0.02", h="0.01")
    B.add_material(m, "Structural steel", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")
    B.add_feature(m, ph, "solid.fixed", geo.select("boundary", xmax=0))
    B.add_feature(m, ph, "solid.bload", geo.select("boundary", xmin=0.1), type="total", Ftz="-150[N]")
    st = m.studies()[0]
    sif = build_sif(m, st, B.study_steps(st)[0], geo, 100).text
    assert f"Force 3 = Real {-150 / (0.02 * 0.01):.12g}" in sif, sif[sif.find("Boundary Condition"):]


def test_coupling_operator_definition():
    from elmerstudio.core.nodes import coupling_operators
    m = B.new_model("3D", ["ht"], "stationary")
    defs = m.component.child("definitions")
    op = m.create("cpl.average", defs, opname="aveop1")
    op.selection.all = False
    op.selection.entities = [1]
    assert coupling_operators(m) == {"aveop1": ("average", "domain", [1])}
