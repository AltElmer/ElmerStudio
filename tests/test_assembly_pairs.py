"""Assemblies and identity pairs (M2): Form an Assembly keeps touching objects separate (their meshes need not
match) and makes an identity pair where they touch; the temperature is continuous across it (Elmer mortar
projection), so two blocks conduct like one. Controls: the pair disabled (the blocks are insulated from each
other), the objects in the other order (the same answer), and a gap between the blocks (no pair at all)."""
import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry


def two_blocks(order=("a", "b"), gap=0.0, sdim="3D"):
    m = B.new_model(sdim, ["ht"], "stationary")
    g = m.component.child("geometry")
    shapes = {"a": dict(x="0"), "b": dict(x=str(1 + gap))}
    for i, key in enumerate(order):
        if sdim == "3D":
            m.create("geom.block", g, index=i, w="1", d="0.4", h="0.4", **shapes[key])
        else:
            m.create("geom.rectangle", g, index=i, w="1", h="0.4", **shapes[key])
    g.child("geom.finalize").props["action"] = "none"           # Form an Assembly
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0="300[K]")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmin=2 + gap), T0="400[K]")
    return m, geo


def test_touching_objects_make_one_identity_pair():
    m, geo = two_blocks()
    assert len(geo.pairs) == 1
    key, src, dst = geo.pairs[0]
    assert key == "blk1|blk2" and len(src) == 1 and len(dst) == 1 and src != dst
    for b in src + dst:                                        # both sides lie on the plane x = 1
        bb = geo.bbox_of["boundary"][b]
        assert bb[0] == pytest.approx(1) and bb[3] == pytest.approx(1)
    pairs = [n for n in m.component.child("definitions").children if n.kind == "pair.identity"]
    assert len(pairs) == 1 and pairs[0].props["objects"] == "blk1 and blk2"
    _m, far = two_blocks(gap=0.1)
    assert far.pairs == []                                     # control: a gap is not a contact
    assert not any(n.kind == "pair.identity" for n in _m.component.child("definitions").children)


def _solve_T(m):
    from test_pipeline import read, solve
    _mesh, sif, _rs, files, _case = solve(m)
    g = read(files[-1])
    return np.asarray(g.points), np.asarray(g.point_data["temperature"]), sif


@pytest.mark.parametrize("sdim", ["3D", "2D"])
def test_temperature_is_continuous_across_the_pair(elmer, sdim):
    m, _geo = two_blocks(sdim=sdim)
    P, T, sif = _solve_T(m)
    assert "Mortar BC = Integer" in sif.text and "Apply Mortar BCs = Logical True" in sif.text
    exact = 300 + 50 * P[:, 0]                                 # one bar from 0 to 2: linear
    assert np.max(np.abs(T - exact)) < 1e-3 * 100, np.max(np.abs(T - exact))
    at = np.abs(P[:, 0] - 1) < 1e-9                            # both sides of the interface (duplicate nodes)
    assert np.allclose(T[at], 350, atol=0.1)
    # control: the same assembly with the pair disabled leaves the two blocks unconnected
    pair = next(n for n in m.component.child("definitions").children if n.kind == "pair.identity")
    pair.enabled = False
    P, T, sif = _solve_T(m)
    assert "Mortar BC" not in sif.text
    assert np.allclose(T[P[:, 0] < 1 - 1e-9], 300, atol=1e-6) and np.allclose(T[P[:, 0] > 1 + 1e-9], 400, atol=1e-6)


def test_the_pair_does_not_depend_on_the_objects_order(elmer):
    m1, _ = two_blocks(("a", "b"))
    m2, _ = two_blocks(("b", "a"))
    P1, T1, _ = _solve_T(m1)
    P2, T2, _ = _solve_T(m2)
    for P, T in ((P1, T1), (P2, T2)):
        assert np.max(np.abs(T - (300 + 50 * P[:, 0]))) < 0.1


def test_a_shaft_in_a_hole_pairs_only_the_curved_faces():
    """The contact is the shaft's mantle and the hole's wall; the shaft's end disks lie in the plate's planes but in
    its hole, so they are not paired (the control)."""
    m = B.new_model("3D", ["ht"], "stationary")
    g = m.component.child("geometry")
    blk = m.create("geom.block", g, index=0, w="2", d="2", h="1", x="-1", y="-1")
    hole = m.create("geom.cylinder", g, index=1, r="0.3", h="1")
    dif = m.create("geom.difference", g, index=2)
    dif.props["input"], dif.props["tools"] = [blk.tag], [hole.tag]
    m.create("geom.cylinder", g, index=3, r="0.3", h="1")
    g.child("geom.finalize").props["action"] = "none"
    geo = build_geometry(m, display=False)
    assert len(geo.pairs) == 1
    _key, src, dst = geo.pairs[0]
    assert len(src) == 1 and len(dst) == 1
    for b in src + dst:
        bb = geo.bbox_of["boundary"][b]
        assert bb[2] == pytest.approx(0) and bb[5] == pytest.approx(1) and bb[3] == pytest.approx(0.3)
