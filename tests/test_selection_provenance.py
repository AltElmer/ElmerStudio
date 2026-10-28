"""Selections that survive geometry edits (M2): typed entity numbers follow their entities when a parameter change
or a new object renumbers the geometry (associative selections), an entity that disappears is reported rather than
silently replaced, a geometry object's own selection ("Resulting objects") follows it through the union, and old
files without signatures pick them up on their first build."""
import os
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core import project
from elmerstudio.core.geometry import build_geometry, named_selections


def _strip():
    m = B.new_model("2D", ["ht"], "stationary")
    m.global_defs.child("params").props["table"] = [{"name": "W", "expr": "1"}]
    g = m.component.child("geometry")
    m.create("geom.rectangle", g, index=0, w="W", h="0.4")
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    left, right = geo.select("boundary", xmax=0), geo.select("boundary", xmin=1)
    ph = m.component.child("physics.ht")
    tl = B.add_feature(m, ph, "ht.temperature", left, T0="300[K]")
    tr = B.add_feature(m, ph, "ht.temperature", right, T0="400[K]")
    build_geometry(m, display=False)                      # records what the selected entities are
    return m, tl, tr


def _edges_at(m, sel_ents):
    geo = build_geometry(m, display=False)
    return [tuple(round(v, 9) for v in geo.bbox_of["boundary"][e]) for e in sel_ents], geo


def test_numbers_follow_their_entities_through_a_renumbering():
    m, tl, tr = _strip()
    before_r = list(tr.selection.entities)
    # widen the strip and put a disc on its left: the edges are numbered differently now
    m.global_defs.child("params").props["table"] = [{"name": "W", "expr": "2"}]
    m.create("geom.circle", m.component.child("geometry"), index=1, r="0.1", x="-0.5", y="0.2")
    geo = build_geometry(m, display=False)
    left_now, right_now = geo.select("boundary", xmax=0, xmin=0), geo.select("boundary", xmin=2)
    assert before_r != right_now                           # the control: the old number is another edge now
    assert tl.selection.entities == left_now and tr.selection.entities == right_now
    assert not tl.meta.get("lost") and not tr.meta.get("lost")
    assert any("renumbered" in msg for msg in geo.messages), geo.messages


def test_a_removed_entity_is_reported_not_replaced():
    m = B.new_model("3D", ["ht"], "stationary")
    g = m.component.child("geometry")
    m.create("geom.block", g, index=0, w="1", d="1", h="1")
    cyl = m.create("geom.cylinder", g, index=1, r="0.2", h="1", x="3")
    geo = build_geometry(m, display=False)
    mantle = [b for b in geo.numbers["boundary"] if geo.bbox_of["boundary"][b][0] > 2.5
              and geo.bbox_of["boundary"][b][5] - geo.bbox_of["boundary"][b][2] > 0.9]
    f = B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", mantle, T0="1")
    build_geometry(m, display=False)
    g.remove(cyl)                                           # the face it selected is gone
    geo = build_geometry(m, display=False)
    assert f.selection.entities == [] and f.meta.get("lost") == mantle
    assert any("no longer exist" in msg for msg in geo.messages)


def test_object_selections_follow_the_object():
    m = B.new_model("3D", ["ht"], "stationary")
    g = m.component.child("geometry")
    m.create("geom.block", g, index=0, w="1", d="1", h="1")
    m.create("geom.block", g, index=1, w="1", d="1", h="1", x="1")
    with m.scoped(m.component):
        tags = [t for t, _lbl in named_selections(m, "domain")]
    assert tags == ["blk1:domain", "blk2:domain"]
    mat = B.add_material(m, "Aluminum")
    mat.selection.named, mat.selection.all = "blk2:domain", False
    geo = build_geometry(m, display=False)
    assert mat.selection.entities == geo.object_domains["blk2"] and len(mat.selection.entities) == 1
    # blk1 grows through blk2: the union splits blk2, and its selection keeps all of blk2's pieces
    g.children[0].props["w"] = "1.5"
    geo = build_geometry(m, display=False)
    pieces = geo.object_domains["blk2"]
    assert mat.selection.entities == pieces and len(pieces) == 2
    bnd = geo.named["blk2:boundary"][1]
    assert set(bnd) == set(geo.boundaries_of_domains(pieces))


def test_signatures_are_saved_and_old_files_get_them(tmp_path=None):
    m, tl, tr = _strip()
    assert tr.selection.sig
    path = os.path.join(tempfile.mkdtemp(prefix="es_sel_"), "s.esm")
    project.save(m, path, None)
    m2, _w = project.load(path)
    tr2 = m2.by_tag(tr.tag)
    assert tr2.selection.sig == tr.selection.sig
    for n in m2.root.walk():                               # an older file: no signatures at all
        if n.selection is not None:
            n.selection.sig = {}
    build_geometry(m2, display=False)
    assert m2.by_tag(tr.tag).selection.sig                  # the first build records them


def test_the_physics_still_applies_where_it_did(elmer):
    """Solve after the renumbering: 300 K on the left edge and 400 K on the right one, as before the edit."""
    from test_pipeline import read, solve
    m, tl, tr = _strip()
    m.global_defs.child("params").props["table"] = [{"name": "W", "expr": "2"}]
    m.create("geom.circle", m.component.child("geometry"), index=1, r="0.1", x="-0.5", y="0.2")
    build_geometry(m, display=False)
    _mesh, _sif, _rs, files, _case = solve(m)
    g = read(files[-1])
    T, P = np.asarray(g.point_data["temperature"]), np.asarray(g.points)
    on_strip = P[:, 0] > -0.05
    assert T[on_strip & (P[:, 0] < 1e-9)].mean() == pytest.approx(300, abs=1e-6)
    assert T[on_strip & (P[:, 0] > 2 - 1e-9)].mean() == pytest.approx(400, abs=1e-6)
