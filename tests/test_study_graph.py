"""Study and result graphs (M2): what each node uses and is used by; a study that starts from another study's
solution (Values of Dependent Variables > Solution), computed in order; and its controls: no initial solution, a
different mesh, a circle of studies."""
import os
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core import deps
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner, study_order


def strip(times="range(0,1e-3,2e-3)"):
    """A copper strip, 300 K left and 400 K right: Study 1 stationary, Study 2 time dependent (starting at 293 K
    unless it takes Study 1's solution)."""
    m = B.new_model("2D", ["ht"], "stationary")
    m.global_defs.child("params").props["table"] = [{"name": "L", "expr": "0.1[m]"}, {"name": "Tl", "expr": "300[K]"}]
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="L", h="L/10")
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0="Tl")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmin=0.1), T0="400[K]")
    st2 = B.add_study(m, "time")
    st2.children[0].props["times"] = times
    return m, m.studies()[0], st2


def test_uses_and_used_by():
    m, st1, st2 = strip()
    params = m.global_defs.child("params")
    rect = m.component.child("geometry").children[0]
    users = deps.used_by(m, params)
    assert rect in users and any(n.kind == "ht.temperature" for n in users)
    assert params in deps.uses(m, rect)
    ds = B.solution_dataset(m, st1)
    pg = m.create("pg2d", m.results)
    pg.props["data"] = ds.tag
    surf = m.create("plot.surface", pg)
    surf.props["data"] = "fromparent"
    assert deps.studies_of(m, surf) == [st1] and deps.studies_of(m, pg) == [st1]
    st2.children[0].props["initsol"] = st1.tag
    assert st1 in deps.uses(m, st2) and [s.tag for s in study_order(m)] == [st1.tag, st2.tag]


def test_a_study_starts_from_another_studys_solution(elmer):
    m, st1, st2 = strip()
    work = tempfile.mkdtemp(prefix="es_graph_")
    StudyRunner(m, st1, work, elmer).run()
    # control: physics-controlled initial values (293.15 K) are far from the steady profile after 1 ms
    StudyRunner(m, st2, work, elmer).run()
    s = Solution(os.path.join(work, st2.tag, "sol_0_0"))
    x = s.topo.nodes[:, 0] * s.scale
    lin = 300 + 1000 * x
    first = s.evaluate("T", 0)
    assert np.max(np.abs(first - lin)) > 20
    st2.children[0].props["initsol"] = st1.tag
    StudyRunner(m, st2, work, elmer).run()
    s = Solution(os.path.join(work, st2.tag, "sol_0_0"))
    assert np.max(np.abs(s.evaluate("T", 0) - lin)) < 0.05   # the steady profile, kept by the same BCs
    # the source study must be on the same mesh
    m.component.child("mesh").props["size"] = "2"                   # Extra fine
    with pytest.raises(RuntimeError, match="on another mesh"):
        StudyRunner(m, st2, work, elmer).run()


def test_controls(elmer):
    m, st1, st2 = strip()
    st2.children[0].props["initsol"] = st1.tag
    with pytest.raises(RuntimeError, match="compute Study 1 first"):
        StudyRunner(m, st2, tempfile.mkdtemp(prefix="es_graph_"), elmer).run()
    st1.children[0].props["initsol"] = st2.tag
    with pytest.raises(RuntimeError, match="in a circle"):
        study_order(m)
