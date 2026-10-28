"""Several components (M2): two components of different dimensions with independent physics, solved by one study,
each with its own geometry, mesh, solution and namespace (comp1.T); a one-way coupling through a coupling operator
(comp2 uses comp1.aveop1(T)); save and reopen; and the controls: a circular coupling, a field of another component
in an equation, another component's name in a result expression."""
import os
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core import project
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner, component_dir, component_order, cross_references


def two_components():
    """comp1 (2D): a 1 x 0.25 strip, T = 300 K at x = 0 and 400 K at x = 1 (average 350 K).
    comp2 (3D): a block, T = comp1.aveop1(T) on one end and 300 K on the other (in mm)."""
    m = B.new_model("2D", ["ht"], "stationary")
    c1 = m.component
    m.create("geom.rectangle", c1.child("geometry"), index=0, w="1", h="0.25")
    B.add_material(m, "Copper", all_domains=True)
    with m.scoped(c1):
        geo1 = build_geometry(m, display=False)
    ht1 = c1.child("physics.ht")
    B.add_feature(m, ht1, "ht.temperature", geo1.select("boundary", xmax=0), T0="300[K]")
    B.add_feature(m, ht1, "ht.temperature", geo1.select("boundary", xmin=1), T0="400[K]")
    ave = m.create("cpl.average", c1.child("definitions"))
    ave.props["opname"] = "aveop1"
    ave.selection.all = True

    c2 = B.add_component(m, "3D")
    with m.scoped(c2):
        g2 = c2.child("geometry")
        g2.props["unit"] = "mm"
        m.create("geom.block", g2, index=0, w="10", d="2", h="2")
        ht2 = B.add_physics(m, "ht")
        B.add_material(m, "Aluminum", all_domains=True)
        geo2 = build_geometry(m, display=False)
        B.add_feature(m, ht2, "ht.temperature", geo2.select("boundary", xmax=0), T0="comp1.aveop1(T)")
        B.add_feature(m, ht2, "ht.temperature", geo2.select("boundary", xmin=10), T0="300[K]")
    for step in m.studies()[0].children:
        if "physics" in step.props:
            step.props["physics"] = {ph.tag: True for ph in B.all_physics(m)}
    return m, c1, c2, ht2


def test_tags_and_namespaces_are_unique():
    m, c1, c2, ht2 = two_components()
    tags = [n.tag for n in m.root.walk()]
    assert len(tags) == len(set(tags)), sorted(t for t in tags if tags.count(t) > 1)
    assert ht2.tag == "ht2" and ht2.label.endswith(" 2")
    assert [c.tag for c in component_order(m)] == ["comp1", "comp2"]
    assert cross_references(m, c2) == [("comp1.aveop1(T)", "comp1", "aveop1", "T")]


def test_two_components_solve_with_a_one_way_coupling(elmer):
    m, c1, c2, _ht2 = two_components()
    work = tempfile.mkdtemp(prefix="es_mc_")
    st = m.studies()[0]
    StudyRunner(m, st, work, elmer).run()
    sd = os.path.join(work, st.tag)
    f1, f2 = os.path.join(component_dir(m, sd, "comp1"), "sol_0_0"), os.path.join(component_dir(m, sd, "comp2"), "sol_0_0")
    s1, s2 = Solution(f1), Solution(f2)
    assert s1.sdim == 2 and s2.sdim == 3 and s1.meta["comp"] == "comp1" and s2.meta["comp"] == "comp2"
    T1, T2 = s1.evaluate("T"), s2.evaluate("T")
    assert T1.min() == pytest.approx(300, abs=1e-6) and T1.max() == pytest.approx(400, abs=1e-6)
    # the coupling: comp1's average temperature (350 K) is comp2's hot end
    assert s2.meta["xrefs"]["comp1.aveop1(T)"] == pytest.approx(350, rel=1e-6)
    assert T2.max() == pytest.approx(350, rel=1e-6) and T2.min() == pytest.approx(300, abs=1e-6)
    # each solution evaluates its own namespace, and refuses the other's by name
    from elmerstudio.core.nodes import bind_results
    with m.scoped(c2):
        b2 = bind_results(m, s2)
    assert np.allclose(b2.evaluate("comp2.T"), T2)
    with pytest.raises(Exception, match="comp1, another component"):
        b2.evaluate("comp1.T")
    # default plots: one solution dataset per component, each plot group on its own component's dataset
    made = B.default_plots(m, st, "stationary")
    dsets = [d for d in m.results.child("results.datasets").children if d.kind == "dset.solution"]
    assert sorted(B.dataset_component(m, d).tag for d in dsets) == ["comp1", "comp2"]
    assert {B.dataset_component(m, m.by_tag(pg.get("data"))).tag for pg in made} == {"comp1", "comp2"}
    # save and reopen away from the scratch folder: both components' solutions come back
    path = os.path.join(tempfile.mkdtemp(prefix="es_mc_save_"), "two.esm")
    project.save(m, path, work)
    m2, work2 = project.load(path)
    assert [c.tag for c in m2.components()] == ["comp1", "comp2"]
    r1 = Solution(os.path.join(component_dir(m2, os.path.join(work2, st.tag), "comp1"), "sol_0_0"))
    r2 = Solution(os.path.join(component_dir(m2, os.path.join(work2, st.tag), "comp2"), "sol_0_0"))
    assert np.allclose(r1.evaluate("T"), T1) and np.allclose(r2.evaluate("T"), T2)


def test_coupling_controls(elmer):
    # a circle of couplings is refused before anything runs
    m, c1, c2, ht2 = two_components()
    t1 = next(f for f in c1.child("physics.ht").children if f.get("T0") == "400[K]")
    m.create("cpl.maximum", c2.child("definitions")).props["opname"] = "maxop1"
    t1.props["T0"] = "comp2.maxop1(T)"
    with pytest.raises(RuntimeError, match="in a circle: comp1 -> comp2 -> comp1|in a circle: comp2 -> comp1 -> comp2"):
        component_order(m)
    # a field of another component cannot enter an equation (only its coupling operators)
    m, c1, c2, ht2 = two_components()
    f = next(f for f in ht2.children if f.get("T0") == "comp1.aveop1(T)")
    f.props["T0"] = "comp1.T"
    with pytest.raises(RuntimeError, match="a field of Component 1 cannot enter the equations"):
        StudyRunner(m, m.studies()[0], tempfile.mkdtemp(prefix="es_mc_"), elmer).run()
    # the coupled component alone in a study: the source is not solved
    m, c1, c2, ht2 = two_components()
    for step in m.studies()[0].children:
        if "physics" in step.props:
            step.props["physics"]["ht"] = False
    with pytest.raises(RuntimeError, match="is not solved before it"):
        StudyRunner(m, m.studies()[0], tempfile.mkdtemp(prefix="es_mc_"), elmer).run()


def test_migration_of_a_one_component_file():
    """A file from before components were namespaced opens unchanged: its dataset has no component and means the
    first one, and its solutions stay where they were (the study folder)."""
    m = B.new_model("2D", ["ht"], "stationary")
    st = m.studies()[0]
    d = B.solution_dataset(m, st)
    d.props.pop("comp", None)
    assert B.dataset_component(m, d) is m.component
    assert component_dir(m, "/w/std1", m.component.tag) == "/w/std1"
