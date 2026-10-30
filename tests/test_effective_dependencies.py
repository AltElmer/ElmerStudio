"""Tests for F07: Effective dependencies - disabled features must not create cycles."""
import os
import tempfile

import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.study_runner import StudyRunner, component_order, cross_references


def test_disabled_return_feature_allows_study():
    """Two components with a working one-way transfer:
    - Active return dependency is refused (correct)
    - Disabled return boundary feature is accepted (defect: currently refused)"""
    # Component 1: 2D strip, T=300K at x=0, 400K at x=1
    m = B.new_model("2D", ["ht"], "stationary")
    c1 = m.component
    m.create("geom.rectangle", c1.child("geometry"), index=0, w="1", h="0.25")
    B.add_material(m, "Copper", all_domains=True)
    with m.scoped(c1):
        geo1 = build_geometry(m, display=False)
    ht1 = c1.child("physics.ht")
    B.add_feature(m, ht1, "ht.temperature", geo1.select("boundary", xmax=0), T0="300[K]")
    B.add_feature(m, ht1, "ht.temperature", geo1.select("boundary", xmin=1), T0="400[K]")
    # Coupling operator in comp1
    ave = m.create("cpl.average", c1.child("definitions"))
    ave.props["opname"] = "aveop1"
    ave.selection.all = True

    # Component 2: 3D block, T = comp1.aveop1(T) at x=0, 300K at x=10
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

    # Both physics solved in the study
    for step in m.studies()[0].children:
        if "physics" in step.props:
            step.props["physics"] = {ph.tag: True for ph in B.all_physics(m)}

    # --- Active return dependency: comp1 uses comp2's result (should be refused) ---
    # Add a return coupling in comp1 that references comp2
    maxop = m.create("cpl.maximum", c2.child("definitions"))
    maxop.props["opname"] = "maxop1"
    # Make comp1's hot boundary depend on comp2's max temperature
    t1 = next(f for f in c1.child("physics.ht").children if f.get("T0") == "400[K]")
    t1.props["T0"] = "comp2.maxop1(T)"

    # This should create a cycle and fail
    with pytest.raises(RuntimeError, match="circle|circular"):
        component_order(m)

    # --- Now DISABLE the return feature in comp1 ---
    t1.enabled = False
    # The cycle should be ignored - disabled feature doesn't create dependency
    order = component_order(m)
    assert len(order) == 2
    assert [c.tag for c in order] == ["comp1", "comp2"]


def test_disabled_domain_feature_cycle_same_behavior():
    """A cycle through a disabled DOMAIN feature behaves the same - accepted."""
    # Setup: comp2 depends on comp1 via boundary condition (comp2 -> comp1)
    # comp1 has a DOMAIN feature that references comp2 (comp1 -> comp2)
    # This creates a cycle. When the domain feature is disabled, the cycle should be broken.
    m = B.new_model("2D", ["ht"], "stationary")
    c1 = m.component
    m.create("geom.rectangle", c1.child("geometry"), index=0, w="1", h="0.25")
    B.add_material(m, "Copper", all_domains=True)
    with m.scoped(c1):
        geo1 = build_geometry(m, display=False)
    ht1 = c1.child("physics.ht")
    B.add_feature(m, ht1, "ht.temperature", geo1.select("boundary", xmax=0), T0="300[K]")
    B.add_feature(m, ht1, "ht.temperature", geo1.select("boundary", xmin=1), T0="400[K]")
    # Coupling operator in comp2 for comp1 to reference
    c2 = B.add_component(m, "3D")
    with m.scoped(c2):
        g2 = c2.child("geometry")
        g2.props["unit"] = "mm"
        m.create("geom.block", g2, index=0, w="10", d="2", h="2")
        ht2 = B.add_physics(m, "ht")
        B.add_material(m, "Aluminum", all_domains=True)
        geo2 = build_geometry(m, display=False)
        maxop = m.create("cpl.maximum", c2.child("definitions"))
        maxop.props["opname"] = "maxop1"
        # comp2's boundary depends on comp1's average (comp2 -> comp1)
        ave = m.create("cpl.average", c1.child("definitions"))
        ave.props["opname"] = "aveop1"
        ave.selection.all = True
        B.add_feature(m, ht2, "ht.temperature", geo2.select("boundary", xmax=0), T0="comp1.aveop1(T)")
        B.add_feature(m, ht2, "ht.temperature", geo2.select("boundary", xmin=10), T0="300[K]")

    for step in m.studies()[0].children:
        if "physics" in step.props:
            step.props["physics"] = {ph.tag: True for ph in B.all_physics(m)}

    # DOMAIN feature in comp1 that references comp2 (comp1 -> comp2)
    # ht.source has Q0 (volumetric heat source) which can be an expression
    heat = m.create("ht.source", ht1)
    heat.selection.all = True
    heat.props["Q0"] = "comp2.maxop1(T)"  # Domain feature in comp1 referencing comp2
    # Enabled: creates cycle comp1 -> comp2 -> comp1
    with pytest.raises(RuntimeError, match="circle|circular"):
        component_order(m)

    # Disabled: cycle broken, should succeed
    heat.enabled = False
    order = component_order(m)
    assert len(order) == 2
    assert [c.tag for c in order] == ["comp1", "comp2"]


def test_disabled_return_feature_boundary_same_behavior():
    """A cycle through a disabled BOUNDARY feature behaves the same - accepted."""
    # Same as first test but with a boundary feature as the return path
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

    # Return via a boundary feature in comp1 (instead of the temperature BC itself)
    # Make the cold boundary of comp1 depend on comp2
    t1_cold = next(f for f in c1.child("physics.ht").children if f.get("T0") == "300[K]")
    t1_cold.props["T0"] = "comp2.maxop1(T)"
    maxop = m.create("cpl.maximum", c2.child("definitions"))
    maxop.props["opname"] = "maxop1"

    # Enabled: cycle detected
    with pytest.raises(RuntimeError, match="circle|circular"):
        component_order(m)

    # Disabled: should be accepted
    t1_cold.enabled = False
    order = component_order(m)
    assert len(order) == 2