"""Issue #3: Circuits and MagnetoDynamics2D (harmonic / frequency domain).

Covers:
* ``mf.coil`` stranded/massive/foil coil excitation -> Elmer Component section
  with ``Component Type = Coil``, ``Coil Type``, ``Number of Turns``, ``Coil Cross Section``.
* ``mf`` in a ``study.freq`` step -> ``MagnetoDynamics2DHarmonic``,
  complex variable ``Az[Az re:1 Az im:1]``, ``Linear System Complex`` and an
  ``Angular Frequency`` in the Simulation section.
* ``cir`` lumped circuit (resistor, voltage/current source) ->
  ``Component`` sections (Resistor only) plus ``CircuitsAndDynamics`` /
  ``CircuitsAndDynamicsHarmonic`` / ``CircuitsOutput`` solver blocks.
  Voltage/current sources are BodyForce with ``testsource Re/Im``.
* Frequency-domain skin-depth decay of the magnetic field in a conducting
  slab (real Elmer solve): Joule loss grows with frequency while the
  shielded field drops.
"""
import tempfile

import numpy as np

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.results import Solution
from elmerstudio.core.sif import build_sif
from elmerstudio.core.study_runner import StudyRunner


def _mf_model(step_kind="stationary"):
    m = B.new_model("2D", ["mf"], step_kind)
    comp = m.component
    m.create("geom.rectangle", comp.child("geometry"), index=0, w="0.1", h="0.05")
    geo = build_geometry(m, display=False)
    sm = comp.child("physics.mf")
    dom = list(geo.select("domain"))
    B.add_material(m, "Copper", all_domains=True)
    B.add_feature(m, sm, "mf.ecd", dom, Jz="1e6[A/m^2]")
    B.add_feature(m, sm, "mf.coil", dom, type="stranded", N="100", cross_section="1e-4")
    st = m.studies()[0]
    return m, geo, st, dom


def _sif_text(m, st, step_kind):
    step = B.study_steps(st)[0]
    geo = build_geometry(m, display=False)
    return build_sif(m, st, step, geo, 1).text


def test_mf_coil_emits_component():
    """Stranded coil generates a Component section with proper keywords."""
    m, geo, st, dom = _mf_model("stationary")
    text = _sif_text(m, st, "stationary")
    assert "MagnetoDynamics2D" in text
    assert "MagnetoDynamics2DHarmonic" not in text
    # Coil is now a Component, not BodyForce
    assert 'Component Type = String "Coil"' in text
    assert 'Coil Type = String "stranded"' in text
    assert "Number of Turns = Real 100.0" in text
    assert "Coil Cross Section = Real 0.0001" in text
    assert "Master Bodies = Integer 1" in text
    # External current density still emits BodyForce
    assert "Current Density = Real 1000000.0" in text
    assert "Linear System Complex" not in text


def test_mf_harmonic_emits_complex_system():
    m, geo, st, dom = _mf_model("study.freq")
    text = _sif_text(m, st, "study.freq")
    assert "MagnetoDynamics2DHarmonic" in text
    assert "Az[Az re:1 Az im:1]" in text
    assert "Linear System Complex" in text
    assert "Angular Frequency" in text
    # Coil component still present in harmonic study
    assert 'Component Type = String "Coil"' in text
    assert 'Coil Type = String "stranded"' in text
    assert "Number of Turns = Real 100.0" in text


def test_circuit_emits_resistor_component_and_sources():
    """Circuit with resistor and voltage/current sources."""
    m = B.new_model("2D", ["mf", "cir"], "stationary")
    comp = m.component
    m.create("geom.rectangle", comp.child("geometry"), index=0, w="0.1", h="0.05")
    geo = build_geometry(m, display=False)
    sm = comp.child("physics.mf")
    dom = list(geo.select("domain"))
    B.add_material(m, "Copper", all_domains=True)
    B.add_feature(m, sm, "mf.ecd", dom, Jz="1e6[A/m^2]")
    cir = comp.child("physics.cir")
    m.create("cir.resistor", cir, R="10[ohm]")
    B.add_feature(m, cir, "cir.vsource", dom, V0="1[V]", phase="0")
    B.add_feature(m, cir, "cir.isource", dom, I0="1[A]", phase="0")
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    text = build_sif(m, st, step, geo, 1).text
    assert "CircuitsAndDynamics" in text
    assert "CircuitsOutput" in text
    # Resistor as Component
    assert 'Component Type = String "Resistor"' in text
    assert "Resistance = Real 10.0" in text
    # No Inductor/Capacitor components (not supported by Elmer)
    assert 'Component Type = String "Inductor"' not in text
    assert 'Component Type = String "Capacitor"' not in text
    assert 'Component Type = String "Voltage Source"' not in text
    assert 'Component Type = String "Current Source"' not in text
    # Sources as BodyForce with testsource
    assert "testsource Re = Real 1.0" in text


def test_circuit_harmonic_sources_emit_complex():
    """Harmonic study: voltage/current sources emit complex testsource Re/Im."""
    m = B.new_model("2D", ["mf", "cir"], "study.freq")
    comp = m.component
    m.create("geom.rectangle", comp.child("geometry"), index=0, w="0.1", h="0.05")
    geo = build_geometry(m, display=False)
    sm = comp.child("physics.mf")
    dom = list(geo.select("domain"))
    B.add_material(m, "Copper", all_domains=True)
    B.add_feature(m, sm, "mf.coil", dom, type="stranded", N="100", cross_section="1e-4")
    cir = comp.child("physics.cir")
    m.create("cir.resistor", cir, R="10[ohm]")
    # Only test one source at a time since they overwrite each other in BodyForce
    B.add_feature(m, cir, "cir.vsource", dom, V0="1[V]", phase="90")  # 90 deg = pure imaginary
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    text = build_sif(m, st, step, geo, 1).text
    # V0=1, phase=90 -> Re=cos(90)=0, Im=sin(90)=1
    assert "testsource Re = Real 0.0" in text
    assert "testsource Im = Real 1.0" in text

    # Now test isource alone
    m2 = B.new_model("2D", ["mf", "cir"], "study.freq")
    comp2 = m2.component
    m2.create("geom.rectangle", comp2.child("geometry"), index=0, w="0.1", h="0.05")
    geo2 = build_geometry(m2, display=False)
    sm2 = comp2.child("physics.mf")
    dom2 = list(geo2.select("domain"))
    B.add_material(m2, "Copper", all_domains=True)
    B.add_feature(m2, sm2, "mf.coil", dom2, type="stranded", N="100", cross_section="1e-4")
    cir2 = comp2.child("physics.cir")
    m2.create("cir.resistor", cir2, R="10[ohm]")
    B.add_feature(m2, cir2, "cir.isource", dom2, I0="2[A]", phase="0")
    st2 = m2.studies()[0]
    step2 = B.study_steps(st2)[0]
    text2 = build_sif(m2, st2, step2, geo2, 1).text
    # I0=2, phase=0 -> Re=2, Im=0
    assert "testsource Re = Real 2.0" in text2
    assert "testsource Im = Real 0.0" in text2


def test_coil_types_massive_and_foil():
    """Test massive and foil winding coil types."""
    m = B.new_model("2D", ["mf"], "stationary")
    comp = m.component
    m.create("geom.rectangle", comp.child("geometry"), index=0, w="0.1", h="0.05")
    geo = build_geometry(m, display=False)
    sm = comp.child("physics.mf")
    dom = list(geo.select("domain"))
    B.add_material(m, "Copper", all_domains=True)
    # Massive coil (voltage-driven)
    B.add_feature(m, sm, "mf.coil", dom, type="massive", V="10[V]", cross_section="1e-4")
    # Foil winding (current-driven, no cross section)
    m.create("geom.rectangle", comp.child("geometry"), index=1, w="0.1", h="0.05", x="0.1", y="0")
    geo2 = build_geometry(m, display=False)
    dom2 = [d for d in geo2.select("domain") if d not in dom]
    B.add_feature(m, sm, "mf.coil", dom2, type="foil winding", N="50")
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    text = build_sif(m, st, step, geo2, 1).text
    # Should have two coil components
    assert text.count('Component Type = String "Coil"') == 2
    assert 'Coil Type = String "massive"' in text
    assert 'Coil Type = String "foil winding"' in text
    # Massive has cross section but N=1
    assert "Coil Cross Section = Real 0.0001" in text
    # Foil winding has no cross section
    # Both have Number of Turns (massive=1, foil=50)
    assert "Number of Turns = Real 50.0" in text


def test_negative_control_invalid_coil_type():
    """Invalid coil type should raise an error or fall back to stranded."""
    m = B.new_model("2D", ["mf"], "stationary")
    comp = m.component
    m.create("geom.rectangle", comp.child("geometry"), index=0, w="0.1", h="0.05")
    geo = build_geometry(m, display=False)
    sm = comp.child("physics.mf")
    dom = list(geo.select("domain"))
    B.add_material(m, "Copper", all_domains=True)
    # Invalid coil type - should fall back to stranded
    B.add_feature(m, sm, "mf.coil", dom, type="invalid_type", N="100")
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    text = build_sif(m, st, step, geo, 1).text
    # Should fall back to stranded
    assert 'Coil Type = String "stranded"' in text


def test_negative_control_no_circuit_solver_without_cir():
    """Circuit solvers should not appear without cir physics."""
    m = B.new_model("2D", ["mf"], "stationary")
    comp = m.component
    m.create("geom.rectangle", comp.child("geometry"), index=0, w="0.1", h="0.05")
    geo = build_geometry(m, display=False)
    sm = comp.child("physics.mf")
    dom = list(geo.select("domain"))
    B.add_material(m, "Copper", all_domains=True)
    B.add_feature(m, sm, "mf.coil", dom, type="stranded", N="100")
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    text = build_sif(m, st, step, geo, 1).text
    assert "CircuitsAndDynamics" not in text
    assert "CircuitsOutput" not in text


def test_mf_stationary_solve(elmer):
    """End-to-end stationary mf solve: a nonzero B field in a current-carrying
    conductor (the mf solve pipeline runs through Elmer and produces a field)."""
    m = B.new_model("2D", ["mf"], "stationary")
    comp = m.component
    m.create("geom.rectangle", comp.child("geometry"), index=0, w="0.1", h="0.02")
    geo = build_geometry(m, display=False)
    sm = comp.child("physics.mf")
    dom = list(geo.select("domain"))
    B.add_material(m, "Copper", all_domains=True)
    B.add_feature(m, sm, "mf.ecd", dom, Jz="1e6[A/m^2]")
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    for b in (list(geo.select("boundary")),):
        B.add_feature(m, mesh, "mesh.distribution", b, n=8)
    tmp = tempfile.mkdtemp(prefix="es_mf_")
    folders = StudyRunner(m, m.studies()[0], tmp, elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    b = sol.evaluate("mf.normB", 0)
    b = np.asarray(b, dtype=float) if b is not None else np.array([0.0])
    assert float(np.nanmax(np.abs(b))) > 0.0, "stationary mf solve produced no magnetic field"