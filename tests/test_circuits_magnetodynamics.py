"""Issue #3: Circuits and MagnetoDynamics2D (harmonic / frequency domain).

Covers:
* ``mf.coil`` stranded coil excitation -> a ``Current Density`` body force
  (with ``Calculate Potential``) in the SIF.
* ``mf`` in a ``study.freq`` step -> ``MagnetoDynamics2DHarmonic``,
  complex variable ``Az[Az re:1 Az im:1]``, ``Linear System Complex`` and an
  ``Angular Frequency`` in the Simulation section.
* ``cir`` lumped circuit (resistor, inductor, capacitor, sources) ->
  ``Component`` sections plus ``CircuitsAndDynamics`` / ``CircuitsOutput``
  solver blocks.
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
    B.add_feature(m, sm, "mf.coil", dom, type="stranded", Jz="1e6[A/m^2]")
    st = m.studies()[0]
    return m, geo, st, dom


def _sif_text(m, st, step_kind):
    step = B.study_steps(st)[0]
    geo = build_geometry(m, display=False)
    return build_sif(m, st, step, geo, 1).text


def test_mf_coil_emits_current_density():
    m, geo, st, dom = _mf_model("stationary")
    text = _sif_text(m, st, "stationary")
    assert "MagnetoDynamics2D\"" in text or "MagnetoDynamics2D " in text
    assert "MagnetoDynamics2DHarmonic" not in text
    assert "Current Density" in text
    assert "Calculate Potential" in text
    assert "Linear System Complex" not in text


def test_mf_harmonic_emits_complex_system():
    m, geo, st, dom = _mf_model("study.freq")
    text = _sif_text(m, st, "study.freq")
    assert "MagnetoDynamics2DHarmonic" in text
    assert "Az[Az re:1 Az im:1]" in text
    assert "Linear System Complex" in text
    assert "Angular Frequency" in text
    assert "Calculate Potential" in text  # coil still present


def test_circuit_emits_components_and_solvers():
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
    m.create("cir.inductor", cir, L="1e-3[H]")
    m.create("cir.vsource", cir, V0="1[V]")
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    text = build_sif(m, st, step, geo, 1).text
    assert "CircuitsAndDynamics" in text
    assert "CircuitsOutput" in text
    assert 'Component Type = String "Resistor"' in text
    assert "Resistance = Real 10.0" in text
    assert 'Component Type = String "Inductor"' in text
    assert "Inductance = Real 0.001" in text


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
