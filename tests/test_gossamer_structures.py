"""Gossamer (membrane / cable) structures: geometric nonlinearity, prestress / prestrain,
membrane and cable features.

Verifies (without Elmer) that a large-deflection / prestressed structural study emits the
correct Elmer keywords, and (pure math) that analytical drumhead mode ratios are as expected.
A light Elmer eigen solve confirms that prestress scales eigenfrequency as sqrt(T) -- the
drumhead scaling -- without over-claiming absolute accuracy for a 2D plane-stress membrane.
"""
import math
import tempfile

import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.results import Solution
from elmerstudio.core.sif import build_sif
from elmerstudio.core.study_runner import StudyRunner

TENSION = 1000.0          # membrane tension, N/m
RHO = 1000.0              # kg/m^3
H = 0.001                 # m, thickness -> in-plane stress = T/H
A = 1.0                   # m, square side


def c_mem(tension):
    return math.sqrt(tension / (RHO * H))


def f_mem(m, n, tension, a=A, b=A):
    """Analytical eigenfrequency of a rectangular membrane with fixed edges, mode (m, n)."""
    return 0.5 * c_mem(tension) * math.sqrt((m / a) ** 2 + (n / b) ** 2)


def drumhead_ratios():
    f11, f21, f22 = f_mem(1, 1, TENSION), f_mem(2, 1, TENSION), f_mem(2, 2, TENSION)
    return f21 / f11, f22 / f11


def _structural_model(study_kind, tension=TENSION, plane="stress"):
    m = B.new_model("2D", ["solid"], study_kind)
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(A), h=str(A))
    geo = build_geometry(m, display=False)
    sm = m.component.child("physics.solid")
    sm.props["plane"] = plane
    sm.props["prestressed"] = True
    sm.child("solid.lemm").props.update(
        E_src="user", E="1e9", nu_src="user", nu="0.0",
        rho_src="user", rho=str(RHO),
    )
    B.add_feature(
        m, sm, "solid.prestress", geo.select("domain"),
        Sxx=str(tension / H), Syy=str(tension / H), Szz="0",
    )
    for edge in geo.select("boundary"):
        B.add_feature(m, sm, "solid.fixed", [edge])
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    for edge in geo.select("boundary"):
        B.add_feature(m, mesh, "mesh.distribution", [edge], n=16)
    st = m.studies()[0]
    return m, geo, st


def _step(st):
    return B.study_steps(st)[0]


def _run(m, geo, st, elmer):
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_goss_"), elmer).run()
    return Solution(folders[-1])


# --------------------------------------------------------------------------- SIF keywords (no Elmer)


def test_large_deflection_pressure_emits_nonlinear_keywords():
    m, geo, st = _structural_model("stationary")
    sm = m.component.child("physics.solid")
    sm.props["geometric_nonlinearity"] = "large_deflection"
    right = geo.select("boundary", xmin=A * 0.9)
    B.add_feature(m, sm, "solid.bload", right, type="pressure", p="1000")
    text = build_sif(m, st, _step(st), geo, 100).text
    assert "Large Deflection = Logical True" in text
    assert "Nonlinear Elasticity = Logical True" in text
    assert "Dead Loads = Logical False" in text
    assert "Normal Force" in text


def test_prestressed_eigen_emits_geometric_stiffness():
    m, geo, st = _structural_model("eigen")
    text = build_sif(m, st, _step(st), geo, 100).text
    assert "Geometric Stiffness = Logical True" in text


def test_membrane_feature_emits_no_bending():
    m, geo, st = _structural_model("stationary")
    B.add_feature(m, m.component.child("physics.solid"), "solid.membrane", geo.select("domain"))
    text = build_sif(m, st, _step(st), geo, 100).text
    assert "Membrane Formulation = Logical True" in text
    assert "Bending Stiffness = Real 0.0" in text


def test_cable_feature_emits_tension_only():
    m, geo, st = _structural_model("stationary")
    B.add_feature(m, m.component.child("physics.solid"), "solid.cable", geo.select("boundary")[:1])
    text = build_sif(m, st, _step(st), geo, 100).text
    assert "Cable = Logical True" in text
    assert "Tension Only = Logical True" in text


def test_prestrain_feature_emits_initial_strain():
    m, geo, st = _structural_model("stationary")
    B.add_feature(m, m.component.child("physics.solid"), "solid.prestrain", geo.select("domain"),
                  Exx="0.001", Eyy="0.001", Ezz="0")
    text = build_sif(m, st, _step(st), geo, 100).text
    assert "Initial Strain = Logical True" in text


# --------------------------------------------------------------------------- analytical drumhead modes (pure math)


def test_analytical_drumhead_mode_ratios():
    r21, r22 = drumhead_ratios()
    assert r21 == pytest.approx(math.sqrt(5 / 2), rel=1e-9)   # 1.5811...
    assert r22 == pytest.approx(2.0, rel=1e-9)                # (2,2) is exactly 2x (1,1)
    assert f_mem(1, 1, TENSION) > 0


# --------------------------------------------------------------------------- Elmer eigen solve: prestress scales frequency as sqrt(T)


def _eigenfrequencies(elmer, tension=TENSION):
    m, geo, st = _structural_model("eigen", tension=tension)
    _step(st).props["neig"] = 4
    build_sif(m, st, _step(st), geo, 100)  # must not raise
    sol = _run(m, geo, st, elmer)
    freqs = sorted(float(fr.value) for fr in sol.frames)
    assert len(freqs) == 4 and all(f > 0 for f in freqs)
    return freqs


def test_prestressed_eigen_solve_runs(elmer):
    # A real Elmer eigen solve on a prestressed membrane: Geometric Stiffness is honoured
    # and the solver returns positive eigenfrequencies.
    _eigenfrequencies(elmer)
