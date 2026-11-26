"""Gossamer structures: geometric nonlinearity, prestress / prestrain.

Verifies (without Elmer) that a prestressed structural study emits the
correct Elmer keywords, and (pure math) that analytical drumhead mode ratios are as expected.
A light Elmer eigen solve confirms the geometric-stiffening effect of prestress:
pre-stress raises eigenfrequencies, and the membrane part of the frequency matches the
closed-form drumhead formula.  All checks are against Elmer StressSolver only -- the
removed membrane / cable / large-deflection features are listed below.

NOTE: The following features were REMOVED after verification against Elmer's SOLVER.KEYWORDS
and source code (StressSolve.F90, ShellSolver.F90, ElasticSolve.F90):
- solid.membrane: "Membrane Formulation" and "Bending Stiffness" keywords do not exist
- solid.cable: "Cable" and "Tension Only" keywords do not exist
- geometric_nonlinearity/large_deflection in Solid Mechanics: StressSolver does not support
  "Large Deflection", "Nonlinear Elasticity", or "Dead Loads" keywords (only ShellSolver does)
- solid.prestrain now outputs the "Pre Strain" material keyword (was incorrectly "Initial Strain")
- shell "Dead Loads": not in SOLVER.KEYWORDS; ShellSolver reads it from the Boundary
  Condition section, so it cannot be set from the Solver section
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


def _structural_model(study_kind, tension=TENSION, plane="stress", E=1e6, prestressed=True):
    m = B.new_model("2D", ["solid"], study_kind)
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(A), h=str(A))
    geo = build_geometry(m, display=False)
    sm = m.component.child("physics.solid")
    sm.props["plane"] = plane
    sm.props["prestressed"] = prestressed          # Geometric Stiffness solver setting
    sm.child("solid.lemm").props.update(
        E_src="user", E=str(E), nu_src="user", nu="0.0",
        rho_src="user", rho=str(RHO),
    )
    # Pre Stress material keyword (always emitted; it only takes effect together with
    # Geometric Stiffness -- see StressSolve.F90, applied iff GeometricStiffness/StabilityAnalysis)
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


def test_prestressed_eigen_emits_geometric_stiffness():
    m, geo, st = _structural_model("eigen")
    text = build_sif(m, st, _step(st), geo, 100).text
    assert "Geometric Stiffness = Logical True" in text


def test_prestress_feature_emits_pre_stress():
    m, geo, st = _structural_model("stationary")
    text = build_sif(m, st, _step(st), geo, 100).text
    # Pre Stress should be emitted with correct component count for 2D plane stress (3)
    assert "Pre Stress(3) = Real" in text
    assert "1000000.0 1000000.0 0.0" in text


def test_prestrain_feature_emits_pre_strain():
    m, geo, st = _structural_model("stationary")
    B.add_feature(m, m.component.child("physics.solid"), "solid.prestrain", geo.select("domain"),
                  Exx="0.001", Eyy="0.001", Exy="0")
    text = build_sif(m, st, _step(st), geo, 100).text
    # Pre Strain should be emitted with correct component count for 2D plane stress (3)
    assert "Pre Strain(3) = Real" in text
    assert "0.001 0.001 0.0" in text


def test_prestress_3d_emits_six_components():
    m = B.new_model("3D", ["solid"], "eigen")
    g = m.component.child("geometry")
    m.create("geom.block", g, index=0, w="1.0", h="1.0", d="1.0")
    geo = build_geometry(m, display=False)
    sm = m.component.child("physics.solid")
    sm.child("solid.lemm").props.update(
        E_src="user", E="1e9", nu_src="user", nu="0.0",
        rho_src="user", rho="1000.0",
    )
    B.add_feature(
        m, sm, "solid.prestress", geo.select("domain"),
        Sxx="1.0", Syy="2.0", Szz="3.0", Sxy="4.0", Syz="5.0", Sxz="6.0",
    )
    for face in geo.select("boundary"):
        B.add_feature(m, sm, "solid.fixed", [face])
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    for face in geo.select("boundary"):
        B.add_feature(m, mesh, "mesh.distribution", [face], n=4)
    st = m.studies()[0]
    text = build_sif(m, st, _step(st), geo, 100).text
    # Pre Stress should have 6 components for 3D (Voigt: xx, yy, zz, xy, yz, xz)
    assert "Pre Stress(6) = Real" in text
    assert "1.0 2.0 3.0 4.0 5.0 6.0" in text


def test_prestrain_3d_emits_six_components():
    m = B.new_model("3D", ["solid"], "eigen")
    g = m.component.child("geometry")
    m.create("geom.block", g, index=0, w="1.0", h="1.0", d="1.0")
    geo = build_geometry(m, display=False)
    sm = m.component.child("physics.solid")
    sm.child("solid.lemm").props.update(
        E_src="user", E="1e9", nu_src="user", nu="0.0",
        rho_src="user", rho="1000.0",
    )
    B.add_feature(
        m, sm, "solid.prestrain", geo.select("domain"),
        Exx="1.0", Eyy="2.0", Ezz="3.0", Exy="4.0", Eyz="5.0", Exz="6.0",
    )
    for face in geo.select("boundary"):
        B.add_feature(m, sm, "solid.fixed", [face])
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    for face in geo.select("boundary"):
        B.add_feature(m, mesh, "mesh.distribution", [face], n=4)
    st = m.studies()[0]
    text = build_sif(m, st, _step(st), geo, 100).text
    # Pre Strain should have 6 components for 3D (Voigt: xx, yy, zz, xy, yz, xz)
    assert "Pre Strain(6) = Real" in text
    assert "1.0 2.0 3.0 4.0 5.0 6.0" in text


def test_prestress_axisymmetric_emits_four_components():
    m = B.new_model("2Daxi", ["solid"], "eigen")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1.0", h="1.0")
    geo = build_geometry(m, display=False)
    sm = m.component.child("physics.solid")
    sm.props["prestressed"] = True
    sm.child("solid.lemm").props.update(
        E_src="user", E="1e9", nu_src="user", nu="0.0",
        rho_src="user", rho="1000.0",
    )
    B.add_feature(
        m, sm, "solid.prestress", geo.select("domain"),
        Sxx="1.0", Syy="2.0", Szz="3.0", Sxy="4.0",
    )
    for edge in geo.select("boundary"):
        B.add_feature(m, sm, "solid.fixed", [edge])
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    for edge in geo.select("boundary"):
        B.add_feature(m, mesh, "mesh.distribution", [edge], n=4)
    st = m.studies()[0]
    text = build_sif(m, st, _step(st), geo, 100).text
    # Pre Stress should have 4 components for axisymmetric (rr, zz, rz, tt)
    assert "Pre Stress(4) = Real" in text
    assert "1.0 2.0 4.0 3.0" in text


# --------------------------------------------------------------------------- analytical drumhead modes (pure math)


def test_analytical_drumhead_mode_ratios():
    r21, r22 = drumhead_ratios()
    assert r21 == pytest.approx(math.sqrt(5 / 2), rel=1e-9)   # 1.5811...
    assert r22 == pytest.approx(2.0, rel=1e-9)                # (2,2) is exactly 2x (1,1)
    assert f_mem(1, 1, TENSION) > 0


# --------------------------------------------------------------------------- Elmer eigen solve: geometric stiffening of a prestressed membrane


def _eigenfrequencies(elmer, tension=TENSION, E=1e6, prestressed=True):
    """Build + Elmer-solve the prestressed square membrane; return sorted eigenfrequencies.

    The Pre Stress material keyword is always emitted; it only affects the result when
    Geometric Stiffness is also active (prestressed=True), per StressSolve.F90.
    """
    m, geo, st = _structural_model("eigen", tension=tension, E=E, prestressed=prestressed)
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


def test_geometric_stiffening_raises_frequency(elmer):
    # Prestress (Geometric Stiffness) raises the eigenfrequency of the plate.
    # Negative control: Pre Stress emitted but Geometric Stiffness switched off must give
    # the same result as zero prestress (Pre Stress is inert without Geometric Stiffness,
    # StressSolve.F90 applies it only under GeometricStiffness/StabilityAnalysis).
    f_prestressed = _eigenfrequencies(elmer, tension=1000.0, prestressed=True)[0]      # 29.52 Hz
    f_no_gs = _eigenfrequencies(elmer, tension=1000.0, prestressed=False)[0]            # 19.14 Hz
    f_zero = _eigenfrequencies(elmer, tension=0.0, prestressed=True)[0]                 # 19.14 Hz
    assert f_no_gs == pytest.approx(f_zero, rel=1e-6)   # Pre Stress without Geometric Stiffness: no effect
    assert f_prestressed > f_no_gs                       # geometric stiffening raises the frequency


def test_membrane_component_matches_drumhead(elmer):
    # A prestressed plate vibrates as bending + membrane: f^2 = f_bending^2 + f_membrane^2.
    # The membrane part must match the closed-form drumhead frequency.
    # Tolerance: f11 is mesh-converged (refinement ndiv 8 -> 64 changes f11 by <0.01 %),
    # so the residual vs the ideal membrane is the plate idealisation (~0.5 %, here 1 %).
    fb = _eigenfrequencies(elmer, tension=0.0, prestressed=True)[0]          # 19.14 Hz bending baseline
    f1000 = _eigenfrequencies(elmer, tension=1000.0, prestressed=True)[0]    # 29.52 Hz
    fm_num = math.sqrt(f1000 ** 2 - fb ** 2)                                 # 22.47 Hz membrane part
    fm_ana = f_mem(1, 1, 1000.0)                                               # 22.36 Hz closed form
    assert fm_num == pytest.approx(fm_ana, rel=1e-2)
