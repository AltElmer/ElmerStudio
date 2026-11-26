"""Gossamer structures: membrane verification.

The prestress/prestrain feature (Solid Mechanics + ``Pre Stress``/``Pre Strain`` material
keywords with ``Geometric Stiffness``) was REMOVED from this branch: it cannot be verified
against the CI Elmer (CSC PPA ~v26.2), which runs ``ElasticSolver`` for this model and rejects
``"Pre Stress"``/``"Pre Strain"`` with
``ERROR:: ElasticSolver: ... are not implemented here``. The governing module is
``StressSolve``/``StressSolver``, which is not the solver CI's Elmer build instantiates here,
so the prestressed eigenfrequency / geometric stiffening claim is not provable on either
supported build and is listed as a known limitation
(see docs/modules/ROOT/pages/known-limitations.adoc).

Two independent checks remain:

1. ``test_analytical_drumhead_mode_ratios`` — pure math: the closed-form eigenfrequency ratios
   of a fixed rectangular membrane. No Elmer solve; passes on every Elmer build.

2. ``test_membrane_uniaxial_tension_matches_analytical`` — solver-backed: a 2D plane-stress
   membrane strip under uniaxial tension solved with Elmer ``StressSolver``, checked against
   the exact linear-elastic tip displacement ``u(L) = F * L / E``. Uses the ``elmer`` fixture,
   so something is actually solved and verified against an independent reference.
"""
import math
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.nodes import bind_results
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


# --------------------------------------------------------------------------- analytical drumhead modes (pure math)


def test_analytical_drumhead_mode_ratios():
    r21, r22 = drumhead_ratios()
    assert r21 == pytest.approx(math.sqrt(5 / 2), rel=1e-9)   # 1.5811...
    assert r22 == pytest.approx(2.0, rel=1e-9)                # (2,2) is exactly 2x (1,1)
    assert f_mem(1, 1, TENSION) > 0


# --------------------------------------------------------------------------- solver-backed membrane check


def _tension_model():
    """2D plane-stress membrane strip fixed at x = 0, pulled by traction Fx at x = L."""
    L, H, E, RHO, F = 1.0, 0.05, 200e9, 7850.0, 1e6
    m = B.new_model("2D", ["solid"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(H))
    geo = build_geometry(m, display=False)
    sm = m.component.child("physics.solid")
    sm.props["plane"] = "stress"
    sm.child("solid.lemm").props.update(E_src="user", E=f"{E!r}[Pa]", nu_src="user", nu="0",
                                        rho_src="user", rho=f"{RHO!r}[kg/m^3]")
    left, right = geo.select("boundary", xmax=0), geo.select("boundary", xmin=L)
    sides = geo.select("boundary", ymax=0) + geo.select("boundary", ymin=H)
    B.add_feature(m, sm, "solid.fixed", left)
    B.add_feature(m, sm, "solid.bload", right, Fx=str(F))
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    for b, n in ((sides, 40), (left + right, 2)):
        B.add_feature(m, mesh, "mesh.distribution", b, n=n)
    return m, geo, m.studies()[0], L, E, F


def tip(sol, i, name="u"):
    x = sol.topo.nodes[:, 0] * sol.scale
    on = np.isclose(x, 1.0, atol=1e-9)
    assert on.sum() >= 3
    return float(np.mean(sol.evaluate(name, i)[on]))


def test_membrane_uniaxial_tension_matches_analytical(elmer):
    """Linear-elastic membrane under uniaxial tension: tip displacement is exact,
    u(L) = F * L / E (unit thickness, nu = 0). Verified against the analytical reference."""
    m, geo, st, L, E, F = _tension_model()
    text = build_sif(m, st, B.study_steps(st)[0], geo, 100).text
    assert "StressSolver" in text
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_gossamer_"), elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    # quasi-static limit: the first (only) stationary frame
    u = tip(sol, 0, "u")
    assert u == pytest.approx(F * L / E, rel=1e-4), u
    assert u > 0
    assert math.isfinite(u)
