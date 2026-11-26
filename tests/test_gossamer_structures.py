"""Gossamer structures: analytical membrane reference (pure math).

The prestress/prestrain feature (Solid Mechanics + ``Pre Stress``/``Pre Strain`` material
keywords with ``Geometric Stiffness``) was REMOVED from this branch: it cannot be verified
against the CI Elmer (CSC PPA ~v26.2), which runs ``ElasticSolver`` and rejects
``"Pre Stress"``/``"Pre Strain"`` with ``ERROR:: ElasticSolver: ... are not implemented here``.
The governing module is ``StressSolve``/``StressSolver``, which is not the solver that CI's
Elmer build instantiates for this model, so the prestressed eigenfrequency / geometric
stiffening claim is not provable on either supported build and is listed as a known
limitation (see docs/modules/ROOT/pages/known-limitations.adoc).

What remains is the closed-form drumhead reference: a pure-math check that the analytical
eigenfrequency ratios of a fixed rectangular membrane are as expected. No Elmer solve is
involved, so this check passes on every Elmer build.
"""
import math

import pytest

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
