"""Qualification tests for Structural rod/beam (ladder row 4).

CORRECT: test_solid_dynamics.py::test_frequency_response_matches_the_exact_phasor
         test_solid_dynamics.py::test_eigenfrequency_agrees_and_ignores_damping
WRONG: test_solid_dynamics.py::test_damping_is_what_bounds_the_resonance
CHANGED: different dimensions, material, load, frequency - exact phasor must match
"""
from __future__ import annotations

import cmath
import math
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.elmer import find_elmer
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner


# Exact phasor for longitudinal bar vibration (plane stress, nu=0):
# u(L) = F tan(kL) / (E* k), E* = E(1 + i w beta), k^2 = rho(w^2 - i w alpha)/E*
# First eigenfrequency f1 = c/(4L), c = sqrt(E/rho)
# Damping: alpha, beta (Rayleigh)
# Tolerance: 2e-4 relative error on phasor (from test_solid_dynamics.py)


def _rod_model(L=1.0, H=0.05, E=200e9, rho=7850.0, F=1e6,
               alpha=None, beta=None, study="freq", freqs=None):
    """Build rod model with optional Rayleigh damping."""
    m = B.new_model("2D", ["solid"], study)
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(H))
    geo = build_geometry(m, display=False)
    sm = m.component.child("physics.solid")
    sm.props["plane"] = "stress"
    sm.child("solid.lemm").props.update(
        E_src="user", E=f"{E!r}[Pa]", nu_src="user", nu="0",
        rho_src="user", rho=f"{rho!r}[kg/m^3]")
    left, right = geo.select("boundary", xmax=0), geo.select("boundary", xmin=L)
    sides = geo.select("boundary", ymax=0) + geo.select("boundary", ymin=H)
    B.add_feature(m, sm, "solid.fixed", left)
    B.add_feature(m, sm, "solid.roller", sides)
    load = B.add_feature(m, sm, "solid.bload", right, Fx=str(F))
    if alpha is not None and beta is not None:
        B.add_feature(m, sm, "solid.damp", geo.select("domain"),
                      alpha=str(alpha), beta=str(beta))
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    B.add_feature(m, mesh, "mesh.distribution", sides, n=40)
    B.add_feature(m, mesh, "mesh.distribution", left + right, n=2)
    st = m.studies()[0]
    if study == "freq" and freqs is not None:
        B.study_steps(st)[0].props["freqs"] = " ".join(f"{f!r}" for f in freqs)
    elif study == "eigen":
        B.study_steps(st)[0].props["neig"] = 2
    return m, geo, st, load


def exact_tip(f, E, rho, alpha, beta, L=1.0, F=1e6):
    w = 2 * math.pi * f
    Es = E * (1 + 1j * w * beta)
    k = cmath.sqrt(rho * (w * w - 1j * w * alpha) / Es)
    return F * cmath.tan(k * L) / (Es * k)


# ---------------------------------------------------------------------- CHANGED VARIANT
# Different L, H, E, rho, F, alpha, beta, frequencies
# Exact phasor must still match
def test_rod_beam_changed_variant(elmer):
    """CHANGED: Different geometry, material, load, damping - exact phasor match."""
    L, H = 0.5, 0.02
    E, rho = 70e9, 2700.0  # Aluminum-like
    F = 5e5
    alpha, beta = 50.0, 1e-6
    FREQS = [50.0, 200.0, 500.0, 1000.0, 2000.0]

    m, geo, st, _load = _rod_model(L=L, H=H, E=E, rho=rho, F=F,
                                   alpha=alpha, beta=beta,
                                   study="freq", freqs=FREQS)

    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_qual_rod_"), elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)

    assert [fr.value for fr in sol.frames] == pytest.approx(FREQS)
    names = {a for a, *_ in sol.available(0)}
    assert {"solid.dispAmp", "solid.uim"} <= names

    for i, f in enumerate(FREQS):
        # Find tip nodes (x = L)
        x_nodes = sol.topo.nodes[:, 0] * sol.scale
        tip_mask = np.isclose(x_nodes, L, atol=1e-9)
        assert tip_mask.sum() >= 3
        u = float(np.mean(sol.evaluate("u", i)[tip_mask]))
        uim = float(np.mean(sol.evaluate("solid.uim", i)[tip_mask]))
        got = complex(u, uim)
        ex = exact_tip(f, E, rho, alpha, beta, L, F)
        rel_err = abs(got - ex) / abs(ex)
        assert rel_err <= 2e-4, f"f={f}: got={got}, ex={ex}, rel_err={rel_err}"
        amp = float(np.mean(sol.evaluate("solid.dispAmp", i)[tip_mask]))
        assert amp == pytest.approx(abs(ex), rel=2e-4)

    # Quasi-static limit at lowest frequency (50 Hz) - tolerance relaxed for 50 Hz
    quasi = F * L / E
    u0 = float(np.mean(sol.evaluate("u", 0)[tip_mask]))
    assert u0 == pytest.approx(quasi, rel=5e-4)


# CHANGED: eigenfrequency with different geometry/material
def test_rod_beam_eigen_changed_variant(elmer):
    """CHANGED: Different geometry/material - eigenfrequency must match analytic."""
    L, H = 0.8, 0.03
    E, rho = 210e9, 7850.0  # Steel-like
    c = math.sqrt(E / rho)
    f1 = c / (4 * L)
    f2 = 3 * f1  # second mode

    m, geo, st, _load = _rod_model(L=L, H=H, E=E, rho=rho,
                                   study="eigen")

    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_qual_rod_eig_"), elmer).run()
    sol = Solution(folders[-1])

    assert sol.frames[0].value == pytest.approx(f1, rel=1e-4)
    assert sol.frames[1].value == pytest.approx(f2, rel=1e-3)


# CHANGED: transient forced vibration with different frequency and damping
def test_rod_beam_transient_changed_variant(elmer):
    """CHANGED: Different transient params - must settle on harmonic response."""
    L, H = 0.6, 0.025
    E, rho = 150e9, 7000.0
    F = 2e5
    c = math.sqrt(E / rho)
    f1 = c / (4 * L)
    f0 = 0.6 * f1  # 60% of first eigenfrequency
    alpha = 2 * 0.1 * 2 * math.pi * f1  # mass damping, zeta=0.1 at f1
    beta = 0.0
    T = 1 / f0
    dt, n = T / 40, 15 * 40  # shorter run

    m, geo, st, load = _rod_model(L=L, H=H, E=E, rho=rho, F=F,
                                   alpha=alpha, beta=beta,
                                   study="time")
    load.props["Fx"] = f"{F!r}*sin(2*pi*{f0!r}[Hz]*t)"
    B.study_steps(st)[0].props.update(times=f"range(0,{dt!r},{n * dt!r})")

    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_qual_rod_tr_"), elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)

    times = np.array([fr.value for fr in sol.frames])
    assert len(times) == n

    # Exact harmonic response at f0 with mass damping alpha
    w = 2 * math.pi * f0
    Es = E * (1 + 1j * w * beta)  # beta=0
    k = cmath.sqrt(rho * (w * w - 1j * w * alpha) / Es)
    uh = F * cmath.tan(k * L) / (Es * k)

    # Last 25 steps should match steady-state
    last = np.arange(n - 25, n)
    got = np.array([float(np.mean(sol.evaluate("u", i)[np.isclose(sol.topo.nodes[:, 0] * sol.scale, L, atol=1e-9)])) for i in last])
    want = np.array([(uh * cmath.exp(1j * 2 * math.pi * f0 * t)).imag for t in times[last]])
    rel_err = np.max(np.abs(got - want)) / abs(uh)
    assert rel_err < 0.01, f"Transient CHANGED variant: rel_err={rel_err}"

    # Early transient must NOT match steady state
    early = np.array([float(np.mean(sol.evaluate("u", i)[np.isclose(sol.topo.nodes[:, 0] * sol.scale, L, atol=1e-9)])) for i in range(25)])
    want0 = np.array([(uh * cmath.exp(1j * 2 * math.pi * f0 * t)).imag for t in times[:25]])
    assert np.max(np.abs(early - want0)) > 0.05 * abs(uh)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])