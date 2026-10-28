"""Solid Mechanics in Time Dependent and Frequency Domain studies, with Rayleigh damping.

A bar in longitudinal vibration: plane stress, nu = 0, rollers on the long sides, fixed at x = 0 and loaded by a
traction F at x = L.  With C = alpha M + beta K the phasor tip displacement is exact,

    u(L) = F tan(k L) / (E* k),   E* = E (1 + i w beta),   k^2 = rho (w^2 - i w alpha) / E*,

the first eigenfrequency is c/(4L) with c = sqrt(E/rho), and a forced transient run settles on Im(u(L) e^{iwt})
for a load F sin(wt).
"""
import cmath
import math
import re
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.project import load, save
from elmerstudio.core.results import Solution
from elmerstudio.core.sif import SifError, build_sif
from elmerstudio.core.study_runner import StudyRunner

L, H, E, RHO, F = 1.0, 0.05, 200e9, 7850.0, 1e6
C = math.sqrt(E / RHO)
F1 = C / (4 * L)                                                    # 1261.9 Hz


def exact_tip(f, alpha=0.0, beta=0.0):
    w = 2 * math.pi * f
    Es = E * (1 + 1j * w * beta)
    k = cmath.sqrt(RHO * (w * w - 1j * w * alpha) / Es)
    return F * cmath.tan(k * L) / (Es * k)


def model(study, damping=None):
    m = B.new_model("2D", ["solid"], study)
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(H))
    geo = build_geometry(m, display=False)
    sm = m.component.child("physics.solid")
    sm.props["plane"] = "stress"
    sm.child("solid.lemm").props.update(E_src="user", E=f"{E!r}[Pa]", nu_src="user", nu="0",
                                        rho_src="user", rho=f"{RHO!r}[kg/m^3]")
    left, right = geo.select("boundary", xmax=0), geo.select("boundary", xmin=L)
    sides = geo.select("boundary", ymax=0) + geo.select("boundary", ymin=H)
    B.add_feature(m, sm, "solid.fixed", left)
    B.add_feature(m, sm, "solid.roller", sides)
    load = B.add_feature(m, sm, "solid.bload", right, Fx=str(F))
    damp = None
    if damping is not None:
        damp = B.add_feature(m, sm, "solid.damp", geo.select("domain"), alpha=str(damping[0]), beta=str(damping[1]))
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    for b, n in ((sides, 40), (left + right, 2)):
        B.add_feature(m, mesh, "mesh.distribution", b, n=n)
    st = m.studies()[0]
    return m, geo, st, load, damp


def tip(sol, i, name="u"):
    x = sol.topo.nodes[:, 0] * sol.scale
    on = np.isclose(x, L, atol=1e-9)
    assert on.sum() >= 3
    return float(np.mean(sol.evaluate(name, i)[on]))


def run(m, geo, st, elmer):
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_sdyn_"), elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    return sol


ALPHA, BETA = 2 * 0.05 * 2 * math.pi * F1, 0.02 / (2 * math.pi * F1)   # zeta = 0.05 + 0.01 at the first mode
FREQS = [1.0, 0.5 * F1, 0.9 * F1, F1, 1.1 * F1, 2.0 * F1]


def test_frequency_response_matches_the_exact_phasor(elmer):
    m, geo, st, _load, _damp = model("freq", damping=(ALPHA, BETA))
    B.study_steps(st)[0].props["freqs"] = " ".join(f"{f!r}" for f in FREQS)
    text = build_sif(m, st, B.study_steps(st)[0], geo, 100).text
    assert "Harmonic Analysis = Logical True" in text and "Rayleigh Damping Alpha" in text
    sol = run(m, geo, st, elmer)
    assert [fr.value for fr in sol.frames] == pytest.approx(FREQS)
    names = {a for a, *_ in sol.available(0)}
    assert {"solid.dispAmp", "solid.uim"} <= names
    for i, f in enumerate(FREQS):
        u = complex(tip(sol, i, "u"), tip(sol, i, "solid.uim"))
        ex = exact_tip(f, ALPHA, BETA)
        assert abs(u - ex) <= 2e-4 * abs(ex), (f, u, ex)
        assert tip(sol, i, "solid.dispAmp") == pytest.approx(abs(ex), rel=2e-4)
    assert tip(sol, 0, "u") == pytest.approx(F * L / E, rel=1e-4)            # quasi-static limit
    # at resonance the response is 90 degrees out of phase with the load: almost all imaginary
    i1 = FREQS.index(F1)
    assert abs(tip(sol, i1, "u")) < 0.05 * abs(tip(sol, i1, "solid.uim"))


def test_damping_is_what_bounds_the_resonance(elmer):
    """Control: the same sweep without the Damping node follows the undamped phasor and not the damped one."""
    m, geo, st, _load, _damp = model("freq")
    fs = [0.9 * F1, 1.1 * F1]
    B.study_steps(st)[0].props["freqs"] = " ".join(f"{f!r}" for f in fs)
    assert "Rayleigh" not in build_sif(m, st, B.study_steps(st)[0], geo, 100).text
    sol = run(m, geo, st, elmer)
    for i, f in enumerate(fs):
        u = tip(sol, i, "u")
        assert abs(tip(sol, i, "solid.uim")) < 1e-9 * abs(u)                  # undamped: real response
        assert u == pytest.approx(exact_tip(f).real, rel=2e-4)
        assert abs(u - exact_tip(f, ALPHA, BETA).real) > 0.05 * abs(u)


def test_eigenfrequency_agrees_and_ignores_damping(elmer):
    m, geo, st, _load, _damp = model("eigen", damping=(ALPHA, BETA))
    B.study_steps(st)[0].props["neig"] = 2
    assert "Rayleigh" not in build_sif(m, st, B.study_steps(st)[0], geo, 100).text
    sol = run(m, geo, st, elmer)
    assert sol.frames[0].value == pytest.approx(F1, rel=1e-4)
    assert sol.frames[1].value == pytest.approx(3 * F1, rel=1e-3)


def test_forced_transient_settles_on_the_harmonic_response(elmer):
    """Load F sin(w t) at half the first eigenfrequency with mass damping (decay rate alpha/2 for every mode):
    after the start-up has decayed the tip follows Im(u_hat e^{iwt}) of the frequency-domain solution."""
    f0, alpha = 0.5 * F1, 2 * 0.1 * 2 * math.pi * F1
    T = 1 / f0
    m, geo, st, load, _damp = model("time", damping=(alpha, 0.0))
    load.props["Fx"] = f"{F!r}*sin(2*pi*{f0!r}[Hz]*t)"
    dt, n = T / 50, 12 * 50
    B.study_steps(st)[0].props.update(times=f"range(0,{dt!r},{n * dt!r})")
    text = build_sif(m, st, B.study_steps(st)[0], geo, 100).text
    assert re.search(r'Timestepping Method = String "?Bossak', text)
    sol = run(m, geo, st, elmer)
    times = np.array([fr.value for fr in sol.frames])
    assert len(times) == n
    uh = exact_tip(f0, alpha, 0.0)
    last = np.arange(n - 50, n)
    got = np.array([tip(sol, i, "u") for i in last])
    want = np.array([(uh * cmath.exp(1j * 2 * math.pi * f0 * t)).imag for t in times[last]])
    assert np.max(np.abs(got - want)) < 0.01 * abs(uh), np.max(np.abs(got - want)) / abs(uh)
    # control: the start-up transient is not the steady state (the comparison above can fail)
    early = np.array([tip(sol, i, "u") for i in range(50)])
    want0 = np.array([(uh * cmath.exp(1j * 2 * math.pi * f0 * t)).imag for t in times[:50]])
    assert np.max(np.abs(early - want0)) > 0.05 * abs(uh)


def test_rollers_are_cartesian_where_elmer_mishandles_them():
    """Elmer's normal-tangential constraint gives a rod on rollers 1306 Hz for the exact 1262 Hz and leaves a
    harmonic solution in rotated components; on axis-aligned boundaries the frequency and eigenfrequency SIFs
    constrain the Cartesian component instead.  Stationary and transient keep the rotated form (correct there),
    and an inclined roller in a frequency study is refused rather than solved wrongly."""
    for kind, nt in (("freq", False), ("eigen", False), ("stationary", True), ("time", True)):
        m, geo, st, _l, _d = model(kind)
        text = build_sif(m, st, B.study_steps(st)[0], geo, 100).text
        assert ("Normal-Tangential Displacement" in text) is nt, kind
        if not nt:
            assert "Displacement 2 = Real 0.0" in text
    m = B.new_model("2D", ["solid"], "freq")
    m.create("geom.polygon", m.component.child("geometry"), index=0, xs="0 1 0", ys="0 0 1")
    geo = build_geometry(m, display=False)
    sm = m.component.child("physics.solid")
    sm.child("solid.lemm").props.update(E_src="user", E="1e9", nu_src="user", nu="0.3", rho_src="user", rho="1000")
    hyp = [b for b, bb in geo.bbox_of["boundary"].items() if bb[3] - bb[0] > 0.5 and bb[4] - bb[1] > 0.5]
    B.add_feature(m, sm, "solid.roller", hyp)
    with pytest.raises(SifError, match="perpendicular to the x, y or z axis"):
        build_sif(m, m.studies()[0], B.study_steps(m.studies()[0])[0], geo, 100)
    # an eigenfrequency study keeps solving such a model as before, with a warning
    eig = B.add_study(m, "eigen")
    res = build_sif(m, eig, B.study_steps(eig)[0], geo, 100)
    assert "Normal-Tangential Displacement" in res.text
    assert any("inclined or curved" in w for w in res.meta["warnings"])


def test_damping_node_persists_and_is_refused_where_meaningless(tmp_path):
    m, geo, st, _load, damp = model("freq", damping=(12.5, 3e-6))
    path = tmp_path / "bar.esm"
    save(m, str(path), None)
    m2, _work = load(str(path))
    d2 = m2.by_tag(damp.tag)
    assert d2.kind == "solid.damp" and d2.get("alpha") == "12.5" and d2.get("beta") == "3e-06"
    assert d2.selection.entities == damp.selection.entities
    # a stationary study writes no damping (nothing to damp); an undefined parameter is an error, not a zero
    m3, geo3, st3, _l, d3 = model("stationary", damping=(1.0, 0.0))
    assert "Rayleigh" not in build_sif(m3, st3, B.study_steps(st3)[0], geo3, 100).text
    m4, geo4, st4, _l, d4 = model("freq", damping=("not_a_param", 0.0))
    with pytest.raises((SifError, ValueError)):
        build_sif(m4, st4, B.study_steps(st4)[0], geo4, 100)
