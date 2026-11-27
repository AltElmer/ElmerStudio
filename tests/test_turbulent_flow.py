"""Turbulent Flow (k-epsilon) for the spf interface.

Gate:  python -m pytest tests/test_turbulent_flow.py -q   exits 0

Only the k-epsilon model is implemented (k-omega SST is not). Every keyword
asserted here is verified against Elmer's SOLVER.KEYWORDS and KESolver.F90.

The end-to-end runs drive ElmerSolver (a segregated k-epsilon channel at
U = 10 m/s, Re = 1e5). Segregated k-epsilon converges only tenths of a
percent per coupling iteration once wall functions are valid, so these tests
use Elmer-matching converging defaults (under-relaxation 0.75 on both
solvers, 2500 segregated iterations, coupling tolerance 1e-5) and each run
takes a couple of minutes.
"""
from __future__ import annotations

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


def _channel(m, turbulence_model="k_epsilon", U0="0.1", hmax="1.5e-3"):
    """2D channel: inlet velocity, pressure outlet, noslip walls.

    The k-epsilon model needs delta_w (wall distance) on every noslip wall
    and a turbulence intensity + length scale at the inlet; these are real
    Elmer requirements, not UI defaults (Node.get does not fall back to Prop
    defaults).

    For turbulent cases, initial k/epsilon are set consistent with the inlet
    turbulence (k = 1.5*(U*I)^2, eps = 0.09^0.75*k^1.5/L) so the segregated
    iteration doesn't waste thousands of steps ramping up from 1e-3.
    """
    H, Lc = 0.01, 0.1
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(Lc), h=str(H))
    B.add_material(m, "Water, liquid", all_domains=True)
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.size", mesh, mode="custom", hmax=hmax, hmin="3e-4", hcurve="0.3")
    m.create("mesh.ftri", mesh)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.spf")
    ph.props["turbulence_model"] = turbulence_model
    if turbulence_model != "laminar":
        # converging segregated defaults: k-epsilon's fixed point is unstable,
        # both solvers run under-relaxed at 0.75 (the Turbulence >
        # Under-relaxation factor Prop default) and the iteration budget is
        # in the thousands (Elmer's own references converge at 1e-4..1e-5)
        ph.props["turb_iters"] = "2500"
        # initial turbulence consistent with inlet: k = 1.5*(U*I)^2, eps = 0.09^0.75*k^1.5/L
        # U0 is string, turb_I=0.05, turb_L=5e-3
        U = float(U0)
        I = 0.05
        L = 5e-3
        k0 = 1.5 * (U * I) ** 2
        ep0 = (0.09 ** 0.75) * (k0 ** 1.5) / L
        init = next(f for f in ph.children if f.kind == "spf.init")
        init.props["k0"] = str(k0)
        init.props["ep0"] = str(ep0)
    inlet = geo.select("boundary", xmax=0)        # left edge (x == 0)
    outlet = geo.select("boundary", xmin=Lc)      # right edge (x == Lc)
    walls = [b for b in geo.select("boundary") if b not in inlet and b not in outlet]
    B.add_feature(m, ph, "spf.inlet", inlet, U0=U0, turb_I="0.05", turb_L="5e-3")
    B.add_feature(m, ph, "spf.outlet", outlet)
    B.add_feature(m, ph, "spf.wall", walls, delta_w="2e-4")
    return geo, ph


def _run_study(m, inst, tmp, coupling_tol="1e-5"):
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    # the study template defaults coupling_tol to 1e-6; Elmer's own turbulent
    # references converge at 1e-4..1e-5, and that is what the runner judges.
    # The looser 1e-5 here is the spf turbulence setup only (finding (a) of the
    # PR #22 review); all other physics keep the 1e-6 global default in sif.py.
    step.props["coupling_tol"] = coupling_tol
    # nonconv="stop": StudyRunner.run() raises StudyNotConverged when the
    # segregated iteration does not reach the coupling tolerance, so reaching
    # here means the turbulent channel converged (not merely ran to the iteration cap).
    st.props["nonconv"] = "stop"
    return StudyRunner(m, st, tmp, inst).run()


def test_turbulence_sif_emits_k_epsilon():
    m = B.new_model("2D", ["spf"], "stationary")
    geo, ph = _channel(m, "k_epsilon")
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    sif = build_sif(m, st, step, geo, 1).text
    # material: Viscosity Model = ke + KE constants (SOLVER.KEYWORDS Material:Real)
    assert 'Viscosity Model = String "ke"' in sif
    assert "KE SigmaK = Real 1.0" in sif
    assert "KE SigmaE = Real 1.3" in sif
    assert "KE C1 = Real 1.44" in sif
    assert "KE C2 = Real 1.92" in sif
    assert "KE Cmu = Real 0.09" in sif
    assert "KE Clip = Real 1e-06" in sif
    # segregated KESolver + FlowSolve under one Equation
    assert 'Procedure = "KESolver" "KESolver"' in sif
    assert "Variable = K-eps[Kinetic Energy:1 Kinetic Dissipation:1]" in sif
    # initial values + inlet turbulence (Kinetic Energy / Kinetic Dissipation)
    assert "Kinetic Energy = Real" in sif
    assert "Kinetic Dissipation = Real" in sif
    # wall functions (Wall Law / Boundary Layer Thickness are genuine Elmer keywords)
    assert "Wall Law = Logical True" in sif
    assert "Boundary Layer Thickness = Real 0.0002" in sif
    # under-relaxed segregated iteration
    assert "Nonlinear System Relaxation Factor" in sif
    # no fake keywords
    assert "Turbulence Model" not in sif
    assert "Turbulence Intensity" not in sif
    assert "Turbulence Length Scale" not in sif


def test_turbulence_laminar_emits_no_model():
    m = B.new_model("2D", ["spf"], "stationary")
    geo, ph = _channel(m, "laminar")
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    sif = build_sif(m, st, step, geo, 1).text
    assert "Viscosity Model" not in sif
    assert "KE " not in sif
    assert "Wall Law" not in sif
    assert "Kinetic Energy" not in sif


def test_turbulent_channel_solve(elmer):
    """End-to-end stationary k-epsilon channel: Elmer accepts the turbulent SIF,
    the segregated run converges to its coupling tolerance (StudyRunner.run()
    raises StudyNotConverged otherwise, so reaching here means converged), and
    turbulence results (k, epsilon) are written to the VTU output."""
    m = B.new_model("2D", ["spf"], "stationary")
    geo, ph = _channel(m, "k_epsilon", U0="10")
    tmp = tempfile.mkdtemp(prefix="es_turb_")
    folders = _run_study(m, elmer, tmp, "1e-5")
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    vel = np.asarray(sol.evaluate("spf.U", 0), dtype=float)
    assert float(np.nanmax(np.abs(vel))) > 0.0, "turbulent channel produced no flow"
    k = np.asarray(sol.evaluate("spf.k", 0), dtype=float)
    ep = np.asarray(sol.evaluate("spf.ep", 0), dtype=float)
    assert np.all(np.isfinite(k)) and float(np.nanmax(np.abs(k))) > 0.0
    assert np.all(np.isfinite(ep)) and float(np.nanmax(np.abs(ep))) > 0.0


def test_turb_relax_default_emits_verified_value():
    """With turb_relax unset, the SIF carries the verified-converging default
    (0.75), not the stale Prop default (0.5)."""
    m = B.new_model("2D", ["spf"], "stationary")
    geo, ph = _channel(m, "k_epsilon", U0="10")
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    sif = build_sif(m, st, step, geo, 1).text
    assert sif.count("Nonlinear System Relaxation Factor = Real 0.75") >= 2
    assert "Nonlinear System Relaxation Factor = Real 0.5" not in sif


@pytest.mark.parametrize("turb_relax", ["0.25", "0.5", "0.9"])
def test_turb_relax_drives_sif(turb_relax):
    """Changing turb_relax changes the under-relaxation emitted for BOTH the
    turbulence (KESolver) and the flow (FlowSolve) segregated solvers.

    Structural guard: on the unfixed code this test FAILS because relax=0.75
    was hardcoded in _spf_solvers and the declared turb_relax Prop was never
    read. After the fix the Prop drives both solvers, so changing it changes
    the emitted relaxation."""
    m = B.new_model("2D", ["spf"], "stationary")
    geo, ph = _channel(m, "k_epsilon", U0="10")
    ph.props["turb_relax"] = turb_relax
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    sif = build_sif(m, st, step, geo, 1).text
    assert sif.count(f"Nonlinear System Relaxation Factor = Real {turb_relax}") >= 2


def test_turbulence_model_feeds_momentum_negative_control():
    """Negative control: on identical channel geometry and Re, the only
    difference between the two builds is the turbulence model. Therefore the
    presence of "Viscosity Model = String ke" in the material -- which
    FlowSolve reads to build the turbulent viscosity of the momentum equation
    -- must be caused by the turbulence model alone. If turbulence did not
    feed momentum, the material would not carry Viscosity Model = ke and this
    test must fail."""
    sifs = {}
    for turbulence_model in ("k_epsilon", "laminar"):
        m = B.new_model("2D", ["spf"], "stationary")
        geo, ph = _channel(m, turbulence_model, U0="10")
        st = m.studies()[0]
        step = B.study_steps(st)[0]
        sifs[turbulence_model] = build_sif(m, st, step, geo, 1).text
    turbulent = sifs["k_epsilon"]
    laminar = sifs["laminar"]
    assert 'Viscosity Model = String "ke"' in turbulent
    assert "KE Cmu = Real 0.09" in turbulent
    assert "Viscosity Model" not in laminar
    assert "KE " not in laminar


def _log_law_error(sol, H, Lc, rho, mu):
    """Max relative deviation of the mean velocity from the log law,
    u+ = (1/kappa) ln y+ + B, evaluated in the log region away from inlet/outlet.
    Returns None when the mesh cannot host the check."""
    arrays = sol.arrays(0)
    if "velocity" not in arrays or "pressure" not in arrays:
        return None
    nodes = sol.topo.nodes
    x, y = nodes[:, 0], nodes[:, 1]
    u = np.asarray(arrays["velocity"])[:, 0]
    p = np.asarray(arrays["pressure"])
    # fully developed: pressure varies linearly in x; slope from the middle half
    mid = (x > 0.35 * Lc) & (x < 0.65 * Lc)
    if mid.sum() < 10:
        return None
    slope = float(np.polyfit(x[mid], p[mid], 1)[0])          # dp/dx
    tau_w = -slope * H / 2.0                                  # force balance
    if not np.isfinite(tau_w) or tau_w <= 0.0:
        return None
    u_tau = math.sqrt(tau_w / rho)
    ydist = np.minimum(y, H - y)
    yp = rho * u_tau * ydist / mu
    up = np.abs(u) / u_tau
    # log region: lower half only (avoid the centerline wake), middle of the
    # channel (away from inlet/outlet disturbances), y+ in the log band
    logregion = (yp > 30) & (yp < 700) & (y < 0.5 * H) & (x > 0.35 * Lc) & (x < 0.65 * Lc) & (up > 1.0)
    if logregion.sum() < 3:
        return None
    kappa, B_ = 0.41, 5.2
    up_log = np.log(yp[logregion]) / kappa + B_
    rel = np.abs(up[logregion] - up_log) / np.maximum(up_log, 1.0)
    return float(np.max(rel)), int(logregion.sum())


def _material(m):
    return m.component.child("materials").children[0]


def test_turbulent_channel_log_law(elmer):
    """Verification: fully developed turbulent channel vs the log law, with the
    tolerance taken from a mesh refinement run (coarse 44% -> fine 10% on this
    mesh, measured; refinement must improve the fit).  The k-epsilon constant
    Cmu=0.09 is wired into the material (KE Cmu = Real 0.09, asserted in
    test_turbulence_sif_emits_k_epsilon); a Cmu=0.18 mutation does NOT worsen
    the log-law fit on this metric (measured 6.5% vs 9.8%), so no
    Cmu-mutation failure is observable here and that claim is reduced to the
    refinement-supported tolerance below."""
    errors = []
    for hmax in ("1.2e-3", "6e-4"):
        m = B.new_model("2D", ["spf"], "stationary")
        geo, ph = _channel(m, "k_epsilon", U0="10", hmax=hmax)
        mat = _material(m)
        rho = float(mat.props.get("Density", 1000.0))
        mu = float(mat.props.get("Viscosity", 0.001))
        tmp = tempfile.mkdtemp(prefix="es_turb_")
        folders = _run_study(m, elmer, tmp, "1e-5")
        sol = Solution(folders[-1])
        bind_results(m, sol, geo)
        err = _log_law_error(sol, 0.01, 0.1, rho, mu)
        assert err is not None, f"mesh hmax={hmax} cannot host the log-law check"
        errors.append(err)
    (err_c, n_c), (err_f, n_f) = errors
    # refinement-supported tolerance: fine-mesh log-law error measured 9.8%,
    # acceptance 15% (~50% margin); refinement must improve (not degrade) the fit
    assert err_f < 0.15, f"fine-mesh log-law error {err_f:.3g} exceeds tolerance ({n_f} points)"
    assert err_f < err_c, f"log-law error {err_c:.3g} (coarse) -> {err_f:.3g} (fine): refinement broke down"


def test_turbulence_results_require_turbulence_interface(elmer):
    """Negative control: a laminar run has no k-epsilon solver, so no turbulence
    results exist in the output (requesting them must fail rather than invent
    numbers)."""
    m = B.new_model("2D", ["spf"], "stationary")
    geo, ph = _channel(m, "laminar", U0="0.1")
    tmp = tempfile.mkdtemp(prefix="es_turb_")
    folders = _run_study(m, elmer, tmp, "1e-5")
    sol = Solution(folders[-1])
    arrays = sol.arrays(0)
    assert "kinetic energy" not in arrays and "kinetic dissipation" not in arrays
