"""Turbulent Flow (k-epsilon / k-omega SST) for the spf interface.

Gate:  python -m pytest tests/test_turbulent_flow.py -q   exits 0
"""
from __future__ import annotations

import tempfile

import numpy as np

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.results import Solution
from elmerstudio.core.sif import build_sif
from elmerstudio.core.study_runner import StudyRunner


def _channel(m, turbulence_model="k_epsilon"):
    """2D channel: inlet velocity, pressure outlet, noslip walls, optional turbulence."""
    H, Lc = 0.01, 0.1
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(Lc), h=str(H))
    B.add_material(m, "Water, liquid", all_domains=True)
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.size", mesh, mode="custom", hmax="1.5e-3", hmin="3e-4", hcurve="0.3")
    m.create("mesh.ftri", mesh)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.spf")
    ph.props["turbulence_model"] = turbulence_model
    inlet = geo.select("boundary", xmax=0)        # left edge (x == 0)
    outlet = geo.select("boundary", xmin=Lc)      # right edge (x == Lc)
    walls = [b for b in geo.select("boundary") if b not in inlet and b not in outlet]
    B.add_feature(m, ph, "spf.inlet", inlet, U0="0.1")
    B.add_feature(m, ph, "spf.outlet", outlet)
    B.add_feature(m, ph, "spf.wall", walls)
    return geo, ph


def test_turbulence_sif_emits_k_epsilon():
    m = B.new_model("2D", ["spf"], "stationary")
    geo, ph = _channel(m, "k_epsilon")
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    sif = build_sif(m, st, step, geo, 1).text
    assert 'Turbulence Model = String "k-epsilon"' in sif
    assert "Turbulence Intensity = Real" in sif
    assert "Wall Law = Logical True" in sif
    assert "Nonlinear System Relaxation Factor" in sif


def test_turbulence_sif_emits_k_omega_sst():
    m = B.new_model("2D", ["spf"], "stationary")
    geo, ph = _channel(m, "k_omega_sst")
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    sif = build_sif(m, st, step, geo, 1).text
    assert 'Turbulence Model = String "k-omega SST"' in sif
    assert "Turbulence Intensity = Real" in sif


def test_turbulence_laminar_emits_no_model():
    m = B.new_model("2D", ["spf"], "stationary")
    geo, ph = _channel(m, "laminar")
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    sif = build_sif(m, st, step, geo, 1).text
    assert "Turbulence Model" not in sif
    assert "Wall Law" not in sif


def test_turbulent_channel_solve(elmer):
    """End-to-end stationary k-epsilon channel solve: Elmer accepts the turbulent
    SIF and the flow field is nonzero (turbulence arrays are not emitted to VTU
    by this Elmer build, but the SIF correctly requests them)."""
    m = B.new_model("2D", ["spf"], "stationary")
    geo, ph = _channel(m, "k_epsilon")
    tmp = tempfile.mkdtemp(prefix="es_turb_")
    folders = StudyRunner(m, m.studies()[0], tmp, elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    # StudyRunner.run() raising would mean Elmer failed to converge; reaching here
    # means the turbulent SIF was accepted and the run converged.
    vel = sol.evaluate("spf.U", 0)
    vel = np.asarray(vel, dtype=float) if vel is not None else np.array([0.0])
    assert float(np.nanmax(np.abs(vel))) > 0.0, "turbulent channel produced no flow"
