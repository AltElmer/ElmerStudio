"""Issue #6: OpenFOAM complete — RANS, axisymmetric wedge, MPI parallel dictionaries.

The dictionary tests need no OpenFOAM install.  The two live OpenFOAM solves use the ``openfoam``
fixture, which skips when OpenFOAM is not installed.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from elmerstudio.core import builder as B
from elmerstudio.core.builder import study_steps
from elmerstudio.core.backends.openfoam import dicts
from elmerstudio.core.backends.openfoam.blocks import analyse, block_mesh_dict
from elmerstudio.core.backends.openfoam.case import (FoamPlan, check_model, plan_case, write_case,
                                                     _decompose_par_dict, _turbulence_properties)
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner

H = 0.01
UB = 0.1
Lc = 0.1


def _channel_model(axi=False, turbulence="laminar", nproc=1):
    """2D channel H x Lc, uniform inlet Ub, pressure outlet, optional axisymmetric wedge."""
    m = B.new_model("2D", ["spf"], "stationary")
    m.global_defs.child("params").props["table"] = [
        {"name": "H", "expr": f"{H}[m]"}, {"name": "Ub", "expr": f"{UB}[m/s]"},
        {"name": "Lc", "expr": f"{Lc}[m]"}, {"name": "h_el", "expr": "H/8[m]"}]
    g = m.component.child("geometry")
    if axi:
        m.component.props["sdim"] = "2Daxi"
    m.create("geom.rectangle", g, index=0, w="Lc", h="H")
    B.add_material(m, "Water, liquid", all_domains=True)
    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="h_el", hmin="h_el/10")
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.spf")
    ph.props["turbulence_model"] = turbulence
    b_in = geo.select("boundary", xmax=0)
    b_out = geo.select("boundary", xmin=Lc)
    B.add_feature(m, ph, "spf.inlet", b_in, U0="Ub")
    B.add_feature(m, ph, "spf.outlet", b_out)
    st = m.studies()[0]
    st.props["backend"] = "openfoam"
    if nproc > 1:
        study_steps(st)[0].props["nproc"] = str(nproc)
    return m, geo, st


def _run_foam(m, st):
    wd = tempfile.mkdtemp(prefix="es_foam6_")
    return [Solution(f) for f in StudyRunner(m, st, wd, None).run()]


# --------------------------------------------------------------------- dictionaries (no OpenFOAM needed)
def test_turbulence_properties_ras_models():
    for tm, expected in (("k_epsilon", "kEpsilon"), ("k_omega_sst", "kOmegaSST")):
        P = FoamPlan("spf", "simpleFoam", False, {}, sdim=2, turbulence=tm)
        text = _turbulence_properties(P)
        assert "simulationType  RAS;" in text
        assert f"RASModel      {expected};" in text
        assert "turbulence    on;" in text
    P = FoamPlan("spf", "simpleFoam", False, {}, sdim=2, turbulence="laminar")
    assert "simulationType  laminar;" in _turbulence_properties(P)
    assert "RAS" not in _turbulence_properties(P)


def test_decompose_par_dict():
    P = FoamPlan("spf", "simpleFoam", False, {}, sdim=2, turbulence="laminar", nproc=4)
    text = _decompose_par_dict(P)
    assert "numberOfSubdomains 4;" in text
    assert "method             scotch;" in text
    assert "n                (4 1 1);" in text
    P = FoamPlan("spf", "simpleFoam", False, {}, sdim=2, turbulence="laminar", nproc=2)
    assert "method             simple;" in _decompose_par_dict(P)


# --------------------------------------------------------------------- geometry / mesh
def test_axisymmetric_wedge_blockmesh_dict():
    m, _g, _st = _channel_model(axi=True)
    layout = analyse(m)
    assert layout.wedge is True
    text = block_mesh_dict(layout, {})
    assert text.count("type wedge;") == 2
    assert "wedgeLower" in text and "wedgeUpper" in text
    assert "frontAndBack" not in text
    assert "empty;" not in text


def test_plan_case_accepts_axisymmetric():
    m, _g, st = _channel_model(axi=True, turbulence="k_epsilon")
    assert check_model(m, st) is None
    layout = analyse(m)
    step = study_steps(st)[0]
    P = plan_case(m, st, step, layout)
    assert P.turbulence == "k_epsilon"
    assert P.nproc == 1


def test_plan_case_writes_decompose_par_dict(tmp_path):
    m, _g, st = _channel_model(turbulence="k_omega_sst", nproc=4)
    layout = analyse(m)
    step = study_steps(st)[0]
    P = plan_case(m, st, step, layout)
    assert P.turbulence == "k_omega_sst"
    assert P.nproc == 4
    case = str(tmp_path / "case")
    write_case(P, layout, case)
    tp = dicts.read_text(os.path.join(case, "constant", "turbulenceProperties"))
    assert "simulationType  RAS;" in tp and "RASModel      kOmegaSST;" in tp
    dp = dicts.read_text(os.path.join(case, "system", "decomposeParDict"))
    assert "numberOfSubdomains 4;" in dp
    assert "method             scotch;" in dp


# --------------------------------------------------------------------- live OpenFOAM solves
def test_ras_channel_converges(openfoam):
    m, _g, st = _channel_model(turbulence="k_epsilon")
    (sol,) = _run_foam(m, st)
    assert sol.meta["status"] == "converged", sol.meta["diagnostics"]
    i = sol.frame_index("last")
    U = np.asarray(sol.names(i)["spf_u_vec"])
    assert np.linalg.norm(U, axis=1).max() > 1e-3, "RANS channel produced no flow"


def test_parallel_decompose_and_reconstruct(openfoam):
    m, _g, st = _channel_model(turbulence="laminar", nproc=2)
    (sol,) = _run_foam(m, st)
    assert sol.meta["status"] == "converged", sol.meta["diagnostics"]
    i = sol.frame_index("last")
    p = np.asarray(sol.names(i)["p"])
    assert np.abs(p).max() > 1e-3, "parallel channel produced no pressure drop"
