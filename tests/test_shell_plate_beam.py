"""Issue #2: shell (ShellSolver) and beam (BeamSolver3D) physics interfaces.

Verifies that the shell/plate and beam physics interfaces are registered, emit a
correct Elmer SIF (ShellSolver / BeamSolver3D + TimoshenkoSolver), and that a
cantilever beam solved end-to-end with Elmer matches analytical beam-deflection
theory (Euler-Bernoulli w = P L^3 / (3 E I), Timoshenko shear correction small
for a slender beam).

Note: ShellSolver / BeamSolver3D require lower-order shell/beam elements that the
current Gmsh meshers (6-node triangles, 9-node quads) do not emit
("CreateCurvedEdges: ... background elements are not yet supported"), so the
real solve below validates the deflection-theory pipeline with the proven
StressSolver on a slender 3D cantilever beam, whose deflection tends to the
analytical beam value as slenderness grows.
"""

import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.sif import build_sif
from elmerstudio.core.study_runner import StudyRunner
from elmerstudio.core.results import Solution
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.builder import add_feature


def _sif(m, st, geo, text_checks):
    step = B.study_steps(st)[0]
    text = build_sif(m, st, step, geo, 100).text
    for check in text_checks:
        assert check in text, f"missing {check!r} in SIF:\n{text}"
    return text


def _make_beam(m, L=1.0, b=0.05, h=0.05):
    """Slender 3D cantilever beam block, fixed at x=0, loaded at x=L."""
    comp = m.component
    geom = comp.child("geometry")
    m.create("geom.block", geom, index=0, w=str(L), d=str(b), h=str(h))
    m.create("geom.finalize", geom, tag="fin")
    geo = build_geometry(m, display=False)
    sm = m.component.child("physics.solid")
    sm.child("solid.lemm").props.update(
        E_src="user", E="200e9[Pa]", nu_src="user", nu="0.3",
        rho_src="user", rho="7850[kg/m^3]")
    left = geo.select("boundary", xmax=0)
    right = geo.select("boundary", xmin=L)
    add_feature(m, sm, "solid.fixed", left)
    add_feature(m, sm, "solid.bload", right, type="total", Ftz="1000")
    mesh = comp.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.size", mesh)
    m.create("mesh.ftet", mesh)
    return m, geo, right


def _tip_deflection(sol, L):
    """Mean out-of-plane (z) displacement of nodes at the loaded tip x=L."""
    x = sol.topo.nodes[:, 0] * sol.scale
    on = np.isclose(x, L, atol=1e-6)
    assert on.sum() >= 3, f"only {on.sum()} tip nodes found"
    w = sol.evaluate("w", 0)[on]
    return float(abs(w).mean())


def test_shell_sif_emits_shellsolver():
    m = B.new_model("3D", ["shell"], "stationary")
    comp = m.component
    geom = comp.child("geometry")
    m.create("geom.block", geom, index=0, w="1.0", d="0.1", h="0.05")
    m.create("geom.finalize", geom, tag="fin")
    geo = build_geometry(m, display=False)
    sm = comp.child("physics.shell")
    sm.child("shell.lemm").props.update(E_src="user", E="200e9[Pa]", nu_src="user", nu="0.3",
                                        rho_src="user", rho="7850[kg/m^3]", h="0.05[m]")
    left = geo.select("boundary", xmax=0)
    right = geo.select("boundary", xmin=1)
    add_feature(m, sm, "solid.fixed", left)
    add_feature(m, sm, "shell.pressure", right, p="1000[Pa]")
    st = m.studies()[0]
    _sif(m, st, geo, [
        "Procedure = \"ShellSolver\" \"ShellSolver\"",
        "Shell Thickness", "Youngs Modulus", "Poisson Ratio",
        "Variable = -dofs 3 Displacement",
    ])


def test_beam_sif_emits_beamsolver3d():
    m = B.new_model("3D", ["beam"], "stationary")
    comp = m.component
    geom = comp.child("geometry")
    m.create("geom.block", geom, index=0, w="1.0", d="0.05", h="0.05")
    m.create("geom.finalize", geom, tag="fin")
    geo = build_geometry(m, display=False)
    sm = comp.child("physics.beam")
    sm.child("beam.lemm").props.update(E_src="user", E="200e9[Pa]", G_src="user", G="77e9[Pa]",
                                       rho_src="user", rho="7850[kg/m^3]", A="0.0025[m^2]",
                                       I2="5.20833e-7[m^4]", I3="5.20833e-7[m^4]", Jt="1.04167e-6[m^4]")
    left = geo.select("boundary", xmax=0)
    right = geo.select("boundary", xmin=1)
    add_feature(m, sm, "beam.fixed", left)
    add_feature(m, sm, "beam.pointload", right, Fz="1000")
    st = m.studies()[0]
    _sif(m, st, geo, [
        "Procedure = \"BeamSolver3D\" \"TimoshenkoSolver\"",
        "Cross Section Area", "Shear Modulus", "Torsional Constant",
        "Theta 1",
    ])


@pytest.mark.slow
def test_cantilever_beam_deflection_matches_theory(elmer):
    """Tip deflection of a slender cantilever under end load P vs P L^3 / (3 E I)."""
    L, b, h, E, P = 1.0, 0.05, 0.05, 200e9, 1000.0
    I = b * h ** 3 / 12.0
    expected = P * L ** 3 / (3.0 * E * I)  # ~3.2 mm
    m = B.new_model("3D", ["solid"], "stationary")
    m, geo, _right = _make_beam(m, L, b, h)
    st = m.studies()[0]
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_beam_"), elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    got = _tip_deflection(sol, L)
    rel = abs(got - expected) / expected
    print(f"cantilever: FEM={got:.6e} m  theory={expected:.6e} m  rel={rel:.3%}")
    assert rel <= 0.10, (got, expected)
