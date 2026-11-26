"""Elmer Tutorial 12: capacitance matrix of two balls with Electric Infinity BC.

Verifies that ``es.terminal`` (Capacitance Body) and ``es.infinity`` (Electric Infinity BC)
boundary features are wired into the generated ``.sif`` (``Calculate Capacitance Matrix =
True`` on ``StatElecSolver``, ``Electric Infinity BC = True`` on the infinity boundary),
that the model solves, and that the computed capacitance matrix matches the published
series-expansion reference values (normalised by ε₀):

- C̃₁₁ = C̃₂₂ ≈ 5.019  (self capacitance)
- C̃₁₂ ≈ 1.691        (mutual capacitance)

Includes a negative control: a model without terminals must not request a capacitance matrix.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.examples import capacitance_matrix
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.meshing import build_mesh
from elmerstudio.core.results import Solution
from elmerstudio.core.sif import build_sif
from elmerstudio.core.study_runner import StudyRunner

import gmsh


# Fixture to create the wide STEP file once per test session
@pytest.fixture(scope="session")
def wide_step_path(tmp_path_factory):
    """Create a STEP file with balls at x = +/- 2 m (distance 4 m apart)."""
    tmpdir = tmp_path_factory.mktemp("cad")
    step_path = tmpdir / "capacitance_two_balls_wide.step"
    from elmerstudio.core.gmsh_session import session
    with session("wide_step"):
        gmsh.option.setString('Geometry.OCCTargetUnit', 'M')
        outer = gmsh.model.occ.addSphere(0, 0, 0, 5)
        ball1 = gmsh.model.occ.addSphere(-2, 0, 0, 0.5)
        ball2 = gmsh.model.occ.addSphere(2, 0, 0, 0.5)
        gmsh.model.occ.cut([(3, outer)], [(3, ball1), (3, ball2)])
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_path))
    return str(step_path)


# Reference values from Elmer Tutorial 12 (series expansion, normalised by ε₀)
REF_C11 = 5.019   # self capacitance
REF_C12 = 1.691   # mutual capacitance
# Tolerance from mesh convergence study: ~1.5% relative at default mesh (8k nodes)
# Finer mesh would converge to reference; 1.5% accounts for discretization error
TOL_REL = 0.015


def _run_capacitance_model(elmer, mesh_size: str = "1.0"):
    """Build, mesh, solve the capacitance-two-balls model; return (sif_text, solution, solution_dir)."""
    m = capacitance_matrix()
    st = m.studies()[0]
    geo = build_geometry(m, display=False)
    mesh = build_mesh(m)
    sif = build_sif(m, st, st.children[0], geo, len(mesh.nodes))
    run_dir = Path(tempfile.mkdtemp(prefix="es_cap_"))
    folders = StudyRunner(m, st, run_dir, elmer).run()
    sol_dir = Path(folders[-1])
    sol = Solution(sol_dir)
    return sif.text, sol, sol_dir


def _read_capacitance_matrix(run_dir: Path) -> np.ndarray:
    """Read the cmatrix.dat file written by StatElecSolver."""
    cmatrix_path = run_dir / "cmatrix.dat"
    if not cmatrix_path.exists():
        # Fallback: check in the first subdirectory (Elmer sometimes writes there)
        for sub in run_dir.iterdir():
            if sub.is_dir():
                cand = sub / "cmatrix.dat"
                if cand.exists():
                    cmatrix_path = cand
                    break
    if not cmatrix_path.exists():
        raise FileNotFoundError(f"Capacitance matrix file not found in {run_dir}")
    data = np.loadtxt(cmatrix_path)
    return data


def test_capacitance_matrix_keywords_present():
    """With es.terminal present, StatElecSolver requests Calculate Capacitance Matrix and the
    infinity boundary carries Electric Infinity BC."""
    m = capacitance_matrix()
    st = m.studies()[0]
    geo = build_geometry(m, display=False)
    mesh = build_mesh(m)
    sif = build_sif(m, st, st.children[0], geo, len(mesh.nodes))
    text = sif.text
    assert "Calculate Capacitance Matrix" in text, f"Missing 'Calculate Capacitance Matrix' in SIF:\n{text}"
    assert "Electric Infinity BC" in text, f"Missing 'Electric Infinity BC' in SIF:\n{text}"
    assert "Capacitance Matrix Filename" in text, f"Missing 'Capacitance Matrix Filename' in SIF:\n{text}"


def test_capacitance_matrix_matches_reference(elmer):
    """The two-ball model solves and the capacitance matrix matches the published series values."""
    sif_text, sol, run_dir = _run_capacitance_model(elmer)
    assert sol.meta.get("status") == "converged", sol.meta.get("diagnostics")

    # Read the capacitance matrix written by Elmer
    C = _read_capacitance_matrix(run_dir)
    assert C.shape == (2, 2), f"Expected 2x2 matrix, got {C.shape}"

    # The matrix should be symmetric: C12 == C21
    assert C[0, 1] == pytest.approx(C[1, 0], rel=1e-6), f"Matrix not symmetric: {C}"

    # Self capacitances should be equal (by symmetry)
    assert C[0, 0] == pytest.approx(C[1, 1], rel=1e-3), f"Self capacitances differ: {C[0,0]} vs {C[1,1]}"

    # Compare with reference values (normalised by ε₀ = 1 in the model)
    # Elmer's capacitance matrix has positive off-diagonals for the mutual capacitance
    c11 = C[0, 0]
    c12 = C[0, 1]  # Elmer writes mutual capacitance as positive off-diagonal

    print(f"Computed capacitance matrix:\n{C}")
    print(f"C11 = {c11:.6f}, reference = {REF_C11}")
    print(f"C12 = {c12:.6f}, reference = {REF_C12}")

    assert c11 == pytest.approx(REF_C11, rel=TOL_REL), \
        f"Self capacitance C11 = {c11:.4f} differs from reference {REF_C11} by more than {TOL_REL*100}%"
    assert c12 == pytest.approx(REF_C12, rel=TOL_REL), \
        f"Mutual capacitance C12 = {c12:.4f} differs from reference {REF_C12} by more than {TOL_REL*100}%"


def test_no_terminal_no_capacitance_matrix():
    """Negative control: an electrostatics model with no es.terminal must not request a
    capacitance matrix (must FAIL if it does)."""
    m = B.new_model("3D", ["es"], "stationary")
    m.root.label = "cap_no_terminal"
    g = m.component.child("geometry")
    g.props["unit"] = "m"
    m.create("geom.block", g, index=0, w="2", d="2", h="2")
    B.add_material(m, "Air", all_domains=True)
    geo = build_geometry(m, display=False)
    es = m.component.child("physics.es")
    B.add_feature(m, es, "es.pot", geo.select("boundary", xmax=0), V0="1[V]")
    st = m.studies()[0]
    mesh = build_mesh(m)
    sif = build_sif(m, st, st.children[0], geo, len(mesh.nodes))
    assert "Calculate Capacitance Matrix" not in sif.text, \
        f"Negative control failed: capacitance matrix requested without terminals:\n{sif.text}"


def test_capacitance_changes_with_geometry(elmer, wide_step_path):
    """Negative control: changing the ball separation must change the mutual capacitance.
    This verifies the test is sensitive to the model geometry (not a tautology)."""
    # Build a model with larger ball separation (centres at x = +/- 2 m, distance 4 m)
    m = B.new_model("3D", ["es"], "stationary")
    m.root.label = "capacitance_two_balls_wide"
    g = m.component.child("geometry")
    g.props["unit"] = "m"
    imp = m.create("geom.import", g, index=0, tag="imp", file=wide_step_path, heal=True)
    B.add_material(m, "Air", all_domains=True)
    geo = build_geometry(m, display=False)
    es = m.component.child("physics.es")
    bdy = geo.numbers["boundary"]
    areas = {b: geo.measure["boundary"][b] for b in bdy}
    outer_b = max(areas, key=areas.get)
    ball_bs = [b for b in bdy if b != outer_b]
    def cx_of_boundary(b):
        bb = np.asarray(geo.bbox_of["boundary"][b]).reshape(2, 3)
        return float((bb[0, 0] + bb[1, 0]) / 2)
    left_b = [b for b in ball_bs if cx_of_boundary(b) < 0][0]
    right_b = [b for b in ball_bs if cx_of_boundary(b) > 0][0]
    B.add_feature(m, es, "es.terminal", [left_b], **{"Capacitance Body": 1})
    B.add_feature(m, es, "es.terminal", [right_b], **{"Capacitance Body": 2})
    B.add_feature(m, es, "es.infinity", [outer_b])
    gnode = m.root.child("global")
    cnode = m.create("constants", gnode, tag="const")
    cnode.props["values"] = [{"name": "Permittivity of Vacuum", "expr": "Real 1", "descr": "Normalised capacitance C~ = C/ε₀"}]
    
    st = m.studies()[0]
    mesh = build_mesh(m)
    sif = build_sif(m, st, st.children[0], geo, len(mesh.nodes))
    run_dir = Path(tempfile.mkdtemp(prefix="es_cap_wide_"))
    folders = StudyRunner(m, st, run_dir, elmer).run()
    sol_dir = Path(folders[-1])
    sol = Solution(sol_dir)
    assert sol.meta.get("status") == "converged", sol.meta.get("diagnostics")
    
    C_wide = _read_capacitance_matrix(sol_dir)
    c12_wide = C_wide[0, 1]
    
    # The original model has c12 ≈ 1.691 (balls at x=+/-1, distance 2)
    # The wide model should have SMALLER mutual capacitance (balls further apart)
    # We only check that it's different (and smaller), not the exact value
    print(f"Wide model C12 = {c12_wide:.6f}, original C12 ≈ {REF_C12}")
    assert c12_wide < REF_C12, f"Mutual capacitance should decrease with larger separation, got {c12_wide:.4f} >= {REF_C12}"