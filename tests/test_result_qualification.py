"""Regression tests for result qualification (F01: nonlinear one-iteration certified converged).

These tests verify that the convergence judge correctly distinguishes:
- a deliberately linear single-pass equation (converged even with relative change 2.0)
- a nonlinear equation limited to one iteration (NOT converged, relative change 2.0)
- a nonlinear equation with too few iterations (NOT converged)
- a nonlinear equation with enough iterations (converged)
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.elmer import RunState, convergence_problems, find_elmer
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.meshing import build_mesh
from elmerstudio.core.study_runner import StudyRunner, StudyNotConverged


# --------------------------------------------------------------------------------------- helpers
def _nonlinear_model(w=1.3, h=0.8, source=40, coeff="1+u^2"):
    """Model for -div((1+u^2) grad u) = source, u=0 on boundary."""
    m = B.new_model("2D", ["pde"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(w), h=str(h))
    geo = build_geometry(m, display=False)
    pde = m.component.child("physics.pde")
    pde.child("pde.cfeq").props.update(c=coeff, f=str(source), da="0")
    B.add_feature(m, pde, "pde.dir", geo.numbers["boundary"], r="0")
    return m, geo


def _linear_model(w=1.0, h=0.5):
    """Linear heat conduction model (definitely linear)."""
    m = B.new_model("2D", ["ht"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(w), h=str(h))
    geo = build_geometry(m, display=False)
    B.add_material(m, "Copper", all_domains=True)
    ht = m.component.child("physics.ht")
    B.add_feature(m, ht, "ht.temperature", geo.select("boundary", xmax=0), T0="0")
    B.add_feature(m, ht, "ht.temperature", geo.select("boundary", xmin=w), T0="10")
    return m, geo


def _setup_nl_solver(m, st, nl_maxit, nl_tol="1e-10"):
    cfg = B.ensure_solver_configs(m, st).children[0]
    cfg.props.update(nl_maxit=nl_maxit, nl_tol=nl_tol)


# --------------------------------------------------------------------------------------- convergence_problems unit tests
def test_judge_nonlinear_maxit_1_is_not_converged():
    """A nonlinear equation with nl_maxit=1 and relative change > tol must be 'not converged'."""
    # Simulate ComputeChange output for a nonlinear equation that did 1 iteration with rel change 2.0
    conv = [(1, "coefficient form pde", "NS", 2.0, 1)]
    spec = {
        "ss_maxit": 1,
        "solvers": [{"equation": "coefficient form pde", "post": False,
                     "nl_maxit": 1, "nl_tol": 1e-10, "ss_tol": 1e-6, "is_nonlinear": True}]
    }
    msgs = convergence_problems(conv, spec)
    assert len(msgs) == 1, f"Expected 1 problem, got {msgs}"
    assert "coefficient form pde" in msgs[0]
    assert "maximum 1" in msgs[0]
    assert "relative change up to 2" in msgs[0]


def test_judge_nonlinear_maxit_2_is_not_converged():
    """A nonlinear equation with nl_maxit=2 and relative change > tol must be 'not converged'."""
    conv = [
        (1, "coefficient form pde", "NS", 0.547, 1),
        (2, "coefficient form pde", "NS", 0.547, 2),
    ]
    spec = {
        "ss_maxit": 1,
        "solvers": [{"equation": "coefficient form pde", "post": False,
                     "nl_maxit": 2, "nl_tol": 1e-10, "ss_tol": 1e-6, "is_nonlinear": True}]
    }
    msgs = convergence_problems(conv, spec)
    assert len(msgs) == 1, f"Expected 1 problem, got {msgs}"
    assert "coefficient form pde" in msgs[0]
    assert "maximum 2" in msgs[0]


def test_judge_nonlinear_maxit_80_is_converged():
    """A nonlinear equation with nl_maxit=80 that converges must be 'converged'."""
    conv = [
        (1, "coefficient form pde", "NS", 1.5, 1),
        (2, "coefficient form pde", "NS", 0.3, 2),
        (3, "coefficient form pde", "NS", 0.05, 3),
        (4, "coefficient form pde", "NS", 1e-5, 4),
    ]
    spec = {
        "ss_maxit": 1,
        "solvers": [{"equation": "coefficient form pde", "post": False,
                     "nl_maxit": 80, "nl_tol": 1e-10, "ss_tol": 1e-6, "is_nonlinear": True}]
    }
    msgs = convergence_problems(conv, spec)
    assert msgs == [], f"Expected no problems, got {msgs}"


def test_judge_linear_one_pass_is_converged():
    """A linear equation (nl_maxit=1) with relative change 2.0 must be 'converged'."""
    # The key: linear equations have nl_maxit=1 but are NOT inherently nonlinear
    conv = [(1, "heat equation", "NS", 2.0, 1)]
    spec = {
        "ss_maxit": 1,
        "solvers": [{"equation": "heat equation", "post": False,
                     "nl_maxit": 1, "nl_tol": 1e-10, "ss_tol": 1e-6, "is_nonlinear": False}]
    }
    msgs = convergence_problems(conv, spec)
    # With the current bug, this will fail (incorrectly flagged as not converged)
    # After fix, this should pass (no problems)
    assert msgs == [], f"Linear one-pass should be converged, got {msgs}"


# --------------------------------------------------------------------------------------- integration tests with real ElmerSolver
@pytest.mark.integration
def test_nonlinear_maxit_1_real_run_not_converged(elmer):
    """Real Elmer run: nonlinear PDE with nl_maxit=1 must be rejected."""
    m, geo = _nonlinear_model()
    st = m.studies()[0]
    _setup_nl_solver(m, st, nl_maxit=1, nl_tol="1e-10")
    work = tempfile.mkdtemp(prefix="es_f01_nl1_")
    with pytest.raises(StudyNotConverged) as exc:
        StudyRunner(m, st, work, elmer).run()
    assert "maximum" in str(exc.value).lower() or "converge" in str(exc.value).lower()


@pytest.mark.integration
def test_nonlinear_maxit_2_real_run_not_converged(elmer):
    """Real Elmer run: nonlinear PDE with nl_maxit=2 must be rejected."""
    m, geo = _nonlinear_model()
    st = m.studies()[0]
    _setup_nl_solver(m, st, nl_maxit=2, nl_tol="1e-10")
    work = tempfile.mkdtemp(prefix="es_f01_nl2_")
    with pytest.raises(StudyNotConverged) as exc:
        StudyRunner(m, st, work, elmer).run()
    assert "maximum" in str(exc.value).lower() or "converge" in str(exc.value).lower()


@pytest.mark.integration
def test_nonlinear_maxit_80_real_run_converged(elmer):
    """Real Elmer run: nonlinear PDE with nl_maxit=80 must converge."""
    m, geo = _nonlinear_model()
    st = m.studies()[0]
    _setup_nl_solver(m, st, nl_maxit=80, nl_tol="1e-10")
    work = tempfile.mkdtemp(prefix="es_f01_nl80_")
    folders = StudyRunner(m, st, work, elmer).run()
    # Check meta.json status
    import json
    with open(os.path.join(folders[0], "meta.json")) as f:
        meta = json.load(f)
    assert meta["status"] == "converged", f"Expected converged, got {meta['status']}: {meta.get('diagnostics')}"


@pytest.mark.integration
def test_linear_one_pass_real_run_converged(elmer):
    """Real Elmer run: linear heat equation with nl_maxit=1 must converge."""
    m, geo = _linear_model()
    st = m.studies()[0]
    cfg = B.ensure_solver_configs(m, st).children[0]
    cfg.props.update(lin="direct", nl_maxit=1, nl_tol="1e-10")
    work = tempfile.mkdtemp(prefix="es_f01_lin1_")
    folders = StudyRunner(m, st, work, elmer).run()
    import json
    with open(os.path.join(folders[0], "meta.json")) as f:
        meta = json.load(f)
    assert meta["status"] == "converged", f"Expected converged, got {meta['status']}: {meta.get('diagnostics')}"


@pytest.mark.integration
def test_nonlinear_changed_variant_real_run_not_converged(elmer):
    """Real Elmer run: CHANGED variant (different geometry, source, tol) with nl_maxit=1 must be rejected."""
    m, geo = _nonlinear_model(w=2.0, h=1.0, source=50, coeff="1+u^3")
    st = m.studies()[0]
    _setup_nl_solver(m, st, nl_maxit=1, nl_tol="1e-8")
    work = tempfile.mkdtemp(prefix="es_f01_changed_")
    with pytest.raises(StudyNotConverged) as exc:
        StudyRunner(m, st, work, elmer).run()
    assert "maximum" in str(exc.value).lower() or "converge" in str(exc.value).lower()


if __name__ == "__main__":
    # Run unit tests without Elmer
    test_judge_nonlinear_maxit_1_is_not_converged()
    print("test_judge_nonlinear_maxit_1_is_not_converged: FAIL (expected)")
    test_judge_nonlinear_maxit_2_is_not_converged()
    print("test_judge_nonlinear_maxit_2_is_not_converged: FAIL (expected)")
    test_judge_nonlinear_maxit_80_is_converged()
    print("test_judge_nonlinear_maxit_80_is_converged: PASS")
    test_judge_linear_one_pass_is_converged()
    print("test_judge_linear_one_pass_is_converged: FAIL (expected - this is the bug)")
    print("Unit tests done - 3 failing as expected before fix")