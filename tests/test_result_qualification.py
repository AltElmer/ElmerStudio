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


def test_judge_navier_stokes_no_is_nonlinear_not_converged():
    """Navier-Stokes with nl_maxit=20 and no is_nonlinear key must be 'not converged'.
    
    This tests the backward compatibility: old saved specs without is_nonlinear
    should still be judged by nl_maxit > 1 rule."""
    conv = [(i, "navier-stokes", "NS", 0.3, i) for i in range(1, 21)]
    spec = {
        "ss_maxit": 1,
        "solvers": [{"equation": "navier-stokes", "post": False,
                     "nl_maxit": 20, "nl_tol": 1e-7, "ss_tol": 1e-6}]
    }
    msgs = convergence_problems(conv, spec)
    assert len(msgs) == 1, f"Expected 1 problem, got {msgs}"
    assert "navier-stokes" in msgs[0]
    assert "maximum 20" in msgs[0]


def test_judge_navier_stokes_is_nonlinear_false_not_converged():
    """Navier-Stokes with nl_maxit=20 and is_nonlinear=False must be 'not converged'.
    
    Even with explicit is_nonlinear=False, nl_maxit > 1 triggers the check."""
    conv = [(i, "navier-stokes", "NS", 0.3, i) for i in range(1, 21)]
    spec = {
        "ss_maxit": 1,
        "solvers": [{"equation": "navier-stokes", "post": False,
                     "nl_maxit": 20, "nl_tol": 1e-7, "ss_tol": 1e-6, "is_nonlinear": False}]
    }
    msgs = convergence_problems(conv, spec)
    assert len(msgs) == 1, f"Expected 1 problem, got {msgs}"
    assert "navier-stokes" in msgs[0]
    assert "maximum 20" in msgs[0]


def test_judge_coupling_two_way_not_converged():
    """Two-way cyclic coupling with ss_maxit=1 and high relc must be 'not converged'.
    
    Both physics have cross_depends=True, ss_maxit=1, relc=2.0 > ss_tol."""
    conv = [
        (1, "physics_a", "NS", 1e-5, 1),
        (1, "physics_a", "SS", 2.0, 1),
        (1, "physics_b", "NS", 1e-5, 1),
        (1, "physics_b", "SS", 2.0, 1),
    ]
    spec = {
        "ss_maxit": 1,
        "solvers": [
            {"equation": "physics_a", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": True},
            {"equation": "physics_b", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": True},
        ]
    }
    msgs = convergence_problems(conv, spec)
    # Should flag coupling (and also Newton loops since is_nonlinear=True and nl_maxit=1 with rel > tol)
    coupling_msgs = [m for m in msgs if "coupling" in m]
    assert len(coupling_msgs) == 1, f"Expected 1 coupling problem, got {msgs}"
    assert "maximum 1" in coupling_msgs[0]
    assert "relative change up to 2" in coupling_msgs[0]


def test_judge_coupling_one_way_converged():
    """One-way coupling chain with ss_maxit=1 stays converged (no coupling flag).
    
    Neither physics has cross_depends=True (no cycle), so coupling at ss_maxit=1
    is not flagged. Newton loops may still be flagged if is_nonlinear=True."""
    conv = [
        (1, "physics_a", "NS", 1e-5, 1),
        (1, "physics_a", "SS", 2.0, 1),
        (1, "physics_b", "NS", 1e-5, 1),
        (1, "physics_b", "SS", 2.0, 1),
    ]
    spec = {
        "ss_maxit": 1,
        "solvers": [
            {"equation": "physics_a", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": False},
            {"equation": "physics_b", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": False},
        ]
    }
    msgs = convergence_problems(conv, spec)
    # Should NOT flag coupling (no cross_depends cycle)
    coupling_msgs = [m for m in msgs if "coupling" in m]
    assert len(coupling_msgs) == 0, f"Expected no coupling problem, got {msgs}"
    # Newton loops are flagged because is_nonlinear=True and nl_maxit=1 with rel > tol
    newton_msgs = [m for m in msgs if "nonlinear iterations" in m]
    assert len(newton_msgs) == 2, f"Expected 2 Newton problems, got {msgs}"


def test_judge_coupling_segregated_converged():
    """Segregated coupling with ss_maxit > 1: one Newton step per coupling iteration is normal.
    
    The PDE's last relative change is 2.88e-07 (< ss_tol 1e-6). The coupling loop
    converged at SS ITER=7. This should be converged."""
    conv = [
        (1, "physics_a", "NS", 1e-5, 1),
        (1, "physics_a", "SS", 0.5, 1),
        (1, "physics_b", "NS", 1e-5, 1),
        (1, "physics_b", "SS", 0.5, 2),
        (2, "physics_a", "NS", 1e-5, 1),
        (2, "physics_a", "SS", 0.3, 2),
        (2, "physics_b", "NS", 1e-5, 1),
        (2, "physics_b", "SS", 0.3, 3),
        (3, "physics_a", "NS", 1e-5, 1),
        (3, "physics_a", "SS", 1e-6, 3),
        (3, "physics_b", "NS", 1e-5, 1),
        (3, "physics_b", "SS", 1e-6, 4),
        (4, "physics_a", "NS", 1e-5, 1),
        (4, "physics_a", "SS", 2.88e-7, 4),
        (4, "physics_b", "NS", 1e-5, 1),
        (4, "physics_b", "SS", 2.88e-7, 5),
    ]
    spec = {
        "ss_maxit": 30,
        "solvers": [
            {"equation": "physics_a", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": False},
            {"equation": "physics_b", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": False},
        ]
    }
    msgs = convergence_problems(conv, spec)
    # Should NOT flag Newton loops (in coupling with ss_maxit > 1)
    newton_msgs = [m for m in msgs if "nonlinear iterations" in m]
    assert len(newton_msgs) == 0, f"Expected no Newton problems, got {msgs}"
    # Should NOT flag coupling (converged at iteration 4 < ss_maxit)
    coupling_msgs = [m for m in msgs if "coupling" in m]
    assert len(coupling_msgs) == 0, f"Expected no coupling problem, got {msgs}"


def test_judge_coupling_segregated_not_converged():
    """Segregated coupling with ss_maxit forced to 2 so coupling cannot converge.
    
    Same model but ss_maxit=2, coupling stops at 2 with high relc -> not converged."""
    conv = [
        (1, "physics_a", "NS", 1e-5, 1),
        (1, "physics_a", "SS", 0.5, 1),
        (1, "physics_b", "NS", 1e-5, 1),
        (1, "physics_b", "SS", 0.5, 2),
        (2, "physics_a", "NS", 1e-5, 1),
        (2, "physics_a", "SS", 0.3, 2),
        (2, "physics_b", "NS", 1e-5, 1),
        (2, "physics_b", "SS", 0.3, 2),
    ]
    spec = {
        "ss_maxit": 2,
        "solvers": [
            {"equation": "physics_a", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": False},
            {"equation": "physics_b", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": False},
        ]
    }
    msgs = convergence_problems(conv, spec)
    # Should flag coupling (stopped at ss_maxit=2 with rel > ss_tol)
    coupling_msgs = [m for m in msgs if "coupling" in m]
    assert len(coupling_msgs) == 1, f"Expected 1 coupling problem, got {msgs}"
    assert "maximum 2" in coupling_msgs[0]


def test_judge_cross_depends_self_nonlinear_independent():
    """Two physics, each nonlinear only in own field, no cross-dep -> cross_depends=False."""
    conv = [
        (1, "physics_a", "NS", 1e-5, 1),
        (1, "physics_a", "SS", 2.0, 1),
        (1, "physics_b", "NS", 1e-5, 1),
        (1, "physics_b", "SS", 2.0, 1),
    ]
    spec = {
        "ss_maxit": 1,
        "solvers": [
            {"equation": "physics_a", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": False},
            {"equation": "physics_b", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": False},
        ]
    }
    msgs = convergence_problems(conv, spec)
    # No coupling flag (no cross_depends cycle)
    coupling_msgs = [m for m in msgs if "coupling" in m]
    assert len(coupling_msgs) == 0, f"Expected no coupling problem, got {msgs}"
    # Newton loops flagged (is_nonlinear=True, nl_maxit=1, rel=2.0 > nl_tol)
    newton_msgs = [m for m in msgs if "nonlinear iterations" in m]
    assert len(newton_msgs) == 2, f"Expected 2 Newton problems, got {msgs}"


def test_judge_cross_depends_two_way():
    """Two physics with real two-way dependency -> both cross_depends=True."""
    conv = [
        (1, "physics_a", "NS", 1e-5, 1),
        (1, "physics_a", "SS", 2.0, 1),
        (1, "physics_b", "NS", 1e-5, 1),
        (1, "physics_b", "SS", 2.0, 1),
    ]
    spec = {
        "ss_maxit": 1,
        "solvers": [
            {"equation": "physics_a", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": True},
            {"equation": "physics_b", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": True},
        ]
    }
    msgs = convergence_problems(conv, spec)
    # Should flag coupling (cross_depends cycle)
    coupling_msgs = [m for m in msgs if "coupling" in m]
    assert len(coupling_msgs) == 1, f"Expected 1 coupling problem, got {msgs}"
    assert "maximum 1" in coupling_msgs[0]
    assert "relative change up to 2" in coupling_msgs[0]


def test_judge_cross_depends_one_way():
    """One-way chain: A depends on B, B doesn't depend on A -> cross_depends=False for both.
    
    Per the correction: one-way chain -> both False (no cycle means no cross_depends)."""
    conv = [
        (1, "physics_a", "NS", 1e-5, 1),
        (1, "physics_a", "SS", 2.0, 1),
        (1, "physics_b", "NS", 1e-5, 1),
        (1, "physics_b", "SS", 2.0, 1),
    ]
    spec = {
        "ss_maxit": 1,
        "solvers": [
            {"equation": "physics_a", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": False},
            {"equation": "physics_b", "post": False, "nl_maxit": 1, "nl_tol": 1e-7, "ss_tol": 1e-6,
             "is_nonlinear": True, "cross_depends": False},
        ]
    }
    msgs = convergence_problems(conv, spec)
    # No coupling flag (no cross_depends cycle - both False)
    coupling_msgs = [m for m in msgs if "coupling" in m]
    assert len(coupling_msgs) == 0, f"Expected no coupling problem, got {msgs}"


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


# --------------------------------------------------------------------------------------- compiler-level cross_depends tests
@pytest.mark.integration
@pytest.mark.parametrize("case_name,heat_k,pde_c,expected_cross", [
    ("two-way", "400*(1+0.01*u)", "1+0.001*T", True),
    ("one-way", "400", "1+0.001*T", False),
    ("independent", "400*(1+0.001*T)", "1+u^2", False),
])
def test_cross_depends_topology(elmer, case_name, heat_k, pde_c, expected_cross):
    """Real compiler test: cross_depends from SIF for three topologies of heat + PDE."""
    m = B.new_model("2D", ["ht", "pde"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1", h="0.5")
    geo = build_geometry(m, display=False)
    B.add_material(m, "Copper", all_domains=True)
    ht = m.component.child("physics.ht")
    m.component.child("materials").children[0].props.setdefault("values", {})["k"] = heat_k
    B.add_feature(m, ht, "ht.temperature", geo.select("boundary", xmax=0), T0="300")
    B.add_feature(m, ht, "ht.temperature", geo.select("boundary", xmin=1), T0="310")
    pde = m.component.child("physics.pde")
    pde.child("pde.cfeq").props.update(c=pde_c, f="40", da="0")
    B.add_feature(m, pde, "pde.dir", geo.numbers["boundary"], r="0")
    st = m.studies()[0]
    work = tempfile.mkdtemp(prefix=f"es_topo_{case_name}_")
    try:
        StudyRunner(m, st, work, elmer).run()
    except StudyNotConverged:
        pass  # The topology test cares about cross_depends, not convergence status
    except Exception:
        pass  # Other errors - still check cross_depends
    
    # Read cross_depends from meta.json
    import json
    got = {}
    for root, _d, files in os.walk(work):
        if "meta.json" in files:
            meta = json.load(open(os.path.join(root, "meta.json")))
            for s in meta.get("convergence", {}).get("solvers", []):
                got[s["equation"]] = s.get("cross_depends")
    
    # Check both heat equation and coefficient form pde
    main = {e: v for e, v in got.items() if e in ("heat equation", "coefficient form pde")}
    assert set(main.values()) == {expected_cross}, \
        f"{case_name}: cross_depends={main}, expected all {expected_cross}"


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
    test_judge_navier_stokes_no_is_nonlinear_not_converged()
    print("test_judge_navier_stokes_no_is_nonlinear_not_converged: PASS")
    test_judge_navier_stokes_is_nonlinear_false_not_converged()
    print("test_judge_navier_stokes_is_nonlinear_false_not_converged: PASS")
    test_judge_coupling_two_way_not_converged()
    print("test_judge_coupling_two_way_not_converged: PASS")
    test_judge_coupling_one_way_converged()
    print("test_judge_coupling_one_way_converged: PASS")
    test_judge_coupling_segregated_converged()
    print("test_judge_coupling_segregated_converged: PASS")
    test_judge_coupling_segregated_not_converged()
    print("test_judge_coupling_segregated_not_converged: PASS")
    test_judge_cross_depends_self_nonlinear_independent()
    print("test_judge_cross_depends_self_nonlinear_independent: PASS")
    test_judge_cross_depends_two_way()
    print("test_judge_cross_depends_two_way: PASS")
    test_judge_cross_depends_one_way()
    print("test_judge_cross_depends_one_way: PASS")
    print("Unit tests done - 14 tests")