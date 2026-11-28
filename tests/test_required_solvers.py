"""ELMERSTUDIO_TEST_REQUIRE: where a solver is installed on purpose, its tests must run (tests/conftest.py)."""
from __future__ import annotations

import os
import subprocess
import sys

from conftest import missing_required

HERE = os.path.dirname(os.path.abspath(__file__))


def test_missing_required_names_what_is_absent():
    solvers = {"here": (lambda: object(), "here not installed"), "gone": (lambda: None, "gone not installed")}
    assert missing_required("", solvers) == []
    assert missing_required("here", solvers) == []
    assert missing_required(" Here , gone,", solvers) == ["gone: gone not installed"]
    unknown = missing_required("elmr", solvers)
    assert len(unknown) == 1 and unknown[0].startswith("elmr: not a solver of the tests")


def _run(require):
    env = dict(os.environ, ELMERSTUDIO_TEST_REQUIRE=require, QT_QPA_PLATFORM="offscreen")
    return subprocess.run([sys.executable, "-m", "pytest", os.path.join(HERE, "test_core.py"), "-q", "-x",
                           "-p", "no:cacheprovider", "-k", "test_unit_expressions"],
                          env=env, capture_output=True, text=True, timeout=300)


def test_a_required_solver_that_is_absent_stops_the_run():
    r = _run("no-such-solver")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "ELMERSTUDIO_TEST_REQUIRE is not met" in r.stdout + r.stderr and "no-such-solver" in r.stdout + r.stderr


def test_without_requirement_the_run_goes_on():
    r = _run("")
    assert r.returncode == 0 and " passed" in r.stdout, r.stdout + r.stderr