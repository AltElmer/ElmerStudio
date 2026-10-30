"""State contract tests for the reference corpus.

Each corpus state must produce a JSON record with explicit state contracts.
With Elmer absent, solved states must be "unavailable", not "solved".
Global Evaluation uses "exprs" (plural table), not "expr" (singular).
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_corpus(outdir: Path, extra_args=None) -> dict:
    """Run reference_corpus.py and return the index of records."""
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "reference_corpus.py"),
        str(outdir),
        "--sizes", "1366x768"
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env, timeout=180)

    index_file = outdir / "index.json"
    assert index_file.exists(), f"No index.json: {result.stdout} {result.stderr}"
    with open(index_file) as f:
        return json.load(f), result


def _get_record(records, state_key: str) -> dict:
    """Get the first record matching state_key."""
    for rec in records:
        if rec["state"] == state_key:
            return rec
    raise AssertionError(f"No record for state {state_key}")


# Module-scoped fixtures to run corpus once per mode

@pytest.fixture(scope="module")
def corpus_without_elmer():
    """Run corpus with --without-elmer flag."""
    with tempfile.TemporaryDirectory() as tmp:
        records, result = _run_corpus(Path(tmp), ["--without-elmer"])
        yield records, result


@pytest.fixture(scope="module")
def corpus_with_elmer():
    """Run corpus with Elmer (default, no flag)."""
    with tempfile.TemporaryDirectory() as tmp:
        records, result = _run_corpus(Path(tmp), [])
        yield records, result


@pytest.fixture(scope="module")
def corpus_without_elmer_require_solved():
    """Run corpus with --without-elmer --require-solved."""
    with tempfile.TemporaryDirectory() as tmp:
        records, result = _run_corpus(Path(tmp), ["--without-elmer", "--require-solved"])
        yield records, result


# Tests for --without-elmer mode

def test_without_elmer_state_08_unavailable(corpus_without_elmer):
    """State 08 must have availability == 'unavailable' when Elmer is absent."""
    records, _ = corpus_without_elmer
    rec = _get_record(records, "08_solved_convergence")
    assert rec["availability"] == "unavailable", f"State 08 availability: {rec['availability']}"
    assert rec["reason"] == "Elmer not available"
    check = next((c for c in rec["checks"] if c["name"] == "solver_ran"), None)
    assert check is not None
    assert check["passed"] is False


def test_without_elmer_state_11_unavailable(corpus_without_elmer):
    """State 11 must have availability == 'unavailable' when Elmer is absent."""
    records, _ = corpus_without_elmer
    rec = _get_record(records, "11_derived_table")
    assert rec["availability"] == "unavailable", f"State 11 availability: {rec['availability']}"
    assert rec["reason"] == "No solution computed"
    check = next((c for c in rec["checks"] if c["name"] == "solution_exists"), None)
    assert check is not None
    assert check["passed"] is False


def test_without_elmer_no_solved_in_status(corpus_without_elmer):
    """No status should claim 'solved' when Elmer is absent."""
    records, _ = corpus_without_elmer
    for rec in records:
        status_str = " ".join(rec.get("status", [])).lower()
        assert "solved" not in status_str, f"State {rec['state']} status claims 'solved': {rec['status']}"


def test_without_elmer_state_10_unavailable(corpus_without_elmer):
    """State 10 must have availability == 'unavailable' when Elmer is absent."""
    records, _ = corpus_without_elmer
    rec = _get_record(records, "10_results_navigation")
    assert rec["availability"] == "unavailable", f"State 10 availability: {rec['availability']}"
    assert rec["reason"] == "No solution computed"


# Tests for with Elmer mode

def test_with_elmer_state_11_nonzero_temperature(corpus_with_elmer):
    """State 11 must have nonzero_temperature check passed with value > 0 when Elmer is present."""
    records, _ = corpus_with_elmer
    rec = _get_record(records, "11_derived_table")
    # With Elmer, it should be available (or if not, at least have the check)
    check = next((c for c in rec["checks"] if c["name"] == "nonzero_temperature"), None)
    assert check is not None, f"nonzero_temperature check missing in {rec['checks']}"
    assert check["passed"] is True, f"nonzero_temperature check failed: {check}"
    assert "value" in check
    assert check["value"] > 0, f"Temperature value not > 0: {check['value']}"


# Tests for --require-solved flag

def test_require_solved_exits_nonzero(corpus_without_elmer_require_solved):
    """--without-elmer --require-solved must exit nonzero."""
    records, result = corpus_without_elmer_require_solved
    assert result.returncode != 0, f"Expected nonzero exit, got {result.returncode}"
    assert "REQUIRE_SOLVED failed" in result.stderr or "REQUIRE_SOLVED failed" in result.stdout


def test_require_solved_names_state_08(corpus_without_elmer_require_solved):
    """--require-solved output must name 08_solved_convergence."""
    records, result = corpus_without_elmer_require_solved
    output = result.stdout + result.stderr
    assert "08_solved_convergence" in output, f"Output doesn't name 08_solved_convergence: {output}"


def test_require_solved_no_png_for_state_08(corpus_without_elmer_require_solved):
    """--require-solved must not write PNG for unavailable states."""
    records, result = corpus_without_elmer_require_solved
    outdir = Path(result.args[0]) if hasattr(result, 'args') else None
    # Find the temp dir from the fixture - we need to check the output directory
    # The fixture yields records and result, but we need the temp dir
    # Let's check the records for image field
    rec = _get_record(records, "08_solved_convergence")
    assert rec["image"] is None, f"State 08 should have no image when require-solved: {rec['image']}"


# Test settle() helper

def test_settle_timeout_raises():
    """settle() with a stub window whose busy() is always True must raise TimeoutError."""
    from tools.reference_corpus import settle

    class StubApp:
        def processEvents(self):
            pass

    class StubWindow:
        def busy(self):
            return True  # Always busy

    app = StubApp()
    w = StubWindow()

    with pytest.raises(TimeoutError):
        settle(app, w, t=0.1)  # Short timeout for test


def test_settle_returns_when_cond_true():
    """settle() with busy=False and cond=True must return without error."""
    from tools.reference_corpus import settle

    class StubApp:
        def processEvents(self):
            pass

    class StubWindow:
        def busy(self):
            return False

    app = StubApp()
    w = StubWindow()

    # Should not raise
    settle(app, w, cond=lambda: True, t=0.1)


# Negative control: run on baseline (should fail because flags don't exist)
# This is documented but not run as a test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])