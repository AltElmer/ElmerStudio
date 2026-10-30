"""Tests for selftest report writer and packaging wrapper (JSON format, schema=1).

The packaging wrapper must reject:
- UI problems (ui=failed)
- Zero frames (frames=0)
- Child exit code != 0 after report
- Substring matches (frames=10 when looking for frames=1)
- Missing schema version
- Stale commit
- Invalid JSON
- Wrong schema version

It must accept only a healthy run with exact JSON report (schema=1, exact integer counts).
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the local packaging/build.py module
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("es_packaging_build", ROOT / "packaging" / "build.py")
build_pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_pkg)


def _run_smoke_with_report(report_content: str, child_exit: int) -> tuple[bool, str]:
    """Run smoke() with a mocked report; return (accepted, output)."""
    with tempfile.TemporaryDirectory() as tmp:
        DIST = Path(tmp)
        report_file = DIST / "selftest.txt"

        def fake_run(cmd, **kwargs):
            report_file.write_text(report_content)
            return subprocess.CompletedProcess(cmd, child_exit)

        with patch.object(build_pkg, "DIST", str(DIST)), \
             patch.object(build_pkg.subprocess, "run", fake_run), \
             patch.object(build_pkg.subprocess, "check_output", return_value="abc123"):
            try:
                build_pkg.smoke()
                return True, "accepted"
            except SystemExit as e:
                return False, str(e)


def _healthy_json_report(commit="abc123"):
    return json.dumps({
        "schema": 1,
        "commit": commit,
        "stages": {
            "geometry": "built",
            "mesh": "built",
            "compute": "converged",
            "plot": "rendered",
            "ui": "ok"
        },
        "counts": {
            "nodes": 20,
            "frames": 1,
            "plots": 1
        }
    }) + "\n"


def test_healthy_json_report_accepted():
    """Fixed code accepts healthy JSON report with schema=1, exact counts, exit 0."""
    report = _healthy_json_report()
    accepted, output = _run_smoke_with_report(report, 0)
    assert accepted is True, f"Expected accept, got: {output}"


def test_json_ui_problems_rejected():
    """Fixed code rejects JSON report with ui=failed."""
    report = json.dumps({
        "schema": 1,
        "commit": "abc123",
        "stages": {
            "geometry": "built",
            "mesh": "built",
            "compute": "converged",
            "plot": "rendered",
            "ui": "failed"
        },
        "counts": {
            "nodes": 20,
            "frames": 1,
            "plots": 1
        }
    }) + "\n"
    accepted, output = _run_smoke_with_report(report, 0)
    assert accepted is False, f"Expected reject, got accept"
    assert "ui" in output.lower() or "stage" in output.lower()


def test_json_zero_frames_rejected():
    """Fixed code rejects JSON report with frames=0."""
    report = json.dumps({
        "schema": 1,
        "commit": "abc123",
        "stages": {
            "geometry": "built",
            "mesh": "built",
            "compute": "converged",
            "plot": "rendered",
            "ui": "ok"
        },
        "counts": {
            "nodes": 20,
            "frames": 0,
            "plots": 1
        }
    }) + "\n"
    accepted, output = _run_smoke_with_report(report, 0)
    assert accepted is False, f"Expected reject, got accept"
    assert "frames" in output.lower() and "expected 1" in output.lower()


def test_json_child_failed_after_report_rejected():
    """Fixed code rejects healthy JSON report when child exit != 0."""
    report = _healthy_json_report()
    accepted, output = _run_smoke_with_report(report, 1)
    assert accepted is False, f"Expected reject, got accept"
    assert "exited with code 1" in output.lower() or "child" in output.lower()


def test_json_ten_frames_rejected_exact_match():
    """Fixed code rejects JSON report with frames=10 (exact integer match, no substring)."""
    report = json.dumps({
        "schema": 1,
        "commit": "abc123",
        "stages": {
            "geometry": "built",
            "mesh": "built",
            "compute": "converged",
            "plot": "rendered",
            "ui": "ok"
        },
        "counts": {
            "nodes": 20,
            "frames": 10,
            "plots": 1
        }
    }) + "\n"
    accepted, output = _run_smoke_with_report(report, 0)
    assert accepted is False, f"Expected reject, got accept"
    assert "frames=10" in output or "expected 1" in output.lower()


def test_json_missing_schema_rejected():
    """Fixed code rejects JSON report without schema version."""
    report = json.dumps({
        "commit": "abc123",
        "stages": {
            "geometry": "built",
            "mesh": "built",
            "compute": "converged",
            "plot": "rendered",
            "ui": "ok"
        },
        "counts": {
            "nodes": 20,
            "frames": 1,
            "plots": 1
        }
    }) + "\n"
    accepted, output = _run_smoke_with_report(report, 0)
    assert accepted is False, f"Expected reject, got accept"
    assert "schema" in output.lower() and "unsupported" in output.lower()


def test_json_stale_commit_rejected():
    """Fixed code rejects JSON report with different commit."""
    report = _healthy_json_report(commit="different_commit")
    accepted, output = _run_smoke_with_report(report, 0)
    assert accepted is False, f"Expected reject, got accept"
    assert "commit" in output.lower() and "match" in output.lower()


def test_json_invalid_rejected():
    """Fixed code rejects invalid JSON."""
    report = "not valid json\n"
    accepted, output = _run_smoke_with_report(report, 0)
    assert accepted is False, f"Expected reject, got accept"
    assert "json" in output.lower() or "valid" in output.lower()


def test_json_wrong_schema_version_rejected():
    """Fixed code rejects JSON report with unsupported schema version."""
    report = json.dumps({
        "schema": 999,
        "commit": "abc123",
        "stages": {
            "geometry": "built",
            "mesh": "built",
            "compute": "converged",
            "plot": "rendered",
            "ui": "ok"
        },
        "counts": {
            "nodes": 20,
            "frames": 1,
            "plots": 1
        }
    }) + "\n"
    accepted, output = _run_smoke_with_report(report, 0)
    assert accepted is False, f"Expected reject, got accept"
    assert "schema" in output.lower() and "unsupported" in output.lower()