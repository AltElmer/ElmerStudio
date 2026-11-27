"""Repo hygiene: no internal scratchpad or working-report paths may be tracked.

The branch is public-facing (AltElmer/ElmerStudio). ``scratchpad/`` and
``reports/`` hold internal working files (review notes, scratchpad checkouts,
per-issue reports) that must never be committed. This test scans the tracked
file list and fails when any path under ``scratchpad/`` or ``reports/`` is
tracked, so the defect cannot be reintroduced unnoticed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _tracked_under(*prefixes: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", *prefixes],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def test_no_scratchpad_or_reports_paths_tracked():
    bad = _tracked_under("scratchpad/", "reports/")
    assert not bad, (
        "scratchpad/reports paths tracked in git (internal working files must not be committed):\n  "
        + "\n  ".join(bad)
    )
