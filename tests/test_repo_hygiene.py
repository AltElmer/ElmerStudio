"""Repo hygiene: no internal scratchpad or working-report paths may be tracked.

The branch is public-facing (AltElmer/ElmerStudio). ``scratchpad/`` and
``reports/`` hold internal working files (review notes, scratchpad checkouts,
per-issue reports) that must never be committed. This test scans the tracked
file list and fails when any path under ``scratchpad/`` or ``reports/`` is
tracked, so the defect cannot be reintroduced unnoticed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_GIT_CANDIDATES = (
    r"C:\Program Files\Git\bin\git.exe",
    r"C:\Program Files (x86)\Git\bin\git.exe",
    r"C:\Program Files\Git\mingw64\bin\git.exe",
    r"C:\Program Files (x86)\Git\mingw64\bin\git.exe",
    r"C:\msys64\usr\bin\git.exe",
    r"C:\cygwin64\bin\git.exe",
)


def _git_cmd() -> list[str]:
    """Locate the git executable robustly.

    ``shutil.which("git")`` covers the common case (Git on PATH). On Windows a
    bare python process may not have Git's bin directory on PATH even though
    Git is installed, so fall back to the standard install locations rather
    than letting ``subprocess`` raise a cryptic FileNotFoundError.
    """
    git = shutil.which("git")
    if git:
        return [git]
    for candidate in _GIT_CANDIDATES:
        if os.path.isfile(candidate):
            return [candidate]
    raise FileNotFoundError(
        "git executable not found on PATH or at the standard Windows install "
        "locations; cannot enumerate tracked files"
    )


def _tracked_under(*prefixes: str) -> list[str]:
    out = subprocess.run(
        _git_cmd() + ["ls-files", "--", *prefixes],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def test_no_scratchpad_or_reports_paths_tracked():
    bad = _tracked_under("scratchpad/", "reports/")
    assert not bad, (
        "scratchpad/reports paths tracked in git (internal working files must not be committed):\n  "
        + "\n  ".join(bad)
    )
