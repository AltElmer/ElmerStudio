"""Eigenmode (eigenfrequency) shapes carry energy on every platform / Elmer build.

Elmer writes eigenmode arrays under build-dependent names ("displacement EigenMode1",
"displacement:eigenmode1", "displacement eigenmode 1", ...).  results.py arrays() promotes the
current mode's array to the plain field name so that every post-processing entry (solid.disp,
solid.u, ...) evaluates the selected mode rather than a silently-all-zero fallback.  A
Solution.zero_modes() detector flags modes whose displacement vectors are numerically all-zero.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elmerstudio.core.examples import cantilever_eigen  # noqa: E402
from elmerstudio.core.elmer import find_elmer  # noqa: E402
from elmerstudio.core.results import Frame, Solution  # noqa: E402
from elmerstudio.core.study_runner import StudyRunner  # noqa: E402


def _solve_eigen():
    m = cantilever_eigen()
    st = m.studies()[0]
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_eigen_"), find_elmer()).run()
    return [Solution(f) for f in folders]


def test_eigen_modes_have_energy(elmer):
    """Every computed eigenmode has non-zero displacement (no zero mode shapes)."""
    sols = _solve_eigen()
    s = sols[0]
    assert s.kind == "study.eigen" and len(s.frames) > 0
    assert s.zero_modes() == [], f"zero-mode detector flagged modes: {s.zero_modes()}"
    for mode in range(1, len(s.frames) + 1):
        i = s.frame_index(mode)
        disp = s.evaluate("solid.disp", i)
        assert float(np.max(np.abs(disp))) > 0.0, f"eigenmode {mode} is all-zero"


def test_zero_modes_detector_negative_control():
    """A synthetic all-zero displacement field must be flagged as zero modes (must FAIL if it
    returns [])."""
    sol = object.__new__(Solution)
    sol.kind = "study.eigen"
    sol.frames = [Frame(file="", label="mode 1", value=100.0, mode=1),
                  Frame(file="", label="mode 2", value=200.0, mode=2)]
    sol._cache = {}

    def fake_arrays(i):
        n = 8
        return {"displacement": np.zeros((n, 3)), "solid.u_vec": np.zeros((n, 3))}

    sol.arrays = fake_arrays
    assert sol.zero_modes() == [1, 2], "negative control: zero-modes detector missed all-zero modes"
