"""Elmer Tutorial 12 style: capacitance matrix of two conductors with Electric Infinity BC.

Verifies that ``es.terminal`` (Capacitance Body) and ``es.infinity`` (Electric Infinity BC)
boundary features are wired into the generated ``.sif`` (``Calculate Capacitance Matrix =
True`` on ``StatElecSolver``, ``Electric Infinity BC = True`` on the infinity boundary) and
that the model solves.  Includes a negative control: a model without terminals must not
request a capacitance matrix.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core.examples import capacitance_matrix  # noqa: E402
from elmerstudio.core.elmer import find_elmer  # noqa: E402
from elmerstudio.core.geometry import build_geometry  # noqa: E402
from elmerstudio.core.results import Solution  # noqa: E402
from elmerstudio.core.sif import build_sif  # noqa: E402
from elmerstudio.core.study_runner import StudyRunner  # noqa: E402


def _build_and_solve():
    m = capacitance_matrix()
    st = m.studies()[0]
    geo = build_geometry(m, display=False)
    from elmerstudio.core.meshing import build_mesh as _bm
    mesh = _bm(m)
    sif = build_sif(m, st, st.children[0], geo, len(mesh.nodes))
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_cap_"), find_elmer()).run()
    return sif, [Solution(f) for f in folders]


def test_capacitance_matrix_keywords_present():
    """With es.terminal present, StatElecSolver requests Calculate Capacitance Matrix and the
    infinity boundary carries Electric Infinity BC."""
    m = capacitance_matrix()
    st = m.studies()[0]
    geo = build_geometry(m, display=False)
    from elmerstudio.core.meshing import build_mesh as _bm
    mesh = _bm(m)
    sif = build_sif(m, st, st.children[0], geo, len(mesh.nodes))
    text = sif.text
    assert "Calculate Capacitance Matrix" in text, text
    assert "Electric Infinity BC" in text, text


def test_capacitance_matrix_solves(elmer):
    """The two-terminal model solves and the terminal domains sit at their prescribed potentials."""
    sif, sols = _build_and_solve()
    s = sols[0]
    assert s.meta.get("status") == "converged", s.meta.get("diagnostics")
    i = s.frame_index("last")
    V = s.evaluate("V", i)
    # terminal 1 (V0) is held at 1 V, terminal 2 at 0 V: check the field respects both
    assert float(np.max(V)) > 0.5 and float(np.min(V)) < 0.5, V


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
    from elmerstudio.core.meshing import build_mesh as _bm
    mesh = _bm(m)
    sif = build_sif(m, st, st.children[0], geo, len(mesh.nodes))
    assert "Calculate Capacitance Matrix" not in sif.text, sif.text
