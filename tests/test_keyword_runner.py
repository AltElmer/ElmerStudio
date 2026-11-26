"""Keyword guard (part b): study_runner flags solver.log lines that report a keyword
as not found / not used / wrong type as user-visible warnings.

Tests here run ElmerSolver (the `elmer` fixture) and are skipped when Elmer is not
installed.
"""
from __future__ import annotations

import os

import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.elmer import run_case_blocking, write_case
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.meshing import build_mesh
from elmerstudio.core.sif import build_sif
from elmerstudio.core.study_runner import keyword_problems

BOGUS_KEY = "BogusDeckKeyZ9"


def _build_example_case(key):
    """Return (sif_text, elmer_mesh) for an example's first study step."""
    from elmerstudio.core.examples import EXAMPLES

    _, maker = EXAMPLES[key]
    m = maker()
    geo = build_geometry(m, display=False)
    st = m.studies()[0]
    step = B.study_steps(st)[0]
    sif = build_sif(m, st, step, geo, 1000).text
    mesh = build_mesh(m)
    return sif, mesh.to_elmer()


def _bogus_sif(sif):
    """Inject an invented keyword into the first Boundary Condition section."""
    anchor = "Boundary Condition 1"
    assert anchor in sif, "example has no Boundary Condition 1 section"
    return sif.replace(anchor, anchor + f"\n  {BOGUS_KEY} = Real 1.23", 1)


def _run_sif(tmp_path, inst, sif, elmer_mesh):
    d = str(tmp_path / "case")
    write_case(d, elmer_mesh, sif)
    return run_case_blocking(d, inst, nproc=1, timeout=240)


# ------------------------------------------------------------------ scanner (no Elmer needed)


def test_keyword_problems_scanner():
    lines = [
        'CheckKeyword:  Unlisted keyword: [bogusdeckkeyz9] in section: [boundary condition 1]',
        "  ListTagKeywords: Setting weight for keywords!",
        "Unlisted keyword: [AnotherBogus] in section: [initial condition 2].",
        "WARNING:: GetPermittivity: Permittivity not defined in material, defaulting to that of vacuum",
    ]
    found = keyword_problems(lines)
    assert len(found) == 2
    assert any("bogusdeckkeyz9" in ln for ln in found)
    assert any("AnotherBogus" in ln for ln in found)


# ------------------------------------------------------------------ negative control: bogus deck is flagged


def test_bogus_deck_flagged(elmer, tmp_path):
    sif, elmer_mesh = _build_example_case("busbar")
    rs = _run_sif(tmp_path, elmer, _bogus_sif(sif), elmer_mesh)
    log = "\n".join(rs.lines)
    assert "unlisted keyword" in log.lower(), f"Elmer did not report the bogus keyword:\n{log[-1500:]}"
    problems = keyword_problems(rs.lines)
    assert problems, f"study_runner keyword_problems() found nothing in:\n{log[-1500:]}"
    assert any(BOGUS_KEY.lower() in ln.lower() for ln in problems), f"bogus keyword not flagged:\n{problems}"


# ------------------------------------------------------------------ pass on main: current output is clean


def test_coil_solver_log_clean_on_main(elmer, tmp_path):
    """The coil example previously emitted `Az` in an Initial Condition, which Elmer
    reports as an unlisted keyword.  After the emitter fix, main's output is clean."""
    sif, elmer_mesh = _build_example_case("coil")
    rs = _run_sif(tmp_path, elmer, sif, elmer_mesh)
    log = "\n".join(rs.lines)
    problems = keyword_problems(rs.lines)
    assert not problems, f"coil solver.log reports keyword problems on main:\n{problems}\n{log[-1500:]}"
