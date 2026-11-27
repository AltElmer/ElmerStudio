"""Headless CLI: ``python -m elmerstudio run <model.esm|--example KEY> ...``.

Verifies end-to-end headless operation (no Qt main window): parameter overrides,
``--export table.csv`` / ``--export plot.png``, structured ``--json`` summary, and
negative controls returning distinct non-zero exit codes for:
- usage errors (exit code 2): bad --param format, unknown parameter, missing file, bad example
- solver non-convergence (exit code 1)
- internal errors (exit code 3)

Negative controls (proving derived values come from real result data):
- test_derived_values_match_independent_read: _derived_table output matches
  an independent read of the same result files via Solution class.
- test_derived_values_change_when_result_corrupted: corrupting the VTU changes
  the derived values.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

# The CLI tests run as subprocesses; the elmer fixture ensures Elmer is available
# and the test is skipped if not.
pytestmark = pytest.mark.usefixtures("elmer")


def _run(args, cwd=None):
    if cwd is None:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, "-m", "elmerstudio", "run"] + args,
        capture_output=True, text=True, cwd=cwd,
    )


def _json(out):
    return json.loads(out)


def test_headless_run_example_json(tmp_path, elmer):
    out = tmp_path / "summary.json"
    res = _run(["--example", "capacitor", "--study", "std1", "--json", str(out)])
    assert res.returncode == 0, res.stderr
    d = _json(out.read_text())
    assert d["ok"] is True
    assert d["solve"]["status"] == "converged"
    assert d["model"]["example"] == "capacitor"
    assert d["derived"]["columns"]
    assert d["derived"]["rows"]


def test_headless_param_override(tmp_path, elmer):
    out = tmp_path / "summary.json"
    res = _run(["--example", "capacitor", "--param", "V0=2[V]", "--json", str(out)])
    assert res.returncode == 0, res.stderr
    d = _json(out.read_text())
    # the 1 kV -> 2 V override scales the potential: max should be ~2 V
    assert abs(d["derived"]["rows"][0][3] - 2.0) < 1e-6, d


def test_headless_exports(tmp_path, elmer):
    table = tmp_path / "t.csv"
    plot = tmp_path / "p.png"
    res = _run(["--example", "capacitor", "--export", f"table:{table}",
                "--export", f"plot:{plot}", "--json", str(tmp_path / "s.json")])
    assert res.returncode == 0, res.stderr
    assert table.is_file() and table.stat().st_size > 0
    assert plot.is_file() and plot.stat().st_size > 0


def test_headless_bad_param_format_returns_exit2():
    res = _run(["--example", "capacitor", "--param", "notanassignment"])
    assert res.returncode == 2, f"expected 2 got {res.returncode}: {res.stderr}"
    d = _json(res.stderr)
    assert d["ok"] is False


def test_headless_unknown_param_returns_exit2():
    res = _run(["--example", "capacitor", "--param", "UNKNOWN_PARAM=42"])
    assert res.returncode == 2, f"expected 2 got {res.returncode}: {res.stderr}"
    d = _json(res.stderr)
    assert d["ok"] is False
    assert "unknown parameter" in d["error"].lower()


def test_headless_missing_file_returns_exit2():
    res = _run(["/no/such/model.esm"])
    assert res.returncode == 2, f"expected 2 got {res.returncode}: {res.stderr}"
    d = _json(res.stderr)
    assert d["ok"] is False


def test_headless_bad_example_returns_exit2():
    res = _run(["--example", "does-not-exist"])
    assert res.returncode == 2, f"expected 2 got {res.returncode}: {res.stderr}"
    d = _json(res.stderr)
    assert d["ok"] is False


def test_headless_bad_export_kind_returns_exit2():
    res = _run(["--example", "capacitor", "--export", "badkind:out.csv"])
    assert res.returncode == 2, f"expected 2 got {res.returncode}: {res.stderr}"
    d = _json(res.stderr)
    assert d["ok"] is False

def test_headless_example_lists_examples():
    res = _run(["--example"])
    assert res.returncode == 0, f"expected 0 got {res.returncode}: {res.stderr}"
    from elmerstudio.core.examples import EXAMPLES
    for name in sorted(EXAMPLES):
        assert name in res.stdout, f"example {name!r} not listed: {res.stdout}"


def _nonlinear_pde_esm(tmp_path, nonconv: str) -> str:
    """A nonlinear stationary PDE (c = 1 + u^2) with nl_maxit=2: iterations run out
    before the tolerance is met, so the run ends with status "not converged"."""
    from elmerstudio.core import builder as B
    from elmerstudio.core.geometry import build_geometry
    from elmerstudio.core.project import save

    m = B.new_model("2D", ["pde"], "stationary")
    m.root.label = "nonlinear_pde"
    g = m.component.child("geometry")
    g.props["unit"] = "m"
    m.create("geom.rectangle", g, index=0, w="1.5", h="1", base="center")
    geo = build_geometry(m, display=False)
    pde = m.component.child("physics.pde")
    pde.child("pde.cfeq").props.update(c="1+u^2", f="20", da="0")
    B.add_feature(m, pde, "pde.dir", geo.numbers["boundary"], r="0")
    st = m.studies()[0]
    st.props["nonconv"] = nonconv
    B.ensure_solver_configs(m, st).children[0].props.update(nl_maxit=2, nl_tol="1e-12")

    path = str(tmp_path / "nonlinear_pde.esm")
    save(m, path, None)
    return path


@pytest.mark.parametrize("nonconv", ["stop", "keep"])
def test_headless_nonconverged_returns_exit1(tmp_path, elmer, nonconv):
    esm = _nonlinear_pde_esm(tmp_path, nonconv)
    out = tmp_path / "summary.json"
    res = _run([esm, "--json", str(out)])
    assert res.returncode == 1, f"expected 1 got {res.returncode}: {res.stderr}"
    d = _json(out.read_text())
    assert d["ok"] is False
    assert d["solve"]["status"] == "not converged"
    assert d["solve"]["diagnostics"], f"no diagnostics: {d}"
    assert d["solve"]["folders"], "no output folders recorded"


def test_headless_corrupt_file_returns_exit3(tmp_path):
    bad = tmp_path / "corrupt.esm"
    bad.write_bytes(b"this is not an ElmerStudio model file")
    out = tmp_path / "summary.json"
    res = _run([str(bad), "--json", str(out)])
    assert res.returncode == 3, f"expected 3 got {res.returncode}: {res.stderr}"
    d = _json(out.read_text())
    assert d["ok"] is False
    assert "zip" in d["error"].lower()


# ---------------------------------------------------------------------------
# Negative controls: derived values come from real result data, not invented
# ---------------------------------------------------------------------------


def test_derived_values_match_independent_read(elmer, tmp_path):
    """_derived_table output matches an independent read via Solution class.

    Runs the study directly (not via CLI subprocess) to preserve the result folder.
    """
    import numpy as np

    from elmerstudio.core.examples import capacitor
    from elmerstudio.core.elmer import find_elmer
    from elmerstudio.core.study_runner import StudyRunner
    from elmerstudio.core.results import Solution
    from elmerstudio.core.project import new_workdir, discard_workdir
    from elmerstudio.core.headless import _derived_table

    model = capacitor()
    study = model.studies()[0]
    workdir = new_workdir()
    inst = find_elmer()
    try:
        folders = StudyRunner(model, study, workdir, inst).run()
        assert folders
        sol = Solution(folders[0])

        # CLI's derived table
        cols, rows = _derived_table(sol)
        cli_max = rows[0][3]  # Electric potential max
        cli_mean = rows[0][2]  # Electric potential mean

        # Independent read
        vals = sol.evaluate("V", sol.frame_index("last"))
        finite = vals[np.isfinite(vals)]
        independent_max = float(finite.max()) if finite.size else float("nan")
        independent_mean = float(finite.mean()) if finite.size else float("nan")

        # Match within tolerance
        assert abs(cli_max - independent_max) < 1e-9, \
            f"_derived_table max {cli_max} != independent read {independent_max}"
        assert abs(cli_mean - independent_mean) < 1e-9, \
            f"_derived_table mean {cli_mean} != independent read {independent_mean}"
    finally:
        discard_workdir(workdir)


def test_derived_values_change_when_result_corrupted(elmer, tmp_path):
    """Negative control: corrupt a result VTU and verify derived values change."""
    import pyvista as pv
    import numpy as np
    from pathlib import Path

    from elmerstudio.core.examples import capacitor
    from elmerstudio.core.elmer import find_elmer
    from elmerstudio.core.study_runner import StudyRunner
    from elmerstudio.core.results import Solution
    from elmerstudio.core.headless import _derived_table
    from elmerstudio.core.project import new_workdir, discard_workdir

    model = capacitor()
    study = model.studies()[0]
    workdir = new_workdir()
    inst = find_elmer()
    try:
        folders = StudyRunner(model, study, workdir, inst).run()
        assert folders
        sol = Solution(folders[0])

        # Original derived max
        cols, rows = _derived_table(sol)
        original_max = rows[0][3]

        # Corrupt the VTU
        result_dir = Path(folders[0]) / "results"
        result_files = list(result_dir.glob("*.vtu"))
        assert result_files, f"no VTU in {result_dir}"
        vtu_path = result_files[0]

        g = pv.read(vtu_path)
        if "potential" not in g.point_data:
            pytest.skip("potential field not in VTU")
        g.point_data["potential"] = np.full_like(g.point_data["potential"], 999.0)
        g.save(vtu_path)

        # Re-read corrupted file via new Solution instance (bypass cache)
        sol2 = Solution(folders[0])
        cols2, rows2 = _derived_table(sol2)
        corrupted_max = rows2[0][3]

        assert abs(corrupted_max - 999.0) < 1e-6, f"corruption failed: {corrupted_max}"
        assert abs(corrupted_max - original_max) > 1.0, \
            "corrupted max should differ significantly from original"
    finally:
        discard_workdir(workdir)