"""Headless CLI: ``python -m elmerstudio run <model.esm|--example KEY> ...``.

Verifies end-to-end headless operation (no Qt main window): parameter overrides,
``--export table.csv`` / ``--export plot.png``, structured ``--json`` summary, and
negative controls returning non-zero exit codes for bad arguments.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

PYTHON = sys.executable  # the venv interpreter running pytest


def _run(args):
    return subprocess.run(
        [PYTHON, "-m", "elmerstudio", "run"] + args,
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def _json(out):
    return json.loads(out)


def test_headless_run_example_json(tmp_path):
    out = tmp_path / "summary.json"
    res = _run(["--example", "capacitor", "--study", "std1", "--json", str(out)])
    assert res.returncode == 0, res.stderr
    d = _json(out.read_text())
    assert d["ok"] is True
    assert d["solve"]["status"] == "converged"
    assert d["model"]["example"] == "capacitor"
    assert d["derived"]["columns"]
    assert d["derived"]["rows"]


def test_headless_param_override(tmp_path):
    out = tmp_path / "summary.json"
    res = _run(["--example", "capacitor", "--param", "V0=2[V]", "--json", str(out)])
    assert res.returncode == 0, res.stderr
    d = _json(out.read_text())
    # the 1 V->2 V override scales the potential: max should be ~2 V
    assert abs(d["derived"]["rows"][0][3] - 2.0) < 1e-6, d


def test_headless_exports(tmp_path):
    table = tmp_path / "t.csv"
    plot = tmp_path / "p.png"
    res = _run(["--example", "capacitor", "--export", f"table:{table}",
                "--export", f"plot:{plot}", "--json", str(tmp_path / "s.json")])
    assert res.returncode == 0, res.stderr
    assert table.is_file() and table.stat().st_size > 0
    assert plot.is_file() and plot.stat().st_size > 0


def test_headless_bad_param_returns_nonzero():
    res = _run(["--example", "capacitor", "--param", "notanassignment"])
    assert res.returncode != 0
    d = _json(res.stderr)
    assert d["ok"] is False


def test_headless_missing_file_returns_nonzero():
    res = _run(["/no/such/model.esm"])
    assert res.returncode != 0
    d = _json(res.stderr)
    assert d["ok"] is False


def test_headless_bad_example_returns_nonzero():
    res = _run(["--example", "does-not-exist"])
    assert res.returncode != 0
    d = _json(res.stderr)
    assert d["ok"] is False
