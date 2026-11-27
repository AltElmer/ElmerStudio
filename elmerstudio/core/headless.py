"""Headless CLI: ``python -m elmerstudio run <model.esm|--example KEY> ...``.

Runs a study without opening the Qt main window, applies parameter overrides,
exports derived-value tables and plot images, and writes a structured JSON
summary of model metadata, solver status, derived values and result statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any

import numpy as np

from .elmer import find_elmer
from .examples import EXAMPLES
from .model import Model
from .project import discard_workdir, load as load_project, new_workdir
from .results import Solution
from .study_runner import StudyRunner, StudyRunError


class HeadlessError(RuntimeError):
    """Base error for headless CLI."""
    exit_code = 2  # usage / bad arguments


class SolverError(HeadlessError):
    """Solver did not converge or failed."""
    exit_code = 1


class InternalError(HeadlessError):
    """Unexpected internal error."""
    exit_code = 3


def _apply_params(model: Model, overrides: dict[str, str]) -> None:
    """Write *overrides* into the model's global ``params`` table so the SIF
    and mesh see them (mirrors the GUI parameter editor).

    Raises HeadlessError(exit_code=2) if *overrides* contains a parameter
    name that does not exist in the model's parameter table.
    """
    params = model.global_defs.child("params")
    if params is None:
        if not overrides:
            return
        raise HeadlessError("model has no parameter table; cannot apply overrides")
    table = list(params.props.get("table") or [])
    names = {str(row.get("name")) for row in table}
    for k, v in overrides.items():
        if k not in names:
            raise HeadlessError(f"unknown parameter '{k}' (available: {sorted(names)})")
        row = next(r for r in table if str(r.get("name")) == k)
        row["expr"] = str(v)
    params.props["table"] = table


def _load_source(source: str | None, example: str | None) -> tuple[Model, str, str | None]:
    """Return ``(model, workdir, source_path)``. *source_path* is None for examples."""
    if example:
        if example not in EXAMPLES:
            raise HeadlessError(f"unknown example '{example}' (try: {', '.join(sorted(EXAMPLES))})")
        entry = EXAMPLES[example]
        factory = entry[1] if isinstance(entry, tuple) else entry
        m = factory()
        return m, new_workdir(), None
    if not source:
        raise HeadlessError("no model file given (pass a .esm path or --example KEY)")
    if not os.path.isfile(source):
        raise HeadlessError(f"model file not found: {source}")
    if not source.lower().endswith(".esm"):
        raise HeadlessError(f"not an ElmerStudio model file (.esm): {source}")
    m, workdir = load_project(source)
    m.file_path = source
    return m, workdir, source


def _pick_study(model: Model, wanted: str | None):
    studies = model.studies()
    if not studies:
        raise HeadlessError("model has no study")
    if wanted:
        for s in studies:
            if s.tag == wanted or s.label == wanted:
                return s
        raise HeadlessError(f"study '{wanted}' not found (have: {[s.tag for s in studies]})")
    return studies[0]


def _derived_table(sol: Solution) -> tuple[list[str], list[list]]:
    """A small derived-values table for the solution's last frame, in COMSOL
    style: frame value plus mean/max/min/volume-integral of the default field."""
    exprs = [r for r in (sol.available() or [])]
    scalar = None
    for a, _l, _u, kind in exprs:
        if kind == "scalar":
            scalar = a
            break
    if scalar is None:
        scalar = "x"
    label, si = sol.describe(scalar)
    nf = len(sol.frames)
    cols = ["frame", "time", f"{label} mean", f"{label} max", f"{label} min", f"{label} volint (m^3)"]
    rows = []
    for fi in range(nf):
        fr = sol.frames[fi]
        try:
            vals = sol.evaluate(scalar, fi)
            finite = vals[np.isfinite(vals)] if vals.size else np.array([float("nan")])
            mean = float(finite.mean()) if finite.size else float("nan")
            mx, mn = sol.maxmin(scalar, fi, "domain")
            integ = sol.integrate(scalar, fi, "domain")
        except Exception:
            mean = mx = mn = integ = float("nan")
        t = fr.value if fr.value is not None else float("nan")
        rows.append([fi + 1, t, mean, mx, mn, integ])
    return cols, rows


def _export_table(cols, rows, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            w.writerow([("" if v is None or (isinstance(v, float) and v != v) else v) for v in row])


def _export_plot(sol: Solution, path: str) -> None:
    """Render the last frame's default scalar field to a PNG (off-screen)."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=[700, 500])
    try:
        g = sol.bulk_grid()
        expr = None
        for a, _l, _u, kind in (sol.available() or []):
            if kind == "scalar":
                expr = a
                break
        if expr is not None:
            try:
                vals = sol.evaluate(expr, sol.frame_index("last"))
                g.point_data[expr] = vals
                pl.add_mesh(g, scalars=expr, show_scalar_bar=True)
            except Exception:
                pl.add_mesh(g, show_scalar_bar=False)
        else:
            pl.add_mesh(g, show_scalar_bar=False)
        pl.camera_position = "iso"
        pl.screenshot(path)
    finally:
        pl.close()


def run_headless(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="elmerstudio run", add_help=True)
    ap.add_argument("source", nargs="?", help="model file (.esm) to run")
    _EXAMPLE_LIST = object()  # sentinel: --example given without a value
    ap.add_argument("--example", nargs="?", const=_EXAMPLE_LIST, default=None,
                    help='run a built-in example instead of a .esm file (capacitor, busbar, ...). '
                         'Pass --example without a value to list them')
    ap.add_argument("--study", help="study tag/label to run (default: first study)")
    ap.add_argument("--param", action="append", default=[], metavar="K=V",
                    help="global parameter override (repeatable), e.g. --param V0=2[V]")
    ap.add_argument("--export", action="append", default=[], metavar="KIND:PATH",
                    help="export kind:path (table:<csv> or plot:<png>), repeatable")
    ap.add_argument("--json", help="write the structured JSON summary to this file (default: stdout)")
    args = ap.parse_args(argv)

    if args.example is _EXAMPLE_LIST:
        print("Built-in examples: " + ", ".join(sorted(EXAMPLES)))
        return 0

    summary: dict[str, Any] = {"ok": False, "model": {}, "study": {}, "solve": {},
                               "derived": {"columns": [], "rows": []}, "exports": {}}
    workdir: str | None = None
    try:
        overrides: dict[str, str] = {}
        for item in args.param:
            if "=" not in item:
                raise HeadlessError(f"--param must be K=V, got: {item!r}")
            k, v = item.split("=", 1)
            overrides[k.strip()] = v.strip()

        model, workdir, source_path = _load_source(args.source, args.example)
        _apply_params(model, overrides)
        study = _pick_study(model, args.study)

        summary["model"] = {
            "label": model.root.label,
            "file": source_path,
            "example": args.example,
            "studies": [{"tag": s.tag, "label": s.label} for s in model.studies()],
            "params": dict(model.parameters()),
        }
        summary["study"] = {"tag": study.tag, "label": study.label}

        inst = find_elmer()
        try:
            folders = StudyRunner(model, study, workdir, inst).run()
        except StudyRunError as exc:
            summary["solve"] = {
                "status": exc.status,
                "folders": [exc.folder],
                "frames": 0,
                "diagnostics": exc.diagnostics,
            }
            summary["ok"] = False
            out = json.dumps(summary, indent=1, default=str)
            if args.json:
                os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
                with open(args.json, "w", encoding="utf-8") as f:
                    f.write(out + "\n")
            else:
                print(out, file=sys.stderr)
            return SolverError("solver run did not converge").exit_code

        sols = [Solution(f) for f in folders]
        last = sols[-1]
        status = last.meta.get("status", "unknown")
        summary["solve"] = {
            "status": status,
            "folders": folders,
            "frames": len(last.frames),
            "diagnostics": last.meta.get("diagnostics"),
        }
        cols, rows = _derived_table(last)
        summary["derived"] = {"columns": cols, "rows": rows}

        for exp in args.export:
            if ":" not in exp:
                raise HeadlessError(f"--export must be KIND:PATH, got: {exp!r}")
            kind, path = exp.split(":", 1)
            kind = kind.strip().lower()
            path = path.strip()
            if kind in ("table", "csv"):
                _export_table(cols, rows, path)
                summary["exports"]["table"] = path
            elif kind in ("plot", "png"):
                _export_plot(last, path)
                summary["exports"]["plot"] = path
            else:
                raise HeadlessError(f"unknown --export kind '{kind}' (use table or plot)")

        summary["ok"] = status == "converged"
        out = json.dumps(summary, indent=1, default=str)
        if args.json:
            if args.json == "-":
                print(out)
            else:
                os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
                with open(args.json, "w", encoding="utf-8") as f:
                    f.write(out + "\n")
        else:
            print(out)
        return 0 if summary["ok"] else SolverError("solver did not converge").exit_code
    except SystemExit:
        raise
    except HeadlessError as exc:
        err = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "model": summary.get("model", {}),
               "study": summary.get("study", {})}
        out = json.dumps(err, indent=1, default=str)
        if args.json:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as f:
                f.write(out + "\n")
        else:
            print(out, file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        err = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "model": summary.get("model", {}),
               "study": summary.get("study", {})}
        out = json.dumps(err, indent=1, default=str)
        if args.json:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as f:
                f.write(out + "\n")
        else:
            print(out, file=sys.stderr)
        return InternalError(str(exc)).exit_code
    finally:
        if workdir:
            discard_workdir(workdir)
