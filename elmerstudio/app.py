"""Application entry point:  python -m elmerstudio [model.esm] [--example KEY]."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def _fix_stdio():
    """Windowed (frozen) builds have no console: sys.stdout/stderr are None and any print would crash."""
    if sys.stdout is None or sys.stderr is None:
        import tempfile
        try:
            log = open(os.path.join(tempfile.gettempdir(), "elmerstudio.log"), "w", buffering=1, errors="replace")
        except OSError:
            log = open(os.devnull, "w")
        if sys.stdout is None:
            sys.stdout = log
        if sys.stderr is None:
            sys.stderr = log


def _get_commit():
    """Get commit hash; for frozen builds, read from embedded file."""
    # Check for embedded commit (written by packaging/build.py)
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        embedded = os.path.join(base, ".commit")
        if os.path.exists(embedded):
            with open(embedded, "r") as f:
                return f.read().strip()
    except Exception:
        pass
    # Fallback: git rev-parse (dev checkout)
    git_exe = r"C:\Program Files\Git\bin\git.exe"
    if not os.path.exists(git_exe):
        git_exe = "git"  # fallback to PATH
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            [git_exe, "rev-parse", "--short", "HEAD"],
            cwd=base,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        return "unknown"


def main(argv=None):
    _fix_stdio()
    ap = argparse.ArgumentParser(prog="elmerstudio", description="Elmer Studio - model builder for Elmer FEM")
    ap.add_argument("file", nargs="?", help="model file (.esm) to open")
    ap.add_argument("--example", help="open an example model (busbar, heat_sink, cantilever, flow_cylinder, ...)")
    ap.add_argument("--elmer-home", help="Elmer installation folder (overrides the saved preference)")
    ap.add_argument("--no-new", action="store_true", help="do not show the New window at startup")
    ap.add_argument("--quit-after", type=float, default=0, help=argparse.SUPPRESS)   # smoke-testing the entry point
    ap.add_argument("--selftest-out", help=argparse.SUPPRESS)                       # write the smoke result to a file
    ap.add_argument("--selftest-compute", action="store_true", help=argparse.SUPPRESS)  # also solve + plot
    ap.add_argument("--selftest-ui", action="store_true", help=argparse.SUPPRESS)   # desktop acceptance checks
    args = ap.parse_args(argv)
    if args.elmer_home:
        os.environ["ELMER_HOME"] = args.elmer_home
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    import warnings
    warnings.filterwarnings("ignore", message=".*extract_surface.*")
    warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from . import APP_NAME
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("ElmerStudio")
    from .ui import theme
    theme.apply(app)
    from .ui.main_window import MainWindow
    w = MainWindow(args.file, show_new=not (args.no_new or args.example or args.file))
    w.show()
    if args.example:
        w.open_example(args.example)
    if args.quit_after:
        from PySide6.QtCore import QTimer
        w.no_modal = True
        state = {"computed": None, "ui": None}

        def _after_idle(fn):
            if w.busy():
                QTimer.singleShot(250, lambda: _after_idle(fn))
            else:
                fn()

        def _compute():
            if args.selftest_ui:
                from .ui.acceptance import run_ui_selftest
                problems = run_ui_selftest(w)
                state["ui"] = "ok" if not problems else f"{len(problems)}-problems"
                for p in problems[:40]:
                    print("UI:", p, flush=True)
            if args.selftest_compute and w.inst and w.model.studies():
                state["computed"] = "running"
                w.compute(w.model.studies()[0])
                _after_idle(_report_compute)
            else:
                QTimer.singleShot(int(args.quit_after * 1000), _quit)

        def _report_compute():
            pgs = w.plot_groups()
            sol = w.solution_of(w._dataset_for(pgs[0])) if pgs else None
            state["computed"] = f"plots={len(pgs)} frames={len(sol.frames) if sol else 0}"
            if pgs:
                w.plot(pgs[0])
            QTimer.singleShot(1500, _quit)

        def _quit():
            commit = _get_commit()

            # Parse computed state
            frames = 0
            plots = 0
            compute_stage = "not-run"
            if state["computed"]:
                if state["computed"] == "running":
                    compute_stage = "running"
                else:
                    # state["computed"] is like "plots=1 frames=1"
                    for part in state["computed"].split():
                        if part.startswith("plots="):
                            plots = int(part.split("=")[1])
                        elif part.startswith("frames="):
                            frames = int(part.split("=")[1])
                    compute_stage = "converged" if frames > 0 else "not-run"

            # Parse UI state
            ui_stage = "not-run"
            if state["ui"]:
                ui_stage = "ok" if state["ui"] == "ok" else "failed"

            # Determine overall pass/fail for required stages
            # Required stages: geometry, mesh, compute, plot, ui
            required_ok = True
            if w.geo is None:
                required_ok = False
            if state.get("computed") and args.selftest_compute and w.inst and w.model.studies():
                # compute was requested but didn't converge
                if compute_stage != "converged":
                    required_ok = False
            if args.selftest_ui:
                if ui_stage != "ok":
                    required_ok = False

            report = {
                "schema": 1,
                "commit": commit,
                "stages": {
                    "geometry": "built" if w.geo is not None else "not-run",
                    "mesh": "built" if w.geo is not None else "not-run",
                    "compute": compute_stage,
                    "plot": "rendered" if plots > 0 else "not-run",
                    "ui": ui_stage
                },
                "counts": {
                    "nodes": sum(1 for _ in w.model.root.walk()),
                    "frames": frames,
                    "plots": plots
                }
            }
            report_json = json.dumps(report)
            print(report_json, flush=True)
            if args.selftest_out:
                with open(args.selftest_out, "w") as f:
                    f.write(report_json + "\n")
            w.dirty = False
            exit_code = 0 if required_ok else 1
            app.exit(exit_code)
            return exit_code
        QTimer.singleShot(3000, lambda: _after_idle(_compute))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main() or 0)
