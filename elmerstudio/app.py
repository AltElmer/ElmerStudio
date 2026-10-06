"""Application entry point:  python -m elmerstudio [model.esm] [--example KEY]."""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="elmerstudio", description="Elmer Studio - model builder for Elmer FEM")
    ap.add_argument("file", nargs="?", help="model file (.esm) to open")
    ap.add_argument("--example", help="open an example model (busbar, heat_sink, cantilever, flow_cylinder, ...)")
    ap.add_argument("--elmer-home", help="Elmer installation folder (overrides the saved preference)")
    ap.add_argument("--no-new", action="store_true", help="do not show the New window at startup")
    ap.add_argument("--quit-after", type=float, default=0, help=argparse.SUPPRESS)   # smoke-testing the entry point
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

        def _quit():
            print(f"OK title={w.windowTitle()!r} nodes={sum(1 for _ in w.model.root.walk())} "
                  f"geometry={'built' if w.geo is not None else 'none'} elmer={w.inst.solver if w.inst else None}",
                  flush=True)
            w.dirty = False
            app.quit()
        QTimer.singleShot(int(args.quit_after * 1000), _quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
