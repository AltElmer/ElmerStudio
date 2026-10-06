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
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
