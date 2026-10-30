"""Walk the NASTRAN-95 tutorial (axial bar, cantilever modes, heat conduction, 3D bar) in the real GUI and save
screenshots.  NASTRAN-95 must be found (Preferences, NASTRAN_HOME or NASTRAN_WSL_HOME).

    python tools/screenshot_nastran.py [outdir]
"""
from __future__ import annotations

import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "tools")]
warnings.filterwarnings("ignore")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402


def model(dim, phys, study, size, hmax, hmin):
    """The tutorial's models: a rectangle (2D) or block (3D) of structural steel with a Size node."""
    from elmerstudio.core import builder as B
    from elmerstudio.core.geometry import build_geometry
    m = B.new_model(dim, [phys], study)
    kw = dict(w=size[0], h=size[1]) if dim == "2D" else dict(w=size[0], h=size[1], d=size[2])
    m.create("geom.rectangle" if dim == "2D" else "geom.block", m.component.child("geometry"), index=0, **kw)
    B.add_material(m, "Structural steel", all_domains=True)
    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax=hmax, hmin=hmin)
    return m, build_geometry(m, display=False)


def bar_2d():
    from elmerstudio.core import builder as B
    m, geo = model("2D", "solid", "stationary", ("1[m]", "5[cm]"), "5[cm]", "2[cm]")
    ph = m.component.child("physics.solid")
    ph.props["plane"] = "stress"
    B.add_feature(m, ph, "solid.fixed", geo.select("boundary", xmax=0))
    bl = B.add_feature(m, ph, "solid.bload", geo.select("boundary", xmin=0.99))
    bl.props.update({"type": "force", "Fx": "20[MPa]", "Fy": "0[MPa]"})
    return m


def modes_2d():
    from elmerstudio.core import builder as B
    m, geo = model("2D", "solid", "eigen", ("1[m]", "5[cm]"), "12.5[mm]", "5[mm]")
    ph = m.component.child("physics.solid")
    ph.props["plane"] = "stress"
    B.add_feature(m, ph, "solid.fixed", geo.select("boundary", xmax=0))
    return m


def heat_2d():
    from elmerstudio.core import builder as B
    m, geo = model("2D", "ht", "stationary", ("1[m]", "20[cm]"), "10[cm]", "5[cm]")
    ph = m.component.child("physics.ht")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0)).props["T0"] = "350[K]"
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmin=0.99)).props["T0"] = "300[K]"
    return m


def bar_3d():
    from elmerstudio.core import builder as B
    m, geo = model("3D", "solid", "stationary", ("1[m]", "5[cm]", "1[cm]"), "10[cm]", "5[cm]")
    ph = m.component.child("physics.solid")
    B.add_feature(m, ph, "solid.fixed", geo.select("boundary", xmax=0))
    bl = B.add_feature(m, ph, "solid.bload", geo.select("boundary", xmin=0.99))
    bl.props.update({"type": "total", "Ftx": "10[kN]", "Fty": "0[N]", "Ftz": "0[N]"})
    return m


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "modules", "ROOT", "images", "screenshots")
    os.makedirs(out, exist_ok=True)
    app = QApplication(sys.argv[:1])
    from elmerstudio.core import project
    from elmerstudio.ui import theme
    from elmerstudio.ui.main_window import MainWindow
    from screenshot_tour import redact_messages
    theme.apply(app)
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    redact_messages(w)
    w.resize(1600, 960)
    w.show()
    saved = []
    errors = []
    sys.excepthook = lambda t, v, tb: errors.append(f"{t.__name__}: {v}")

    def shot(name, widget=None):
        from PySide6.QtCore import QPoint, QRectF
        from PySide6.QtGui import QImage, QPainter
        path = os.path.join(out, f"tutorial_nastran_{name}.png")
        if widget is not None:                 # a dialog: grab it alone
            widget.grab().save(path)
            saved.append(path)
            return
        w.settings.refresh()
        app.processEvents()
        pm = w.grab()
        pl = w.graphics.plotter
        if pl is not None and w.central.currentIndex() == 0 and w.gstack.currentIndex() == 0:
            pl.render()
            img = pl.screenshot(return_img=True)
            qi = QImage(img.copy().data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888).copy()
            widget = pl.interactor
            pos = widget.mapTo(w, QPoint(0, 0))
            p = QPainter(pm)
            p.drawImage(QRectF(pos.x(), pos.y(), widget.width(), widget.height()), qi)
            p.end()
        pm.save(path)
        saved.append(path)

    def idle(fn, timeout=900):
        t0 = time.time()

        def poll():
            if not w.busy():
                QTimer.singleShot(500, fn)
            elif time.time() - t0 > timeout:
                errors.append("timeout")
                finish()
            else:
                QTimer.singleShot(250, poll)
        QTimer.singleShot(250, poll)

    def load(m, then):
        w._set_model(m, None, project.new_workdir())
        w.build_geometry()
        idle(then)

    def solve_and_shot(name, then):
        st = w.model.studies()[0]
        w.set_prop(st, "backend", "nastran")
        w.compute(st)

        def done():
            pg = w.plot_groups()[0] if w.plot_groups() else None
            if pg is not None:
                w.tree.select(pg)
                w.plot(pg)
            QTimer.singleShot(1500, lambda: (shot(name), then()))
        idle(done, timeout=1800)

    def s1():
        load(bar_2d(), s1b)

    def s1b():
        st = w.model.studies()[0]
        w.tree.select(st)
        w.set_prop(st, "backend", "nastran")
        w.update_actions()
        QTimer.singleShot(800, lambda: (shot("1_solver_backend"), s2()))

    def s2():
        st = w.model.studies()[0]
        before = set(w.findChildren(QDialog))
        w.show_nastran_inputs(st)
        dlg = next(iter(set(w.findChildren(QDialog)) - before), None)
        if dlg is None:
            errors.append("no input deck dialog")
            s3()
            return
        dlg.show()
        QTimer.singleShot(800, lambda: (shot("2_input_deck", dlg), dlg.close(), s3()))

    def s3():
        solve_and_shot("3_bar_stress", s4)

    def s4():
        load(modes_2d(), lambda: solve_and_shot("4_mode_shape", s5))

    def s5():
        load(heat_2d(), lambda: solve_and_shot("5_temperature", s6))

    def s6():
        load(bar_3d(), lambda: solve_and_shot("6_bar_3d_stress", finish))

    def finish():
        w.dirty = False
        print("\n".join(saved))
        print("ERRORS:", errors)
        sys.stdout.flush()
        QTimer.singleShot(200, app.quit)

    QTimer.singleShot(500, s1)
    app.exec()


if __name__ == "__main__":
    main()
