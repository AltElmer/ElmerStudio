"""Walk the COMSOL Busbar tutorial in the real GUI and save screenshots of each tutorial stage.

    python tools/screenshot_tutorial.py [outdir]
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
from PySide6.QtWidgets import QApplication  # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "screenshots")
    os.makedirs(out, exist_ok=True)
    app = QApplication(sys.argv[:1])
    from elmerstudio.core import builder as B
    from elmerstudio.core import project
    from elmerstudio.core.geometry import build_geometry
    from elmerstudio.ui import theme
    from elmerstudio.ui.main_window import MainWindow
    from screenshot_tour import redact_messages
    from test_tutorial_busbar import bolt_contact_faces, build_model, setup_physics
    theme.apply(app)
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    redact_messages(w)
    w.resize(1600, 960)
    w.show()
    saved = []
    errors = []
    sys.excepthook = lambda t, v, tb: errors.append(f"{t.__name__}: {v}")

    def shot(name):
        from PySide6.QtCore import QPoint, QRectF
        from PySide6.QtGui import QImage, QPainter
        app.processEvents()
        pm = w.grab()
        pl = w.graphics.plotter
        if pl is not None and w.central.currentIndex() == 0:
            if w.gstack.currentIndex() == 0:
                pl.render()
                img = pl.screenshot(return_img=True)
                qi = QImage(img.copy().data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888).copy()
                widget = pl.interactor
            else:
                qi = None
            if qi is not None:
                pos = widget.mapTo(w, QPoint(0, 0))
                p = QPainter(pm)
                p.drawImage(QRectF(pos.x(), pos.y(), widget.width(), widget.height()), qi)
                p.end()
        path = os.path.join(out, f"tutorial_busbar_{name}.png")
        pm.save(path)
        saved.append(path)

    def idle(fn, timeout=1500):
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

    S = {}

    def s1():
        m = build_model()
        w._set_model(m, None, project.new_workdir())
        g = m.component.child("geometry")
        wp1 = g.children[0]
        w.tree.select(wp1.children[3])              # Fillet 1 inside Work Plane 1: 2D sketch with point numbers
        QTimer.singleShot(1200, lambda: (shot("1_workplane_fillet_points"), s2()))

    def s2():
        g = w.model.component.child("geometry")
        wp2 = [c for c in g.children if c.kind == "geom.workplane"][1]
        S["wp2"] = wp2
        w.tree.select(wp2)                           # face-parallel plane: geometry so far, boundary numbers
        QTimer.singleShot(1500, lambda: (shot("2_face_parallel_pick"), s3()))

    def s3():
        w.build_geometry()
        w.tree.select(w.model.component.child("geometry"))
        idle(lambda: (shot("3_geometry"), s4()))

    def s4():
        m = w.model
        geo = build_geometry(m, display=False)
        setup_physics(m, geo, bolt_contact_faces(geo))
        defs = m.component.child("definitions")
        op = w.create_node("cpl.average", defs, select=False)
        op.selection.all = True
        st = m.studies()[0]
        w.create_node("study.sweep", st, index=0, select=False, pname="wbb", pvalues="range(5,1,10)", punit="cm")
        w.tree.refresh()
        mat = [c for c in m.component.child("materials").children][1]
        w.tree.select(mat)
        QTimer.singleShot(1500, lambda: (shot("4_titanium_material"), s5()))

    def s5():
        w.compute(w.model.studies()[0])
        idle(lambda: (shot("5_results_temperature"), s6()), timeout=2400)

    def s6():
        pgs = w.plot_groups()
        pot = next((p for p in pgs if "Potential" in p.label), None)
        if pot is not None:
            w.tree.select(pot)
        QTimer.singleShot(1500, lambda: (shot("6_multislice_potential"), s7()))

    def s7():
        res = w.model.results
        ev = w.create_node("eval.global", res.child("results.derived"))
        ev.props["data"] = w.datasets()[0].tag
        ev.props["solnum"] = "all"
        ev.props["exprs"] = [{"expr": "aveop1(T)", "unit": "", "descr": "Average temperature"}]
        w.evaluate_node(ev)
        w.add_plot_group("pg1d")
        pg = w.tree.current()
        for c in list(pg.children):
            pg.remove(c)
        gg = w.create_node("plot.globalgraph", pg)
        gg.props.update({"data": w.datasets()[0].tag, "expr": "aveop1(T)", "reduce": "global"})
        w.tree.select(pg)
        w.plot(pg)
        QTimer.singleShot(1500, lambda: (shot("7_aveop_vs_wbb"), finish()))

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
