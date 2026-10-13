"""Walk the COMSOL Wrench tutorial (stationary solid mechanics, mesh convergence) in the real GUI and save screenshots.

    python tools/screenshot_wrench.py [outdir]
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
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "modules", "ROOT", "images", "screenshots")
    os.makedirs(out, exist_ok=True)
    app = QApplication(sys.argv[:1])
    from elmerstudio.core import project
    from elmerstudio.ui import theme
    from elmerstudio.ui.main_window import MainWindow
    from screenshot_tour import redact_messages
    from test_tutorial_wrench import build_model
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
        w.settings.refresh()              # show the values this script set, not the ones the window was opened with
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
        path = os.path.join(out, f"tutorial_wrench_{name}.png")
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

    def s1():
        m, _geo, _socket, _end = build_model(convergence=True)
        w._set_model(m, None, project.new_workdir())
        w.build_geometry()
        w.tree.select(m.component.child("geometry"))
        idle(lambda: (shot("1_geometry"), s2()))

    def s2():
        solid = w.model.component.child("physics.solid")
        fixed = next(c for c in solid.children if c.kind == "solid.fixed")
        w.tree.select(fixed)                         # the six socket faces highlighted
        QTimer.singleShot(1500, lambda: (shot("2_fixed_constraint"), s3()))

    def s3():
        solid = w.model.component.child("physics.solid")
        load = next(c for c in solid.children if c.kind == "solid.bload")
        w.tree.select(load)                          # Total force on the handle end
        QTimer.singleShot(1200, lambda: (face_the(load), QTimer.singleShot(800, lambda: (
            shot("3_boundary_load_total_force"), s4()))))

    def face_the(node):
        """Turn the camera so that the node's selected faces point at the viewer (they face away by default)."""
        import numpy as np
        geo, pl = w.geo, w.graphics.plotter
        ents = list(node.selection.entities) if node.selection is not None else []
        if pl is None or geo is None or not ents:
            return
        bb = np.array([geo.bbox_of["boundary"][e] for e in ents])
        c_sel = (bb[:, :3].min(0) + bb[:, 3:].max(0)) / 2
        c_all = (np.array(geo.bbox[:3]) + np.array(geo.bbox[3:])) / 2
        d = c_sel - c_all
        d = d / (np.linalg.norm(d) or 1.0) + np.array([0.3, -0.6, 0.6])
        pl.view_vector(tuple(d), viewup=(0, 0, 1))
        pl.reset_camera()
        pl.render()

    def s4():
        mesh = w.model.component.child("mesh")
        size = next(c for c in mesh.children if c.kind == "mesh.size")
        w.build_mesh(then=lambda: None)
        # select the Size node first, then draw the mesh once its selection highlight has been shown
        idle(lambda: (w.tree.select(size), QTimer.singleShot(1000, lambda: (
            w.show_mesh(), QTimer.singleShot(1500, lambda: (shot("4_mesh_size"), s5()))))))

    def s5():
        w.compute(w.model.studies()[0])
        idle(lambda: (s5b()), timeout=1800)

    def s5b():
        pg = next((p for p in w.plot_groups() if "Stress" in p.label), None)
        if pg is not None:
            w.tree.select(pg)
            w.plot(pg)
        QTimer.singleShot(1500, lambda: (shot("5_stress_von_mises"), s6()))

    def s6():
        w.add_plot_group("pg3d")
        pg = w.tree.current()
        pg.label = "Safety"
        surf = next(c for c in pg.children if c.kind == "plot.surface")
        surf.props["expr"] = "solid.mises > 200[MPa]"
        surf.props["unit"] = ""
        w.tree.refresh(select=pg)
        w.plot(pg)
        QTimer.singleShot(1500, lambda: (shot("6_safety_plot"), s7()))

    def s7():
        res = w.model.results
        ev = w.create_node("eval.max", res.child("results.derived"))
        ev.props["data"] = w.datasets()[0].tag
        ev.props["solnum"] = "all"
        ev.props["exprs"] = [{"expr": "solid.mises", "unit": "MPa", "descr": "von Mises stress"}]
        w.evaluate_node(ev)
        w.tree.select(ev)
        QTimer.singleShot(1500, lambda: (shot("7_volume_maximum"), s8()))

    def s8():
        w.add_plot_group("pg1d")
        pg = w.tree.current()
        for c in list(pg.children):
            pg.remove(c)
        gg = w.create_node("plot.globalgraph", pg)
        gg.props.update({"data": w.datasets()[0].tag, "expr": "w", "reduce": "min"})
        w.tree.select(pg)
        w.plot(pg)
        QTimer.singleShot(1500, lambda: (shot("8_tip_deflection_vs_hd"), finish()))

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
