"""Drive the real GUI through a workflow and save window screenshots (visual regression aid).

    python tools/screenshot_tour.py [example] [outdir]
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def main():
    import warnings
    warnings.filterwarnings("ignore")
    example =sys.argv[1] if len(sys.argv) > 1 else "busbar"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "docs", "screenshots")
    os.makedirs(out, exist_ok=True)
    app = QApplication(sys.argv[:1])
    from elmerstudio.ui import theme
    theme.apply(app)
    from elmerstudio.ui.main_window import MainWindow
    w = MainWindow(None, show_new=False)
    w.resize(1600, 960)
    w.show()
    steps = []
    log = []

    def shot(name):
        """Widget-tree grab (works on a locked/headless desktop) + VTK framebuffer composited in."""
        from PySide6.QtCore import QPoint, QRectF
        from PySide6.QtGui import QImage, QPainter
        app.processEvents()
        pm = w.grab()
        pl = w.graphics.plotter
        if pl is not None and w.gstack.currentIndex() == 0 and w.central.currentIndex() == 0:
            pl.render()
            img = pl.screenshot(return_img=True)
            if img is not None:
                h, wd = img.shape[:2]
                qi = QImage(img.copy().data, wd, h, 3 * wd, QImage.Format_RGB888).copy()
                widget = pl.interactor
                pos = widget.mapTo(w, QPoint(0, 0))
                p = QPainter(pm)
                p.setRenderHint(QPainter.SmoothPixmapTransform)
                p.drawImage(QRectF(pos.x(), pos.y(), widget.width(), widget.height()), qi)
                p.end()
        path = os.path.join(out, f"{example}_{name}.png")
        pm.save(path)
        log.append(path)

    def wait_idle(next_fn, timeout=600):
        t0 = time.time()

        def poll():
            if not w.busy():
                QTimer.singleShot(400, next_fn)
            elif time.time() - t0 > timeout:
                print("timeout")
                app.quit()
            else:
                QTimer.singleShot(200, poll)
        QTimer.singleShot(300, poll)

    def s1():
        w.open_example(example)
        wait_idle(s2)

    def s2():
        shot("1_geometry")
        ph = next((c for c in w.model.component.children if c.kind.startswith("physics.")), None)
        bnd = next((f for f in ph.children if f.selection is not None and f.selection.level == "boundary"
                    and not f.meta.get("default")), None) if ph else None
        if bnd is not None:
            w.tree.select(bnd)
        QTimer.singleShot(1200, s3)

    def s3():
        shot("2_physics_selection")
        mat = next((c for c in w.model.component.child("materials").children), None)
        if mat is not None:
            w.tree.select(mat)
        QTimer.singleShot(1000, s4)

    def s4():
        shot("3_material")
        w.tree.select(w.model.component.child("mesh"))
        w.build_mesh()
        wait_idle(s5)

    def s5():
        shot("4_mesh")
        st = w.model.studies()[0]
        w.tree.select(st)
        w.compute(st)
        wait_idle(s6, timeout=900)

    def s6():
        QTimer.singleShot(1500, lambda: (shot("5_results"), s7()))

    def s7():
        pgs = w.plot_groups()
        if len(pgs) > 1:
            w.tree.select(pgs[1])
            QTimer.singleShot(1500, lambda: (shot("6_results2"), finish()))
        else:
            finish()

    def finish():
        w.dirty = False
        print("\n".join(log))
        sys.stdout.flush()
        QTimer.singleShot(300, app.quit)

    QTimer.singleShot(800, s1)
    app.exec()


if __name__ == "__main__":
    main()
