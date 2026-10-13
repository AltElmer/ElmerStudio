"""Screenshots for the documentation: the Quick Start model built step by step in the real GUI, and an
annotated overview of the desktop.

    python tools/screenshot_docs.py [outdir]

Quick Start model: a copper bar, 20 degC on one end, 1e5 W/m^2 into the other, insulated elsewhere.
The temperature is linear, T(x) = T0 + q0*x/k, so the maximum is T0 + q0*L/k = 45 degC; the script checks it.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
warnings.filterwarnings("ignore")

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "modules", "ROOT", "images", "screenshots")
    os.makedirs(out, exist_ok=True)
    app = QApplication(sys.argv[:1])
    from elmerstudio.core import builder as B
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
    saved, errors, checks = [], [], []
    sys.excepthook = lambda t, v, tb: errors.append(f"{t.__name__}: {v}")

    def grab():
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
        return pm

    def shot(name, pm=None):
        if pm is None and not name.endswith("maximum"):
            w.dock_msg.raise_()
        pm = pm or grab()
        path = os.path.join(out, f"{name}.png")
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

    def later(fn, ms=1200):
        QTimer.singleShot(ms, fn)

    def click(node, level, ents):
        """Select a feature, then click its entities in the Graphics window (what a user does)."""
        w.tree.select(node)
        app.processEvents()
        for e in ents:
            w.on_entity_clicked(level, e, False)

    # ------------------------------------------------------------------------------ quick start
    def s1():
        m = B.new_model("3D", ["ht"], "stationary")        # what the Model Wizard creates: 3D, Heat Transfer, Stationary
        w._set_model(m, None, project.new_workdir())
        w.dock_msg.raise_()                                 # the restored desktop may have another tab in front
        pnode = m.global_defs.child("params")
        pnode.props["table"] = [
            {"name": "L", "expr": "10[cm]", "descr": "Bar length"},
            {"name": "W", "expr": "2[cm]", "descr": "Bar width"},
            {"name": "H", "expr": "1[cm]", "descr": "Bar height"},
            {"name": "T0", "expr": "20[degC]", "descr": "Temperature of the cold end"},
            {"name": "q0", "expr": "1e5[W/m^2]", "descr": "Heat flux into the hot end"}]
        w.tree.select(pnode)
        later(lambda: (shot("qs_1_parameters"), s2()))

    def s2():
        w.act["geom.block"].trigger()
        blk = w.tree.current()
        for k, v in (("w", "L"), ("d", "W"), ("h", "H")):
            w.set_prop(blk, k, v)
        w.build_geometry()
        w.tree.select(blk)
        idle(lambda: (shot("qs_2_block"), s3()))

    def s3():
        w.add_library_material("Copper")
        w.tree.select(w.model.component.child("materials").children[0])
        later(lambda: (shot("qs_3_material"), s4()))

    def s4():
        ht = w.model.component.child("physics.ht")
        geo = w.geo
        cold = geo.select("boundary", xmax=0)
        hot = geo.select("boundary", xmin=float(geo.bbox[3]))
        t = w.create_node("ht.temperature", ht)
        w.set_prop(t, "T0", "T0")
        click(t, "boundary", cold)
        later(lambda: (shot("qs_4_temperature"), s5(ht, hot)))

    def s5(ht, hot):
        q = w.create_node("ht.heatflux", ht)
        w.set_prop(q, "q0", "q0")
        click(q, "boundary", hot)
        pl = w.graphics.plotter
        if pl is not None:                                  # turn the bar so that its far end (x = L) faces us
            pl.view_vector((1.0, -1.1, 0.7), viewup=(0, 0, 1))
            pl.reset_camera()
        later(lambda: (shot("qs_5_heat_flux"), s6()))

    def s6():
        w.tree.select(w.model.component.child("mesh"))
        w.build_mesh()
        idle(lambda: (shot("qs_6_mesh"), s7()))

    def s7():
        st = w.model.studies()[0]
        w.tree.select(st)
        w.compute(st)
        idle(lambda: later(lambda: (shot("qs_7_temperature"), s8()), 1500), timeout=1200)

    def s8():
        w.add_results_child("results.datasets", "dset.cutline")
        cl = w.tree.current()
        for k, v in (("x0", "0"), ("y0", "W/2"), ("z0", "H/2"), ("x1", "L"), ("y1", "W/2"), ("z1", "H/2")):
            w.set_prop(cl, k, v)
        w.add_plot_group("pg1d")
        pg = w.tree.current()
        lg = pg.children[0]
        lg.props.update({"data": cl.tag, "expr": "T", "unit": "degC"})
        w.tree.select(lg)
        w.plot(pg)
        later(lambda: (shot("qs_8_line_graph"), s9()), 1500)

    def s9():
        w.add_results_child("results.derived", "eval.max")
        n = w.tree.current()
        n.props["exprs"] = [{"expr": "T", "unit": "degC", "descr": "Maximum temperature"}]
        w.evaluate_node(n)
        w.dock_table.raise_()
        tab = w.table.table
        try:
            vals = [tab.item(0, j).text() for j in range(tab.columnCount())]
            checks.append(f"Volume Maximum row: {vals}")
            num = next(float(v) for v in vals if _isnum(v))
            if abs(num - 45.0) > 0.05:
                errors.append(f"maximum temperature {num} degC, expected 45")
        except Exception as exc:                       # noqa: BLE001
            errors.append(f"table: {exc}")
        later(lambda: (shot("qs_9_maximum"), s10()))

    # ------------------------------------------------------------------------------ desktop overview
    def s10():
        w.tree.select(w.plot_groups()[0])
        w.dock_msg.raise_()
        later(lambda: overview(), 1500)

    def overview():
        pm = grab()
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)

        orange = QColor(230, 81, 0)
        font = QFont("Segoe UI", 13, QFont.Bold)

        def box(widget, n, text, where, inset=3):
            """Outline a pane and put a numbered label inside it ('bl' bottom-left, 'r' right end, 'c' center)."""
            r = QRect(widget.mapTo(w, QPoint(0, 0)), widget.size()).adjusted(inset, inset, -inset, -inset)
            p.setPen(QPen(orange, 3))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r, 6, 6)
            p.setFont(font)
            label = f"{n}  {text}"
            tw = p.fontMetrics().horizontalAdvance(label) + 28
            th = 34
            if where == "bl":
                pill = QRect(r.left() + 14, r.bottom() - th - 14, tw, th)
            elif where == "r":
                pill = QRect(r.right() - tw - 14, r.center().y() - th // 2, tw, th)
            else:
                pill = QRect(r.center().x() - tw // 2, r.center().y() - th // 2, tw, th)
            p.setPen(Qt.NoPen)
            p.setBrush(orange)
            p.drawRoundedRect(pill, th // 2, th // 2)
            p.setPen(QColor("white"))
            p.drawText(pill, Qt.AlignCenter, label)

        box(w.ribbon, 1, "Ribbon and Quick Access Toolbar", "r", 1)
        box(w.dock_tree, 2, "Model Builder", "bl")
        box(w.dock_settings, 3, "Settings", "bl")
        box(w.graphics.toolbar, 4, "Graphics toolbar", "r", 0)
        box(w.graphics.plotter.interactor if w.graphics.plotter else w.graphics, 5, "Graphics window", "bl")
        box(w.dock_msg, 6, "Messages, Progress, Log, Table", "r")
        p.end()
        shot("ui_overview_annotated", pm)
        tb = w.graphics.toolbar
        r = QRect(tb.mapTo(w, QPoint(0, 0)), tb.size())
        shot("ui_graphics_toolbar", grab().copy(r.adjusted(0, 0, -max(0, r.width() - 520), 0)))
        finish()

    def finish():
        w.dirty = False
        print("\n".join(saved))
        print("\n".join(checks))
        print("ERRORS:", errors)
        sys.stdout.flush()
        QTimer.singleShot(200, app.quit)

    QTimer.singleShot(500, s1)
    app.exec()


def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
