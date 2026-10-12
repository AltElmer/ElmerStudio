"""Walk the COMSOL 'Essentials of Postprocessing and Visualization' workflow in the real GUI; save screenshots.

    python tools/screenshot_postprocessing.py [outdir]

Stand-in models (see tests/test_tutorial_postprocessing.py): a spinning pulley (2D, parametric sweep over n),
the heat sink example (3D) with a sweep over the dissipated power, and the axisymmetric coil example.
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
    from elmerstudio.core import project
    from elmerstudio.core.examples import coil_axi, heat_sink
    from elmerstudio.ui import theme
    from elmerstudio.ui.main_window import MainWindow
    from screenshot_tour import redact_messages
    from test_tutorial_postprocessing import pulley_model
    theme.apply(app)
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    redact_messages(w)
    w.resize(1600, 960)
    w.show()
    saved = []
    errors = []
    sys.excepthook = lambda t, v, tb: errors.append(f"{t.__name__}: {v}")
    w.messages.add_orig = w.messages.add

    def watch(msg, level="info", *a, **k):
        if level == "error":
            errors.append(msg)
        return w.messages.add_orig(msg, level, *a, **k)
    w.messages.add = watch

    def shot(name):
        from PySide6.QtCore import QPoint, QRectF
        from PySide6.QtGui import QImage, QPainter
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
        path = os.path.join(out, f"tutorial_post_{name}.png")
        pm.save(path)
        saved.append(path)

    def idle(fn, timeout=1800):
        t0 = time.time()

        def poll():
            if not w.busy():
                QTimer.singleShot(600, fn)
            elif time.time() - t0 > timeout:
                errors.append("timeout")
                finish()
            else:
                QTimer.singleShot(250, poll)
        QTimer.singleShot(250, poll)

    def later(fn, ms=1500):
        QTimer.singleShot(ms, fn)

    S = {}
    res = lambda: w.model.results  # noqa: E731
    derived = lambda: res().child("results.derived")  # noqa: E731
    dsets = lambda: res().child("results.datasets")  # noqa: E731
    sol_ds = lambda: next(d for d in w.datasets() if d.kind == "dset.solution")  # noqa: E731

    # ------------------------------------------------------------------ pulley
    def p1():
        m, geo, hub, right, left = pulley_model()
        S.update(right=right, left=left)
        w._set_model(m, None, project.new_workdir())
        w.compute(m.studies()[0])
        idle(lambda: (shot("pulley_1_stress"), p2()), timeout=2400)

    def p2():
        mx = w.create_node("eval.max", derived())
        mx.props.update(solnum="all", exprs=[{"expr": "solid.disp", "unit": "mm", "descr": ""},
                                             {"expr": "solid.mises", "unit": "MPa", "descr": ""}])
        mx.label = "Surface Maximum 1"
        w.settings.show_node(mx)
        w.evaluate_node(mx)
        later(lambda: (shot("pulley_2_surface_maximum"), p3()))

    def p3():
        cp = w.create_node("dset.cutpoint", dsets(), select=False)
        cp.props.update(x="19", y="54")
        w.add_plot_group("pg1d")
        pg = w.tree.current()
        for c in list(pg.children):
            pg.remove(c)
        g = w.create_node("plot.pointgraph", pg)
        g.props.update(data=cp.tag, expr="solid.mises", unit="MPa", marker="o")
        pg.label = "Stress at Cut Point"
        w.tree.refresh(select=g)
        w.plot(pg)
        later(lambda: (shot("pulley_3_point_graph"), p4()))

    def p4():
        pe = w.create_node("eval.point", derived())
        pe.selection.entities = S["right"] + S["left"]
        pe.props.update(solnum="all", exprs=[{"expr": "u", "unit": "mm", "descr": "Radial displacement"}])
        w.tree.refresh(select=pe)
        w.evaluate_node(pe)
        later(lambda: (shot("pulley_4_point_evaluation"), p5()))

    def p5():
        ge = w.create_node("eval.global", derived())
        ge.props.update(solnum="all", exprs=[{"expr": "diam-2*R", "unit": "", "descr": "Diameter change"}])
        w.evaluate_node(ge)
        tbl = w.tables()[-1]
        w.add_plot("plot.tablegraph")
        g = w.tree.current()
        g.props.update(table=tbl.tag, xcol=1, marker="o")
        pg = g.parent
        pg.props.update(ylabel="Diameter change (m)")
        w.tree.select(g)
        w.plot(pg)
        later(lambda: (shot("pulley_5_table_graph"), p6()))

    def p6():
        w.add_plot_group("pg2d")
        pg = w.tree.current()
        pg.label = "Stress with annotations"
        surf = pg.children[0]
        surf.props.update(expr="solid.mises", unit="MPa")
        con = w.create_node("plot.contour", pg, select=False)
        con.props.update(expr="solid.mises", unit="MPa", levels=8, labels=True, coloring="uniform", ucolor="black",
                         colorleg=False)
        arr = w.create_node("plot.arrow", pg, select=False)
        arr.props.update(expr="solid.u_vec", xg=14, yg=14, color="black")
        ann = w.create_node("plot.annotation", pg, select=False)
        ann.props.update(text="eval(solid.mises, MPa)", x="19", y="54")
        w.tree.refresh(select=ann)
        w.plot(pg)
        later(lambda: (shot("pulley_6_annotation_contours_arrows"), h1()))

    # ------------------------------------------------------------------ heat sink (3D)
    def h1():
        m = heat_sink()
        st = m.studies()[0]
        m.create("study.sweep", st, index=0, pname="P0", pvalues="range(5,5,20)", punit="W")
        w._set_model(m, None, project.new_workdir())
        w.compute(st)
        idle(lambda: h2(), timeout=2400)

    def h2():
        w.add_plot_group("pg3d")
        pg = w.tree.current()
        pg.label = "Isosurfaces and streamlines"
        pg.children[0].props.update(expr="T", unit="degC", wireframe=True, coloring="uniform", ucolor="gray",
                                    colorleg=False)
        iso = w.create_node("plot.isosurface", pg, select=False)
        iso.props.update(expr="T", unit="degC", levels=6)
        stl = w.create_node("plot.streamline", pg, select=False)
        stl.props.update(expr="ht.tflux", ltype="tube", nseeds=30, positioning="start")
        sel = w.create_node("plot.selection", stl, select=False)
        sel.selection.entities = [b for b in w.geo.numbers["boundary"] if w.geo.bbox_of["boundary"][b][5] < 1e-3]
        w.tree.refresh(select=pg)
        w.plot(pg)
        later(lambda: (shot("heatsink_7_isosurface_streamlines"), h3()))

    def h3():
        cpl = w.create_node("dset.cutplane", dsets(), select=False)
        cpl.props.update(data=sol_ds().tag, ptype="quick", plane="zx", coord="20")
        w.add_plot_group("pg2d")
        pg = w.tree.current()
        pg.props["data"] = cpl.tag
        pg.label = "Cut plane y = 20 mm"
        pg.children[0].props.update(expr="T", unit="degC", cmap="ThermalLight")
        arr = w.create_node("plot.arrow", pg, select=False)
        arr.props.update(expr="ht.tflux", xg=24, yg=18, color="black")
        S["cutplane"] = cpl
        w.tree.refresh(select=pg)
        w.plot(pg)
        later(lambda: (shot("heatsink_8_cut_plane"), h4()))

    def h4():
        pe = w.create_node("dset.parext", dsets(), select=False)
        pe.props.update(data=S["cutplane"].tag, levelscale="1")
        w.add_plot_group("pg3d")
        pg = w.tree.current()
        pg.props["data"] = pe.tag
        pg.label = "Parametric extrusion over P0"
        pg.children[0].props.update(expr="T", unit="degC", cmap="ThermalLight")
        w.tree.refresh(select=pg)
        w.plot(pg)
        later(lambda: (shot("heatsink_9_parametric_extrusion"), c1()))

    # ------------------------------------------------------------------ coil (2D axisymmetric)
    def c1():
        m = coil_axi()
        w._set_model(m, None, project.new_workdir())
        w.compute(m.studies()[0])
        idle(lambda: c2(), timeout=2400)

    def c2():
        rv = w.create_node("dset.revolution", dsets(), select=False)
        rv.props.update(data=sol_ds().tag, start="-90", angle="225", layers=48)
        w.add_plot_group("pg3d")
        pg = w.tree.current()
        pg.props["data"] = rv.tag
        pg.label = "Revolution 2D"
        surf = pg.children[0]
        surf.props.update(expr="mf.normB", unit="mT")
        sel = w.create_node("plot.selection", surf, select=False)      # core and coil only (hide the air)
        sel.props["level"] = sel.selection.level = "domain"
        dom = w.geo.numbers["domain"]
        big = max(dom, key=lambda d: w.geo.bbox_of["domain"][d][3] - w.geo.bbox_of["domain"][d][0])
        sel.selection.entities = [d for d in dom if d != big]
        w.tree.refresh(select=pg)
        w.plot(pg)
        later(lambda: (shot("coil_10_revolution"), c3()))

    def c3():
        w.add_plot_group("pgpolar")
        pg = w.tree.current()
        pol = pg.children[0]
        pol.props.update(expr="mf.normB", unit="mT", xc="0", yc="0", radius="40", a0="-90", a1="90", npts=181)
        pg.label = "Polar plot |B| on r = 40 mm"
        w.tree.refresh(select=pol)
        w.plot(pg)
        later(lambda: (shot("coil_11_polar"), finish()))

    def finish():
        w.dirty = False
        print("\n".join(saved))
        print("ERRORS:", errors)
        sys.stdout.flush()
        QTimer.singleShot(200, app.quit)

    QTimer.singleShot(500, p1)
    app.exec()


if __name__ == "__main__":
    main()
