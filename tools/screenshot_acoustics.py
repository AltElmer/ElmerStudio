"""Walk the acoustics postprocessing of the COMSOL notebook in the real GUI; save screenshots.

    python tools/screenshot_acoustics.py [outdir]

Stand-in models (see tests/test_tutorial_acoustics.py): a loudspeaker (piston in a closed box) radiating into
air, and the rigid spherical scatterer in a plane wave.
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
    from test_tutorial_acoustics import loudspeaker_model, scatterer_model
    theme.apply(app)
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    redact_messages(w)
    w.resize(1600, 960)
    w.show()
    saved, errors = [], []
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
        path = os.path.join(out, f"tutorial_acoustics_{name}.png")
        pm.save(path)
        saved.append(path)

    def idle(fn, timeout=2400):
        t0 = time.time()

        def poll():
            if not w.busy():
                QTimer.singleShot(600, fn)
            elif time.time() - t0 > timeout:
                errors.append("timeout")
                finish()
            else:
                QTimer.singleShot(300, poll)
        QTimer.singleShot(300, poll)

    def later(fn, ms=1800):
        QTimer.singleShot(ms, fn)

    S = {}
    res = lambda: w.model.results  # noqa: E731
    dsets = lambda: res().child("results.datasets")  # noqa: E731
    sol_ds = lambda: next(d for d in w.datasets() if d.kind == "dset.solution")  # noqa: E731

    # ------------------------------------------------------------------ loudspeaker (3D)
    def s1():
        m, geo, piston, outer = loudspeaker_model(freqs="200 300 450 650 900")
        S.update(piston=piston, outer=outer)
        w._set_model(m, None, project.new_workdir())
        w.compute(m.studies()[0])
        idle(s2)

    def s2():
        w.add_plot_group("pg3d")
        pg = w.tree.current()
        pg.label = "Acoustic Pressure, Isosurfaces"
        for c in list(pg.children):
            pg.remove(c)
        iso = w.create_node("plot.isosurface", pg, select=False)
        iso.props.update(expr="acpr.p_t", levels=8, cmap="Wave")
        w.tree.refresh(select=iso)
        w.plot(pg)
        later(lambda: (shot("1_isosurfaces"), s3()))

    def s3():
        grid = w.create_node("dset.grid3d", dsets(), select=False)
        grid.props.update(data=sol_ds().tag, xmin="-1.5", xmax="1.5", ymin="-1.5", ymax="1.5", zmin="-1.5",
                          zmax="1.5", xres=35, yres=35, zres=35)
        S["grid"] = grid
        w.add_plot_group("pg3d")
        pg = w.tree.current()
        pg.label = "Sound Pressure Level, Grid 3D"
        surf = pg.children[0]
        surf.props.update(expr="acpr.Lp", cmap="RainbowLight")
        sel = w.create_node("plot.selection", surf, select=False)
        sel.selection.entities = [b for b in w.geo.numbers["boundary"] if b not in S["outer"]]
        ms = w.create_node("plot.multislice", pg, select=False)
        ms.props.update(data=grid.tag, expr="acpr.efc1.Lp_pext", nx=1, ny=1, nz=0, colorleg=True)
        w.tree.refresh(select=ms)
        w.plot(pg)
        later(lambda: (shot("2_grid3d_multislice"), s4()), 3000)

    def s4():
        sp = w.create_node("dset.paramcurve", dsets(), select=False)
        sp.props.update(data=sol_ds().tag, smin="0", smax="1", npts=600, xexpr="(0.6+0.6*s)*cos(8*pi*s)",
                        yexpr="(0.6+0.6*s)*sin(8*pi*s)", zexpr="-0.9+1.8*s")
        sp.label = "Spiral"
        w.add_plot_group("pg3d")
        pg = w.tree.current()
        pg.label = "Sound Pressure Level along a spiral"
        surf = pg.children[0]
        surf.props.update(expr="acpr.Lp")
        sel = w.create_node("plot.selection", surf, select=False)
        sel.selection.entities = [b for b in w.geo.numbers["boundary"] if b not in S["outer"]]
        ln = w.create_node("plot.line", pg, select=False)
        ln.props.update(data=sp.tag, expr="acpr.efc1.Lp_pext", ltype="tube", radius="0.012")
        w.tree.refresh(select=ln)
        w.plot(pg)
        later(lambda: (shot("3_parametric_curve_spiral"), s5()), 2500)

    def s5():
        w.add_plot_group("pgpolar")
        pg = w.tree.current()
        pg.label = "Radiation Pattern"
        pg.remove(pg.children[0])
        rp = w.create_node("plot.radiation", pg, select=False)
        rp.props.update(radius="1[m]", plane="xz", a0="0", a1="360", npts=181, params="1 3 5")
        w.tree.refresh(select=rp)
        w.plot(pg)
        later(lambda: (shot("4_radiation_pattern_polar"), s6()))

    def s6():
        w.add_plot_group("pg3d")
        pg = w.tree.current()
        pg.label = "Radiation Pattern 3D"
        for c in list(pg.children):
            pg.remove(c)
        surf = w.create_node("plot.surface", pg, select=False)
        surf.props.update(expr="acpr.Lp", coloring="uniform", ucolor="gray", colorleg=False)
        sel = w.create_node("plot.selection", surf, select=False)
        sel.selection.entities = [b for b in w.geo.numbers["boundary"] if b not in S["outer"]]
        rp = w.create_node("plot.radiation", pg, select=False)
        rp.props.update(radius="1[m]", npts=180, range="30")
        w.tree.refresh(select=rp)
        w.plot(pg)
        later(lambda: (shot("5_radiation_pattern_3d"), s7()), 2500)

    def s7():
        w.add_plot("plot.directivity")
        d = w.tree.current()
        d.props.update(radius="1[m]", npts=91, a0="-180", a1="180", levels=13)
        d.parent.label = "Directivity"
        w.tree.select(d)
        w.plot(d.parent)
        later(lambda: (shot("6_directivity"), c1()), 2500)

    # ------------------------------------------------------------------ spherical scatterer (2D axisymmetric)
    def c1():
        m, geo, inner, outer = scatterer_model()
        w._set_model(m, None, project.new_workdir())
        w.compute(m.studies()[0])
        idle(c2)

    def c2():
        rv = w.create_node("dset.revolution", dsets(), select=False)
        rv.props.update(data=sol_ds().tag, start="-90", angle="270", layers=72)
        w.add_plot_group("pg3d")
        pg = w.tree.current()
        pg.props.update(data=rv.tag, solnum="0")
        pg.label = "Sound Pressure Level, 500 Hz"
        pg.children[0].props.update(expr="acpr.Lp")
        w.tree.refresh(select=pg)
        w.plot(pg)
        later(lambda: (shot("7_scatterer_spl_revolution"), c3()), 2500)

    def c3():
        w.add_plot_group("pgpolar")
        pg = w.tree.current()
        pg.label = "Scattered field at 3 m"
        pg.remove(pg.children[0])
        rp = w.create_node("plot.radiation", pg, select=False)
        rp.props.update(radius="3[m]", a0="-90", a1="90", npts=181, params="1 2")
        w.tree.refresh(select=rp)
        w.plot(pg)
        later(lambda: (shot("8_scatterer_scattered_polar"), finish()))

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
