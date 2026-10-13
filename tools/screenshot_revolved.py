"""Derived Values in the revolved geometry (2D axisymmetric) in the real GUI; save screenshots.

    python tools/screenshot_revolved.py [outdir]

A cylinder r < R, 0 < z < H of electrolyte (primary current distribution) with a disc electrode r < R/2 at z = 0
and the counter electrode on the top (the test, tests/test_electrochemistry.py::test_revolved_integration_
axisymmetric, uses the whole bottom for a closed form).  Shows the "Compute integral in revolved geometry" setting,
the Surface Integration of cd.iloc over the electrode next to the electrode current cd.es1.Itot (with the current
density and potential plotted), and Volume Integration of 1 with the setting checked (pi R^2 H) and unchecked by a
click (R H), with the cylinder drawn as a Revolution 2D dataset.
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
from PySide6.QtWidgets import QApplication, QCheckBox  # noqa: E402

R, H, SIGMA, V = 2e-3, 5e-3, 5.0, 0.1
LABEL = "Compute integral in revolved geometry"


def cylinder_model():
    from elmerstudio.core import builder as B
    from elmerstudio.core.geometry import build_geometry
    m = B.new_model("2Daxi", ["cd"], "stationary")
    m.create("geom.polygon", m.component.child("geometry"), index=0,          # the vertex at r = R/2 splits the
             xs=f"0 {R / 2} {R} {R} 0", ys=f"0 0 0 {H} {H}")                 # bottom into disc and insulator
    geo = build_geometry(m, display=False)
    cd = m.component.child("physics.cd")
    cd.props["dist"] = "primary"
    cd.child("cd.elyte").props.update(sigma_src="user", sigma=str(SIGMA))
    bottom = geo.select("boundary", ymax=1e-9, xmax=R / 2 * (1 + 1e-6))
    if len(bottom) != 1 or len(geo.select("boundary", ymax=1e-9)) != 2:
        raise RuntimeError(f"expected a split bottom boundary, got {bottom}")
    an = B.add_feature(m, cd, "cd.es", bottom, phis=str(V))
    B.add_feature(m, cd, "cd.pot", geo.select("boundary", ymin=H - 1e-9), phil0="0")
    return m, bottom, an


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "screenshots")
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
    saved, errors, checks = [], [], []
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
        path = os.path.join(out, f"revolved_{name}.png")
        pm.save(path)
        saved.append(path)

    def idle(fn, timeout=600):
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

    def checkbox():
        boxes = [b for b in w.settings.findChildren(QCheckBox) if b.text() == LABEL and b.isVisible()]
        return boxes[0] if boxes else None

    def last_value():
        return float(w.tables()[-1].props["rows"][-1][-1])

    S = {}
    derived = lambda: w.model.results.child("results.derived")  # noqa: E731
    dsets = lambda: w.model.results.child("results.datasets")  # noqa: E731
    sol_ds = lambda: next(d for d in w.datasets() if d.kind == "dset.solution")  # noqa: E731

    def show(node, pg):
        """Evaluate *node* and keep its settings in view while the Graphics window shows the results plot *pg*.
        Plot last: showing a node with a selection activates it, which redraws the bare geometry."""
        w.tree.refresh(select=node)
        w.evaluate_node(node)
        w.settings.show_node(node)
        app.processEvents()
        w.plot(pg)

    def s1():
        m, bottom, an = cylinder_model()
        S.update(bottom=bottom, an=an)
        w._set_model(m, None, project.new_workdir())
        w.compute(m.studies()[0])
        idle(s2)

    def s2():
        pg = next(p for p in w.model.results.children if p.kind == "pg2d" and "Current Density" in p.label)
        pg.children[0].props.update(expr="cd.IlMag", unit="A/m^2", cmap="ThermalLight")
        con = w.create_node("plot.contour", pg, select=False)            # equipotentials of the electrolyte
        con.props.update(expr="cd.phil", unit="V", levels=12, coloring="uniform", ucolor="black", colorleg=False)
        S["pg2d"] = pg
        si = w.create_node("eval.surfint", derived())
        si.selection.all, si.selection.entities = False, list(S["bottom"])
        si.props["exprs"] = [{"expr": "cd.iloc", "unit": "", "descr": "Electrode current (revolved)"}]
        si.label = "Surface Integration: electrode current"
        show(si, pg)
        cb = checkbox()
        checks.append(f"surfint checkbox shown and checked: {cb is not None and cb.isChecked()}")
        checks.append(f"surfint column: {w.tables()[-1].props['columns'][-1]}")
        S["iint"] = last_value()
        later(lambda: (shot("1_surface_integration_cd_iloc"), s3()))

    def s3():
        ge = w.create_node("eval.global", derived())
        ge.props["exprs"] = [{"expr": f"cd.{S['an'].tag}.Itot", "unit": "", "descr": "Electrode current (reaction loads)"}]
        show(ge, S["pg2d"])
        S["itot"] = last_value()
        checks.append(f"Surface Integration of cd.iloc = {S['iint']:.9g} A; cd.{S['an'].tag}.Itot = {S['itot']:.9g} A")
        later(lambda: (shot("2_global_evaluation_itot"), s4()))

    def s4():
        rv = w.create_node("dset.revolution", dsets(), select=False)
        rv.props.update(data=sol_ds().tag, start="-90", angle="270", layers=48)     # cut away a quarter
        w.add_plot_group("pg3d")
        pg = w.tree.current()
        pg.props["data"] = rv.tag
        pg.label = "Revolved geometry: electrolyte potential"
        pg.children[0].props.update(expr="cd.phil", unit="V", cmap="RainbowLight")
        S["pg3d"] = pg
        vi = w.create_node("eval.volint", derived())
        vi.props["exprs"] = [{"expr": "1", "unit": "", "descr": "Volume"}]
        show(vi, pg)
        S["vi"] = vi
        checks.append(f"revolved volume {last_value():.9g} m^3 (pi R^2 H = {3.141592653589793 * R * R * H:.9g})")
        later(lambda: (shot("3_volume_integration_revolved"), s5()))

    def s5():
        cb = checkbox()
        if cb is None:
            errors.append("checkbox not found on the Volume Integration settings")
            return finish()
        cb.click()                                     # through the Settings window, like a user
        app.processEvents()
        checks.append(f"after the click, node revolved = {S['vi'].get('revolved')}")
        w.evaluate_node(S["vi"])
        app.processEvents()
        w.plot(S["pg2d"])                              # unchecked: the r-z plane is what gets integrated
        checks.append(f"r-z plane area {last_value():.9g} m^2 (R H = {R * H:.9g})")
        later(lambda: (shot("4_volume_integration_unchecked"), finish()))

    def finish():
        w.dirty = False
        print("\n".join(saved))
        print("\n".join(checks))
        print("ERRORS:", errors)
        sys.stdout.flush()
        QTimer.singleShot(200, app.quit)

    QTimer.singleShot(500, s1)
    app.exec()


if __name__ == "__main__":
    main()
