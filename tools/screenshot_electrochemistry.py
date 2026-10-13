"""Electrochemistry tutorials in the real GUI; save screenshots for docs/modules/tutorials/pages/electrochemistry.adoc.

    python tools/screenshot_electrochemistry.py [outdir]

1. The 1D cell of tests/test_electrochemistry.py (strip 100 um x 200 um, Butler-Volmer anode at the bottom, electrolyte
   potential 0 on the top) with a Parametric Sweep of the electrode potential Vapp = range(0.05,0.05,0.25) V:
   Electrode Surface settings, the electrolyte potential, a Global Evaluation of cd.es1.Itot over the sweep and the
   polarization curve as a Global plot.
2. Newman's disc electrode (2D axisymmetric, primary): quarter circle R = 30 mm united with a rectangle that splits
   the bottom at r = a = 1 mm; the user mesh of the test; the electrolyte potential and the electrode current.
The closed forms are printed next to the values the GUI computed.
"""
from __future__ import annotations

import math
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "tools")]
warnings.filterwarnings("ignore")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

A_DISC, R_FAR, SIG_DISC, V_DISC = 1e-3, 30e-3, 5.0, 1.0


def disc_model():
    """Newman's disc, as in tests/test_electrochemistry.py::test_newman_disc_axisymmetric."""
    from elmerstudio.core import builder as B
    from elmerstudio.core.geometry import build_geometry
    a, Rf = A_DISC, R_FAR
    m = B.new_model("2Daxi", ["cd"], "stationary")
    g = m.component.child("geometry")
    q = m.create("geom.circle", g, index=0, r=str(Rf), sector="90")
    d = m.create("geom.rectangle", g, index=1, w=str(a), h=str(a / 4))
    m.create("geom.union", g, index=2, input=[q.tag, d.tag])
    geo = build_geometry(m, display=False)
    cd = m.component.child("physics.cd")
    cd.props["dist"] = "primary"
    cd.child("cd.elyte").props.update(sigma_src="user", sigma=str(SIG_DISC))
    bb = geo.bbox_of["boundary"]
    disc = [b for b in geo.numbers["boundary"] if bb[b][4] < 1e-9 and bb[b][3] <= a + 1e-9]
    far = [b for b in geo.numbers["boundary"] if bb[b][3] > 0.5 * Rf and bb[b][4] > 0.5 * Rf]
    an = B.add_feature(m, cd, "cd.es", disc, phis=str(V_DISC))
    B.add_feature(m, cd, "cd.pot", far, phil0="0")
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.size", mesh, mode="custom", hmax=str(Rf / 12), hmin=str(a / 400), hcurve="0.3", hnarrow="1")
    edge = [p for p in geo.numbers["point"]
            if abs(geo.bbox_of["point"][p][0] - a) < 1e-6 and geo.bbox_of["point"][p][1] < 1e-6]
    s = m.create("mesh.size", mesh, scope="point", mode="custom", hmax=str(a / 200), hmin=str(a / 400))
    s.selection.level, s.selection.all, s.selection.entities = "point", False, edge
    s2 = m.create("mesh.size", mesh, scope="boundary", mode="custom", hmax=str(a / 25), hmin=str(a / 400))
    s2.selection.level, s2.selection.all, s2.selection.entities = "boundary", False, disc
    m.create("mesh.ftri", mesh)
    return m, an, disc, far, edge


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "modules", "ROOT", "images", "screenshots")
    os.makedirs(out, exist_ok=True)
    app = QApplication(sys.argv[:1])
    from elmerstudio.core import project
    from elmerstudio.ui import theme
    from elmerstudio.ui.main_window import MainWindow
    from screenshot_tour import redact_messages
    from test_electrochemistry import KINETICS, exact_current, strip
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
        path = os.path.join(out, f"tutorial_echem_{name}.png")
        pm.save(path)
        saved.append(path)

    def idle(fn, timeout=900):
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
    derived = lambda: w.model.results.child("results.derived")  # noqa: E731
    sol_ds = lambda: next(d for d in w.datasets() if d.kind == "dset.solution")  # noqa: E731

    def show(node, pg):
        w.tree.refresh(select=node)
        w.evaluate_node(node)
        w.settings.show_node(node)
        app.processEvents()
        w.plot(pg)

    # ------------------------------------------------------------------ 1D cell, Butler-Volmer, polarization sweep
    def c1():
        m, _geo, an = strip("bv", V=0.1, i0=50.0, sigma=20.0, W=1e-4, L=2e-4)
        S["an"] = an
        w._set_model(m, None, project.new_workdir())
        w.build_geometry()
        idle(c2)

    def c2():
        w.tree.select(S["an"])
        later(lambda: (shot("1_cell_electrode_surface"), c3()), 3000)

    def c3():
        st = w.model.studies()[0]
        w.create_node("study.sweep", st, index=0, select=False, pname="Vapp", pvalues="range(0.05,0.05,0.25)", punit="V")
        w.tree.refresh()
        w.compute(st)
        idle(c4)

    def c4():
        pg = next(p for p in w.model.results.children if p.kind == "pg2d" and "Potential" in p.label)
        S["pg_cell"] = pg
        ge = w.create_node("eval.global", derived())
        ge.props["data"] = sol_ds().tag
        ge.props["solnum"] = "all"
        ge.props["exprs"] = [{"expr": f"cd.{S['an'].tag}.Itot", "unit": "", "descr": "Anode current per unit depth"}]
        show(ge, pg)
        rows = w.tables()[-1].props["rows"]
        for r in rows:
            try:
                v, cur = float(r[0]), float(r[-1])
            except (TypeError, ValueError):
                continue
            ex = exact_current(KINETICS["bv"], v, 2e-4, 20.0) * 1e-4
            checks.append(f"Vapp={v:g} V: Itot={cur:.9g} A/m, exact {ex:.9g}, rel {cur / ex - 1:.2e}")
        later(lambda: (shot("2_cell_polarization_table"), c5()))

    def c5():
        w.add_plot_group("pg1d")
        pg = w.tree.current()
        pg.label = "Polarization curve"
        for c in list(pg.children):
            pg.remove(c)
        gg = w.create_node("plot.globalgraph", pg)
        gg.props.update({"data": sol_ds().tag, "expr": f"cd.{S['an'].tag}.Itot", "reduce": "global",
                         "marker": "o"})
        w.tree.refresh(select=gg)
        w.plot(pg)
        later(lambda: (shot("3_cell_polarization_curve"), d1()))

    # ------------------------------------------------------------------ Newman's disc electrode
    def d1():
        m, an, disc, far, edge = disc_model()
        S.update(dan=an, disc=disc)
        w.dirty = False
        w._set_model(m, None, project.new_workdir())
        w.build_geometry()
        idle(d2)

    def d2():
        w.tree.select(w.model.component.child("physics.cd"))     # Current distribution type: Primary
        later(lambda: (shot("4_disc_primary_interface"), d3()))

    def d3():
        w.build_mesh()
        idle(lambda: later(lambda: (shot("5_disc_mesh"), d4())))

    def d4():
        w.compute(w.model.studies()[0])
        idle(d5)

    def d5():
        pg = next(p for p in w.model.results.children if p.kind == "pg2d" and "Potential" in p.label)
        ge = w.create_node("eval.global", derived())
        ge.props["exprs"] = [{"expr": f"cd.{S['dan'].tag}.Itot", "unit": "", "descr": "Disc current"},
                             {"expr": f"{V_DISC}/cd.{S['dan'].tag}.Itot", "unit": "", "descr": "Resistance V/I"}]
        show(ge, pg)
        row = w.tables()[-1].props["rows"][-1]
        I = float(row[-2]) if len(row) > 1 else float("nan")
        Rex = 1 / (4 * SIG_DISC * A_DISC) - 1 / (2 * math.pi * SIG_DISC * R_FAR)
        checks.append(f"disc: row {row}; I = {I:.9g} A, R = {V_DISC / I:.9g} ohm, exact {Rex:.9g}, "
                      f"rel {V_DISC / I / Rex - 1:.2e}")
        later(lambda: (shot("6_disc_potential_current"), finish()))

    def finish():
        w.dirty = False
        print("\n".join(saved))
        print("\n".join(checks))
        print("ERRORS:", errors)
        sys.stdout.flush()
        QTimer.singleShot(200, app.quit)

    QTimer.singleShot(500, c1)
    app.exec()


if __name__ == "__main__":
    main()
