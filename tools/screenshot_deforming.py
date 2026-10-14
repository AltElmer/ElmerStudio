"""Deforming electrode tutorial in the real GUI (a wavy anode dissolves and levels); save screenshots.

    python tools/screenshot_deforming.py [outdir]

Checks what the tutorial states: the anode amplitude decays as exp(-k coth(kL) hbar) with the mean removed thickness
hbar (linear theory), and a flat-cell estimate of the removal rate.
"""
from __future__ import annotations

import math
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]
warnings.filterwarnings("ignore")

from PySide6.QtCore import QPoint, QRectF, QTimer  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "modules", "ROOT", "images", "screenshots")
    os.makedirs(out, exist_ok=True)
    app = QApplication(sys.argv[:1])
    import numpy as np
    from elmerstudio.core import builder as B
    from elmerstudio.core import project
    from elmerstudio.core.geometry import build_geometry
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
    S = {}

    def grab():
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
        return pm

    def shot(name):
        w.dock_msg.raise_()
        path = os.path.join(out, f"tutorial_deform_{name}.png")
        grab().save(path)
        saved.append(path)

    def idle(fn, timeout=3600):
        t0 = time.time()

        def poll():
            if not w.busy():
                QTimer.singleShot(500, fn)
            elif time.time() - t0 > timeout:
                errors.append("timeout")
                finish()
            else:
                QTimer.singleShot(300, poll)
        QTimer.singleShot(300, poll)

    def later(fn, ms=1200):
        QTimer.singleShot(ms, fn)

    # ------------------------------------------------------------------ the model, as the tutorial builds it
    def s1():
        m = B.new_model("2D", ["cd"], "time")
        m.root.label = "levelling"
        m.global_defs.child("params").props["table"] = [
            {"name": "lam", "expr": "200[um]", "descr": "Wavelength of the surface roughness"},
            {"name": "L", "expr": "300[um]", "descr": "Gap from the mean anode surface to the cathode"},
            {"name": "a0", "expr": "10[um]", "descr": "Initial amplitude of the roughness"},
            {"name": "V", "expr": "50[mV]", "descr": "Cell voltage"},
            {"name": "sigma", "expr": "10[S/m]", "descr": "Electrolyte conductivity"}]
        w._set_model(m, None, project.new_workdir())
        g = m.component.child("geometry")
        g.props["unit"] = "um"
        pc = m.create("geom.paramcurve", g, index=0, xexpr="lam*s", yexpr="a0*cos(2*pi*s)", npts=121)
        pc.label = "Anode profile"
        rest = m.create("geom.polygon", g, index=1, ptype="open", xs="lam lam 0 0", ys="a0 L L a0")
        cs = m.create("geom.tosolid", g, index=2)
        cs.props["input"] = [pc.tag, rest.tag]
        for i, (name, lim) in enumerate({"Anode": dict(ymax="L/2"), "Cathode": dict(ymin="L/2"),
                                         "Left wall": dict(xmax="lam/10"), "Right wall": dict(xmin="0.9*lam")}.items()):
            b = m.create("geom.boxsel", g, index=3 + i, level="boundary", **lim)
            b.label = name
            S[name] = b.tag
        build_geometry(m, display=False)
        cd = m.component.child("physics.cd")
        cd.props["dist"] = "primary"
        cd.child("cd.elyte").props.update(sigma_src="user", sigma="sigma")
        an = m.create("cd.es", cd, phis="0[V]", kin="primary", moving=True)
        an.label = "Anode"
        an.selection.named = S["Anode"]
        S["an"] = an
        ca = m.create("cd.pot", cd, phil0="-V")
        ca.label = "Cathode"
        ca.selection.named = S["Cathode"]
        for wname in ("Left wall", "Right wall"):
            sy = m.create("cd.sym", cd)
            sy.selection.named = S[wname]
        S["dd"] = m.create("dg.ddom", m.component.child("definitions"))
        mesh = m.component.child("mesh")
        mesh.props["sequence"] = "user"
        m.create("mesh.mapped", mesh)
        for name, n in (("Anode", 40), ("Cathode", 40), ("Left wall", 30), ("Right wall", 30)):
            d = m.create("mesh.distribution", mesh, n=n)
            d.selection.named = S[name]
        st = m.studies()[0]
        step = next(c for c in st.children if c.kind == "study.time")
        step.props.update(times="range(0,60,600)")
        S["step"] = step
        w.tree.refresh()
        w.build_geometry()
        idle(lambda: (w.tree.select(an), later(lambda: (shot("1_electrode_surface"), s2()))))

    def s2():
        w.tree.select(S["dd"])
        later(lambda: (shot("2_deforming_domain"), s3()))

    def s3():
        w.tree.select(S["step"])
        later(lambda: (shot("3_time_dependent"), s4()))

    def s4():
        w.compute(w.model.studies()[0])
        idle(lambda: later(s5, 1500))

    def s5():
        pgs = w.plot_groups()
        pot = next((p for p in pgs if "Potential" in p.label), pgs[0])
        w.tree.select(pot)
        w.plot(pot)
        later(lambda: (shot("4_potential_final"), s6()), 1500)

    def s6():
        pg = next((p for p in w.plot_groups() if p.label.startswith("Electrode Shape")), None)
        if pg is None:
            errors.append("no Electrode Shape default plot")
            return s7()
        w.tree.select(pg)
        w.plot(pg)
        later(lambda: (shot("5_electrode_shape"), s7()), 1500)

    def s7():
        m = w.model
        defs = m.component.child("definitions")
        mx = w.create_node("cpl.maximum", defs, select=False)
        mn = w.create_node("cpl.minimum", defs, select=False)
        av = w.create_node("cpl.average", defs, select=False)
        for op in (mx, mn, av):
            op.props["level"] = "boundary"
            op.selection.level, op.selection.all, op.selection.named = "boundary", False, S["Anode"]
        w.build_geometry()
        idle(lambda: s8(mx, mn, av))

    def s8(mx, mn, av):
        res = w.model.results
        ev = w.create_node("eval.global", res.child("results.derived"))
        ev.props["data"] = w.datasets()[0].tag
        ev.props["solnum"] = "all"
        names = tuple((op.get("opname") or op.tag) for op in (mx, mn, av))
        S["ops"] = names
        ev.props["exprs"] = [{"expr": f"({names[0]}(y)-{names[1]}(y))/2", "unit": "um", "descr": "Amplitude"},
                             {"expr": f"-{names[2]}(y)", "unit": "um", "descr": "Mean removed thickness"}]
        w.evaluate_node(ev)
        w.dock_table.raise_()
        tab = w.table.table
        try:
            rows = [[tab.item(r, c).text() for c in range(tab.columnCount())] for r in range(tab.rowCount())]
            checks.append(f"table rows: {len(rows)} first {rows[0]} last {rows[-1]}")
            amp = np.array([float(r[1]) for r in rows])
            hb = np.array([float(r[2]) for r in rows])
            k = 2 * math.pi / 200e-6
            slope = np.polyfit((hb - hb[0]) * 1e-6, np.log(amp), 1)[0]
            checks.append(f"levelling slope {slope:.1f} 1/m vs -k coth(kL) {-k / math.tanh(k * 300e-6):.1f} "
                          f"(ratio {slope / (-k / math.tanh(k * 300e-6)):.4f}); amplitude {amp[0]:.3f} -> {amp[-1]:.3f} um, "
                          f"removed {hb[-1] - hb[0]:.2f} um")
        except Exception as exc:                         # noqa: BLE001
            errors.append(f"table: {exc}")
        w.tree.select(ev)
        later(lambda: (shot_table("6_amplitude_table"), s9()))

    def shot_table(name):
        path = os.path.join(out, f"tutorial_deform_{name}.png")
        grab().save(path)
        saved.append(path)

    def s9():
        w.add_plot_group("pg1d")
        pg = w.tree.current()
        pg.label = "Amplitude"
        for c in list(pg.children):
            pg.remove(c)
        gg = w.create_node("plot.globalgraph", pg)
        mx, mn, _av = S["ops"]
        gg.props.update({"data": w.datasets()[0].tag, "expr": f"({mx}(y)-{mn}(y))/2", "reduce": "global",
                         "marker": "o"})
        pg.props.update({"ylog": True, "ylabel": "Amplitude (m)", "title_type": "manual",
                         "title": "Amplitude of the anode roughness"})
        w.tree.select(pg)
        w.plot(pg)
        later(lambda: (shot("7_amplitude_vs_time"), finish()), 1500)

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
