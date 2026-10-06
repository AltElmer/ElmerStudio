"""Drive the real main window through most commands; fail on any exception or error message.

    python tests/test_gui_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMenu, QMessageBox  # noqa: E402

ERRORS: list[str] = []
LOG: list[str] = []


def hook(t, v, tb):
    ERRORS.append("".join(traceback.format_exception(t, v, tb)))


def main():
    sys.excepthook = hook
    app = QApplication(sys.argv[:1])
    from elmerstudio.core import builder as B
    from elmerstudio.core import project
    from elmerstudio.core.model import REGISTRY
    from elmerstudio.ui import theme
    from elmerstudio.ui.main_window import MainWindow
    theme.apply(app)
    # never block on modal UI during the test
    QMenu.exec = lambda self, *a, **k: None
    QDialog.exec = lambda self, *a, **k: 0
    QMessageBox.warning = staticmethod(lambda *a, **k: ERRORS.append(f"warning dialog: {a[2] if len(a) > 2 else a}") or 0)
    QMessageBox.critical = staticmethod(lambda *a, **k: ERRORS.append(f"critical dialog: {a[2] if len(a) > 2 else a}") or 0)
    QMessageBox.information = staticmethod(lambda *a, **k: LOG.append(f"info: {a[2] if len(a) > 2 else a}") or 0)
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    orig_add = w.messages.add

    def msg_add(text, kind="info"):
        if kind == "error":
            ERRORS.append(f"message: {text}")
        LOG.append(f"[{kind}] {text}")
        orig_add(text, kind)
    w.messages.add = msg_add
    w.resize(1500, 900)
    w.show()
    tmp = tempfile.mkdtemp(prefix="es_gui_")
    queue = []

    def step(fn):
        queue.append(fn)
        return fn

    def run_next():
        if not queue:
            finish()
            return
        fn = queue.pop(0)
        print(f"STEP {fn.__name__}", flush=True)
        try:
            fn()
        except Exception:
            ERRORS.append(f"in {fn.__name__}: {traceback.format_exc()}")
        wait_idle(run_next)

    def wait_idle(nxt, timeout=600):
        t0 = time.time()

        def poll():
            app.processEvents()
            if not w.busy():
                QTimer.singleShot(150, nxt)
            elif time.time() - t0 > timeout:
                ERRORS.append("timeout waiting for job")
                finish()
            else:
                QTimer.singleShot(100, poll)
        QTimer.singleShot(100, poll)

    S = {}

    @step
    def new_model():
        m = B.new_model("3D", ["ht", "solid"], "stationary")
        w._set_model(m, None, project.new_workdir())
        B.add_coupling(m, "mp.te")
        w.tree.refresh()

    @step
    def geometry():
        w.act["geom.block"].trigger()
        blk = w.tree.current()
        assert blk.kind == "geom.block", blk
        w.set_prop(blk, "w", "0.1")
        w.set_prop(blk, "d", "0.04")
        w.set_prop(blk, "h", "0.02")
        w.act["geom.cylinder"].trigger()
        cyl = w.tree.current()
        w.set_prop(cyl, "r", "0.008")
        w.set_prop(cyl, "h", "0.02")
        w.set_prop(cyl, "x", "0.05")
        w.set_prop(cyl, "y", "0.02")
        w.act["geom.difference"].trigger()
        dif = w.tree.current()
        assert dif.props["input"] == [blk.tag] and dif.props["tools"] == [cyl.tag], dif.props
        S["dif"] = dif
        w.build_geometry()

    @step
    def check_geometry():
        assert w.geo is not None and w.geo.count("domain") == 1, w.geo
        LOG.append(f"geometry: {w.geo.count('boundary')} boundaries")

    @step
    def physics_and_selection():
        m = w.model
        w.add_library_material("Aluminum")
        ht = m.component.child("physics.ht")
        temp = w.create_node("ht.temperature", ht)
        w.activate_selection(temp)
        left = w.geo.select("boundary", xmax=0)
        right = w.geo.select("boundary", xmin=0.1)
        w.on_entity_clicked("boundary", left[0], False)
        assert temp.selection.entities == left, temp.selection.entities
        w.set_prop(temp, "T0", "350[K]")
        hf = w.create_node("ht.heatflux", ht)
        w.set_selection(hf, entities=right)
        w.set_prop(hf, "type", "convective", rebuild=True)
        sol = m.component.child("physics.solid")
        fx = w.create_node("solid.fixed", sol)
        w.set_selection(fx, entities=left)
        S["temp"] = temp

    @step
    def tree_ops():
        n_undo = len(w.undo_stack)
        temp = S["temp"]
        w.tree.select(temp)
        w.duplicate_node(temp)
        dup = w.tree.current()
        assert dup is not temp and dup.kind == temp.kind
        w.set_enabled(dup, False)
        w.set_enabled(dup, True)
        w.move_node(dup, -1)
        w.delete_node(dup)
        w.undo()
        w.undo()
        w.redo()
        w.redo()
        assert len(w.undo_stack) >= n_undo
        # restore exactly: delete any leftover duplicate
        ht = w.model.component.child("physics.ht")
        temps = [c for c in ht.children if c.kind == "ht.temperature"]
        for extra in temps[1:]:
            w.delete_node(extra)
        S["temp"] = temps[0]

    @step
    def settings_and_menus_for_all_nodes():
        for n in list(w.model.root.walk()):
            print(f"  node {n.kind} {n.tag}", flush=True)
            w.settings.show_node(n)
            print("    settings ok", flush=True)
            w.context_menu(n, w.pos())
            app.processEvents()

    @step
    def mesh():
        w.tree.select(w.model.component.child("mesh"))
        w.set_mesh_size("6")
        w.build_mesh()

    @step
    def mesh_quality():
        assert w.mesh is not None and not w.mesh_stale
        w.mesh_statistics()
        LOG.append(f"mesh: {w.mesh.stats['elements']} elements")

    @step
    def compute():
        w.compute(w.model.studies()[0])

    @step
    def results():
        pgs = w.plot_groups()
        assert pgs, "no default plots"
        LOG.append("default plots: " + ", ".join(p.label for p in pgs))
        for pg in pgs:
            w.tree.select(pg)
            app.processEvents()
        w.add_plot_group("pg3d")
        pg = w.tree.current()
        for k in ("plot.volume", "plot.slice", "plot.isosurface", "plot.arrow", "plot.streamline", "plot.mesh", "plot.line"):
            w.tree.select(pg)
            w.add_plot(k)
            f = w.tree.current()
            if k in ("plot.arrow", "plot.streamline"):
                w.set_prop(f, "expr", "ht.tflux")
            w.plot(pg)
            app.processEvents()
        surf = pg.children[0]
        w.tree.select(surf)
        w.add_plot("plot.deform")
        w.set_prop(surf, "unit", "degC")
        w.set_prop(surf, "expr", "T")
        w.plot(pg)
        w.add_results_child("results.datasets", "dset.cutline")
        cl = w.tree.current()
        w.add_plot_group("pg1d")
        pg1 = w.tree.current()
        pg1.children[0].props["data"] = cl.tag
        pg1.children[0].props["expr"] = "T"
        w.plot(pg1)
        for k in ("eval.volint", "eval.surfint", "eval.avg", "eval.max", "eval.point", "eval.global"):
            w.add_results_child("results.derived", k)
            n = w.tree.current()
            if "exprs" in n.props:
                n.props["exprs"] = [{"expr": "T", "unit": "", "descr": ""}]
            w.evaluate_node(n)
            LOG.append(f"{k}: {[w.table.table.item(0, j).text() for j in range(w.table.table.columnCount())] if w.table.table.rowCount() else 'empty'}")
        for k, ext in (("export.image", "png"), ("export.data", "csv"), ("export.vtu", "vtu"), ("export.report", "html")):
            w.add_results_child("results.export", k)
            n = w.tree.current()
            n.props["file"] = os.path.join(tmp, f"out_{k.split('.')[1]}.{ext}")
            if k == "export.image":
                n.props["source"] = pgs[0].tag
            w.export_node(n)
            assert os.path.isfile(n.props["file"]), n.props["file"]
        w.show_sif()

    @step
    def save_load():
        fn = os.path.join(tmp, "smoke.esm")
        w.file_path = fn
        assert w.file_save()
        w.open_file(fn)
        assert w.solution_of(w._dataset_for(w.plot_groups()[0])) is not None, "solution not restored"

    @step
    def transient_sweep_2d():
        m = B.new_model("2D", ["ht"], "time")
        m.global_defs.child("params").props["table"] = [{"name": "Q", "expr": "1e5[W/m^3]", "descr": ""}]
        w._set_model(m, None, project.new_workdir())
        w.act["geom.rectangle"].trigger()
        r = w.tree.current()
        w.set_prop(r, "w", "0.02")
        w.set_prop(r, "h", "0.01")
        w.add_library_material("Copper")
        ht = m.component.child("physics.ht")
        hs = w.create_node("ht.source", ht)
        w.set_selection(hs, all_=True)
        w.set_prop(hs, "Q0", "Q")
        st = m.studies()[0]
        step = B.study_steps(st)[0]
        w.set_prop(step, "times", "range(0,1,5)")
        w.tree.select(st)
        w.add_sweep()
        sw = w.tree.current()
        w.set_prop(sw, "pname", "Q")
        w.set_prop(sw, "pvalues", "1e5 2e5")
        w.set_prop(sw, "punit", "W/m^3")
        w.compute(st)

    @step
    def transient_results():
        pgs = w.plot_groups()
        assert pgs
        pg = pgs[0]
        sol = w.solution_of(w._dataset_for(pg))
        assert sol is not None and len(sol.frames) == 5, (sol and len(sol.frames))
        for i in range(len(sol.frames)):
            w.set_prop(pg, "solnum", str(i))
            w.plot(pg)
        w.add_plot_group("pg1d")
        pg1 = w.tree.current()
        for c in list(pg1.children):
            pg1.remove(c)
        g = w.create_node("plot.globalgraph", pg1)
        g.props["data"] = w.datasets()[0].tag
        g.props["expr"] = "T"
        w.plot(pg1)
        LOG.append(f"transient frames: {sol.frame_labels()}")

    def finish():
        w.dirty = False
        print("\n".join(LOG[-40:]))
        print(f"\nERRORS: {len(ERRORS)}")
        for e in ERRORS:
            print("-" * 60)
            print(e[-2500:])
        sys.stdout.flush()
        QTimer.singleShot(100, app.quit)

    QTimer.singleShot(300, run_next)
    app.exec()
    sys.exit(1 if ERRORS else 0)


if __name__ == "__main__":
    main()
