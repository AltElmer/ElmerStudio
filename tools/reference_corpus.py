"""Capture the desktop reference corpus: the twelve task states of the COMSOL-parity review, at fixed window sizes.

    python tools/reference_corpus.py [outdir] [--sizes 1366x768,1920x1080]
    QT_SCALE_FACTOR=1.5 python tools/reference_corpus.py out150     # another display scaling

For every state it writes a screenshot and a JSON record: the selected node, the active entity level, the enabled
commands, the focus, the model revision, the expected next action, and the COMSOL documentation page that describes
the same state (the COMSOL Multiphysics Reference Manual, 6.3 at doc.comsol.com). No COMSOL installation or licence
is needed or used: the reference is the vendor's documentation, and the record keeps that apart from Elmer Studio's
own design choices (the "design" field), e.g. Escape restoring a field, which the manual does not describe. Compare the two by eye, state by state; tests/test_desktop.py
and the packaged app's --selftest-ui check the behaviour automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
os.environ.setdefault("ELMERSTUDIO_RECOVERY_DIR", tempfile.mkdtemp(prefix="es_corpus_recovery_"))

DOC = "https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/"
STATES = [
    ("01_new_model_wizard", "New model and the wizard's space dimension page", "comsol_ref_modeling.19.003.html",
     "Pick a space dimension, then physics and a study"),
    ("02_parameters_units", "Parameters with a forward reference and units", "comsol_ref_definitions.21.002.html",
     "Use the parameters in geometry dimensions"),
    ("03_workplane_extrude", "Work plane, its sketch, extruded", "comsol_ref_geometry.23.010.html",
     "Build All (F8)"),
    ("04_boolean_stale", "Boolean geometry after a topology-changing parameter edit", "comsol_ref_geometry.23.016.html",
     "Build the geometry again: it is marked as needing an update"),
    ("05_material_missing", "A material missing a property the physics needs", "comsol_ref_materials.25.04.html",
     "Enter the missing property"),
    ("06_boundary_selection", "Boundary selection with a hidden face reached by Up/Down",
     "comsol_ref_visualizationselection.22.06.html", "Click to add the highlighted boundary"),
    ("07_mesh_quality", "Local mesh size and the element quality plot", "comsol_ref_mesh.24.03.html",
     "Refine where the quality is low"),
    ("08_solved_convergence", "Stationary solve and its convergence plot", "comsol_ref_solver.36.109.html",
     "Look at the default plots"),
    ("09_invalid_input", "An invalid expression kept as a draft, with the message linked to its node",
     "comsol_ref_modeling.19.020.html", "Fix the expression or press Esc"),
    ("10_results_navigation", "Result plot with its dataset and solution choice", "comsol_ref_results.30.001.html",
     "Choose another solution or parameter value"),
    ("11_derived_table", "A derived value evaluated into a table", "comsol_ref_results.30.090.html",
     "Copy or export the table"),
    ("12_recovery", "Recovery of an interrupted session", "comsol_ref_environment.18.22.html",
     "Open the recovered model and save it"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default=os.path.join(ROOT, ".tmp", "reference_corpus"))
    ap.add_argument("--sizes", default="1366x768,1920x1080")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
    app = QApplication(sys.argv[:1])
    from elmerstudio.core import builder as B
    from elmerstudio.core import project, recovery
    from elmerstudio.ui import theme
    from elmerstudio.ui.main_window import MainWindow
    from elmerstudio.ui.wizard import ModelWizard
    theme.apply(app)
    for name in ("warning", "critical", "information", "question"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))
    w = MainWindow(None, show_new=False)
    w.no_modal = True
    w.qs = QSettings(os.path.join(tempfile.mkdtemp(), "s.ini"), QSettings.IniFormat)
    w.show()

    def settle(cond=lambda: True, t=180):
        t0 = time.time()
        while time.time() - t0 < t and (w.busy() or not cond()):
            app.processEvents()
            time.sleep(0.02)
        for _ in range(25):
            app.processEvents()
            time.sleep(0.01)

    def model(sdim="3D", physics=("ht",)):
        settle()
        m = B.new_model(sdim, list(physics), "stationary")
        w._set_model(m, None, project.new_workdir())
        return m

    index = []

    def capture(key, extra_widget=None):
        title, page, nxt = next((t, p, n) for k, t, p, n in STATES if k == key)
        for (W, H) in sizes:
            w.resize(W, H)
            settle()
            png = f"{key}_{W}x{H}.png"
            (extra_widget or w).grab().save(os.path.join(args.outdir, png))
            cur = w.tree.current()
            fw = app.focusWidget() or w.focusWidget()
            rec = {"state": key, "title": title, "size": [W, H], "scale": os.environ.get("QT_SCALE_FACTOR", "1"),
                   "image": png, "selected_node": cur.label if cur else None,
                   "selected_kind": cur.kind if cur else None, "entity_level": w.graphics.level,
                   "layout": w.layout_mode, "ribbon_tab": w.ribbon.current_tab(),
                   "enabled_commands": sorted(n for n, a in w.act.items() if a.isEnabled()),
                   "disabled_commands": {n: a.toolTip().split("Unavailable: ")[-1] for n, a in w.act.items()
                                         if not a.isEnabled()},
                   "focus": (type(fw).__name__ + (f":{fw.accessibleName()}" if fw.accessibleName() else "")) if fw else None,
                   "revision": w.rev, "modified": w.dirty,
                   "status": [b.text() for b in (w.st_geo, w.st_mesh, w.st_sol) if b.isVisible()],
                   "expected_next_action": nxt, "reference": DOC + page,
                   "design": "Elmer Studio design choice where the reference is silent; see "
                             "docs/modules/ROOT/pages/comsol-users.adoc"}
            with open(os.path.join(args.outdir, f"{key}_{W}x{H}.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1)
            index.append(rec)
        print("captured", key, flush=True)

    # 1 wizard
    wz = ModelWizard(w)
    wz.show()
    settle()
    capture("01_new_model_wizard", wz)
    wz.close()
    # 2 parameters
    m = model()
    params = m.global_defs.child("params")
    params.props["table"] = [{"name": "W", "expr": "2*H", "descr": "width (forward reference)"},
                             {"name": "H", "expr": "10[mm]", "descr": "height"}]
    w.tree.refresh(select=params)
    capture("02_parameters_units")
    # 3 work plane -> extrude
    g = m.component.child("geometry")
    wp = m.create("geom.workplane", g, index=0)
    m.create("geom.rectangle", wp, w="W", h="H")
    ext = m.create("geom.extrude", g, index=1)
    ext.props["input"] = [wp.tag]
    ext.props["dist"] = "H"
    w.tree.refresh(select=ext)
    w.build_geometry()
    settle(lambda: w.geo is not None)
    capture("03_workplane_extrude")
    # 4 boolean + parameter edit
    cyl = m.create("geom.cylinder", g, index=2, r="H/4", h="H")
    dif = m.create("geom.difference", g, index=3)
    dif.props["input"], dif.props["tools"] = [ext.tag], [cyl.tag]
    w.build_geometry()
    settle(lambda: not w.geo_stale)
    w.tree.refresh(select=params)
    w.set_prop(params, "table", [{"name": "W", "expr": "4*H", "descr": "width"},
                                 {"name": "H", "expr": "10[mm]", "descr": "height"}], rebuild=True)
    w.tree.select(g)
    capture("04_boolean_stale")
    w.build_geometry()
    settle(lambda: not w.geo_stale)
    # 5 material missing a property
    mat = w.create_node("material", m.component.child("materials"))
    mat.selection.all = True
    w.tree.refresh(select=mat)
    capture("05_material_missing")
    w.delete_node(mat)
    w.add_library_material("Copper")                  # the first material: all domains
    settle()
    # 6 boundary selection, hidden face
    feat = B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", [], T0="350[K]")
    w.tree.refresh(select=feat)
    settle()
    try:
        import numpy as np
        b = w.geo.bbox
        r = w.graphics.plotter.renderer
        r.SetWorldPoint((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2, 1.0)
        r.WorldToDisplay()
        w.graphics._mouse = r.GetDisplayPoint()[:2]
        w.graphics.cycle_candidate(1)
        w.graphics.cycle_candidate(1)
        del np
    except Exception as exc:  # noqa: BLE001 - no graphics
        print("selection:", exc)
    capture("06_boundary_selection")
    # two temperatures on two faces, so the solution has a gradient to show
    w.set_selection(feat, entities=[1])
    B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", [w.geo.count("boundary")], T0="300[K]")
    w.tree.refresh()
    # 7 mesh quality
    w.set_prop(m.component.child("mesh"), "size", "fine", rebuild=True)
    w.tree.select(m.component.child("mesh"))
    w.show_mesh(quality=True)
    settle(lambda: w.mesh is not None and not w.mesh_stale)
    capture("07_mesh_quality")
    # 8 solve
    st = m.studies()[0]
    if w.inst is not None:
        w.compute(st)
        settle(lambda: not w.busy(), 600)
        if w.central.indexOf(w.conv) >= 0:
            w.central.setCurrentWidget(w.conv)
    capture("08_solved_convergence")
    w.central.setCurrentIndex(0)
    # 9 invalid input
    w.tree.select(feat)
    settle()
    e = w.settings.editor("T0")
    if e is not None:
        e.setFocus()
        e.setText("350[K")
        w.settings._commit_expr(e, next(p for p in __import__("elmerstudio.core.model", fromlist=["REGISTRY"])
                                         .REGISTRY[feat.kind].props if p.key == "T0"), lambda v: None)
    capture("09_invalid_input")
    if e is not None:
        e.setText(e.committed)
    # 10 results navigation
    pgs = w.plot_groups()
    if pgs:
        w.tree.select(pgs[0])
    capture("10_results_navigation")
    # 11 derived value
    ev = w.create_node("eval.global", m.results.child("results.derived"))
    ev.props["expr"] = "T"
    if w.solution_of(w._dataset_for(ev)) is not None:
        w.evaluate_node(ev)
    capture("11_derived_table")
    # 12 recovery: a checkpoint of a session that is gone, offered in the Recovery Files window
    s = recovery.Session(os.environ["ELMERSTUDIO_RECOVERY_DIR"])
    s.checkpoint(m.to_dict(), os.path.join(tempfile.gettempdir(), "busbar.esm"), None, "busbar.esm")
    s.note_job("Computing Study 1")
    info = json.load(open(os.path.join(s.dir, "info.json")))
    info["pid"], info["process_started"] = 999999, 1.0
    json.dump(info, open(os.path.join(s.dir, "info.json"), "w"))
    shown = {}
    orig = QDialog.exec

    def grab_exec(dlg, *a):
        dlg.resize(720, 320)
        dlg.show()
        settle()
        shown["dlg"] = dlg
        capture("12_recovery", dlg)
        return 0
    QDialog.exec = grab_exec
    w.show_recovery_files()
    QDialog.exec = orig
    recovery.discard(info | {"dir": s.dir})
    with open(os.path.join(args.outdir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    print(f"{len(index)} records in {args.outdir}")
    w.dirty = False
    w.close()


if __name__ == "__main__":
    main()
