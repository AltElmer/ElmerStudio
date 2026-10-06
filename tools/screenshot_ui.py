"""Grab the dialogs and ribbon tabs (New, Model Wizard pages, Add Material, each ribbon tab)."""
from __future__ import annotations

import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def main():
    out = os.path.join(ROOT, "docs", "screenshots")
    os.makedirs(out, exist_ok=True)
    app = QApplication(sys.argv[:1])
    from elmerstudio.core.examples import EXAMPLES
    from elmerstudio.ui import theme
    from elmerstudio.ui.main_window import MainWindow
    from elmerstudio.ui.wizard import ModelWizard, NewDialog
    theme.apply(app)
    saved = []

    def save(widget, name):
        app.processEvents()
        p = os.path.join(out, f"ui_{name}.png")
        widget.grab().save(p)
        saved.append(p)

    d = NewDialog([(k, t) for k, (t, _f) in EXAMPLES.items()])
    d.show()
    save(d, "new")
    d.close()
    wz = ModelWizard()
    wz.show()
    save(wz, "wizard_1_dim")
    wz._dim("3D")
    for i in range(wz.tree.topLevelItemCount()):
        top = wz.tree.topLevelItem(i)
        if top.text(0) == "Heat Transfer":
            wz.tree.setCurrentItem(top.child(0))
            wz._add()
    save(wz, "wizard_2_physics")
    wz._to_study()
    save(wz, "wizard_3_study")
    wz.close()
    w = MainWindow(None, show_new=False)
    w.resize(1600, 960)
    w.show()

    def tabs():
        w.open_example("heat_sink")
        QTimer.singleShot(2500, step2)

    def step2():
        ph = w.model.component.child("physics.ht")
        w.tree.select(ph)
        for name in ("Geometry", "Physics", "Mesh", "Study", "Results"):
            w.ribbon.show_tab(name)
            app.processEvents()
            save(w.ribbon, f"ribbon_{name.lower()}")
        w.ribbon.show_tab("Home")
        mat = w.model.component.child("materials").children[0]
        w.tree.select(mat)
        w.show_material_browser()
        QTimer.singleShot(800, step3)

    def step3():
        save(w, "materials_window")
        w.dirty = False
        print("\n".join(saved))
        QTimer.singleShot(200, app.quit)

    QTimer.singleShot(500, tabs)
    app.exec()


if __name__ == "__main__":
    main()
