#!/usr/bin/env python
"""2D steady heat conduction MWE: NASTRAN-95 backend through Elmer Studio API.

Builds a 2D plate with temperature boundary conditions, solves with NASTRAN (HEAT 1),
prints measured vs theory, saves .esm project, and writes offscreen PNG.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure we can import elmerstudio when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from elmerstudio.core import builder as B
from elmerstudio.core import project
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.meshing import build_mesh
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner
import numpy as np


def main():
    # 2D plate: L=1.0m, H=0.2m
    L, H_plate = 1.0, 0.2
    T_left, T_right = 350.0, 300.0

    m = B.new_model("2D", ["ht"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(H_plate))
    B.add_material(m, "Structural steel", all_domains=True)

    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.1", hmin="0.05")

    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")

    bnd_left = geo.select("boundary", xmax=0)
    bnd_right = geo.select("boundary", xmin=0.99 * L)
    assert bnd_left and bnd_right

    t1 = B.add_feature(m, ph, "ht.temperature", bnd_left)
    t1.props["T0"] = f"{T_left}[K]"

    t2 = B.add_feature(m, ph, "ht.temperature", bnd_right)
    t2.props["T0"] = f"{T_right}[K]"

    st = m.studies()[0]
    st.props["backend"] = "nastran"

    wd = project.new_workdir()
    try:
        folders = StudyRunner(m, st, wd, None, None).run()
        assert folders
        folder = folders[0]
        sol = Solution(folder)

        assert sol.meta["status"] == "converged"
        assert sol.meta["backend"] == "nastran"

        coords = sol.topo.nodes * sol.scale
        T_field = sol.names(0)["T"]
        T_exact = T_left + (T_right - T_left) * (coords[:, 0] / L)
        max_err = np.max(np.abs(T_field - T_exact))
        print(f"Heat conduction max temperature error: {max_err:.4e} K")

        # Save .esm
        out_esm = os.path.join(os.path.dirname(__file__), "heat_2d.esm")
        project.save(m, out_esm, wd)
        print(f"Saved project: {out_esm}")

        # Write offscreen PNG of temperature field
        import pyvista as pv
        from elmerstudio.core.results import Topology
        plotter = pv.Plotter(off_screen=True)
        g = Topology(mesh_result=sol.topo).bulk_grid()
        g.point_data["T"] = T_field
        plotter.add_mesh(g, scalars="T", cmap="hot")
        png_path = os.path.join(os.path.dirname(__file__), "heat_2d.png")
        plotter.screenshot(png_path)
        plotter.close()
        print(f"Saved PNG: {png_path}")

    finally:
        project.discard_workdir(wd)
        import shutil
        shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    main()