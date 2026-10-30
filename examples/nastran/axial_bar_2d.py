#!/usr/bin/env python
"""2D axial bar MWE: NASTRAN-95 backend through Elmer Studio API.

Builds a 2D axial bar, solves with NASTRAN, prints measured vs theory,
saves .esm project, and writes offscreen PNG of displacement field.
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
    # 2D axial bar: L=1.0m, H=0.05m, thickness=0.01m
    L, H = 1.0, 0.05
    E = 2.0e11
    P = 10000.0
    A = H * 0.01  # cross-section
    delta_theory = (P * L) / (E * A)
    sigma_theory = P / A

    m = B.new_model("2D", ["solid"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(H))
    B.add_material(m, "Structural steel", all_domains=True)

    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.05", hmin="0.02")

    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")

    bnd_left = geo.select("boundary", xmax=0)
    bnd_right = geo.select("boundary", xmin=0.99 * L)
    assert bnd_left and bnd_right

    B.add_feature(m, ph, "solid.fixed", bnd_left)
    bload = B.add_feature(m, ph, "solid.bload", bnd_right)
    bload.props["Fx"] = f"{P}[N]"
    bload.props["Fy"] = "0[N]"

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
        tip_mask = np.isclose(coords[:, 0], L, atol=1e-4)
        ux = sol.names(0)["u"][tip_mask]
        mean_ux = np.mean(ux)
        err_disp = abs(mean_ux - delta_theory) / delta_theory * 100

        print(f"2D bar elongation: measured {mean_ux:.4e} m, theory {delta_theory:.4e} m, error {err_disp:.2f}%")

        if "vonmises" in sol.names(0):
            vm = sol.names(0)["vonmises"]
            interior = (coords[:, 0] > 0.2 * L) & (coords[:, 0] < 0.8 * L)
            if np.any(interior):
                mean_vm = np.mean(vm[interior])
                err_stress = abs(mean_vm - sigma_theory) / sigma_theory * 100
                print(f"2D bar stress: measured {mean_vm:.3e} Pa, theory {sigma_theory:.3e} Pa, error {err_stress:.2f}%")

        # Save .esm
        out_esm = os.path.join(os.path.dirname(__file__), "axial_bar_2d.esm")
        project.save(m, out_esm, wd)
        print(f"Saved project: {out_esm}")

        # Write offscreen PNG of displacement magnitude
        import pyvista as pv
        from elmerstudio.core.results import Topology
        plotter = pv.Plotter(off_screen=True)
        g = Topology(mesh_result=sol.topo).bulk_grid()
        g.point_data["disp_mag"] = sol.names(0)["disp_mag"]
        plotter.add_mesh(g, scalars="disp_mag", cmap="viridis")
        png_path = os.path.join(os.path.dirname(__file__), "axial_bar_2d.png")
        plotter.screenshot(png_path)
        plotter.close()
        print(f"Saved PNG: {png_path}")

    finally:
        project.discard_workdir(wd)
        import shutil
        shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    main()