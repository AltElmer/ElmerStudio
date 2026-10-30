#!/usr/bin/env python
"""2D cantilever eigenfrequency MWE: NASTRAN-95 backend through Elmer Studio API.

Builds a 2D cantilever beam, solves eigenfrequencies with NASTRAN (SOL 3),
prints measured vs theory, saves .esm project, and writes offscreen PNG.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import math

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
    # 2D cantilever: L=1.0m, H=0.05m
    L, H = 1.0, 0.05
    E, rho = 2.0e11, 7850.0
    f1_theory = (1.875104**2 / (2 * math.pi)) * math.sqrt(E * H**2 / (12 * rho)) / L**2

    m = B.new_model("2D", ["solid"], "eigen")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(H))
    B.add_material(m, "Structural steel", all_domains=True)

    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.0125", hmin="0.005")

    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")
    ph.props["plane"] = "stress"        # beam theory uses E

    bnd_left = geo.select("boundary", xmax=0)
    assert bnd_left
    B.add_feature(m, ph, "solid.fixed", bnd_left)

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
        assert sol.meta["nastran"]["sol"] == 3

        freqs = sol.meta["freqs"]
        print(f"Computed natural frequencies (Hz): {freqs}")
        rel_err = abs(freqs[0] - f1_theory) / f1_theory * 100
        print(f"First frequency: {freqs[0]:.2f} Hz, theory: {f1_theory:.2f} Hz, error {rel_err:.2f}%")

        ratio_theory = (4.694091 / 1.875104) ** 2
        ratio_meas = freqs[1] / freqs[0]
        ratio_err = abs(ratio_meas - ratio_theory) / ratio_theory * 100
        print(f"f2/f1 ratio: {ratio_meas:.4f}, theory: {ratio_theory:.4f}, error {ratio_err:.2f}%")

        # Save .esm
        out_esm = os.path.join(os.path.dirname(__file__), "cantilever_modes_2d.esm")
        project.save(m, out_esm, wd)
        print(f"Saved project: {out_esm}")

        # Write offscreen PNG of the first mode shape (deformed, colored by the displacement magnitude)
        import pyvista as pv
        from elmerstudio.core.results import Topology
        plotter = pv.Plotter(off_screen=True)
        g = Topology(mesh_result=sol.topo).bulk_grid()
        mode1 = sol.names(0)
        disp = np.column_stack([mode1["u"], mode1["v"], np.zeros_like(mode1["u"])])
        g.point_data["mode1_mag"] = np.linalg.norm(disp, axis=1)
        g.points = g.points + 0.2 * disp / np.abs(disp).max()
        plotter.add_mesh(g, scalars="mode1_mag", cmap="viridis")
        plotter.view_xy()
        png_path = os.path.join(os.path.dirname(__file__), "cantilever_modes_2d.png")
        plotter.screenshot(png_path)
        plotter.close()
        print(f"Saved PNG: {png_path}")

    finally:
        project.discard_workdir(wd)
        import shutil
        shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    main()