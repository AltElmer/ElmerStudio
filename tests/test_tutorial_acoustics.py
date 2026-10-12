"""Acoustics walkthroughs of the COMSOL notebook, reproduced with Elmer Studio (Elmer HelmholtzSolver).

* Essentials of Postprocessing, *vented loudspeaker enclosure*: isosurfaces of the acoustic pressure around a
  speaker, legend as filled level blocks.  Stand-in: a piston in a closed box radiating into a sphere of air
  closed by Spherical Wave Radiation (COMSOL uses a PML).
* "How to Use the Boundary Element Method in Acoustics Modeling", *Loudspeaker Radiation* postprocessing: a
  Grid 3D dataset evaluated beyond the mesh (multislice of the sound pressure level), the far-field Radiation
  Pattern (polar and 3D), the Directivity plot and a Parametric Curve 3D spiral.  Elmer has no boundary
  elements, so the exterior field comes from the Exterior Field Calculation (Kirchhoff-Helmholtz integral over
  the outer surface of the FEM air domain), which is what COMSOL's FEM models use.
* *Spherical Scatterer: BEM Benchmark*: a rigid sphere in a plane wave, compared with the analytic series.

Verification: the pulsating sphere (monopole) and the rigid-sphere scatterer have closed-form solutions.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core.geometry import build_geometry  # noqa: E402

from elmerstudio.core.examples import AIR_C as C0, AIR_RHO as RHO0, _air  # noqa: E402
from elmerstudio.core.examples import loudspeaker_setup as loudspeaker_model  # noqa: E402
from elmerstudio.core.examples import scatterer_setup as scatterer_model  # noqa: E402
from elmerstudio.core.examples import sphere_shell_setup as sphere_shell_model  # noqa: E402


def _solve(m, geo, prefix):
    from elmerstudio.core.elmer import find_elmer
    from elmerstudio.core.nodes import bind_results
    from elmerstudio.core.results import Solution
    from elmerstudio.core.study_runner import StudyRunner
    folders = StudyRunner(m, m.studies()[0], tempfile.mkdtemp(prefix=prefix), find_elmer()).run()
    return bind_results(m, Solution(folders[-1]), geo)

# --------------------------------------------------------------------------- verification
def test_pulsating_sphere_radiation_and_exterior_field(elmer):
    """Spherical Wave Radiation and the exterior field against p = rho c u (ika/(1+ika)) (a/r) exp(-ik(r-a))."""
    from elmerstudio.core.acoustics import ExteriorField
    a, R = 0.1, 0.6
    m, geo, inner, outer = sphere_shell_model(a, R, mesh="1")
    ac = m.component.child("physics.acpr")
    B.add_feature(m, ac, "acpr.nacc", inner, an="1")
    B.add_feature(m, ac, "acpr.swr", outer)
    sol = _solve(m, geo, "es_mono_")
    P = sol.topo.nodes * sol.scale
    r = np.linalg.norm(P[:, :2], axis=1)
    sel = (r > 0.2) & (r < 0.5)
    efc = ExteriorField(sol, outer, C0)
    for i, fr in enumerate(sol.frames):
        k, w = 2 * math.pi * fr.value / C0, 2 * math.pi * fr.value
        exact = lambda rr: RHO0 * C0 / (1j * w) * (1j * k * a / (1 + 1j * k * a)) * (a / rr) * np.exp(-1j * k * (rr - a))  # noqa: E731
        pw = sol.arrays(i)["pressure wave"]
        p = pw[:, 0] + 1j * pw[:, 1]
        err = np.abs(p[sel] - exact(r[sel])).max() / np.abs(exact(r[sel])).max()
        assert err < 2e-3, (fr.value, err)                                 # FEM + spherical wave radiation
        th = np.radians(np.linspace(-80, 80, 9))
        X = np.column_stack([3 * np.cos(th), np.zeros_like(th), 3 * np.sin(th)])     # 5 x the mesh radius
        e2 = np.abs(efc.pressure(i, X) - exact(3.0)).max() / abs(exact(3.0))
        assert e2 < 0.015, (fr.value, e2)                                  # Kirchhoff-Helmholtz integral


def test_pulsating_sphere_3d_exterior_field(elmer):
    """3D: the exterior field from the outer (radiation) surface of a coarse lambda/5 mesh, 4 x beyond it."""
    from elmerstudio.core.acoustics import point_values
    a, R = 0.1, 0.5
    m = B.new_model("3D", ["acpr"], "freq")
    g = m.component.child("geometry")
    o, s = m.create("geom.sphere", g, r=str(R)), m.create("geom.sphere", g, r=str(a))
    m.create("geom.difference", g, input=[o.tag], tools=[s.tag])
    _air(m)
    geo = build_geometry(m, display=False)
    bb = geo.bbox_of["boundary"]
    inner = [b for b in geo.numbers["boundary"] if bb[b][3] - bb[b][0] < 3 * a]
    outer = [b for b in geo.numbers["boundary"] if b not in inner]
    ac = m.component.child("physics.acpr")
    B.add_feature(m, ac, "acpr.nacc", inner, an="1")
    B.add_feature(m, ac, "acpr.swr", outer)
    B.add_feature(m, ac, "acpr.efc", outer)
    B.study_steps(m.studies()[0])[0].props["freqs"] = "500"
    sol = _solve(m, geo, "es_mono3_")
    k, w = 2 * math.pi * 500 / C0, 2 * math.pi * 500
    X = np.array([[2.0, 0, 0], [0, 0, 2.0], [0, 1.5, -1.0], [-1.2, -1.2, 1.2]])
    rr = np.linalg.norm(X, axis=1)
    exact = RHO0 * C0 / (1j * w) * (1j * k * a / (1 + 1j * k * a)) * (a / rr) * np.exp(-1j * k * (rr - a))
    p = point_values(sol, "acpr.efc1.pext", 0, X) + 1j * point_values(sol, "imag(acpr.efc1.pext)", 0, X)
    assert np.abs(p - exact).max() / np.abs(exact).max() < 0.02


def test_spherical_scatterer_benchmark(elmer):
    """Rigid sphere in a plane wave: scattered field in the mesh and beyond it vs the analytic series."""
    from elmerstudio.core.acoustics import point_values, rigid_sphere_scattered
    a, R = 0.25, 1.0
    m, geo, inner, outer = scatterer_model(a, R)
    sol = _solve(m, geo, "es_scat_")
    P = sol.topo.nodes * sol.scale
    r = np.linalg.norm(P[:, :2], axis=1)
    th = np.arctan2(P[:, 0], P[:, 1])
    sel = (r > 1.2 * a) & (r < 0.8 * R)
    for i, fr in enumerate(sol.frames):
        k = 2 * math.pi * fr.value / C0
        ps = sol.evaluate("acpr.p_s", i) + 1j * sol.evaluate("acpr.p_sim", i)      # result variables
        exact = rigid_sphere_scattered(k, a, r[sel], th[sel])
        err = np.abs(ps[sel] - exact).max() / np.abs(exact).max()
        assert err < 0.05, (fr.value, err)
        # exterior field (from the sphere surface) at 3 m, beyond the mesh
        tt = np.linspace(0, math.pi, 13)
        Lp = point_values(sol, "acpr.efc1.Lp_pext", i, np.column_stack([3 * np.sin(tt), 3 * np.cos(tt)]))
        Lp_exact = 20 * np.log10(np.abs(rigid_sphere_scattered(k, a, np.full_like(tt, 3.0), tt)) / math.sqrt(2) / 20e-6)
        assert np.abs(Lp - Lp_exact).max() < 0.8, (fr.value, Lp - Lp_exact)     # dB


# --------------------------------------------------------------------------- loudspeaker postprocessing
def test_loudspeaker_postprocessing(elmer):
    from matplotlib.figure import Figure

    from elmerstudio.core.datasets import resolve
    from elmerstudio.ui.plots import build_scene, directivity_map, draw_1d, radiation_polar
    m, geo, piston, outer = loudspeaker_model(freqs="300 800")
    sol = _solve(m, geo, "es_spk_")
    assert sorted(sol.efc) == ["efc1"]
    fi = len(sol.frames) - 1
    dsets = m.results.child("results.datasets")
    sds = B.solution_dataset(m, m.studies()[0], create=True)
    load = lambda d: sol  # noqa: E731
    # Grid 3D reaching beyond the air domain: mesh values inside, exterior field outside, continuous across
    grid = m.create("dset.grid3d", dsets, data=sds.tag, xmin="-1.5", xmax="1.5", ymin="-1.5", ymax="1.5",
                    zmin="-1.5", zmax="1.5", xres=13, yres=13, zres=13)
    gv, _ = resolve(m, grid, load)
    Lp = gv.evaluate("acpr.efc1.Lp_pext", fi)
    assert np.isfinite(Lp).all()
    X = gv.topo.nodes
    rr = np.linalg.norm(X, axis=1)
    # no jump at the edge of the mesh: just inside (mesh) vs just outside (Kirchhoff-Helmholtz), 8 directions
    from elmerstudio.core.acoustics import point_values
    ang = np.radians(np.arange(0, 360, 45) + 10)
    dirs = np.column_stack([np.cos(ang), np.zeros_like(ang), np.sin(ang)])
    L_in = point_values(sol, "acpr.efc1.Lp_pext", fi, 0.47 * dirs)
    L_out = point_values(sol, "acpr.efc1.Lp_pext", fi, 0.53 * dirs)
    assert np.abs(L_in - L_out).max() < 2.0, (L_in, L_out)                     # dB (spreading alone: 1.0 dB)
    far = rr > 1.2
    # spherical spreading: about -6 dB per doubling of distance in the far field
    d1 = np.median(Lp[(rr > 0.9) & (rr < 1.1)]) - np.median(Lp[(rr > 1.8) & (rr < 2.2)])
    assert 3.0 < d1 < 9.0, d1
    assert far.any()
    # Radiation Pattern (polar): louder in front (+z) than behind at 800 Hz
    pol = m.create("plot.radiation", m.create("pgpolar", m.results), radius="1[m]", plane="xz", a0="0", a1="360",
                   npts=73)
    th, v, label, unit = radiation_polar(pol, sol, fi, m)
    assert unit == "dB" and label == "Exterior-field sound pressure level"
    front, back = v[np.argmin(np.abs(th - math.pi / 2))], v[np.argmin(np.abs(th - 3 * math.pi / 2))]
    assert front > back + 1.0, (front, back)
    fig = Figure()
    draw_1d(fig, pol.parent, m, lambda d, s: (sol, fi, sol.scale), "last")
    assert fig.axes[0].name == "polar" and len(fig.axes[0].lines) == 1
    # 3D radiation pattern in a 3D plot group
    pg3 = m.create("pg3d", m.results)
    m.create("plot.radiation", pg3, radius="1[m]", npts=72)
    sc = build_scene(pg3, sol, fi, 3, model=m)
    assert sc.items and "Radiation Pattern" in sc.title
    # Directivity: on-axis normalization gives 0 dB at 0 degrees for every frequency
    dirp = m.create("plot.directivity", m.create("pg1d", m.results), data=sds.tag, radius="1[m]", npts=37)
    fr, ang, M, label, unit = directivity_map(dirp, sol, m)
    j0 = int(np.argmin(np.abs(ang)))
    assert len(fr) == 2 and np.allclose(M[j0], 0, atol=1e-9)
    assert (M[np.abs(ang) > 60] < 0.5).all()                                     # off-axis not louder
    fig = Figure()
    draw_1d(fig, dirp.parent, m, lambda d, s: (sol, fi, sol.scale), "last")
    assert fig.axes[0].get_title().startswith("Directivity")
    # Parametric Curve 3D: a spiral around the speaker, sound pressure level along it
    spiral = m.create("dset.paramcurve", dsets, data=sds.tag, smin="0", smax="1", npts=120,
                      xexpr="(0.7+0.5*s)*cos(6*pi*s)", yexpr="(0.7+0.5*s)*sin(6*pi*s)", zexpr="-0.6+1.2*s")
    cv, _ = resolve(m, spiral, load)
    Ls = cv.evaluate("acpr.efc1.Lp_pext", fi)
    assert np.isfinite(Ls).all() and Ls.std() > 0.1
    pgl = m.create("pg1d", m.results)
    m.create("plot.linegraph", pgl, data=spiral.tag, expr="acpr.efc1.Lp_pext")
    fig = Figure()
    draw_1d(fig, pgl, m, lambda d, s: (cv, fi, cv.scale), "last")
    assert len(fig.axes[0].lines) == 1
    # isosurfaces of the acoustic pressure with a filled (blocked) legend
    pgi = m.create("pg3d", m.results)
    m.create("plot.isosurface", pgi, expr="acpr.p_t", levels=8)
    sc = build_scene(pgi, sol, fi, 3, model=m)
    assert sc.legends and sc.legends[0].get("bins") == 9


if __name__ == "__main__":
    m, geo, piston, outer = loudspeaker_model()
    print("piston", piston, "outer", outer)
