"""Primary and Secondary Current Distribution interface (cd) against closed forms.

* 1D cell (strip): anode with each kinetics type, counter electrode at a fixed electrolyte potential.
  The current solves i = i_loc(V - i L / sigma) exactly (one unknown, solved here by bisection); every
  number below is the conservative electrode current from Elmer's reaction loads (cd.es1.Itot).
* Newman's disc electrode (2D axisymmetric): the primary resistance of a disc of radius a flush in an
  insulating plane is 1/(4 sigma a); with a hemispherical counter electrode at R the far field adds
  -1/(2 pi sigma R).  Checks the 2 pi of axisymmetric currents and the reaction-flux evaluation.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core.geometry import build_geometry  # noqa: E402

F, R, T = 96485.33212, 8.314462618, 298.15
f = F / (R * T)


def _solve(m, geo, prefix):
    from elmerstudio.core.elmer import find_elmer
    from elmerstudio.core.nodes import bind_results
    from elmerstudio.core.results import Solution
    from elmerstudio.core.study_runner import StudyRunner
    folders = StudyRunner(m, m.studies()[0], tempfile.mkdtemp(prefix=prefix), find_elmer()).run()
    return [bind_results(m, Solution(fo), geo) for fo in folders]


def strip(kin, V=0.1, i0=50.0, sigma=20.0, W=1e-4, L=2e-4, **kp):
    m = B.new_model("2D", ["cd"], "stationary")
    g = m.component.child("geometry")
    m.create("geom.rectangle", g, index=0, w=str(W), h=str(L))
    m.global_defs.child("params").props["table"] = [{"name": "Vapp", "expr": f"{V}[V]", "descr": "Cell voltage"}]
    geo = build_geometry(m, display=False)
    cd = m.component.child("physics.cd")
    cd.child("cd.elyte").props.update(sigma_src="user", sigma=str(sigma))
    an = B.add_feature(m, cd, "cd.es", geo.select("boundary", ymax=1e-9), phis="Vapp", kin=kin, i0=str(i0), **kp)
    B.add_feature(m, cd, "cd.pot", geo.select("boundary", ymin=L - 1e-9), phil0="0")
    return m, geo, an


def exact_current(iloc, V, L, sigma):
    """Root of i = iloc(V - i L / sigma) (i between 0 and the primary current)."""
    lo, hi = 0.0, V * sigma / L
    for _ in range(200):
        x = 0.5 * (lo + hi)
        lo, hi = (x, hi) if iloc(V - x * L / sigma) - x > 0 else (lo, x)
    return 0.5 * (lo + hi)


KINETICS = {
    "primary": lambda eta: 1e30 * eta,
    "linear": lambda eta: 50.0 * f * eta,
    "bv": lambda eta: 50.0 * (math.exp(0.5 * f * eta) - math.exp(-0.5 * f * eta)),
    "tafel_a": lambda eta: 50.0 * 10 ** (eta / 0.118),
    "lim": lambda eta: (lambda ia, ic: ia / (1 + ia / 100) - ic / (1 + ic / 100))(
        50.0 * math.exp(0.5 * f * eta), 50.0 * math.exp(-0.5 * f * eta)),
}


@pytest.mark.parametrize("kin", list(KINETICS))
def test_strip_kinetics(kin, elmer):
    V, L, W, sigma = 0.1, 2e-4, 1e-4, 20.0
    m, geo, an = strip(kin, V=V, L=L, W=W, sigma=sigma, ilim="100[A/m^2]")
    sol = _solve(m, geo, f"es_cd_{kin}_")[-1]
    i = sol.frame_index("last")
    I = float(sol.evaluate(f"cd.{an.tag}.Itot", i)[0])
    iex = exact_current(KINETICS[kin], V, L, sigma) if kin != "primary" else V * sigma / L
    assert abs(I / (iex * W) - 1) < 1e-9, (kin, I, iex * W)
    # local current density from the reaction, at every anode node (corners included)
    P = sol.topo.nodes
    bot = np.abs(P[:, 1]) < 1e-12
    assert np.allclose(sol.evaluate("cd.iloc", i)[bot], iex, rtol=1e-8)
    if kin not in ("primary",):
        assert np.allclose(sol.evaluate(f"cd.{an.tag}.iloc", i)[bot], iex, rtol=1e-8)   # kinetics expression
    # the counter electrode carries the same current out of the electrolyte
    top = np.abs(P[:, 1] - L) < 1e-12
    assert np.allclose(sol.evaluate("cd.nIl", i)[top], iex, rtol=1e-8)


def test_strip_tabulated_kinetics(elmer):
    """User-defined i(eta) through an Interpolation function of tabulated polarization data (piecewise linear)."""
    V, L, W, sigma = 0.3, 2e-4, 1e-4, 20.0
    pts = [(0.0, 0.0), (0.05, 40.0), (0.1, 150.0), (0.2, 600.0), (0.3, 2000.0)]
    m, geo, an = strip("expr", V=V, L=L, W=W, sigma=sigma, iexpr="pol(eta)")
    m.create("func.interp", m.global_defs, fname="pol",
             table=[{"x": str(x), "y": str(y)} for x, y in pts])
    sol = _solve(m, geo, "es_cd_tab_")[-1]
    I = float(sol.evaluate(f"cd.{an.tag}.Itot", sol.frame_index("last"))[0])
    iex = exact_current(lambda e: float(np.interp(e, *zip(*pts))), V, L, sigma)
    assert abs(I / (iex * W) - 1) < 1e-6, (I, iex * W)


def test_strip_clamped_kinetics(elmer):
    """User-defined i(eta) with nested two-argument min/max clamps (a saturating Tafel law with a floor and a
    ceiling, as fitted polarization data are often written).  MATC's own min/max are one-argument reductions,
    so the export calls helper functions; nested clamps must not blow the expression up."""
    V, L, W, sigma = 0.3, 2e-4, 1e-4, 20.0
    law = "min(max(j0*exp(min(eta/b,30))/(1+j0*exp(min(eta/b,30))/jl),0),20*jl)"
    m, geo, an = strip("expr", V=V, L=L, W=W, sigma=sigma, iexpr=law)
    m.global_defs.child("params").props["table"] += [
        {"name": "j0", "expr": "5[A/m^2]", "descr": ""}, {"name": "b", "expr": "0.04[V]", "descr": ""},
        {"name": "jl", "expr": "2000[A/m^2]", "descr": ""}]
    sol = _solve(m, geo, "es_cd_clamp_")[-1]
    text = open(os.path.join(sol.folder, "case.sif")).read()
    assert "$ function stmin" in text and max(len(s) for s in text.splitlines()) < 4000
    I = float(sol.evaluate(f"cd.{an.tag}.Itot", sol.frame_index("last"))[0])

    def law_py(eta):
        jt = 5 * math.exp(min(eta / 0.04, 30))
        return min(max(jt / (1 + jt / 2000), 0), 40000)
    iex = exact_current(law_py, V, L, sigma)
    assert abs(I / (iex * W) - 1) < 1e-6, (I, iex * W)


def test_polarization_sweep(elmer):
    """Parametric sweep of the electrode potential: a polarization (V-I) curve, monotonic and exact."""
    V, L, W, sigma = 0.1, 2e-4, 1e-4, 20.0
    m, geo, an = strip("bv", V=V, L=L, W=W, sigma=sigma)
    m.create("study.sweep", m.studies()[0], index=0, pname="Vapp", pvalues="range(0.05,0.05,0.25)", punit="V")
    sols = _solve(m, geo, "es_cd_sweep_")
    rows = sorted((s.meta["sweep_value"], float(s.evaluate(f"cd.{an.tag}.Itot", s.frame_index("last"))[0]))
                  for s in sols)
    for Vs, I in rows:
        assert abs(I / (exact_current(KINETICS["bv"], Vs, L, sigma) * W) - 1) < 1e-8, (Vs, I)
    assert all(b[1] > a[1] for a, b in zip(rows, rows[1:]))


def test_newman_disc_axisymmetric(elmer):
    a, Rf, sigma, V = 1e-3, 30e-3, 5.0, 1.0
    m = B.new_model("2Daxi", ["cd"], "stationary")
    g = m.component.child("geometry")
    q = m.create("geom.circle", g, index=0, r=str(Rf), sector="90")
    d = m.create("geom.rectangle", g, index=1, w=str(a), h=str(a / 4))
    m.create("geom.union", g, index=2, input=[q.tag, d.tag])
    geo = build_geometry(m, display=False)
    cd = m.component.child("physics.cd")
    cd.props["dist"] = "primary"
    cd.child("cd.elyte").props.update(sigma_src="user", sigma=str(sigma))
    bb = geo.bbox_of["boundary"]
    disc = [b for b in geo.numbers["boundary"] if bb[b][4] < 1e-9 and bb[b][3] <= a + 1e-9]
    far = [b for b in geo.numbers["boundary"] if bb[b][3] > 0.5 * Rf and bb[b][4] > 0.5 * Rf]
    an = B.add_feature(m, cd, "cd.es", disc, phis=str(V))
    B.add_feature(m, cd, "cd.pot", far, phil0="0")
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.size", mesh, mode="custom", hmax=str(Rf / 12), hmin=str(a / 400), hcurve="0.3", hnarrow="1")
    edge = [p for p in geo.numbers["point"] if abs(geo.bbox_of["point"][p][0] - a) < 1e-6 and geo.bbox_of["point"][p][1] < 1e-6]
    s = m.create("mesh.size", mesh, scope="point", mode="custom", hmax=str(a / 200), hmin=str(a / 400))
    s.selection.level, s.selection.all, s.selection.entities = "point", False, edge
    s2 = m.create("mesh.size", mesh, scope="boundary", mode="custom", hmax=str(a / 25), hmin=str(a / 400))
    s2.selection.level, s2.selection.all, s2.selection.entities = "boundary", False, disc
    m.create("mesh.ftri", mesh)
    sol = _solve(m, geo, "es_cd_disc_")[-1]
    I = float(sol.evaluate(f"cd.{an.tag}.Itot", sol.frame_index("last"))[0])
    R_exact = 1 / (4 * sigma * a) - 1 / (2 * math.pi * sigma * Rf)
    assert abs((V / I) / R_exact - 1) < 0.005, (V / I, R_exact)      # measured 0.03 %


def test_wavy_anode_mapped_mesh(elmer):
    """Primary current over a sinusoidal anode y = a cos(kx) (Parametric Curve + open Polygon -> Convert to Solid,
    Mapped mesh graded toward the anode): the relative current modulation is a k coth(kL) (Wagner)."""
    from elmerstudio.core.meshing import build_mesh
    m = B.new_model("2D", ["cd"], "stationary")
    g = m.component.child("geometry")
    g.props["unit"] = "um"
    m.global_defs.child("params").props["table"] = [{"name": "lam", "expr": "100", "descr": "Wavelength (um)"},
                                                     {"name": "a0", "expr": "1", "descr": "Amplitude (um)"},
                                                     {"name": "L", "expr": "200", "descr": "Gap (um)"}]
    pc = m.create("geom.paramcurve", g, index=0, xexpr="lam*s", yexpr="a0*cos(2*pi*s)", npts=81)
    po = m.create("geom.polygon", g, index=1, ptype="open", xs="lam lam 0 0", ys="a0 L L a0")
    cs = m.create("geom.tosolid", g, index=2)
    cs.props["input"] = [pc.tag, po.tag]
    geo = build_geometry(m, display=False)
    assert (geo.count("domain"), geo.count("boundary"), geo.count("point")) == (1, 4, 4)
    bb = geo.bbox_of["boundary"]
    anode = [b for b in geo.numbers["boundary"] if bb[b][4] < 2]
    top = [b for b in geo.numbers["boundary"] if bb[b][1] > 150]
    left = [b for b in geo.numbers["boundary"] if b not in anode + top and bb[b][3] < 1]
    right = [b for b in geo.numbers["boundary"] if b not in anode + top + left]
    cd = m.component.child("physics.cd")
    cd.props["dist"] = "primary"
    cd.child("cd.elyte").props.update(sigma_src="user", sigma="20")
    an = B.add_feature(m, cd, "cd.es", anode, phis="0")
    B.add_feature(m, cd, "cd.pot", top, phil0="-0.02")
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    m.create("mesh.distribution", mesh, n=80).selection.entities = anode + top
    m.create("mesh.distribution", mesh, n=60, ratio="20").selection.entities = right
    m.create("mesh.distribution", mesh, n=60, ratio="20", reverse=True).selection.entities = left
    assert build_mesh(m).stats["types"] == {"Quadrilateral 9": 4800}
    sol = _solve(m, geo, "es_cd_wavy_")[-1]
    i = sol.frame_index("last")
    il = sol.evaluate("cd.iloc", i)
    P = sol.topo.nodes * sol.scale
    on = np.isfinite(il) & (P[:, 1] < 2e-6)
    k = 2 * math.pi / 100e-6
    c0, c1 = np.linalg.lstsq(np.column_stack([np.ones(on.sum()), np.cos(k * P[on, 0])]), il[on], rcond=None)[0]
    assert abs((c1 / c0) / (1e-6 * k / math.tanh(k * 200e-6)) - 1) < 2e-3          # measured 0.02 %
    assert abs(sol.evaluate(f"cd.{an.tag}.Itot", i)[0] / (20 * 0.02 / 200e-6 * 100e-6) - 1) < 1e-3


def test_sweep_all_combinations(elmer):
    """Parametric Sweep over two parameters, All combinations: every run sees both values (2 x 3 = 6 runs)."""
    V, L, W = 0.1, 2e-4, 1e-4
    m, geo, an = strip("linear", V=V, L=L, W=W, sigma=20.0)
    m.global_defs.child("params").props["table"].append({"name": "sig", "expr": "20[S/m]", "descr": ""})
    m.component.child("physics.cd").child("cd.elyte").props["sigma"] = "sig"
    m.create("study.sweep", m.studies()[0], index=0, stype="all", pname="Vapp", pvalues="0.05 0.1 0.2", punit="V",
             more=[{"name": "sig", "values": "10 20", "unit": "S/m"}])
    sols = _solve(m, geo, "es_cd_combo_")
    seen = set()
    for s in sols:
        p = s.meta["sweep_params"]
        I = float(s.evaluate(f"cd.{an.tag}.Itot", s.frame_index("last"))[0])
        iex = exact_current(KINETICS["linear"], p["Vapp"], L, p["sig"])
        assert abs(I / (iex * W) - 1) < 1e-9, (p, I)
        assert s.params["sig"] == p["sig"]                     # result expressions see every swept parameter
        seen.add((round(p["Vapp"], 6), p["sig"]))
    assert len(seen) == 6


def test_tertiary_three_resistances(elmer):
    """Secondary Current Distribution + Transport of Diluted Species coupled at the anode (tertiary): linear
    kinetics (polarization length Lam), an equilibrium potential shifted by the local product concentration
    (linearized Nernst) and a Nernst diffusion layer delta under a bulk electrolyte.  Ohmic, charge-transfer and
    concentration resistances in series: i = sigma V / (L + Lam + beta delta), beta = sigma R T/(n^2 F^2 D c_b).
    The species flux comes from the anode's own kinetics through an Electrode Surface Coupling (nu i/(n F))."""
    sigma, V, L, dl, lam, D, n, beta, T_ = 20.0, 0.02, 200e-6, 2e-6, 15e-6, 1e-9, 2.0, 2.0, 298.15
    cb = sigma * R * T_ / (n ** 2 * F ** 2 * D * beta)
    m = B.new_model("2D", ["cd", "tds"], "stationary")
    m.global_defs.child("params").props["table"] = [
        {"name": k, "expr": v, "descr": ""} for k, v in (
            ("sig", f"{sigma}[S/m]"), ("Vc", f"{V}[V]"), ("Lam", f"{lam}[m]"), ("Dc", f"{D}[m^2/s]"),
            ("nel", f"{n}"), ("cb", f"{cb}[mol/m^3]"), ("Tk", f"{T_}[K]"))]
    g = m.component.child("geometry")
    m.create("geom.rectangle", g, index=0, w="20e-6", h=str(dl))                  # Nernst layer
    m.create("geom.rectangle", g, index=1, w="20e-6", h=str(L - dl), y=str(dl))  # bulk
    geo = build_geometry(m, display=False)
    layer = geo.select("domain", ymax=dl * (1 + 1e-9))
    anode, top = geo.select("boundary", ymax=1e-12), geo.select("boundary", ymin=L * (1 - 1e-9))
    edge = geo.select("boundary", ymin=dl * (1 - 1e-9), ymax=dl * (1 + 1e-9))
    cd = m.component.child("physics.cd")
    cd.child("cd.elyte").props.update(sigma_src="user", sigma="sig")
    an = B.add_feature(m, cd, "cd.es", anode, phis="Vc", kin="expr", iexpr="sig/Lam*eta",
                       Eeq="R_const*Tk/(nel*F_const)*(c-cb)/cb")          # product accumulation opposes dissolution
    B.add_feature(m, cd, "cd.pot", top, phil0="0")
    tds = m.component.child("physics.tds")
    tds.selection.all, tds.selection.entities = False, layer
    tds.child("tds.tp").props["D"] = "Dc"
    tds.child("tds.init").props["c0"] = "cb"
    B.add_feature(m, tds, "tds.esc", anode, es=an.tag, nu="1", nel="nel")
    B.add_feature(m, tds, "tds.conc", edge, c0="cb")
    m.studies()[0].child("study.stationary").props.update(coupling=100, coupling_tol="1e-11")
    sol = _solve(m, geo, "es_cd_tertiary_")[-1]
    i = sol.frame_index("last")
    I = float(sol.evaluate(f"cd.{an.tag}.Itot", i)[0]) / 20e-6
    exact = sigma * V / (L + lam + beta * dl)
    assert abs(I / exact - 1) < 1e-8, (I, exact)
    cs = float(np.nanmax(sol.evaluate("c", i)))                               # surface enrichment c_b + i delta/(nFD)
    assert abs((cs - cb) / (exact * dl / (n * F * D)) - 1) < 1e-6, (cs, cb)
    ctrl = sigma * V / (L + lam)                                              # without the concentration resistance
    assert abs(I / ctrl - 1) > 0.01


def test_box_selection_follows_a_swept_geometry(elmer):
    """Sweeping the wave amplitude to 0 renumbers the boundaries (the flat anode ties the left wall on position).
    Electrode features on Box Selections follow; each run reports its own electrode boundaries."""
    from elmerstudio.core.geometry import build_geometry as bg
    V, L, lam, sigma = 0.1, 200e-6, 100e-6, 20.0
    m = B.new_model("2D", ["cd"], "stationary")
    m.global_defs.child("params").props["table"] = [
        {"name": "lam", "expr": "100[um]", "descr": ""}, {"name": "a0", "expr": "1[um]", "descr": ""},
        {"name": "L", "expr": "200[um]", "descr": ""}, {"name": "Vapp", "expr": f"{V}[V]", "descr": ""}]
    g = m.component.child("geometry")
    g.props["unit"] = "um"
    pc = m.create("geom.paramcurve", g, index=0, xexpr="lam*s", yexpr="a0*sin(2*pi*s)", npts=41)
    po = m.create("geom.polygon", g, index=1, ptype="open", xs="lam lam 0 0", ys="0 L L 0")
    cs = m.create("geom.tosolid", g, index=2)
    cs.props["input"] = [pc.tag, po.tag]
    bs_an = m.create("geom.boxsel", g, index=3, level="boundary", ymax="L/2")
    bs_ca = m.create("geom.boxsel", g, index=4, level="boundary", ymin="L/2")
    wavy, flat = bg(m, display=False), bg(m, overrides={"a0": 0.0}, display=False)
    assert wavy.named[bs_an.tag][1] != flat.named[bs_an.tag][1]          # the sweep does renumber
    cd = m.component.child("physics.cd")
    cd.child("cd.elyte").props.update(sigma_src="user", sigma=str(sigma))
    an = B.add_feature(m, cd, "cd.es", [], phis="Vapp", kin="linear", i0="50")
    an.selection.named = bs_an.tag
    m.create("cd.pot", cd, phil0="0").selection.named = bs_ca.tag
    geo = bg(m, display=False)
    assert an.selection.entities == wavy.named[bs_an.tag][1]             # building points the feature at it
    m.create("study.sweep", m.studies()[0], index=0, pname="a0", pvalues="1 0", punit="um")
    sols = {round(s.meta["sweep_params"]["a0"] * 1e6): s for s in _solve(m, geo, "es_cd_boxsel_")}
    assert sols[0].meta["selections"][an.tag] == flat.named[bs_an.tag][1]
    assert an.selection.entities == wavy.named[bs_an.tag][1]             # restored after the sweep
    iex = exact_current(KINETICS["linear"], V, L, sigma) * lam
    I = {a: float(s.evaluate(f"cd.{an.tag}.Itot", s.frame_index("last"))[0]) for a, s in sols.items()}
    assert abs(I[0] / iex - 1) < 1e-8, I                                  # flat run: exact 1D cell
    assert 0 < I[1] / iex - 1 < 1e-2, I                                   # wavy: slightly more area, more current


def test_face_currents_add_up_in_3d(elmer):
    """3D, quadratic tetrahedra: Surface Integration of cd.iloc over each face of a plate electrode adds up to the
    electrode current from the reaction loads (cd.es1.Itot) to round-off.  Integrating the nodal flux with the
    quadratic shape functions instead would drop the corner loads (control below)."""
    m = B.new_model("3D", ["cd"], "stationary")
    g = m.component.child("geometry")
    g.props["unit"] = "mm"
    bath = m.create("geom.block", g, index=0, w="20", d="20", h="10", x="-10", y="-10", z="-10")
    plate = m.create("geom.block", g, index=1, w="6", d="1", h="8", x="-3", y="-0.5", z="-5")
    m.create("geom.difference", g, index=2).props.update(input=[bath.tag], tools=[plate.tag])
    faces = [m.create("geom.boxsel", g, index=3 + i, level="boundary", **lim) for i, lim in enumerate([
        dict(xmin="-3", xmax="3", ymin="-0.5", ymax="-0.5", zmin="-5", zmax="0"),
        dict(xmin="-3", xmax="3", ymin="0.5", ymax="0.5", zmin="-5", zmax="0"),
        dict(xmin="-3", xmax="-3", ymin="-0.5", ymax="0.5", zmin="-5", zmax="0"),
        dict(xmin="3", xmax="3", ymin="-0.5", ymax="0.5", zmin="-5", zmax="0"),
        dict(xmin="-3", xmax="3", ymin="-0.5", ymax="0.5", zmin="-5", zmax="-5")])]
    plate_sel = m.create("geom.boxsel", g, index=8, level="boundary", xmin="-3", xmax="3", ymin="-0.5", ymax="0.5",
                         zmin="-5", zmax="0")
    wall = m.create("geom.boxsel", g, index=9, level="boundary", ymin="10")
    cd = m.component.child("physics.cd")
    cd.props["dist"] = "primary"
    cd.child("cd.elyte").props.update(sigma_src="user", sigma="1")
    an = m.create("cd.es", cd, phis="1")
    an.selection.named = plate_sel.tag
    m.create("cd.es", cd, phis="0").selection.named = wall.tag
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.size", mesh, mode="custom", hmax="2.5", hmin="0.1", hgrad="1.4")
    m.create("mesh.ftet", mesh)
    geo = build_geometry(m, display=False)
    assert len(an.selection.entities) == 5
    sol = _solve(m, geo, "es_cd_faces3d_")[-1]
    i = sol.frame_index("last")
    I = float(sol.evaluate(f"cd.{an.tag}.Itot", i)[0])
    parts = [sol.integrate("cd.iloc", i, "boundary", geo.named[f.tag][1]) for f in faces]
    assert abs(sum(parts) / I - 1) < 1e-9, (parts, I)
    assert abs(sol.integrate("cd.iloc", i, "boundary", an.selection.entities) / I - 1) < 1e-9
    consistent = sol._integrate_values(sol.evaluate("cd.iloc", i), "boundary", an.selection.entities)
    assert abs(consistent / I - 1) > 1e-3                                # the control: shape-function quadrature


def test_uniform_current_density_on_quadratic_triangles(elmer):
    """3D box, anode and cathode on opposite faces: the current density is uniform, so cd.iloc must read sigma V/L
    at every anode node -- corner nodes of the 6-node triangles included (dividing the nodal loads by a lumped
    measure gave 0 there, since a uniform flux loads only the mid-side nodes)."""
    sigma, V, L = 5.0, 1.0, 1e-3
    m = B.new_model("3D", ["cd"], "stationary")
    g = m.component.child("geometry")
    m.create("geom.block", g, index=0, w="2e-3", d="1e-3", h=str(L))
    geo = build_geometry(m, display=False)
    cd = m.component.child("physics.cd")
    cd.props["dist"] = "primary"
    cd.child("cd.elyte").props.update(sigma_src="user", sigma=str(sigma))
    an = B.add_feature(m, cd, "cd.es", geo.select("boundary", zmax=1e-12), phis=str(V))
    B.add_feature(m, cd, "cd.es", geo.select("boundary", zmin=L * (1 - 1e-9)), phis="0")
    sol = _solve(m, geo, "es_cd_box3d_")[-1]
    i = sol.frame_index("last")
    j = sol.evaluate("cd.iloc", i)
    on = np.isfinite(j) & (np.abs(sol.topo.nodes[:, 2] * sol.scale) < 1e-12)
    assert on.sum() > 20
    assert np.allclose(j[on], sigma * V / L, rtol=1e-6), (j[on].min(), j[on].max())
    I = float(sol.evaluate(f"cd.{an.tag}.Itot", i)[0])
    assert abs(sol.integrate("cd.iloc", i, "boundary", an.selection.entities) / I - 1) < 1e-9


def test_curve_coordinates_follow_length_units():
    """Parametric Curve and Polygon coordinates: expressions carrying units (directly or through parameters)
    are converted to the geometry unit; plain numbers are in the geometry unit (COMSOL semantics)."""
    m = B.new_model("2D", ["cd"], "stationary")
    m.global_defs.child("params").props["table"] = [{"name": "W", "expr": "5[mm]", "descr": ""},
                                                     {"name": "H", "expr": "2", "descr": "plain number"}]
    g = m.component.child("geometry")
    g.props["unit"] = "mm"
    pc = m.create("geom.paramcurve", g, index=0, xexpr="W*s", yexpr="0.1[mm]*sin(2*pi*s)", npts=41)
    po = m.create("geom.polygon", g, index=1, ptype="open", xs="W W 0 0", ys="0 H H 0")
    cs = m.create("geom.tosolid", g, index=2)
    cs.props["input"] = [pc.tag, po.tag]
    geo = build_geometry(m, display=False)
    b = geo.bbox
    assert abs(b[3] - 5.0) < 1e-6 and abs(b[4] - 2.0) < 1e-6 and abs(b[1] + 0.1) < 1e-3, b
