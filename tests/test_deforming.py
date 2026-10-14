"""Deforming electrodes (Deforming Domain + Electrode Surface > Deforming electrode surface) in a Time Dependent study.

Checks against closed forms:
* a flat anode receding under a primary current distribution: the gap grows as the metal dissolves,
  i = sigma V / (L0 + h), dh/dt = c i with c = M/(n F rho)  ->  (L0 + h)^2 = L0^2 + 2 c sigma V t  (Heun: 2nd order);
* a sinusoidal anode levels at d(ln a)/d(hbar) = -k coth(kL) (linear theory; all material constants cancel), and a
  uniform recession velocity (current efficiency cancelling the current) gives no levelling (negative control);
* remeshing keeps the boundary numbers and the solution continuous.
"""
import math
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner

F = 96485.33212
M, NEL, RHO = 55.845e-3, 2.0, 7870.0
C = M / (NEL * F * RHO)


def cell(elmer, amp_um, n_x, n_y, times, eff="1", integrator="heun", lam_um=200.0, gap_um=300.0, V=0.05, sigma=10.0,
         remesh=False, qmin="0.2"):
    m = B.new_model("2D", ["cd"], "time")
    m.global_defs.child("params").props["table"] = [
        {"name": "lam", "expr": f"{lam_um}[um]", "descr": ""}, {"name": "L", "expr": f"{gap_um}[um]", "descr": ""},
        {"name": "a0", "expr": f"{amp_um}[um]", "descr": ""}, {"name": "V", "expr": f"{V}[V]", "descr": ""}]
    g = m.component.child("geometry")
    g.props["unit"] = "um"
    pc = m.create("geom.paramcurve", g, index=0, xexpr="lam*s", yexpr="a0*cos(2*pi*s)", npts=121)
    rest = m.create("geom.polygon", g, index=1, ptype="open", xs="lam lam 0 0", ys="a0 L L a0")
    cs = m.create("geom.tosolid", g, index=2)
    cs.props["input"] = [pc.tag, rest.tag]
    sel = {}
    for i, (name, lim) in enumerate({"Anode": dict(ymax="L/2"), "Cathode": dict(ymin="L/2"),
                                     "Left": dict(xmax="lam/10"), "Right": dict(xmin="0.9*lam")}.items()):
        b = m.create("geom.boxsel", g, index=3 + i, level="boundary", **lim)
        sel[name] = b.tag
    build_geometry(m, display=False)
    cd = m.component.child("physics.cd")
    cd.props["dist"] = "primary"
    cd.child("cd.elyte").props.update(sigma_src="user", sigma=f"{sigma}[S/m]")
    an = m.create("cd.es", cd, phis="0[V]", kin="primary", moving=True, eff=eff, M=f"{M}[kg/mol]", nel=str(NEL),
                  rhom=f"{RHO}[kg/m^3]")
    an.selection.named = sel["Anode"]
    m.create("cd.pot", cd, phil0="-V").selection.named = sel["Cathode"]
    for w in ("Left", "Right"):
        m.create("cd.sym", cd).selection.named = sel[w]
    m.create("dg.ddom", m.component.child("definitions"), integrator=integrator, remesh=remesh, qmin=qmin)
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    for name, n in (("Anode", n_x), ("Cathode", n_x), ("Left", n_y), ("Right", n_y)):
        m.create("mesh.distribution", mesh, n=n).selection.named = sel[name]
    geo = build_geometry(m, display=False)
    st = m.studies()[0]
    step = next(c for c in st.children if c.kind == "study.time")
    step.props["times"] = times
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_def_"), elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    return m, sol, an, lam_um * 1e-6, gap_um * 1e-6, V, sigma


def anode_profile(sol, an, i):
    sol.frame_index(i)
    ents = sol.meta["selections"][an.tag]
    ids = np.unique(np.concatenate([np.asarray(c).ravel() for b in ents for _gt, c in sol.topo.bnd[b]]))
    P = sol.topo.nodes[ids] * sol.scale
    o = np.argsort(P[:, 0])
    return P[o, 0], P[o, 1]


def test_flat_anode_recedes_like_the_closed_form(elmer):
    """(L0 + h)^2 = L0^2 + 2 c sigma V t: the computed recession at every output time, Heun time stepping."""
    L0, V, sigma = 300e-6, 0.05, 10.0
    tend = 20e-6 / (C * sigma * V / L0)                 # about 20 um of metal removed
    m, sol, an, lam, L0, V, sigma = cell(elmer, 0.0, 6, 12, f"range(0,{tend / 5!r},{tend!r})")
    assert sol.meta["moving_mesh"] and len(sol.frames) == 6
    ents = sol.meta["selections"][an.tag]
    for i, fr in enumerate(sol.frames):
        x, y = anode_profile(sol, an, i)
        h = -float(np.mean(y))
        exact = math.sqrt(L0 ** 2 + 2 * C * sigma * V * fr.value) - L0
        assert abs(h - exact) <= 2e-4 * max(exact, 1e-9) + 1e-12, (i, h, exact)
        assert np.ptp(y) < 1e-12                           # stays flat
        # the current falls as the gap grows: I = sigma V lam / (L0 + h) per metre of depth
        itot = sol.integrate("cd.iloc", i, "boundary", ents)
        assert abs(itot / (sigma * V * lam / (L0 + h)) - 1) < 1e-9, (i, itot)


def test_wavy_anode_levels_at_the_analytic_rate_and_the_control_does_not(elmer):
    lam, a0 = 200e-6, 2e-6
    k = 2 * math.pi / lam
    tend = 16e-6 / (C * 10.0 * 0.05 / 300e-6)

    def slope(sol, an):
        amps, hb = [], []
        for i in range(len(sol.frames)):
            x, y = anode_profile(sol, an, i)
            amps.append(2 * np.trapezoid(y * np.cos(k * x), x) / lam)
            hb.append(-np.trapezoid(y, x) / lam)
        hb = np.array(hb) - hb[0]
        return np.polyfit(hb, np.log(np.abs(amps)), 1)[0], amps

    m, sol, an, *_ = cell(elmer, a0 * 1e6, 30, 36, f"range(0,{tend / 8!r},{tend!r})")
    s, amps = slope(sol, an)
    kl = k * 300e-6
    assert abs(s / (-k / math.tanh(kl)) - 1) < 0.01, s      # linear theory holds to O((a k)^2) = 0.4 %
    assert amps[-1] / amps[0] < 0.8
    m, ctrl, an, *_ = cell(elmer, a0 * 1e6, 30, 36, f"range(0,{tend / 4!r},{tend!r})", eff="1[A/m^2]/iloc")
    s_ctrl, amps_ctrl = slope(ctrl, an)
    assert abs(s_ctrl / (-k)) < 0.02, s_ctrl                # uniform recession: no levelling


def test_flat_anode_recedes_in_3d(elmer):
    """3D, tetrahedra: the bottom face of a box dissolves, the side walls slide; same closed form as in 2D."""
    L0, V, sigma = 300e-6, 0.05, 10.0
    m = B.new_model("3D", ["cd"], "time")
    g = m.component.child("geometry")
    g.props["unit"] = "um"
    m.create("geom.block", g, index=0, w="200", d="100", h="300")
    geo = build_geometry(m, display=False)
    cd = m.component.child("physics.cd")
    cd.props["dist"] = "primary"
    cd.child("cd.elyte").props.update(sigma_src="user", sigma=f"{sigma}[S/m]")
    an = B.add_feature(m, cd, "cd.es", geo.select("boundary", zmax=0), phis="0[V]", kin="primary", moving=True,
                       M=f"{M}[kg/mol]", nel=str(NEL), rhom=f"{RHO}[kg/m^3]")
    B.add_feature(m, cd, "cd.pot", geo.select("boundary", zmin=300), phil0=f"-{V}[V]")
    m.create("dg.ddom", m.component.child("definitions"))
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.size", mesh, mode="custom", hmax="60", hmin="10")
    m.create("mesh.ftet", mesh)
    st = m.studies()[0]
    tend = 20e-6 / (C * sigma * V / L0)
    next(c for c in st.children if c.kind == "study.time").props["times"] = f"range(0,{tend / 2!r},{tend!r})"
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_def3_"), elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    ents = list(an.selection.entities)
    for i, fr in enumerate(sol.frames):
        sol.frame_index(i)
        ids = np.unique(np.concatenate([np.asarray(c).ravel() for b in ents for _gt, c in sol.topo.bnd[b]]))
        z = sol.topo.nodes[ids, 2] * sol.scale
        h = -float(np.mean(z))
        exact = math.sqrt(L0 ** 2 + 2 * C * sigma * V * fr.value) - L0
        assert abs(h - exact) <= 2e-4 * max(exact, 1e-9) + 1e-12, (i, h, exact)
        assert np.ptp(z) < 1e-12
        lo, hi = sol.topo.nodes.min(axis=0) * sol.scale, sol.topo.nodes.max(axis=0) * sol.scale
        assert abs(lo[0]) < 1e-12 and abs(hi[0] - 200e-6) < 1e-12        # the side walls did not move
        itot = sol.integrate("cd.iloc", i, "boundary", ents)
        assert abs(itot / (sigma * V * 200e-6 * 100e-6 / (L0 + h)) - 1) < 1e-6, (i, itot)


def test_transport_limited_levelling_with_prescribed_mesh_velocity(elmer):
    """Transport-limited dissolution: a saturated wavy anode under a diffusion layer whose flat outer edge sits a fixed
    delta above the MEAN surface.  The anode moves with the species flux (Prescribed Normal Mesh Velocity
    -tds.ntflux*Vm), the layer edge with the anode's mean recession (an Average coupling operator); linear theory:
    d(ln a)/d(hbar) = -k coth(k delta)."""
    lam, delta, a0, csat, D, Vm = 200e-6, 50e-6, 1e-6, 1000.0, 1e-9, 7.1e-6
    k = 2 * math.pi / lam
    m = B.new_model("2D", ["tds"], "time")
    m.global_defs.child("params").props["table"] = [
        {"name": "lam", "expr": f"{lam}[m]", "descr": ""}, {"name": "dl", "expr": f"{delta}[m]", "descr": ""},
        {"name": "a0", "expr": f"{a0}[m]", "descr": ""}, {"name": "Vm", "expr": f"{Vm}[m^3/mol]", "descr": ""}]
    g = m.component.child("geometry")
    g.props["unit"] = "um"
    pc = m.create("geom.paramcurve", g, index=0, xexpr="lam*s", yexpr="a0*cos(2*pi*s)", npts=121)
    rest = m.create("geom.polygon", g, index=1, ptype="open", xs="lam lam 0 0", ys="a0 dl dl a0")
    m.create("geom.tosolid", g, index=2).props["input"] = [pc.tag, rest.tag]
    sel = {}
    for i, (name, lim) in enumerate({"Anode": dict(ymax="dl/2"), "Edge": dict(ymin="dl/2"),
                                     "Left": dict(xmax="lam/10"), "Right": dict(xmin="0.9*lam")}.items()):
        sel[name] = m.create("geom.boxsel", g, index=3 + i, level="boundary", **lim).tag
    build_geometry(m, display=False)
    tds = m.component.child("physics.tds")
    tds.child("tds.tp").props["D"] = f"{D}[m^2/s]"
    m.create("tds.conc", tds, c0=f"{csat}[mol/m^3]").selection.named = sel["Anode"]
    m.create("tds.conc", tds, c0="0").selection.named = sel["Edge"]
    for w in ("Left", "Right"):
        m.create("tds.sym", tds).selection.named = sel[w]
    defs = m.component.child("definitions")
    ave = m.create("cpl.average", defs, level="boundary")
    ave.selection.level, ave.selection.all, ave.selection.named = "boundary", False, sel["Anode"]
    m.create("dg.ddom", defs)
    m.create("dg.pnmv", defs, vn="-tds.ntflux*Vm").selection.named = sel["Anode"]
    m.create("dg.pnmv", defs, vn=f"aveop1(tds.ntflux*Vm)").selection.named = sel["Edge"]
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    for name, n in (("Anode", 40), ("Edge", 40), ("Left", 24), ("Right", 24)):
        m.create("mesh.distribution", mesh, n=n).selection.named = sel[name]
    geo = build_geometry(m, display=False)
    v0 = D * csat / delta * Vm
    tend = 16e-6 / v0
    st = m.studies()[0]
    next(c for c in st.children if c.kind == "study.time").props["times"] = f"range(0,{tend / 8!r},{tend!r})"
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_tll_"), elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    an = next(f for f in defs.children if f.kind == "dg.pnmv")
    amps, hb, gap = [], [], []
    for i in range(len(sol.frames)):
        x, y = anode_profile(sol, an, i)
        amps.append(2 * np.trapezoid(y * np.cos(k * x), x) / lam)
        hb.append(-np.trapezoid(y, x) / lam)
        ytop = sol.topo.nodes[:, 1].max() * sol.scale
        gap.append(ytop + hb[-1])
    hb = np.array(hb) - hb[0]
    s = np.polyfit(hb, np.log(np.abs(amps)), 1)[0]
    assert hb[-1] > 10e-6
    assert np.ptp(gap) < 2e-3 * delta, gap                  # the layer edge kept its distance from the mean surface
    assert abs(s / (-k / math.tanh(k * delta)) - 1) < 0.01, s


def test_remeshing_keeps_boundaries_and_solution(elmer):
    """A strongly receding wavy anode, remeshed on the way: frames on new meshes keep the boundary numbers, and the
    anode current follows the same smooth decay before and after a remeshing."""
    tend = 60e-6 / (C * 10.0 * 0.05 / 300e-6)
    m, sol, an, *_ = cell(elmer, 2.0, 16, 10, f"range(0,{tend / 6!r},{tend!r})", remesh=True, qmin="0.85")
    assert sol.meta["remeshes"] >= 1
    assert len(set(sol.meta["frame_mesh"])) >= 2
    ents = sol.meta["selections"][an.tag]
    itot = []
    for i in range(len(sol.frames)):
        sol.frame_index(i)
        assert all(b in sol.topo.bnd for b in ents)
        x, y = anode_profile(sol, an, i)
        assert np.all(np.diff(x) > 0)
        itot.append(float(sol.integrate("cd.iloc", i, "boundary", ents)))
    h = np.array([-np.mean(anode_profile(sol, an, i)[1]) for i in range(len(sol.frames))])
    expect = 10.0 * 0.05 * 200e-6 / (300e-6 + h)            # ~ flat-cell current at the present gap
    assert np.all(np.abs(np.array(itot) / expect - 1) < 0.01), (itot, expect)
