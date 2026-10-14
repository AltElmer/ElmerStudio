"""Streamline diffusion (SUPG) in Transport of Diluted Species, and the Normal flow outlet of Laminar Flow.

1. Convection against a fixed outlet value (textbook 1D boundary layer): u c' = D c'', c(0) = 0, c(L) = 1, so
   c = (exp(Pe x/L) - 1)/(exp(Pe) - 1).  At an element Peclet number of 2.5 the Galerkin solution oscillates from
   node to node; the stabilized one stays monotone and matches the exact solution away from the layer.
2. The Leveque problem (textbook): Poiseuille flow over a wall that consumes the species.  Its wall flux and its
   similarity profile, up to the outlet: stabilized transport and a parallel (Normal flow) outflow.
3. Saved models: a Transport of Diluted Species interface without the setting opens unstabilized (its results stay
   as they were); a new one is stabilized.
"""
import math
import tempfile

import numpy as np

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.model import Model
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.results import Solution
from elmerstudio.core.sif import build_sif
from elmerstudio.core.study_runner import StudyRunner


def _solve(m, geo, prefix, elmer):
    st = m.studies()[0]
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix=prefix), elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    return sol


def layer(elmer, stab, pe=200.0, n=80, L=1e-3, D=1e-9):
    m = B.new_model("2D", ["tds"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(L / 20))
    geo = build_geometry(m, display=False)
    inlet, outlet = geo.select("boundary", xmax=0), geo.select("boundary", xmin=L)
    tds = m.component.child("physics.tds")
    tds.props.update(stab=stab, disc="1")
    tds.child("tds.tp").props.update(D=f"{D}[m^2/s]", ux=f"{pe * D / L}[m/s]")
    B.add_feature(m, tds, "tds.conc", inlet, c0="0")
    B.add_feature(m, tds, "tds.conc", outlet, c0="1[mol/m^3]")
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    for b, k in ((inlet, 1), (outlet, 1), (geo.select("boundary", ymax=0), n), (geo.select("boundary", ymin=L / 20), n)):
        B.add_feature(m, mesh, "mesh.distribution", b, n=k)
    sol = _solve(m, geo, "es_supg_", elmer)
    x = sol.topo.nodes[:, 0] * sol.scale
    c = sol.evaluate("c", sol.frame_index("last"))
    bottom = np.abs(sol.topo.nodes[:, 1]) < 1e-12
    o = np.argsort(x[bottom])
    xs, cs = x[bottom][o], c[bottom][o]
    exact = np.expm1(pe * xs / L) / math.expm1(pe)
    return xs, cs, exact


def test_streamline_diffusion_removes_the_oscillation(elmer):
    xs, cg, exact = layer(elmer, False)                      # element Peclet number u h/(2D) = 1.25
    _, cs, _ = layer(elmer, True)
    dg, ds = np.diff(cg), np.diff(cs)
    assert np.any(dg < -1e-2), "Galerkin should oscillate at this element Peclet number"   # the control
    assert np.all(ds > -1e-9), ds.min()                       # monotone
    assert cs.min() > -1e-6 and cs.max() < 1 + 1e-6
    away = xs < 0.8e-3                                        # outside the outlet layer the exact c is < 1e-17
    assert np.max(np.abs(cs[away] - exact[away])) < 1e-6
    assert cg.min() < -0.05, cg.min()                         # Galerkin undershoots (c = -0.11 next to the layer)
    # the layer itself (D/u = 5 um) is thinner than an element (12.5 um): there, no scheme is accurate


def leveque(elmer, normal=True):
    L, H, U, D = 10e-3, 1e-3, 0.05, 2e-9
    m = B.new_model("2D", ["spf", "tds"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(H))
    geo = build_geometry(m, display=False)
    sel = lambda **kw: geo.select("boundary", **kw)          # noqa: E731
    wall, top, inlet, outlet = sel(ymax=0), sel(ymin=H), sel(xmax=0), sel(xmin=L)
    spf = m.component.child("physics.spf")
    spf.child("spf.fp").props.update(rho_src="user", rho="1000[kg/m^3]", mu_src="user", mu="1e-3[Pa*s]")
    B.add_feature(m, spf, "spf.inlet", inlet, spec="field", ux=f"6*{U}*(y/{H})*(1-y/{H})", uy="0")
    B.add_feature(m, spf, "spf.outlet", outlet, normal=normal)
    tds = m.component.child("physics.tds")
    tds.child("tds.tp").props.update(usrc="spf", D=f"{D}[m^2/s]")
    B.add_feature(m, tds, "tds.conc", inlet + top, c0="1[mol/m^3]")
    B.add_feature(m, tds, "tds.conc", wall, c0="0")
    B.add_feature(m, tds, "tds.outflow", outlet)
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    for b, k, kw in ((wall, 100, {}), (top, 100, {}), (inlet, 60, {"ratio": "30", "reverse": True}),
                     (outlet, 60, {"ratio": "30"})):
        B.add_feature(m, mesh, "mesh.distribution", b, n=k, **kw)
    sol = _solve(m, geo, "es_leveque_", elmer)
    P = sol.topo.nodes * sol.scale
    i = sol.frame_index("last")
    c = sol.evaluate("c", i)
    gamma = 6 * U / H
    g43 = math.gamma(4 / 3)

    def profile(y, x):
        eta = y * (gamma / (9 * D * x)) ** (1 / 3)
        s = np.linspace(0, max(float(eta.max()), 1e-9), 4001)
        f = np.concatenate([[0.0], np.cumsum(0.5 * (np.exp(-s[1:] ** 3) + np.exp(-s[:-1] ** 3)) * np.diff(s))])
        return np.interp(eta, s, f) / g43

    rms = []
    for xs in (0.5 * L, L):
        col = np.abs(P[:, 0] - xs) < 1e-9
        y = P[col, 1]
        near = y <= 6 * (9 * D * xs / gamma) ** (1 / 3)
        rms.append(float(np.sqrt(np.mean((c[col][near] - profile(y[near], xs)) ** 2))))
    q = -np.asarray(sol.evaluate("tds.ntflux", i), float)
    ids = np.unique(np.concatenate([np.asarray(cc).ravel() for b in wall for _gt, cc in sol.topo.bnd[b]]))
    x, qq = P[ids, 0], -q[ids]
    o = np.argsort(x)
    x, qq = x[o], qq[o]
    on = x >= 0.5e-3
    flux = float(np.trapezoid(qq[on], x[on]))
    exact = D * (gamma / (9 * D)) ** (1 / 3) / g43 * 1.5 * (L ** (2 / 3) - x[on][0] ** (2 / 3))
    return rms, flux / exact - 1


def test_leveque_wall_flux_and_profile_to_the_outlet(elmer):
    rms, err = leveque(elmer, normal=True)
    assert abs(err) < 0.02, err
    assert max(rms) < 0.01, rms
    rms0, _ = leveque(elmer, normal=False)                   # a plain pressure outlet disturbs the outlet station
    assert rms0[1] > 2 * rms[1], (rms0, rms)


def test_streamline_diffusion_in_the_sif_and_in_saved_models():
    m = B.new_model("2D", ["tds"], "stationary")
    tds = m.component.child("physics.tds")
    assert tds.get("stab") is True                           # new interfaces are stabilized
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1", h="1")
    geo = build_geometry(m, display=False)
    B.add_feature(m, tds, "tds.flux", geo.select("boundary", xmax=0), N0="2")
    st = m.studies()[0]
    text = build_sif(m, st, B.study_steps(st)[0], geo, 100).text
    assert '"AdvectionDiffusion"' in text and "Concentration Flux" in text and "Field Flux" not in text
    tds.props["stab"] = False
    text = build_sif(m, st, B.study_steps(st)[0], geo, 100).text
    assert '"ModelPDE"' in text and "Field Flux" in text
    tds.props["stab"] = True
    B.add_feature(m, tds, "tds.reac", None, k="0.1").selection.all = True   # a first-order rate: Galerkin, warned
    res = build_sif(m, st, B.study_steps(st)[0], geo, 100)
    assert '"ModelPDE"' in res.text and any("without streamline diffusion" in w for w in res.meta["warnings"])
    d = m.to_dict()
    stack = [d["root"]]                                      # strip the setting, as an older file has it
    while stack:
        x = stack.pop()
        if x["kind"] == "physics.tds":
            x["props"].pop("stab", None)
        stack.extend(x.get("children", []))
    assert Model.from_dict(d).component.child("physics.tds").get("stab") is False


def test_normal_flow_outlet_in_the_sif():
    m = B.new_model("2D", ["spf"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="2", h="1")
    geo = build_geometry(m, display=False)
    spf = m.component.child("physics.spf")
    spf.child("spf.fp").props.update(rho_src="user", rho="1000", mu_src="user", mu="1e-3")
    out = B.add_feature(m, spf, "spf.outlet", geo.select("boundary", xmin=2))
    st = m.studies()[0]
    assert "Normal-Tangential Velocity" not in build_sif(m, st, B.study_steps(st)[0], geo, 100).text
    out.props["normal"] = True
    text = build_sif(m, st, B.study_steps(st)[0], geo, 100).text
    assert "Normal-Tangential Velocity = Logical True" in text and "Velocity 2 = Real 0.0" in text
