"""OpenFOAM backend: closed-form checks, their controls, numbering, the convergence judge, refusals, persistence.

Every positive check has a control that must fail or change when the mapping is wrong.  Needs OpenFOAM for the
solver runs (the refusal checks without geometry run everywhere).

Run:  python -m pytest tests/test_openfoam.py -s     (-s prints the measured numbers)
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core import project
from elmerstudio.core.backends.openfoam import dicts
from elmerstudio.core.backends.openfoam.blocks import BlockRefusal
from elmerstudio.core.backends.openfoam.case import FoamRefusal, check_model
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyNotConverged, StudyRunner

H, UB = 0.01, 0.001


def run(m, st=None, events=None):
    st = st or m.studies()[0]
    st.props["backend"] = "openfoam"
    wd = tempfile.mkdtemp(prefix="es_foam_")
    cb = (lambda k, p: events.append((k, p))) if events is not None else None
    return [Solution(f) for f in StudyRunner(m, st, wd, None, on_event=cb).run()]


def channel(N=16, Lf=2.0, inlet="parabolic", rho=None, mu=None, swap=False, two_domains=False):
    """2D channel H x Lf*H, N cells across; parabolic (developed) or uniform inlet at x=0 (x=L when swapped)."""
    m = B.new_model("2D", ["spf"], "stationary")
    m.global_defs.child("params").props["table"] = [
        {"name": "H", "expr": f"{H}[m]", "descr": ""}, {"name": "Ub", "expr": f"{UB}[m/s]", "descr": ""},
        {"name": "Lc", "expr": f"{Lf}*H", "descr": ""}, {"name": "h_el", "expr": f"H/{N}", "descr": ""}]
    g = m.component.child("geometry")
    if two_domains:
        m.create("geom.rectangle", g, index=0, w="Lc/2", h="H")
        m.create("geom.rectangle", g, index=1, w="Lc/2", h="H", x="Lc/2")
    else:
        m.create("geom.rectangle", g, index=0, w="Lc", h="H")
    B.add_material(m, "Water, liquid", all_domains=True)
    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="h_el", hmin="h_el/10")
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.spf")
    fp = next(f for f in ph.children if f.kind == "spf.fp")
    if rho is not None:
        fp.props.update({"rho_src": "user", "rho": f"{rho}[kg/m^3]"})
    if mu is not None:
        fp.props.update({"mu_src": "user", "mu": f"{mu}[Pa*s]"})
    L = Lf * H
    x_in, x_out = (L, 0.0) if swap else (0.0, L)
    b_in = geo.select("boundary", xmin=x_in, xmax=x_in)
    b_out = geo.select("boundary", xmin=x_out, xmax=x_out)
    if inlet == "parabolic":
        B.add_feature(m, ph, "spf.inlet", b_in, spec="field", ux="6*Ub*y*(H-y)/H^2", uy="0")
    else:
        B.add_feature(m, ph, "spf.inlet", b_in, U0="Ub")
    B.add_feature(m, ph, "spf.outlet", b_out)
    return m, geo, b_in, b_out


def column(sol, name, x):
    """(y, values) of a node field on the grid line x."""
    P = sol.topo.nodes * sol.scale
    s = np.isclose(P[:, 0], x, rtol=0, atol=1e-12)
    o = np.argsort(P[s, 1])
    return P[s, 1][o], sol.names(sol.frame_index("last"))[name][s][o]


def mu_of(sol):
    return sol.meta["foam"]["nu"] * sol.meta["foam"]["rho"]


def interior_dp(sol, L):
    """p(L/4) - p(3L/4): mean over the grid lines (the developed region, away from inlet and outlet)."""
    return column(sol, "p", L / 4)[1].mean() - column(sol, "p", 3 * L / 4)[1].mean()


def kinematic_p(sol) -> np.ndarray:
    case = os.path.join(sol.folder, "foam")
    t = max((float(d), d) for d in os.listdir(case) if _is_time(d) and float(d) > 0)[1]
    return dicts.read_internal_field(os.path.join(case, t, "p"), 1)[0]


def _is_time(d):
    try:
        float(d)
        return True
    except ValueError:
        return False


# =========================================================================== refusals (no OpenFOAM needed)
def test_refusals_without_geometry_name_the_reason():
    m = B.new_model("3D", ["solid"], "stationary")
    assert "not supported by the OpenFOAM backend" in check_model(m, m.studies()[0])
    m = B.new_model("3D", ["solid"], "eigen")
    assert "Eigenfrequency" in check_model(m, m.studies()[0])
    m = B.new_model("2D", ["acpr"], "freq")
    assert "Frequency Domain" in check_model(m, m.studies()[0])
    m = B.new_model("2Daxi", ["ht"], "stationary")
    assert "axisymmetric" in check_model(m, m.studies()[0])
    m = B.new_model("2D", ["spf"], "stationary")
    B.add_feature(m, m.component.child("physics.spf"), "spf.vf", [1])
    assert "Volume Force" in check_model(m, m.studies()[0])
    m = B.new_model("2D", ["ht"], "stationary")
    B.add_feature(m, m.component.child("physics.ht"), "ht.radiation", [1])
    assert "Radiation" in check_model(m, m.studies()[0])
    m = B.new_model("2D", ["spf", "ht"], "stationary")
    assert "one at a time" in check_model(m, m.studies()[0])
    # control: the supported models pass
    m = B.new_model("2D", ["spf"], "stationary")
    assert check_model(m, m.studies()[0]) is None
    m = B.new_model("3D", ["ht"], "time")
    assert check_model(m, m.studies()[0]) is None


def test_old_studies_default_to_elmer():
    from elmerstudio.core.backends import study_backend
    from elmerstudio.core.model import Model
    m = B.new_model("2D", ["ht"], "stationary")
    d = m.to_dict()

    def strip(n):
        if isinstance(n, dict):
            n.pop("backend", None)
            for v in n.values():
                strip(v)
        elif isinstance(n, list):
            for v in n:
                strip(v)
    strip(d)
    assert '"backend"' not in str(d).replace("'", '"')
    st = Model.from_dict(d).studies()[0]                # a file from before the setting: the default applies
    assert st.get("backend") == "elmer" and study_backend(st) == "elmer"
    st.props["backend"] = "openfoam"
    assert study_backend(st) == "openfoam"


# =========================================================================== refusals at Compute
def test_a_circle_is_refused_before_anything_runs(openfoam):
    m = B.new_model("2D", ["ht"], "stationary")
    m.create("geom.circle", m.component.child("geometry"), index=0, r="0.1")
    B.add_material(m, "Copper", all_domains=True)
    with pytest.raises(BlockRefusal) as e:
        run(m)
    msg = str(e.value)
    print("circle:", msg)
    assert "curved" in msg and "Elmer" in msg


def test_a_rectangle_union_with_a_circle_is_refused(openfoam):
    m = B.new_model("2D", ["ht"], "stationary")
    g = m.component.child("geometry")
    m.create("geom.rectangle", g, index=0, w="1", h="1")
    m.create("geom.circle", g, index=1, r="0.2", x="1", y="0.5")
    B.add_material(m, "Copper", all_domains=True)
    with pytest.raises(BlockRefusal, match="curved"):
        run(m)


def test_solid_mechanics_and_eigenfrequency_are_refused_at_compute(openfoam):
    for phys, study, needle in (("solid", "stationary", "Laminar Flow and Heat Transfer"),
                                ("solid", "eigen", "Eigenfrequency")):
        m = B.new_model("3D", [phys], study)
        m.create("geom.block", m.component.child("geometry"), index=0, w="1", d="0.1", h="0.1")
        B.add_material(m, "Structural steel", all_domains=True)
        st = m.studies()[0]
        st.props["backend"] = "openfoam"
        wd = tempfile.mkdtemp(prefix="es_foam_")
        with pytest.raises(FoamRefusal) as e:
            StudyRunner(m, st, wd, None).run()
        print(f"{phys}/{study}:", e.value)
        assert needle in str(e.value) and "Elmer" in str(e.value)
        assert not os.path.isdir(os.path.join(wd, st.tag)), "nothing may be written before the refusal"


# =========================================================================== Laminar Flow
def test_poiseuille_profile_and_pressure_drop(openfoam):
    """Developed channel flow, N = 16 cells across, 2H long: u(y) = 6 Ub y (H-y)/H^2, dp/dx = -12 mu Ub/H^2."""
    m, geo, b_in, b_out = channel(N=16, Lf=2.0)
    sol = run(m)[0]
    assert sol.meta["status"] == "converged", sol.meta["diagnostics"]
    L = 2 * H
    y, u = column(sol, "u", L / 2)
    uerr = np.max(np.abs(u - 6 * UB * y * (H - y) / H ** 2)) / (1.5 * UB)
    mu = mu_of(sol)
    dp_i = interior_dp(sol, L) / (12 * mu * UB * (L / 2) / H ** 2) - 1
    i = sol.frame_index("last")
    dp_in = sol.integrate("p", i, "boundary", b_in, average=True) / (12 * mu * UB * L / H ** 2) - 1
    vmax = np.abs(sol.names(i)["v"]).max() / UB
    print(f"poiseuille N=16: u max err {uerr:.3e} (of 1.5Ub); dp interior {dp_i:+.3e}, inlet-outlet {dp_in:+.3e}; "
          f"|v|max/Ub {vmax:.2e}; {sol.meta['foam']['iterations']} SIMPLE iterations")
    assert uerr < 1e-2 and abs(dp_i) < 1e-2 and abs(dp_in) < 1e-2 and vmax < 1e-2
    # the result names and units of an Elmer solution
    keys = {a for a, *_r in sol.available(i)}
    assert {"spf.U", "spf.u", "spf.v", "spf.u_vec", "p"} <= keys and {"u", "v", "U"} <= set(sol.names(i))


def test_poiseuille_mesh_convergence_is_second_order(openfoam):
    errs_dp, errs_u = [], []
    for N in (8, 16, 32):
        m, *_r = channel(N=N, Lf=2.0)
        sol = run(m)[0]
        L = 2 * H
        errs_dp.append(abs(interior_dp(sol, L) / (12 * mu_of(sol) * UB * (L / 2) / H ** 2) - 1))
        y, u = column(sol, "u", L / 2)
        errs_u.append(np.max(np.abs(u - 6 * UB * y * (H - y) / H ** 2)) / (1.5 * UB))
    p_dp = [math.log2(errs_dp[k] / errs_dp[k + 1]) for k in range(2)]
    p_u = [math.log2(errs_u[k] / errs_u[k + 1]) for k in range(2)]
    print("dp errors", [f"{e:.3e}" for e in errs_dp], "orders", [f"{p:.2f}" for p in p_dp])
    print("u errors ", [f"{e:.3e}" for e in errs_u], "orders", [f"{p:.2f}" for p in p_u])
    assert all(1.8 < p < 2.2 for p in p_dp + p_u)


def test_density_conversion_is_exercised(openfoam):
    """Kinematic pressure: (a) with nu fixed (rho and mu both doubled) the OpenFOAM case is the same, the velocity
    is identical and p in Pa doubles exactly; (b) with mu fixed, p in Pa stays (physics) while the kinematic p
    in the case files halves."""
    base = run(channel(N=8, rho=1000, mu=1e-3)[0])[0]
    same_nu = run(channel(N=8, rho=2000, mu=2e-3)[0])[0]
    same_mu = run(channel(N=8, rho=2000, mu=1e-3)[0])[0]
    i = 0
    u0, p0 = base.names(i)["u"], base.names(i)["p"]
    du = np.max(np.abs(same_nu.names(i)["u"] - u0)) / UB
    rp = same_nu.names(i)["p"][np.abs(p0) > 1e-3 * np.abs(p0).max()] / p0[np.abs(p0) > 1e-3 * np.abs(p0).max()]
    print(f"rho x2, nu fixed: |du|/Ub {du:.1e}, p ratio {rp.min():.12f}..{rp.max():.12f}")
    assert du < 1e-12 and np.allclose(rp, 2.0, rtol=1e-9, atol=0)
    pk0, pk1 = kinematic_p(base), kinematic_p(same_mu)
    big = np.abs(pk0) > 1e-3 * np.abs(pk0).max()
    kr = pk1[big] / pk0[big]
    dpa = np.max(np.abs(same_mu.names(i)["p"] - p0)) / np.abs(p0).max()
    print(f"rho x2, mu fixed: kinematic p ratio {np.median(kr):.6f}, p (Pa) max rel change {dpa:.2e}")
    # Re doubles, which moves the discrete entrance adjustment a little (a few 1e-3 here): physics, not units
    assert np.allclose(np.median(kr), 0.5, rtol=1e-2) and dpa < 1e-2


def test_swapping_inlet_and_outlet_flips_the_flow(openfoam):
    m, geo, b_in, b_out = channel(N=8, inlet="uniform", swap=True)
    sol = run(m)[0]
    L = 2 * H
    i = sol.frame_index("last")
    y, u = column(sol, "u", L / 2)
    p_right = sol.integrate("p", i, "boundary", b_in, average=True)
    p_left = sol.integrate("p", i, "boundary", b_out, average=True)
    print(f"swapped: centre u {u[len(u) // 2]:.4g} m/s, p(x=L) {p_right:.4g} Pa, p(x=0) {p_left:.4g} Pa")
    assert u[len(u) // 2] < -1.2 * UB and p_right > p_left
    # control: the same channel the right way round flows in +x
    m2, *_r = channel(N=8, inlet="uniform")
    y, u2 = column(run(m2)[0], "u", L / 2)
    assert u2[len(u2) // 2] > 1.2 * UB


def pipeline_channel():
    """The model of tests/test_pipeline.py::test_poiseuille_2d, unchanged (uniform inlet, 10 H long, 0.5 mm)."""
    m = B.new_model("2D", ["spf"], "stationary")
    Lc = 0.1
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(Lc), h=str(H))
    B.add_material(m, "Water, liquid", all_domains=True)
    mesh_node = m.component.child("mesh")
    mesh_node.props["sequence"] = "user"
    m.create("mesh.size", mesh_node, mode="custom", hmax="H_el", hmin="1e-5", hcurve="0.3")
    m.create("mesh.ftri", mesh_node)
    m.global_defs.child("params").props["table"] = [{"name": "H_el", "expr": "0.5[mm]", "descr": "element size"}]
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.spf")
    B.add_feature(m, ph, "spf.inlet", geo.select("boundary", xmax=0), U0=str(UB))
    B.add_feature(m, ph, "spf.outlet", geo.select("boundary", xmin=Lc))
    return m


def test_against_elmer_and_the_closed_form(openfoam, elmer):
    """The Elmer pipeline test's channel, solved by both backends: the outlet centreline against 1.5 Q/H, each with
    its own inflow Q.  They differ in Q: Elmer's no-slip corner nodes take one element off the uniform inlet
    (Q = Ub (H - h) by the nodes), OpenFOAM imposes Ub on every inlet face (Q = Ub H)."""
    sol = run(pipeline_channel())[0]
    of_max = column(sol, "u", 0.1)[1].max()
    q_of = UB * H                                               # the face fluxes of the fixed-value inlet
    me = pipeline_channel()
    se = Solution(StudyRunner(me, me.studies()[0], tempfile.mkdtemp(prefix="es_elmer_"), elmer).run()[0])
    P = se.topo.nodes * se.scale

    def prof(x):
        s = np.isclose(P[:, 0], x, atol=1e-9)
        o = np.argsort(P[s, 1])
        return P[s, 1][o], se.names(0)["u"][s][o]
    q_el = np.trapezoid(*prof(0.0)[::-1])
    el_max = prof(0.1)[1].max()
    e_of, e_el = of_max / (1.5 * q_of / H) - 1, el_max / (1.5 * q_el / H) - 1
    print(f"centreline at outlet vs 1.5 Q/H: OpenFOAM {of_max:.6g} ({e_of:+.2e}, Q = Ub H), Elmer {el_max:.6g} "
          f"({e_el:+.2e}, Q = {q_el / (UB * H):.4f} Ub H); OpenFOAM/Elmer {of_max / el_max:.4f} vs inflow ratio "
          f"{q_of / q_el:.4f}; blockMesh {sol.meta['foam']['cells']} cells")
    assert abs(e_of) < 1e-2 and abs(e_el) < 1e-2
    assert abs((of_max / el_max) / (q_of / q_el) - 1) < 1e-2    # the backends agree once the inflow is the same
    # the same result variables, with the same units
    flow = lambda s: {(a, u) for a, _l, u, _k in s.available(0) if a.startswith("spf.") or a == "p"}  # noqa: E731
    assert flow(se) <= flow(sol), flow(se) - flow(sol)


def test_boundary_integrals_use_studio_numbers(openfoam):
    """Two domains (an interior boundary at x = L/2): the flow through the outlet and through the interior boundary,
    integrated by Derived Values on Studio's boundary numbers, equals the inflow Ub H.

    OpenFOAM's own face fluxes balance to the solver tolerance; Studio integrates the node interpolant, whose
    error is O(h^2): it must fall about fourfold from 20 to 40 cells across."""
    from elmerstudio.core.backends.openfoam.polymesh import PolyMesh
    from elmerstudio.ui.plots import evaluate_derived
    errs = {}
    for N in (20, 40):
        m, geo, b_in, b_out = channel(N=N, Lf=4.0, inlet="uniform", two_domains=True)
        sol = run(m)[0]
        L = 4 * H
        mid = [b for b in geo.select("boundary", xmin=L / 2, xmax=L / 2)]
        walls = [b for b in geo.numbers["boundary"] if b not in b_in + b_out + mid]
        assert len(mid) == 1 and len(walls) == 4 and sorted(sol.topo.bulk) == [1, 2]
        der = m.results.child("results.derived")
        node = m.create("eval.surfint", der, exprs=[{"expr": "u", "unit": "", "descr": ""}])
        i = sol.frame_index("last")

        def flow(ents):
            return evaluate_derived(node, m, sol, [i], ents)[1][0][-1]
        q_out, q_mid = flow(b_out), flow(mid)
        q_len = sol.integrate("1", i, "boundary", b_out)
        q_wall = max(abs(flow([w])) for w in walls)
        # the conservative flux through patch bnd<outlet> (zero-gradient U: face value = cell value)
        case = os.path.join(sol.folder, "foam")
        pm = PolyMesh(case)
        t = max((float(d), d) for d in os.listdir(case) if _is_time(d) and float(d) > 0)[1]
        U = dicts.read_internal_field(os.path.join(case, t, "U"), 3)[0]
        fl = pm.patch_faces(f"bnd{b_out[0]}")
        pts = pm.points[pm.faces[fl]]
        area = np.linalg.norm(np.cross(pts[:, 1] - pts[:, 0], pts[:, 3] - pts[:, 0]), axis=1) / L_thick(pm)
        q_face = float(np.sum(U[pm.owner[fl], 0] * area))
        assert sorted(pm.zones) == ["dom1", "dom2"] and len(pm.zones["dom1"]) == len(pm.zones["dom2"])
        errs[N] = (q_out / (UB * H) - 1, q_mid / (UB * H) - 1, q_face / (UB * H) - 1)
        print(f"N={N} mass balance: outlet {errs[N][0]:+.3e}, interior {errs[N][1]:+.3e} (Studio, of Ub H); "
              f"OpenFOAM face flux {errs[N][2]:+.1e}; walls {q_wall:.1e}; outlet length {q_len / H - 1:+.1e}")
        assert q_wall == 0.0 and abs(q_len / H - 1) < 1e-12       # controls: the numbers reach the right entities
        assert abs(errs[N][2]) < 1e-6
    for k in (0, 1):
        assert abs(errs[20][k]) < 1e-2 and abs(errs[40][k]) < abs(errs[20][k]) / 3


def L_thick(pm) -> float:
    """The z thickness of a 2D (one cell thick) mesh."""
    return float(np.ptp(pm.points[:, 2]))


def test_pressure_driven_flow_converts_the_inlet_pressure(openfoam):
    """Inlet with pressure p0 (Pa), outlet p = 0: Q = H^3 p0/(12 mu L) per unit depth.  An inlet pressure handed to
    OpenFOAM in Pa instead of p0/rho would make Q a thousand times too large."""
    m, geo, b_in, b_out = channel(N=16, Lf=4.0, rho=1000, mu=1e-3)
    inlet = next(f for f in m.component.child("physics.spf").children if f.kind == "spf.inlet")
    inlet.props.update({"type": "pressure", "p0": "0.012[Pa]"})
    sol = run(m)[0]
    L = 4 * H
    q = sol.integrate("u", 0, "boundary", b_in)
    q_exact = H ** 3 * 0.012 / (12 * 1e-3 * L)
    y, u = column(sol, "u", L / 2)
    print(f"pressure-driven: Q {q:.6e} vs {q_exact:.6e} m^2/s ({q / q_exact - 1:+.2e}); p at inlet "
          f"{sol.integrate('p', 0, 'boundary', b_in, average=True):.6g} Pa")
    assert abs(q / q_exact - 1) < 1e-2
    assert abs(sol.integrate("p", 0, "boundary", b_in, average=True) / 0.012 - 1) < 1e-9


def test_slip_walls_give_plug_flow(openfoam):
    m, geo, b_in, b_out = channel(N=8, inlet="uniform")
    walls = [b for b in geo.numbers["boundary"] if b not in b_in + b_out]
    B.add_feature(m, m.component.child("physics.spf"), "spf.wall", walls, cond="slip")
    ev = []
    sol = run(m, events=ev)[0]
    err = np.max(np.abs(sol.names(0)["u"] - UB)) / UB
    said = [p for k, p in ev if k == "message" and "one more iteration" in str(p)]
    print(f"slip walls: max |u - Ub|/Ub {err:.1e}; status {sol.meta['status']}; {said}")
    assert err < 1e-6 and sol.meta["status"] == "converged"
    # control: the no-slip channel is not a plug flow
    m2, *_r = channel(N=8, inlet="uniform")
    assert np.max(np.abs(run(m2)[0].names(0)["u"] - UB)) / UB > 0.3


def test_symmetry_halves_the_channel(openfoam):
    """Lower half of the channel, symmetry on y = H/2: the same profile as the full channel; v = 0 there."""
    m = B.new_model("2D", ["spf"], "stationary")
    m.global_defs.child("params").props["table"] = [
        {"name": "H", "expr": f"{H}[m]", "descr": ""}, {"name": "Ub", "expr": f"{UB}[m/s]", "descr": ""}]
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="2*H", h="H/2")
    B.add_material(m, "Water, liquid", all_domains=True)
    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="H/16", hmin="H/160")
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.spf")
    B.add_feature(m, ph, "spf.inlet", geo.select("boundary", xmax=0), spec="field", ux="6*Ub*y*(H-y)/H^2")
    B.add_feature(m, ph, "spf.outlet", geo.select("boundary", xmin=2 * H))
    top = geo.select("boundary", ymin=H / 2)
    B.add_feature(m, ph, "spf.symmetry", top)
    sol = run(m)[0]
    full = run(channel(N=16)[0])[0]
    y, u = column(sol, "u", H)
    yf, uf = column(full, "u", H)
    half = yf <= H / 2 + 1e-12
    diff = np.max(np.abs(u - uf[half])) / (1.5 * UB)
    v_top = np.max(np.abs(sol.names(0)["v"][np.isclose(sol.topo.nodes[:, 1] * sol.scale, H / 2)])) / UB
    print(f"symmetry: half vs full channel max diff {diff:.1e} (of 1.5Ub); |v| on the symmetry plane {v_top:.1e}")
    assert diff < 1e-3 and v_top < 1e-3


def test_outlet_normal_flow_removes_the_tangential_velocity(openfoam):
    m, geo, b_in, b_out = channel(N=8, Lf=1.0, inlet="uniform")
    outlet = next(f for f in m.component.child("physics.spf").children if f.kind == "spf.outlet")
    outlet.props["normal"] = True
    sol = run(m)[0]
    y, v = column(sol, "v", H)
    m2, *_r = channel(N=8, Lf=1.0, inlet="uniform")              # control: without it, the developing flow has v
    y2, v2 = column(run(m2)[0], "v", H)
    print(f"normal flow outlet: max |v| {np.abs(v).max():.2e} m/s; free outlet {np.abs(v2).max():.2e} m/s")
    assert np.abs(v).max() < 1e-12 and np.abs(v2).max() > 1e-6


def test_3d_plates_with_symmetric_sides_match_2d(openfoam):
    """A 3D box between walls y = 0, H, symmetry on the z faces: plane Poiseuille flow, as in 2D."""
    m = B.new_model("3D", ["spf"], "stationary")
    m.global_defs.child("params").props["table"] = [
        {"name": "H", "expr": f"{H}[m]", "descr": ""}, {"name": "Ub", "expr": f"{UB}[m/s]", "descr": ""}]
    m.create("geom.block", m.component.child("geometry"), index=0, w="2*H", d="H", h="H/4")
    B.add_material(m, "Water, liquid", all_domains=True)
    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="H/16", hmin="H/160")
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.spf")
    B.add_feature(m, ph, "spf.inlet", geo.select("boundary", xmax=0), spec="field", ux="6*Ub*y*(H-y)/H^2")
    B.add_feature(m, ph, "spf.outlet", geo.select("boundary", xmin=2 * H))
    B.add_feature(m, ph, "spf.symmetry", geo.select("boundary", zmax=0) + geo.select("boundary", zmin=H / 4))
    sol = run(m)[0]
    P = sol.topo.nodes * sol.scale
    s = np.isclose(P[:, 0], H)
    u = sol.names(0)["u"][s]
    err = np.max(np.abs(u - 6 * UB * P[s, 1] * (H - P[s, 1]) / H ** 2)) / (1.5 * UB)
    w = np.max(np.abs(sol.names(0)["w"])) / UB
    print(f"3D plates: u max err {err:.3e} (of 1.5Ub), |w|max/Ub {w:.1e}, {sol.meta['foam']['cells']} cells")
    assert sol.sdim == 3 and err < 1e-2 and w < 1e-4


def test_distribution_sets_cells_and_grading(openfoam):
    """A Distribution on the bottom edge: n cells, last/first = ratio along the edge; Reverse flips it.  A
    Distribution over two block intervals is refused."""
    def widths(ratio, reverse=False, split=False):
        m = B.new_model("2D", ["ht"], "stationary")
        g = m.component.child("geometry")
        if split:
            m.create("geom.rectangle", g, index=0, w="1", h="0.5")
            m.create("geom.rectangle", g, index=1, w="0.5", h="0.5", y="0.5")
        else:
            m.create("geom.rectangle", g, index=0, w="1", h="0.5")
        B.add_material(m, "Copper", all_domains=True)
        geo = build_geometry(m, display=False)
        ph = m.component.child("physics.ht")
        B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0="300[K]")
        mn = m.component.child("mesh")
        mn.props.update({"sequence": "user", "order": "1"})
        m.create("mesh.size", mn, mode="custom", hmax="0.1", hmin="0.01")
        m.create("mesh.distribution", mn, n=10, ratio=str(ratio), reverse=reverse).selection.entities = \
            geo.select("boundary", ymax=0)
        gm = build_mesh(m)                                     # Gmsh reads the same Distribution (direction)
        gx = np.sort(gm.nodes[np.isclose(gm.nodes[:, 1], 0), 0])
        sol = run(m)[0]
        P = sol.topo.nodes * sol.scale
        xs = np.sort(P[np.isclose(P[:, 1], 0), 0])
        return np.diff(xs), np.diff(gx)
    from elmerstudio.core.meshing import build_mesh
    d, g = widths(4)
    dr, gr = widths(4, reverse=True)
    print(f"distribution: {len(d)} cells, left/right width ratio {d[-1] / d[0]:.6f} (Gmsh {g[-1] / g[0]:.6f}); "
          f"reversed {dr[-1] / dr[0]:.6f} (Gmsh {gr[-1] / gr[0]:.6f})")
    assert len(d) == 10 and len(g) == 10
    assert abs(d[-1] / d[0] - g[-1] / g[0]) < 1e-6 and abs(dr[-1] / dr[0] - gr[-1] / gr[0]) < 1e-6
    assert abs(max(d[-1] / d[0], d[0] / d[-1]) - 4) < 1e-6 and abs((d[-1] / d[0]) * (dr[-1] / dr[0]) - 1) < 1e-9
    with pytest.raises(BlockRefusal, match="exactly one"):
        widths(4, split=True)


def test_parametric_sweep_runs_one_case_per_value(openfoam):
    m = slab(Q=1e6, n=20)
    st = m.studies()[0]
    sw = m.create("study.sweep", st, index=0, pname="Q", pvalues="1e6 2e6", punit="W/m^3")
    assert sw is not None
    sols = run(m, st)
    t = [s.names(0)["T"].max() for s in sols]
    print(f"sweep: Tmax {t[0]:.6f}, {t[1]:.6f} K (ratio {t[1] / t[0]:.9f}); sweep values "
          f"{[s.meta['sweep_value'] for s in sols]}")
    assert len(sols) == 2 and abs(t[1] / t[0] - 2) < 1e-8


def test_two_components_are_two_cases(openfoam):
    """A study over two components solves each as its own OpenFOAM case, into the component folders Elmer uses;
    a refusal in the second component names it."""
    from elmerstudio.core.study_runner import component_dir
    m = B.new_model("2D", ["ht"], "stationary")
    comps = [m.component, B.add_component(m, "2D")]
    temps = [(300, 400), (500, 200)]
    for comp, (t0, t1) in zip(comps, temps):
        with m.scoped(comp):
            if comp is comps[1]:
                B.add_physics(m, "ht")
            m.create("geom.rectangle", comp.child("geometry"), index=0, w="1", h="0.2")
            B.add_material(m, "Copper", all_domains=True)
            geo = build_geometry(m, display=False)
            ph = comp.child("physics.ht") or next(c for c in comp.children if c.kind == "physics.ht")
            B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0=f"{t0}[K]")
            B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmin=1), T0=f"{t1}[K]")
    st = m.studies()[0]
    B.study_steps(st)[0].props["physics"] = {ph.tag: True for c in comps for ph in B.physics_nodes(m, c)}
    st.props["backend"] = "openfoam"
    wd = tempfile.mkdtemp(prefix="es_foam_")
    folders = StudyRunner(m, st, wd, None).run()
    assert len(folders) == 2
    for comp, folder, (t0, t1) in zip(comps, folders, temps):
        assert os.path.normpath(os.path.dirname(folder)) == os.path.normpath(component_dir(m, os.path.join(wd, st.tag),
                                                                                           comp.tag))
        sol = Solution(folder)
        x = sol.topo.nodes[:, 0] * sol.scale
        err = np.max(np.abs(sol.names(0)["T"] - (t0 + (t1 - t0) * x)))
        print(f"{comp.tag}: {sol.meta['comp']}, max |T - Texact| {err:.1e} K in {folder}")
        assert sol.meta["comp"] == comp.tag and err < 1e-6
    with m.scoped(comps[1]):
        ph2 = next(c for c in comps[1].children if c.kind == "physics.ht")
        B.add_feature(m, ph2, "ht.radiation", [1])
    why = check_model(m, st)
    assert why is not None and "Radiation" in why and comps[1].label in why, why


# =========================================================================== convergence judge
def test_too_few_iterations_is_not_converged_and_follows_the_policy(openfoam):
    m, *_r = channel(N=8)
    st = m.studies()[0]
    sols = B.ensure_solver_configs(m, st)
    sols.children[0].props["nl_maxit"] = 5
    with pytest.raises(StudyNotConverged) as e:
        run(m, st)
    print("stop:", e.value)
    assert "maximum of 5 iterations" in str(e.value)
    st.props["nonconv"] = "keep"
    ev = []
    sol = run(m, st, ev)[0]
    assert sol.meta["status"] == "not converged" and sol.meta["foam"]["iterations"] == 5
    assert any(k == "warning" and "NOT CONVERGED" in p for k, p in ev)
    # control: with the default limit the same model converges
    sols.children[0].props["nl_maxit"] = 0
    assert run(m, st)[0].meta["status"] == "converged"


def test_a_linear_solve_that_stops_early_is_not_converged(openfoam):
    m = slab(Q=1e6)
    st = m.studies()[0]
    sols = B.ensure_solver_configs(m, st)
    sols.children[0].props.update({"maxit": 1, "tol": "1e-14"})
    st.props["nonconv"] = "keep"
    sol = run(m, st)[0]
    print("linear:", sol.meta["diagnostics"])
    assert sol.meta["status"] == "not converged" and "above the tolerance" in sol.meta["diagnostics"][0]


def test_a_solver_crash_is_failed(openfoam, monkeypatch):
    from elmerstudio.core.backends.openfoam import case as foam_case
    real = foam_case._schemes
    monkeypatch.setattr(foam_case, "_schemes", lambda P: real(P).replace("linearUpwind grad(U)", "noSuchScheme"))
    m, *_r = channel(N=8)
    st = m.studies()[0]
    st.props["backend"] = "openfoam"
    wd = tempfile.mkdtemp(prefix="es_foam_")
    with pytest.raises(RuntimeError) as e:
        StudyRunner(m, st, wd, None).run()
    print("crash:", e.value)
    meta = Solution(os.path.join(wd, st.tag, "sol_0_0")).meta
    assert meta["status"] == "failed" and "noSuchScheme" in str(e.value)


# =========================================================================== heat transfer
def block(T_left="300[K]", T_right="400[K]"):
    m = B.new_model("3D", ["ht"], "stationary")
    m.create("geom.block", m.component.child("geometry"), index=0, w="1", d="0.2", h="0.1")
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0=T_left)
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmin=1), T0=T_right)
    return m


def slab(Q=1e6, n=40, k=None, rho=None, cp=None, right=None):
    """Slab 0 < x < 0.1 of steel, T = 0 at x = 0 (and at x = L unless *right* gives another condition)."""
    m = B.new_model("3D", ["ht"], "stationary")
    m.global_defs.child("params").props["table"] = [
        {"name": "L", "expr": "0.1[m]", "descr": ""}, {"name": "Q", "expr": f"{Q}[W/m^3]", "descr": ""}]
    m.create("geom.block", m.component.child("geometry"), index=0, w="L", d="0.02", h="0.02")
    B.add_material(m, "Structural steel", all_domains=True)
    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax=f"L/{n}", hmin=f"L/{10 * n}")
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    solid = next(f for f in ph.children if f.kind == "ht.solid")
    for key, v, u in (("k", k, "W/(m*K)"), ("rho", rho, "kg/m^3"), ("Cp", cp, "J/(kg*K)")):
        if v is not None:
            solid.props.update({f"{key}_src": "user", key: f"{v}[{u}]"})
    if Q:
        B.add_feature(m, ph, "ht.source", [1], Q0="Q")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0="0[K]")
    if right is None:
        B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmin=0.1), T0="0[K]")
    else:
        B.add_feature(m, ph, "ht.heatflux", geo.select("boundary", xmin=0.1), **right)
    return m


def test_heat_block_linear_is_exact(openfoam):
    sol = run(block())[0]
    T = sol.names(0)["T"]
    x = sol.topo.nodes[:, 0] * sol.scale
    err = np.max(np.abs(T - (300 + 100 * x)))
    sol2 = run(block("400[K]", "300[K]"))[0]                       # control: the temperatures swapped
    err2 = np.max(np.abs(sol2.names(0)["T"] - (400 - 100 * x)))
    print(f"block: max |T - Texact| {err:.2e} K; swapped {err2:.2e} K; cells {sol.meta['foam']['cells']}")
    assert err < 1e-6 and err2 < 1e-6 and np.max(np.abs(sol2.names(0)["T"] - (300 + 100 * x))) > 50


def test_heat_source_slab_and_its_controls(openfoam):
    """Tmax = Q L^2/(8 k); steady T does not depend on rho or Cp (DT and the source both carry 1/(rho Cp));
    doubling k halves it."""
    sol = run(slab())[0]
    exact = 1e6 * 0.1 ** 2 / (8 * 44.5)
    Tmax = sol.names(0)["T"].max()
    s_rc = run(slab(rho=1000, cp=100))[0]
    s_k = run(slab(k=89))[0]
    r_rc = np.max(np.abs(s_rc.names(0)["T"] - sol.names(0)["T"])) / Tmax
    r_k = s_k.names(0)["T"].max() / Tmax
    print(f"slab: Tmax {Tmax:.6f} vs {exact:.6f} K, rel err {Tmax / exact - 1:+.2e}; rho,Cp changed: max rel diff "
          f"{r_rc:.1e}; k doubled: Tmax ratio {r_k:.9f}")
    assert abs(Tmax / exact - 1) < 1e-3 and r_rc < 1e-8 and abs(r_k - 0.5) < 1e-8


def test_heat_source_slab_against_elmer(openfoam, elmer):
    m = slab()
    sol = run(m)[0]
    me = slab()
    se = Solution(StudyRunner(me, me.studies()[0], tempfile.mkdtemp(prefix="es_elmer_"), elmer).run()[0])
    a, b = sol.names(0)["T"].max(), se.names(0)["T"].max()
    exact = 1e6 * 0.1 ** 2 / (8 * 44.5)
    print(f"slab Tmax: OpenFOAM {a:.6f}, Elmer {b:.6f}, exact {exact:.6f} K; OpenFOAM vs Elmer {a / b - 1:+.2e}")
    assert abs(a / b - 1) < 1e-3


def test_heat_flux_and_convection_are_exact(openfoam):
    """1D: T(0) = 0; at x = L a general inward flux q0 gives T(L) = q0 L/k; a convective flux gives
    T(L) = h Text/(k/L + h).  Both profiles are linear, which the discretization reproduces exactly."""
    k, L = 44.5, 0.1
    out = {}
    for name, right, exact in (
            ("flux", {"type": "general", "q0": "1e4[W/m^2]"}, 1e4 * L / k),
            ("flux-", {"type": "general", "q0": "-1e4[W/m^2]"}, -1e4 * L / k),
            ("conv", {"type": "convective", "h": "100[W/(m^2*K)]", "Text": "400[K]"}, 100 * 400 / (k / L + 100))):
        sol = run(slab(Q=0, n=10, right=right))[0]
        x = sol.topo.nodes[:, 0] * sol.scale
        T = sol.names(0)["T"]
        out[name] = (np.max(np.abs(T - exact * x / L)), T[np.isclose(x, L)].mean(), exact)
    print("flux/convection: " + "; ".join(f"{n} T(L) {v[1]:.6f} vs {v[2]:.6f}, max err {v[0]:.1e}"
                                          for n, v in out.items()))
    assert all(v[0] < 1e-6 for v in out.values())
    assert out["flux-"][1] < 0 < out["flux"][1]                   # control: the sign of the inward flux


def test_transient_conduction_against_the_series_solution(openfoam):
    """Bar 0 < x < L, insulated except T(0) = T1 from t = 0, initially T0: the Fourier series solution at
    t = 0.1, 0.25 and 0.5 s.  Control: the same comparison one output step late fails the tolerance."""
    L, T0, T1 = 0.01, 293.15, 393.15
    m = B.new_model("2D", ["ht"], "time")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h="0.001")
    B.add_material(m, "Copper", all_domains=True)
    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax=str(L / 50), hmin=str(L / 500))
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    next(f for f in ph.children if f.kind == "ht.init").props["T0"] = f"{T0}[K]"
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0=f"{T1}[K]")
    step = B.study_steps(m.studies()[0])[0]
    step.props.update({"times": "range(0,0.05,0.5)", "substeps": 10})
    sol = run(m)[0]
    assert sol.meta["status"] == "converged", sol.meta["diagnostics"]
    a = sol.meta["foam"]["DT"]
    x = sol.topo.nodes[:, 0] * sol.scale

    def series(t):
        th = np.zeros_like(x)
        for n in range(200):
            lam = (2 * n + 1) * math.pi / (2 * L)
            th += 4 / ((2 * n + 1) * math.pi) * np.sin(lam * x) * math.exp(-lam ** 2 * a * t)
        return T1 + (T0 - T1) * th
    times = [fr.value for fr in sol.frames]
    assert len(times) == 10 and abs(times[-1] - 0.5) < 1e-12
    errs, late = {}, {}
    for t in (0.1, 0.25, 0.5):
        i = int(np.argmin(np.abs(np.array(times) - t)))
        T = sol.names(i)["T"]
        errs[t] = np.max(np.abs(T - series(t))) / (T1 - T0)
        late[t] = np.max(np.abs(T - series(t + 0.05))) / (T1 - T0)
    print("transient conduction, max err / dT: " + ", ".join(f"t={t}: {e:.2e} (one step late {late[t]:.2e})"
                                                              for t, e in errs.items()))
    assert all(e < 2e-3 for e in errs.values()) and all(v > 2e-3 for v in late.values())


def test_startup_flow_develops_to_poiseuille(openfoam):
    """pimpleFoam: a channel started from rest with a developed inflow; nu = 1e-3 m^2/s so the viscous time
    H^2/(pi^2 nu) is 10 ms.  Early the centreline lags the parabola, late it is the steady profile."""
    m, *_r = channel(N=16, rho=1, mu=1e-3)
    st = m.studies()[0]
    B.add_step(m, st, "time")
    stat = B.study_steps(st)[0]
    stat.enabled = False
    step = B.study_steps(st)[0]
    step.props.update({"times": "range(0,0.01,0.3)", "substeps": 2})
    sol = run(m, st)[0]
    assert sol.meta["status"] == "converged" and sol.meta["foam"]["app"] == "pimpleFoam", sol.meta["diagnostics"]
    L = 2 * H
    P = sol.topo.nodes * sol.scale
    s = np.isclose(P[:, 0], L / 2)
    y = P[s, 1]
    ue = 6 * UB * y * (H - y) / H ** 2
    early = np.max(np.abs(sol.names(0)["u"][s] - ue)) / (1.5 * UB)
    late = np.max(np.abs(sol.names(len(sol.frames) - 1)["u"][s] - ue)) / (1.5 * UB)
    print(f"start-up: t={sol.frames[0].value:g} s deviation {early:.3e}, t={sol.frames[-1].value:g} s {late:.3e}; "
          f"continuity {sol.meta['foam']['continuity']:.1e}")
    assert early > 5e-2 and late < 1e-2


# =========================================================================== persistence
def test_a_saved_openfoam_solution_reopens_elsewhere(openfoam):
    from elmerstudio.ui.plots import build_scene
    m, *_r = channel(N=8)
    st = m.studies()[0]
    st.props["backend"] = "openfoam"
    wd = project.new_workdir()
    folder = StudyRunner(m, st, wd, None).run()[0]
    u0 = Solution(folder).names(0)["u"]
    out = os.path.join(tempfile.mkdtemp(prefix="es_saved_"), "channel.esm")
    project.save(m, out, wd)
    project.discard_workdir(wd)
    shutil.rmtree(wd, ignore_errors=True)
    assert not os.path.exists(folder)
    m2, wd2 = project.load(out)
    sol = Solution(os.path.join(wd2, st.tag, "sol_0_0"))
    assert sol.meta["backend"] == "openfoam" and sol.meta["status"] == "converged"
    assert np.array_equal(sol.names(0)["u"], u0)
    pgs = B.default_plots(m2, m2.studies()[0], "stationary")
    assert pgs
    scene = build_scene(pgs[0], sol, sol.frame_index("last"), sol.sdim, False, model=m2)
    assert scene is not None
    assert not any("foam/" in n for n in project.solution_files(wd2))   # the case inputs stay out of the file
