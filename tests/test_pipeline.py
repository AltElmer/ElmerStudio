"""End-to-end pipeline checks against closed-form solutions (needs Elmer installed).

Run:  python tests/test_pipeline.py            (prints one line per case)
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core.elmer import find_elmer, result_files, run_case_blocking, write_case  # noqa: E402
from elmerstudio.core.geometry import build_geometry  # noqa: E402
from elmerstudio.core.meshing import build_mesh  # noqa: E402
from elmerstudio.core.sif import build_sif  # noqa: E402

INST = find_elmer()

try:
    import pytest
    pytestmark = pytest.mark.skipif(INST is None, reason="Elmer (ElmerSolver) not installed")
except ImportError:  # running as a plain script
    pass


def solve(model, step_index=0):
    st = model.studies()[0]
    step = B.study_steps(st)[step_index]
    mesh = build_mesh(model)
    sif = build_sif(model, st, step, mesh.geometry, len(mesh.nodes))
    case = tempfile.mkdtemp(prefix="es_test_")
    write_case(case, mesh.to_elmer(), sif.text)
    rs = run_case_blocking(case, INST)
    files = result_files(case)
    if not files:
        raise AssertionError(f"no results; errors={rs.errors[-5:]} case={case}")
    return mesh, sif, rs, files, case


def read(f):
    import pyvista as pv
    return pv.read(f)


def bulk(g):
    """Keep only volume/area cells (VTU also carries boundary elements)."""
    import pyvista as pv
    dims = np.array([pv.CellType(int(c)).name for c in g.celltypes])
    return g


def test_heat_block_linear():
    m = B.new_model("3D", ["ht"], "stationary")
    m.create("geom.block", m.component.child("geometry"), index=0, w="1", d="0.2", h="0.1")
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0="300[K]")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmin=1), T0="400[K]")
    mesh, sif, rs, files, case = solve(m)
    g = read(files[-1])
    T = np.asarray(g.point_data["temperature"])
    err = np.max(np.abs(T - (300 + 100 * g.points[:, 0])))
    print(f"heat block: max |T - Texact| = {err:.2e} K  ({len(mesh.nodes)} nodes, order {mesh.order})  {case}")
    assert err < 1e-6
    return g


def test_heat_source_slab():
    """Slab 0<x<L with uniform Q, T=0 at both ends: Tmax = Q L^2 / (8 k)."""
    m = B.new_model("3D", ["ht"], "stationary")
    m.root.child("global").child("params").props["table"] = [
        {"name": "L", "expr": "0.1[m]", "descr": "slab"}, {"name": "Q", "expr": "1e6[W/m^3]", "descr": ""}]
    m.create("geom.block", m.component.child("geometry"), index=0, w="L", d="0.02", h="0.02")
    B.add_material(m, "Structural steel", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    B.add_feature(m, ph, "ht.source", [1], Q0="Q")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0) + geo.select("boundary", xmin=0.1), T0="0[K]")
    mesh, sif, rs, files, case = solve(m)
    g = read(files[-1])
    T = np.asarray(g.point_data["temperature"])
    exact = 1e6 * 0.1 ** 2 / (8 * 44.5)
    rel = abs(T.max() - exact) / exact
    print(f"heat source slab: Tmax {T.max():.4f} vs {exact:.4f} K, rel err {rel:.2e}  {case}")
    assert rel < 1e-3


def test_cantilever_tip():
    """Euler-Bernoulli tip deflection of an end-loaded cantilever: d = F L^3 / (3 E I)."""
    m = B.new_model("3D", ["solid"], "stationary")
    L, b, h = 1.0, 0.05, 0.1
    m.create("geom.block", m.component.child("geometry"), index=0, w=str(L), d=str(b), h=str(h))
    B.add_material(m, "Structural steel", all_domains=True)
    m.component.child("mesh").props["size"] = "4"
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")
    B.add_feature(m, ph, "solid.fixed", geo.select("boundary", xmax=0))
    F = 1000.0
    B.add_feature(m, ph, "solid.bload", geo.select("boundary", xmin=L), Fz=f"-{F / (b * h)}")
    mesh, sif, rs, files, case = solve(m)
    g = read(files[-1])
    uz = np.asarray(g.point_data["displacement"])[:, 2]
    I = b * h ** 3 / 12
    exact = -F * L ** 3 / (3 * 200e9 * I)
    tip = uz[np.isclose(g.points[:, 0], L)].mean()
    rel = abs(tip - exact) / abs(exact)
    print(f"cantilever: tip uz {tip:.4e} vs EB {exact:.4e} m, rel diff {rel:.2%} (shear+3D effects ~1-2%)  {case}")
    assert rel < 0.05
    assert "vonmises" in [k.lower() for k in g.point_data.keys()], list(g.point_data.keys())


def test_eigen_beam():
    m = B.new_model("3D", ["solid"], "eigen")
    L, b, h = 1.0, 0.05, 0.05
    m.create("geom.block", m.component.child("geometry"), index=0, w=str(L), d=str(b), h=str(h))
    B.add_material(m, "Structural steel", all_domains=True)
    m.component.child("mesh").props["size"] = "4"
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")
    B.add_feature(m, ph, "solid.fixed", geo.select("boundary", xmax=0))
    mesh, sif, rs, files, case = solve(m)
    I = b * h ** 3 / 12
    A = b * h
    f1 = (1.8751 ** 2) / (2 * math.pi) * math.sqrt(200e9 * I / (7850 * A * L ** 4))
    freqs = [math.sqrt(abs(l)) / (2 * math.pi) for l in rs.eigen]
    g = read(files[-1])
    print(f"eigen beam: f = {[round(f, 2) for f in freqs[:4]]} Hz, EB f1 = {f1:.2f} Hz; fields {list(g.point_data.keys())[:6]} files {len(files)}  {case}")
    assert freqs and abs(freqs[0] - f1) / f1 < 0.03


def test_capacitor_2d():
    """Parallel plates in 2D: uniform field V/d."""
    m = B.new_model("2D", ["es"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="0.1", h="0.01")
    B.add_material(m, "Air", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.es")
    B.add_feature(m, ph, "es.pot", geo.select("boundary", ymin=0.01), V0="100[V]")
    B.add_feature(m, ph, "es.ground", geo.select("boundary", ymax=0))
    mesh, sif, rs, files, case = solve(m)
    g = read(files[-1])
    V = np.asarray(g.point_data["potential"])
    err = np.max(np.abs(V - 100 * g.points[:, 1] / 0.01))
    print(f"capacitor 2D: max |V - Vexact| = {err:.2e} V; fields {list(g.point_data.keys())}  {case}")
    assert err < 1e-6


def test_poiseuille_2d():
    """Channel flow with parabolic inlet developing: centerline ~1.5*U for fully developed."""
    m = B.new_model("2D", ["spf"], "stationary")
    H, Lc = 0.01, 0.1
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(Lc), h=str(H))
    B.add_material(m, "Water, liquid", all_domains=True)
    mesh_node = m.component.child("mesh")       # user-controlled mesh with a custom Size node
    mesh_node.props["sequence"] = "user"
    m.create("mesh.size", mesh_node, mode="custom", hmax="H_el", hmin="1e-5", hcurve="0.3")
    m.create("mesh.ftri", mesh_node)
    m.global_defs.child("params").props["table"] = [{"name": "H_el", "expr": "0.5[mm]", "descr": "element size"}]
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.spf")
    U = 0.001   # Re_Dh = 20 -> entrance length ~0.02 m < channel length 0.1 m
    B.add_feature(m, ph, "spf.inlet", geo.select("boundary", xmax=0), U0=str(U))
    B.add_feature(m, ph, "spf.outlet", geo.select("boundary", xmin=Lc))
    mesh, sif, rs, files, case = solve(m)
    g = read(files[-1])
    vel = np.asarray(g.point_data["velocity"])
    P = g.points

    def prof(x):
        s = np.isclose(P[:, 0], x, atol=1e-9)
        o = np.argsort(P[s, 1])
        return P[s, 1][o], vel[s, 0][o]
    yi, ui = prof(0.0)
    Q = np.trapezoid(ui, yi)                      # discrete inflow (corner nodes are no-slip)
    yo, uo = prof(Lc)
    umax_exact = 1.5 * Q / H
    err = abs(uo.max() - umax_exact) / umax_exact
    print(f"poiseuille 2D: outlet centerline {uo.max():.6g} vs 1.5*Q/H {umax_exact:.6g} m/s, rel err {err:.2%} "
          f"(Q_out/Q_in = {np.trapezoid(uo, yo) / Q:.6f}); fields {list(g.point_data.keys())}  {case}")
    assert err < 0.01


def test_diffusion_1d_like():
    m = B.new_model("2D", ["tds"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1", h="0.1")
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.tds")
    B.add_feature(m, ph, "tds.conc", geo.select("boundary", xmax=0), c0="1")
    B.add_feature(m, ph, "tds.conc", geo.select("boundary", xmin=1), c0="0")
    mesh, sif, rs, files, case = solve(m)
    g = read(files[-1])
    c = np.asarray(g.point_data["concentration"])
    err = np.max(np.abs(c - (1 - g.points[:, 0])))
    print(f"diffusion: max |c - cexact| = {err:.2e}  {case}")
    assert err < 1e-6


def test_acoustic_duct():
    """1D standing wave in a closed-open duct driven by pressure p0 at x=0, p=0 at x=L: p = p0 sin(k(L-x))/sin(kL)."""
    m = B.new_model("2D", ["acpr"], "freq")
    L = 1.0
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h="0.05")
    B.add_material(m, "Air", all_domains=True)
    m.component.child("mesh").props["size"] = "3"
    st = m.studies()[0]
    B.study_steps(st)[0].props["freqs"] = "100"
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.acpr")
    B.add_feature(m, ph, "acpr.pressure", geo.select("boundary", xmax=0), p0="1")
    B.add_feature(m, ph, "acpr.soft", geo.select("boundary", xmin=L))
    mesh, sif, rs, files, case = solve(m)
    g = read(files[-1])
    keys = list(g.point_data.keys())
    p = np.asarray(g.point_data[[k for k in keys if "pressure" in k.lower()][0]])
    pre = p[:, 0] if p.ndim == 2 else p
    k = 2 * math.pi * 100 / 343.2
    exact = np.sin(k * (L - g.points[:, 0])) / np.sin(k * L)
    err = np.max(np.abs(pre - exact))
    print(f"acoustic duct: max |p - pexact| = {err:.2e} Pa; fields {keys}  {case}")
    assert err < 2e-2


def test_transient_heat():
    m = B.new_model("2D", ["ht"], "time")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="0.01", h="0.001")
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0="393.15[K]")
    st = m.studies()[0]
    B.study_steps(st)[0].props["times"] = "range(0,0.05,0.5)"
    mesh, sif, rs, files, case = solve(m)
    g = read(files[-1])
    print(f"transient heat: {len(files)} output files; T range {np.ptp(g.point_data['temperature']):.3f}  {case}")
    assert len(files) == 10


if __name__ == "__main__":
    print("Elmer:", INST.solver if INST else None)
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    only = sys.argv[1:]
    fails = 0
    for t in tests:
        if only and not any(o in t.__name__ for o in only):
            continue
        try:
            t()
        except Exception as exc:  # report and continue
            fails += 1
            print(f"FAIL {t.__name__}: {exc}")
    print("failures:", fails)
