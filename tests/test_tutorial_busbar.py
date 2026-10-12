"""COMSOL 'Introduction to COMSOL Multiphysics', Example 2: the Busbar (Joule heating), built step by step.

Geometry follows Appendix A; physics, materials, mesh and study follow the tutorial chapter.
Run as a script for a report, or via pytest (needs Elmer for the solve checks).
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core.geometry import build_geometry  # noqa: E402

PARAMS = [("L", "9[cm]", "Length of lower part of the busbar"), ("rad_1", "6[mm]", "Radius of the titanium bolts"),
          ("tbb", "5[mm]", "Thickness of the busbar"), ("wbb", "5[cm]", "Width of the busbar"),
          ("mh", "3[mm]", "Maximum element size"), ("htc", "5[W/(m^2*K)]", "Heat transfer coefficient"),
          ("Vtot", "20[mV]", "Applied electric potential")]


def build_model(sweep=False):
    m = B.new_model("3D", ["@jh"], "stationary")
    m.root.label = "busbar_tutorial"
    m.global_defs.child("params").props["table"] = [{"name": n, "expr": e, "descr": d} for n, e, d in PARAMS]
    g = m.component.child("geometry")
    mk = lambda kind, parent, **p: m.create(kind, parent, index=len([c for c in parent.children if c.kind != "geom.finalize"]), **p)  # noqa: E731
    # Work Plane 1 (xz-plane): L profile, two fillets, extrude wbb
    wp1 = mk("geom.workplane", g, ptype="quick", plane="xz")
    r1 = mk("geom.rectangle", wp1, w="L+2*tbb", h="0.1")
    r2 = mk("geom.rectangle", wp1, w="L+tbb", h="0.1-tbb", x="0", y="tbb")
    mk("geom.difference", wp1, input=[r1.tag], tools=[r2.tag])
    mk("geom.fillet2d", wp1, points="3", r="tbb")
    mk("geom.fillet2d", wp1, points="6", r="2*tbb")
    mk("geom.extrude", g, input=[wp1.tag], dist="wbb")
    # Work Plane 2 (face parallel, face 8): bolt 1
    wp2 = mk("geom.workplane", g, ptype="face", face=8)
    mk("geom.circle", wp2, r="rad_1")
    mk("geom.extrude", g, input=[wp2.tag], dist="-2*tbb")
    # Work Plane 3 (face parallel, face 4): bolts 2a, 2b
    wp3 = mk("geom.workplane", g, ptype="face", face=4)
    c1 = mk("geom.circle", wp3, r="rad_1", x="-L/2+1.5[cm]", y="-wbb/4")
    mk("geom.copy", wp3, input=[c1.tag], dx="0", dy="wbb/2")
    mk("geom.extrude", g, input=[wp3.tag], dist="-2*tbb")
    return m


def report_geometry(m):
    geo = build_geometry(m, display=False)
    print(f"domains {geo.count('domain')}, boundaries {geo.count('boundary')}, edges {geo.count('edge')}")
    for d in geo.numbers["domain"]:
        print(f"  domain {d}: bbox [mm] {np.round(np.array(geo.bbox_of['domain'][d]) * 1000, 2).tolist()}")
    return geo


def bolt_contact_faces(geo):
    """The exposed circular end faces of the three bolts (what the tutorial calls boundaries 8, 15, 43).

    Returns (lower_a, lower_b, upper): flat faces of a bolt domain (2..7) adjacent to that domain only, at the
    extreme end of the protrusion away from the copper busbar (domain with the largest volume)."""
    copper = max(geo.numbers["domain"], key=lambda d: np.prod(np.ptp(np.array(geo.bbox_of["domain"][d]).reshape(2, 3), axis=0)))
    bolts = [d for d in geo.numbers["domain"] if d != copper]
    ends = []
    for b in geo.numbers["boundary"]:
        ups = geo.adj_up["boundary"][b]
        if len(ups) != 1 or ups[0] == copper:
            continue
        bb = np.array(geo.bbox_of["boundary"][b]).reshape(2, 3)
        flat = np.ptp(bb, axis=0)
        if (flat < 1e-9).sum() != 1:
            continue
        # a bolt end disk: protruding beyond the copper bounding box
        cb = np.array(geo.bbox_of["domain"][copper]).reshape(2, 3)
        axis = int(np.argmin(flat))
        val = bb[0, axis]
        if val < cb[0, axis] - 1e-9 or val > cb[1, axis] + 1e-9:
            ends.append((b, axis, ups[0]))
    lower = [b for b, ax, _ in ends if ax == 2]
    upper = [b for b, ax, _ in ends if ax != 2]
    assert len(lower) == 2 and len(upper) == 1, (ends, bolts)
    return lower[0], lower[1], upper[0]


def setup_physics(m, geo, bolt_faces):
    """Materials, Joule heating BCs, user mesh as in the tutorial. bolt_faces = (ground a, ground b, potential)."""
    B.add_material(m, "Copper", all_domains=True)
    copper = max(geo.numbers["domain"], key=lambda d: np.prod(np.ptp(np.array(geo.bbox_of["domain"][d]).reshape(2, 3), axis=0)))
    B.add_material(m, "Titanium beta-21S", [d for d in geo.numbers["domain"] if d != copper])
    g_a, g_b, pot = bolt_faces
    ec = m.component.child("physics.ec")
    B.add_feature(m, ec, "ec.pot", [pot], V0="Vtot")
    B.add_feature(m, ec, "ec.ground", [g_a, g_b])
    ht = m.component.child("physics.ht")
    ext = [b for b in geo.exterior_boundaries() if b not in bolt_faces]
    B.add_feature(m, ht, "ht.heatflux", ext, type="convective", h="htc", Text="293.15[K]")
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.size", mesh, mode="custom", hmax="mh", hmin="mh-mh/3", hcurve="0.2")
    m.create("mesh.ftet", mesh)


def solve(m, sweep_wbb=False):
    from elmerstudio.core.elmer import find_elmer
    from elmerstudio.core.results import Solution
    from elmerstudio.core.study_runner import StudyRunner
    st = m.studies()[0]
    if sweep_wbb:
        m.create("study.sweep", st, index=0, pname="wbb", pvalues="range(5,1,10)", punit="cm")
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_busbar_"), find_elmer()).run()
    return [Solution(f) for f in folders]


def run(sweep=False):
    m = build_model()
    geo = report_geometry(m)
    faces = bolt_contact_faces(geo)
    print("bolt contact faces (ground, ground, potential):", faces)
    setup_physics(m, geo, faces)
    sols = solve(m, sweep)
    rows = []
    for s in sols:
        i = s.frame_index("last")
        T = s.evaluate("T", i)
        Tavg = s.integrate("T", i, "domain", None, average=True)
        rows.append((s.meta.get("sweep_value"), float(T.max()), Tavg, float(T.min())))
        print(f"wbb={s.meta.get('sweep_value')}  Tmax={T.max():.2f} K  Tavg={Tavg:.2f} K  nodes={len(s.topo.nodes)}")
    return geo, rows


def test_busbar_tutorial(elmer):
    """Checks only what the tutorial text states (it prints no computed temperature values)."""
    geo, rows = run(sweep=True)
    assert geo.count("domain") == 7                       # "The domains 2, 3, 4, 5, 6, and 7 are highlighted"
    wbb, tmax, tavg, tmin = rows[0]
    assert tmax > 300                                     # "substantially higher than the ambient temperature of 293 K"
    assert tmax - tmin < 10                              # "temperature difference in the device is less than 10 K"
    tavgs = [r[2] for r in rows]
    assert all(a > b for a, b in zip(tavgs, tavgs[1:])), tavgs  # "the average temperature decreases as the width increases"


if __name__ == "__main__":
    run(sweep="--sweep" in sys.argv)
