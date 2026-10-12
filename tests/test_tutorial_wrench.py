"""COMSOL 'Introduction to COMSOL Multiphysics', Example 1 (the wrench), reproduced with Elmer Studio.

COMSOL imports a fixed binary geometry (wrench.mphbin), which is not available; this uses the parametric
hex-socket wrench from the elmer-multiphysics-tutorials project, built with work plane + booleans + extrude.
Tutorial workflow kept: Structural steel, Fixed Constraint (bolt/socket), Boundary Load as *Total force*
F = 150 N, von Mises plot, safety plot solid.mises > 200[MPa], Volume Maximum, and the mesh-convergence
Parametric Sweep on hd with Size: hmax = h0/hd, hmin = h0/(4*hd), curvature factor 0.1.
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

PARAMS = [("F", "150[N]", "Applied load"), ("hd", "1", "Element size divider"), ("h0", "1[cm]", "Starting element size"),
          ("R", "18[mm]", "Head outer radius"), ("AF", "17[mm]", "Hex across flats"), ("Rc", "AF/sqrt(3)", "Hex circumradius"),
          ("Lh", "150[mm]", "Handle length"), ("W", "12[mm]", "Handle width"), ("T", "6[mm]", "Wrench thickness")]


def build_model(convergence=False):
    m = B.new_model("3D", ["solid"], "stationary")
    m.root.label = "wrench_tutorial"
    m.global_defs.child("params").props["table"] = [{"name": n, "expr": e, "descr": d} for n, e, d in PARAMS]
    g = m.component.child("geometry")
    wp = m.create("geom.workplane", g, index=0, ptype="quick", plane="xy")
    head = m.create("geom.circle", wp, r="R")
    handle = m.create("geom.rectangle", wp, w="Lh", h="W", y="-W/2")
    uni = m.create("geom.union", wp, input=[head.tag, handle.tag], interior=False)
    xs = " ".join(f"Rc*cos({i}*pi/3)" for i in range(6))
    ys = " ".join(f"Rc*sin({i}*pi/3)" for i in range(6))
    hexa = m.create("geom.polygon", wp, xs=xs, ys=ys)
    m.create("geom.difference", wp, input=[uni.tag], tools=[hexa.tag])
    m.create("geom.extrude", g, index=1, input=[wp.tag], dist="T")
    B.add_material(m, "Structural steel", all_domains=True)
    geo = build_geometry(m, display=False)
    solid = m.component.child("physics.solid")
    socket = [b for b in geo.numbers["boundary"]
              if np.linalg.norm(np.array(geo.bbox_of["boundary"][b])[[0, 1, 3, 4]]) < 0.02 and
              geo.bbox_of["boundary"][b][5] - geo.bbox_of["boundary"][b][2] > 0.005]   # the six hex flats
    end = geo.select("boundary", xmin=0.15 - 1e-9)                                     # handle end face
    B.add_feature(m, solid, "solid.fixed", socket)
    B.add_feature(m, solid, "solid.bload", end, type="total", Ftz="-F")
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.size", mesh, mode="custom", hmax="h0/hd", hmin="h0/(4*hd)", hcurve="0.1", hnarrow="0.2")
    m.create("mesh.ftet", mesh)
    st = m.studies()[0]
    if convergence:
        m.create("study.sweep", st, index=0, pname="hd", pvalues="range(1,1,4)", punit="")
    defs = m.component.child("definitions")
    op = m.create("cpl.maximum", defs, opname="maxop1")
    return m, geo, socket, end


def run(convergence=False):
    from elmerstudio.core.elmer import find_elmer
    from elmerstudio.core.results import Solution
    from elmerstudio.core.nodes import coupling_operators
    from elmerstudio.core.study_runner import StudyRunner
    m, geo, socket, end = build_model(convergence)
    print(f"geometry: {geo.count('domain')} domain, {geo.count('boundary')} boundaries; socket faces {socket}, end {end}")
    folders = StudyRunner(m, m.studies()[0], tempfile.mkdtemp(prefix="es_wrench_"), find_elmer()).run()
    rows = []
    for f in folders:
        s = Solution(f)
        s.operators = coupling_operators(m)
        i = s.frame_index("last")
        mm = s.maxmin_locations("solid.mises", i)
        tip = float(np.nanmin(s.evaluate("w", i)))
        unsafe = s.integrate("solid.mises > 200[MPa]", i, "domain", None) / s.integrate("1", i, "domain", None)
        reac = s.integrate("1", i, "boundary", end)
        rows.append((s.meta.get("sweep_value"), mm["max"], mm["xmax"], tip, unsafe, len(s.topo.nodes),
                     float(np.nanmean(s.evaluate("maxop1(solid.mises)", i)))))
        print(f"hd={s.meta.get('sweep_value')}  nodes={len(s.topo.nodes):6d}  max mises={mm['max'] / 1e6:7.1f} MPa at "
              f"{np.round(np.array(mm['xmax']) * 1000, 1).tolist()} mm  tip w={tip * 1000:.3f} mm  "
              f"volume fraction > 200 MPa = {unsafe:.4f}  end area={reac * 1e6:.1f} mm2")
    return rows


def test_wrench_tutorial(elmer):
    rows = run(convergence=True)
    E, W, T = 200e9, 0.012, 0.006
    L = 0.150 - 0.017 / 3 ** 0.5        # cantilever measured from the socket corners
    I = W * T ** 3 / 12
    nodes = [r[5] for r in rows]
    assert nodes == sorted(nodes) and len(set(nodes)) == len(nodes), nodes   # each hd refines the mesh
    for _hd, mises, xmax, tip, unsafe, _n, maxop in rows:
        assert maxop == mises                                          # maxop1(...) equals Volume Maximum
        assert 0.85 < abs(tip) / (150 * L ** 3 / (3 * E * I)) < 1.05    # head is stiffer than a slender beam
        assert 0 < unsafe < 0.2                                         # a small region exceeds 200 MPa (safety plot)
    # mesh convergence of the tip deflection (the peak stress sits in a sharp re-entrant corner of this stand-in
    # geometry and is singular; COMSOL's wrench has fillets there, which is why its maximum converges)
    t = [abs(r[3]) for r in rows]
    assert all(b >= a for a, b in zip(t, t[1:])) and (t[-1] - t[-2]) < (t[1] - t[0]), t


if __name__ == "__main__":
    run(convergence="--sweep" in sys.argv)
