"""The Plate interface (Elmer SmitcSolver, Reissner-Mindlin), solved with Elmer and compared with thin-plate theory.

D = E h^3 / (12 (1 - nu^2)) is the bending stiffness.  References (Timoshenko and Woinowsky-Krieger, Theory of
Plates and Shells; Leissa, Vibration of Plates):

* square plate of side a under a uniform load q, centre deflection
      clamped edges           w = 0.001265 q a^4 / D
      simply supported edges  w = 0.004062 q a^4 / D
  so the two edge conditions differ by a factor 3.2, and w goes as h^-3;
* strip of length L clamped at x = 0 with nu = 0 (a beam, D = E h^3 / 12):
      uniform load q          w(L) = q L^4 / (8 D),  slope at the tip  q L^3 / (6 D)
      tip deflection w0       w(L/2) = 5 w0 / 16
      tip rotation theta0     w(L) = theta0 L / 2          (a pure end moment: w = theta0 x^2 / (2 L))
* eigenfrequencies of the simply supported square plate
      f_mn = (pi / 2) sqrt(D / (rho h)) (m^2 + n^2) / a^2,
  which go as h, and of the clamped one  f_1 = 35.99 / (2 pi a^2) sqrt(D / (rho h)).

The plates here have a/h = 50..100, where the shear deformation that SmitcSolver includes changes these numbers by
less than the tolerances.
"""
import math
import os
import re
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.model import node_type
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.physics import PHYSICS, WIZARD_TREE
from elmerstudio.core.results import Solution
from elmerstudio.core.sif import build_sif
from elmerstudio.core.study_runner import StudyRunner

A, H, E, NU, RHO, Q = 1.0, 0.01, 70e9, 0.3, 2700.0, 1000.0
LS, BS = 1.0, 0.2                                    # the strip


def stiffness(h=H, nu=NU):
    return E * h ** 3 / (12 * (1 - nu ** 2))


def model(study="stationary", w=A, b=A, h=H, nu=NU, n=(20, 20)):
    m = B.new_model("2D", ["plate"], study)
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(w), h=str(b))
    geo = build_geometry(m, display=False)
    pl = m.component.child("physics.plate")
    pl.child("plate.lemm").props.update(h=f"{h!r}[m]", E_src="user", E=f"{E!r}[Pa]", nu_src="user", nu=str(nu),
                                        rho_src="user", rho=f"{RHO!r}[kg/m^3]")
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    ends = geo.select("boundary", xmax=0) + geo.select("boundary", xmin=w)
    sides = geo.select("boundary", ymax=0) + geo.select("boundary", ymin=b)
    B.add_feature(m, mesh, "mesh.distribution", sides, n=n[0])
    B.add_feature(m, mesh, "mesh.distribution", ends, n=n[1])
    return m, geo, pl


def square(edge, study="stationary", h=H, q=Q):
    m, geo, pl = model(study, h=h)
    B.add_feature(m, pl, edge, geo.select("boundary"))
    load = B.add_feature(m, pl, "plate.load", geo.select("domain"), q=str(q)) if study == "stationary" else None
    return m, geo, load


def strip():
    m, geo, pl = model(w=LS, b=BS, nu=0.0, n=(40, 8))
    B.add_feature(m, pl, "plate.clamped", geo.select("boundary", xmax=0))
    return m, geo, pl, geo.select("boundary", xmin=LS)


def solve(m, geo, elmer):
    folders = StudyRunner(m, m.studies()[0], tempfile.mkdtemp(prefix="es_plate_"), elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    return sol


def at(sol, expr, x, y, i=None):
    """The value at the mesh node at (x, y)."""
    i = sol.frame_index("last") if i is None else i
    xy = sol.topo.nodes[:, :2] * sol.scale
    k = int(np.argmin(np.hypot(xy[:, 0] - x, xy[:, 1] - y)))
    assert math.hypot(xy[k, 0] - x, xy[k, 1] - y) < 1e-9
    return float(sol.evaluate(expr, i)[k])


def sif_text(m, geo):
    st = m.studies()[0]
    return build_sif(m, st, B.study_steps(st)[0], geo, 100).text


def sections(text):
    """[(section name, {keyword: value})] of a .sif text."""
    out = []
    for name, body in re.findall(r"^([A-Z][A-Za-z ]*?)(?: \d+)?\n(.*?)^End$", text, flags=re.M | re.S):
        kw = {}
        for line in body.splitlines():
            if "=" in line and not line.lstrip().startswith("!"):
                key, val = line.split("=", 1)
                kw[re.sub(r"\(\d+\)$", "", key.strip())] = val.strip()
        out.append((name, kw))
    return out


# --------------------------------------------------------------------------------------------- without a solver
def test_the_interface_is_offered_in_2d_with_linear_elements_only():
    pd = PHYSICS["plate"]
    assert pd.dims == ("2D",) and pd.studies == ("stationary", "eigen")
    assert "plate" in [p for cat, groups in WIZARD_TREE if cat == "Structural Mechanics" for _g, ps in groups for p in ps]
    disc = next(p for p in node_type("physics.plate").props if p.key == "disc")
    assert disc.default == "1" and [c[0] for c in disc.choices] == ["1"]
    # the other interfaces keep both orders
    assert [c[0] for c in next(p for p in node_type("physics.solid").props if p.key == "disc").choices] == ["1", "2"]


def test_shell_and_beam_interfaces_that_never_ran_are_gone():
    assert "shell" not in PHYSICS and "beam" not in PHYSICS


def test_every_setting_reaches_the_sif():
    m, geo, pl = model()
    left, right = geo.select("boundary", xmax=0), geo.select("boundary", xmin=A)
    bottom, top = geo.select("boundary", ymax=0), geo.select("boundary", ymin=A)
    B.add_feature(m, pl, "plate.clamped", left)
    B.add_feature(m, pl, "plate.simple", right)
    B.add_feature(m, pl, "plate.disp", bottom, w0="1[mm]", onthx=True, thx0="0.002", onthy=True, thy0="0.003")
    B.add_feature(m, pl, "plate.disp", top, w0="2[mm]")
    B.add_feature(m, pl, "plate.load", geo.select("domain"), q="1[kPa]")
    secs = sections(sif_text(m, geo))
    solver = next(kw for name, kw in secs if name == "Solver" and "Smitc" in kw.get("Procedure", ""))
    assert solver["Procedure"] == '"Smitc" "SmitcSolver"' and solver["Variable"] == "-dofs 3 Deflection"
    mat = next(kw for name, kw in secs if name == "Material")
    assert {k: float(v.split()[-1]) for k, v in mat.items()} == {
        "Youngs Modulus": E, "Poisson Ratio": NU, "Density": RHO, "Thickness": H}
    assert [float(kw["Pressure"].split()[-1]) for name, kw in secs if name == "Body Force"] == [1000.0]
    bcs = {int(kw["Target Boundaries"].split()[-1]): {k: float(v.split()[-1]) for k, v in kw.items()
                                                      if k.startswith("Deflection")}
           for name, kw in secs if name == "Boundary Condition"}
    assert bcs[left[0]] == {"Deflection 1": 0.0, "Deflection 2": 0.0, "Deflection 3": 0.0}
    assert bcs[right[0]] == {"Deflection 1": 0.0}
    assert bcs[bottom[0]] == {"Deflection 1": 1e-3, "Deflection 2": 0.002, "Deflection 3": 0.003}
    assert bcs[top[0]] == {"Deflection 1": 2e-3}                 # rotations are not prescribed unless ticked


def test_a_quadratic_mesh_is_refused():
    """SmitcSolver has elements for 3-node triangles and 4-node quadrilaterals only."""
    m, geo, _load = square("plate.clamped")
    sif_text(m, geo)                                             # the automatic order is linear
    m.component.child("mesh").props["order"] = "2"
    with pytest.raises(ValueError, match="linear"):
        sif_text(m, geo)


def test_the_keywords_are_ones_elmer_knows(elmer):
    """Elmer runs with Check Keywords "Warn", so a keyword it does not know would be dropped without a word."""
    from elmerstudio.core.elmer import module_dir
    cands = [os.path.join(d, "SOLVER.KEYWORDS") for d in (module_dir(elmer), os.path.join(
        elmer.home, "share", "elmersolver", "lib")) if d]
    path = next((c for c in cands if os.path.isfile(c)), None)
    if path is None:
        pytest.skip("SOLVER.KEYWORDS not found in this Elmer installation")
    known = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            mt = re.match(r"\s*(\w+):\w+:\s*'([^']+)'", line)
            if mt:
                known.add((mt.group(1).lower(), mt.group(2).lower()))
    m, geo, pl = model("eigen")
    B.add_feature(m, pl, "plate.clamped", geo.select("boundary", xmax=0))
    B.add_feature(m, pl, "plate.disp", geo.select("boundary", xmin=A), onthx=True, onthy=True)
    B.add_feature(m, pl, "plate.load", geo.select("domain"), q="1")
    names = {"Material": "material", "Body Force": "bodyforce", "Boundary Condition": "bc"}
    checked = 0
    for name, kw in sections(sif_text(m, geo)):
        if name == "Solver" and "Smitc" in kw.get("Procedure", ""):
            keys, sec = [k for k in kw if k.startswith("Eigen")], "solver"
        elif name in names:
            keys, sec = [k for k in kw if k not in ("Target Boundaries", "Name")], names[name]
        else:
            continue
        for k in keys:
            assert (sec, k.lower()) in known, f"{name}: Elmer does not know '{k}'"
            checked += 1
    assert checked >= 10


# ------------------------------------------------------------------------------------------------- with Elmer
def test_clamped_square_plate_and_thickness_cubed(elmer):
    m, geo, load = square("plate.clamped")
    sol = solve(m, geo, elmer)
    i = sol.frame_index("last")
    assert {"plate.w", "plate.thx", "plate.thy"} <= {a[0] for a in sol.available(i)}
    w1 = at(sol, "plate.w", A / 2, A / 2)
    assert w1 == pytest.approx(0.001265 * Q * A ** 4 / stiffness(), rel=0.015)       # 1.973e-4 m
    assert w1 == pytest.approx(float(np.max(sol.evaluate("plate.w", i))))             # the centre is the maximum
    # the edge is held and does not rotate
    for name in ("plate.w", "plate.thx", "plate.thy"):
        assert at(sol, name, 0.0, A / 2) == 0.0
    # twice the thickness: an eighth of the deflection
    m2, geo2, _l = square("plate.clamped", h=2 * H)
    assert w1 / at(solve(m2, geo2, elmer), "plate.w", A / 2, A / 2) == pytest.approx(8.0, rel=0.01)
    # without the load the plate stays flat
    load.enabled = False
    assert "Pressure" not in sif_text(m, geo)
    sol0 = solve(m, geo, elmer)
    assert float(np.max(np.abs(sol0.evaluate("plate.w", sol0.frame_index("last"))))) == 0.0


def test_simply_supported_square_plate(elmer):
    m, geo, _load = square("plate.simple")
    sol = solve(m, geo, elmer)
    w = at(sol, "plate.w", A / 2, A / 2)
    ref = 0.004062 * Q * A ** 4 / stiffness()                                        # 6.337e-4 m
    assert w == pytest.approx(ref, rel=0.015)
    assert abs(w - 0.001265 * Q * A ** 4 / stiffness()) > 0.5 * ref                  # not the clamped plate
    assert at(sol, "plate.w", 0.0, A / 2) == 0.0
    assert abs(at(sol, "plate.thx", 0.0, A / 2)) > 1e-3                               # the edge rotates


def test_load_that_depends_on_the_deflection(elmer):
    """A simply supported plate on an elastic foundation, entered as the load q - k*wp.  Navier's series:
    w(a/2, a/2) = sum over odd m, n of  16 q / (pi^2 m n) (-1)^((m+n)/2-1) / (D pi^4 (m^2+n^2)^2 / a^4 + k).
    With k = 1e6 N/m^3 the foundation carries part of the load: 4.48e-4 m against 6.34e-4 m without it."""
    def navier(k):
        return sum(16 * Q / (math.pi ** 2 * mm * nn) * (-1) ** ((mm + nn) // 2 - 1)
                   / (stiffness() * math.pi ** 4 * (mm * mm + nn * nn) ** 2 / A ** 4 + k)
                   for mm in range(1, 60, 2) for nn in range(1, 60, 2))

    k = 1e6
    assert navier(0.0) == pytest.approx(0.004062 * Q * A ** 4 / stiffness(), rel=1e-3)     # the series itself
    ref = navier(k)
    assert ref == pytest.approx(4.484e-4, rel=1e-3)
    m, geo, pl = model()
    B.add_feature(m, pl, "plate.simple", geo.select("boundary"))
    B.add_feature(m, pl, "plate.load", geo.select("domain"), q=f"{Q}[Pa] - {k}[N/m^3]*wp")
    text = sif_text(m, geo)
    assert "Pressure = Variable Deflection 1" in text
    assert int(re.search(r"Nonlinear System Max Iterations = Integer (\d+)", text).group(1)) > 1
    assert at(solve(m, geo, elmer), "plate.w", A / 2, A / 2) == pytest.approx(ref, rel=0.015)


def test_cantilever_strip_under_uniform_load(elmer):
    m, geo, pl, _tip = strip()
    B.add_feature(m, pl, "plate.load", geo.select("domain"), q=str(Q))
    sol = solve(m, geo, elmer)
    D = stiffness(nu=0.0)
    assert at(sol, "plate.w", LS, BS / 2) == pytest.approx(Q * LS ** 4 / (8 * D), rel=0.01)        # 21.43 mm
    assert at(sol, "plate.thx", LS, BS / 2) == pytest.approx(Q * LS ** 3 / (6 * D), rel=0.01)      # dw/dx there
    assert abs(at(sol, "plate.thy", LS, BS / 2)) < 1e-6 * Q * LS ** 3 / (6 * D)


def test_prescribed_deflection_and_rotation(elmer):
    w0, th0 = 2e-3, 5e-3
    m, geo, pl, tip = strip()
    d = B.add_feature(m, pl, "plate.disp", tip, w0=f"{w0}[m]")
    sol = solve(m, geo, elmer)
    assert at(sol, "plate.w", LS, BS / 2) == pytest.approx(w0, rel=1e-9)
    assert at(sol, "plate.w", LS / 2, BS / 2) == pytest.approx(5 * w0 / 16, rel=0.01)
    # a rotation alone: the tip is free to move and follows the end moment
    d.props.update(onw=False, onthx=True, thx0=str(th0))
    sol = solve(m, geo, elmer)
    assert at(sol, "plate.thx", LS, BS / 2) == pytest.approx(th0, rel=1e-9)
    assert at(sol, "plate.w", LS, BS / 2) == pytest.approx(th0 * LS / 2, rel=0.01)
    assert at(sol, "plate.w", LS / 2, BS / 2) == pytest.approx(th0 * LS / 8, rel=0.01)
    # unticked, the value is not applied
    d.props.update(onthx=False)
    sol = solve(m, geo, elmer)
    assert float(np.max(np.abs(sol.evaluate("plate.w", sol.frame_index("last"))))) == 0.0


def test_eigenfrequencies(elmer):
    def freqs(edge, h):
        m, geo, _l = square(edge, "eigen", h=h)
        sol = solve(m, geo, elmer)
        return sol, [fr.value for fr in sol.frames]

    f0 = math.sqrt(stiffness() / (RHO * H)) / A ** 2
    sol, f = freqs("plate.simple", H)
    ref = [math.pi / 2 * f0 * k for k in (2, 5, 5, 8)]                               # 48.41, 121.0, 121.0, 193.6 Hz
    assert f[:4] == pytest.approx(ref, rel=0.01)
    # the first mode has no nodal line and its largest deflection at the centre
    w = sol.evaluate("plate.w", 0)
    assert abs(at(sol, "plate.w", A / 2, A / 2, 0)) == pytest.approx(float(np.max(np.abs(w))))
    assert np.all(w * np.sign(at(sol, "plate.w", A / 2, A / 2, 0)) >= 0)
    # twice the thickness: twice the frequency
    assert freqs("plate.simple", 2 * H)[1][0] / f[0] == pytest.approx(2.0, rel=0.01)
    # clamped edges stiffen the plate
    assert freqs("plate.clamped", H)[1][0] == pytest.approx(35.99 / (2 * math.pi) * f0, rel=0.015)     # 88.25 Hz
