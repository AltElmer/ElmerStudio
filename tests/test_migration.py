"""Migration of a charged species in the electrolyte potential (Transport of Diluted Species > Migration in electric
field, Nernst-Planck with the Nernst-Einstein mobility D/RT).

Along a strip with a linear potential phi = V y/L, fixed concentrations at both ends and no flow, the steady profile
solves D c'' + z D F/(RT) V/L c' = 0:  c = c0 + (c1 - c0) (exp(P y/L) - 1)/(exp(P) - 1),  P = -z F V/(R T).
Without migration it is linear (control: the migration term must be live).
"""
import math
import tempfile

import numpy as np

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner

F, R, T = 96485.33212, 8.314462618, 298.15


def strip(elmer, mig, z=1, V=0.05, L=1e-3, W=2e-4, c0=1.0, c1=2.0):
    m = B.new_model("2D", ["cd", "tds"], "stationary")
    g = m.component.child("geometry")
    m.create("geom.rectangle", g, index=0, w=str(W), h=str(L))
    geo = build_geometry(m, display=False)
    bottom, top = geo.select("boundary", ymax=0), geo.select("boundary", ymin=L)
    cd = m.component.child("physics.cd")
    cd.child("cd.elyte").props.update(sigma_src="user", sigma="1[S/m]")
    B.add_feature(m, cd, "cd.pot", bottom, phil0="0[V]")
    B.add_feature(m, cd, "cd.pot", top, phil0=f"{V}[V]")
    tds = m.component.child("physics.tds")
    tds.child("tds.tp").props.update(D="1e-9[m^2/s]", mig=mig, z=str(z), Tm=f"{T}[K]")
    B.add_feature(m, tds, "tds.conc", bottom, c0=f"{c0}[mol/m^3]")
    B.add_feature(m, tds, "tds.conc", top, c0=f"{c1}[mol/m^3]")
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    for b, n in ((bottom, 4), (top, 4), (geo.select("boundary", xmax=0), 60), (geo.select("boundary", xmin=W), 60)):
        B.add_feature(m, mesh, "mesh.distribution", b, n=n)
    st = m.studies()[0]
    next(c for c in st.children if c.kind == "study.stationary").props.update(coupling_tol="1e-12")
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_mig_"), elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    y = sol.topo.nodes[:, 1] * sol.scale
    c = sol.evaluate("c", sol.frame_index("last"))
    P = -z * F * V / (R * T)
    exact = c0 + (c1 - c0) * (np.exp(P * y / L) - 1) / (math.exp(P) - 1)
    linear = c0 + (c1 - c0) * y / L
    return c, exact, linear


def test_migration_gives_the_exact_exponential_profile(elmer):
    c, exact, linear = strip(elmer, True)
    assert np.max(np.abs(c - exact)) < 1e-4, np.max(np.abs(c - exact))      # quadratic elements, 60 along
    assert np.max(np.abs(exact - linear)) > 0.2                               # the effect is large
    c2, exact2, _ = strip(elmer, True, z=-2)                                   # an anion, twice the charge
    assert np.max(np.abs(c2 - exact2)) < 1e-3, np.max(np.abs(c2 - exact2))


def test_without_migration_the_profile_is_linear(elmer):
    c, exact, linear = strip(elmer, False)
    assert np.max(np.abs(c - linear)) < 1e-9
