"""Time Dependent output times with growing steps (several ranges, as in COMSOL's list of times).

A sine mode decaying by diffusion, c = exp(-t/tau) sin(pi x/L) with tau = L^2/(D pi^2) = 1 s, marched with 0.01 s
steps to 0.1 s and 0.1 s steps to 1 s: the solution is saved at every output time and decays at the exact rate.
"""
import math
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.results import Solution
from elmerstudio.core.sif import SifError, build_sif, time_groups
from elmerstudio.core.study_runner import StudyRunner

TIMES = "range(0,0.01,0.1) range(0.2,0.1,1)"


def test_time_groups():
    assert time_groups([0, 0.01, 0.02, 0.03, 0.13, 0.23]) == pytest.approx([(0.01, 3), (0.1, 2)])
    g = time_groups([0, 0.5, 1.0, 1.0, 2.0, 3.0])                 # a time repeated where two ranges meet
    assert [n for _dt, n in g] == [2, 2] and g[1][0] == pytest.approx(1.0)
    with pytest.raises(SifError):
        time_groups([0, 1, 0.5])


def model(times=TIMES):
    L, D = 1e-3, 1e-6 / math.pi ** 2                                # tau = L^2/(D pi^2) = 1 s
    m = B.new_model("2D", ["tds"], "time")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(L / 10))
    geo = build_geometry(m, display=False)
    tds = m.component.child("physics.tds")
    tds.child("tds.tp").props["D"] = f"{D!r}[m^2/s]"
    tds.child("tds.init").props["c0"] = f"sin(pi*x/{L!r}[m])*1[mol/m^3]"
    B.add_feature(m, tds, "tds.conc", geo.select("boundary", xmax=0) + geo.select("boundary", xmin=L), c0="0")
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    for b, n in ((geo.select("boundary", ymax=0), 40), (geo.select("boundary", ymin=L / 10), 40),
                 (geo.select("boundary", xmax=0), 2), (geo.select("boundary", xmin=L), 2)):
        B.add_feature(m, mesh, "mesh.distribution", b, n=n)
    st = m.studies()[0]
    B.study_steps(st)[0].props.update(times=times, bdf="2")
    return m, geo, st, L


def test_growing_steps_in_the_sif():
    m, geo, st, _ = model()
    res = build_sif(m, st, B.study_steps(st)[0], geo, 100)
    assert "Timestep Intervals(2) = 10 9" in res.text and "Timestep Sizes(2) = 0.01 0.1" in res.text
    assert res.meta["times"] == pytest.approx([0.01 * i for i in range(1, 11)] + [0.1 * i for i in range(2, 11)])


def test_growing_steps_decay_at_the_exact_rate(elmer):
    # BDF2's own error is ~(2/9)(dt/tau)^2 per tau: 0.2 % at 0.1 s steps, 2e-5 at 0.01 s
    m, geo, st, L = model("range(0,0.001,0.01) range(0.02,0.01,1)")
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_dt_"), elmer).run()
    sol = Solution(folders[-1])
    bind_results(m, sol, geo)
    times = [fr.value for fr in sol.frames]
    assert times[0] == pytest.approx(0.001) and times[9] == pytest.approx(0.01) and times[-1] == pytest.approx(1.0)
    assert len(times) == 109
    x = sol.topo.nodes[:, 0] * sol.scale
    mid = np.argmin(np.abs(x - L / 2) + np.abs(sol.topo.nodes[:, 1]))
    for i in (9, 10, 108):                                          # the end of the fine steps, one step on, 1 s
        c = sol.evaluate("c", i)
        assert c[mid] == pytest.approx(math.exp(-times[i]), rel=2e-4), (times[i], c[mid])
