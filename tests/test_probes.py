"""Probes: a quantity at every output time (or parameter value), in the study's Probe Table and Probe Plot.

The decaying sine of test_time_steps, c = exp(-t) sin(pi x/L) on [0, L] x [0, L/10]: its domain average is
(2/pi) exp(-t), its maximum and its value at (L/2, y) exp(-t), its integral (2/pi) exp(-t) L^2/10.
"""
import math
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core import probes as P
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.project import load, save
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner
from test_time_steps import model as decay_model


def with_probes(m, geo, L):
    defs = m.component.child("definitions")
    avg = m.create("probe.domain", defs, expr="c", ptype="avg")
    mx = m.create("probe.domain", defs, expr="c", ptype="max")
    pt = m.create("probe.point", defs, expr="c", x=f"{L / 2!r}", y=f"{L / 20!r}")
    integ = m.create("probe.domain", defs, expr="c", ptype="int")
    edge = B.add_feature(m, defs, "probe.boundary", geo.select("boundary", xmin=L), expr="c", ptype="avg")  # c = 0
    bad = m.create("probe.global", defs, expr="no_such_variable")
    off = m.create("probe.domain", defs, expr="c")
    off.enabled = False
    return avg, mx, pt, integ, edge, bad, off


def test_probe_table_follows_the_exact_decay(elmer):
    m, geo, st, L = decay_model("range(0,0.05,0.5)")
    avg, mx, pt, integ, edge, bad, off = with_probes(m, geo, L)
    runner = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_probe_"), elmer)
    lives = []
    runner.on_event = lambda kind, p: lives.append(p) if kind == "live" else None
    folders = runner.run()
    sol = bind_results(m, Solution(folders[-1]), geo)
    cols, rows = P.table(m, [(None, sol)])
    assert cols[0] == "Time (s)" and len(cols) == 1 + 6                 # the disabled probe has no column
    assert cols[1].startswith(f"{avg.label}: ") and "(mol/m^3)" in cols[1]
    assert cols[4].endswith("(mol/m^3*m^2)") or cols[4].endswith("(mol/m)")   # integral over a 2D domain
    t = np.array([r[0] for r in rows])
    assert t == pytest.approx([0.05 * i for i in range(1, 11)])
    a, x, p, i_, e, b = (np.array([r[k] for r in rows]) for k in range(1, 7))
    assert a == pytest.approx(2 / math.pi * np.exp(-t), rel=3e-3)
    assert x == pytest.approx(np.exp(-t), rel=3e-3)
    assert p == pytest.approx(np.exp(-t), rel=3e-3)
    assert i_ == pytest.approx(2 / math.pi * np.exp(-t) * L * L / 10, rel=3e-3)
    assert np.allclose(e, 0, atol=1e-9)
    assert np.isnan(b).all()                                           # a bad expression is a blank column
    # the live view of the run (the SIF's metadata, frames as written) reads the same numbers
    folder, meta = lives[-1]
    live = bind_results(m, Solution(folder, meta), geo)
    cols2, rows2 = P.table(m, [(None, live)], frames=range(4))
    assert cols2 == cols and np.allclose(np.array(rows2, float)[:, :6], np.array(rows, float)[:4, :6])


def test_probes_over_a_parametric_sweep(elmer):
    m = B.new_model("2D", ["ht"], "stationary")
    m.global_defs.child("params").props["table"] = [{"name": "Th", "expr": "300[K]", "descr": ""}]
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1", h="0.1")
    from elmerstudio.core.geometry import build_geometry
    geo = build_geometry(m, display=False)
    ht = m.component.child("physics.ht")
    ht.child("ht.solid").props.update(k_src="user", k="1", rho_src="user", rho="1", Cp_src="user", Cp="1")
    B.add_feature(m, ht, "ht.temperature", geo.select("boundary", xmax=0), T0="273.15[K]")
    B.add_feature(m, ht, "ht.temperature", geo.select("boundary", xmin=1), T0="Th")
    st = m.studies()[0]
    m.create("study.sweep", st, index=0, pname="Th", pvalues="300 400 500", punit="K")
    m.create("probe.domain", m.component.child("definitions"), expr="T", ptype="avg")
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_probe_sw_"), elmer).run()
    sols = [bind_results(m, Solution(f), geo) for f in folders]
    cols, rows = P.table(m, [(s.meta["sweep_value"], s) for s in sols])
    assert cols[0] == "Th" and [r[0] for r in rows] == pytest.approx([300, 400, 500])
    assert [r[1] for r in rows] == pytest.approx([(273.15 + v) / 2 for v in (300, 400, 500)], rel=1e-9)


def test_probes_persist_and_need_an_expression(tmp_path):
    m, geo, st, L = decay_model()
    nodes = with_probes(m, geo, L)
    path = tmp_path / "probes.esm"
    save(m, str(path), None)
    m2, _w = load(str(path))
    kinds = [n.kind for n in m2.component.child("definitions").children if n.kind.startswith("probe.")]
    assert kinds == [n.kind for n in nodes]
    assert m2.by_tag(nodes[4].tag).selection.entities == nodes[4].selection.entities
    assert [n.tag for n in P.probes(m2)] == [n.tag for n in nodes if n.enabled]
    empty = m.create("probe.global", m.component.child("definitions"))
    with pytest.raises(ValueError, match="no expression"):
        P.value(empty, None, 0, m)
