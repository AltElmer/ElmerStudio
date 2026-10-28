"""Electrode Surface with a total current (galvanostatic): the study finds the electrode potential.

On the 1D strip of test_electrochemistry (Butler-Volmer anode, counter electrode at phil = 0) a current density i
needs the electrode potential V = (2/f) asinh(i/(2 i0)) + i L/sigma exactly.
"""
import math

import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.sif import build_sif
from test_electrochemistry import _solve, f, strip

L, W, SIGMA, I0 = 2e-4, 1e-4, 20.0, 50.0


def exact_potential(i):
    return 2 / f * math.asinh(i / (2 * I0)) + i * L / SIGMA


@pytest.mark.parametrize("i", [400.0, -250.0])
def test_total_current_finds_the_electrode_potential(i, elmer):
    m, geo, an = strip("bv", V=0.0, L=L, W=W, sigma=SIGMA, i0=I0)
    an.props.update(control="current", Iset=f"{i * W!r}[A]", phis="0[V]")
    sol = _solve(m, geo, "es_gal_")[-1]
    fi = sol.frame_index("last")
    assert float(sol.evaluate(f"cd.{an.tag}.Itot", fi)[0]) == pytest.approx(i * W, rel=1e-6)
    assert sol.params[f"phis_{an.tag}"] == pytest.approx(exact_potential(i), abs=1e-6)
    # the overpotential result uses the potential found, not the first guess
    eta = 2 / f * math.asinh(i / (2 * I0))
    P = sol.topo.nodes
    on = abs(P[:, 1]) < 1e-12
    assert sol.evaluate(f"cd.{an.tag}.eta", fi)[on] == pytest.approx(eta, abs=1e-6)


def test_potential_control_is_unchanged_and_misuse_is_refused(elmer):
    m, geo, an = strip("bv", V=0.1, L=L, W=W, sigma=SIGMA, i0=I0)
    st = m.studies()[0]
    assert "phis_" not in build_sif(m, st, B.study_steps(st)[0], geo, 100).text
    cathode = [c for c in m.component.child("physics.cd").children if c.kind == "cd.pot"][0]
    two = B.add_feature(m, m.component.child("physics.cd"), "cd.es", cathode.selection.entities,
                        control="current", Iset="0[A]")
    an.props["control"] = "current"
    with pytest.raises(RuntimeError, match="one electrode per component"):
        _solve(m, geo, "es_gal2_")
    two.enabled = False
    t = B.add_study(m, "time")
    m.root.children.remove(st)
    with pytest.raises(RuntimeError, match="Stationary study"):
        _solve(m, geo, "es_gal3_")
    assert t in m.studies()
