"""Qualification tests for Two transport species (ladder row 3).

CORRECT: test_model_integrity.py::test_two_species_have_their_own_variables_and_keywords
WRONG: test_model_integrity.py (SifError on stab=False, lines 291-298; keyword collision)
CHANGED: different dimensions, values, units, order from existing test
"""
from __future__ import annotations

import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.elmer import find_elmer
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner


# Tolerance: analytic solution for 1D strip with c=2 at x=0, c=5 at x=L is linear.
# D1=4e-9, D2=7e-9. Max error should be < 1e-12 for linear profile on fine mesh.
# CHANGED variant uses different D values, domain length, and BC values.


def _two_species_model(L=0.7, H=0.16, D1=4e-9, D2=7e-9, c1_left=2, c1_right=5, c2_left=11, c2_right=1):
    """Two species on a 2D strip: c and dye, with different D and BCs."""
    m = B.new_model("2D", ["tds"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(H))
    geo = build_geometry(m, display=False)
    p1 = m.component.child("physics.tds")
    p2 = B.add_physics(m, "tds")
    p2.props["species"] = "dye"
    p1.child("tds.tp").props["D"] = f"{D1}[m^2/s]"
    p2.child("tds.tp").props["D"] = f"{D2}[m^2/s]"
    for ph, left, right in [(p1, str(c1_left), str(c1_right)), (p2, str(c2_left), str(c2_right))]:
        B.add_feature(m, ph, "tds.conc", geo.select("boundary", xmax=0), c0=left)
        B.add_feature(m, ph, "tds.conc", geo.select("boundary", xmin=L), c0=right)
    return m, geo


def _solve_two_species(m, geo, elmer):
    """Solve and return (c_vals, dye_vals) at nodes."""
    st = m.studies()[0]
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_qual_2sp_"), elmer).run()
    sol = Solution(folders[-1])
    from elmerstudio.core.nodes import bind_results
    bind_results(m, sol, geo)
    c = sol.evaluate("c")
    dye = sol.evaluate("dye")
    x = sol.topo.nodes[:, 0]
    return x, c, dye


# ---------------------------------------------------------------------- CHANGED VARIANT
# Different geometry (1.5 x 0.3), different D (2e-9, 1e-8), different BCs (1/4, 8/2),
# different unit interpretation (all SI). The analytic profiles are still linear.
# Exact: c(x) = c_left + (c_right - c_left) * x/L
#        dye(x) = dye_left + (dye_right - dye_left) * x/L
def test_two_species_changed_variant(elmer):
    """CHANGED: Different dims, D, BCs - analytic linear profiles must match."""
    L, H = 1.5, 0.3
    D1, D2 = 2e-9, 1e-8
    c1_left, c1_right = 1.0, 4.0
    c2_left, c2_right = 8.0, 2.0

    m, geo = _two_species_model(L=L, H=H, D1=D1, D2=D2,
                                c1_left=c1_left, c1_right=c1_right,
                                c2_left=c2_left, c2_right=c2_right)
    x, c, dye = _solve_two_species(m, geo, elmer)

    c_exact = c1_left + (c1_right - c1_left) * x / L
    dye_exact = c2_left + (c2_right - c2_left) * x / L

    err_c = float(np.max(np.abs(c - c_exact)))
    err_dye = float(np.max(np.abs(dye - dye_exact)))

    # Mesh is fine enough that linear profile error should be tiny
    assert err_c < 1e-12, f"CHANGED variant species c max error {err_c}"
    assert err_dye < 1e-12, f"CHANGED variant species dye max error {err_dye}"


# Additional CHANGED: swap species order (dye first, then c)
# The variable namespacing must still work (c2, c3 instead of c, c2)
def test_two_species_changed_order(elmer):
    """CHANGED: Species added in different order - namespacing must still work."""
    m = B.new_model("2D", ["tds"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1.0", h="0.2")
    geo = build_geometry(m, display=False)

    # Add dye first, then c
    p_dye = m.component.child("physics.tds")
    p_dye.props["species"] = "dye"
    p_c = B.add_physics(m, "tds")  # this gets species="c2"

    p_dye.child("tds.tp").props["D"] = "5e-9[m^2/s]"
    p_c.child("tds.tp").props["D"] = "3e-9[m^2/s]"

    B.add_feature(m, p_dye, "tds.conc", geo.select("boundary", xmax=0), c0="10")
    B.add_feature(m, p_dye, "tds.conc", geo.select("boundary", xmin=1.0), c0="3")
    B.add_feature(m, p_c, "tds.conc", geo.select("boundary", xmax=0), c0="5")
    B.add_feature(m, p_c, "tds.conc", geo.select("boundary", xmin=1.0), c0="1")

    st = m.studies()[0]
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_qual_2sp_ord_"), elmer).run()
    sol = Solution(folders[-1])
    from elmerstudio.core.nodes import bind_results
    bind_results(m, sol, geo)

    # Variables should be "dye" and "c2" (second species gets c2)
    x = sol.topo.nodes[:, 0]
    dye = sol.evaluate("dye")
    c2 = sol.evaluate("c2")

    dye_exact = 10 + (3 - 10) * x / 1.0
    c2_exact = 5 + (1 - 5) * x / 1.0

    err_dye = float(np.max(np.abs(dye - dye_exact)))
    err_c2 = float(np.max(np.abs(c2 - c2_exact)))

    assert err_dye < 1e-12, f"CHANGED order dye max error {err_dye}"
    assert err_c2 < 1e-12, f"CHANGED order c2 max error {err_c2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])