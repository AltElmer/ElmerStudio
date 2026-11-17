"""Qualification tests for Assembly conduction (ladder row 5).

CORRECT: test_assembly_pairs.py::test_temperature_is_continuous_across_the_pair
WRONG: test_assembly_pairs.py (pair.enabled=False control, lines 61-66)
CHANGED: test_assembly_pairs.py::test_the_pair_does_not_depend_on_the_objects_order

Row 5 appears complete in D4-M1. This file adds an additional CHANGED variant
with different geometry/material parameters.
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


# Tolerance: series thermal resistance gives linear profile. Error < 1e-3 K.
# CHANGED variant: different lengths, conductivities, boundary temps.


def _assembly_model(L1=0.7, L2=1.1, H=0.12, k1=30, k2=150, T_left=300, T_right=360, pair_enabled=True):
    """Two blocks in series with identity pair at interface."""
    m = B.new_model("2D", ["ht"], "stationary")
    g = m.component.child("geometry")
    m.create("geom.rectangle", g, index=0, w=str(L1), h=str(H))
    m.create("geom.rectangle", g, index=1, w=str(L2), h=str(H), x=str(L1))
    g.child("geom.finalize").props["action"] = "none"  # Form Assembly
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    d1 = geo.select("domain", xmax=L1)
    d2 = geo.select("domain", xmin=L1)
    default = ph.child("ht.solid")
    default.selection.all = False
    default.selection.entities = d1
    default.props.update(k_src="user", k=str(k1))
    B.add_feature(m, ph, "ht.solid", d2, k_src="user", k=str(k2))
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0=str(T_left))
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmin=L1+L2), T0=str(T_right))
    pair = next(n for n in m.component.child("definitions").children if n.kind == "pair.identity")
    pair.enabled = pair_enabled
    return m, geo, L1, L2, k1, k2, T_left, T_right


def _solve_assembly(m, geo, elmer):
    st = m.studies()[0]
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_qual_asm_"), elmer).run()
    sol = Solution(folders[-1])
    x = sol.topo.nodes[:, 0]
    T = sol.evaluate("T")
    return x, T


# ---------------------------------------------------------------------- ADDITIONAL CHANGED VARIANT
# Different dimensions, conductivities, boundary temperatures
def test_assembly_changed_variant(elmer):
    """CHANGED: Different L1, L2, k1, k2, T_left, T_right - analytic profile must match."""
    L1, L2 = 0.5, 1.5
    k1, k2 = 50, 80
    T_left, T_right = 280, 400
    m, geo, _, _, _, _, _, _ = _assembly_model(L1=L1, L2=L2, k1=k1, k2=k2,
                                               T_left=T_left, T_right=T_right)
    x, T = _solve_assembly(m, geo, elmer)

    # Series resistance: q = (T_right - T_left) / (L1/k1 + L2/k2)
    q = (T_right - T_left) / (L1/k1 + L2/k2)
    T_exact = np.where(x <= L1,
                       T_left + q * x / k1,
                       T_left + q * L1 / k1 + q * (x - L1) / k2)
    err = float(np.max(np.abs(T - T_exact)))
    assert err < 1e-3, f"CHANGED variant max error {err} K"


# CHANGED: 3D version with different geometry
def test_assembly_3d_changed_variant(elmer):
    """CHANGED: 3D blocks with different dimensions - same physics."""
    m = B.new_model("3D", ["ht"], "stationary")
    g = m.component.child("geometry")
    m.create("geom.block", g, index=0, w="0.5", d="0.3", h="0.3")
    m.create("geom.block", g, index=1, w="1.5", d="0.3", h="0.3", x="0.5")
    g.child("geom.finalize").props["action"] = "none"
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    d1 = geo.select("domain", xmax=0.5)
    d2 = geo.select("domain", xmin=0.5)
    default = ph.child("ht.solid")
    default.selection.all = False
    default.selection.entities = d1
    default.props.update(k_src="user", k="50")
    B.add_feature(m, ph, "ht.solid", d2, k_src="user", k="80")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0="280")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmin=2.0), T0="400")
    pair = next(n for n in m.component.child("definitions").children if n.kind == "pair.identity")
    pair.enabled = True

    st = m.studies()[0]
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_qual_asm3d_"), elmer).run()
    sol = Solution(folders[-1])
    x = sol.topo.nodes[:, 0]
    T = sol.evaluate("T")

    q = (400 - 280) / (0.5/50 + 1.5/80)
    T_exact = np.where(x <= 0.5,
                       280 + q * x / 50,
                       280 + q * 0.5 / 50 + q * (x - 0.5) / 80)
    err = float(np.max(np.abs(T - T_exact)))
    assert err < 1e-2, f"3D CHANGED variant max error {err} K"  # 3D mesh coarser


if __name__ == "__main__":
    pytest.main([__file__, "-v"])