"""Issue #12: magnetic energy density ``mf.Wm = 0.5 * |B| * |H|`` and Elmer's global ``Magnetic Field Energy`` scalar.

Uses the 2D axisymmetric ``coil_axi`` example (copper coil around an iron core, mur=4000).

Verified behaviour:
* ``mf.Wm`` is computable pointwise as ``0.5 * mf.normB * mf.normH``.
* Elmer's global ``Magnetic Field Energy`` (printed to the solver log) is exposed as
  ``mf.W_mag`` / ``magnetic_field_energy`` and is positive.
Negative control:
* Integrating the nodal (shape-function-smeared) ``mf.Wm`` across the high-mur core|air
  interface overestimates the energy vs Elmer's elemental value (full energy = scalar * 2 pi
  for an axisymmetric model); a naive uniform-permeability energy also disagrees.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from elmerstudio.core.examples import coil_axi
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner


def _solve(coil_axi_fn=coil_axi):
    import tempfile

    from elmerstudio.core.elmer import find_elmer

    inst = find_elmer()
    m = coil_axi_fn()
    st = m.studies()[0]
    work = tempfile.mkdtemp(prefix="es_mf_energy_")
    folders = StudyRunner(m, st, work, inst).run()
    return Solution(folders[-1])


@pytest.mark.parametrize("key", ["mf_wm", "magnetic_field_energy", "mf_w_mag"])
def test_mf_energy_evaluable(key, elmer):
    """mf.Wm derives from the fields; Elmer's scalar Magnetic Field Energy is exposed."""
    sol = _solve()
    i = sol.frame_index("last")
    v = sol.evaluate(key, i)
    assert np.isfinite(v).all()
    if key == "magnetic_field_energy" or key == "mf_w_mag":
        assert float(v[0]) > 0          # global scalar, broadcast onto every node
    else:
        assert float(np.nanmax(v)) >= 0  # energy density


def test_wm_derives_from_b_and_h(elmer):
    """mf.Wm == 0.5 * mf.normB * mf.normH pointwise."""
    sol = _solve()
    i = sol.frame_index("last")
    wm = sol.evaluate("mf_wm", i)
    b = sol.evaluate("mf_normB", i)
    h = sol.evaluate("mf_normH", i)
    assert b.size and h.size and wm.size == b.size
    assert np.allclose(wm, 0.5 * b * h, rtol=1e-6, atol=1e-12), \
        f"mf.Wm mismatch: max abs diff {float(np.nanmax(np.abs(wm - 0.5*b*h))):.3e}"


def test_nodal_smearing_disagrees_across_mur_jump(elmer):
    """Negative control: nodal-smeared mf.Wm overestimates the energy across the mur=4000 jump.

    Elmer reports the elemental energy per radian for an axisymmetric model (full energy =
    scalar * 2 pi). The nodal (smeared) integral of mf.Wm must NOT match it.
    """
    sol = _solve()
    i = sol.frame_index("last")
    nodal = sol.integrate("mf_wm", i, level="domain", revolved=True)
    elmer_full = sol.evaluate("magnetic_field_energy", i)[0] * 2 * math.pi
    assert np.isfinite(nodal) and np.isfinite(elmer_full) and elmer_full > 0
    rel = abs(nodal - elmer_full) / elmer_full
    assert rel > 0.10, f"smearing did not corrupt the integral (rel {rel:.2%}): nodal {nodal:.6e} vs elemental {elmer_full:.6e}"


def test_uniform_permeability_energy_disagrees(elmer):
    """Negative control: a naive uniform-permeability energy 0.5*B^2/(2*mu0) does NOT match Elmer's scalar.

    Across the iron core (mur=4000) the real energy density 0.5*B*H is far below 0.5*B^2/(2*mu0),
    so assuming one permeability everywhere corrupts the integral.
    """
    sol = _solve()
    i = sol.frame_index("last")
    scalar = sol.evaluate("magnetic_field_energy", i)[0]
    b = sol.evaluate("mf_normB", i)
    naive = 0.5 * b * b / (4 * math.pi * 1e-7)
    naive_integral = sol._integrate_values(naive, "domain", revolved=True)
    assert np.isfinite(naive_integral) and naive_integral > 0
    rel = abs(naive_integral - scalar) / scalar
    assert rel > 0.20, f"uniform-mu energy (wrongly) matches Elmer scalar: naive {naive_integral:.6e} vs {scalar:.6e} (rel {rel:.2%})"
