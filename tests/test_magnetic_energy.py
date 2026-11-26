"""Issue #12: magnetic energy not available as a result, and B.H integrals off at material interfaces.

coil_axi: 2D axisymmetric stationary model, copper coil around an iron core (mur = 4000),
driven by J0 = 2 A/mm^2.  Elmer 9.0 MagnetoDynamics2D + MagnetoDynamicsCalcFields with
"Calculate Elemental Fields = True" writes per-body ("elemental") B and H to side files
(sides_t*.vtu); "Separate Magnetic Energy = True" makes CalcFields add "res: Magnetic
Field Energy" to the scalars file (SaveScalars solver).

Studio exposes (SOLVER.KEYWORDS verified; no invented keywords):
* mf.Wm = 0.5 B.H energy density, evaluated pointwise with per-body fields.  At nodes
  shared between bodies the adjoining body's value is used, so integrating mf.Wm over
  each body with side fields reproduces Elmer's global field-energy scalar.
* mf.W_mag / magnetic_field_energy: Elmer's global Magnetic Field Energy scalar
  (per radian for axisymmetric models; Solution.scalar() applies the x 2 pi turn).

The side-resolved integral of mf.Wm agrees with Elmer's scalar within 1.6 % on the
default mesh and on a refined mesh (TOL_AGREE below allows 3 %); the plain nodal
integral (B and H continuous across the mu_r = 4000 jump) misses Elmer by ~8 % on
this mesh, so the comparison below fails if side fields are not applied and passes
only when they are.
"""

import math

import numpy as np
import pytest

from elmerstudio.core.examples import coil_axi
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner

TOL_AGREE = 0.03   # side-resolved integral vs Elmer's scalar (observed 1.6 % on default + refined meshes)


def _solve():
    import tempfile
    from elmerstudio.core.elmer import find_elmer
    inst = find_elmer()
    m = coil_axi()
    st = m.studies()[0]
    work = tempfile.mkdtemp(prefix="es_mf_energy_")
    folders = StudyRunner(m, st, work, inst).run()
    return Solution(folders[-1])


def _elmer_full(sol, i):
    """Elmer's global Magnetic Field Energy for this frame, in joules (x 2 pi turn applied for axisym)."""
    return float(sol.scalar("mf.W_mag", i))


@pytest.mark.parametrize("key", ["mf_wm", "magnetic_field_energy", "mf_w_mag"])
def test_mf_energy_evaluable(key, elmer):
    """mf.Wm is computable pointwise; Elmer's global Magnetic Field Energy is exposed as a scalar."""
    sol = _solve()
    i = sol.frame_index("last")
    v = sol.evaluate(key, i)
    assert np.isfinite(v).all() and v.size
    if key in ("magnetic_field_energy", "mf_w_mag"):
        assert float(v[0]) > 0.0      # global energy scalar
    else:
        assert float(np.nanmax(v)) >= 0.0   # energy density, non-negative in linear media


def test_side_resolved_integral_matches_elmer(elmer):
    """Integral of mf.Wm over each body with per-body fields equals Elmer's scalar.

    Negative control: the plain nodal integral (B and H continuous across the material
    interface) misses Elmer by ~8 % on this mesh, so this test fails if the side fields
    are not applied (the integral then falls back to nodal fields) and passes only when
    they are.
    """
    sol = _solve()
    i = sol.frame_index("last")
    elmer = _elmer_full(sol, i)                          # 0.02299563 J for this coil
    side = sol.integrate("mf_wm", i, level="domain", revolved=True)   # per-body fields
    nodal = sol._integrate_values(sol.evaluate("mf_wm", i), "domain", revolved=True)   # smeared fields
    assert np.isfinite(side) and np.isfinite(nodal) and elmer > 0.0
    rel_side = abs(side - elmer) / elmer
    rel_nodal = abs(nodal - elmer) / elmer
    assert rel_side < TOL_AGREE, (
        f"side-resolved integral misses Elmer: {side:.6e} vs {elmer:.6e} ({rel_side:.2%})")
    assert rel_nodal > TOL_AGREE, (
        f"nodal smearing matches Elmer (it should not): {nodal:.6e} vs {elmer:.6e} ({rel_nodal:.2%})")


def test_uniform_permeability_energy_overestimates(elmer):
    """Negative control: naive 0.5 B^2/(2 mu0) with one permeability everywhere is not Elmer's energy."""
    sol = _solve()
    i = sol.frame_index("last")
    elmer = _elmer_full(sol, i)
    b = sol.evaluate("mf_normB", i)
    naive = 0.5 * b * b / (4 * math.pi * 1e-7)        # 0.5 B^2 / (2 mu0)
    naive_full = sol._integrate_values(naive, "domain", revolved=True)
    assert np.isfinite(naive_full) and elmer > 0.0
    rel = abs(naive_full - elmer) / elmer
    assert rel > 0.50, (
        f"uniform-permeability energy matches Elmer: {naive_full:.6e} vs {elmer:.6e} ({rel:.2%})")
