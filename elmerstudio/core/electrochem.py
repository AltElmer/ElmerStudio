"""Electrochemistry results: electrode currents from Elmer's reaction loads, overpotentials, dissolution rates.

COMSOL names are kept: for an Electrode Surface with tag ``es1`` of the Current Distribution interface

    cd.es1.Itot   total electrode current entering the electrolyte (A; per metre of depth in 2D,
                  the full revolution in 2D axisymmetry) -- the sum of the nodal reactions, which is
                  exactly conservative (an element-gradient integral is not)
    cd.es1.eta    overpotential phis_ext - Eeq - phil (V)
    cd.es1.iloc   local current density from the kinetics expression i(eta) (A/m^2)
    cd.es1.vdis   metal dissolution (recession) velocity M i / (n F rho) (m/s), from the conservative cd.iloc
"""
from __future__ import annotations

import math

from .physics import F_CONST, cd_eta, cd_kinetics


def electrode_results(model, sol, params=None):
    """({global name: (array, boundaries, factor)}, {variable: expression}, flux boundaries) for Electrode Surfaces."""
    comp = model.component
    gl, var, flux = {}, {}, set()
    if comp is None:
        return gl, var, flux
    axi = bool(sol.meta.get("axi"))
    factor = 2 * math.pi if axi else 1.0                 # Elmer's axisymmetric loads are per radian
    for ph in comp.children:
        if ph.kind != "physics.cd" or not ph.enabled:
            continue
        for f in ph.children:
            if f.kind in ("cd.es", "cd.pot", "cd.ncd") and f.enabled and f.selection is not None:
                flux.update(int(b) for b in f.selection.entities)
            if f.kind != "cd.es" or not f.enabled or f.selection is None:
                continue
            bnds = sorted(int(b) for b in f.selection.entities)
            t = f.tag
            gl[f"cd.{t}.Itot"] = ("phil loads", bnds, factor)
            eta = cd_eta(f)
            var[f"cd.{t}.eta"] = eta
            if ph.get("dist", "secondary") != "primary" and f.get("kin", "bv") != "primary":
                i, _di = cd_kinetics(f)
                var[f"cd.{t}.iloc"] = "(" + i.replace("eta", eta) + ")"
            var[f"cd.{t}.vdis"] = (f"(({f.get('M', '55.845[g/mol]')})/(({f.get('nel', '2')})*{F_CONST!r}*"
                                   f"({f.get('rhom', '7870[kg/m^3]')}))*cd.iloc)")
    return gl, var, flux
