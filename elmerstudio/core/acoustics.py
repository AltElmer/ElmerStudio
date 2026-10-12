"""Acoustics postprocessing: the exterior field (Kirchhoff-Helmholtz integral) and point evaluation of
far-field quantities, as COMSOL's Exterior Field Calculation (acpr.efc1.pext, acpr.efc1.Lp_pext).

The time convention is exp(+i w t), the one Elmer's HelmholtzSolver and COMSOL use: outgoing waves are
exp(-i k r), and the free-space Green's function is G = exp(-i k R) / (4 pi R).  For a point x outside the
closed surface S (normal n pointing away from the sources)

    p(x) = sum over S of  p(y) dG/dn_y - G dp/dn_y  dS_y .

2D axisymmetric solutions are integrated over the revolved surface; 2D planar models are not supported
(that needs Hankel functions, i.e. an infinitely long source).
"""
from __future__ import annotations

import math
import re

import numpy as np

from . import units
from .model import selection_entities

P_REF = 20e-6                       # reference pressure for the sound pressure level (air)
EFC_NAME = re.compile(r"(?<![\w.])acpr\.(efc\d+)\.(pext|Lp_pext|abs_pext|p_ext)(?![\w(])")


class ExteriorField:
    """Exterior field of one solution from the pressure on a set of boundaries."""

    def __init__(self, sol, boundaries, c, axi=None, incident=(), bc=None):
        self.sol = sol
        self.bc = dict(bc or {})                 # {boundary: ('swr', center SI, beta) | ('hard',)}: dp/dn from the BC
        self.incident = list(incident)          # [(p0, direction)]: plane waves subtracted (scattered field)
        self.bnds = sorted(int(b) for b in boundaries)
        self.c = float(c)
        self.axi = bool(sol.meta.get("axi")) if axi is None else bool(axi)
        if sol.sdim == 2 and not self.axi:
            raise ValueError("Exterior field calculation needs a 3D or 2D axisymmetric model")
        self._geom = None
        self._fields: dict = {}

    # ---------------------------------------------------------------- surface quadrature (frame independent)
    def _geometry(self):
        """Quadrature points as linear combinations of mesh nodes: (ids (Q,2|3), shape weights, y, n, w)."""
        if self._geom is not None:
            return self._geom
        g = self.sol.boundary_grid()
        if not g.n_cells or "bnd" not in g.cell_data:
            raise ValueError("Exterior field: the solution has no boundary elements")
        cells = np.where(np.isin(np.asarray(g.cell_data["bnd"]), self.bnds))[0]
        if not len(cells):
            raise ValueError(f"Exterior field: boundaries {self.bnds} have no elements")
        P = np.asarray(self.sol.topo.nodes, float) * self.sol.scale
        if self.sol.sdim == 2:
            segs, sb = [], []
            bcell = np.asarray(g.cell_data["bnd"])
            for ci in cells:
                ids = np.asarray(g.get_cell(int(ci)).point_ids)
                if len(ids) == 3:                      # quadratic edge: end, end, middle
                    segs += [(ids[0], ids[2]), (ids[2], ids[1])]
                    sb += [bcell[ci]] * 2
                else:
                    segs.append((ids[0], ids[1]))
                    sb.append(bcell[ci])
            segs = np.array(segs)
            self._bq = np.tile(np.array(sb), 2)
            a, b = P[segs[:, 0], :2], P[segs[:, 1], :2]
            t = b - a
            L = np.linalg.norm(t, axis=1)
            nrm = np.column_stack([t[:, 1], -t[:, 0]]) / L[:, None]
            mid = (a + b) / 2
            zc = float(np.sum(mid[:, 1] * L * np.abs(mid[:, 0])) / max(np.sum(L * np.abs(mid[:, 0])), 1e-300))
            flip = np.sum(nrm * (mid - np.array([0.0, zc])), axis=1) < 0      # point away from the enclosed volume
            nrm[flip] *= -1
            gp = np.array([0.5 - 0.5 / math.sqrt(3), 0.5 + 0.5 / math.sqrt(3)])
            ids, sw, y, n, w = [], [], [], [], []
            for s in gp:
                ids.append(segs)
                sw.append(np.tile([1 - s, s], (len(segs), 1)))
                y.append(a + s * t)
                n.append(nrm)
                w.append(L / 2)
            self._geom = (np.vstack(ids), np.vstack(sw), np.vstack(y), np.vstack(n), np.concatenate(w))
            self._group, self._h = 2, float(L.mean())
        else:
            sub = g.extract_cells(cells)
            root_ids = np.asarray(sub.point_data["vtkOriginalPointIds"]).copy()
            surf = _surface(sub, 1)
            sid = np.asarray(surf.point_data["vtkOriginalPointIds"])
            if np.any(sid < 0):
                surf = _surface(sub, 0)
                sid = np.asarray(surf.point_data["vtkOriginalPointIds"])
            surf = surf.triangulate()
            tri = surf.faces.reshape(-1, 4)[:, 1:]
            tri = root_ids[sid[tri]]
            tb = np.asarray(surf.cell_data["bnd"]) if "bnd" in surf.cell_data else np.full(len(tri), -1)
            self._bq = np.tile(tb, 3)
            A, B, C = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
            cr = np.cross(B - A, C - A)
            area = np.linalg.norm(cr, axis=1) / 2
            nrm = cr / (2 * area[:, None])
            cen = (A + B + C) / 3
            ctr = np.sum(cen * area[:, None], axis=0) / area.sum()
            flip = np.sum(nrm * (cen - ctr), axis=1) < 0
            nrm[flip] *= -1
            ids, sw, y = [], [], []
            for i, j in ((0, 1), (1, 2), (2, 0)):        # edge-midpoint rule (exact for quadratics)
                ids.append(tri[:, [i, j]])
                sw.append(np.full((len(tri), 2), 0.5))
                y.append((P[tri[:, i]] + P[tri[:, j]]) / 2)
            self._geom = (np.vstack(ids), np.vstack(sw), np.vstack(y), np.vstack([nrm] * 3),
                          np.concatenate([area / 3] * 3))
            self._group, self._h = 3, float(np.sqrt(2 * area.mean()))
        return self._geom

    def _surface_fields(self, i):
        """Complex pressure and its gradient at the mesh nodes for frame *i*."""
        if i in self._fields:
            return self._fields[i]
        arr = self.sol.arrays(i)
        pw = arr.get("pressure wave")
        if pw is None:
            raise ValueError("Exterior field: the solution has no acoustic pressure")
        g = self.sol.bulk_grid().copy()
        g.points = np.asarray(self.sol.topo.nodes, float) * self.sol.scale
        g.point_data["pr"] = np.asarray(pw[:, 0], float)
        g.point_data["pi"] = np.asarray(pw[:, 1], float)
        gr = g.compute_derivative(scalars="pr", gradient="gr")
        gi = g.compute_derivative(scalars="pi", gradient="gi")
        p = pw[:, 0] + 1j * pw[:, 1]
        grad = np.asarray(gr.point_data["gr"]) + 1j * np.asarray(gi.point_data["gi"])
        self._fields = {i: (p, grad)}
        return self._fields[i]

    def wavenumber(self, i):
        f = self.sol.frames[i].value if 0 <= i < len(self.sol.frames) else 0.0
        if not f:
            raise ValueError("Exterior field: needs a Frequency Domain solution")
        return 2 * math.pi * f / self.c

    def quadrature(self, i):
        """(y, n, w, p, dp/dn) at the 3D quadrature points for frame *i*."""
        ids, sw, y, n, w = self._geometry()
        p, grad = self._surface_fields(i)
        k = self.wavenumber(i)
        pq = np.sum(p[ids] * sw, axis=1)
        gq = np.sum(grad[ids] * sw[:, :, None], axis=1)
        nd = n.shape[1]
        dpdn = np.sum(gq[:, :nd] * n, axis=1)              # numerical normal derivative (total field)
        # 3D positions of the quadrature points ((r, z) -> (r, 0, z) on axisymmetric models)
        y3 = np.column_stack([y[:, 0], np.zeros(len(y)), y[:, 1]]) if self.sol.sdim == 2 else y
        incs = [(p0, np.asarray(d, float)) for p0, d in self.incident]
        for b, spec in self.bc.items():                   # where the boundary condition fixes dp/dn, use it
            sel = self._bq == b
            if not sel.any():
                continue
            if spec[0] == "hard":
                dpdn[sel] = 0.0
            elif spec[0] == "swr":
                ctr, beta = np.asarray(spec[1], float), float(spec[2])
                rv = y3[sel] - ctr
                r = np.linalg.norm(rv, axis=1)
                g = np.zeros(sel.sum(), complex)
                for p0, d in incs:
                    pi_ = p0 * np.exp(-1j * k * (y3[sel] @ d))
                    g += pi_ * (1j * k * (1 - (rv @ d) / r) + beta / r)
                dpdn[sel] = -(1j * k + beta / r) * pq[sel] + g
        for p0, d in incs:                                # radiate the scattered field only
            pinc = p0 * np.exp(-1j * k * (y3 @ d))
            n3 = np.column_stack([n[:, 0], np.zeros(len(n)), n[:, 1]]) if self.sol.sdim == 2 else n
            pq = pq - pinc
            dpdn = dpdn - (-1j * k) * (n3 @ d) * pinc
        if self.sol.sdim == 2:                            # revolve the meridian quadrature (r, z) around z
            rmax = float(np.max(np.abs(y[:, 0]))) or 1.0
            nphi = int(max(48, 4 * k * rmax + 24))
            phi = 2 * math.pi * np.arange(nphi) / nphi
            cphi, sphi = np.cos(phi), np.sin(phi)
            r, z = y[:, 0][:, None], y[:, 1][:, None]
            Y = np.stack([r * cphi, r * sphi, z * np.ones(nphi)], axis=-1).reshape(-1, 3)
            N = np.stack([n[:, 0][:, None] * cphi, n[:, 0][:, None] * sphi,
                          np.broadcast_to(n[:, 1][:, None], (len(n), nphi))], axis=-1).reshape(-1, 3)
            W = (w[:, None] * np.abs(r) * (2 * math.pi / nphi) * np.ones(nphi)).ravel()
            return Y, N, W, np.repeat(pq, nphi), np.repeat(dpdn, nphi)
        return y, n, w, pq, dpdn

    def pressure(self, i, X):
        """Exterior pressure (complex, Pa) at points *X* (SI, 3D Cartesian; for axisymmetric models x = r cos phi)."""
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] == 2:
            X = np.column_stack([X, np.zeros(len(X))])
        k = self.wavenumber(i)
        fine = self.quadrature(i)
        coarse = self._coarse(fine)
        out, dmin = _kirchhoff(X, k, *coarse, want_dmin=True)
        near = dmin < 4 * self._h                           # close to the surface: the full rule
        if near.any():
            out[near] = _kirchhoff(X[near], k, *fine)[0]
        return out

    def _coarse(self, q):
        """Lump groups of quadrature points (same element) into one: for points a few elements away."""
        Y, N, W, p, dpdn = q
        g = self._group
        if g <= 1 or len(Y) % g:
            self._h = 0.0
            return q
        nq = len(Y) // g
        if self.sol.sdim == 2:      # meridian points are ordered [gauss point][segment] before the revolution
            nphi = len(Y) // len(self._geom[2])
            idx = np.arange(len(Y)).reshape(g, -1, nphi)          # (gauss, segment, phi)
            idx = idx.transpose(1, 2, 0).reshape(-1, g)
        else:
            idx = np.arange(len(Y)).reshape(g, nq).T                # (triangle, 3 edge midpoints)
        Wg = W[idx]
        Wt = Wg.sum(axis=1)
        Yc = (Y[idx] * Wg[:, :, None]).sum(axis=1) / Wt[:, None]
        Nc = N[idx].mean(axis=1)
        Nc /= np.linalg.norm(Nc, axis=1)[:, None]
        pc = (p[idx] * Wg).sum(axis=1) / Wt
        dc = (dpdn[idx] * Wg).sum(axis=1) / Wt
        return Yc, Nc, Wt, pc, dc


def _kirchhoff(X, k, Y, N, W, p, dpdn, want_dmin=False):
    """sum_q W [p dG/dn - G dp/dn] at points X, G = exp(-ikR)/(4 pi R); real float32 arithmetic in chunks."""
    X = np.asarray(X, float)
    out = np.empty(len(X), complex)
    dmin = np.full(len(X), np.inf)
    A, Bv = p * W, dpdn * W
    Ar, Ai = A.real.astype(np.float32), A.imag.astype(np.float32)
    Br, Bi = Bv.real.astype(np.float32), Bv.imag.astype(np.float32)
    Yf, Nf = Y.astype(np.float32), N.astype(np.float32)
    kf = np.float32(k)
    chunk = max(1, int(3e6 // max(len(Y), 1)))
    for s in range(0, len(X), chunk):
        Xs = X[s:s + chunk].astype(np.float32)
        D = Xs[:, None, :] - Yf[None, :, :]
        R = np.sqrt(np.einsum("mqd,mqd->mq", D, D))
        if want_dmin:
            dmin[s:s + chunk] = R.min(axis=1)
        invR = 1 / R
        cosn = np.einsum("mqd,qd->mq", D, Nf) * invR
        T, U = cosn * invR, cosn * kf
        P = T * Ar - U * Ai - Br
        Q = T * Ai + U * Ar - Bi
        kR = kf * R
        c, sn = np.cos(kR), np.sin(kR)
        g = invR * np.float32(1 / (4 * math.pi))
        out.real[s:s + chunk] = (g * (c * P + sn * Q)).sum(axis=1, dtype=np.float64)
        out.imag[s:s + chunk] = (g * (c * Q - sn * P)).sum(axis=1, dtype=np.float64)
    return out, dmin


def _surface(grid, level):
    from .datasets import _surface as surface_with_ids
    return surface_with_ids(grid, level)


def spl(p):
    """Sound pressure level (dB re 20 uPa) of a complex amplitude."""
    return 20 * np.log10(np.maximum(np.abs(p) / math.sqrt(2), 1e-30) / P_REF)


def uses_efc(expr) -> bool:
    return bool(EFC_NAME.search(str(expr)))


def to_cartesian(sol, P):
    """Plot/evaluation coordinates (SI) -> 3D Cartesian points for the exterior field ((r, z) -> (r, 0, z))."""
    P = np.atleast_2d(np.asarray(P, float))
    if sol.sdim == 2 and sol.meta.get("axi"):
        return np.column_stack([P[:, 0], np.zeros(len(P)), P[:, 1]])
    if P.shape[1] == 2:
        return np.column_stack([P, np.zeros(len(P))])
    return P


def point_values(sol, expr, i, P, efc=None):
    """Values of *expr* at points *P* (SI, in the solution's coordinates).

    Expressions with exterior-field variables (acpr.efc1.pext, ...Lp_pext, ...abs_pext) take the mesh pressure
    where the point lies in the mesh and the Kirchhoff-Helmholtz integral elsewhere; other expressions are
    interpolated from the mesh (NaN outside it).
    """
    P = np.atleast_2d(np.asarray(P, float))
    if not uses_efc(expr):
        return sol.point_eval(expr, i, P[:, :max(2, sol.sdim)] if sol.sdim == 2 else P)
    efc = efc if efc is not None else getattr(sol, "efc", {}) or {}
    Pm = P[:, :2] if sol.sdim == 2 else P
    pin = sol.point_eval("acpr.p_t", i, Pm) + 1j * sol.point_eval("acpr.p_im", i, Pm)
    names = {}
    for tag in sorted(set(m.group(1) for m in EFC_NAME.finditer(str(expr)))):
        if tag not in efc:
            raise units.ExprError(f"No Exterior Field Calculation named {tag} (add one to the acoustics interface)")
        p = pin.copy()
        out = ~np.isfinite(p)
        if efc[tag].incident and (~out).any():          # pext is the scattered field: remove the incident wave
            k = efc[tag].wavenumber(i)
            Xin = to_cartesian(sol, P[~out])
            for p0, d in efc[tag].incident:
                p[~out] -= p0 * np.exp(-1j * k * (Xin @ np.asarray(d, float)))
        if out.any():
            p[out] = efc[tag].pressure(i, to_cartesian(sol, P[out]))
        names[f"{tag}_pext"] = names[f"{tag}_p_ext"] = p
        names[f"{tag}_abs_pext"] = np.abs(p)
        names[f"{tag}_Lp_pext"] = spl(p)
    X = to_cartesian(sol, P)
    names.update({"x": X[:, 0], "y": X[:, 1], "z": X[:, 2]})
    if sol.sdim == 2 and sol.meta.get("axi"):
        names.update({"r": P[:, 0], "z": P[:, 1]})
    names.update(getattr(sol, "params", {}) or {})
    if 0 <= i < len(sol.frames):
        names["freq"] = sol.frames[i].value
    e = EFC_NAME.sub(lambda m: f"{m.group(1)}_{m.group(2)}", str(expr))
    v = units.evaluate(e, names)
    v = np.asarray(v)
    if np.iscomplexobj(v):
        v = np.real(v)
    return np.broadcast_to(np.asarray(v, float), (len(P),)).copy()


def efc_operators(model, sol, geo=None):
    """{efcN: ExteriorField} for the model's Exterior Field Calculation features."""
    out = {}
    comp = model.component
    if comp is None:
        return out
    params = {}
    try:
        params = dict(model.parameters())
    except Exception:
        pass
    for ph in comp.children:
        if ph.kind != "physics.acpr" or not ph.enabled:
            continue
        for f in ph.children:
            if f.kind != "acpr.efc" or not f.enabled or f.selection is None:
                continue
            bnds = selection_entities(f, sol.meta) if not f.selection.all else sorted(int(b) for b in sol.topo.bnd)
            if not bnds:
                continue
            c = _sound_speed(model, ph, bnds, geo, params)
            inc = []
            from .physics import INCIDENT_DIRS
            for w in ph.children:
                if w.kind == "acpr.swr" and w.enabled and w.get("incident", "none") == "plane":
                    try:
                        p0 = float(units.evaluate(str(w.get("p0", "1[Pa]")), params))
                    except Exception:
                        p0 = 1.0
                    inc.append((p0, INCIDENT_DIRS.get(w.get("dir", "+z"), (0, 0, 1))))
            bc = _flux_conditions(ph, bnds, geo, params, sol)
            key = (f.tag, tuple(sorted(bnds)), c, tuple(inc), tuple(sorted((b, str(v)) for b, v in bc.items())))
            cache = sol.__dict__.setdefault("_efc_cache", {})
            if key in cache:                       # keep the surface quadrature between plots
                out[f.tag] = cache[key]
                continue
            try:
                out[f.tag] = cache[key] = ExteriorField(sol, bnds, c, incident=inc, bc=bc)
            except ValueError:
                continue
    return out


def _flux_conditions(ph, bnds, geo, params, sol):
    """{boundary: spec} where the acoustics boundary condition fixes dp/dn (later features override earlier)."""
    axi = bool(sol.meta.get("axi"))
    out: dict = {}
    for f in ph.children:
        if not f.enabled or f.selection is None or f.type.selection != "boundary":
            continue
        if f.meta.get("default"):
            if geo is None:
                continue
            ents = [b for b in bnds if len(geo.adj_up["boundary"].get(b, [])) == 1]   # exterior boundaries only
        else:
            ents = [b for b in bnds if b in set(selection_entities(f, sol.meta))]
        for b in ents:
            if f.kind == "acpr.hard":
                out[b] = ("hard",)
            elif f.kind == "acpr.swr":
                ev = lambda key: float(units.evaluate(str(f.get(key, "0")), params))  # noqa: E731
                ctr = (0.0, 0.0, ev("z0")) if axi else (ev("x0"), ev("y0"), ev("z0") if sol.sdim == 3 else 0.0)
                out[b] = ("swr", ctr, 1.0 if (axi or sol.sdim == 3) else 0.5)
            elif f.kind != "acpr.efc":
                out.pop(b, None)                      # other conditions: use the computed gradient
    return out


def incident_variables(model, sol):
    """Result variables for an incident plane wave (total-field scattering): acpr.p_i (background, real part),
    acpr.p_s (scattered, real part), acpr.abs_ps and acpr.Lp_s."""
    comp = model.component
    out = {}
    if comp is None:
        return out
    try:
        params = dict(model.parameters())
    except Exception:
        params = {}
    from .physics import INCIDENT_DIRS
    re_terms, im_terms = [], []
    for ph in comp.children:
        if ph.kind != "physics.acpr" or not ph.enabled:
            continue
        for w in ph.children:
            if w.kind == "acpr.swr" and w.enabled and w.get("incident", "none") == "plane":
                bnds = list(w.selection.entities) if w.selection is not None else []
                c = _sound_speed(model, ph, bnds, None, params)
                d = INCIDENT_DIRS.get(w.get("dir", "+z"), (0, 0, 1))
                try:
                    p0 = float(units.evaluate(str(w.get("p0", "1[Pa]")), params))
                except Exception:
                    p0 = 1.0
                xyz = ("x", "y", "z")
                ph_ = f"(2*pi*freq/{c!r})*(" + "+".join(f"({di})*{xi}" for di, xi in zip(d, xyz) if di) + ")"
                re_terms.append(f"{p0!r}*cos({ph_})")
                im_terms.append(f"(-{p0!r})*sin({ph_})")
    if re_terms:
        out["acpr.p_i"] = "(" + "+".join(re_terms) + ")"
        out["acpr.p_iim"] = "(" + "+".join(im_terms) + ")"
        out["acpr.p_s"] = "(acpr.p_t-acpr.p_i)"
        out["acpr.p_sim"] = "(acpr.p_im-acpr.p_iim)"
        out["acpr.abs_ps"] = "sqrt(acpr.p_s^2+acpr.p_sim^2)"
        out["acpr.Lp_s"] = "(20*log10(max(acpr.abs_ps/sqrt(2),1e-30)/20e-6))"
    return out


INCIDENT_LABELS = [("acpr.p_i", "Background (incident) pressure field, real part", "Pa"),
                   ("acpr.p_s", "Scattered pressure field, real part", "Pa"),
                   ("acpr.abs_ps", "Scattered pressure amplitude", "Pa"),
                   ("acpr.Lp_s", "Scattered sound pressure level", "dB")]
EFC_LABELS = [("Lp_pext", "Exterior-field sound pressure level", "dB"), ("pext", "Exterior-field pressure, real part", "Pa"),
              ("abs_pext", "Exterior-field pressure amplitude", "Pa")]


def acoustic_hmax(model, per_wavelength=5):
    """Physics-controlled mesh for pressure acoustics: c_min / (per_wavelength * f_max) in metres, or None."""
    comp = model.component
    if comp is None or not any(c.kind == "physics.acpr" and c.enabled for c in comp.children):
        return None
    try:
        params = dict(model.parameters())
    except Exception:
        params = {}
    from .sif import parse_list
    fmax = 0.0
    for st in model.studies():
        for s in st.children:
            if s.kind == "study.freq" and s.enabled:
                try:
                    fac = units.unit_info(s.get("funit", "Hz"))[0]
                    fmax = max([fmax] + [f * fac for f in parse_list(s.get("freqs", "500"), params)])
                except Exception:
                    continue
    if fmax <= 0:
        return None
    cs = []
    for ph in comp.children:
        if ph.kind == "physics.acpr" and ph.enabled:
            pam = next((f for f in ph.children if f.kind == "acpr.pam"), None)
            if pam is not None and pam.get("c_src", "mat") == "user":
                try:
                    cs.append(float(units.evaluate(str(pam.get("c", "343")), params)))
                except Exception:
                    pass
    mats = comp.child("materials")
    for m in (mats.children if mats else []):
        val = (m.get("values") or {}).get("c", "") if m.kind == "material" and m.enabled else ""
        if str(val).strip():
            try:
                cs.append(float(units.evaluate(str(val), params)))
            except Exception:
                pass
    c = min(cs) if cs else 343.0
    return c / (per_wavelength * fmax)


def _sound_speed(model, ph, bnds, geo, params):
    """Speed of sound next to the surface: Pressure Acoustics feature (user value) or its material."""
    pam = next((f for f in ph.children if f.kind == "acpr.pam"), None)
    if pam is not None and pam.get("c_src", "mat") == "user":
        try:
            return float(units.evaluate(str(pam.get("c", "343")), params))
        except Exception:
            pass
    mats = model.component.child("materials")
    doms = set()
    if geo is not None:
        for b in bnds:
            doms.update(geo.adj_up["boundary"].get(b, []))
    for m in (mats.children if mats else []):
        if m.kind != "material" or not m.enabled or m.selection is None:
            continue
        if geo is not None and doms and not (set(m.selection.resolve(geo.numbers["domain"])) & doms):
            continue
        val = (m.get("values") or {}).get("c", "")
        if str(val).strip():
            try:
                return float(units.evaluate(str(val), params))
            except Exception:
                continue
    return 343.0


# --------------------------------------------------------------------------- spherical Bessel functions
def spherical_jn(n_max, x):
    """j_0..j_n_max at x > 0 (Miller's downward recurrence, normalized with j_0)."""
    x = float(x)
    start = n_max + int(2 * math.sqrt(40 * (n_max + 1))) + 20
    j = np.zeros(start + 2)
    j[start] = 1e-300
    for n in range(start, 0, -1):
        j[n - 1] = (2 * n + 1) / x * j[n] - j[n + 1]
        if abs(j[n - 1]) > 1e250:
            j /= 1e250
    j *= (math.sin(x) / x) / j[0]
    return j[:n_max + 1]


def spherical_yn(n_max, x):
    """y_0..y_n_max at x > 0 (upward recurrence, stable for y_n)."""
    x = float(x)
    y = np.zeros(n_max + 1)
    y[0] = -math.cos(x) / x
    if n_max >= 1:
        y[1] = -math.cos(x) / x ** 2 - math.sin(x) / x
    for n in range(1, n_max):
        y[n + 1] = (2 * n + 1) / x * y[n] - y[n - 1]
    return y


def legendre(n_max, t):
    t = np.asarray(t, float)
    P = [np.ones_like(t), t.copy()]
    for n in range(1, n_max):
        P.append(((2 * n + 1) * t * P[n] - n * P[n - 1]) / (n + 1))
    return P[:n_max + 1]


def rigid_sphere_scattered(k, a, r, theta, p0=1.0, n_max=None):
    """Scattered pressure of a rigid sphere (radius a) in the plane wave p0 exp(-i k z), exp(+i w t) convention."""
    n_max = n_max or int(k * a + 4 * (k * a) ** (1 / 3) + 10)
    ja, ya = spherical_jn(n_max + 1, k * a), spherical_yn(n_max + 1, k * a)
    r = np.atleast_1d(np.asarray(r, float))
    theta = np.atleast_1d(np.asarray(theta, float))
    Pn = legendre(n_max, np.cos(theta))
    out = np.zeros(np.broadcast(r, theta).shape, complex)
    hr = {}
    for rr in np.unique(r):
        hr[rr] = spherical_jn(n_max, k * rr) - 1j * spherical_yn(n_max, k * rr)
    H = np.array([hr[rr] for rr in np.broadcast_to(r, out.shape).ravel()]).reshape(out.shape + (n_max + 1,))
    for n in range(n_max + 1):
        djn = ja[n - 1] - (n + 1) / (k * a) * ja[n] if n else -ja[1]
        dyn = ya[n - 1] - (n + 1) / (k * a) * ya[n] if n else -ya[1]
        dhn = djn - 1j * dyn
        out += -p0 * (2 * n + 1) * (-1j) ** n * (djn / dhn) * H[..., n] * Pn[n]
    return out
