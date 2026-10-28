"""Probes (COMSOL: Definitions > Probes): a quantity watched at every output time or parameter value of a study.

Domain and Boundary Probes reduce an expression over their selection (average, maximum, minimum or integral), a
Point Probe reads it at a point and a Global Variable Probe evaluates a scalar such as aveop1(T).  After a solve,
and while a time-dependent study runs, every probe of the computed component fills a column of the study's Probe
Table, which a Probe Plot Group draws against time (or frequency, or the swept parameter).
"""
from __future__ import annotations

import math

import numpy as np

from . import units
from .model import NodeType, Prop, register

PROBE_KINDS = ("probe.domain", "probe.boundary", "probe.point", "probe.global")
_TYPES = [("avg", "Average"), ("max", "Maximum"), ("min", "Minimum"), ("int", "Integral")]
_EXPR = [Prop("expr", "Expression:", "rexpr", "", section="Expression"),
         Prop("unit", "Unit:", "runit", "", section="Expression"),
         Prop("descr", "Description:", "string", "", section="Expression")]

register(NodeType("probe.domain", "Domain Probe", "dom", "average", selection="domain", selection_all=True,
                  description="The average, maximum, minimum or integral of an expression over the selected domains, "
                              "at every output time or parameter value: in the Probe Table and Probe Plot.",
                  props=[Prop("ptype", "Type:", "choice", "avg", choices=_TYPES, section="Probe Settings"), *_EXPR]))
register(NodeType("probe.boundary", "Boundary Probe", "bnd", "average", selection="boundary", selection_all=True,
                  description="The average, maximum, minimum or integral of an expression over the selected "
                              "boundaries, at every output time or parameter value.",
                  props=[Prop("ptype", "Type:", "choice", "avg", choices=_TYPES, section="Probe Settings"), *_EXPR]))
register(NodeType("probe.point", "Domain Point Probe", "ppb", "point_evaluation",
                  description="An expression at a point of the geometry, at every output time or parameter value.",
                  props=[Prop("x", "x:", default="0", unit="LEN", section="Point Selection"),
                         Prop("y", "y:", default="0", unit="LEN", section="Point Selection"),
                         Prop("z", "z:", default="0", unit="LEN", section="Point Selection",
                              visible=lambda p: p.get("__sdim", 3) == 3), *_EXPR]))
register(NodeType("probe.global", "Global Variable Probe", "var", "global_evaluation",
                  description="A scalar expression (a coupling operator such as aveop1(T), a parameter expression) at "
                              "every output time or parameter value.", props=[*_EXPR]))


def probes(model, comp=None) -> list:
    """The enabled probes of a component (the active one by default)."""
    comp = comp or model.component
    defs = comp.child("definitions") if comp is not None else None
    return [n for n in (defs.children if defs else []) if n.kind in PROBE_KINDS and n.enabled]


def _measure_unit(si: str, dim: int) -> str:
    m = {3: "m^3", 2: "m^2", 1: "m"}.get(dim, "")
    if not si or si == "1":
        return m
    return f"{si}*{m}" if m else si


def column(node, sol) -> str:
    """Probe Table column header: the probe's label, the quantity and its unit."""
    expr = str(node.get("expr", "")).strip()
    label, si = sol.describe(expr) if expr else ("", "")
    unit = str(node.get("unit", "")).strip() or si
    if node.kind in ("probe.domain", "probe.boundary") and node.get("ptype", "avg") == "int" \
            and not str(node.get("unit", "")).strip():
        dim = (sol.sdim + (1 if sol.meta.get("axi") else 0)) - (0 if node.kind == "probe.domain" else 1)
        unit = _measure_unit(si, dim)
    what = str(node.get("descr", "")).strip() or label or expr
    head = f"{node.label}: {what}"
    return f"{head} ({unit})" if unit else head


def _point(node, sol, model):
    """The point in SI (its coordinates are in the geometry's length unit)."""
    vals = []
    for key in ("x", "y", "z")[:max(2, sol.sdim)]:
        try:
            v = float(np.real(units.evaluate(str(node.get(key, "0")) or "0", model.parameters())))
        except Exception as exc:
            raise ValueError(f"{node.label}: {key} = {node.get(key)!r}: {exc}") from None
        vals.append(v)
    comp = node.ancestor("component")
    geom = comp.child("geometry") if comp is not None else None
    fac = units.unit_info(geom.get("unit", "m"))[0] if geom is not None else 1.0
    return [v * fac for v in vals]


def value(node, sol, fi: int, model, geo=None) -> float:
    """One probe at one frame (SI unless the probe names a unit)."""
    expr = str(node.get("expr", "")).strip()
    if not expr:
        raise ValueError(f"{node.label}: no expression")
    axi = bool(sol.meta.get("axi"))
    if node.kind in ("probe.domain", "probe.boundary"):
        level = "domain" if node.kind == "probe.domain" else "boundary"
        ents = None
        if node.selection is not None and not node.selection.all:
            ents = list(node.selection.entities)
            if not ents:
                raise ValueError(f"{node.label}: nothing selected")
        t = node.get("ptype", "avg")
        if t in ("avg", "int"):
            v = sol.integrate(expr, fi, level, ents, average=(t == "avg"), revolved=axi)
        else:
            v = sol.maxmin_locations(expr, fi, level, ents)[t]
    elif node.kind == "probe.point":
        v = float(sol.point_eval(expr, fi, [_point(node, sol, model)])[0])
    else:
        v = float(np.nanmean(sol.evaluate(expr, fi)))
    unit = str(node.get("unit", "")).strip()
    if unit and not (node.kind in ("probe.domain", "probe.boundary") and node.get("ptype") == "int"):
        _l, si = sol.describe(expr)
        if unit != si:
            v = float(units.from_si(v, unit))
    return float(v)


def lead(sol) -> str:
    return {"study.time": f"Time ({sol.meta.get('tunit', 's')})", "study.freq": "freq (Hz)",
            "study.eigen": "eigfreq (Hz)"}.get(sol.kind, "Solution")


def table(model, solutions, frames=None) -> tuple[list[str], list[list]]:
    """Probe Table of a study: [(sweep value or None, Solution)] -> columns, rows.  A sweep gives one row per
    parameter value (its last output), a time-dependent or frequency study one row per output."""
    nodes = probes(model)
    if not nodes or not solutions:
        return [], []
    first = solutions[0][1]
    sweep = len(solutions) > 1 or solutions[0][0] is not None
    cols = [str(first.meta.get("sweep_param") or "Parameter value")] if sweep else [lead(first)]
    cols += [column(n, first) for n in nodes]
    rows = []
    for val, sol in solutions:
        idx = [sol.frame_index("last")] if sweep else (frames if frames is not None else range(len(sol.frames)))
        for fi in idx:
            if fi < 0:
                continue
            row = [float(val) if sweep and val is not None else float(sol.frames[fi].value)]
            for n in nodes:
                try:
                    row.append(value(n, sol, fi, model))
                except Exception:
                    row.append(float("nan"))
            rows.append(row)
    return cols, rows


def finite(rows) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for r in rows for x in r)
