"""Model checks that do not need a solve: units of every expression (COMSOL's orange unit warnings).

The unit of a value is deduced from its expression (units in brackets, parameters, variables, the dependent
variables and coordinates) and compared with the unit of the setting. As in COMSOL the value is used anyway (its SI
number), so a mismatch is a warning, not an error: 2930[K] typed as a density is 2930 kg/m^3.
"""
from __future__ import annotations

from . import units
from .materials import MATPROPS
from .model import REGISTRY, Model, Node

_COORD = {"x": "m", "y": "m", "z": "m", "r": "m", "t": "s", "freq": "Hz", "omega": "1/s"}


def symbol_dims(model: Model, comp: Node | None = None) -> dict:
    """{name: (Dim or None, plain-number flag)} of the names an expression of *comp* can use."""
    from .physics import PHYSICS
    out: dict = {k: (units.unit_dims(u), False) for k, u in _COORD.items()}
    try:
        out.update(units.parameter_dims(model.parameter_rows()))
    except Exception:  # noqa: BLE001
        pass
    comps = [comp] if comp is not None else model.components()
    for c in comps:
        for ph in c.children:
            pd = next((d for d in PHYSICS.values() if d.kind == ph.kind), None)
            if pd is None:
                continue
            for sym, (_var, _lbl, u) in pd.depvars.items():
                out.setdefault(sym, (None if pd.id == "pde" else units.unit_dims(u), False))
        defs = c.child("definitions")
        rows = [(str(r.get("name", "")).strip(), str(r.get("expr", "0")))
                for n in (defs.children if defs else []) if n.kind == "variables" and n.enabled
                for r in (n.props.get("table") or []) if str(r.get("name", "")).strip()]
        for _ in range(len(rows) + 1):                     # variables may use each other, in any order
            changed = False
            for name, e in rows:
                try:
                    d, pure, _w = units.expr_dims(e, out, _operator_dims(c))
                except units.ExprError:
                    d, pure = None, False
                if out.get(name) != (d, pure):
                    out[name] = (d, pure)
                    changed = True
            if not changed:
                break
    return out


def _operator_dims(comp: Node | None) -> dict:
    """Coupling operators: an average, maximum or minimum keeps its argument's unit; an integral's is unknown here."""
    ops = {}
    defs = comp.child("definitions") if comp is not None else None
    for n in (defs.children if defs else []):
        if n.kind.startswith("cpl.") and n.enabled:
            name = (n.get("opname") or n.tag).strip()
            keep = n.kind != "cpl.integration"
            ops[name] = (lambda args, keep=keep: args[0].dim if keep and args else None)
    return ops


def node_unit_warnings(model: Model, node: Node, dims: dict | None = None) -> list[str]:
    """Unit messages of one node's settings: '<setting>: <message>'."""
    nt = REGISTRY.get(node.kind)
    if nt is None:
        return []
    comp = node.ancestor("component") if node.kind != "component" else node
    dims = dims if dims is not None else symbol_dims(model, comp)
    ops = _operator_dims(comp)
    out = []
    for p in nt.props:
        if p.kind == "expr" and (p.unit or p.unit == "") and p.key in node.props:
            val = str(node.props.get(p.key, "")).strip()
            if not val:
                continue
            for w in units.unit_warnings(val, p.unit if p.unit else None, dims, ops):
                out.append(f"{p.label.rstrip(': ') or p.key}: {w}")
        elif p.kind == "table":
            for r in node.props.get(p.key) or []:
                e = str(r.get("expr", "")).strip()
                if e:
                    for w in units.unit_warnings(e, None, dims, ops):
                        out.append(f"{r.get('name', '?')}: {w}")
        elif p.kind == "matprops":
            for k, e in (node.props.get("values") or {}).items():
                if k in MATPROPS and str(e).strip():
                    for w in units.unit_warnings(str(e), MATPROPS[k][2], dims, ops):
                        out.append(f"{MATPROPS[k][0]}: {w}")
    return out


def unit_report(model: Model) -> dict[str, list[str]]:
    """{node tag: [unit messages]} for the whole model."""
    out: dict[str, list[str]] = {}
    cache: dict = {}
    for n in model.root.walk():
        comp = n.ancestor("component")
        key = comp.tag if comp is not None else ""
        if key not in cache:
            cache[key] = symbol_dims(model, comp)
        w = node_unit_warnings(model, n, cache[key])
        if w:
            out[n.tag] = w
    return out
