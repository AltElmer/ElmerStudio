"""The model's dependency graph: which nodes a node uses, and which use it.

Edges come from what the settings name: parameters, variables, functions and coupling operators in expressions;
another component's operators (comp1.aveop1(T)); geometry objects and selections; the materials a physics feature
reads; the physics, mesh and other studies a study needs; the dataset, table or plot group a results node shows.
The GUI uses it to say what a node is used by, to warn before deleting something in use, and to mark results whose
study is out of date; the runner orders components and studies with the same edges.
"""
from __future__ import annotations

import re

from .model import Model, Node

_NAME = re.compile(r"(?<![\w.])([A-Za-z_]\w*)(?:\.(\w+))?")


def _strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _strings(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _strings(x)


def _index(model: Model) -> dict:
    """Name -> node for everything an expression can name, per component (key None: global)."""
    idx: dict = {None: {}}
    g = model.global_defs
    for n in (g.children if g else []):
        if not n.enabled:
            continue
        if n.kind == "params":
            for r in n.props.get("table") or []:
                if str(r.get("name", "")).strip():
                    idx[None][str(r["name"]).strip()] = n
        elif n.kind.startswith("func."):
            idx[None][str(n.get("fname", n.tag)).strip()] = n
    for c in model.components():
        d = idx.setdefault(c.tag, {})
        defs = c.child("definitions")
        for n in (defs.children if defs else []):
            if n.kind == "variables":
                for r in n.props.get("table") or []:
                    if str(r.get("name", "")).strip():
                        d[str(r["name"]).strip()] = n
            elif n.kind.startswith("cpl."):
                d[str(n.get("opname") or n.tag).strip()] = n
    return idx


def uses(model: Model, node: Node, idx: dict | None = None) -> list[Node]:
    """The nodes *node* depends on directly."""
    idx = idx if idx is not None else _index(model)
    comp = model.component_of(node)
    scope = {**idx[None], **(idx.get(comp.tag, {}) if comp is not None else {})}
    comps = {c.tag: c for c in model.components()}
    out: list[Node] = []

    def add(n):
        if n is not None and n is not node and n not in out:
            out.append(n)
    k = node.kind
    for key, v in node.props.items():
        if key in ("physics", "label"):
            continue
        for s in _strings(v):
            for m in _NAME.finditer(s):
                name, attr = m.group(1), m.group(2)
                if name in comps and attr:                       # another component's operator
                    add(idx.get(name, {}).get(attr) or comps[name])
                elif name in scope:
                    add(scope[name])
    if node.selection is not None and node.selection.named:
        add(model.by_tag(node.selection.named.split(":")[0]))
    if k.startswith("geom."):
        for key in ("input", "tools"):
            for t in node.props.get(key) or []:
                add(model.by_tag(t))
    if comp is not None and node.parent is not None and node.parent.kind.startswith("physics.") \
            and any(str(v) == "mat" for kk, v in node.props.items() if kk.endswith("_src")):
        add(comp.child("materials"))
    if k.startswith("study.") and isinstance(node.props.get("physics"), dict):
        for t, on in node.props["physics"].items():
            if on:
                add(model.by_tag(t))
    if k == "study":
        for s in node.children:
            for d in uses(model, s, idx):
                add(d)
        for c in model.components():
            if any(d.ancestor("component") is c or d is c for d in out):
                add(c.child("mesh"))
    for key in ("initsol", "study", "data", "table", "source"):
        t = node.props.get(key)
        if isinstance(t, str) and t and t != "fromparent":
            add(model.by_tag(t))
    if node.props.get("data") == "fromparent" or (k.startswith("plot.") and "data" not in node.props):
        add(node.ancestor("pg"))
    return out


def used_by(model: Model, node: Node) -> list[Node]:
    """The nodes that depend directly on *node*."""
    idx = _index(model)
    return [n for n in model.root.walk() if n is not node and node in uses(model, n, idx)]


def upstream(model: Model, node: Node, kind_prefix: str = "") -> list[Node]:
    """Everything *node* depends on, directly or not (optionally only kinds starting with *kind_prefix*)."""
    idx = _index(model)
    seen, todo, out = set(), [node], []
    while todo:
        n = todo.pop()
        for d in uses(model, n, idx):
            if id(d) in seen:
                continue
            seen.add(id(d))
            todo.append(d)
            if d.kind.startswith(kind_prefix):
                out.append(d)
    return out


def studies_of(model: Model, node: Node) -> list[Node]:
    """The studies whose solutions a results node shows."""
    return [d for d in upstream(model, node, "study") if d.kind == "study"]
