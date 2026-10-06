"""Model tree: nodes, node-type registry, selections and (de)serialization.

The tree mirrors the COMSOL Model Builder::

    root
      Global Definitions -> Parameters 1, ...
      Component 1 (comp1)
        Definitions, Geometry 1, Materials, <physics>..., Multiphysics, Mesh 1
      Study 1 -> steps, Solver Configurations
      Results -> Datasets, Derived Values, Tables, plot groups, Export
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from . import units

LEVELS = ("domain", "boundary", "edge", "point")


# --------------------------------------------------------------------------- property schema
@dataclass
class Prop:
    key: str
    label: str
    kind: str = "expr"              # expr | string | choice | bool | int | vector | table | objects | text | file | color | info | matprops | physics_table
    default: Any = ""
    unit: str = ""                  # SI unit shown next to the field
    choices: list = field(default_factory=list)   # [(value, label)]
    section: str = "General"
    visible: Callable[[dict], bool] | None = None
    symbol: str = ""                # e.g. "k" shown left of the field
    tip: str = ""
    n: int = 3                      # vector length
    readonly: bool = False
    columns: list = field(default_factory=list)  # table columns


@dataclass
class NodeType:
    kind: str
    title: str                      # type name shown in the Settings header
    prefix: str                     # tag prefix, e.g. "blk"
    icon: str = "physics"
    props: list[Prop] = field(default_factory=list)
    selection: str | None = None    # domain | boundary | edge | point | None
    selection_all: bool = False     # default "All <level>s"
    selection_locked: bool = False  # default features: selection not editable
    children: Callable[["Node"], list[str]] | list[str] | None = None
    deletable: bool = True
    can_disable: bool = True
    renamable: bool = True
    equation: str | None = None     # mathtext shown in an Equation section
    numbered: bool = True           # label gets " 1", " 2"...
    label: str | None = None        # base label (defaults to title)
    group: str = ""                 # menu grouping hint
    sections: list[str] = field(default_factory=list)
    description: str = ""
    dims: tuple = (1, 2, 3)         # spatial dims where available

    def base_label(self):
        return self.label or self.title

    def prop(self, key) -> Prop | None:
        for p in self.props:
            if p.key == key:
                return p
        return None

    def child_kinds(self, node: "Node") -> list[str]:
        if self.children is None:
            return []
        if callable(self.children):
            return list(self.children(node))
        return list(self.children)


REGISTRY: dict[str, NodeType] = {}


def register(nt: NodeType) -> NodeType:
    REGISTRY[nt.kind] = nt
    return nt


def node_type(kind: str) -> NodeType:
    try:
        return REGISTRY[kind]
    except KeyError:
        raise KeyError(f"Unknown node kind '{kind}'") from None


# --------------------------------------------------------------------------- selection
@dataclass
class Selection:
    level: str = "domain"
    entities: list[int] = field(default_factory=list)
    all: bool = False

    def resolve(self, available: Iterable[int]) -> list[int]:
        avail = sorted(set(available))
        if self.all:
            return avail
        s = set(avail)
        return [e for e in self.entities if e in s]

    def to_dict(self):
        return {"level": self.level, "entities": list(self.entities), "all": self.all}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("level", "domain"), [int(x) for x in d.get("entities", [])], bool(d.get("all", False)))


# --------------------------------------------------------------------------- node
class Node:
    def __init__(self, kind: str, tag: str = "", label: str = "", props: dict | None = None):
        self.kind = kind
        self.tag = tag
        self.label = label
        self.props: dict[str, Any] = {}
        self.children: list[Node] = []
        self.parent: Node | None = None
        self.enabled = True
        self.selection: Selection | None = None
        self.meta: dict[str, Any] = {}
        nt = REGISTRY.get(kind)
        if nt:
            for p in nt.props:
                if p.kind != "info":
                    self.props[p.key] = copy.deepcopy(p.default)
            if nt.selection:
                self.selection = Selection(nt.selection, [], nt.selection_all)
        if props:
            self.props.update(props)

    # -- tree
    @property
    def type(self) -> NodeType:
        return node_type(self.kind)

    def add(self, child: "Node", index: int | None = None) -> "Node":
        child.parent = self
        if index is None:
            self.children.append(child)
        else:
            self.children.insert(index, child)
        return child

    def remove(self, child: "Node"):
        self.children.remove(child)
        child.parent = None

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def find(self, kind: str | None = None, tag: str | None = None) -> "Node | None":
        for n in self.walk():
            if (kind is None or n.kind == kind) and (tag is None or n.tag == tag):
                return n
        return None

    def find_all(self, pred) -> list["Node"]:
        return [n for n in self.walk() if pred(n)]

    def child(self, kind: str) -> "Node | None":
        for c in self.children:
            if c.kind == kind:
                return c
        return None

    def ancestor(self, kind_prefix: str) -> "Node | None":
        n = self.parent
        while n is not None:
            if n.kind.startswith(kind_prefix):
                return n
            n = n.parent
        return None

    def path(self) -> list["Node"]:
        out = []
        n = self
        while n is not None:
            out.append(n)
            n = n.parent
        return out[::-1]

    def is_active(self) -> bool:
        return all(n.enabled for n in self.path())

    def get(self, key, default=None):
        return self.props.get(key, default)

    def __repr__(self):
        return f"<Node {self.kind} {self.tag!r} {self.label!r}>"

    # -- serialization
    def to_dict(self) -> dict:
        d = {"kind": self.kind, "tag": self.tag, "label": self.label, "props": self.props}
        if not self.enabled:
            d["enabled"] = False
        if self.selection is not None:
            d["selection"] = self.selection.to_dict()
        if self.meta:
            d["meta"] = self.meta
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        n = cls(d["kind"], d.get("tag", ""), d.get("label", ""))
        n.props.update(copy.deepcopy(d.get("props", {})))
        n.enabled = d.get("enabled", True)
        if "selection" in d:
            n.selection = Selection.from_dict(d["selection"])
        n.meta = copy.deepcopy(d.get("meta", {}))
        for c in d.get("children", []):
            n.add(cls.from_dict(c))
        return n


# --------------------------------------------------------------------------- model
class Model:
    """Owns the node tree and the bookkeeping for tags and labels."""

    def __init__(self):
        self.root = Node("root", "root", "Untitled")
        self.file_path: str | None = None
        self.listeners: list[Callable[[str, Any], None]] = []

    # -- notifications (the GUI subscribes)
    def notify(self, what: str, obj=None):
        for cb in list(self.listeners):
            cb(what, obj)

    # -- unique tags/labels
    def _used_tags(self) -> set[str]:
        return {n.tag for n in self.root.walk()}

    def new_tag(self, prefix: str) -> str:
        used = self._used_tags()
        i = 1
        while f"{prefix}{i}" in used:
            i += 1
        return f"{prefix}{i}"

    def new_label(self, base: str, parent: Node | None = None) -> str:
        """Numbered label unique among siblings (COMSOL numbers features per parent, e.g. per interface)."""
        scope = parent.children if parent is not None else list(self.root.walk())
        labels = {n.label for n in scope}
        i = 1
        while f"{base} {i}" in labels:
            i += 1
        return f"{base} {i}"

    def create(self, kind: str, parent: Node, index: int | None = None, label: str | None = None,
               tag: str | None = None, **props) -> Node:
        nt = node_type(kind)
        n = Node(kind)
        n.tag = tag or self.new_tag(nt.prefix)
        if label:
            n.label = label
        elif nt.numbered:
            n.label = self.new_label(nt.base_label(), parent)
        else:
            n.label = nt.base_label()
        n.props.update(props)
        parent.add(n, index)
        return n

    # -- typed accessors
    @property
    def global_defs(self) -> Node:
        return self.root.child("global")

    def components(self) -> list[Node]:
        return [c for c in self.root.children if c.kind == "component"]

    @property
    def component(self) -> Node | None:
        comps = self.components()
        return comps[0] if comps else None

    def studies(self) -> list[Node]:
        return [c for c in self.root.children if c.kind == "study"]

    @property
    def results(self) -> Node | None:
        return self.root.child("results")

    def by_tag(self, tag: str) -> Node | None:
        for n in self.root.walk():
            if n.tag == tag:
                return n
        return None

    # -- parameters
    def parameter_rows(self) -> list[tuple[str, str]]:
        rows = []
        g = self.global_defs
        if g is None:
            return rows
        for n in g.children:
            if n.kind == "params" and n.enabled:
                for r in n.props.get("table", []):
                    name = str(r.get("name", "")).strip()
                    if name and re.match(r"^[A-Za-z_]\w*$", name):
                        rows.append((name, str(r.get("expr", ""))))
        return rows

    def parameters(self, overrides: dict | None = None) -> dict:
        vals, _ = units.evaluate_parameters(self.parameter_rows(), overrides)
        return vals

    def eval(self, expr, overrides=None, default=None):
        try:
            v = units.evaluate(expr, self.parameters(overrides))
            return float(v.real if isinstance(v, complex) else v)
        except Exception:
            if default is not None:
                return default
            raise

    # -- io
    def to_dict(self) -> dict:
        from .. import __version__
        return {"format": "elmerstudio-model", "version": __version__, "root": self.root.to_dict()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        m = cls()
        m.root = Node.from_dict(d["root"])
        return m

    def snapshot(self) -> str:
        return json.dumps(self.root.to_dict())

    def restore(self, snap: str):
        self.root = Node.from_dict(json.loads(snap))


# --------------------------------------------------------------------------- helpers
def space_dim(comp: Node | None) -> int:
    if comp is None:
        return 3
    return {"3D": 3, "2D": 2, "2Daxi": 2, "1D": 1}.get(comp.get("sdim", "3D"), 3)


def is_axisymmetric(comp: Node | None) -> bool:
    return comp is not None and comp.get("sdim") == "2Daxi"


def level_dim(level: str, sdim: int) -> int:
    return {"domain": sdim, "boundary": sdim - 1, "edge": 1, "point": 0}[level]


def level_name(level: str, plural=False) -> str:
    return {"domain": "Domain", "boundary": "Boundary", "edge": "Edge", "point": "Point"}[level] + \
        (("s" if level != "boundary" else "") if plural else "")


def level_plural(level: str) -> str:
    return {"domain": "domains", "boundary": "boundaries", "edge": "edges", "point": "points"}[level]
