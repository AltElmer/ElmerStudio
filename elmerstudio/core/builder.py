"""Model-construction API (the scripting interface the GUI also uses).

Example::

    m = new_model("3D", ["ht"], "stationary")
    geom = m.component.child("geometry")
    blk = m.create("geom.block", geom, index=-1, w="0.1", d="0.02", h="0.01")
"""
from __future__ import annotations

from . import materials as _materials  # noqa: F401  (registers node types)
from . import nodes as _nodes  # noqa: F401
from . import geometry as _geometry  # noqa: F401
from . import meshing as _meshing  # noqa: F401
from .model import Model, Node, node_type
from .physics import BUNDLES, COUPLINGS, PHYSICS, STUDY_TYPES, physics_def


def _component_children(node):
    return []


def new_model(sdim: str | None = "3D", physics: list[str] | None = None, study: str | None = "stationary") -> Model:
    m = Model()
    root = m.root
    m.create("global", root, tag="global")
    m.create("params", m.global_defs, tag="param", label="Parameters 1")
    if sdim:
        add_component(m, sdim)
        for pid in physics or []:
            if pid.startswith("@"):
                add_bundle(m, pid)
            else:
                add_physics(m, pid)
    m.create("results", root, tag="results")
    res = m.results
    m.create("results.datasets", res, tag="dsets")
    m.create("results.derived", res, tag="dvals")
    m.create("results.tables", res, tag="tbls")
    m.create("results.export", res, tag="exps")
    if sdim and study:
        add_study(m, study)
    return m


def add_component(m: Model, sdim: str) -> Node:
    idx = 1 if m.global_defs is not None else 0
    comp = m.create("component", m.root, index=idx, tag=m.new_tag("comp"), label=None)
    comp.props["sdim"] = sdim
    comp.label = f"Component {comp.tag[4:]}"
    m.create("definitions", comp, tag=m.new_tag("defs") if m.by_tag("defs") else "defs")
    geom = m.create("geometry", comp, tag=m.new_tag("geom"))
    geom.label = f"Geometry {geom.tag[4:]}"
    m.create("geom.finalize", geom, tag="fin")
    m.create("materials", comp, tag=m.new_tag("matlnk") if m.by_tag("materials") else "materials")
    mesh = m.create("mesh", comp, tag=m.new_tag("mesh"))
    mesh.label = f"Mesh {mesh.tag[4:]}"
    return comp


def _insert_index(comp: Node) -> int:
    """Physics go after Materials and existing physics, before Multiphysics/Mesh."""
    for i, c in enumerate(comp.children):
        if c.kind in ("multiphysics", "mesh"):
            return i
    return len(comp.children)


def add_physics(m: Model, pid: str, comp: Node | None = None) -> Node:
    comp = comp or m.component
    pd = PHYSICS[pid]
    existing = [c for c in comp.children if c.kind == pd.kind]
    tag = pd.id if not existing and not m.by_tag(pd.id) else m.new_tag(pd.id)
    label = pd.name if not existing else f"{pd.name} {len(existing) + 1}"
    ph = m.create(pd.kind, comp, index=_insert_index(comp), tag=tag, label=label)
    for kind, level in pd.defaults:
        f = m.create(kind, ph)
        f.meta["default"] = True
        if f.selection is not None:
            f.selection.all = True
    # physics table entries in existing studies
    for st in m.studies():
        for step in st.children:
            if step.kind.startswith("study.") and "physics" in step.props:
                step.props["physics"][ph.tag] = True
    return ph


def add_feature(m: Model, ph: Node, kind: str, entities=None, **props) -> Node:
    f = m.create(kind, ph)
    f.props.update(props)
    if entities is not None and f.selection is not None:
        f.selection.entities = list(entities)
        f.selection.all = False
    return f


def multiphysics_node(m: Model, comp: Node | None = None, create=True) -> Node | None:
    comp = comp or m.component
    mp = comp.child("multiphysics")
    if mp is None and create:
        idx = next((i for i, c in enumerate(comp.children) if c.kind == "mesh"), len(comp.children))
        mp = m.create("multiphysics", comp, index=idx, tag="mp" if not m.by_tag("mp") else m.new_tag("mp"))
    return mp


def add_coupling(m: Model, kind: str, comp: Node | None = None) -> Node:
    comp = comp or m.component
    mp = multiphysics_node(m, comp)
    c = m.create(kind, mp)
    return c


def add_bundle(m: Model, key: str):
    title, _icon, pids, couplings = BUNDLES[key]
    comp = m.component
    for pid in pids:
        if not any(c.kind == f"physics.{pid}" for c in comp.children):
            add_physics(m, pid)
    for k in couplings:
        add_coupling(m, k)


def physics_nodes(m: Model, comp: Node | None = None) -> list[Node]:
    comp = comp or m.component
    if comp is None:
        return []
    return [c for c in comp.children if c.kind.startswith("physics.")]


def add_study(m: Model, step_kind: str = "stationary", label: str | None = None) -> Node:
    idx = next((i for i, c in enumerate(m.root.children) if c.kind == "results"), len(m.root.children))
    st = m.create("study", m.root, index=idx, tag=m.new_tag("std"))
    st.label = label or f"Study {st.tag[3:]}"
    add_step(m, st, step_kind)
    return st


def add_step(m: Model, st: Node, step_kind: str) -> Node:
    kind = step_kind if step_kind.startswith("study.") else f"study.{step_kind}"
    idx = next((i for i, c in enumerate(st.children) if c.kind == "study.solvers"), len(st.children))
    step = m.create(kind, st, index=idx)
    step.label = f"Step {sum(1 for c in st.children if c.kind in ('study.stationary', 'study.time', 'study.eigen', 'study.freq'))}: {node_type(kind).title}"
    step.props["physics"] = {ph.tag: True for ph in physics_nodes(m)}
    return step


def study_steps(st: Node) -> list[Node]:
    return [c for c in st.children if c.kind in ("study.stationary", "study.time", "study.eigen", "study.freq") and c.enabled]


def study_sweep(st: Node) -> Node | None:
    for c in st.children:
        if c.kind == "study.sweep" and c.enabled:
            return c
    return None


def ensure_solver_configs(m: Model, st: Node) -> Node:
    sols = st.child("study.solvers")
    if sols is None:
        sols = m.create("study.solvers", st, tag=m.new_tag("sols"))
    have = {c.props.get("physics") for c in sols.children}
    for ph in physics_nodes(m):
        if ph.tag not in have:
            s = m.create("study.solver", sols)
            s.props["physics"] = ph.tag
            s.label = f"Solver: {ph.label} ({ph.tag})"
            pd = physics_def(ph.kind)
            if pd.id in ("ht", "tds", "pde", "es", "ec"):
                s.props["lin"] = "auto"
    return sols


def add_material(m: Model, name: str, domains=None, all_domains=False) -> Node:
    return _materials.make_material(m, m.component.child("materials"), name, domains, all_domains)


def solution_dataset(m: Model, st: Node, create=True) -> Node | None:
    dsets = m.results.child("results.datasets")
    for d in dsets.children:
        if d.kind == "dset.solution" and d.props.get("study") == st.tag:
            return d
    if not create:
        return None
    d = m.create("dset.solution", dsets, tag=m.new_tag("dset"))
    d.props["study"] = st.tag
    d.label = f"{st.label}/Solution {d.tag[4:]}"
    return d


def default_plots(m: Model, st: Node, step_kind: str) -> list[Node]:
    """Create COMSOL-style default plot groups for the solved physics (if not present)."""
    comp = m.component
    from .model import space_dim
    sd = space_dim(comp)
    ds = solution_dataset(m, st)
    made = []
    existing = {n.meta.get("default_plot") for n in m.results.children}
    for ph in physics_nodes(m):
        step = study_steps(st)[0] if study_steps(st) else None
        if step is not None and not step.props.get("physics", {}).get(ph.tag, True):
            continue
        pd = physics_def(ph.kind)
        if not pd.plots:
            continue
        for spec in pd.plots(ph, sd, step_kind):
            key = f"{st.tag}:{ph.tag}:{spec['label']}"
            if key in existing:
                continue
            pg_kind = "pg3d" if sd == 3 else "pg2d"
            idx = next((i for i, c in enumerate(m.results.children) if c.kind == "results.export"), len(m.results.children))
            pg =m.create(pg_kind, m.results, index=idx, label=spec["label"])
            pg.props["data"] = ds.tag
            pg.meta["default_plot"] = key
            if step_kind == "eigen":
                pg.props["solnum"] = "0"     # COMSOL shows the first eigenmode by default
            t = spec["type"]
            kind = {"surface": "plot.surface", "volume": "plot.volume", "slice": "plot.slice",
                    "isosurface": "plot.isosurface", "contour": "plot.contour"}[t]
            if sd == 2 and kind in ("plot.volume", "plot.slice", "plot.isosurface"):
                kind = "plot.surface" if kind != "plot.isosurface" else "plot.contour"
            f = m.create(kind, pg)
            f.props["expr"] = spec["expr"]
            f.props["cmap"] = spec.get("cmap", "RainbowLight")
            if spec.get("deform"):
                d = m.create("plot.deform", f)
            if spec.get("contour") and sd == 2:
                c = m.create("plot.contour", pg)
                c.props["expr"] = spec["contour"]
                c.props["levels"] = 20
                c.props["colorleg"] = False
                c.props["cmap"] = "GrayScale"
            made.append(pg)
    return made
