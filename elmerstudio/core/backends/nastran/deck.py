"""Model -> NASTRAN-95 card deck: refusal rules, planning, and Bulk Data writer.

Supports:
- Solid Mechanics (physics.solid): Linear Statics (SOL 1), Normal Modes (SOL 3)
- Heat Transfer in Solids (physics.ht): Steady Heat Conduction (HEAT 1, fixed temperatures, heat flux, convection)
- Elements: CTRIA3, CQUAD4 (2D, unit depth), CTETRA, CHEXA, CPENTA (3D)
- Properties & Materials: PSHELL, MAT1, MAT4
- BCs & Loads: SPC1, SPC, FORCE (consistent nodal loads of boundary and body loads), GRAV
- Heat flux: CHBDY (LINE, AREA3, AREA4), PHBDY, QBDY1, ambient SPC for convection
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ... import units
from ...builder import physics_nodes, study_steps
from ...model import Model, Node, is_axisymmetric
from ...physics import COUPLINGS, FEATURES, physics_def

SUPPORTED_PHYSICS = {"solid": "Solid Mechanics", "ht": "Heat Transfer in Solids"}
SUPPORTED_FEATURES = {
    "solid": {
        "solid.lemm", "solid.init", "solid.fixed", "solid.disp",
        "solid.roller", "solid.symmetry", "solid.bload", "solid.bodyload",
        "solid.gravity", "solid.free"
    },
    "ht": {
        "ht.solid", "ht.init", "ht.temperature", "ht.insulation", "ht.symmetry",
        "ht.heatflux"
    }
}

STEP_MAP = {
    "study.stationary": 1,
    "study.stat": 1,
    "study.eigen": 3,
    "study.time": 9,          # DISP 9 = direct transient (SOL 109)
}

# Linear NASTRAN cells for each gmsh element type, as (card, node positions in the gmsh row).  A 6-node triangle
# becomes 4 CTRIA3 and a 10-node tetrahedron 8 CTETRA, so every mesh node stays a grid point.  The deck numbers
# its elements through this table and the result converter maps element stresses back through the same table.
TRI6_SPLIT = ((0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5))
TET10_SPLIT = ((0, 4, 6, 7), (4, 1, 5, 8), (6, 5, 2, 9), (7, 8, 9, 3),
               (4, 6, 7, 8), (4, 5, 8, 9), (6, 5, 9, 8), (7, 8, 9, 5))
BULK_CELLS = {
    2: [("CTRIA3", (0, 1, 2))],
    3: [("CQUAD4", (0, 1, 2, 3))],
    9: [("CTRIA3", c) for c in TRI6_SPLIT],
    4: [("CTETRA", (0, 1, 2, 3))],
    11: [("CTETRA", c) for c in TET10_SPLIT],
    5: [("CHEXA", tuple(range(8)))],
    6: [("CPENTA", tuple(range(6)))],
}
# Boundary elements split the same way (a 3-node edge of a 6-node triangle is two 2-node edges of its CTRIA3s).
BOUNDARY_CELLS = {1: ((0, 1),), 8: ((0, 2), (2, 1)), 2: ((0, 1, 2),), 9: TRI6_SPLIT, 3: ((0, 1, 2, 3),)}
# Not CTRIM6 (NASTRAN-95's 6-node membrane): it fails the constant stress patch test on coarse free meshes, with
# vertex stresses 30-50% off in uniform tension, so 6-node triangles are split into constant strain CTRIA3s.  These
# are exact for uniform states and converge like linear triangles in bending (a finer mesh than Elmer's needs).
_GMSH_NAMES = {1: "2-node line", 8: "3-node line", 10: "9-node quadrilateral", 16: "8-node quadrilateral",
               17: "20-node hexahedron", 12: "27-node hexahedron", 18: "15-node prism", 13: "18-node prism",
               7: "pyramid", 14: "pyramid", 19: "pyramid"}


class NastranRefusal(RuntimeError):
    """The model cannot be solved by the NASTRAN-95 backend."""


def _fail(msg: str):
    raise NastranRefusal(msg + " Choose Elmer as the solver backend for this study.")


def check_model(model: Model, study: Node) -> str | None:
    """Check if model/study is supported by NASTRAN-95. Returns refusal reason or None."""
    try:
        _check(model, study)
    except NastranRefusal as exc:
        return str(exc)
    return None


def _check(model: Model, study: Node):
    comps = model.components()
    if not comps:
        _fail("The model has no component.")
    steps = study_steps(study)
    if not steps:
        _fail(f"{study.label} has no study steps.")

    for step in steps:
        if step.kind not in STEP_MAP:
            _fail(f"{step.label} is not a Stationary, Eigenfrequency, or Time Dependent step, the only study steps mapped to NASTRAN-95.")
        if str(step.get("initsol", "") or ""):
            _fail(f"{step.label} takes its initial values from another study's solution.")

    for comp in comps:
        with model.scoped(comp):
            _check_component(model, study, comp)


def _check_component(model: Model, study: Node, comp: Node):
    where = f" ({comp.label})" if len(model.components()) > 1 else ""
    if is_axisymmetric(comp):
        _fail(f"2D axisymmetric components{where} are not currently supported by the NASTRAN-95 backend.")

    steps = study_steps(study)
    for step in steps:
        sel = step.props.get("physics", {})
        active = [ph for ph in physics_nodes(model, comp) if ph.enabled and sel.get(ph.tag, True)]
        if len(active) > 1:
            _fail(f"{step.label} solves {len(active)} physics interfaces together{where} "
                  f"({', '.join(p.label for p in active)}); NASTRAN-95 backend solves one at a time.")
        for ph in active:
            pid = physics_def(ph.kind).id
            if pid not in SUPPORTED_PHYSICS:
                _fail(f"{ph.label} is not supported by the NASTRAN-95 backend (solves Solid Mechanics and Heat Transfer only).")
            if step.kind == "study.time" and pid == "ht":
                _fail(f"{step.label} is a Time Dependent heat transfer step; NASTRAN-95 backend only supports transient for Solid Mechanics.")
            for f in ph.children:
                if not f.enabled or f.kind in SUPPORTED_FEATURES[pid]:
                    continue
                if f.kind in FEATURES and FEATURES[f.kind].section == "post":
                    continue
                if f.kind in ("ht.source", "ht.radiation"):
                    _fail(f"{f.label} ({ph.label}) is not mapped to NASTRAN-95 (heat sources and radiation not supported).")
                _fail(f"{f.label} ({ph.label}) is not mapped to NASTRAN-95.")
            for f in ph.children:
                if f.enabled and f.kind == "solid.gravity" and f.get("rho_src", "mat") != "mat":
                    # GRAV multiplies the acceleration by the density on MAT1, which comes from the material model
                    _fail(f"{f.label} ({ph.label}) uses its own density; NASTRAN-95 applies gravity with the "
                          f"material's density only.")

    mp = comp.child("multiphysics")
    for c in (mp.children if mp is not None and mp.enabled else []):
        if c.enabled and c.kind in COUPLINGS:
            _fail(f"{c.label} is a multiphysics coupling; NASTRAN-95 backend currently solves single physics.")


@dataclass
class NastranPlan:
    physics: str              # solid | ht
    app: str                  # DISP | HEAT
    sol: int                  # 1 | 3 | 109
    title: str
    spc_id: int = 1
    load_id: int = 1
    eigr_id: int = 1
    neig: int = 6
    thickness: float = 1.0    # 2D: per unit depth, as Elmer solves 2D models
    plane_strain: bool = False                              # 2D Solid Mechanics in plane strain (set by write_deck)
    domain_nu: dict = field(default_factory=dict)           # domain -> Poisson's ratio of the model (set by write_deck)
    # Transient fields
    tstep: str | None = None        # TSTEP card reference
    tend: str = "1.0"               # End time
    dt: str = "0.01"                # Time step
    nout: str = "10"                # Output every N steps


def plan_deck(model: Model, study: Node, step: Node) -> NastranPlan:
    """Create a NastranPlan from model, study, and step."""
    comp = model.component
    sel = step.props.get("physics", {})
    active = [ph for ph in physics_nodes(model, comp) if ph.enabled and sel.get(ph.tag, True)]
    ph = active[0] if active else None
    pid = physics_def(ph.kind).id if ph else "solid"

    sol_num = STEP_MAP.get(step.kind, 1)
    app = "HEAT" if pid == "ht" else "DISP"

    plan = NastranPlan(
        physics=pid,
        app=app,
        sol=sol_num,
        title=f"{study.label} - {step.label}",
    )
    if step.kind == "study.eigen":
        plan.neig = int(step.get("neig", 6) or 6)
    if step.kind == "study.time":
        # For transient: SOL 109 (DISP 9 direct transient)
        plan.sol = 109
        plan.app = "DISP"
        # Store time step info from step properties
        plan.tstep = step.get("tstep", None)
        plan.tend = step.get("tend", "1.0")
        plan.dt = step.get("dt", "0.01")
        plan.nout = step.get("nout", "10")
    return plan


def _eval_expr(expr_str: str, model: Model, overrides: dict | None = None) -> float:
    """Evaluate a mathematical expression in model context."""
    if not expr_str:
        return 0.0
    val = units.evaluate(expr_str, model.parameters(overrides))
    return float(np.real(val))


def _fmt8(v: float) -> str:
    """Format a float to fit within NASTRAN's 8-character small field limit.
    
    NASTRAN-95 Bulk Data rules require all real numbers to contain a decimal point.
    """
    v = float(v)
    if v == 0.0 or v == 0:
        return "0.0"
    s = f"{v:.6g}"
    if "e" not in s and "E" not in s:
        if "." not in s:
            s = (s + ".0") if len(s) <= 6 else (s + ".")
        if len(s) <= 8:
            return s
    # Exponential formatting with guaranteed decimal point
    for prec in (3, 2, 1, 0):
        s = f"{v:.{prec}e}".replace("e+0", "+").replace("e-0", "-").replace("e+", "+").replace("e-", "-")
        if "." not in s:
            for sign in ("+", "-"):
                if sign in s:
                    p = s.split(sign, 1)
                    s = f"{p[0]}.{sign}{p[1]}"
                    break
        if len(s) <= 8:
            return s
    return f"{v:.1e}".replace("e+0", "+").replace("e-0", "-")


def _measure_normal(pts: np.ndarray, sdim: int) -> tuple[float, np.ndarray]:
    """Length (2D) or area (3D) of a linear boundary cell and its unit normal (orientation not fixed)."""
    if len(pts) == 2:
        t = pts[1] - pts[0]
        a = float(np.linalg.norm(t))
        return a, np.array([t[1], -t[0], 0.0]) / (a or 1.0)
    c = np.cross(pts[2] - pts[0], pts[3] - pts[1]) if len(pts) == 4 else np.cross(pts[1] - pts[0], pts[2] - pts[0])
    a = 0.5 * float(np.linalg.norm(c))
    return a, c / (2 * a or 1.0)


_TET_OF = {4: ((0, 1, 2, 3),), 5: ((0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 4, 6, 7), (1, 3, 4, 6)),
           6: ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5))}


def _bulk_cells(mesh, nodes: np.ndarray, thickness: float):
    """(node indices, area*thickness or volume, domain) of every linear NASTRAN cell, in element-ID order."""
    for dom, blocks in mesh.bulk.items():
        for gt, conn in blocks:
            for row in conn:
                for card, pos in BULK_CELLS[gt]:
                    ids = [int(row[p]) for p in pos]
                    p = nodes[ids]
                    if card in ("CTRIA3", "CQUAD4"):
                        m = _measure_normal(p, 3)[0] * thickness
                    else:
                        m = sum(abs(np.linalg.det(np.array([p[t[1]] - p[t[0]], p[t[2]] - p[t[0]], p[t[3]] - p[t[0]]])))
                                for t in _TET_OF[len(ids) if len(ids) != 8 else 5]) / 6.0
                    yield ids, m, dom


_FACES = {2: ((0, 1), (1, 2), (2, 0)), 9: ((0, 1), (1, 2), (2, 0)), 3: ((0, 1), (1, 2), (2, 3), (3, 0)),
          4: ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)), 11: ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
          5: ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
          6: ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5))}


def _face_parents(mesh, nodes: np.ndarray, sdim: int) -> dict:
    """Corner node set of every element side -> centroid of the element it belongs to (boundary sides are unique)."""
    out = {}
    for blocks in mesh.bulk.values():
        for gt, conn in blocks:
            ncorner = {2: 3, 9: 3, 3: 4, 4: 4, 11: 4, 5: 8, 6: 6}[gt]
            for row in conn:
                c = nodes[row[:ncorner]].mean(0)
                for face in _FACES[gt]:
                    out[frozenset(int(row[k]) for k in face)] = c
    return out


def write_deck(plan: NastranPlan, mesh, model: Model, out_path: str, overrides: dict | None = None):
    """Generate NASTRAN Bulk Data card deck and write to out_path."""
    # 1. Executive Control Deck
    exec_lines = [
        "NASTRAN TITLEOPT=-1",
        "ID ELMERSTUDIO,NASTRAN95",
        f"APP {plan.app}",
        f"SOL {plan.sol}",
        "TIME 100",
        "CEND",
    ]

    # Grid coordinates: every mesh node, quadratic mid-side nodes included (they are corners of the split cells).
    nodes = np.asarray(mesh.nodes, float) * getattr(mesh, "scale", 1.0)
    sdim = getattr(mesh, "sdim", 3)
    if sdim == 2:
        nodes = nodes.copy()
        nodes[:, 2] = 0.0
    for blocks in mesh.bulk.values():
        for gt, _conn in blocks:
            if gt not in BULK_CELLS:
                raise NastranRefusal(f"The mesh has {_GMSH_NAMES.get(gt, f'type {gt}')} elements, which are not "
                                     f"mapped to NASTRAN-95: use a free triangular (2D) or tetrahedral (3D) mesh.")
    # PS=3456 for 2D solid (plane membrane: no out-of-plane translation, no rotations); omit for HEAT
    # For 3D solid: add PS=456 to constrain rotations 4,5,6 (no stiffness from CTETRA/CHEXA)
    if sdim == 2 and plan.physics == "solid":
        default_spc = "3456"
    elif sdim == 3 and plan.physics == "solid":
        default_spc = "456"
    else:
        default_spc = ""

    grid_cards = []
    for i, pt in enumerate(nodes, start=1):
        x, y, z = pt[0], pt[1], pt[2] if sdim == 3 else 0.0
        if default_spc:
            grid_cards.append(f"GRID,{i},,{_fmt8(x)},{_fmt8(y)},{_fmt8(z)},,{default_spc}")
        else:
            grid_cards.append(f"GRID,{i},,{_fmt8(x)},{_fmt8(y)},{_fmt8(z)}")

    # Materials and Properties per domain
    comp = model.component
    active_ph = next((ph for ph in comp.children if ph.kind.startswith("physics.") and ph.enabled), None)
    # Elmer solves 2D Solid Mechanics in plane strain unless the interface says plane stress.  NASTRAN-95's
    # membrane is plane stress; plane strain is the same in-plane problem with E/(1-nu^2) and nu/(1-nu).
    plan.plane_strain = (sdim == 2 and plan.physics == "solid" and active_ph is not None
                         and active_ph.get("plane", "strain") != "stress")

    mat_cards = []
    prop_cards = []
    domain_mids: dict[int, int] = {}
    mid_counter = 1

    for dom_id in mesh.bulk.keys():
        mid = mid_counter
        mid_counter += 1
        domain_mids[dom_id] = mid

        E = 2.0e11
        nu = 0.3
        rho = 7850.0
        k_therm = 50.0
        cp_therm = 450.0

        mats = comp.child("materials") if comp else None
        mat_vals: dict[str, str] = {}
        if mats:
            for mn in mats.children:
                if mn.enabled and mn.kind == "material":
                    sel = mn.selection
                    if sel is None or sel.all or dom_id in sel.entities:
                        mat_vals = mn.get("values") or {}
                        break

        if active_ph:
            for child in active_ph.children:
                if child.enabled and child.kind in ("solid.lemm", "ht.solid"):
                    sel = child.selection
                    if sel is None or sel.all or dom_id in sel.entities:
                        try:
                            params = model.parameters(overrides)
                            if child.kind == "solid.lemm":
                                E_src = child.get("E_src", "mat")
                                nu_src = child.get("nu_src", "mat")
                                rho_src = child.get("rho_src", "mat")

                                E_str = child.get("E", "2.0e11[Pa]") if E_src == "user" else mat_vals.get("E", "2.0e11")
                                nu_str = child.get("nu", "0.3") if nu_src == "user" else mat_vals.get("nu", "0.3")
                                rho_str = child.get("rho", "7850[kg/m^3]") if rho_src == "user" else mat_vals.get("rho", "7850")

                                E = float(np.real(units.evaluate(str(E_str), params)))
                                nu = float(np.real(units.evaluate(str(nu_str), params)))
                                rho = float(np.real(units.evaluate(str(rho_str), params)))
                            elif child.kind == "ht.solid":
                                k_src = child.get("k_src", "mat")
                                rho_src = child.get("rho_src", "mat")
                                cp_src = child.get("Cp_src", "mat")

                                k_str = child.get("k", "50[W/(m*K)]") if k_src == "user" else mat_vals.get("k", "50")
                                rho_str = child.get("rho", "7850[kg/m^3]") if rho_src == "user" else mat_vals.get("rho", "7850")
                                cp_str = child.get("Cp", "450[J/(kg*K)]") if cp_src == "user" else mat_vals.get("Cp", "450")

                                k_therm = float(np.real(units.evaluate(str(k_str), params)))
                                rho = float(np.real(units.evaluate(str(rho_str), params)))
                                cp_therm = float(np.real(units.evaluate(str(cp_str), params)))
                        except Exception:
                            pass

        if plan.physics == "solid":
            plan.domain_nu[dom_id] = nu
            if plan.plane_strain:
                E, nu = E / (1 - nu * nu), nu / (1 - nu)
            mat_cards.append(f"MAT1,{mid},{_fmt8(E)},,{_fmt8(nu)},{_fmt8(rho)}")
            if sdim == 2:
                prop_cards.append(f"PSHELL,{mid},{mid},{_fmt8(plan.thickness)},{mid},1.0,{mid}")
            # 3D solid elements (CTETRA, CHEXA, CPENTA) reference MID directly, no PSOLID
        else:
            mat_cards.append(f"MAT4,{mid},{_fmt8(k_therm)},{_fmt8(cp_therm)},{_fmt8(rho)}")
            if sdim == 2:
                prop_cards.append(f"PSHELL,{mid},{mid},{_fmt8(plan.thickness)}")
            # 3D thermal solid elements (CTETRA, CHEXA, CPENTA) reference MID directly, no PSOLID

    elem_cards = []
    eid = 1
    for dom_id, blocks in mesh.bulk.items():
        pid = domain_mids.get(dom_id, 1)          # PSHELL ID for 2D elements, MAT ID for 3D solids (same number)
        for gt, conn in blocks:
            for row in conn + 1:                  # 1-based grid IDs
                for card, pos in BULK_CELLS[gt]:
                    elem_cards.append(f"{card},{eid},{pid}," + ",".join(str(row[p]) for p in pos))
                    eid += 1

    spc_cards: list[str] = []
    load_cards: list[str] = []
    forces: dict[int, np.ndarray] = {}           # grid ID -> nodal force vector (N)

    def add_force(nid: int, vec):
        forces[nid] = forces.get(nid, np.zeros(3)) + vec

    bnd_nodes: dict[int, set[int]] = {}
    for bnd_id, blocks in mesh.bnd.items():
        s = set()
        for _gt, conn in blocks:
            s.update((conn.ravel() + 1).tolist())
        bnd_nodes[bnd_id] = s

    def selected(f, level):
        sel = f.selection
        pool = mesh.bnd if level == "boundary" else mesh.bulk
        if sel is None:
            return []
        return sorted(pool) if sel.all else [e for e in sel.entities if e in pool]

    def boundary_cells(ents):
        """(grid IDs, area, unit normal, parent element key) of the linear boundary cells."""
        out = []
        for b in ents:
            for gt, conn in mesh.bnd.get(b, []):
                if gt not in BOUNDARY_CELLS:
                    raise NastranRefusal(f"Boundary {b} has {_GMSH_NAMES.get(gt, f'type {gt}')} elements, which are "
                                         f"not mapped to NASTRAN-95.")
                for row in conn:
                    corners = frozenset(int(x) for x in row[: (2 if gt in (1, 8) else 4 if gt == 3 else 3)])
                    for pos in BOUNDARY_CELLS[gt]:
                        ids = [int(row[p]) for p in pos]
                        a, n = _measure_normal(nodes[ids], sdim)
                        out.append(([i + 1 for i in ids], a * (plan.thickness if sdim == 2 else 1.0), n, corners))
        return out

    parent = None

    def outward(cells):
        """Turn each normal away from the bulk element that owns the boundary element."""
        nonlocal parent
        if parent is None:
            parent = _face_parents(mesh, nodes, sdim)
        for ids, a, n, corners in cells:
            c = parent.get(corners)
            if c is not None and np.dot(n, nodes[[i - 1 for i in ids]].mean(0) - c) < 0:
                n = -n
            yield ids, a, n

    def axis_dof(f, ents) -> str:
        """The DOF normal to a boundary that is perpendicular to the x, y or z axis (a roller there is an SPC)."""
        normals = np.array([np.abs(cell[2]) for cell in boundary_cells(ents)])
        for k in range(sdim):
            if len(normals) and np.all(normals[:, k] > 1 - 1e-6):
                return str(k + 1)
        raise NastranRefusal(f"{f.label} is on a boundary that is not perpendicular to the x, y or z axis; "
                             f"NASTRAN-95 constrains displacements in the global axes only.")

    # Heat flux tracking: collect all ht.heatflux features first
    ht_flux_features = []
    if active_ph and plan.physics == "ht":
        for f in active_ph.children:
            if f.enabled and f.kind == "ht.heatflux":
                ht_flux_features.append(f)

    # CHBDY/PHBDY/QBDY1/MAT4 tracking for heat flux
    chbdy_eid = 1
    chbdy_cards: list[str] = []
    phbdy_cards: list[str] = []
    qbdy1_cards: list[str] = []
    mat4_conv_cards: list[str] = []  # MAT4 for convection (film coefficient)
    ambient_grid_map: dict[int, int] = {}  # boundary grid ID -> ambient grid ID
    next_ambient_grid = len(nodes) + 1
    chbdy_to_mat4: dict[int, int] = {}  # CHBDY EID -> MAT4 MID

    # Process heat flux features
    for f in ht_flux_features:
        if not f.enabled:
            continue
        flux_type = f.get("type", "general")
        bents = selected(f, "boundary")
        cells = boundary_cells(bents)
        for ids, area, normal in outward(cells):
            # Create CHBDY element for this boundary cell
            if sdim == 2:
                # 2D: LINE element (2 nodes)
                if len(ids) != 2:
                    continue
                chbdy_type = "LINE"
            else:
                # 3D: AREA3 (3 nodes) or AREA4 (4 nodes)
                if len(ids) == 3:
                    chbdy_type = "AREA3"
                elif len(ids) == 4:
                    chbdy_type = "AREA4"
                else:
                    continue
            
            if flux_type == "convective":
                # Convection: CHBDY with ambient points (GA1, GA2...), PHBDY referencing MAT4 with film coefficient
                # Create ambient grid points for each node of the CHBDY
                ambient_ids = []
                for gid in ids:
                    if gid not in ambient_grid_map:
                        ambient_grid_map[gid] = next_ambient_grid
                        next_ambient_grid += 1
                    ambient_ids.append(ambient_grid_map[gid])
                
                # Create MAT4 for convection if not already created for this feature
                # Use a unique MID per feature for convection
                mat4_mid = 2000 + len(mat4_conv_cards)
                h = _eval_expr(f.get("h", "10"), model, overrides)
                # MAT4: MID, K (film coefficient), CP (thermal capacity per unit area, 0 for steady)
                mat4_conv_cards.append(f"MAT4,{mat4_mid},{_fmt8(h)},0.0")
                chbdy_to_mat4[chbdy_eid] = mat4_mid
                
                # CHBDY with ambient points
                ga_str = ",".join(str(aid) for aid in ambient_ids)
                chbdy_cards.append(f"CHBDY,{chbdy_eid},,{chbdy_type}," + 
                                   ",".join(str(gid) for gid in ids) + 
                                   "," + ga_str + ",")
                # PHBDY references the MAT4 MID for convection
                phbdy_cards.append(f"PHBDY,{chbdy_eid},{mat4_mid},,,,,,")
            else:
                # General heat flux: CHBDY + PHBDY + QBDY1
                chbdy_cards.append(f"CHBDY,{chbdy_eid},,{chbdy_type}," + 
                                   ",".join(str(gid) for gid in ids) + ",")
                phbdy_cards.append(f"PHBDY,{chbdy_eid},,,,,,,")
                # QBDY1 for this CHBDY
                q0 = _eval_expr(f.get("q0", "0"), model, overrides)
                qbdy1_cards.append(f"QBDY1,{plan.load_id},{_fmt8(q0)},{chbdy_eid}")
            
            chbdy_eid += 1

    # Solid mechanics boundary conditions
    if active_ph:
        for f in active_ph.children:
            if not f.enabled:
                continue
            bents = selected(f, "boundary")
            target_nodes: set[int] = set()
            for ent in bents:
                target_nodes.update(bnd_nodes[ent])
            if f.selection is not None and not f.selection.all:
                for ent in f.selection.entities:
                    if ent in getattr(mesh, "point_nodes", {}) and ent not in mesh.bnd:
                        target_nodes.add(mesh.point_nodes[ent] + 1)
            node_list = sorted(target_nodes)
            chunks = [node_list[i:i + 7] for i in range(0, len(node_list), 7)]

            if f.kind == "solid.fixed":
                dofs = "12" if sdim == 2 else "123"
                spc_cards += [f"SPC1,{plan.spc_id},{dofs}," + ",".join(map(str, c)) for c in chunks]

            elif f.kind == "solid.disp":
                for dof_idx, c in enumerate("xyz"[:sdim], start=1):
                    if f.get(f"on{c}", c == "x"):
                        val = _eval_expr(f.get(f"u{c}", "0"), model, overrides)
                        spc_cards += [f"SPC,{plan.spc_id},{n},{dof_idx},{_fmt8(val)}" for n in node_list]

            elif f.kind in ("solid.roller", "solid.symmetry") and node_list:
                dof = axis_dof(f, bents)
                spc_cards += [f"SPC1,{plan.spc_id},{dof}," + ",".join(map(str, c)) for c in chunks]

            elif f.kind == "solid.bload" and bents:
                cells = boundary_cells(bents)
                t = f.get("type", "force")
                if t == "pressure":
                    p = _eval_expr(f.get("p", "0"), model, overrides)
                    for ids, a, n in outward(cells):          # Elmer's Normal Force -p: p pushes into the body
                        for i in ids:
                            add_force(i, -p * n * a / len(ids))
                else:
                    if t == "total":
                        area = sum(cell[1] for cell in cells)
                        if not area > 0:
                            raise NastranRefusal(f"{f.label}: total force needs a boundary selection with nonzero area")
                        trac = np.array([_eval_expr(f.get(f"Ft{c}", "0"), model, overrides) / area
                                         if k < sdim else 0.0 for k, c in enumerate("xyz")])
                    else:
                        trac = np.array([_eval_expr(f.get(f"F{c}", "0"), model, overrides) if k < sdim else 0.0
                                         for k, c in enumerate("xyz")])
                    for ids, a, _n, _c in cells:
                        for i in ids:
                            add_force(i, trac * a / len(ids))

            elif f.kind == "solid.bodyload":
                fv = np.array([_eval_expr(f.get(f"F{c}", "0"), model, overrides) if k < sdim else 0.0
                               for k, c in enumerate("xyz")])
                doms = set(selected(f, "domain"))
                for row, vol, dom in _bulk_cells(mesh, nodes, plan.thickness):
                    if dom in doms:
                        for i in row:
                            add_force(i + 1, fv * vol / len(row))

            elif f.kind == "solid.gravity":
                doms = set(selected(f, "domain"))
                if doms != set(mesh.bulk):
                    raise NastranRefusal(f"{f.label} acts on part of the model; NASTRAN-95's gravity load acts on all "
                                         f"of it.")
                g = np.array([_eval_expr(f.get(f"g{c}", "0"), model, overrides) if k < sdim else 0.0
                              for k, c in enumerate("xyz")])
                g_mag = float(np.linalg.norm(g))
                if g_mag > 1e-30:
                    load_cards.append(f"GRAV,{plan.load_id},0,{_fmt8(g_mag)},{_fmt8(g[0] / g_mag)},"
                                      f"{_fmt8(g[1] / g_mag)},{_fmt8(g[2] / g_mag)}")

            elif f.kind == "ht.temperature":
                T_val = _eval_expr(f.get("T0", "293.15[K]"), model, overrides)
                spc_cards += [f"SPC,{plan.spc_id},{n},1,{_fmt8(T_val)}" for n in node_list]

    for nid in sorted(forces):
        vec = forces[nid]
        mag = float(np.linalg.norm(vec))
        if mag > 1e-30:
            nx, ny, nz = vec / mag
            load_cards.append(f"FORCE,{plan.load_id},{nid},0,{_fmt8(mag)},{_fmt8(nx)},{_fmt8(ny)},{_fmt8(nz)}")

    case_lines = [
        f"TITLE = {plan.title}",
        "ECHO = NONE",
        "MAXLINES = 99999999",          # the default (20000 printed lines) ends larger runs before the results
    ]
    if plan.physics == "ht":
        # NASTRAN-95 HEAT 1: THERMAL = ALL requests temperature output
        case_lines.append("THERMAL = ALL")
    else:
        case_lines.append("DISPLACEMENT = ALL")
        case_lines.append("STRESS = ALL")
        case_lines.append("ELFORCE = ALL")
        case_lines.append("SPCFORCE = ALL")
    if spc_cards:
        case_lines.append(f"SPC = {plan.spc_id}")
    if load_cards and plan.sol != 3:
        case_lines.append(f"LOAD = {plan.load_id}")
    if plan.sol == 3:
        case_lines.append(f"METHOD = {plan.eigr_id}")
    if plan.sol == 109:
        # Transient direct: need TSTEP for time integration
        case_lines.append(f"TSTEP = {plan.spc_id}")  # reuse spc_id as tstep_id
        case_lines.append(f"DLOAD = {plan.load_id}")

    bulk_lines = ["BEGIN BULK"]
    bulk_lines.extend(grid_cards)
    bulk_lines.extend(mat_cards)
    bulk_lines.extend(prop_cards)
    bulk_lines.extend(elem_cards)
    bulk_lines.extend(spc_cards)
    bulk_lines.extend(load_cards)
    bulk_lines.extend(chbdy_cards)
    bulk_lines.extend(phbdy_cards)
    bulk_lines.extend(qbdy1_cards)
    bulk_lines.extend(mat4_conv_cards)

    # Ambient grid points for convection: create GRID and SPC for each
    for gid, ambient_id in ambient_grid_map.items():
        # Place ambient point at same location as the boundary grid point
        pt = nodes[gid - 1]
        x, y, z = pt[0], pt[1], pt[2] if sdim == 3 else 0.0
        bulk_lines.append(f"GRID,{ambient_id},,{_fmt8(x)},{_fmt8(y)},{_fmt8(z)}")
        # SPC to set Text temperature - use the Text from the first ht.heatflux feature
        T_ext = 293.15
        for f in ht_flux_features:
            if f.get("type", "general") == "convective":
                T_ext = _eval_expr(f.get("Text", "293.15[K]"), model, overrides)
                break
        bulk_lines.append(f"SPC,{plan.spc_id},{ambient_id},1,{_fmt8(T_ext)}")

    if plan.sol == 3:
        # EIGR: SID, METHOD, F1, F2, NE, ND, NZ, E, +abc.  FEER (tridiagonal reduction) about F1 = 0 gives the
        # lowest ND modes, as Elmer's eigen solver does, without a frequency window (INV) and without the dense
        # Givens method, whose time estimate exceeds any limit beyond a few hundred grid points.
        bulk_lines.append(f"EIGR,{plan.eigr_id},FEER,0.0,,,{plan.neig},,,+EIG1")
        bulk_lines.append("+EIG1,MASS")
    elif plan.sol == 109:
        # Transient direct (SOL 109, DISP 9)
        # TSTEP: SID, N1, DT1, NO1, N2, DT2, NO2...
        # Use uniform time step from plan.dt, output every plan.nout steps
        tend = _eval_expr(plan.tend, model, overrides)
        dt = _eval_expr(plan.dt, model, overrides)
        nsteps = max(1, int(tend / dt + 0.5))
        nout = int(plan.nout) if plan.nout else 1
        bulk_lines.append(f"TSTEP,{plan.spc_id},{nsteps},{_fmt8(dt)},{nout}")
        
        # TABLED1 for step load (constant from t=0): TABLED1, TID, X, Y...
        # Step function: 0 at t<0, 1 at t>=0
        tabled1_id = 999
        bulk_lines.append(f"TABLED1,{tabled1_id},0.0,0.0,{_fmt8(dt)},1.0,{_fmt8(tend)},1.0")
        
        # TLOAD1: SID, L (DAREA), M (DELAY), TF (TABLED1)
        bulk_lines.append(f"TLOAD1,{plan.load_id},{plan.load_id},0,{tabled1_id}")
        
        # DAREA: SID, EID, COMP, VALUE... for each grid point force component
        # Convert FORCE cards to DAREA entries
        # DAREA format: DAREA, SID, EID, C1, V1, C2, V2...
        # Group forces by grid point and component
        force_by_grid = defaultdict(lambda: defaultdict(float))
        for nid, vec in forces.items():
            for comp_idx, val in enumerate(vec, start=1):
                if abs(val) > 1e-30:
                    force_by_grid[nid][comp_idx] += val
        
        # Also include SPC forces (non-zero prescribed displacements) as DAREA
        # For simplicity, we only handle FORCE loads here; SPC loads are enforced via SPC cards
        
        dareas_by_grid = defaultdict(list)
        for gid, comps in force_by_grid.items():
            for comp, val in comps.items():
                dareas_by_grid[gid].append((comp, val))
        
        if dareas_by_grid:
            dare_area_id = plan.load_id  # reuse load_id for DAREA set
            for gid, comps in dareas_by_grid.items():
                line = f"DAREA,{dare_area_id},{gid}"
                for comp, val in comps:
                    line += f",{comp},{_fmt8(val)}"
                bulk_lines.append(line)

    bulk_lines.append("ENDDATA")

    lines = exec_lines + case_lines + bulk_lines
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="\n") as f:
        f.write(text)
    return text