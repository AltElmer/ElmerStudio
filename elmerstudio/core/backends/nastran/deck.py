"""Model -> NASTRAN-95 card deck: refusal rules, planning, and Bulk Data writer.

Supports:
- Solid Mechanics (physics.solid): Linear Statics (SOL 1), Normal Modes (SOL 3), Transient (SOL 9)
- Heat Transfer in Solids (physics.ht): Steady Heat Conduction (HEAT 1)
- Elements: CTRIA3, CQUAD4, CTETRA, CHEXA, CPENTA, CROD
- Properties & Materials: PSHELL, PSOLID, PROD, MAT1, MAT4
- BCs & Loads: SPC1, SPC, FORCE, GRAV
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from ... import units
from ...builder import physics_nodes, step_components, study_steps
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
        "ht.solid", "ht.init", "ht.temperature", "ht.heatflux", "ht.insulation", "ht.symmetry"
    }
}

STEP_MAP = {
    "study.stationary": 1,
    "study.stat": 1,
    "study.eigen": 3,
    "study.freq": 8,
    "study.time": 9,
    "study.trans": 9,
}


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
            _fail(f"{step.label} has step type {step.kind}, which is not mapped to NASTRAN-95.")
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
            for f in ph.children:
                if not f.enabled or f.kind in SUPPORTED_FEATURES[pid]:
                    continue
                if f.kind in FEATURES and FEATURES[f.kind].section == "post":
                    continue
                _fail(f"{f.label} ({ph.label}) is not mapped to NASTRAN-95.")

    mp = comp.child("multiphysics")
    for c in (mp.children if mp is not None and mp.enabled else []):
        if c.enabled and c.kind in COUPLINGS:
            _fail(f"{c.label} is a multiphysics coupling; NASTRAN-95 backend currently solves single physics.")


@dataclass
class NastranPlan:
    physics: str              # solid | ht
    app: str                  # DISP | HEAT
    sol: int                  # 1 | 3 | 8 | 9
    title: str
    spc_id: int = 1
    load_id: int = 1
    eigr_id: int = 1
    neig: int = 6
    thickness: float = 0.01   # default 2D shell thickness
    transient: bool = False
    dt: float = 0.01
    nsteps: int = 100


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
    elif step.kind == "study.time":
        plan.transient = True
        plan.dt = float(step.get("step", 0.01) or 0.01)
        plan.nsteps = int(step.get("nsteps", 100) or 100)
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

    # Grid coordinates – keep ALL nodes (quadratic mid-sides are used by CTRIM6).
    nodes = np.asarray(mesh.nodes, float) * getattr(mesh, "scale", 1.0)
    sdim = getattr(mesh, "sdim", 3)
    # PS=3456 for 2D solid (constrains 1-2-3-4-5-6 in plane); omit for HEAT
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
        pid = domain_mids.get(dom_id, 1)          # property ID for 2D elements (PSHELL/PTRIM6/PROD)
        mid = domain_mids.get(dom_id, 1)          # material ID for 3D elements (CTETRA/CHEXA/CPENTA)
        for gt, conn in blocks:
            if gt == 9:
                # 6-node quadratic triangle
                # Elmer/gmsh order: corners [n1,n2,n3], mids [n4(1-2),n5(2-3),n6(3-1)]
                # CTRIM6 needs: corner, mid, corner, mid, corner, mid (counter-clockwise)
                for row in conn:
                    n = row + 1  # 1-based
                    # Split each 6-node quadratic triangle into 4 linear CTRIA3
                    # sub-triangles. Keeps ALL nodes in the deck; no renumbering or
                    # reverse mapping needed. CTRIA3 is a shell element (PSHELL) so
                    # it captures both membrane and bending behaviour.
                    elem_cards.append(f"CTRIA3,{eid},{pid},{n[0]},{n[3]},{n[5]}")
                    eid += 1
                    elem_cards.append(f"CTRIA3,{eid},{pid},{n[3]},{n[1]},{n[4]}")
                    eid += 1
                    elem_cards.append(f"CTRIA3,{eid},{pid},{n[5]},{n[4]},{n[2]}")
                    eid += 1
                    elem_cards.append(f"CTRIA3,{eid},{pid},{n[3]},{n[4]},{n[5]}")
                    eid += 1
            elif gt == 10:              # quadratic quad -> CQUAD4 (corners only)
                for row in conn:
                    n = row + 1
                    elem_cards.append(f"CQUAD4,{eid},{pid},{n[0]},{n[1]},{n[2]},{n[3]}")
                    eid += 1
            elif gt == 11:              # quadratic tetra (10-node) -> 8 linear CTETRA
                # gmsh/Elmer order: corners [0,1,2,3], mids [4(0-1),5(1-2),6(2-0),7(0-3),8(1-3),9(2-3)]
                # Split into 8 linear CTETRA keeping all 10 nodes
                for row in conn:
                    n = row + 1  # 1-based
                    # 4 corner tets (each corner + its 3 adjacent mids)
                    elem_cards.append(f"CTETRA,{eid},{mid},{n[0]},{n[4]},{n[6]},{n[7]}")
                    eid += 1
                    elem_cards.append(f"CTETRA,{eid},{mid},{n[4]},{n[1]},{n[5]},{n[8]}")
                    eid += 1
                    elem_cards.append(f"CTETRA,{eid},{mid},{n[6]},{n[5]},{n[2]},{n[9]}")
                    eid += 1
                    elem_cards.append(f"CTETRA,{eid},{mid},{n[7]},{n[8]},{n[9]},{n[3]}")
                    eid += 1
                    # 4 interior tets (partition the central octahedron of mid-edge nodes)
                    elem_cards.append(f"CTETRA,{eid},{mid},{n[4]},{n[6]},{n[7]},{n[8]}")
                    eid += 1
                    elem_cards.append(f"CTETRA,{eid},{mid},{n[4]},{n[5]},{n[8]},{n[9]}")
                    eid += 1
                    elem_cards.append(f"CTETRA,{eid},{mid},{n[6]},{n[5]},{n[9]},{n[8]}")
                    eid += 1
                    elem_cards.append(f"CTETRA,{eid},{mid},{n[7]},{n[8]},{n[9]},{n[5]}")
                    eid += 1
            elif gt == 17:              # quadratic hexa -> CHEXA (corners only)
                for row in conn:
                    n = row + 1
                    elem_cards.append(f"CHEXA,{eid},{mid},{n[0]},{n[1]},{n[2]},{n[3]},{n[4]},{n[5]},{n[6]},{n[7]}")
                    eid += 1
            else:
                c = conn + 1
                if gt == 1:
                    for row in c:
                        elem_cards.append(f"CROD,{eid},{pid},{row[0]},{row[1]}")
                        eid += 1
                elif gt == 2:
                    for row in c:
                        elem_cards.append(f"CTRIA3,{eid},{pid},{row[0]},{row[1]},{row[2]}")
                        eid += 1
                elif gt in (3, 16):
                    for row in c:
                        elem_cards.append(f"CQUAD4,{eid},{pid},{row[0]},{row[1]},{row[2]},{row[3]}")
                        eid += 1
                elif gt == 4:
                    for row in c:
                        elem_cards.append(f"CTETRA,{eid},{mid},{row[0]},{row[1]},{row[2]},{row[3]}")
                        eid += 1
                elif gt == 5:
                    for row in c:
                        elem_cards.append(f"CHEXA,{eid},{mid},{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},{row[7]}")
                        eid += 1
                elif gt == 6:
                    for row in c:
                        elem_cards.append(f"CPENTA,{eid},{mid},{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}")
                        eid += 1

    spc_cards: list[str] = []
    load_cards: list[str] = []

    bnd_nodes: dict[int, set[int]] = {}
    for bnd_id, blocks in mesh.bnd.items():
        s = set()
        for _gt, conn in blocks:
            s.update((conn.ravel() + 1).tolist())
        bnd_nodes[bnd_id] = s

    if active_ph:
        for f in active_ph.children:
            if not f.enabled:
                continue
            sel = f.selection
            target_nodes: set[int] = set()
            if sel is not None:
                if sel.all:
                    for s in bnd_nodes.values():
                        target_nodes.update(s)
                else:
                    for ent in sel.entities:
                        if ent in bnd_nodes:
                            target_nodes.update(bnd_nodes[ent])
                        elif ent in getattr(mesh, "point_nodes", {}):
                            target_nodes.add(mesh.point_nodes[ent] + 1)

            if not target_nodes:
                continue

            if f.kind == "solid.fixed":
                # 3D solid elements (CTETRA/CHEXA/CPENTA) only have 3 translational DOFs
                dofs = "12" if sdim == 2 else "123"
                node_list = sorted(target_nodes)
                for chunk in [node_list[i:i + 7] for i in range(0, len(node_list), 7)]:
                    spc_cards.append(f"SPC1,{plan.spc_id},{dofs}," + ",".join(str(n) for n in chunk))

            elif f.kind == "solid.disp":
                for n in sorted(target_nodes):
                    for dof_idx, dof_name in enumerate(["u0", "v0", "w0"], start=1):
                        if dof_idx > sdim:
                            break
                        if f.get(f"{dof_name}_active", True):
                            val = _eval_expr(f.get(dof_name, "0"), model, overrides)
                            spc_cards.append(f"SPC,{plan.spc_id},{n},{dof_idx},{_fmt8(val)}")

            elif f.kind == "solid.roller":
                node_list = sorted(target_nodes)
                for chunk in [node_list[i:i + 7] for i in range(0, len(node_list), 7)]:
                    spc_cards.append(f"SPC1,{plan.spc_id},2," + ",".join(str(n) for n in chunk))

            elif f.kind == "solid.bload":
                n_count = len(target_nodes)
                if n_count > 0:
                    Fx = _eval_expr(f.get("Fx", "0"), model, overrides) / n_count
                    Fy = _eval_expr(f.get("Fy", "0"), model, overrides) / n_count
                    Fz = _eval_expr(f.get("Fz", "0"), model, overrides) if sdim == 3 else 0.0
                    mag = math.sqrt(Fx ** 2 + Fy ** 2 + Fz ** 2)
                    if mag > 1e-30:
                        nx, ny, nz = Fx / mag, Fy / mag, Fz / mag
                        for n in sorted(target_nodes):
                            load_cards.append(f"FORCE,{plan.load_id},{n},0,{_fmt8(mag)},{_fmt8(nx)},{_fmt8(ny)},{_fmt8(nz)}")

            elif f.kind in ("solid.gravity", "solid.bodyload"):
                gx = _eval_expr(f.get("gx", "0"), model, overrides)
                gy = _eval_expr(f.get("gy", "-9.81"), model, overrides)
                gz = _eval_expr(f.get("gz", "0"), model, overrides) if sdim == 3 else 0.0
                g_mag = math.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
                if g_mag > 1e-30:
                    load_cards.append(f"GRAV,{plan.load_id},0,{_fmt8(g_mag)},{_fmt8(gx/g_mag)},{_fmt8(gy/g_mag)},{_fmt8(gz/g_mag)}")

            elif f.kind == "ht.temperature":
                T_val = _eval_expr(f.get("T0", "293.15[K]"), model, overrides)
                for n in sorted(target_nodes):
                    spc_cards.append(f"SPC,{plan.spc_id},{n},1,{_fmt8(T_val)}")

    case_lines = [
        f"TITLE = {plan.title}",
        "ECHO = NONE",
    ]
    if plan.physics == "ht":
        # NASTRAN-95 HEAT 1: THERMAL = ALL requests temperature output
        case_lines.append("THERMAL = ALL")
    else:
        case_lines.append("DISPLACEMENT = ALL")
        case_lines.append("STRESS = ALL")
        case_lines.append("ELFORCE = ALL")
    if spc_cards:
        case_lines.append(f"SPC = {plan.spc_id}")
    if load_cards and plan.sol != 3:
        case_lines.append(f"LOAD = {plan.load_id}")
    if plan.sol == 3:
        case_lines.append(f"METHOD = {plan.eigr_id}")

    bulk_lines = ["BEGIN BULK"]
    bulk_lines.extend(grid_cards)
    bulk_lines.extend(mat_cards)
    bulk_lines.extend(prop_cards)
    bulk_lines.extend(elem_cards)
    bulk_lines.extend(spc_cards)
    bulk_lines.extend(load_cards)

    if plan.sol == 3:
        # EIGR: SID, METHOD, F1, F2, NE, ND, NZ, E, +abc
        # For 3D: use INV (inverse power) instead of GIV (dense Givens) to avoid huge time estimates
        # FEER exceeded MAXLINES; INV is iterative and efficient for few modes
        # For 2D: GIV with ND works fine
        if sdim == 3:
            # INV: F1=lower freq bound, F2=upper freq bound, NE=est roots in range, ND=num vectors
            bulk_lines.append(f"EIGR,{plan.eigr_id},INV,0.0,1000.0,{plan.neig},{plan.neig},,,+EIG1")
        else:
            # GIV: set ND (field 7) = number of roots to print; F1/F2 ignored when ND > 0
            bulk_lines.append(f"EIGR,{plan.eigr_id},GIV,,,{plan.neig},{plan.neig},,,+EIG1")
        bulk_lines.append("+EIG1,MASS")

    bulk_lines.append("ENDDATA")

    lines = exec_lines + case_lines + bulk_lines
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="\n") as f:
        f.write(text)
    return text
