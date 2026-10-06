"""Geometry sequence: COMSOL-like feature nodes built with the gmsh OpenCASCADE kernel.

Each feature produces a named geometry object (``blk1``, ``uni1`` ...). The finalizing
``Form Union`` fragments everything into a conforming cell complex whose domains,
boundaries, edges and points are numbered by position, as COMSOL does.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import units
from .gmsh_session import session
from .model import Model, Node, NodeType, Prop, register, space_dim

LENGTH_UNITS = [("m", "m"), ("mm", "mm"), ("cm", "cm"), ("um", "µm"), ("nm", "nm"), ("km", "km"),
                ("in", "in"), ("ft", "ft")]
OCC_PAD = 1e-7
OCC_UNIT = {"m": "M", "mm": "MM", "cm": "CM", "um": "UM", "nm": "NM", "km": "KM", "in": "IN", "ft": "FT"}

_axis_choices = [("x", "x-axis"), ("y", "y-axis"), ("z", "z-axis"), ("cartesian", "Cartesian")]
_base_choices = [("corner", "Corner"), ("center", "Center")]


def _pos3(section="Position"):
    return [Prop("x", "x:", default="0", unit="LEN", section=section),
            Prop("y", "y:", default="0", unit="LEN", section=section),
            Prop("z", "z:", default="0", unit="LEN", section=section)]


def _pos2(section="Position"):
    return [Prop("x", "x:", default="0", unit="LEN", section=section),
            Prop("y", "y:", default="0", unit="LEN", section=section)]


def _axis():
    return [Prop("axis", "Axis type:", "choice", "z", choices=_axis_choices, section="Axis"),
            Prop("ax", "x:", default="0", section="Axis", visible=lambda p: p.get("axis") == "cartesian"),
            Prop("ay", "y:", default="0", section="Axis", visible=lambda p: p.get("axis") == "cartesian"),
            Prop("az", "z:", default="1", section="Axis", visible=lambda p: p.get("axis") == "cartesian")]


def _rot(section="Rotation Angle"):
    return [Prop("rot", "Rotation:", default="0", unit="deg", section=section)]


def _keep():
    return [Prop("keep", "Keep input objects", "bool", False, section="Input")]


def _layers_note():
    return []


GEOM3D = ["geom.block", "geom.sphere", "geom.cylinder", "geom.cone", "geom.torus", "geom.ellipsoid",
          "geom.workplane", "geom.extrude", "geom.revolve", "geom.union", "geom.difference",
          "geom.intersection", "geom.move", "geom.rotate", "geom.scale", "geom.mirror", "geom.array",
          "geom.copy", "geom.import", "geom.fillet3d", "geom.chamfer3d"]
GEOM2D = ["geom.rectangle", "geom.circle", "geom.ellipse", "geom.polygon", "geom.union", "geom.difference",
          "geom.intersection", "geom.move", "geom.rotate", "geom.scale", "geom.mirror", "geom.array",
          "geom.copy", "geom.import"]
WP_KINDS = ["geom.rectangle", "geom.circle", "geom.ellipse", "geom.polygon", "geom.union", "geom.difference",
            "geom.intersection", "geom.move", "geom.rotate", "geom.copy", "geom.array"]


def _geom_children(node: Node):
    comp = node.ancestor("component")
    sd = space_dim(comp)
    return GEOM3D if sd == 3 else GEOM2D


register(NodeType("geometry", "Geometry", "geom", "geometry", label="Geometry", numbered=True,
                  deletable=False, can_disable=False, children=_geom_children,
                  props=[Prop("unit", "Length unit:", "choice", "m", choices=LENGTH_UNITS, section="Units"),
                         Prop("angunit", "Angular unit:", "choice", "deg", choices=[("deg", "Degrees"), ("rad", "Radians")],
                              section="Units"),
                         Prop("repair_tol", "Default repair tolerance:", "choice", "auto",
                              choices=[("auto", "Automatic"), ("relative", "Relative")], section="Advanced")]))

# ---- 3D primitives
register(NodeType("geom.block", "Block", "blk", "block", dims=(3,), props=[
    Prop("w", "Width:", default="1", unit="LEN", section="Size and Shape"),
    Prop("d", "Depth:", default="1", unit="LEN", section="Size and Shape"),
    Prop("h", "Height:", default="1", unit="LEN", section="Size and Shape"),
    Prop("base", "Base:", "choice", "corner", choices=_base_choices, section="Position"),
    *_pos3(), *_axis(), *_rot()]))
register(NodeType("geom.sphere", "Sphere", "sph", "sphere", dims=(3,), props=[
    Prop("r", "Radius:", default="1", unit="LEN", section="Size"),
    *_pos3(), *_axis(), *_rot()]))
register(NodeType("geom.cylinder", "Cylinder", "cyl", "cylinder", dims=(3,), props=[
    Prop("r", "Radius:", default="1", unit="LEN", section="Size and Shape"),
    Prop("h", "Height:", default="1", unit="LEN", section="Size and Shape"),
    *_pos3(), *_axis(), *_rot()]))
register(NodeType("geom.cone", "Cone", "cone", "cone", dims=(3,), props=[
    Prop("r", "Bottom radius:", default="1", unit="LEN", section="Size and Shape"),
    Prop("h", "Height:", default="1", unit="LEN", section="Size and Shape"),
    Prop("spec", "Specify top size using:", "choice", "radius",
         choices=[("radius", "Top radius"), ("angle", "Semiangle")], section="Size and Shape"),
    Prop("rtop", "Top radius:", default="0", unit="LEN", section="Size and Shape",
         visible=lambda p: p.get("spec") == "radius"),
    Prop("ang", "Semiangle:", default="30", unit="deg", section="Size and Shape",
         visible=lambda p: p.get("spec") == "angle"),
    *_pos3(), *_axis(), *_rot()]))
register(NodeType("geom.torus", "Torus", "tor", "torus", dims=(3,), props=[
    Prop("rmaj", "Major radius:", default="1", unit="LEN", section="Size and Shape"),
    Prop("rmin", "Minor radius:", default="0.5", unit="LEN", section="Size and Shape"),
    Prop("angle", "Revolution angle:", default="360", unit="deg", section="Size and Shape"),
    *_pos3(), *_axis(), *_rot()]))
register(NodeType("geom.ellipsoid", "Ellipsoid", "elp", "sphere", dims=(3,), props=[
    Prop("a", "a-semiaxis:", default="1", unit="LEN", section="Size"),
    Prop("b", "b-semiaxis:", default="0.6", unit="LEN", section="Size"),
    Prop("c", "c-semiaxis:", default="0.4", unit="LEN", section="Size"),
    *_pos3(), *_axis(), *_rot()]))

# ---- 2D primitives (also used inside work planes)
register(NodeType("geom.rectangle", "Rectangle", "r", "rectangle", dims=(2, 3), props=[
    Prop("w", "Width:", default="1", unit="LEN", section="Size and Shape"),
    Prop("h", "Height:", default="1", unit="LEN", section="Size and Shape"),
    Prop("base", "Base:", "choice", "corner", choices=_base_choices, section="Position"),
    *_pos2(), *_rot()]))
register(NodeType("geom.circle", "Circle", "c", "circle", dims=(2, 3), props=[
    Prop("r", "Radius:", default="1", unit="LEN", section="Size and Shape"),
    Prop("sector", "Sector angle:", default="360", unit="deg", section="Size and Shape"),
    Prop("base", "Base:", "choice", "center", choices=_base_choices, section="Position"),
    *_pos2(), *_rot()]))
register(NodeType("geom.ellipse", "Ellipse", "e", "ellipse", dims=(2, 3), props=[
    Prop("a", "a-semiaxis:", default="1", unit="LEN", section="Size and Shape"),
    Prop("b", "b-semiaxis:", default="0.5", unit="LEN", section="Size and Shape"),
    *_pos2(), *_rot()]))
register(NodeType("geom.polygon", "Polygon", "pol", "polygon", dims=(2, 3), props=[
    Prop("xs", "x:", "string", "0 1 1 0", section="Coordinates", tip="Space- or comma-separated x coordinates"),
    Prop("ys", "y:", "string", "0 0 1 1", section="Coordinates", tip="Space- or comma-separated y coordinates")]))

# ---- work plane, extrude, revolve
register(NodeType("geom.workplane", "Work Plane", "wp", "work_plane", dims=(3,), children=WP_KINDS, props=[
    Prop("plane", "Plane:", "choice", "xy", choices=[("xy", "xy-plane"), ("yz", "yz-plane"), ("zx", "zx-plane")],
         section="Plane Definition"),
    Prop("offset", "Offset:", default="0", unit="LEN", section="Plane Definition",
         tip="Offset along the plane normal (z for xy, x for yz, y for zx)")]))
register(NodeType("geom.extrude", "Extrude", "ext", "extrude", dims=(3,), props=[
    Prop("input", "Input objects:", "objects", [], section="General"),
    Prop("dist", "Distance from work plane:", default="1", unit="LEN", section="Distances"),
    Prop("reverse", "Reverse direction", "bool", False, section="Distances"),
    Prop("keep", "Keep input objects", "bool", False, section="General")]))
register(NodeType("geom.revolve", "Revolve", "rev", "revolve", dims=(3,), props=[
    Prop("input", "Input objects:", "objects", [], section="General"),
    Prop("angle", "End angle:", default="360", unit="deg", section="Revolution Angles"),
    Prop("px", "Point on axis, xw:", default="0", unit="LEN", section="Revolution Axis"),
    Prop("py", "Point on axis, yw:", default="0", unit="LEN", section="Revolution Axis"),
    Prop("dx", "Direction, xw:", default="0", section="Revolution Axis"),
    Prop("dy", "Direction, yw:", default="1", section="Revolution Axis"),
    Prop("keep", "Keep input objects", "bool", False, section="General")]))

# ---- booleans
register(NodeType("geom.union", "Union", "uni", "union", props=[
    Prop("input", "Input objects:", "objects", [], section="Union"),
    Prop("keep", "Keep input objects", "bool", False, section="Union"),
    Prop("interior", "Keep interior boundaries", "bool", True, section="Union")]))
register(NodeType("geom.difference", "Difference", "dif", "difference", props=[
    Prop("input", "Objects to add:", "objects", [], section="Difference"),
    Prop("tools", "Objects to subtract:", "objects", [], section="Difference"),
    Prop("keep", "Keep input objects", "bool", False, section="Difference")]))
register(NodeType("geom.intersection", "Intersection", "int", "intersection", props=[
    Prop("input", "Input objects:", "objects", [], section="Intersection"),
    Prop("keep", "Keep input objects", "bool", False, section="Intersection")]))

# ---- transforms
_vec3 = lambda sec, pre="", d=("0", "0", "0"), unit="LEN": [  # noqa: E731
    Prop(pre + "x", "x:", default=d[0], unit=unit, section=sec),
    Prop(pre + "y", "y:", default=d[1], unit=unit, section=sec),
    Prop(pre + "z", "z:", default=d[2], unit=unit, section=sec, visible=lambda p: p.get("__sdim", 3) == 3)]
register(NodeType("geom.move", "Move", "mov", "move", props=[
    Prop("input", "Input objects:", "objects", [], section="Input"), *_keep(),
    *_vec3("Displacement", "d")]))
register(NodeType("geom.copy", "Copy", "copy", "transform_copy", props=[
    Prop("input", "Input objects:", "objects", [], section="Input"),
    *_vec3("Displacement", "d")]))
register(NodeType("geom.rotate", "Rotate", "rot", "rotate", props=[
    Prop("input", "Input objects:", "objects", [], section="Input"), *_keep(),
    Prop("angle", "Angle:", default="45", unit="deg", section="Rotation"),
    Prop("axis", "Axis of rotation:", "choice", "z", choices=_axis_choices, section="Rotation",
         visible=lambda p: p.get("__sdim", 3) == 3),
    Prop("ax", "x:", default="0", section="Rotation", visible=lambda p: p.get("axis") == "cartesian"),
    Prop("ay", "y:", default="0", section="Rotation", visible=lambda p: p.get("axis") == "cartesian"),
    Prop("az", "z:", default="1", section="Rotation", visible=lambda p: p.get("axis") == "cartesian"),
    *_vec3("Center of Rotation", "c")]))
register(NodeType("geom.scale", "Scale", "sca", "scale", props=[
    Prop("input", "Input objects:", "objects", [], section="Input"), *_keep(),
    Prop("factor", "Scale factor:", default="1", section="Scale Factor"),
    *_vec3("Center of Scaling", "c")]))
register(NodeType("geom.mirror", "Mirror", "mir", "mirror", props=[
    Prop("input", "Input objects:", "objects", [], section="Input"),
    Prop("keep", "Keep input objects", "bool", True, section="Input"),
    *_vec3("Point on Plane of Reflection", "p"),
    *_vec3("Normal Vector to Plane of Reflection", "n", ("1", "0", "0"), unit="")]))
register(NodeType("geom.array", "Array", "arr", "array", props=[
    Prop("input", "Input objects:", "objects", [], section="Input"),
    Prop("nx", "x size:", "int", 2, section="Size"),
    Prop("ny", "y size:", "int", 1, section="Size"),
    Prop("nz", "z size:", "int", 1, section="Size", visible=lambda p: p.get("__sdim", 3) == 3),
    *_vec3("Displacement", "d", ("1", "1", "1"))]))
register(NodeType("geom.fillet3d", "Fillet", "fil", "fillet", dims=(3,), props=[
    Prop("input", "Input object:", "objects", [], section="Input"),
    Prop("edges", "Edges (numbers, after build of input):", "string", "", section="Edges",
         tip="Edge numbers of the input object, space separated; empty = all edges"),
    Prop("r", "Radius:", default="0.1", unit="LEN", section="Radius")]))
register(NodeType("geom.chamfer3d", "Chamfer", "cha", "chamfer", dims=(3,), props=[
    Prop("input", "Input object:", "objects", [], section="Input"),
    Prop("edges", "Edges (numbers, after build of input):", "string", "", section="Edges"),
    Prop("d", "Distance:", default="0.1", unit="LEN", section="Distance")]))
register(NodeType("geom.import", "Import", "imp", "import", props=[
    Prop("file", "Filename:", "file", "", section="Import",
         tip="STEP (*.step, *.stp), IGES (*.iges, *.igs) or BREP (*.brep) file"),
    Prop("heal", "Repair imported geometry", "bool", True, section="Import")]))
register(NodeType("geom.finalize", "Form Union/Assembly", "fin", "form_union", label="Form Union",
                  numbered=False, deletable=False, can_disable=False, props=[
                      Prop("action", "Action:", "choice", "union",
                           choices=[("union", "Form a union"), ("none", "Keep objects separate (no fragment)")],
                           section="Finalize"),
                      Prop("tol", "Repair tolerance (relative):", default="1e-6", section="Finalize")]))


# --------------------------------------------------------------------------- results
@dataclass
class GeometryResult:
    sdim: int
    unit: str
    scale: float                       # geometry unit -> meters
    bbox: tuple                        # (xmin,ymin,zmin,xmax,ymax,zmax) in geometry units
    numbers: dict = field(default_factory=dict)       # level -> [1..N]
    tag_of: dict = field(default_factory=dict)        # level -> {number: gmsh tag}
    num_of: dict = field(default_factory=dict)        # level -> {gmsh tag: number}
    adj_up: dict = field(default_factory=dict)        # level -> {number: [numbers one level up]}
    adj_down: dict = field(default_factory=dict)      # level -> {number: [numbers one level down]}
    display: dict = field(default_factory=dict)       # level -> {number: geometry arrays}
    bbox_of: dict = field(default_factory=dict)       # level -> {number: (xmin,ymin,zmin,xmax,ymax,zmax)}
    objects: list = field(default_factory=list)       # object names alive before finalize
    messages: list = field(default_factory=list)
    built_upto: str = ""

    @property
    def diag(self) -> float:
        b = self.bbox
        return float(math.dist(b[:3], b[3:])) or 1.0

    def count(self, level):
        return len(self.numbers.get(level, []))

    def boundaries_of_domains(self, doms):
        out = set()
        for d in doms:
            out.update(self.adj_down.get("domain", {}).get(d, []))
        return sorted(out)

    def select(self, level, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, tol=None):
        """Entities whose bounding box lies inside the given (inclusive) box - like COMSOL box selection."""
        tol = max(self.diag * 1e-6, 1e-12) if tol is None else tol
        lo = [xmin, ymin, zmin]
        hi = [xmax, ymax, zmax]
        out = []
        for num, bb in self.bbox_of.get(level, {}).items():
            ok = True
            for i in range(3):
                if lo[i] is not None and bb[i] < lo[i] - tol:
                    ok = False
                if hi[i] is not None and bb[i + 3] > hi[i] + tol:
                    ok = False
            if ok:
                out.append(num)
        return out

    def exterior_boundaries(self):
        up = self.adj_up.get("boundary", {})
        return [b for b in self.numbers.get("boundary", []) if len(up.get(b, [])) <= 1]


LEVEL_DIM3 = {"domain": 3, "boundary": 2, "edge": 1, "point": 0}
LEVEL_DIM2 = {"domain": 2, "boundary": 1, "point": 0}


def levels_for(sdim):
    return LEVEL_DIM3 if sdim == 3 else LEVEL_DIM2


class GeometryError(RuntimeError):
    pass


def _floats(s: str) -> list[float]:
    return [float(x) for x in re.split(r"[\s,;]+", str(s).strip()) if x]


class GeometryBuilder:
    """Builds a geometry node sequence into the active gmsh model."""

    def __init__(self, model: Model, geom: Node, overrides: dict | None = None):
        self.model = model
        self.geom = geom
        comp = geom.ancestor("component")
        self.sdim = space_dim(comp)
        self.unit = geom.get("unit", "m")
        self.ufac = units.unit_info(self.unit)[0]
        self.params = model.parameters(overrides)
        self.param_has_units = self._param_units()
        self.objects: dict[str, list[tuple[int, int]]] = {}
        self.wp_objects: dict[str, dict] = {}
        self.messages: list[str] = []

    # ---- expression helpers (COMSOL semantics: unitless numbers are in the geometry unit)
    def _param_units(self):
        flags = {}
        for name, expr in self.model.parameter_rows():
            has = "[" in expr
            if not has:
                try:
                    has = any(flags.get(s, False) for s in units.free_symbols(expr))
                except Exception:
                    has = False
            flags[name] = has
        return flags

    def _has_units(self, expr) -> bool:
        s = str(expr)
        if "[" in s:
            return True
        try:
            return any(self.param_has_units.get(n, False) for n in units.free_symbols(s))
        except Exception:
            return False

    def L(self, node: Node, key: str) -> float:
        expr = node.get(key, "0")
        try:
            v = float(np.real(units.evaluate(expr, self.params)))
        except Exception as exc:
            raise GeometryError(f"{node.label}: cannot evaluate '{expr}' ({exc})") from exc
        return v / self.ufac if self._has_units(expr) else v

    def A(self, node: Node, key: str) -> float:
        """Angle in radians (unitless numbers are degrees)."""
        expr = node.get(key, "0")
        try:
            v = float(np.real(units.evaluate(expr, self.params)))
        except Exception as exc:
            raise GeometryError(f"{node.label}: cannot evaluate '{expr}' ({exc})") from exc
        if self._has_units(expr):
            return v
        return math.radians(v) if self.geom.get("angunit", "deg") == "deg" else v

    def N(self, node: Node, key: str) -> float:
        expr = node.get(key, "0")
        try:
            return float(np.real(units.evaluate(expr, self.params)))
        except Exception as exc:
            raise GeometryError(f"{node.label}: cannot evaluate '{expr}' ({exc})") from exc

    # ---- object bookkeeping
    def _take(self, names, keep=False) -> list[tuple[int, int]]:
        out = []
        for nm in names or []:
            if nm not in self.objects:
                raise GeometryError(f"Input object '{nm}' does not exist (already consumed or not built)")
            out.extend(self.objects[nm])
            if not keep:
                del self.objects[nm]
        return out

    def _input(self, node, key="input", keep=None):
        names = node.get(key, []) or []
        if isinstance(names, str):
            names = [s for s in re.split(r"[\s,]+", names) if s]
        if not names:
            raise GeometryError(f"{node.label}: no input objects selected")
        keep = node.get("keep", False) if keep is None else keep
        return names, self._take(names, keep)

    # ---- placement
    def _axis_vec(self, node):
        ax = node.get("axis", "z")
        if ax == "x":
            return (1.0, 0.0, 0.0)
        if ax == "y":
            return (0.0, 1.0, 0.0)
        if ax == "z":
            return (0.0, 0.0, 1.0)
        v = np.array([self.N(node, "ax"), self.N(node, "ay"), self.N(node, "az")], float)
        n = np.linalg.norm(v)
        if n == 0:
            raise GeometryError(f"{node.label}: zero axis vector")
        return tuple(v / n)

    def _orient(self, occ, dimtags, node, origin):
        """Rotate objects built along +z so that +z maps to the node axis, then apply rotation angle."""
        axis = np.array(self._axis_vec(node))
        z = np.array([0.0, 0.0, 1.0])
        rot = self.A(node, "rot") if "rot" in node.props else 0.0
        if rot:
            occ.rotate(dimtags, *origin, 0, 0, 1, rot)
        c = np.cross(z, axis)
        s = np.linalg.norm(c)
        if s > 1e-12:
            ang = math.atan2(s, float(np.dot(z, axis)))
            occ.rotate(dimtags, *origin, *(c / s), ang)
        elif np.dot(z, axis) < 0:
            occ.rotate(dimtags, *origin, 1, 0, 0, math.pi)

    # ---- build
    def build(self, upto: Node | None = None, display=True, finalize=True) -> GeometryResult:
        import gmsh
        occ = gmsh.model.occ
        gmsh.option.setNumber("Geometry.OCCParallel", 1)
        gmsh.option.setString("Geometry.OCCTargetUnit", OCC_UNIT.get(self.unit, "M"))
        built_upto = ""
        fin_node = None
        for feat in self.geom.children:
            if feat.kind == "geom.finalize":
                fin_node = feat
                continue
            if not feat.enabled:
                continue
            try:
                self._build_feature(occ, feat)
            except GeometryError:
                raise
            except Exception as exc:
                raise GeometryError(f"{feat.label}: {exc}") from exc
            built_upto = feat.tag
            if upto is not None and feat is upto:
                finalize = False
                break
        occ.synchronize()
        object_names = list(self.objects.keys())
        if finalize and fin_node is not None and fin_node.get("action", "union") == "union":
            all_dt = [dt for v in self.objects.values() for dt in v]
            top = max((d for d, _ in all_dt), default=0)
            if len(all_dt) > 1:
                try:
                    occ.fragment(all_dt, [])
                except Exception as exc:
                    raise GeometryError(f"Form Union failed: {exc}") from exc
                occ.removeAllDuplicates()
            occ.synchronize()
            built_upto = fin_node.tag
        res = self._number(object_names)
        res.built_upto = built_upto
        if display:
            self._tessellate(res)
        res.messages = self.messages
        return res

    def _build_feature(self, occ, f: Node):
        k = f.kind
        sd = self.sdim
        if k == "geom.block":
            w, d, h = self.L(f, "w"), self.L(f, "d"), self.L(f, "h")
            x, y, z = self.L(f, "x"), self.L(f, "y"), self.L(f, "z")
            if min(w, d, h) <= 0:
                raise GeometryError(f"{f.label}: size must be positive")
            if f.get("base") == "center":
                ox, oy, oz = x - w / 2, y - d / 2, z - h / 2
            else:
                ox, oy, oz = x, y, z
            t = occ.addBox(ox, oy, oz, w, d, h)
            dt = [(3, t)]
            self._orient(occ, dt, f, (x, y, z))
            self.objects[f.tag] = dt
        elif k == "geom.sphere":
            r = self.L(f, "r")
            x, y, z = self.L(f, "x"), self.L(f, "y"), self.L(f, "z")
            t = occ.addSphere(x, y, z, r)
            self.objects[f.tag] = [(3, t)]
        elif k == "geom.ellipsoid":
            a, b, c = self.L(f, "a"), self.L(f, "b"), self.L(f, "c")
            x, y, z = self.L(f, "x"), self.L(f, "y"), self.L(f, "z")
            t = occ.addSphere(x, y, z, 1.0)
            occ.dilate([(3, t)], x, y, z, a, b, c)
            dt = [(3, t)]
            self._orient(occ, dt, f, (x, y, z))
            self.objects[f.tag] = dt
        elif k == "geom.cylinder":
            r, h = self.L(f, "r"), self.L(f, "h")
            x, y, z = self.L(f, "x"), self.L(f, "y"), self.L(f, "z")
            ax = self._axis_vec(f)
            t = occ.addCylinder(x, y, z, ax[0] * h, ax[1] * h, ax[2] * h, r)
            dt = [(3, t)]
            rot = self.A(f, "rot")
            if rot:
                occ.rotate(dt, x, y, z, *ax, rot)
            self.objects[f.tag] = dt
        elif k == "geom.cone":
            r, h = self.L(f, "r"), self.L(f, "h")
            if f.get("spec") == "angle":
                rt = max(r - h * math.tan(self.A(f, "ang")), 0.0)
            else:
                rt = self.L(f, "rtop")
            x, y, z = self.L(f, "x"), self.L(f, "y"), self.L(f, "z")
            ax = self._axis_vec(f)
            t = occ.addCone(x, y, z, ax[0] * h, ax[1] * h, ax[2] * h, r, rt)
            self.objects[f.tag] = [(3, t)]
        elif k == "geom.torus":
            R, r = self.L(f, "rmaj"), self.L(f, "rmin")
            x, y, z = self.L(f, "x"), self.L(f, "y"), self.L(f, "z")
            ang = self.A(f, "angle")
            t = occ.addTorus(x, y, z, R, r, angle=ang if ang < 2 * math.pi - 1e-9 else 2 * math.pi)
            dt = [(3, t)]
            self._orient(occ, dt, f, (x, y, z))
            self.objects[f.tag] = dt
        elif k in ("geom.rectangle", "geom.circle", "geom.ellipse", "geom.polygon"):
            self.objects[f.tag] = self._prim2d(occ, f, z=0.0)
            if sd == 3 and f.parent is self.geom:
                raise GeometryError(f"{f.label}: 2D primitives must be placed in a Work Plane in 3D")
        elif k == "geom.workplane":
            self._workplane(occ, f)
        elif k == "geom.extrude":
            self._extrude(occ, f)
        elif k == "geom.revolve":
            self._revolve(occ, f)
        elif k == "geom.union":
            names, dts = self._input(f)
            if len(dts) < 1:
                raise GeometryError(f"{f.label}: needs input objects")
            if len(dts) == 1:
                out = dts
            elif f.get("interior", True):
                out, _ = occ.fragment(dts[:1], dts[1:])
            else:
                out, _ = occ.fuse(dts[:1], dts[1:])
            self.objects[f.tag] = out
        elif k == "geom.difference":
            names, add = self._input(f)
            sub_names = f.get("tools", []) or []
            sub = self._take(sub_names, f.get("keep", False))
            if not sub:
                raise GeometryError(f"{f.label}: select objects to subtract")
            out, _ = occ.cut(add, sub)
            self.objects[f.tag] = out
        elif k == "geom.intersection":
            names, dts = self._input(f)
            if len(dts) < 2:
                raise GeometryError(f"{f.label}: needs at least two input objects")
            cur = dts[:1]
            for o in dts[1:]:
                cur, _ = occ.intersect(cur, [o])
            self.objects[f.tag] = cur
        elif k in ("geom.move", "geom.copy"):
            keep = True if k == "geom.copy" else f.get("keep", False)
            names, dts = self._input(f, keep=keep)
            if keep:
                dts = occ.copy(dts)
            dz = self.L(f, "dz") if sd == 3 else 0.0
            occ.translate(dts, self.L(f, "dx"), self.L(f, "dy"), dz)
            self.objects[f.tag] = dts
        elif k == "geom.rotate":
            keep = f.get("keep", False)
            names, dts = self._input(f, keep=keep)
            if keep:
                dts = occ.copy(dts)
            ax = self._axis_vec(f) if sd == 3 else (0, 0, 1)
            cz = self.L(f, "cz") if sd == 3 else 0.0
            occ.rotate(dts, self.L(f, "cx"), self.L(f, "cy"), cz, *ax, self.A(f, "angle"))
            self.objects[f.tag] = dts
        elif k == "geom.scale":
            keep = f.get("keep", False)
            names, dts = self._input(f, keep=keep)
            if keep:
                dts = occ.copy(dts)
            s = self.N(f, "factor")
            cz = self.L(f, "cz") if sd == 3 else 0.0
            occ.dilate(dts, self.L(f, "cx"), self.L(f, "cy"), cz, s, s, s if sd == 3 else 1.0)
            self.objects[f.tag] = dts
        elif k == "geom.mirror":
            keep = f.get("keep", True)
            names, dts = self._input(f, keep=keep)
            if keep:
                dts = occ.copy(dts)
            n = np.array([self.N(f, "nx"), self.N(f, "ny"), self.N(f, "nz") if sd == 3 else 0.0])
            p = np.array([self.L(f, "px"), self.L(f, "py"), self.L(f, "pz") if sd == 3 else 0.0])
            if np.linalg.norm(n) == 0:
                raise GeometryError(f"{f.label}: zero normal vector")
            n = n / np.linalg.norm(n)
            occ.mirror(dts, n[0], n[1], n[2], -float(np.dot(n, p)))
            self.objects[f.tag] = dts
        elif k == "geom.array":
            names, dts = self._input(f, keep=False)
            nx, ny = int(f.get("nx", 1)), int(f.get("ny", 1))
            nz = int(f.get("nz", 1)) if sd == 3 else 1
            dx, dy = self.L(f, "dx"), self.L(f, "dy")
            dz = self.L(f, "dz") if sd == 3 else 0.0
            out = list(dts)
            for i in range(nx):
                for j in range(ny):
                    for kk in range(nz):
                        if i == j == kk == 0:
                            continue
                        c = occ.copy(dts)
                        occ.translate(c, i * dx, j * dy, kk * dz)
                        out.extend(c)
            self.objects[f.tag] = out
        elif k in ("geom.fillet3d", "geom.chamfer3d"):
            names, dts = self._input(f, keep=False)
            vols = [t for d, t in dts if d == 3]
            occ.synchronize()
            import gmsh
            edges = []
            for v in vols:
                _, ed = gmsh.model.getAdjacencies(3, v)
                faces = gmsh.model.getBoundary([(3, v)], combined=False, oriented=False, recursive=False)
                for _, ft in faces:
                    for _, et in gmsh.model.getBoundary([(2, ft)], combined=False, oriented=False):
                        edges.append(abs(et))
            edges = sorted(set(edges))
            want = [int(x) for x in _floats(f.get("edges", ""))] if str(f.get("edges", "")).strip() else None
            if want:
                edges = [edges[i - 1] for i in want if 0 < i <= len(edges)]
            if k == "geom.fillet3d":
                out = occ.fillet(vols, edges, [self.L(f, "r")])
            else:
                faces = []
                for e in edges:
                    up, _ = gmsh.model.getAdjacencies(1, e)
                    faces.append(int(up[0]))
                out = occ.chamfer(vols, edges, faces, [self.L(f, "d")])
            self.objects[f.tag] = out
        elif k == "geom.import":
            path = str(f.get("file", "")).strip()
            if not path:
                raise GeometryError(f"{f.label}: no file selected")
            import os
            if not os.path.isfile(path):
                raise GeometryError(f"{f.label}: file not found: {path}")
            out = occ.importShapes(path, highestDimOnly=True)
            if f.get("heal", True):
                try:
                    out = occ.healShapes(out) or out
                except Exception:
                    pass
            self.objects[f.tag] = out
        else:
            raise GeometryError(f"Unsupported geometry feature {k}")

    # ---- 2D primitives (built in the local xy-plane at height z)
    def _prim2d(self, occ, f: Node, z=0.0):
        k = f.kind
        if k == "geom.rectangle":
            w, h = self.L(f, "w"), self.L(f, "h")
            x, y = self.L(f, "x"), self.L(f, "y")
            if f.get("base") == "center":
                x0, y0 = x - w / 2, y - h / 2
            else:
                x0, y0 = x, y
            t = occ.addRectangle(x0, y0, z, w, h)
        elif k == "geom.circle":
            r = self.L(f, "r")
            x, y = self.L(f, "x"), self.L(f, "y")
            if f.get("base") == "corner":
                x, y = x + r, y + r
            sec = self.A(f, "sector")
            if sec >= 2 * math.pi - 1e-9:
                t = occ.addDisk(x, y, z, r, r)
            else:
                c = occ.addCircle(x, y, z, r, angle1=0.0, angle2=sec)
                p0 = occ.addPoint(x, y, z)
                p1 = occ.addPoint(x + r, y, z)
                p2 = occ.addPoint(x + r * math.cos(sec), y + r * math.sin(sec), z)
                l1 = occ.addLine(p0, p1)
                l2 = occ.addLine(p2, p0)
                loop = occ.addCurveLoop([l1, c, l2])
                t = occ.addPlaneSurface([loop])
            x0, y0 = x, y
        elif k == "geom.ellipse":
            a, b = self.L(f, "a"), self.L(f, "b")
            x, y = self.L(f, "x"), self.L(f, "y")
            if a >= b:
                t = occ.addDisk(x, y, z, a, b)
            else:
                t = occ.addDisk(x, y, z, b, a)
                occ.rotate([(2, t)], x, y, z, 0, 0, 1, math.pi / 2)
            x0, y0 = x, y
        else:  # polygon
            xs = [float(np.real(units.evaluate(v, self.params))) for v in re.split(r"[\s,;]+", str(f.get("xs", "")).strip()) if v]
            ys = [float(np.real(units.evaluate(v, self.params))) for v in re.split(r"[\s,;]+", str(f.get("ys", "")).strip()) if v]
            if len(xs) != len(ys) or len(xs) < 3:
                raise GeometryError(f"{f.label}: need at least 3 points with matching x and y counts")
            pts = [occ.addPoint(xx, yy, z) for xx, yy in zip(xs, ys)]
            lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
            loop = occ.addCurveLoop(lines)
            t = occ.addPlaneSurface([loop])
            x0, y0 = xs[0], ys[0]
        dt = [(2, t)]
        if "rot" in f.props:
            rot = self.A(f, "rot")
            if rot:
                occ.rotate(dt, x0, y0, z, 0, 0, 1, rot)
        return dt

    # ---- work planes
    def _wp_transform(self, occ, dts, wp: Node):
        off = self.L(wp, "offset")
        plane = wp.get("plane", "xy")
        if plane == "xy":
            if off:
                occ.translate(dts, 0, 0, off)
        elif plane == "yz":   # local (x,y) -> global (y,z), normal +x
            occ.rotate(dts, 0, 0, 0, 1, 1, 1, 2 * math.pi / 3)
            if off:
                occ.translate(dts, off, 0, 0)
        elif plane == "zx":   # local (x,y) -> global (z,x), normal +y
            occ.rotate(dts, 0, 0, 0, 1, 1, 1, -2 * math.pi / 3)
            if off:
                occ.translate(dts, 0, off, 0)

    def _wp_normal(self, wp: Node):
        return {"xy": (0, 0, 1), "yz": (1, 0, 0), "zx": (0, 1, 0)}[wp.get("plane", "xy")]

    def _workplane(self, occ, wp: Node):
        outer = self.objects
        self.objects = {}
        try:
            saved_sdim = self.sdim
            self.sdim = 2
            for f in wp.children:
                if f.enabled:
                    self._build_feature(occ, f)
            self.sdim = saved_sdim
            dts = [dt for v in self.objects.values() for dt in v]
        finally:
            self.objects, inner = outer, self.objects
        if not dts:
            raise GeometryError(f"{wp.label}: the work plane is empty")
        self._wp_transform(occ, dts, wp)
        self.objects[wp.tag] = dts
        self.wp_objects[wp.tag] = {"normal": self._wp_normal(wp), "node": wp}

    def _extrude(self, occ, f: Node):
        names, dts = self._input(f)
        dist = self.L(f, "dist") * (-1 if f.get("reverse") else 1)
        normal = (0, 0, 1)
        for nm in names:
            if nm in self.wp_objects:
                normal = self.wp_objects[nm]["normal"]
        faces = [dt for dt in dts if dt[0] == 2]
        out = occ.extrude(faces, normal[0] * dist, normal[1] * dist, normal[2] * dist)
        vols = [dt for dt in out if dt[0] == 3]
        if not f.get("keep", False):
            occ.remove(faces, recursive=True)
        self.objects[f.tag] = vols

    def _revolve(self, occ, f: Node):
        names, dts = self._input(f)
        wp = None
        for nm in names:
            if nm in self.wp_objects:
                wp = self.wp_objects[nm]["node"]
        px, py = self.L(f, "px"), self.L(f, "py")
        dx, dy = self.N(f, "dx"), self.N(f, "dy")
        # map the in-plane axis to global coordinates
        def g(x, y):
            plane = wp.get("plane", "xy") if wp else "xy"
            off = self.L(wp, "offset") if wp else 0.0
            if plane == "xy":
                return (x, y, off)
            if plane == "yz":
                return (off, x, y)
            return (y, off, x)
        p = g(px, py)
        q = np.array(g(px + dx, py + dy)) - np.array(p)
        if wp is not None and wp.get("plane", "xy") != "xy":
            q = q  # offsets cancel in the difference
        faces = [dt for dt in dts if dt[0] == 2]
        ang = self.A(f, "angle")
        out = occ.revolve(faces, *p, *q, min(ang, 2 * math.pi))
        vols = [dt for dt in out if dt[0] == 3]
        if not f.get("keep", False):
            occ.remove(faces, recursive=True)
        self.objects[f.tag] = vols

    # ---- numbering
    def _number(self, object_names) -> GeometryResult:
        import gmsh
        sd = self.sdim
        lv = levels_for(sd)
        # In 3D, drop free 2D faces that are not part of a volume only if volumes exist? keep all (COMSOL keeps).
        try:
            bb = gmsh.model.getBoundingBox(-1, -1)
        except Exception:
            bb = (0, 0, 0, 1, 1, 1)
        if bb[0] > bb[3]:
            bb = (0, 0, 0, 1, 1, 1)
        res = GeometryResult(sd, self.unit, self.ufac, tuple(bb))
        tol = max(res.diag * 1e-9, 1e-14)
        for level, dim in lv.items():
            ents = [t for d, t in gmsh.model.getEntities(dim)]
            keys = []
            boxes = {}
            for t in ents:
                b = gmsh.model.getBoundingBox(dim, t)
                # OpenCASCADE pads boxes by Precision::Confusion (1e-7); remove it
                b = tuple(b[i] + OCC_PAD for i in range(3)) + tuple(b[i] - OCC_PAD for i in range(3, 6))
                boxes[t] = b
                keys.append((tuple(round(v / tol) for v in b), t))
            keys.sort()
            res.bbox_of[level] = {i + 1: boxes[t] for i, (_, t) in enumerate(keys)}
            res.numbers[level] = list(range(1, len(keys) + 1))
            res.tag_of[level] = {i + 1: t for i, (_, t) in enumerate(keys)}
            res.num_of[level] = {t: i + 1 for i, (_, t) in enumerate(keys)}
        order = list(lv.keys())   # domain, boundary, (edge), point
        for i, level in enumerate(order):
            dim = lv[level]
            up_level = order[i - 1] if i > 0 else None
            down_level = order[i + 1] if i + 1 < len(order) else None
            res.adj_up[level] = {}
            res.adj_down[level] = {}
            for num, tag in res.tag_of[level].items():
                up, down = gmsh.model.getAdjacencies(dim, tag)
                if up_level:
                    res.adj_up[level][num] = sorted(res.num_of[up_level][int(t)] for t in up
                                                    if int(t) in res.num_of[up_level])
                if down_level:
                    res.adj_down[level][num] = sorted(res.num_of[down_level][int(t)] for t in down
                                                      if int(t) in res.num_of[down_level])
        res.objects = object_names
        return res

    # ---- display tessellation
    def _tessellate(self, res: GeometryResult):
        import gmsh
        sd = res.sdim
        diag = res.diag
        try:
            gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 12)
            gmsh.option.setNumber("Mesh.MeshSizeMin", diag / 2000)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 28)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.model.mesh.generate(2 if sd >= 2 else 1)
        except Exception as exc:
            self.messages.append(f"Display tessellation incomplete: {exc}")
        disp = {"boundary": {}, "domain": {}, "edge": {}, "point": {}}
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        if len(ntags):
            idx = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
            idx[ntags.astype(np.int64)] = np.arange(len(ntags))
            xyz = ncoords.reshape(-1, 3)
        else:
            idx, xyz = np.zeros(1, int), np.zeros((0, 3))
        face_level = "boundary" if sd == 3 else "domain"
        for num, tag in res.tag_of.get(face_level, {}).items():
            tris = []
            try:
                types, _, conns = gmsh.model.mesh.getElements(2, tag)
                for t, c in zip(types, conns):
                    if t == 2:
                        tris.append(idx[c.astype(np.int64)].reshape(-1, 3))
                    elif t == 3:
                        q = idx[c.astype(np.int64)].reshape(-1, 4)
                        tris.append(np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]))
            except Exception:
                pass
            if tris:
                T = np.vstack(tris)
                used = np.unique(T)
                disp[face_level][num] = (xyz[used].copy(), np.searchsorted(used, T))
        curve_level = "edge" if sd == 3 else "boundary"
        for num, tag in res.tag_of.get(curve_level, {}).items():
            segs = []
            try:
                types, _, conns = gmsh.model.mesh.getElements(1, tag)
                for t, c in zip(types, conns):
                    if t == 1:
                        segs.append(idx[c.astype(np.int64)].reshape(-1, 2))
            except Exception:
                pass
            if segs:
                S = np.vstack(segs)
                used = np.unique(S)
                disp[curve_level][num] = (xyz[used].copy(), np.searchsorted(used, S))
        for num, tag in res.tag_of.get("point", {}).items():
            try:
                disp["point"][num] = np.array(gmsh.model.getValue(0, tag, []), float)
            except Exception:
                pass
        res.display = disp
        try:
            gmsh.model.mesh.clear()
        except Exception:
            pass


def build_geometry(model: Model, geom: Node | None = None, upto: Node | None = None, overrides=None,
                   display=True) -> GeometryResult:
    comp = model.component
    geom = geom or comp.child("geometry")
    with session("geometry"):
        return GeometryBuilder(model, geom, overrides).build(upto=upto, display=display)


def object_names_before(geom: Node, node: Node) -> list[str]:
    """Names of objects that exist just before *node* in the sequence (for input pickers)."""
    alive: list[str] = []
    seq = geom.children if node.parent is geom else node.parent.children
    for f in seq:
        if f is node:
            break
        if not f.enabled or f.kind == "geom.finalize":
            continue
        consumed = []
        for key in ("input", "tools"):
            v = f.get(key) or []
            if f.get("keep", False) or f.kind == "geom.copy":
                continue
            consumed.extend(v)
        alive = [a for a in alive if a not in consumed]
        alive.append(f.tag)
    return alive
