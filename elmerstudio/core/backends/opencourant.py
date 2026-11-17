"""OpenCourant / OpenRadioss explicit-dynamics backend (external process).

Generates Radioss block-format starter (``*_0000.rad``) and engine
(``*_0001.rad``) decks from a Gmsh mesh plus model properties, and ingests
OpenCourant/OpenRadioss result output (legacy VTK animation frames and the
energy time history) back into Studio datasets.

The starter/engine binaries are *only ever invoked as external processes* --
this backend never links against them -- so the GPLv2 / AGPLv3 separation is
preserved.  When no starter/engine binary is installed, deck generation and
result ingestion still work; the live solve step is skipped.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "OpenCourantInstall", "find_opencourant",
    "write_starter_deck", "write_engine_deck",
    "parse_vtk", "parse_vtk_animation", "parse_energy_history",
    "energy_balanced", "OpCourantStudy",
]


@dataclass
class OpenCourantInstall:
    starter: str | None      # path to the starter binary (or None)
    engine: str | None       # path to the engine binary (or None)
    root: str | None         # install root used to locate them

    @property
    def available(self) -> bool:
        return self.starter is not None and self.engine is not None


def find_opencourant(preferred: str | None = None) -> OpenCourantInstall | None:
    """Locate an OpenCourant / OpenRadioss installation.

    Looks for ``starter`` and ``engine`` (the Radioss solver executables) on
    PATH or under an explicit root, returning None when neither is present
    (in which case deck generation and postprocessing still work).
    """
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates += [os.environ.get("OPENCOURANT_ROOT", ""),
                   os.environ.get("OPENRADIOSS_ROOT", ""),
                   "/opt/openradiuss", "/opt/OpenCourant",
                   os.path.join(os.path.expanduser("~"), "OpenCourant"),
                   os.path.join(os.path.expanduser("~"), "OpenRadioss")]
    seen = set()
    for root in candidates:
        if not root or root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        starter = shutil.which("starter", path=os.path.join(root, "bin"))
        engine = shutil.which("engine", path=os.path.join(root, "bin"))
        if starter or engine:
            return OpenCourantInstall(starter, engine, root)
    starter = shutil.which("starter")
    engine = shutil.which("engine")
    if starter or engine:
        return OpenCourantInstall(starter, engine, None)
    return None


# --------------------------------------------------------------------------- Radioss block-format decks

def _fmt(val: float | str) -> str:
    if isinstance(val, str):
        return val
    s = f"{val:.10g}"
    return s if s not in ("-0", "") else "0"


def _block(keyword: str, rows: list[list[Any]]) -> str:
    lines = [keyword]
    for row in rows:
        lines.append(" ".join(_fmt(v) for v in row))
    return "\n".join(lines) + "\n"


def write_starter_deck(nodes, elements, props: dict, path: str) -> str:
    """Write a Radioss starter deck (``*_0000.rad``).

    ``nodes`` is an (N,3) array of coordinates; ``elements`` is a list of
    ``(gmsh_type, conn)`` tuples with 0-based node indices.  ``props`` may
    carry: ``end_time``, ``output_interval``, ``materials`` (list of dicts with
    ``law`` in {"law2","law36"} plus law parameters), ``initial_velocity``,
    ``rigid_walls``, ``contact`` (bool), ``time_step`` (mass scaling).
    Returns the deck text (also written to ``path``).
    """
    nodes = np.asarray(nodes, float)
    n_nodes = len(nodes)
    mats = props.get("materials") or [{"law": "law2", "rho": 7850.0, "E": 200e9, "nu": 0.3,
                                       "sig0": 250e6, "h": 2000e6}]
    end_time = float(props.get("end_time", 1.0))
    dt_cst = float(props.get("time_step", end_time / 1000.0))
    out = []
    out.append("/RUN\n# ElmerStudio OpenCourant starter deck\n# Generated from Gmsh mesh: %d nodes, %d elements\n"
               % (n_nodes, len(elements)))
    out.append("/UNIT/1\n# Unit system: SI (m, kg, s, Pa)\n")
    out.append("/ANIM\n# Output animation\n")
    out.append(_block("/DT/NODA/CST", [[dt_cst]]))
    out.append("# Materials\n")
    mid = 1
    for m in mats:
        law = str(m.get("law", "law2")).lower()
        rho = float(m.get("rho", 7850.0))
        if law == "law36":   # bilinear elastoplastic
            E = float(m.get("E", 200e9)); sig0 = float(m.get("sig0", 250e6))
            et = float(m.get("et", 2000e6))
            out.append(_block(f"/MAT/LAW36 {mid}", [[E, 0.3, rho, sig0, et]]))
        else:                # Johnson-Cook law2
            E = float(m.get("E", 200e9)); sig0 = float(m.get("sig0", 250e6))
            h = float(m.get("h", 2000e6))
            out.append(_block(f"/MAT/LAW2 {mid}", [[E, 0.3, rho, sig0, h]]))
        mid += 1
    out.append("# Properties & parts\n")
    pid = 1
    for m in mats:
        out.append(_block(f"/PROP/SHELL {pid}", [[pid, mid, 1.0]]))  # placeholder thickness 1
        out.append(_block(f"/PART {pid}", [[pid, f"part {pid}", mid]]))
        pid += 1
    out.append("# Nodes\n")
    out.append(_block("/NODE", [[i + 1] + list(nodes[i]) for i in range(n_nodes)]))
    out.append("# Elements\n")
    eid = 1
    for _gtype, conn in elements:
        conn = np.atleast_2d(np.asarray(conn, int))
        for row in conn:
            conn1 = [int(c) + 1 for c in row]
            out.append(_block(f"/ELEMENT/SHELL {eid}", [[eid, pid if mats else 1] + conn1]))
            eid += 1
    iv = props.get("initial_velocity")
    if iv:
        out.append("# Initial velocity\n")
        out.append(_block("/INIVEL", [[1, 1] + [float(v) for v in iv]]))
    rw = props.get("rigid_walls")
    if rw:
        out.append("# Rigid walls\n")
        for i, wall in enumerate(rw, 1):
            out.append(_block(f"/RWALL {i}", [[1, 1] + [float(v) for v in wall]]))
    if props.get("contact"):
        out.append("# Surface contact (Type 7)\n")
        out.append("/INTER/TYPE7 1\n# surface-to-surface contact\n")
    out.append("/END\n")
    deck = "".join(out)
    with open(path, "w", newline="\n") as f:
        f.write(deck)
    return deck


def write_engine_deck(props: dict, path: str) -> str:
    """Write a Radioss engine deck (``*_0001.rad``): end time, output interval."""
    end_time = float(props.get("end_time", 1.0))
    out_interval = float(props.get("output_interval", end_time / 10.0))
    dt_cst = float(props.get("time_step", end_time / 1000.0))
    deck = (
        "/RUN\n# ElmerStudio OpenCourant engine deck\n"
        f"# output interval {out_interval:.10g} s\n"
        "/ANIM\n# Output animation\n"
        f"/DT/NODA/CST\n{dt_cst:.10g}\n"
        f"/END\n"
    )
    with open(path, "w", newline="\n") as f:
        f.write(deck)
    return deck


# --------------------------------------------------------------------------- VTK animation ingestion

def _read_vtk_blocks(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    blocks: dict = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.split()[0] in ("POINTS", "CELLS", "CELL_TYPES", "POINT_DATA", "CELL_DATA"):
            header = ln.split()[0]
            parts = ln.split()[1:2]
            i += 1
            if header == "POINTS":
                n = int(parts[0])
                data = []
                for _ in range(n):
                    data.append([float(v) for v in lines[i].split()]); i += 1
                blocks["POINTS"] = np.asarray(data, float)
            elif header == "CELL_TYPES":
                n = int(parts[0])
                data = []
                for _ in range(n):
                    data.append(int(lines[i])); i += 1
                blocks["CELL_TYPES"] = np.asarray(data, int)
            elif header == "POINT_DATA" or header == "CELL_DATA":
                n = int(parts[0])
                fields: dict = {}
                while i < len(lines):
                    l = lines[i]
                    if l in ("POINTS", "CELLS", "CELL_TYPES", "POINT_DATA", "CELL_DATA"):
                        break
                    if l.upper().startswith("VECTORS") or l.upper().startswith("SCALARS") or l.upper().startswith("LOOKUP_TABLE"):
                        if l.upper().startswith("VECTORS"):
                            _, name, _ = l.split(None, 2)
                            i += 1
                            vals = []
                            for _ in range(n):
                                vals.append([float(v) for v in lines[i].split()]); i += 1
                            fields[name] = np.asarray(vals, float)
                        else:
                            _, name = l.split(None, 1)
                            i += 1
                            # skip RANGE / LOOKUP_TABLE lines until data
                            while i < len(lines) and not lines[i][0].isdigit() and lines[i] not in ("-",):
                                i += 1
                            vals = []
                            for _ in range(n):
                                vals.append(float(lines[i])); i += 1
                            fields[name] = np.asarray(vals, float)
                        continue
                    i += 1
                blocks[header] = fields
        else:
            i += 1
    return blocks


def parse_vtk(path: str) -> tuple[float, dict[str, Any]]:
    """Parse one legacy VTK unstructured-grid file.

    Returns ``(time, fields)`` where ``fields`` maps VTK field names to numpy
    arrays (one row per point).  ``time`` comes from a ``FIELD`` block if
    present, otherwise from the trailing number in the file name
    (``frame-0001.vtk`` -> 1.0), defaulting to 0.
    """
    with open(path, "r", errors="replace") as f:
        text = f.read()
    blocks = _read_vtk_blocks(text)
    time = 0.0
    fdata = blocks.get("FIELD")
    if isinstance(fdata, dict) and "time" in fdata:
        time = float(fdata["time"])
    else:
        m = _NAME_RE.search(os.path.basename(path))
        if m:
            time = float(m.group(1))
    pd = blocks.get("POINT_DATA", {})
    fields = {k: v for k, v in pd.items() if isinstance(v, np.ndarray)}
    fields["nodes"] = blocks.get("POINTS")
    return time, fields


_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)")
_NAME_RE = re.compile(r"(?:^|[^0-9])(\d+(?:\.\d+)?)(?=\.vtk$|\.vtu$)")


def parse_vtk_animation(source: str) -> list[tuple[float, dict[str, Any]]]:
    """Parse a VTK animation: a single ``.vtk`` file, or a directory of them.

    Returns a list of ``(time, fields)`` sorted by time.  Directory frames are
    ordered by the trailing number in their name (``disp-0001.vtk``, ...).
    """
    frames: list[tuple[float, dict[str, Any]]] = []
    if os.path.isdir(source):
        files = sorted(
            (os.path.join(source, fn) for fn in os.listdir(source) if fn.lower().endswith((".vtk", ".vtu"))),
            key=lambda p: (float(_NAME_RE.search(os.path.basename(p)).group(1))
                           if _NAME_RE.search(os.path.basename(p)) else 0.0),
        )
        for fp in files:
            frames.append(parse_vtk(fp))
    else:
        frames.append(parse_vtk(source))
    frames.sort(key=lambda t: t[0])
    return frames


# --------------------------------------------------------------------------- energy time history

_ENERGY_RE = re.compile(r"^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)")


def parse_energy_history(path: str) -> dict[str, "np.ndarray"]:
    """Parse an OpenCourant energy time history (``time  Eint  Ekin  [Etot]``).

    Returns arrays ``t``, ``Eint``, ``Ekin``, ``Etot`` (Etot = Eint + Ekin when
    the column is missing).
    """
    t, Eint, Ekin, Etot = [], [], [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = _ENERGY_RE.match(line)
            if not m:
                continue
            t.append(float(m.group(1))); Eint.append(float(m.group(2))); Ekin.append(float(m.group(3)))
            if m.lastindex and m.lastindex >= 4:
                Etot.append(float(m.group(4)))
            else:
                Etot.append(Eint[-1] + Ekin[-1])
    import numpy as np
    return {"t": np.asarray(t), "Eint": np.asarray(Eint), "Ekin": np.asarray(Ekin), "Etot": np.asarray(Etot)}


def energy_balanced(energy: dict[str, "np.ndarray"], tol: float = 1e-3) -> bool:
    """True when Etot ~= Eint + Ekin within ``tol`` (relative, over the run)."""
    import numpy as np
    Etot = energy["Etot"]; Esum = energy["Eint"] + energy["Ekin"]
    if Etot.size == 0:
        return True
    scale = np.max(np.abs(Etot)) or 1.0
    return bool(np.all(np.abs(Etot - Esum) <= tol * scale))


# --------------------------------------------------------------------------- external-process solver

class OpCourantStudy:
    """Run an explicit-dynamics study through the external starter/engine.

    The binaries are located via :func:`find_opencourant` and invoked as
    external processes; nothing is linked against OpenCourant/OpenRadioss.
    """

    def __init__(self, install: OpenCourantInstall | None = None):
        self.inst = install or find_opencourant()
        self.logs: list[str] = []

    def available(self) -> bool:
        return bool(self.inst and self.inst.available)

    def run(self, starter_deck: str, engine_deck: str, case: str) -> int:
        """Write both decks into ``case`` and run starter then engine.

        Returns the engine's exit code, or raises ``RuntimeError`` when no
        binary is installed.  Appends solver output to ``self.logs``.
        """
        if not self.available():
            raise RuntimeError("OpenCourant/OpenRadioss starter/engine not installed")
        os.makedirs(case, exist_ok=True)
        base = os.path.join(case, "model")
        with open(base + "_0000.rad", "w", newline="\n") as f:
            f.write(starter_deck)
        with open(base + "_0001.rad", "w", newline="\n") as f:
            f.write(engine_deck)
        rc = 0
        for deck, binpath in (("_0000.rad", self.inst.starter), ("_0001.rad", self.inst.engine)):
            cmd = [binpath, base + deck]
            self.logs.append(f"===== {' '.join(cmd)} =====")
            r = subprocess.run(cmd, cwd=case, capture_output=True, text=True, errors="replace", timeout=3600)
            self.logs.append(r.stdout)
            self.logs.append(r.stderr)
            rc = max(rc, r.returncode)
            if r.returncode != 0:
                break
        return rc

    # ------------------------------------------------------------------ study-runner entry point

    def run_opencourant(self, sr) -> list[str]:
        """Entry point from :class:`elmerstudio.core.study_runner.StudyRunner`.

        Builds the mesh, writes the starter/engine decks into the component
        case folder, and runs starter+engine when the binaries are installed
        (otherwise the decks are written and the live solve is skipped).
        Returns the list of solution folders (empty here: results are read
        back from the VTK/energy files the solver writes).
        """
        from .geometry import build_geometry
        from .meshing import build_mesh
        comp = sr.model.component
        case = sr.comp_dir(comp)
        with sr.model.scoped(comp):
            build_geometry(sr.model, display=False)
        mesh = build_mesh(sr.model)
        nodes = np.asarray(mesh.nodes, float)
        elements = [(gtype, np.asarray(conn, int))
                    for _dom, lst in (mesh.bulk or {}).items() for (gtype, conn) in lst]
        study = sr.study
        props = {
            "end_time": float(study.get("end_time", 1.0) or 1.0),
            "output_interval": float(study.get("output_interval", 0.1) or 0.1),
            "time_step": float(study.get("time_step", 0.001) or 0.001),
            "materials": study.get("materials") or [{"law": "law2"}],
            "initial_velocity": study.get("initial_velocity"),
            "rigid_walls": study.get("rigid_walls"),
            "contact": bool(study.get("contact")),
        }
        starter_path = os.path.join(case, "model_0000.rad")
        engine_path = os.path.join(case, "model_0001.rad")
        self.logs.append(f"===== OpenCourant decks -> {case} =====")
        write_starter_deck(nodes, elements, props, starter_path)
        write_engine_deck(props, engine_path)
        self.inst = find_opencourant()
        if self.available():
            rc = self.run(
                open(starter_path).read(), open(engine_path).read(), case)
            self.logs.append(f"engine exit code {rc}")
        else:
            self.logs.append("OpenCourant/OpenRadioss not installed: decks written, live solve skipped")
        return []
