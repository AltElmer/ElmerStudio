"""Parser for NASTRAN-95 printed output.

Extracts:
- Displacements: (T1, T2, T3) per grid point
- Eigenvectors: (T1, T2, T3) per grid point for each extracted mode
- Stresses: von Mises equivalent stress per element (and, for membranes, the principal stresses)
- Eigenvalues: Mode number, cyclic frequency (Hz), radian frequency (rad/s), generalized mass
- Temperatures: Point temperatures for heat transfer
- Fatal error detection
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

RE_USER_FATAL = re.compile(r"USER FATAL MESSAGE\s+(\d+)[\s,\(]*(.*)", re.IGNORECASE)
RE_SYSTEM_FATAL = re.compile(r"SYSTEM FATAL MESSAGE\s+(\d+)[\s,\(]*(.*)", re.IGNORECASE)
# The GINO file system stops the run without a fatal message when a scratch file outgrows it, e.g.
#   "I/O SUBSYSTEM ERROR NUMBER       105" then "I/O ERROR #     0 ON FILE       67 NAME=KELM     UNIT=  48"
RE_IO_ERROR = re.compile(r"I/O SUBSYSTEM ERROR NUMBER\s+(\d+)")
RE_IO_FILE = re.compile(r"ON FILE\s+\d+\s+NAME=(\S+)")
RE_EIGEN_HEADER = re.compile(r"R\s*E\s*A\s*L\s+E\s*I\s*G\s*E\s*N\s*V\s*A\s*L\s*U\s*E\s*S")
RE_EIGVEC_HEADER = re.compile(r"R E A L   E I G E N V E C T O R   N O \.\s+(\d+)")


@dataclass
class NastranResult:
    displacements: dict[int, list[float]] = field(default_factory=dict)  # nid -> [ux, uy, uz]
    eigenvectors: dict[int, dict[int, list[float]]] = field(default_factory=dict)  # mode -> nid -> [ux, uy, uz]
    stresses: dict[int, float] = field(default_factory=dict)             # eid -> von Mises (in-plane for membranes)
    principal: dict[int, tuple[float, float]] = field(default_factory=dict)  # membrane eid -> (major, minor)
    temperatures: dict[int, float] = field(default_factory=dict)         # nid -> T
    eigenvalues: list[dict] = field(default_factory=list)                # list of {mode, freq_hz, omega, mass}
    fatal_messages: list[str] = field(default_factory=list)
    success: bool = True


def _floats(tokens) -> list[float]:
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def _grid_row(line: str):
    """(grid ID, [T1, T2, T3]) of a displacement/eigenvector table row, or None.  The row may carry the Fortran
    carriage control '0' in front of the ID ("0      12      G  ...")."""
    tokens = line.split()
    if tokens and tokens[0] == "0":
        tokens = tokens[1:]
    elif tokens and tokens[0].startswith("0") and tokens[0][1:].isdigit():
        tokens = [tokens[0][1:]] + tokens[1:]
    if len(tokens) >= 5 and tokens[0].isdigit() and tokens[1] in ("G", "S"):
        try:
            return int(tokens[0]), [float(tokens[2]), float(tokens[3]), float(tokens[4])]
        except ValueError:
            return None
    return None


def _von_mises_plane(s1: float, s2: float) -> float:
    return (s1 * s1 - s1 * s2 + s2 * s2) ** 0.5


def parse_output(out_text: str) -> NastranResult:
    """Parse NASTRAN-95 print output text."""
    res = NastranResult()
    lines = out_text.splitlines()

    # Check for fatal errors (not warnings or informational messages)
    for i, ln in enumerate(lines):
        m_io = RE_IO_ERROR.search(ln)
        if m_io:
            m_file = next((RE_IO_FILE.search(x) for x in lines[i + 1:i + 4] if RE_IO_FILE.search(x)), None)
            res.fatal_messages.append(
                f"I/O subsystem error {m_io.group(1)}" + (f" on file {m_file.group(1)}" if m_file else "")
                + ": the model is too large for NASTRAN-95's scratch files; use a coarser mesh")
            res.success = False
            continue
        m_uf = RE_USER_FATAL.search(ln)
        if m_uf:
            res.fatal_messages.append(f"USER FATAL MESSAGE {m_uf.group(1)}: {m_uf.group(2).strip()}")
            res.success = False
            continue
        m_sf = RE_SYSTEM_FATAL.search(ln)
        if m_sf:
            res.fatal_messages.append(f"SYSTEM FATAL MESSAGE {m_sf.group(1)}: {m_sf.group(2).strip()}")
            res.success = False
            continue
        # Only catch actual FATAL messages, not warnings or "NON-FATAL MESSAGES"
        if " FATAL MESSAGE " in ln or ln.startswith("*** USER FATAL") or ln.startswith("*** SYSTEM FATAL"):
            res.fatal_messages.append(ln.strip())
            res.success = False

    # State machine for section parsing
    section = None
    section_subtype = None  # "MEMBRANE" (CTRIA3/CQUAD4), "ROD" or "CTETRA" (solids) for the stress section
    mode = 0

    for line in lines:
        m_vec = RE_EIGVEC_HEADER.search(line)
        if m_vec:
            section, mode = "EIGVEC", int(m_vec.group(1))
            continue
        if "D I S P L A C E M E N T   V E C T O R" in line:
            section = "DISP"
            continue
        if "T E M P E R A T U R E   V E C T O R" in line:
            section = "TEMP"
            continue
        if "S T R E S S" in line or "F O R C E S   I N   R O D" in line:
            section = "STRESS"
            if "R O D" in line:
                section_subtype = "ROD"
            elif "T R I A" in line or "Q U A D" in line:
                section_subtype = "MEMBRANE"
            elif "T E T R A" in line or "S O L I D" in line or "H E X A" in line or "P E N T A" in line:
                section_subtype = "CTETRA"
            else:
                section_subtype = None
            continue
        if RE_EIGEN_HEADER.search(line):
            section = "EIGEN"
            continue
        if "PAGE" in line or ("NASTRAN" in line and "TITLE" in line) or "END OF JOB" in line:
            # Page banner or end of job: a table continues on the next page under its header again
            section = None
            section_subtype = None
            continue

        if section in ("DISP", "EIGVEC"):
            row = _grid_row(line)
            if row is not None:
                nid, vec = row
                (res.displacements if section == "DISP" else res.eigenvectors.setdefault(mode, {}))[nid] = vec

        elif section == "TEMP":
            # Compact rows "ID  TYPE  T(ID)  T(ID+1) ...": consecutive grid IDs from the first one
            tokens = line.split()
            if len(tokens) >= 3 and tokens[0].isdigit():
                for j, v in enumerate(_floats(tokens[2:])):
                    res.temperatures[int(tokens[0]) + j] = v

        elif section == "STRESS":
            # Element rows start with the element ID, possibly behind the carriage control "0":
            #   CTRIA3/CQUAD4: "0  <eid>  <fibre>  NORMAL-X NORMAL-Y SHEAR-XY ANGLE MAJOR MINOR MAX-SHEAR"
            #                  (the second fibre follows on a row without an ID and is the same for a membrane)
            #   CTETRA:        "<eid>  SIGMA-XX SIGMA-YY SIGMA-ZZ TAU-YZ TAU-XZ TAU-XY TAU-0 P"
            tokens = line.split()
            if tokens and tokens[0] == "0":
                tokens = tokens[1:]
            elif tokens and tokens[0].startswith("0") and tokens[0][1:].isdigit():
                tokens = [tokens[0][1:]] + tokens[1:]
            if len(tokens) < 3 or not tokens[0].isdigit():
                continue
            eid, vals = int(tokens[0]), _floats(tokens[1:])
            if section_subtype == "MEMBRANE" and len(vals) >= 8:
                s1, s2 = vals[5], vals[6]
                res.principal[eid] = (s1, s2)
                res.stresses[eid] = _von_mises_plane(s1, s2)
            elif section_subtype == "CTETRA" and len(vals) >= 6:
                sx, sy, sz, tyz, txz, txy = vals[:6]
                res.stresses[eid] = ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2
                                     + 6 * (txy ** 2 + tyz ** 2 + txz ** 2)) ** 0.5 / 2 ** 0.5
            elif section_subtype == "ROD" and vals:
                res.stresses[eid] = abs(vals[0])

        elif section == "EIGEN":
            # Columns in REAL EIGENVALUES table:
            # MODE NO., EXTRACTION ORDER, EIGENVALUE, RADIAN FREQUENCY, CYCLIC FREQUENCY, GENERALIZED MASS, GENERALIZED STIFFNESS
            # Only accept full table rows (7 columns) to avoid reading subsequent eigenvector/stress output
            tokens = line.split()
            if len(tokens) == 7 and (tokens[0].isdigit() or (tokens[0].startswith("0") and tokens[0][1:].isdigit())):
                try:
                    mode_no = int(tokens[0])
                    eigval = float(tokens[2])
                    omega = float(tokens[3])
                    freq_hz = float(tokens[4])
                    gen_mass = float(tokens[5])
                    gen_stiff = float(tokens[6])
                    # Rows of extracted modes only: gen_mass ~ 1.0, gen_stiff == eigval (zeros pad the table)
                    if freq_hz > 0.0 and abs(gen_mass - 1.0) < 0.1 and abs(gen_stiff - eigval) / max(abs(eigval), 1.0) < 0.01:
                        res.eigenvalues.append({
                            "mode": mode_no,
                            "eigenvalue": eigval,
                            "omega": omega,
                            "freq_hz": freq_hz,
                            "gen_mass": gen_mass
                        })
                except (ValueError, IndexError):
                    pass

    return res
