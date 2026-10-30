"""Parser for NASTRAN-95 output reports and punch files.

Extracts:
- Displacements: (T1, T2, T3) per grid point
- Stresses: Element stresses (normal, shear, principal) and von Mises equivalent stress
- Eigenvalues: Mode number, cyclic frequency (Hz), radian frequency (rad/s), generalized mass
- Temperatures: Point temperatures for heat transfer
- Fatal error detection
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

RE_USER_FATAL = re.compile(r"USER FATAL MESSAGE\s+(\d+)[\s,\(]*(.*)", re.IGNORECASE)
RE_SYSTEM_FATAL = re.compile(r"SYSTEM FATAL MESSAGE\s+(\d+)[\s,\(]*(.*)", re.IGNORECASE)
RE_EIGEN_HEADER = re.compile(r"R\s*E\s*A\s*L\s+E\s*I\s*G\s*E\s*N\s*V\s*A\s*L\s*U\s*E\s*S")
RE_SUBCASE = re.compile(r"SUBCASE\s+(\d+)|EIGENVALUE\s+NO\.\s+(\d+)")


@dataclass
class NastranResult:
    displacements: dict[int, list[float]] = field(default_factory=dict)  # nid -> [ux, uy, uz]
    stresses: dict[int, float] = field(default_factory=dict)             # eid -> von Mises (or axial stress)
    temperatures: dict[int, float] = field(default_factory=dict)         # nid -> T
    eigenvalues: list[dict] = field(default_factory=list)                # list of {mode, freq_hz, omega, mass}
    fatal_messages: list[str] = field(default_factory=list)
    success: bool = True


def parse_output(out_text: str) -> NastranResult:
    """Parse NASTRAN-95 print output text."""
    res = NastranResult()
    lines = out_text.splitlines()

    # Check for fatal errors (not warnings or informational messages)
    for ln in lines:
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
    section_subtype = None  # e.g. "CTRIA3", "QUAD4", "ROD" for stress section
    i = 0
    n_lines = len(lines)

    # Track if we're reading temperature multi-point compact header
    temp_multi_point = False
    current_temp_id = None

    while i < n_lines:
        line = lines[i]

        if "D I S P L A C E M E N T   V E C T O R" in line:
            section = "DISP"
            i += 1
            continue
        elif "T E M P E R A T U R E   V E C T O R" in line:
            section = "TEMP"
            temp_multi_point = False
            current_temp_id = None
            i += 1
            continue
        elif "S T R E S S" in line or "F O R C E S   I N   R O D" in line:
            section = "STRESS"
            if "R O D" in line:
                section_subtype = "ROD"
            elif "T R I A" in line or "T R I" in line:
                section_subtype = "CTRIA3"
            elif "Q U A D" in line:
                section_subtype = "QUAD4"
            elif "T E T R A" in line or "S O L I D" in line:
                section_subtype = "CTETRA"
            else:
                section_subtype = None
            i += 1
            continue
        elif "R E A L   E I G E N V A L U E S" in line:
            section = "EIGEN"
            i += 1
            continue
        elif "PAGE" in line or ("NASTRAN" in line and "TITLE" in line) or "END OF JOB" in line:
            # Page banner or end of job - end any active section
            section = None
            section_subtype = None
            temp_multi_point = False
            current_temp_id = None

        if section == "DISP":
            # Header typically: POINT ID.   TYPE          T1             T2             T3
            tokens = line.strip().split()
            if len(tokens) >= 5 and tokens[0].isdigit() and tokens[1] in ("G", "1", "0"):
                try:
                    nid = int(tokens[0])
                    ux = float(tokens[2])
                    uy = float(tokens[3])
                    uz = float(tokens[4])
                    res.displacements[nid] = [ux, uy, uz]
                except (ValueError, IndexError):
                    pass
            elif line.strip().startswith("0") and len(tokens) >= 5 and tokens[0][1:].isdigit():
                # Fortran carriage control '0' attached to number
                try:
                    nid = int(tokens[0][1:])
                    ux = float(tokens[2])
                    uy = float(tokens[3])
                    uz = float(tokens[4])
                    res.displacements[nid] = [ux, uy, uz]
                except (ValueError, IndexError):
                    pass

        elif section == "TEMP":
            # Detect multi-point compact header and data lines.
            # Header: "POINT ID.   TYPE   ID VALUE    ID+1 VALUE   ..."
            # If "ID+1" is present, we have compact format.
            if "ID+1" in line:
                temp_multi_point = True
                i += 1
                continue
            # Data line: "1  S  350  300  300  350  345  340"
            tokens = line.strip().split()
            if len(tokens) >= 3 and tokens[0].isdigit():
                try:
                    base_id = int(tokens[0])
                    val_tokens = tokens[2:]
                    float_vals = []
                    for t in val_tokens:
                        try:
                            float_vals.append(float(t))
                        except ValueError:
                            pass
                    # Compact: assign to consecutive IDs.
                    for j, v in enumerate(float_vals):
                        nid = base_id + j
                        res.temperatures[nid] = v
                except (ValueError, IndexError):
                    pass

        elif section == "STRESS":
            # Handle ROD, QUAD, TRIA, SOLID stresses.
            # CTRIA3/CQUAD4 format with Fortran carriage control '0':
            #   "0       1   -5.000000E-03  1.000001E+07 ... MAJOR  MINOR MAX_SHEAR"
            tokens = line.strip().split()
            eid = None
            if not tokens:
                pass
            elif len(tokens) >= 3 and tokens[0].startswith("0") and len(tokens[0]) > 1 and tokens[0][1].isdigit():
                # eid is split out after the leading '0' carriage control
                # The rest of tokens[0] after '0' is the element ID
                try:
                    eid = int(tokens[0][1:])
                except ValueError:
                    pass
            elif tokens and tokens[0].startswith("0") and len(tokens) >= 4 and tokens[1].isdigit():
                # Format: "0  <eid>  -5.0E-03  ..."
                try:
                    eid = int(tokens[1])
                except ValueError:
                    pass
            else:
                # No carriage control
                if len(tokens) >= 3 and tokens[0].isdigit():
                    try:
                        eid = int(tokens[0])
                    except ValueError:
                        pass
            if eid is not None:
                vals = []
                for t in tokens[1:]:
                    try:
                        vals.append(float(t))
                    except ValueError:
                        pass
                if vals:
                    # CTRIA3 format includes ANGLE, MAJOR, MINOR, MAX_SHEAR
                    # -> MAJOR principal (von Mises equivalent) is 2nd-to-last before MAX_SHEAR
                    # QUAD4 / test format ends with VON MISES (last value)
                    # CTETRA format: SIGMA-XX, SIGMA-YY, SIGMA-ZZ, TAU-YZ, TAU-XZ, TAU-XY, TAU-0, P
                    if section_subtype == "CTRIA3" and len(vals) >= 3:
                        res.stresses[eid] = vals[-3]
                    elif section_subtype == "CTETRA" and len(vals) >= 6:
                        # Compute von Mises from 6 stress components
                        sx, sy, sz = vals[0], vals[1], vals[2]
                        tyz, txz, txy = vals[3], vals[4], vals[5]
                        vm = ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2 + 6 * (txy ** 2 + tyz ** 2 + txz ** 2)) ** 0.5 / 2 ** 0.5
                        res.stresses[eid] = vm
                    else:
                        res.stresses[eid] = vals[-1]

        elif section == "EIGEN":
            # Columns in REAL EIGENVALUES table:
            # MODE NO., EXTRACTION ORDER, EIGENVALUE, RADIAN FREQUENCY, CYCLIC FREQUENCY, GENERALIZED MASS, GENERALIZED STIFFNESS
            # Only accept full table rows (7 columns) to avoid reading subsequent eigenvector/stress output
            tokens = line.strip().split()
            if len(tokens) == 7 and (tokens[0].isdigit() or (tokens[0].startswith("0") and tokens[0][1:].isdigit())):
                try:
                    mode = int(tokens[0][1:] if tokens[0].startswith("0") and not tokens[0].isdigit() else tokens[0])
                    eigval = float(tokens[2])
                    omega = float(tokens[3])
                    freq_hz = float(tokens[4])
                    gen_mass = float(tokens[5])
                    gen_stiff = float(tokens[6])
                    # Verify it's a valid table row: gen_mass ~ 1.0, gen_stiff == eigval
                    if freq_hz > 0.0 and abs(gen_mass - 1.0) < 0.1 and abs(gen_stiff - eigval) / max(abs(eigval), 1.0) < 0.01:
                        res.eigenvalues.append({
                            "mode": mode,
                            "eigenvalue": eigval,
                            "omega": omega,
                            "freq_hz": freq_hz,
                            "gen_mass": gen_mass
                        })
                except (ValueError, IndexError):
                    pass

        i += 1

    return res
