"""Locate an Elmer installation, write a case directory and run/parse ElmerSolver.

Discovery order: explicit setting -> ELMER_HOME -> PATH -> common install locations
on Windows, macOS and Linux. Nothing here depends on a GPU or vendor library.
"""
from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field

EXE = ".exe" if os.name == "nt" else ""


@dataclass
class ElmerInstall:
    home: str
    bin: str
    solver: str
    grid: str | None = None
    solver_mpi: str | None = None
    mpiexec: str | None = None
    version: str = ""

    def env(self) -> dict:
        env = dict(os.environ)
        env["ELMER_HOME"] = self.home
        sep = os.pathsep
        extra = [self.bin]
        lib = os.path.join(self.home, "lib")
        if os.path.isdir(lib):
            extra.append(lib)
        env["PATH"] = sep.join(extra + [env.get("PATH", "")])
        if platform.system() == "Linux":
            env["LD_LIBRARY_PATH"] = sep.join([lib, os.path.join(lib, "elmersolver"), env.get("LD_LIBRARY_PATH", "")])
        elif platform.system() == "Darwin":
            env["DYLD_LIBRARY_PATH"] = sep.join([lib, os.path.join(lib, "elmersolver"), env.get("DYLD_LIBRARY_PATH", "")])
        env.setdefault("OMP_NUM_THREADS", "1")
        return env


def _candidate_homes() -> list[str]:
    c = []
    if os.environ.get("ELMER_HOME"):
        c.append(os.environ["ELMER_HOME"])
    w = shutil.which("ElmerSolver") or shutil.which("ElmerSolver_mpi")
    if w:
        c.append(os.path.dirname(os.path.dirname(os.path.realpath(w))))
    home = os.path.expanduser("~")
    sysname = platform.system()
    pats = []
    if sysname == "Windows":
        for root in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", ""),
                     os.environ.get("LOCALAPPDATA", ""), "C:\\", home):
            if root:
                pats += [os.path.join(root, "Elmer*"), os.path.join(root, "ElmerFEM*")]
        pats += [os.path.join(home, "*", "ElmerFEM*"), os.path.join(home, "*", "*", "ElmerFEM*"),
                 os.path.join(home, "*", "*", "*", "ElmerFEM*"), os.path.join(home, "*", "*", "*", "Elmer *")]
    elif sysname == "Darwin":
        pats += ["/Applications/Elmer*.app/Contents/MacOS", "/Applications/Elmer*", "/usr/local/opt/elmer",
                 "/opt/homebrew/opt/elmer", "/opt/local", "/usr/local"]
    else:
        pats += ["/usr", "/usr/local", "/opt/elmer*", "/opt/Elmer*", os.path.join(home, "elmer*"),
                 os.path.join(home, ".local")]
    for p in pats:
        c.extend(sorted(glob.glob(p), reverse=True))
    seen, out = set(), []
    for h in c:
        h = os.path.normpath(h)
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def find_elmer(preferred: str | None = None) -> ElmerInstall | None:
    homes = ([preferred] if preferred else []) + _candidate_homes()
    for h in homes:
        if not h or not os.path.isdir(h):
            continue
        for b in (os.path.join(h, "bin"), h):
            solver = os.path.join(b, "ElmerSolver" + EXE)
            solver_mpi = os.path.join(b, "ElmerSolver_mpi" + EXE)
            if os.path.isfile(solver) or os.path.isfile(solver_mpi):
                grid = os.path.join(b, "ElmerGrid" + EXE)
                home = h if b != h else os.path.dirname(h)
                inst = ElmerInstall(home, b, solver if os.path.isfile(solver) else solver_mpi,
                                    grid if os.path.isfile(grid) else None,
                                    solver_mpi if os.path.isfile(solver_mpi) else None,
                                    shutil.which("mpiexec") or shutil.which("mpirun"))
                return inst
    return None


def elmer_version(inst: ElmerInstall, timeout=20) -> str:
    try:
        r = subprocess.run([inst.solver, "-v"], capture_output=True, text=True, timeout=timeout, env=inst.env(),
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        txt = r.stdout + r.stderr
        m = re.search(r"Version:\s*([^\n]+)", txt)
        v = m.group(1).strip() if m else ""
        m2 = re.search(r"Compiled:\s*([^\n]+)", txt)
        if m2 and "ompiled" not in v:
            v += f" (compiled {m2.group(1).strip()})"
        inst.version = v
        return v
    except Exception:
        return ""


# --------------------------------------------------------------------------- log parsing
RE_CHANGE = re.compile(r"ComputeChange:\s+(NS|SS)\s+\(ITER=(\d+)\)\s+\(NRM,RELC\):\s+\(\s*([-+\d.Ee]+)\s+([-+\d.Ee]+)\s*\)\s*::\s*(.+)")
RE_TIME = re.compile(r"MAIN:\s+Time:\s+(\d+)/(\d+)\s+([-+\d.Ee]+)")
RE_SCAN = re.compile(r"MAIN:\s+Scanning:\s+(\d+)/(\d+)")
RE_EIG = re.compile(r"EigenSolve:\s+(\d+):\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)")
RE_EIG2 = re.compile(r"^\s*(\d+):\s+\(?\s*([-+\d.Ee]+)\s*,?\s+([-+\d.Ee]+)\)?\s*$")
RE_ERR = re.compile(r"^(ERROR::|.*Fatal|.*ERROR)", re.I)
RE_WARN = re.compile(r"^WARNING::\s*(.*)")
RE_STEADY = re.compile(r"MAIN:\s+Steady state iteration:\s+(\d+)")


@dataclass
class RunState:
    lines: list = field(default_factory=list)
    conv: list = field(default_factory=list)        # (index, equation, kind, relc)
    eigen: list = field(default_factory=list)       # eigenvalues (omega^2)
    step: tuple = (0, 0)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    finished: bool = False
    in_eigen: bool = False

    def feed(self, line: str):
        self.lines.append(line)
        m = RE_CHANGE.search(line)
        if m:
            self.conv.append((len(self.conv) + 1, m.group(5).strip(), m.group(1), float(m.group(4)), int(m.group(2))))
            return "conv"
        m = RE_TIME.search(line)
        if m:
            self.step = (int(m.group(1)), int(m.group(2)))
            return "step"
        m = RE_SCAN.search(line)
        if m:
            self.step = (int(m.group(1)), int(m.group(2)))
            return "step"
        if "Computed" in line and "Eigen Values" in line:
            self.in_eigen = True
            self.eigen = []
            return None
        m = RE_EIG.search(line)
        if m:
            self.eigen.append(float(m.group(2)))
            return "eigen"
        if self.in_eigen:
            m = RE_EIG2.match(line.split(":", 1)[1] if line.startswith("EigenSolve") else line)
            if m:
                self.eigen.append(float(m.group(2)))
                return "eigen"
        m = RE_WARN.search(line)
        if m:
            self.warnings.append(m.group(1).strip())
            return "warning"
        if "ELMER SOLVER FINISHED" in line.upper():
            self.finished = True
            return "finished"
        if RE_ERR.search(line) and "Linear System Abort" not in line:
            self.errors.append(line.strip())
            return "error"
        return None


def write_case(case_dir: str, elmer_mesh, sif_text: str):
    os.makedirs(case_dir, exist_ok=True)
    elmer_mesh.write(os.path.join(case_dir, "mesh"))
    with open(os.path.join(case_dir, "case.sif"), "w") as f:
        f.write(sif_text)
    with open(os.path.join(case_dir, "ELMERSOLVER_STARTINFO"), "w") as f:
        f.write("case.sif\n1\n")
    res = os.path.join(case_dir, "results")
    if os.path.isdir(res):
        for p in glob.glob(os.path.join(res, "*")):
            try:
                os.remove(p)
            except OSError:
                pass
    os.makedirs(res, exist_ok=True)


def solver_command(inst: ElmerInstall, nproc: int = 1) -> list[str]:
    if nproc > 1 and inst.solver_mpi and inst.mpiexec:
        return [inst.mpiexec, "-n", str(nproc), inst.solver_mpi, "case.sif"]
    return [inst.solver, "case.sif"]


def partition_command(inst: ElmerInstall, nproc: int) -> list[str] | None:
    if nproc > 1 and inst.grid:
        return [inst.grid, "2", "2", "mesh", "-partdual", "-metiskway", str(nproc)]
    return None


def run_case_blocking(case_dir: str, inst: ElmerInstall, nproc=1, on_line=None, timeout=None) -> RunState:
    st = RunState()
    pc = partition_command(inst, nproc)
    if pc:
        subprocess.run(pc, cwd=case_dir, env=inst.env(), capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    p = subprocess.Popen(solver_command(inst, nproc), cwd=case_dir, env=inst.env(), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace",
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for line in p.stdout:
        line = line.rstrip("\n")
        ev = st.feed(line)
        if on_line:
            on_line(line, ev, st)
    p.wait(timeout=timeout)
    st.returncode = p.returncode
    return st


def result_files(case_dir: str, name="case") -> list[str]:
    res = os.path.join(case_dir, "results")
    files = sorted(glob.glob(os.path.join(res, f"{name}_t*.vtu")))
    if not files:
        files = sorted(glob.glob(os.path.join(res, f"{name}_t*.pvtu")))
    if not files:
        files = sorted(glob.glob(os.path.join(res, "*.vtu")) + glob.glob(os.path.join(res, "*.pvtu")))
    return files
