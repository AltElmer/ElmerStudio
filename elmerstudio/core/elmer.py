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
    build_tree: bool = False          # a CMake build directory of elmerfem (from-source Elmer)

    def env(self) -> dict:
        env = dict(os.environ)
        if self.build_tree:
            # an elmerfem build tree: the solver, its modules and three runtime libraries live in five places
            h = self.home
            dirs = [os.path.join(h, *p) for p in (("fem", "src"), ("fem", "src", "modules"), ("fhutiter", "src"),
                                                    ("matc", "src"), ("mathlibs", "src", "arpack"))]
            rt = os.environ.get("MSYS2_UCRT_BIN", r"C:\msys64\ucrt64\bin" if os.name == "nt" else "")
            if rt and os.path.isdir(rt):
                dirs.append(rt)
            env["ELMER_HOME"] = dirs[0]
            env["ELMER_LIB"] = dirs[1]
            env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
            if platform.system() == "Linux":
                env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [env.get("LD_LIBRARY_PATH", "")])
            elif platform.system() == "Darwin":
                env["DYLD_LIBRARY_PATH"] = os.pathsep.join(dirs + [env.get("DYLD_LIBRARY_PATH", "")])
            env.setdefault("OMP_NUM_THREADS", "1")
            return env
        # only point ELMER_HOME at self-contained installs; distro packages (/usr) use compiled-in paths
        if os.path.isdir(os.path.join(self.home, "share", "elmersolver")) and self.home not in ("/usr", "/usr/local"):
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
        bt = os.path.join(h, "fem", "src", "ElmerSolver" + EXE)
        if os.path.isfile(bt):                      # CMake build tree of elmerfem
            grid = os.path.join(h, "elmergrid", "src", "ElmerGrid" + EXE)
            return ElmerInstall(h, os.path.dirname(bt), bt, grid if os.path.isfile(grid) else None,
                                build_tree=True)
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


_CAPS: dict = {}


def module_dir(inst: ElmerInstall) -> str | None:
    """Folder holding the solver module libraries (StatCurrentSolve.dll/.so, ...)."""
    cands = [os.path.join(inst.home, "fem", "src", "modules")] if inst.build_tree else []
    cands += [os.path.join(inst.home, "share", "elmersolver", "lib"), os.path.join(inst.home, "lib", "elmersolver"),
              os.path.join(inst.home, "lib")]
    for c in cands:
        if glob.glob(os.path.join(c, "StatCurrentSolve*")):
            return c
    return None


def capabilities(inst: ElmerInstall | None) -> dict:
    """What this Elmer can do beyond the 9.0 release, detected from its module binaries (cached):

    diffusion_current -- StatCurrentSolve adds F*sum z_i D_i grad c_i (multi-ion electrolytes; ElmerCSC #871)
    advdiff_migration -- AdvectionDiffusion has the ion-migration keywords
    user_solvers      -- a Fortran compiler can build user solvers against this Elmer
    """
    if inst is None:
        return {"diffusion_current": False, "advdiff_migration": False, "user_solvers": False}
    key = (inst.home, inst.solver)
    if key in _CAPS:
        return _CAPS[key]

    def has(module, *needles):
        d = module_dir(inst)
        for f in glob.glob(os.path.join(d, module + ".*")) if d else []:
            try:
                with open(f, "rb") as fh:
                    blob = fh.read().lower()
                if all(n.lower().encode() in blob for n in needles):
                    return True
            except OSError:
                continue
        return False

    caps = {"diffusion_current": has("StatCurrentSolve", "diffusion current"),
            "advdiff_migration": has("AdvectionDiffusion", "migration"),
            "user_solvers": bool(fortran_compiler(inst))}
    _CAPS[key] = caps
    return caps


def fortran_compiler(inst: ElmerInstall) -> str | None:
    """elmerf90 of an installed Elmer, or gfortran next to a build tree (MSYS2 UCRT64 on Windows)."""
    if not inst.build_tree:
        f = os.path.join(inst.bin, "elmerf90" + (".bat" if os.name == "nt" else ""))
        if os.path.isfile(f):
            return f
    rt = os.environ.get("MSYS2_UCRT_BIN", r"C:\msys64\ucrt64\bin" if os.name == "nt" else "")
    g = os.path.join(rt, "gfortran" + EXE) if rt else ""
    if g and os.path.isfile(g):
        return g
    return shutil.which("gfortran")


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


def convergence_problems(conv, spec) -> list[str]:
    """Iterations that stopped at their maximum without meeting the tolerance, from Elmer's ComputeChange lines
    (RunState.conv: (n, equation, NS|SS, relative change, iteration)) and the SIF's convergence spec
    (sif meta["convergence"]).  Elmer itself only reports these at an output level the runner does not log."""
    if not spec:
        return []
    solvers = {s["equation"]: s for s in spec.get("solvers", []) if not s.get("post")}
    ss_maxit = int(spec.get("ss_maxit", 1) or 1)
    nl_bad: dict = {}                                  # equation -> [count, total, worst relc, maxit, tol]
    ss_bad = [0, 0, 0.0]                               # blocks at the maximum, blocks, worst relc
    nl_cur: dict = {}                                  # equation -> (iteration, relc) of the Newton loop running
    nl_block: dict = {}                                # equation -> its Newton loop in the latest coupling iteration
    ss_last: dict = {}
    has_ss: set = set()

    def judge_nl(eq, seq):
        it, rel = seq
        sp = solvers[eq]
        rec = nl_bad.setdefault(eq, [0, 0, 0.0, sp["nl_maxit"], sp["nl_tol"]])
        rec[1] += 1
        if sp["nl_maxit"] > 1 and it >= sp["nl_maxit"] and rel > sp["nl_tol"]:
            rec[0] += 1
            rec[2] = max(rec[2], rel)

    def close_ss():
        top = max(it for it, _r in ss_last.values())
        ss_bad[1] += 1
        if ss_maxit > 1 and top >= ss_maxit:
            over = [rel for eq, (_it, rel) in ss_last.items() if rel > solvers[eq]["ss_tol"]]
            if over:
                ss_bad[0] += 1
                ss_bad[2] = max(ss_bad[2], max(over))
        # only the Newton loops of the last coupling iteration count: earlier ones run on a partner field that is
        # still changing, and stopping them at their maximum is how a segregated iteration normally proceeds
        for eq, seq in nl_block.items():
            judge_nl(eq, seq)
        nl_block.clear()
        ss_last.clear()

    for _n, eq, kind, relc, it in conv:
        eq = eq.strip().lower()
        if eq not in solvers:
            continue
        if kind == "NS":
            if eq in nl_cur and it <= nl_cur[eq][0]:   # a new Newton loop (no coupling line in between)
                seq = nl_cur.pop(eq)
                if eq not in has_ss:
                    judge_nl(eq, seq)
            nl_cur[eq] = (it, relc)
        elif kind == "SS":
            has_ss.add(eq)
            if eq in ss_last and it <= ss_last[eq][0]:
                close_ss()
            ss_last[eq] = (it, relc)
            if eq in nl_cur:
                nl_block[eq] = nl_cur.pop(eq)
    if ss_last:
        close_ss()
    for eq, seq in nl_cur.items():
        judge_nl(eq, seq)
    out = []
    for eq, (bad, total, worst, maxit, tol) in nl_bad.items():
        if bad:
            out.append(f"nonlinear iterations of '{eq}' reached the maximum {maxit} without meeting the tolerance "
                       f"{tol:g} (relative change up to {worst:.3g}) in {bad} of {total} solve(s)")
    if ss_bad[0]:
        out.append(f"the coupling iterations between the physics reached the maximum {ss_maxit} without meeting "
                   f"the coupling tolerance (relative change up to {ss_bad[2]:.3g}) in {ss_bad[0]} of {ss_bad[1]} "
                   f"step(s)")
    return out


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
