"""Locate an OpenFOAM installation and build the command lines that run its utilities in a case folder.

OpenFOAM's binaries resolve only after its ``etc/bashrc`` is sourced.  Three ways to get there:

* ``msys``: the ESI Windows (MinGW) tree.  ``etc/bashrc`` must be sourced inside the tree's OWN bundled msys64 bash,
  with HOME=/home/ofuser and MSYSTEM=MSYS, and case paths translated by that bash's cygpath;
* ``linux``: a Linux install (the openfoam.com packages put it in /usr/lib/openfoam/openfoamNNNN);
* ``wsl``: a Linux install inside WSL, driven from Windows through wsl.exe (case paths through wslpath).

Discovery order: explicit setting -> OPENFOAM_ROOT -> WM_PROJECT_DIR -> common install locations -> (Windows) WSL.
OPENFOAM_ROOT may name a Windows tree, a Linux project directory, or ``wsl:/usr/lib/openfoam/openfoam2606``.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass

_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class FoamInstall:
    root: str                 # msys: the tree holding msys64; linux/wsl: the OpenFOAM project directory
    bash: str                 # msys/linux: bash; wsl: wsl.exe
    bashrc: str               # as the bash sees it
    kind: str                 # msys | linux | wsl
    version: str = ""

    def env(self) -> dict:
        env = dict(os.environ)
        if self.kind == "msys":
            env.update({"MSYSTEM": "MSYS", "HOME": "/home/ofuser", "CHERE_INVOKING": "1"})
        for k in ("WM_PROJECT_DIR", "FOAM_APPBIN", "FOAM_LIBBIN"):   # a sourced environment of another install
            env.pop(k, None)
        return env

    def posix_path(self, path: str) -> str:
        path = os.path.abspath(path)
        if self.kind == "linux":
            return path
        if self.kind == "msys":
            q = path.replace("'", "'\\''")
            argv = [self.bash, "-lc", f"cygpath -u '{q}'"]
        else:
            argv = [self.bash, "-e", "wslpath", "-u", path]
        r = subprocess.run(argv, capture_output=True, text=True, env=self.env(), timeout=60, creationflags=_NOWIN)
        out = r.stdout.strip()
        if r.returncode != 0 or not out:
            raise RuntimeError(f"Could not translate the case path {path} for OpenFOAM: {r.stderr.strip()}")
        return out

    def argv(self, command: str, case: str) -> list[str]:
        """bash -lc "source etc/bashrc; cd <case>; exec <command>": the exit code is the command's, and the
        command's process id goes to .foam_pid so that Stop can kill it inside WSL."""
        c = self.posix_path(case).replace("'", "'\\''")
        inner = f"source '{self.bashrc}' >/dev/null 2>&1; cd '{c}' || exit 97; echo $$ > .foam_pid; exec {command}"
        if self.kind == "wsl":
            return [self.bash, "-e", "bash", "-lc", inner]
        return [self.bash, "-lc", inner]

    def popen_kwargs(self) -> dict:
        if os.name == "nt":
            return {"creationflags": _NOWIN}
        return {"start_new_session": True}

    def kill(self, proc: subprocess.Popen, case: str):
        """Stop the running utility and everything it started."""
        if self.kind == "wsl":
            try:
                with open(os.path.join(case, ".foam_pid")) as f:
                    pid = int(f.read().strip())
                subprocess.run([self.bash, "-e", "kill", "-9", str(pid)], capture_output=True, timeout=30,
                               creationflags=_NOWIN)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, creationflags=_NOWIN)
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            proc.kill()
        except OSError:
            pass

    def describe(self) -> str:
        where = {"msys": "Windows (bundled msys64)", "linux": "Linux", "wsl": "WSL"}[self.kind]
        return f"OpenFOAM {self.version} ({where}) at {self.root}"


def _windows_tree(root: str) -> FoamInstall | None:
    bash = os.path.join(root, "msys64", "usr", "bin", "bash.exe")
    if not os.path.isfile(bash) or os.path.getsize(bash) == 0:      # a 0-byte cloud placeholder is no install
        return None
    rcs = sorted(glob.glob(os.path.join(root, "msys64", "home", "ofuser", "OpenFOAM", "OpenFOAM-v*", "etc", "bashrc")),
                 reverse=True)
    if not rcs:
        return None
    proj = os.path.basename(os.path.dirname(os.path.dirname(rcs[0])))
    return FoamInstall(root, bash, f"/home/ofuser/OpenFOAM/{proj}/etc/bashrc", "msys", proj.replace("OpenFOAM-", ""))


def _linux_tree(root: str) -> FoamInstall | None:
    rc = os.path.join(root, "etc", "bashrc")
    bash = shutil.which("bash") or "/bin/bash"
    if not os.path.isfile(rc) or not os.path.isfile(bash):
        return None
    return FoamInstall(root, bash, rc, "linux", _version_of(root))


def _version_of(root: str) -> str:
    m = re.search(r"(v?\d{4})", os.path.basename(os.path.normpath(root)))
    return ("v" + m.group(1).lstrip("v")) if m else os.path.basename(os.path.normpath(root))


_WSL: dict = {}


def _wsl_tree(root: str | None = None) -> FoamInstall | None:
    """An OpenFOAM inside the default WSL distribution (looked up once per process)."""
    wsl = shutil.which("wsl") if os.name == "nt" else None
    if not wsl:
        return None
    key = root or ""
    if key not in _WSL:
        probe = f"ls -d {root}/etc/bashrc" if root else "ls -d /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc"
        try:
            r = subprocess.run([wsl, "-e", "bash", "-lc", probe + " 2>/dev/null"], capture_output=True, text=True,
                               timeout=60, creationflags=_NOWIN)
            rcs = sorted((ln.strip() for ln in r.stdout.splitlines() if ln.strip().endswith("/etc/bashrc")), reverse=True)
        except (OSError, subprocess.SubprocessError):
            rcs = []
        _WSL[key] = rcs[0] if rcs else None
    rc = _WSL[key]
    if not rc:
        return None
    proj = rc[: -len("/etc/bashrc")]
    return FoamInstall(proj, wsl, rc, "wsl", _version_of(proj))


def _candidates() -> list[str]:
    c = [os.environ.get("OPENFOAM_ROOT", ""), os.environ.get("WM_PROJECT_DIR", "")]
    home = os.path.expanduser("~")
    if os.name == "nt":
        pats = [os.path.join(r, "OpenFOAM*", "v*") for r in ("C:\\", home, os.path.join(home, "Apps"))]
        pats += [os.path.join(r, "OpenFOAM*") for r in ("C:\\", home, os.path.join(home, "Apps"),
                                                         os.environ.get("ProgramFiles", r"C:\Program Files"))]
    else:
        pats = ["/usr/lib/openfoam/openfoam*", "/opt/openfoam*", "/opt/OpenFOAM*/OpenFOAM-*",
                os.path.join(home, "OpenFOAM", "OpenFOAM-*")]
    for p in pats:
        c.extend(sorted(glob.glob(p), reverse=True))
    return c


def find_openfoam(preferred: str | None = None) -> FoamInstall | None:
    seen = set()
    for root in ([preferred] if preferred else []) + _candidates():
        if not root or root in seen:
            continue
        seen.add(root)
        if root.lower().startswith("wsl"):
            inst = _wsl_tree(root.split(":", 1)[1] if ":" in root else None)
        elif not os.path.isdir(root):
            continue
        else:
            inst = _windows_tree(root) if os.name == "nt" else _linux_tree(root)
        if inst is not None:
            return inst
    return _wsl_tree() if os.name == "nt" else None
