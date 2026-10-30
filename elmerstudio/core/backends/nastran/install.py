"""Locate a NASTRAN-95 installation and execute runs with its rigid format sequences.

Discovery order:
1. NASTRAN_WSL_HOME environment variable (WSL build path, Windows only)
2. Explicit setting (Preferences > NASTRAN installation folder)
3. Environment variables: NASTRAN_ROOT, NASTRAN_EXE, NASTRAN95_ROOT, NASTRAN_HOME
4. Standard local paths and PATH lookup
5. Windows WSL installation if on Windows
"""
from __future__ import annotations

import glob
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass

_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class NastranInstall:
    root: str                   # directory containing the executable or repo root
    exe: str                    # path to nastran executable
    rf_dir: str                 # directory containing rigid format files (AERO*, DISP*, HEAT*)
    kind: str                   # windows | linux | macos | wsl
    version: str = "95 (v1.0.0)"
    wsl_distro: str | None = None

    def env(self) -> dict:
        env = dict(os.environ)
        if self.rf_dir:
            env["RFDIR"] = self.rf_dir
        return env

    def prepare_case(self, case_dir: str):
        """Ensure the rigid format directory 'rf' is accessible inside case_dir.

        NASTRAN-95 opens './rf/<sequence>' during execution. We link or copy it.
        """
        target = os.path.join(case_dir, "rf")
        if os.path.exists(target):
            return
        if not self.rf_dir or not os.path.isdir(self.rf_dir):
            return

        # WSL: skip here; argv creates the symlink inside WSL ( avoids broken
        # Windows symlinks when later zipping the project for save/reload )
        if self.kind == "wsl":
            return

        if os.name == "nt":
            try:
                import _winapi
                _winapi.CreateJunction(self.rf_dir, target)
                return
            except Exception:
                pass
            try:
                os.symlink(self.rf_dir, target, target_is_directory=True)
                return
            except Exception:
                pass
            try:
                shutil.copytree(self.rf_dir, target)
            except Exception:
                pass
        else:
            try:
                os.symlink(self.rf_dir, target)
            except Exception:
                try:
                    shutil.copytree(self.rf_dir, target)
                except Exception:
                    pass

    def argv(self, case_dir: str, deck_file: str) -> list[str]:
        """Command line list to execute NASTRAN."""
        if self.kind == "wsl":
            wsl = shutil.which("wsl") or "wsl.exe"
            # NASTRAN-95 on WSL has I/O errors on /mnt/c (Windows drive).
            # Copy the deck to a WSL-native temp dir, run there, then copy results back.
            wsl_case = case_dir.replace("\\", "/").replace("C:", "/mnt/c")
            rel_deck = os.path.basename(deck_file)
            out_files = "case.out case.err"
            # Build command without nested quotes in f-string; write to a temp script
            # to avoid exec replacing shell and losing copy-back.
            script = (
                "tmpd=$(mktemp -d) && "
                "cp -r '" + wsl_case + "'/* " + "$tmpd/ && "
                "cd " + "$tmpd && "
                "RFDIR='" + self.rf_dir + "' " + self.exe + " < '" + rel_deck + "' && "
                "cp -f " + out_files + " '" + wsl_case + "/' 2>/dev/null || true"
            )
            return [wsl, "-e", "bash", "-lc", script]
        return [self.exe]

    def popen_kwargs(self) -> dict:
        if os.name == "nt" and self.kind != "wsl":
            return {"creationflags": _NOWIN}
        return {"start_new_session": True}

    def kill(self, proc: subprocess.Popen, case_dir: str | None = None):
        """Terminate the running NASTRAN process and any children."""
        if os.name == "nt" and self.kind != "wsl":
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, timeout=10, creationflags=_NOWIN)
            except Exception:
                pass
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        try:
            proc.kill()
        except Exception:
            pass

    def describe(self) -> str:
        where = {"windows": "Windows native", "linux": "Linux native", "macos": "macOS native", "wsl": "Linux (WSL)"}.get(self.kind, self.kind)
        return f"NASTRAN {self.version} ({where}) at {self.exe}"


_WSL_CACHE: NastranInstall | None = None


def _find_wsl_nastran() -> NastranInstall | None:
    global _WSL_CACHE
    if os.name != "nt":
        return None
    if _WSL_CACHE is not None:
        return _WSL_CACHE
    wsl = shutil.which("wsl")
    if not wsl:
        return None
    probe = (
        "for d in /mnt/c/Users/*/Downloads/*/NASTRAN-95 /mnt/c/Users/*/Downloads/*/nastran* "
        "/usr/local/bin /opt/nastran* ~/.local/bin; do "
        "  if [ -x \"$d/dist/bin/nastran\" ] && [ -d \"$d/rf\" ]; then echo \"$d/dist/bin/nastran|$d/rf\"; exit 0; fi; "
        "  if [ -x \"$d/nastran\" ] && [ -d \"$d/rf\" ]; then echo \"$d/nastran|$d/rf\"; exit 0; fi; "
        "done; which nastran 2>/dev/null || true"
    )
    try:
        r = subprocess.run([wsl, "-e", "bash", "-lc", probe], capture_output=True, text=True, timeout=15, creationflags=_NOWIN)
        out = r.stdout.strip()
        if out and "|" in out:
            exe, rf = out.split("|", 1)
            _WSL_CACHE = NastranInstall(root=os.path.dirname(exe), exe=exe, rf_dir=rf, kind="wsl")
            return _WSL_CACHE
        elif out and out.endswith("nastran"):
            _WSL_CACHE = NastranInstall(root=os.path.dirname(out), exe=out, rf_dir="", kind="wsl")
            return _WSL_CACHE
    except Exception:
        pass
    return None


def _check_dir(d: str) -> NastranInstall | None:
    if not os.path.isdir(d):
        return None
    exe_names = ["nastran.exe", "nastran", "nastrn.exe", "nastrn"]
    found_exe = None
    for name in exe_names:
        cand = os.path.join(d, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            found_exe = cand
            break
        cand_dist = os.path.join(d, "dist", "bin", name)
        if os.path.isfile(cand_dist) and os.access(cand_dist, os.X_OK):
            found_exe = cand_dist
            break

    if not found_exe:
        return None

    # Check for rf directory
    rf_dir = os.path.join(d, "rf")
    if not os.path.isdir(rf_dir):
        rf_parent = os.path.join(os.path.dirname(d), "rf")
        if os.path.isdir(rf_parent):
            rf_dir = rf_parent
        else:
            rf_dir = ""

    # Determine OS kind safely (os.uname() doesn't exist on Windows)
    if os.name == "nt" and found_exe.lower().endswith(".exe"):
        kind = "windows"
    elif os.name != "nt":
        import platform
        kind = "macos" if platform.system() == "Darwin" else "linux"
    else:
        kind = "linux"  # WSL or Linux-style path on Windows
    return NastranInstall(root=d, exe=found_exe, rf_dir=rf_dir, kind=kind)


def find_nastran(preferred: str | None = None) -> NastranInstall | None:
    """Find an installed NASTRAN-95 package.

    On Windows, NASTRAN_WSL_HOME environment variable can be set to the WSL path
    of a verified build (e.g., /home/<user>/NASTRAN-95). This takes precedence
    over auto-discovery to avoid broken native builds.
    """
    candidates = []
    if preferred:
        candidates.append(preferred)

    # NASTRAN_WSL_HOME takes precedence on Windows (explicit WSL build selection)
    if os.name == "nt":
        wsl_home = os.environ.get("NASTRAN_WSL_HOME")
        # Git Bash mangles /home/... into C:/Program Files/Git/home/...
        # Undo that by detecting the pattern and keeping only from /home/.
        if wsl_home and "Program Files/Git/home" in wsl_home:
            wsl_home = wsl_home.replace("\\", "/")
            idx = wsl_home.find("/home/")
            if idx >= 0:
                wsl_home = wsl_home[idx:]
        if wsl_home:
            # Build the WSL executable path (use forward slashes for WSL)
            wsl_home = wsl_home.replace("\\", "/")
            wsl_exe = f"{wsl_home}/dist/bin/nastran"
            wsl_rf = f"{wsl_home}/rf"
            # Verify via WSL that the file exists
            wsl = shutil.which("wsl") or "wsl.exe"
            try:
                result = subprocess.run(
                    [wsl, "-e", "bash", "-c", f"[ -x '{wsl_exe}' ] && [ -d '{wsl_rf}' ] && echo OK"],
                    capture_output=True, text=True, timeout=10, creationflags=_NOWIN
                )
                if "OK" in result.stdout:
                    return NastranInstall(
                        root=wsl_home, exe=wsl_exe, rf_dir=wsl_rf, kind="wsl",
                        wsl_distro="default"
                    )
            except Exception:
                pass
    
    for env_var in ("NASTRAN_ROOT", "NASTRAN_EXE", "NASTRAN95_ROOT", "NASTRAN_HOME"):
        val = os.environ.get(env_var)
        if val:
            if os.path.isfile(val):
                candidates.append(os.path.dirname(val))
            else:
                candidates.append(val)

    # Standard locations (no hardcoded personal paths)
    home = os.path.expanduser("~")
    if os.name == "nt":
        candidates.extend([
            os.path.join(home, "Downloads", "nastran95-v1.0.0-windows-x86_64"),
            os.path.join(home, "Downloads", "NASTRAN-95"),
            r"C:\nastran95",
            r"C:\Program Files\NASTRAN-95",
        ])
        # Glob downloads
        candidates.extend(glob.glob(os.path.join(home, "Downloads", "*nastran*")))
    else:
        candidates.extend([
            "/usr/local/bin",
            "/opt/nastran95",
            "/opt/NASTRAN-95",
            os.path.join(home, ".local", "bin"),
            os.path.join(home, "Downloads", "NASTRAN-95"),
        ])

    which_nast = shutil.which("nastran") or shutil.which("nastran.exe")
    if which_nast:
        candidates.append(os.path.dirname(which_nast))

    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        inst = _check_dir(cand)
        if inst:
            return inst

    # Fallback to WSL auto-discovery if on Windows
    if os.name == "nt":
        wsl_inst = _find_wsl_nastran()
        if wsl_inst:
            return wsl_inst

    return None
