"""Recovery checkpoints: the unsaved model of a running session, so a crash or power loss does not lose it.

Every session owns a folder ``<root>/<pid>-<process start time>/`` holding ``model.json`` (the model) and
``info.json`` (when, which file, which scratch folder, which job was running). The window writes a checkpoint when
the model has unsaved changes (on a timer and before a computation starts) and removes the folder when it closes
normally. A folder whose process is no longer running is left over from an interrupted session; the next start offers
to recover it. The scratch folder of the interrupted session (its solutions) is reused when it still exists.

The root is ``$ELMERSTUDIO_RECOVERY_DIR`` when set (tests), else a per-user application-data folder.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import sys
import tempfile

FORMAT = 1


def root() -> str:
    env = os.environ.get("ELMERSTUDIO_RECOVERY_DIR")
    if env:
        return env
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return os.path.join(base, "ElmerStudio", "recovery")


def _proc_start(pid: int) -> float | None:
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:  # noqa: BLE001 - no such process, or no access
        return None


def _atomic_write(path: str, text: str):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".part_", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


class Session:
    """This process's recovery folder."""

    def __init__(self, base: str | None = None):
        self.pid = os.getpid()
        self.started = _proc_start(self.pid) or 0.0
        self.dir = os.path.join(base or root(), f"{self.pid}-{int(self.started)}")
        self.info: dict = {}

    def checkpoint(self, model_dict: dict, file_path: str | None, workdir: str | None, label: str = ""):
        os.makedirs(self.dir, exist_ok=True)
        _atomic_write(os.path.join(self.dir, "model.json"), json.dumps(model_dict))
        self.info.update({"format": FORMAT, "pid": self.pid, "process_started": self.started,
                          "saved": _dt.datetime.now().isoformat(timespec="seconds"), "file_path": file_path or "",
                          "label": label, "workdir": workdir or ""})
        self._write_info()

    def note_job(self, title: str | None):
        """Record the job that is running (None: none), so a crash during it can be explained."""
        if title:
            self.info["job"] = {"title": title, "started": _dt.datetime.now().isoformat(timespec="seconds")}
        else:
            self.info.pop("job", None)
        if os.path.isfile(os.path.join(self.dir, "model.json")):
            self._write_info()

    def _write_info(self):
        _atomic_write(os.path.join(self.dir, "info.json"), json.dumps(self.info, indent=1))

    def has_checkpoint(self) -> bool:
        return os.path.isfile(os.path.join(self.dir, "model.json"))

    def discard(self):
        """The model was saved or closed on purpose: nothing to recover."""
        shutil.rmtree(self.dir, ignore_errors=True)
        self.info = {}


def _alive(info: dict) -> bool:
    pid = int(info.get("pid", -1))
    if pid == os.getpid():
        return True
    started = _proc_start(pid)
    # a different process may have reused the pid: compare its start time
    return started is not None and abs(started - float(info.get("process_started", 0))) < 2.0


def orphans(base: str | None = None) -> list[dict]:
    """Checkpoints of sessions that ended without closing their model: newest first. Each entry is its info plus
    ``dir``."""
    base = base or root()
    out = []
    if not os.path.isdir(base):
        return out
    for name in os.listdir(base):
        d = os.path.join(base, name)
        try:
            with open(os.path.join(d, "info.json"), encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, ValueError):
            continue
        if not os.path.isfile(os.path.join(d, "model.json")) or _alive(info):
            continue
        info["dir"] = d
        out.append(info)
    out.sort(key=lambda i: i.get("saved", ""), reverse=True)
    return out


def load(entry: dict):
    """(Model, info) of an orphaned checkpoint."""
    from .model import Model
    with open(os.path.join(entry["dir"], "model.json"), encoding="utf-8") as f:
        return Model.from_dict(json.load(f)), entry


def discard(entry: dict):
    shutil.rmtree(entry["dir"], ignore_errors=True)
