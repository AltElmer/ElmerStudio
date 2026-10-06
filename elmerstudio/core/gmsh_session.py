"""Single shared gmsh session.

gmsh is a process-wide singleton and not re-entrant, so every geometry/mesh job takes
this lock. ``interruptible=False`` lets it run off the main thread (no SIGINT hook).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

import gmsh

_lock = threading.RLock()
_initialized = False


@contextmanager
def session(model_name: str = "model"):
    global _initialized
    with _lock:
        if not _initialized:
            try:
                gmsh.initialize(readConfigFiles=False, interruptible=False)
            except TypeError:  # older gmsh without the keyword
                gmsh.initialize()
            _initialized = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.clear()
        gmsh.model.add(model_name)
        try:
            yield gmsh
        finally:
            pass


def logger_messages() -> list[str]:
    try:
        return gmsh.logger.get()
    except Exception:
        return []
