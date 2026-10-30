"""Solver backends behind a study: ElmerSolver (the default), OpenFOAM (blockMesh), or NASTRAN-95.

The choice is the study's ``backend`` property; models without it are Elmer studies."""
from __future__ import annotations

BACKENDS = [("elmer", "Elmer"), ("openfoam", "OpenFOAM (blockMesh)"), ("nastran", "NASTRAN-95")]


def study_backend(study) -> str:
    b = str(study.get("backend", "elmer") or "elmer") if study is not None else "elmer"
    return b if b in dict(BACKENDS) else "elmer"
