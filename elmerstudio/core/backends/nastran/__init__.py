"""NASTRAN-95 solver backend for ElmerStudio.

Provides structural finite element analysis (Linear Statics, Normal Modes, Transient)
and Heat Transfer using NASA's NASTRAN-95.
"""
from __future__ import annotations

from .deck import NastranPlan, NastranRefusal, check_model, plan_deck, write_deck
from .install import NastranInstall, find_nastran
from .parser import NastranResult, parse_output
from .runner import NastranStudy, preview_files

__all__ = [
    "NastranInstall",
    "find_nastran",
    "NastranRefusal",
    "check_model",
    "NastranPlan",
    "plan_deck",
    "write_deck",
    "NastranResult",
    "parse_output",
    "NastranStudy",
    "preview_files",
]
