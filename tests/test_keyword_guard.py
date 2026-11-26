"""Keyword guard (part a): every `Keyword = ...` Elmer Studio emits must be listed
in the installed SOLVER.KEYWORDS or in tests/keyword_allowlist.txt.

The guard checks keyword *names* (SOLVER.KEYWORDS is the source of truth for names);
part (b) (study_runner scanning solver.log for "Unlisted keyword" / "not used"
messages) catches keywords that are listed somewhere but used in the wrong section,
e.g. a Boundary Condition keyword emitted in an Initial Condition.
"""
from __future__ import annotations

import os
import re

import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.examples import EXAMPLES
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.sif import build_sif

_KEYWORD = re.compile(r"^\s+([A-Za-z][A-Za-z0-9 ]*?)\s*=\s", re.M)
_ARRAY_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")


def _solver_keywords_path(inst) -> str:
    return os.path.join(inst.home, "share", "elmersolver", "lib", "SOLVER.KEYWORDS")


def _load_valid_keywords(path: str) -> set[str]:
    """Keyword *names* listed in SOLVER.KEYWORDS: '<Section>:<Type>: '<Keyword>''."""
    names: set[str] = set()
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("$"):
                continue
            m = re.match(r"^[A-Za-z]+\:\w+\:\s*'(.+?)'\s*$", line)
            if m:
                names.add(_base(m.group(1)).lower())
    return names


def _load_allowlist(path: str) -> set[str]:
    names: set[str] = set()
    if not os.path.isfile(path):
        return names
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split("#", 1)[0].strip()
            if name:
                names.add(_base(name).lower())
    return names


def _base(name: str) -> str:
    """Strip an Elmer array suffix such as `Coordinate Mapping(3)` -> `Coordinate Mapping`."""
    return _ARRAY_SUFFIX.sub("", name).strip()


def _emit_keywords(sif_text: str) -> set[str]:
    return {_base(m.group(1)).lower() for m in _KEYWORD.finditer(sif_text)}


def _missing(sif_text: str, valid: set[str], allowed: set[str]) -> set[str]:
    return _emit_keywords(sif_text) - (valid | allowed)


def _build_all_sifs():
    """Yield (example_key, sif_text) for every node/study type the package emits."""
    for key, (_, maker) in sorted(EXAMPLES.items()):
        m = maker()
        geo = build_geometry(m, display=False)
        st = m.studies()[0]
        step = B.study_steps(st)[0]
        sif = build_sif(m, st, step, geo, 1000).text
        yield key, sif


# ------------------------------------------------------------------ guard itself


@pytest.fixture
def keyword_rules(elmer):
    """(valid keyword names, allowlist names) from the installed Elmer + allowlist."""
    valid = _load_valid_keywords(_solver_keywords_path(elmer))
    allowlist = os.path.join(os.path.dirname(__file__), "keyword_allowlist.txt")
    allowed = _load_allowlist(allowlist)
    return valid, allowed


def test_emitted_keywords_are_listed(keyword_rules):
    """Every keyword Elmer Studio emits is in SOLVER.KEYWORDS (or the allowlist)."""
    valid, allowed = keyword_rules
    problems = {}
    for key, sif in _build_all_sifs():
        miss = _missing(sif, valid, allowed)
        if miss:
            problems[key] = sorted(miss)
    assert not problems, f"emitted keywords not in SOLVER.KEYWORDS/allowlist: {problems}"


def test_allowlist_entries_cite_an_elmer_source():
    """Documented contract: every allowlist entry cites the Elmer source that reads it."""
    allowlist = os.path.join(os.path.dirname(__file__), "keyword_allowlist.txt")
    with open(allowlist, errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "#" not in stripped:
                name = stripped.split()[0] if stripped else ""
                raise AssertionError(
                    f"tests/keyword_allowlist.txt:{lineno}: entry {name!r} has no Elmer source citation"
                )


# ------------------------------------------------------------------ negative control


def test_invented_keyword_is_rejected(keyword_rules):
    """An invented keyword must fail the guard (not in SOLVER.KEYWORDS or allowlist)."""
    valid, allowed = keyword_rules
    key, sif = next(iter(_build_all_sifs()))
    bogus = sif.replace("Boundary Condition 1", "Boundary Condition 1\n  BogusKeyXyz = Real 1.23", 1)
    miss = _missing(bogus, valid, allowed)
    assert "boguskeyxyz" in miss, f"invented keyword not rejected; missing={sorted(miss)}"
