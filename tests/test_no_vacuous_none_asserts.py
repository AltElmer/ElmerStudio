"""Structural guard: no test may gate an ``assert`` behind ``if <x> is not None:``.

This is the defect class behind PR #20 finding F1: ``col_map`` was built from the
wrong metadata filename (``circuit.names`` instead of ``circuit.dat.names``), so it
stayed empty, every ``if col is not None: assert ...`` block was silently skipped,
and the test passed vacuously. An ``assert`` placed inside an ``is not None`` guard
only runs when the value happens to be present -- a missing value turns the check
into a no-op instead of a failure.

The guard scans every file under ``tests/`` and fails if it finds an
``if ... is not None:`` line whose immediately following non-blank line is an
(indented) ``assert``. Such constructs must be rewritten as unconditional
``assert`` statements (the value's presence should itself be asserted, not used
to skip the check).
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"

IF_NONE_RE = re.compile(r"^\s*if\s+.+\s+is\s+not\s+None\s*:\s*$")
ASSERT_RE = re.compile(r"^\s*assert\b")


def _collect_violations(root: Path) -> list[str]:
    violations: list[str] = []
    for py in sorted(root.rglob("*.py")):
        if py.name == Path(__file__).name:
            continue  # do not scan this guard itself
        lines = py.read_text().splitlines()
        for i, line in enumerate(lines):
            if not IF_NONE_RE.match(line):
                continue
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and ASSERT_RE.match(lines[j]):
                violations.append(f"{py}:{i + 1}: {line.strip()}  -> {lines[j].strip()}")
    return violations


def test_no_vacuous_is_not_none_assert_guard():
    violations = _collect_violations(TESTS_DIR)
    assert not violations, (
        "vacuous 'is not None'-guarded assert found (F1 defect class):\n"
        + "\n".join(violations)
    )
