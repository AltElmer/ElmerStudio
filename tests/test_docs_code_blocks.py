"""Every Python source block in docs/modules/reference/pages/programming.adoc
and docs/modules/reference/pages/headless.adoc must run against the real API.

Each block is executed end to end (all blocks of a page share one namespace,
like a notebook, with CWD redirected to a scratch directory), and every
module-level attribute a block touches is resolved against the real
elmerstudio.core modules. A snippet that calls a nonexistent builder method
(add_parameters, add_block, add_difference ...) fails here, so the
documentation cannot drift from the API unnoticed.
"""
from __future__ import annotations

import ast
import builtins
import csv
import json
import os
import re
from pathlib import Path

import pytest

from elmerstudio.core import builder as B
from elmerstudio.core.elmer import find_elmer
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.headless import _derived_table, run_headless
from elmerstudio.core.project import discard_workdir, load, new_workdir, save
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner

try:  # numpy is optional for the doc snippets themselves
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

ROOT = Path(__file__).resolve().parent.parent
PAGES = [
    ROOT / "docs" / "modules" / "reference" / "pages" / "programming.adoc",
    ROOT / "docs" / "modules" / "reference" / "pages" / "headless.adoc",
]

BLOCK_RE = re.compile(r"\[source,python\]\n----\n(.*?)\n----", re.S)

NAMESPACE = {
    "B": B,
    "build_geometry": build_geometry,
    "run_headless": run_headless,
    "_derived_table": _derived_table,
    "StudyRunner": StudyRunner,
    "find_elmer": find_elmer,
    "new_workdir": new_workdir,
    "discard_workdir": discard_workdir,
    "save": save,
    "load": load,
    "Solution": Solution,
    "csv": csv,
    "json": json,
    "os": os,
    "np": np,
}


def _python_blocks() -> list[tuple[str, str]]:
    found = []
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        found.extend((page.name, block) for block in BLOCK_RE.findall(text))
    return found


def _resolve_against_real_modules(block: str) -> None:
    """Every attribute access whose root is a module-level name must exist on
    the real module. Names created at runtime (m, sol, folders ...) are
    checked by executing the block instead.
    """
    tree = ast.parse(block)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = []
        cur = node
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            continue  # attribute on a call result: checked at run time
        root = cur.id
        if root not in NAMESPACE and root not in dir(builtins):
            continue  # free name (e.g. print): executed at run time
        if root in NAMESPACE and NAMESPACE[root] is not None:
            obj = NAMESPACE[root]
            for attr in reversed(chain):
                if not hasattr(obj, attr):
                    raise AssertionError(
                        f"{root}.{'.'.join(reversed(chain))} does not exist on the real "
                        f"{type(obj).__module__}.{type(obj).__qualname__}; the docs block "
                        f"calls an API that is not implemented"
                    )
                obj = getattr(obj, attr)


def test_docs_code_blocks_run_end_to_end(tmp_path, elmer):
    ns = dict(NAMESPACE)
    ns["tmp_path"] = tmp_path
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        for page_name, block in _python_blocks():
            _resolve_against_real_modules(block)
            exec(compile(block, f"<docs {page_name}>", "exec"), ns)
    finally:
        os.chdir(old_cwd)
