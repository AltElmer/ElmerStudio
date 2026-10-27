"""Context help: the documentation page (and section) that describes a Model Builder node.

The Reference pages are generated from the node registry (tools/gen_docs_reference.py), with one anchor per node
type, so F1 on a node opens the section of the page that lists exactly the settings the Settings window shows.
"""
from __future__ import annotations

# reference page -> node kinds it documents (a kind matches a prefix exactly or as "<prefix>.<...>")
REFERENCE_PAGES = [
    ("definitions", ["global", "params", "func", "definitions", "variables", "cpl"]),
    ("geometry", ["geometry", "geom"]),
    ("materials", ["materials", "material"]),
    ("mesh", ["mesh"]),
    ("studies", ["study"]),
    ("datasets", ["dset"]),
    ("derived-values", ["eval", "table"]),
    ("plots", ["pg3d", "pg2d", "pg1d", "pgpolar", "plot"]),
    ("export", ["export"]),
    ("multiphysics", ["multiphysics", "mp"]),
]
# nodes without a reference section: the user-guide page about them
GUIDE_PAGES = {"root": "user-guide/models-and-files", "component": "user-guide/models-and-files",
               "results": "user-guide/results", "results.datasets": "user-guide/results",
               "results.derived": "user-guide/results", "results.tables": "user-guide/results",
               "results.export": "user-guide/results"}


def anchor(kind: str) -> str:
    return kind.replace(".", "-").replace("@", "")


def _matches(kind: str, prefixes) -> bool:
    return any(kind == p or kind.startswith(p + ".") for p in prefixes)


def help_page(kind: str) -> tuple[str, str]:
    """(page path below the site root without .html, anchor or "") for a node kind."""
    if kind in GUIDE_PAGES:
        return GUIDE_PAGES[kind], ""
    from .physics import PHYSICS
    for pid, d in PHYSICS.items():
        if kind == d.kind:
            return f"reference/physics-{pid}", ""
        if _matches(kind, [pid]):
            return f"reference/physics-{pid}", anchor(kind)
    for page, prefixes in REFERENCE_PAGES:
        if _matches(kind, prefixes):
            return f"reference/{page}", anchor(kind)
    return "user-interface", ""


def help_url(kind: str | None, base: str) -> str:
    if not kind:
        return base
    page, anc = help_page(kind)
    return base.rstrip("/") + f"/{page}.html" + (f"#{anc}" if anc else "")
