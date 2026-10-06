"""Model files (.esm): a zip holding model.json and, optionally, the computed solutions."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile

from . import builder as _builder  # noqa: F401  (registers all node types before loading)
from .model import Model


def new_workdir() -> str:
    return tempfile.mkdtemp(prefix="elmerstudio_")


def save(model: Model, path: str, workdir: str | None, include_solutions=True):
    tmp = path + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.json", model.to_json())
        if include_solutions and workdir and os.path.isdir(workdir):
            for root, _dirs, files in os.walk(workdir):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, workdir)
                    if rel.startswith("_"):
                        continue
                    if "results" in rel.split(os.sep) or f in ("meta.json", "mesh.npz", "case.sif", "solver.log"):
                        z.write(full, "work/" + rel.replace(os.sep, "/"))
    os.replace(tmp, path)


def load(path: str) -> tuple[Model, str]:
    workdir = new_workdir()
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("model.json").decode("utf-8"))
        for name in z.namelist():
            if name.startswith("work/") and not name.endswith("/"):
                target = os.path.join(workdir, *name[5:].split("/"))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    m = Model.from_dict(data)
    m.file_path = path
    return m, workdir
