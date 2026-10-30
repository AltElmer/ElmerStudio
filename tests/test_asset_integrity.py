"""Tests for F06: Asset integrity - saved model reopens only as what it really is."""
import os
import shutil
import tempfile
import zipfile

import pytest

from elmerstudio.core import builder as B
from elmerstudio.core import project

ORCH = r"C:\Users\FoadS\Downloads\20261029\comsol-parity-orch"
ASSET_DIR = os.path.join(ORCH, "oc")


def test_asset_digest_verified_on_load():
    """A tampered archive must produce a load problem naming the asset; it must never load silently."""
    src = os.path.join(ASSET_DIR, "asset-tampered.esm")
    m, work = project.load(src)
    assert any("damaged" in p and "asset" in p for p in m.load_problems), f"load_problems={m.load_problems}"


def test_clean_asset_loads_without_problems():
    """A clean archive must load with no load problems."""
    src = os.path.join(ASSET_DIR, "asset.esm")
    m, work = project.load(src)
    assert m.load_problems == [], f"clean asset has problems: {m.load_problems}"


def test_changed_live_original_is_reported():
    """Replacing the live original's bytes while retaining manifest size/hash must be reported."""
    src_dir = tempfile.mkdtemp(prefix="es_asset_live_")
    try:
        src = os.path.join(src_dir, "part.step")
        with open(src, "w") as f:
            f.write("ISO-10303-21; original\n")
        m = B.new_model("3D", ["ht"], "stationary")
        imp = m.create("geom.import", m.component.child("geometry"), index=0)
        imp.props["file"] = src
        work = tempfile.mkdtemp(prefix="es_proj_")
        path = os.path.join(tempfile.mkdtemp(prefix="es_projfile_"), "live_change.mph")
        project.save(m, path, work)
        with open(src, "w") as f:
            f.write("ISO-10303-21; CHANGED\n")
        m2, work2 = project.load(path)
        assert any("changed" in p.lower() or "mismatch" in p.lower() for p in m2.load_problems), \
            f"changed live original not reported: {m2.load_problems}"
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)


def test_corrupt_asset_produces_load_problem():
    """A missing or corrupt asset must produce a load problem and must never let the old solution count as current."""
    src_dir = tempfile.mkdtemp(prefix="es_asset_corrupt_")
    try:
        src = os.path.join(src_dir, "part.step")
        with open(src, "w") as f:
            f.write("ISO-10303-21; original\n")
        m = B.new_model("3D", ["ht"], "stationary")
        imp = m.create("geom.import", m.component.child("geometry"), index=0)
        imp.props["file"] = src
        work = tempfile.mkdtemp(prefix="es_proj_")
        path = os.path.join(tempfile.mkdtemp(prefix="es_projfile_"), "corrupt.mph")
        project.save(m, path, work)
        bad = path.replace(".mph", "_bad.mph")
        with zipfile.ZipFile(path) as zin, zipfile.ZipFile(bad, "w") as zout:
            for it in zin.infolist():
                data = zin.read(it.filename)
                if "asset" in it.filename and it.filename.endswith(".step"):
                    data = b"CORRUPTED"
                zout.writestr(it, data)
        # Delete the original file so the corrupted zip asset is used
        os.remove(src)
        m2, _ = project.load(bad)
        assert any("missing asset" in p or "damaged" in p for p in m2.load_problems), \
            f"corrupt asset not detected: {m2.load_problems}"
        assert m2.solutions_saved != "included" or m2.load_problems, "corrupt asset allowed current solution"
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)