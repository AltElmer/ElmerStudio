"""Tests for F06: Asset integrity - saved model reopens only as what it really is."""
import json
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


def test_solutions_current_clean_asset():
    """A clean archive must have solutions_current = True."""
    src = os.path.join(ASSET_DIR, "asset.esm")
    m, work = project.load(src)
    assert m.solutions_current is True, f"clean asset: solutions_current={m.solutions_current}"


def test_solutions_current_tampered_asset():
    """A tampered archive must have solutions_current = False."""
    src = os.path.join(ASSET_DIR, "asset-tampered.esm")
    m, work = project.load(src)
    assert m.solutions_current is False, f"tampered asset: solutions_current={m.solutions_current}"


def test_solutions_current_changed_live_original():
    """A changed live original must have solutions_current = False."""
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
        assert m2.solutions_current is False, f"changed live original: solutions_current={m2.solutions_current}"
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)


def test_solutions_current_missing_asset():
    """A missing asset must have solutions_current = False."""
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
        os.remove(src)
        m2, _ = project.load(bad)
        assert m2.solutions_current is False, f"missing asset: solutions_current={m2.solutions_current}"
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)


def test_solutions_current_no_digest_archive():
    """An archive without digests must have solutions_current = True and report 'digest not available'."""
    src_dir = tempfile.mkdtemp(prefix="es_asset_nodigest_")
    try:
        src = os.path.join(src_dir, "part.step")
        with open(src, "w") as f:
            f.write("ISO-10303-21; original\n")
        m = B.new_model("3D", ["ht"], "stationary")
        imp = m.create("geom.import", m.component.child("geometry"), index=0)
        imp.props["file"] = src
        work = tempfile.mkdtemp(prefix="es_proj_")
        path = os.path.join(tempfile.mkdtemp(prefix="es_projfile_"), "no_digest.mph")
        project.save(m, path, work)
        # Modify manifest to remove sha256
        bad = path.replace(".mph", "_nodigest.mph")
        with zipfile.ZipFile(path) as zin, zipfile.ZipFile(bad, "w") as zout:
            for it in zin.infolist():
                data = zin.read(it.filename)
                if it.filename == "manifest.json":
                    manifest = json.loads(data.decode("utf-8"))
                    # Remove sha256 from assets
                    for key, info in manifest.get("assets", {}).items():
                        info.pop("sha256", None)
                        info.pop("size", None)
                    data = json.dumps(manifest).encode("utf-8")
                zout.writestr(it, data)
        m2, _ = project.load(bad)
        assert m2.solutions_current is True, f"no-digest asset: solutions_current={m2.solutions_current}"
        assert any("digest not available" in p for p in m2.load_problems), \
            f"no-digest asset missing problem: {m2.load_problems}"
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)


def test_no_digest_archive_still_relinks_to_embedded_when_original_missing():
    """An archive without digests must still extract and relink to the embedded copy when the original is missing."""
    src_dir = tempfile.mkdtemp(prefix="es_asset_nodigest_missing_")
    try:
        src = os.path.join(src_dir, "part.step")
        with open(src, "w") as f:
            f.write("ISO-10303-21; original\n")
        m = B.new_model("3D", ["ht"], "stationary")
        imp = m.create("geom.import", m.component.child("geometry"), index=0)
        imp.props["file"] = src
        work = tempfile.mkdtemp(prefix="es_proj_")
        path = os.path.join(tempfile.mkdtemp(prefix="es_projfile_"), "no_digest_missing.mph")
        project.save(m, path, work)
        # Modify manifest to remove sha256
        bad = path.replace(".mph", "_nodigest_missing.mph")
        with zipfile.ZipFile(path) as zin, zipfile.ZipFile(bad, "w") as zout:
            for it in zin.infolist():
                data = zin.read(it.filename)
                if it.filename == "manifest.json":
                    manifest = json.loads(data.decode("utf-8"))
                    for key, info in manifest.get("assets", {}).items():
                        info.pop("sha256", None)
                        info.pop("size", None)
                    data = json.dumps(manifest).encode("utf-8")
                zout.writestr(it, data)
        # Delete the original file so the embedded copy must be used
        os.remove(src)
        m2, work2 = project.load(bad)
        # The asset should be relinked to the embedded copy (file exists in work2/_assets)
        imp2 = m2.by_tag("imp1")
        asset_path = imp2.props["file"]
        assert os.path.isfile(asset_path), f"embedded asset not extracted: {asset_path}"
        assert "ISO-10303-21; original" in open(asset_path).read(), f"embedded asset content wrong"
        assert m2.solutions_current is True, f"no-digest missing original: solutions_current={m2.solutions_current}"
        assert any("digest not available" in p for p in m2.load_problems), \
            f"no-digest asset missing problem: {m2.load_problems}"
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)