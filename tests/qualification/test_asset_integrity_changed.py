"""CHANGED variant tests for Imported CAD edit (ladder row 8).

CORRECT: test_asset_integrity.py::test_asset_digest_verified_on_load
         test_asset_integrity.py::test_clean_asset_loads_without_problems
WRONG: test_asset_integrity.py::test_changed_live_original_is_reported
       test_asset_integrity.py::test_corrupt_asset_produces_load_problem
CHANGED: Different geometry, physics, units - same integrity guarantees

Note: These tests don't build the imported geometry (fake STEP content),
so they don't add boundary features that require geometry selection.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile

import pytest

from elmerstudio.core import builder as B
from elmerstudio.core import project


def test_asset_integrity_changed_geometry():
    """CHANGED: Different STEP geometry file, different physics (structural)."""
    src_dir = tempfile.mkdtemp(prefix="es_asset_changed_")
    try:
        src = os.path.join(src_dir, "bracket.step")
        with open(src, "w") as f:
            f.write("ISO-10303-21; HEADER; FILE_NAME('bracket.step'); ENDSEC; DATA; ENDSEC; END-ISO-10303-21;\n")

        m = B.new_model("3D", ["solid"], "stationary")
        imp = m.create("geom.import", m.component.child("geometry"), index=0)
        imp.props["file"] = src
        B.add_material(m, "Structural steel", all_domains=True)

        work = tempfile.mkdtemp(prefix="es_proj_")
        path = os.path.join(tempfile.mkdtemp(prefix="es_projfile_"), "bracket.mph")
        project.save(m, path, work)

        m2, work2 = project.load(path)
        assert m2.load_problems == [], f"clean changed asset has problems: {m2.load_problems}"
        assert m2.solutions_current is True, f"solutions_current={m2.solutions_current}"

        # Tamper with the embedded asset
        bad = path.replace(".mph", "_bad.mph")
        with zipfile.ZipFile(path) as zin, zipfile.ZipFile(bad, "w") as zout:
            for it in zin.infolist():
                data = zin.read(it.filename)
                if "asset" in it.filename and it.filename.endswith(".step"):
                    data = b"ISO-10303-21; CORRUPTED\n"
                zout.writestr(it, data)

        os.remove(src)
        m3, _ = project.load(bad)
        assert any("missing asset" in p or "damaged" in p for p in m3.load_problems), \
            f"corrupt changed asset not detected: {m3.load_problems}"
        assert m3.solutions_current is False, f"corrupt asset allowed current solution"

    finally:
        shutil.rmtree(src_dir, ignore_errors=True)


def test_asset_integrity_changed_units():
    """CHANGED: Same geometry but different unit system (cm instead of mm)."""
    src_dir = tempfile.mkdtemp(prefix="es_asset_units_")
    try:
        src = os.path.join(src_dir, "part_cm.step")
        with open(src, "w") as f:
            f.write("ISO-10303-21; cm units\n")

        m = B.new_model("3D", ["ht"], "stationary")
        m.global_defs.child("params").props["table"] = [
            {"name": "L", "expr": "10[cm]", "descr": "Length in cm"},
        ]
        imp = m.create("geom.import", m.component.child("geometry"), index=0)
        imp.props["file"] = src

        work = tempfile.mkdtemp(prefix="es_proj_")
        path = os.path.join(tempfile.mkdtemp(prefix="es_projfile_"), "units_cm.mph")
        project.save(m, path, work)

        m2, work2 = project.load(path)
        assert m2.load_problems == [], f"cm units asset has problems: {m2.load_problems}"
        assert m2.solutions_current is True

    finally:
        shutil.rmtree(src_dir, ignore_errors=True)


def test_asset_integrity_multiphysics():
    """CHANGED: Multiphysics model with imported CAD."""
    src_dir = tempfile.mkdtemp(prefix="es_asset_mp_")
    try:
        src = os.path.join(src_dir, "thermal_struct.step")
        with open(src, "w") as f:
            f.write("ISO-10303-21; multiphysics\n")

        m = B.new_model("3D", ["ht", "solid"], "stationary")
        imp = m.create("geom.import", m.component.child("geometry"), index=0)
        imp.props["file"] = src
        B.add_material(m, "Copper", all_domains=True)
        B.add_material(m, "Structural steel", all_domains=True)

        work = tempfile.mkdtemp(prefix="es_proj_")
        path = os.path.join(tempfile.mkdtemp(prefix="es_projfile_"), "multiphysics.mph")
        project.save(m, path, work)

        m2, work2 = project.load(path)
        assert m2.load_problems == [], f"multiphysics asset has problems: {m2.load_problems}"
        assert m2.solutions_current is True

    finally:
        shutil.rmtree(src_dir, ignore_errors=True)


def test_asset_integrity_time_dependent():
    """CHANGED: Time-dependent model with imported CAD."""
    src_dir = tempfile.mkdtemp(prefix="es_asset_td_")
    try:
        src = os.path.join(src_dir, "transient.step")
        with open(src, "w") as f:
            f.write("ISO-10303-21; transient\n")

        m = B.new_model("3D", ["tds"], "time")
        imp = m.create("geom.import", m.component.child("geometry"), index=0)
        imp.props["file"] = src
        B.add_feature(m, m.component.child("physics.tds"), "tds.conc",
                      [], c0="0")  # Empty selection - don't build geometry

        work = tempfile.mkdtemp(prefix="es_proj_")
        path = os.path.join(tempfile.mkdtemp(prefix="es_projfile_"), "transient.mph")
        project.save(m, path, work)

        m2, work2 = project.load(path)
        assert m2.load_problems == [], f"time-dependent asset has problems: {m2.load_problems}"
        assert m2.solutions_current is True

    finally:
        shutil.rmtree(src_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])