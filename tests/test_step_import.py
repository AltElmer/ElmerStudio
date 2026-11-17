"""Issue #10: STEP import domain repair.

Verifies that importing STEP files through ``geom.import`` keeps every highest-dimension entity
(solids in 3D) as a separate domain, that the optional repair (``heal``) never drops those
entities (falling back to the imported geometry with a warning instead), that coordinates are
scaled from the file's millimetres into metres, and that a multi-body STEP preserves each solid
as its own domain.  A real electrostatics solve on an imported two-conductor geometry proves the
imported mesh is Elmer-ready.

Run as a script for a report, or via pytest (needs Elmer for the solve checks).
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gmsh  # noqa: E402
from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core.geometry import GeometryError, build_geometry  # noqa: E402
from elmerstudio.core.elmer import find_elmer  # noqa: E402
from elmerstudio.core.results import Solution  # noqa: E402
from elmerstudio.core.study_runner import StudyRunner  # noqa: E402

TOL = 1e-6


@pytest.fixture(scope="session", autouse=True)
def _gmsh():
    gmsh.initialize()
    yield
    gmsh.finalize()


def _final():
    gmsh.model.occ.synchronize()


def make_step_single(path):
    """A single solid 10 mm cube authored in millimetres."""
    gmsh.clear()
    gmsh.model.add("single")
    gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10)
    _final()
    gmsh.write(path)


def make_step_multi(path):
    """Two separate solid cubes (one body per solid) in millimetres."""
    gmsh.clear()
    gmsh.model.add("multi")
    gmsh.model.occ.addBox(0, 0, 0, 4, 4, 4)
    gmsh.model.occ.addBox(10, 0, 0, 4, 4, 4)
    _final()
    gmsh.write(path)


def make_step_composite(path):
    """A 10 mm cube containing two 1 mm cubes (3 solids in the STEP)."""
    gmsh.clear()
    gmsh.model.add("composite")
    gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10)
    gmsh.model.occ.addBox(4, 0, 0, 1, 1, 1)
    gmsh.model.occ.addBox(-4, 0, 0, 1, 1, 1)
    _final()
    gmsh.write(path)


def build_imported_model(step_path, unit="mm", repair=True):
    """An Elmer model whose geometry is the imported STEP, with es physics ready to solve."""
    m = B.new_model("3D", ["es"], "stationary")
    m.root.label = "step_import"
    m.global_defs.child("params").props["table"] = [
        {"name": "V0", "expr": "1[V]", "descr": "applied potential"},
    ]
    g = m.component.child("geometry")
    g.props["unit"] = unit
    fin = g.child("geom.finalize")
    idx = list(g.children).index(fin)
    imp = m.create("geom.import", g, index=idx, tag="imp", label="Import 1")
    imp.props["file"] = step_path
    imp.props["heal"] = repair
    imp.props["unit"] = unit
    return m


def box_boundaries(geo, cx):
    """Boundaries whose centroid x-coordinate (in metres) is within 5 mm of *cx*."""
    out = []
    for b in geo.numbers["boundary"]:
        bb = np.array(geo.bbox_of["boundary"][b]).reshape(2, 3)
        if abs(bb.mean(axis=0)[0] - cx) < 0.005:
            out.append(b)
    return out


def setup_physics(m, geo):
    ec = m.component.child("physics.es")
    # boxes are at x = 0 and x = 10 mm = 0.01 m (geometry is scaled to metres)
    B.add_feature(m, ec, "es.pot", box_boundaries(geo, 0.0), V0="V0")
    B.add_feature(m, ec, "es.ground", box_boundaries(geo, 0.01))
    B.add_material(m, "Air", all_domains=True)
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.size", mesh, mode="custom", hmax="2", hmin="1", hcurve="0.2")
    m.create("mesh.ftet", mesh)


def solve(m, st, folder):
    B.ensure_solver_configs(m, st)
    folders = StudyRunner(m, st, folder, find_elmer()).run()
    return [Solution(f) for f in folders]


def _imported_geometry(step_path, repair=True, unit="mm"):
    m = B.new_model("3D", None, None)
    g = m.component.child("geometry")
    g.props["unit"] = unit
    fin = g.child("geom.finalize")
    idx = list(g.children).index(fin)
    imp = m.create("geom.import", g, index=idx, tag="imp")
    imp.props["file"] = step_path
    imp.props["heal"] = repair
    imp.props["unit"] = unit
    return build_geometry(m, display=False)


def test_step_import_single_domain(elmer):
    """A single-solid STEP imports as exactly one domain with boundaries, scaled mm -> m."""
    with tempfile.TemporaryDirectory() as d:
        step = os.path.join(d, "single.step")
        make_step_single(step)
        for repair in (True, False):
            geo = _imported_geometry(step, repair=repair)
            assert geo.count("domain") == 1, geo.count("domain")
            assert geo.count("boundary") > 0
            d_ = geo.numbers["domain"][0]
            vol = float(geo.measure["domain"][d_])
            assert vol > 0
            # cube 10 mm side = 0.01 m -> 1e-6 m^3
            assert abs(vol - 1e-6) < 1e-9, vol


def test_step_import_multi_domain(elmer):
    """A multi-body STEP (two separate solids) preserves each solid as its own domain."""
    with tempfile.TemporaryDirectory() as d:
        step = os.path.join(d, "multi.step")
        make_step_multi(step)
        geo = _imported_geometry(step, repair=True)
        assert geo.count("domain") == 2, geo.count("domain")
        assert geo.count("boundary") > 0


def test_step_import_composite(elmer):
    """A 10 mm cube containing two 1 mm cubes -> three solids as domains."""
    with tempfile.TemporaryDirectory() as d:
        step = os.path.join(d, "composite.step")
        make_step_composite(step)
        geo = _imported_geometry(step, repair=True)
        assert geo.count("domain") == 3, geo.count("domain")
        vols = sorted(float(geo.measure["domain"][dm]) for dm in geo.numbers["domain"])
        big = 1e-6
        small = 1e-9
        assert abs(vols[2] - big) < big * 1e-3, vols
        assert abs(vols[0] - small) < small * 1e-3, vols
        assert abs(vols[1] - small) < small * 1e-3, vols


def test_step_import_solve_electrostatics(elmer):
    """Two imported conductors (potential vs ground) solve in Elmer and produce a Potential field."""
    with tempfile.TemporaryDirectory() as d:
        step = os.path.join(d, "multi.step")
        make_step_multi(step)
        m = build_imported_model(step, unit="mm", repair=True)
        geo = build_geometry(m, display=False)
        assert geo.count("domain") == 2
        setup_physics(m, geo)
        st = m.studies()[0]
        sols = solve(m, st, tempfile.mkdtemp(prefix="es_step_"))
        assert len(sols) == 1
        s = sols[0]
        i = s.frame_index("last")
        V = s.evaluate("V", i)
        assert V is not None and len(V) > 0
        # one conductor at 1 V, the other at 0 V: the field must span both values
        assert float(V.max()) > 0.5
        assert float(V.min()) < 0.5


def test_step_import_missing_file():
    """Negative control: importing a missing file raises a clear GeometryError."""
    m = B.new_model("3D", None, None)
    g = m.component.child("geometry")
    fin = g.child("geom.finalize")
    idx = list(g.children).index(fin)
    imp = m.create("geom.import", g, index=idx, tag="imp")
    imp.props["file"] = "/nonexistent/does-not-exist.step"
    imp.props["heal"] = True
    try:
        build_geometry(m, display=False)
    except GeometryError as e:
        assert "file not found" in str(e).lower(), str(e)
    else:
        raise AssertionError("expected GeometryError for a missing STEP file")


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v", "--no-header"]))
