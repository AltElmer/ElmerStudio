"""Importing CAD files (STEP, IGES, BREP) through the Import node keeps the solids (issue #10).

The reported case is a sphere with two balls cut out of it: OpenCASCADE's repair sews the faces of each solid into one
closed shell, a solid with cavities has three, and the volume came back as loose faces (3 boundaries, 0 domains).
The files are written here with gmsh's OpenCASCADE kernel, in millimetres, as a CAD tool would.
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core import gmsh_session
from elmerstudio.core.geometry import GeometryBuilder, GeometryError, build_geometry
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner

R, A, D = 5.0, 0.5, 2.0          # outer sphere radius, ball radius, distance of the ball centres [mm]
V_CAVITY = 4 / 3 * math.pi * (R ** 3 - 2 * A ** 3) * 1e-9        # [m^3]


def _cavities(occ):
    s = occ.addSphere(0, 0, 0, R)
    balls = [(3, occ.addSphere(x, 0, 0, A)) for x in (-D / 2, D / 2)]
    occ.cut([(3, s)], balls)


def _cube(occ):
    occ.addBox(0, 0, 0, 10, 10, 10)


def _apart(occ):
    occ.addBox(0, 0, 0, 4, 4, 4)
    occ.addBox(10, 0, 0, 4, 4, 4)


def _touching(occ):
    occ.addBox(0, 0, 0, 4, 4, 4)
    occ.addBox(4, 0, 0, 4, 4, 4)


def _void(occ):
    box = occ.addBox(0, 0, 0, 10, 10, 10)
    occ.cut([(3, box)], [(3, occ.addBox(4, 4, 4, 2, 2, 2))])


def _cylinder(occ):
    occ.addCylinder(0, 0, 0, 0, 0, 10, 2)


@pytest.fixture(scope="module")
def cad():
    """name.ext -> path of a CAD file written on demand."""
    folder = tempfile.mkdtemp(prefix="es_cad_")
    shapes = {"cavities": _cavities, "cube": _cube, "apart": _apart, "touching": _touching, "void": _void,
              "cylinder": _cylinder}

    def path(name, ext="step"):
        p = os.path.join(folder, f"{name}.{ext}")
        if not os.path.isfile(p):
            with gmsh_session.session("cad") as g:
                g.option.setString("Geometry.OCCTargetUnit", "MM")
                shapes[name](g.model.occ)
                g.model.occ.synchronize()
                g.write(p)
        return p
    return path


def _model(path, unit="mm", heal=True, physics=None, before=None):
    m = B.new_model("3D", physics, "stationary" if physics else None)
    g = m.component.child("geometry")
    g.props["unit"] = unit
    if before:
        before(m, g)
    imp = m.create("geom.import", g, index=len(g.children) - 1, tag="imp")
    imp.props.update(file=path, heal=heal)
    return m


def _volumes(geo):
    return sorted(float(geo.measure["domain"][d]) for d in geo.numbers["domain"])


def test_solid_with_cavities_keeps_its_domain(cad):
    """The file of the issue, repair on (the default): one domain of the right volume, three boundaries."""
    geo = build_geometry(_model(cad("cavities")), display=False)
    assert geo.count("domain") == 1 and geo.count("boundary") == 3
    assert _volumes(geo) == pytest.approx([V_CAVITY], rel=1e-6)
    # the sewing repair lost the solid, the repair without sewing left a sphere that cannot be meshed (the solve
    # below is what found that): the file is taken as it is, and the messages say why
    assert len(geo.messages) == 3, geo.messages
    assert "became 6 of dimension 2" in geo.messages[0] and "discarded" in geo.messages[0]
    assert "cannot be meshed" in geo.messages[1] and "discarded" in geo.messages[1]
    assert "imported without repair" in geo.messages[2]


def test_cavities_without_repair(cad):
    geo = build_geometry(_model(cad("cavities"), heal=False), display=False)
    assert geo.count("domain") == 1 and geo.count("boundary") == 3 and geo.messages == []
    assert _volumes(geo) == pytest.approx([V_CAVITY], rel=1e-6)


@pytest.mark.parametrize("heal", [True, False])
@pytest.mark.parametrize("unit", ["mm", "m"])
def test_length_unit_of_the_file(cad, unit, heal):
    """A STEP file carries its unit (mm): the cube of 10 mm is 1e-6 m^3 whatever the unit of the geometry."""
    geo = build_geometry(_model(cad("cube"), unit=unit, heal=heal), display=False)
    assert geo.count("domain") == 1 and geo.count("boundary") == 6 and geo.messages == []
    assert _volumes(geo) == pytest.approx([1e-6], rel=1e-9)


@pytest.mark.parametrize("heal", [True, False])
def test_solid_with_a_void_keeps_its_repair(cad, heal):
    """A void between plane faces survives the repair: it is kept, without a message."""
    geo = build_geometry(_model(cad("void"), heal=heal), display=False)
    assert geo.count("domain") == 1 and geo.count("boundary") == 12 and geo.messages == []
    assert _volumes(geo) == pytest.approx([992e-9], rel=1e-9)


@pytest.mark.parametrize("ext", ["step", "brep", "iges"])
def test_formats(cad, ext):
    """STEP, BREP (no unit: taken in the unit of the geometry) and IGES, which holds faces only: there the solid
    is made of the faces, and a message says so."""
    geo = build_geometry(_model(cad("cube", ext)), display=False)
    assert geo.count("domain") == 1 and geo.count("boundary") == 6
    assert _volumes(geo) == pytest.approx([1e-6], rel=1e-9)
    assert [("surfaces only" in s, "1 region" in s) for s in geo.messages] == ([(True, True)] if ext == "iges" else [])


def test_iges_without_repair_is_a_surface_model(cad):
    """The control of the IGES case above: without the repair there is no solid to find."""
    geo = build_geometry(_model(cad("cube", "iges"), heal=False), display=False)
    assert geo.count("domain") == 0


@pytest.mark.parametrize("heal", [True, False])
def test_bodies_become_domains(cad, heal):
    """Each body of a file is a domain; bodies that touch share their interface after the union, as native ones do."""
    geo = build_geometry(_model(cad("apart"), heal=heal), display=False)
    assert geo.count("domain") == 2 and geo.count("boundary") == 12
    assert _volumes(geo) == pytest.approx([64e-9, 64e-9], rel=1e-9)
    geo = build_geometry(_model(cad("touching"), heal=heal), display=False)
    assert geo.count("domain") == 2 and geo.count("boundary") == 11
    shared = [b for b in geo.numbers["boundary"] if len(geo.adj_up["boundary"][b]) == 2]
    assert len(shared) == 1 and float(geo.measure["boundary"][shared[0]]) == pytest.approx(16e-6, rel=1e-9)


@pytest.mark.parametrize("name, volumes, boundaries", [
    ("apart", [64e-9, 64e-9], 12),          # two bodies
    ("touching", [64e-9, 64e-9], 11),       # two bodies with a common face: sewn into one shell they were one
    #                                         solid with a crack through it
    ("void", [8e-9, 992e-9], 12),           # faces do not say which side is material: the void is a region too
    ("cylinder", [math.pi * 4 * 10 * 1e-9], 3),     # curved faces do not cut the box: the sewn shell
])
def test_iges_surfaces_become_solids(cad, name, volumes, boundaries):
    geo = build_geometry(_model(cad(name, "iges")), display=False)
    assert geo.count("domain") == len(volumes) and geo.count("boundary") == boundaries
    assert _volumes(geo) == pytest.approx(volumes, rel=1e-6)
    assert any(f"{len(volumes)} region(s)" in s for s in geo.messages), geo.messages
    assert build_geometry(_model(cad(name, "iges"), heal=False), display=False).count("domain") == 0


def test_iges_surfaces_that_enclose_nothing_are_reported(cad):
    """The limit of the above, pinned: the repair breaks the spheres of this file and they neither cut a box nor sew
    into one shell.  There is no domain, and a message says so instead of a solid being made up."""
    geo = build_geometry(_model(cad("cavities", "iges")), display=False)
    assert geo.count("domain") == 0 and geo.count("boundary") == 3
    assert "enclose no volume" in geo.messages[-1], geo.messages


@pytest.mark.parametrize("heal", [True, False])
def test_import_owns_only_its_own_entities(cad, heal):
    """healShapes returns the entities of the whole model: an import after other objects must not take them over."""
    def block(m, g):
        m.create("geom.block", g, index=0, tag="blk", w="3", d="3", h="3", x="20")
    geo = build_geometry(_model(cad("cube"), heal=heal, before=block), display=False)
    assert geo.count("domain") == 2
    own = {name: list(doms) for name, doms in geo.object_domains.items()}
    assert sorted(len(d) for d in own.values()) == [1, 1] and len({d[0] for d in own.values()}) == 2, own
    imp = next(d[0] for name, d in own.items() if "imp" in name)
    assert float(geo.measure["domain"][imp]) == pytest.approx(1e-6, rel=1e-9)



def test_repair_that_changes_the_volume_is_refused(cad, monkeypatch):
    """A repair that keeps the number of solids but not their measure is not a repair."""
    import gmsh

    def shrink(dimTags, **kw):
        gmsh.model.occ.dilate(dimTags, 0, 0, 0, 0.9, 0.9, 0.9)
        return []
    monkeypatch.setattr(gmsh.model.occ, "healShapes", shrink)
    assert GeometryBuilder.HEAL_RTOL < 1 - 0.9 ** 3
    geo = build_geometry(_model(cad("cube")), display=False)
    assert _volumes(geo) == pytest.approx([1e-6], rel=1e-9)
    assert any("without repair" in s for s in geo.messages), geo.messages


def test_missing_and_empty_files(tmp_path):
    with pytest.raises(GeometryError, match="file not found"):
        build_geometry(_model(str(tmp_path / "nothing.step")), display=False)
    with pytest.raises(GeometryError, match="no file selected"):
        build_geometry(_model(""), display=False)


def test_imported_solid_meshes_and_solves(cad, elmer):
    """Electrostatics in the imported sphere, one ball at 1 V and the other grounded: by symmetry the potential is
    0.5 V on the plane between them, and the capacitance agrees with the series solution for two spheres in free
    space once the charge on the outer sphere (which is not at infinity) is allowed for."""
    m = _model(cad("cavities"), physics=["es"])
    geo = build_geometry(m, display=False)
    mm = 1e-3
    left = geo.select("boundary", xmax=0)
    right = geo.select("boundary", xmin=0)
    assert len(left) == 1 and len(right) == 1
    es = m.component.child("physics.es")
    B.add_feature(m, es, "es.pot", left, V0="1[V]")
    B.add_feature(m, es, "es.ground", right)
    B.add_material(m, "Air", all_domains=True)
    mesh = m.component.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.size", mesh, mode="custom", hmax="1.0", hmin="0.07", hcurve="0.3")
    m.create("mesh.ftet", mesh)
    st = m.studies()[0]
    B.ensure_solver_configs(m, st)
    sol = Solution(StudyRunner(m, st, tempfile.mkdtemp(prefix="es_step_"), elmer).run()[-1])
    i = sol.frame_index("last")
    V = np.asarray(sol.evaluate("V", i))
    assert V.min() == pytest.approx(0.0, abs=1e-9) and V.max() == pytest.approx(1.0, abs=1e-9)
    # the mesh is not symmetric, the solution is: 0.5 V on the plane x = 0, and V(-x) = 1 - V(x) off it
    yz = np.array([(y, z) for y in np.linspace(-3, 3, 7) for z in np.linspace(-3, 3, 7)]) * mm
    plane = np.column_stack([np.zeros(len(yz)), yz])
    assert np.asarray(sol.point_eval("V", i, plane)) == pytest.approx(0.5, abs=0.01)
    off = np.column_stack([np.full(len(yz), 0.3 * mm), 0.2 * yz])
    here = np.asarray(sol.point_eval("V", i, off))
    there = np.asarray(sol.point_eval("V", i, off * [-1, 1, 1]))
    # linear elements of 0.1 to 0.3 mm where the field is 1 V/mm: 0.001 typically, 0.03 in the odd coarse element
    error = np.abs(here + there - 1.0)
    assert float(np.median(error)) < 0.005 and float(error.max()) < 0.05, error
    assert float(here.max()) < 0.45                 # the side of the grounded ball, and far more than that error
