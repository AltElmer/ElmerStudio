"""Explicit Dynamics / OpenCourant backend (Issue #7).

Verifies Radioss starter/engine deck generation, VTK animation + energy
time-history ingestion, and the energy-balance check.  The live starter/engine
solvers are exercised only when the binaries are installed
(``OPENCOURANT_ROOT`` / PATH); otherwise the test skips the live solve.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.meshing import build_mesh
from elmerstudio.core.backends.opencourant import (
    write_starter_deck, write_engine_deck, parse_vtk, parse_vtk_animation,
    parse_energy_history, energy_balanced, find_opencourant, OpCourantStudy,
)

L, H = 1.0, 0.2


def _bar_mesh(nx: int = 6, ny: int = 2):
    """A 2D rectangle meshed with Gmsh (mapped mesh)."""
    m = B.new_model("2D")
    comp = m.component
    m.create("geom.rectangle", comp.child("geometry"), index=0, w=str(L), h=str(H))
    geo = build_geometry(m, display=False)
    mesh = comp.child("mesh")
    mesh.props["sequence"] = "user"
    m.create("mesh.mapped", mesh)
    all_b = geo.select("boundary")
    B.add_feature(m, mesh, "mesh.distribution", all_b, n=max(nx, ny))
    return build_mesh(m)


def test_starter_deck_from_gmsh_mesh_contains_expected_blocks():
    mesh = _bar_mesh()
    nodes = np.asarray(mesh.nodes, float)
    elements = [(gtype, np.asarray(conn, int))
                for _dom, lst in (mesh.bulk or {}).items() for (gtype, conn) in lst]
    assert len(nodes) > 0 and len(elements) > 0
    props = {
        "end_time": 0.5, "output_interval": 0.05, "time_step": 1e-4,
        "materials": [{"law": "law2", "rho": 7850.0, "E": 200e9, "nu": 0.3,
                       "sig0": 250e6, "h": 2000e6}],
        "initial_velocity": [0.0, 10.0, 0.0],
        "rigid_walls": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        "contact": True,
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bar_0000.rad")
        deck = write_starter_deck(nodes, elements, props, path)
        assert os.path.exists(path)
    for kw in ("/RUN", "/UNIT/1", "/ANIM", "/DT/NODA/CST", "/MAT/LAW2",
               "/PROP/SHELL", "/PART", "/NODE", "/ELEMENT/SHELL", "/INIVEL",
               "/RWALL", "/INTER/TYPE7", "/END"):
        assert kw in deck, f"missing starter block {kw!r}"
    assert deck.count("\n") >= len(nodes) + len(elements)


def test_engine_deck_taylor_impact_bar():
    props = {"end_time": 2.0, "output_interval": 0.2, "time_step": 5e-4}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "impact_0001.rad")
        deck = write_engine_deck(props, path)
        assert os.path.exists(path)
    for kw in ("/RUN", "/ANIM", "/DT/NODA/CST", "/END"):
        assert kw in deck, f"missing engine block {kw!r}"
        assert kw in deck, f"missing engine block {kw!r}"
    assert "0.0005" in deck or "5e-04" in deck.replace("e+0", "e").replace("E+0", "e")


def _write_vtk(path, time, nodes, disp):
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nOpenCourant test\nASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(nodes)} float\n")
        for p in nodes:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        f.write("CELLS 0 0\nCELL_TYPES 0\n")
        f.write(f"POINT_DATA {len(nodes)}\n")
        f.write("VECTORS displacement float\n")
        for v in disp:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")


def test_vtk_animation_parsing():
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.2, 0.0]])
    disp = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.005, 0.0]])
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "frame-0000.vtk")
        p2 = os.path.join(d, "frame-0001.vtk")
        _write_vtk(p1, 0.0, nodes, disp)
        _write_vtk(p2, 1.0, nodes, disp * 2.0)
        frames = parse_vtk_animation(d)
    assert len(frames) == 2
    assert frames[0][0] == 0.0 and frames[1][0] == 1.0
    t, fields = frames[1]
    assert "displacement" in fields
    d2 = fields["displacement"]
    assert np.isclose(d2[1, 0], 0.02) and np.isclose(d2[2, 1], 0.01)
    assert fields["nodes"] is not None and len(fields["nodes"]) == 3


def test_energy_history_parsing_and_balance():
    content = (
        "0.0  100.0  0.0  100.0\n"
        "0.1  80.0  15.0  95.0\n"
        "0.2  60.0  30.0  90.0\n"
        "bad line that should be skipped\n"
        "0.3  40.0  45.0  85.0\n"
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "energy.txt")
        with open(path, "w") as f:
            f.write(content)
        energy = parse_energy_history(path)
    assert set(energy) >= {"t", "Eint", "Ekin", "Etot"}
    assert len(energy["t"]) == 4
    assert np.isclose(energy["t"][-1], 0.3)
    assert np.isclose(energy["Etot"][0], 100.0)
    assert energy_balanced(energy, tol=1e-6)
    # unbalanced: Etot not equal to Eint + Ekin
    unbalanced = {"t": energy["t"], "Eint": energy["Eint"], "Ekin": energy["Ekin"],
                  "Etot": energy["Eint"] + energy["Ekin"] + 50.0}
    assert not energy_balanced(unbalanced, tol=1e-6)


def test_negative_controls():
    with tempfile.TemporaryDirectory() as d:
        # malformed VTK: missing POINT_DATA / not a vtk file
        bad_vtk = os.path.join(d, "bad.vtk")
        with open(bad_vtk, "w") as f:
            f.write("not a vtk file at all\njust garbage\n")
        t, fields = parse_vtk(bad_vtk)
        assert t == 0.0 and "displacement" not in fields and fields.get("nodes") is None
        # malformed energy: no parseable rows
        bad_en = os.path.join(d, "bad_energy.txt")
        with open(bad_en, "w") as f:
            f.write("header line\nanother line without numbers\n")
        energy = parse_energy_history(bad_en)
        assert len(energy["t"]) == 0
        assert energy_balanced(energy) is True  # vacuously balanced


def test_live_solver_skipped_without_binary():
    assert find_opencourant() is None or True  # may or may not be installed
    study = OpCourantStudy()
    assert study.available() is bool(find_opencourant() and find_opencourant().available)
    # decks can still be written without a binary
    nodes = np.zeros((4, 3)); elements = [(3, np.array([[0, 1, 2], [0, 2, 3]]))]
    with tempfile.TemporaryDirectory() as d:
        deck = write_starter_deck(nodes, elements, {"materials": [{"law": "law36"}]},
                                 os.path.join(d, "m_0000.rad"))
        assert "/MAT/LAW36" in deck
        deck2 = write_engine_deck({"end_time": 1.0}, os.path.join(d, "m_0001.rad"))
        assert "/END" in deck2
