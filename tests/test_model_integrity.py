"""Integrity of what Studio computes, saves and shows.

1. A solve is accepted only when Elmer converged: a linear solver that runs out of iterations fails the study; Newton
   or coupling iterations that stop at their maximum stop it too, unless the study keeps such solutions, marked.
2. Parameters with units are recognized whatever the row order; circular definitions are named.
3. The model file stores every solution file (moving meshes included) with a manifest that detects damage, refuses
   unsafe archive paths, and packages imported geometry files.
4. Interfaces that would overwrite each other's Elmer variable or keywords are refused; species transport may be
   added more than once, each species with its own variable.
5. Results evaluate with the parameter values the solution was computed with.
"""
import itertools
import json
import os
import shutil
import tempfile
import zipfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core import project, units
from elmerstudio.core.elmer import convergence_problems
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.results import Solution
from elmerstudio.core.sif import SifError, build_sif
from elmerstudio.core.study_runner import StudyNotConverged, StudyRunner


def _sif(m, geo):
    st = m.studies()[0]
    return build_sif(m, st, B.study_steps(st)[0], geo, 100)


def _meta(folder):
    with open(os.path.join(folder, "meta.json")) as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------- 1. acceptance
def _conduction(w=2.0, h=0.25, t_right=3.0):
    m = B.new_model("2D", ["ht"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(w), h=str(h))
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    ht = m.component.child("physics.ht")
    B.add_feature(m, ht, "ht.temperature", geo.select("boundary", xmax=0), T0="0")
    B.add_feature(m, ht, "ht.temperature", geo.select("boundary", xmin=w), T0=str(t_right))
    return m


def test_a_linear_solver_out_of_iterations_fails_the_study(elmer):
    m = _conduction()
    st = m.studies()[0]
    cfg = B.ensure_solver_configs(m, st).children[0]
    cfg.props.update(lin="iterative", iter="CG", prec="none", maxit=1, tol="1e-14")
    work = tempfile.mkdtemp(prefix="es_linabort_")
    with pytest.raises(RuntimeError):
        StudyRunner(m, st, work, elmer).run()
    meta = _meta(os.path.join(work, st.tag, "sol_0_0"))
    assert meta["status"] == "failed" and meta["diagnostics"], meta.get("diagnostics")
    cfg.props.update(lin="direct")                                      # the control: same model, converged
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_linok_"), elmer).run()
    sol = Solution(folders[0])
    assert _meta(folders[0])["status"] == "converged"
    x = sol.topo.nodes[:, 0] * sol.scale
    assert np.max(np.abs(sol.evaluate("T", sol.frame_index("last")) - 1.5 * x)) < 1e-8


def _nonlinear_pde(nl_maxit, nl_tol, keep=False):
    """-div((1 + u^2) grad u) = 20, u = 0 on the boundary: nonlinear in u."""
    m = B.new_model("2D", ["pde"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1.5", h="1")
    geo = build_geometry(m, display=False)
    pde = m.component.child("physics.pde")
    pde.child("pde.cfeq").props.update(c="1+u^2", f="20", da="0")
    B.add_feature(m, pde, "pde.dir", geo.numbers["boundary"], r="0")
    st = m.studies()[0]
    st.props["nonconv"] = "keep" if keep else "stop"
    B.ensure_solver_configs(m, st).children[0].props.update(nl_maxit=nl_maxit, nl_tol=nl_tol)
    return m, st


def test_newton_iterations_at_their_maximum_stop_or_are_kept_marked(elmer):
    m, st = _nonlinear_pde(2, "1e-12")
    with pytest.raises(StudyNotConverged):
        StudyRunner(m, st, tempfile.mkdtemp(prefix="es_nlstop_"), elmer).run()
    m, st = _nonlinear_pde(2, "1e-12", keep=True)
    events = []
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_nlkeep_"), elmer,
                          on_event=lambda k, p: events.append((k, p))).run()
    meta = _meta(folders[0])
    assert meta["status"] == "not converged" and "coefficient form pde" in meta["diagnostics"][0]
    assert any(k == "warning" and "NOT CONVERGED" in str(p) for k, p in events)
    m, st = _nonlinear_pde(60, "1e-8")                                  # the control: enough iterations
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_nlok_"), elmer).run()
    assert _meta(folders[0])["status"] == "converged"


def test_convergence_verdict_from_the_iteration_log():
    spec = {"ss_maxit": 4, "solvers": [{"equation": "a", "post": False, "nl_maxit": 3, "nl_tol": 1e-8,
                                        "ss_tol": 1e-6},
                                       {"equation": "b", "post": False, "nl_maxit": 1, "nl_tol": 1e-8,
                                        "ss_tol": 1e-6}]}

    def log(blocks):                       # [[(a Newton relcs), b relc, (a SS relc, b SS relc)], ...] per coupling it.
        out, n = [], 0
        for j, (newton, b_rel, (sa, sb)) in enumerate(blocks, 1):
            for i, r in enumerate(newton, 1):
                out.append((n := n + 1, "A", "NS", r, i))
            out.append((n := n + 1, "A", "SS", sa, j))
            out.append((n := n + 1, "B", "NS", b_rel, 1))
            out.append((n := n + 1, "B", "SS", sb, j))
        return out
    # Newton at its maximum in early coupling iterations, converged in the last: accepted
    ok = log([((1e-1, 1e-3, 1e-5), 1e-2, (1, 1)), ((1e-4, 1e-9), 1e-7, (1e-3, 1e-4)),
              ((1e-10,), 1e-12, (1e-8, 1e-9))])
    assert convergence_problems(ok, spec) == []
    # Newton at its maximum in the last coupling iteration
    bad_nl = log([((1e-1, 1e-3, 1e-5), 1e-2, (1, 1)), ((1e-4, 1e-6, 1e-7), 1e-9, (1e-8, 1e-9))])
    msgs = convergence_problems(bad_nl, spec)
    assert len(msgs) == 1 and "'a'" in msgs[0] and "1 of 1" in msgs[0]
    # coupling iterations at their maximum
    bad_ss = log([((1e-9,), 1e-9, (1, 1))] * 3 + [((1e-9,), 1e-9, (1e-3, 1e-9))])
    msgs = convergence_problems(bad_ss, spec)
    assert len(msgs) == 1 and "coupling" in msgs[0]
    assert convergence_problems(bad_ss, None) == []                    # no spec (an old solution): no verdict


# ---------------------------------------------------------------------------------------- 2. parameter units
def test_unit_flags_do_not_depend_on_row_order():
    rows = [("span", "2*half"), ("half", "7[mm]"), ("gap", "span+1[mm]"), ("count", "4"), ("ratio", "count/2")]
    expect = {"span": True, "half": True, "gap": True, "count": False, "ratio": False}
    for perm in itertools.permutations(rows):
        assert units.parameter_unit_flags(list(perm)) == expect, perm
        assert units.expr_has_units("span/3", list(perm)) and not units.expr_has_units("ratio+1", list(perm))


def test_geometry_size_does_not_depend_on_parameter_order():
    widths = []
    for rows in ([("wd", "3*base"), ("base", "4[mm]")], [("base", "4[mm]"), ("wd", "3*base")]):
        m = B.new_model("2D", ["ht"], "stationary")
        m.by_tag("param").props["table"] = [{"name": n, "expr": e} for n, e in rows]
        m.create("geom.rectangle", m.component.child("geometry"), index=0, w="wd", h="base")
        geo = build_geometry(m, display=False)
        widths.append((geo.bbox[3] - geo.bbox[0]) * geo.scale)
    # 12 mm both ways (the row-order bug made one of them 12 m); gmsh pads bounding boxes by ~1e-7
    assert widths[0] == pytest.approx(0.012, rel=1e-4) and widths[1] == pytest.approx(widths[0], rel=1e-9)


def test_circular_parameters_are_named():
    _vals, errs = units.evaluate_parameters([("p", "q+1"), ("q", "2*p"), ("r", "zz*2")])
    assert "p -> q -> p" in errs["p"] or "q -> p -> q" in errs["p"], errs
    assert "undefined" in errs["r"] and "zz" in errs["r"], errs


# ------------------------------------------------------------------------------------------------ 3. model file
def _fake_solution(work, moving=True):
    """A solution folder as the runner writes one (a deforming run: frame coordinates and remeshed topologies)."""
    sol = os.path.join(work, "std1", "sol_0_0")
    os.makedirs(os.path.join(sol, "results"))
    os.makedirs(os.path.join(sol, "mesh"))                             # Elmer's case mesh: not stored
    with open(os.path.join(sol, "meta.json"), "w") as f:
        json.dump({"kind": "study.time", "status": "converged", "moving_mesh": moving}, f)
    np.savez_compressed(os.path.join(sol, "mesh.npz"), nodes=np.arange(12.0).reshape(4, 3))
    np.savez_compressed(os.path.join(sol, "frames.npz"), f0=np.zeros((4, 3)), f1=np.ones((4, 3)))
    np.savez_compressed(os.path.join(sol, "mesh_1.npz"), nodes=np.full((5, 3), 2.5))
    with open(os.path.join(sol, "results", "case_t0001.vtu"), "wb") as f:
        f.write(b"<VTKFile/>" * 40)
    with open(os.path.join(sol, "mesh", "mesh.nodes"), "w") as f:
        f.write("1 -1 0 0 0\n")
    return sol


def test_model_file_keeps_moving_mesh_files_and_detects_damage():
    m = B.new_model("2D", ["ht"], "stationary")
    work = tempfile.mkdtemp(prefix="es_proj_")
    sol = _fake_solution(work)
    path = os.path.join(tempfile.mkdtemp(prefix="es_projfile_"), "moving.mph")
    project.save(m, path, work)
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        manifest = json.loads(z.read("manifest.json"))
    assert {"work/std1/sol_0_0/frames.npz", "work/std1/sol_0_0/mesh_1.npz"} <= names
    assert not any("/mesh/" in n for n in names)                       # the Elmer case mesh is not stored
    assert set(manifest["files"]) == {n for n in names if n.startswith("work/")}
    shutil.rmtree(work)                                                 # the session is gone: only the file remains
    m2, work2 = project.load(path)
    assert m2.load_problems == [] and m2.solutions_saved == "included"
    for name in ("frames.npz", "mesh_1.npz", "meta.json"):
        a = os.path.join(work2, "std1", "sol_0_0", name)
        assert os.path.isfile(a), name
    with np.load(os.path.join(work2, "std1", "sol_0_0", "mesh_1.npz")) as z:
        assert float(z["nodes"][0, 0]) == 2.5
    # a damaged archive: one stored file altered, another removed
    bad = path.replace(".mph", "_bad.mph")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(bad, "w") as zout:
        for it in zin.infolist():
            if it.filename.endswith("frames.npz"):
                continue
            data = zin.read(it.filename)
            if it.filename.endswith(".vtu"):
                data = data[:-5] + b"XXXXX"
            zout.writestr(it, data)
    m3, _ = project.load(bad)
    assert any(p.startswith("missing") and "frames.npz" in p for p in m3.load_problems), m3.load_problems
    assert any(p.startswith("damaged") and ".vtu" in p for p in m3.load_problems), m3.load_problems
    _ = sol


def test_saved_without_solutions_is_not_damage():
    m = B.new_model("2D", ["ht"], "stationary")
    work = tempfile.mkdtemp(prefix="es_proj_")
    _fake_solution(work)
    path = os.path.join(tempfile.mkdtemp(prefix="es_projfile_"), "lean.mph")
    project.save(m, path, work, include_solutions=False)
    m2, work2 = project.load(path)
    assert m2.load_problems == [] and m2.solutions_saved == "excluded"
    assert not os.path.exists(os.path.join(work2, "std1"))


@pytest.mark.parametrize("entry", ["work/../escaped.txt", "work/../../escaped.txt", "work/C:/escaped.txt"])
def test_unsafe_archive_paths_are_refused(entry):
    m = B.new_model("2D", ["ht"], "stationary")
    path = os.path.join(tempfile.mkdtemp(prefix="es_projfile_"), "unsafe.mph")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("model.json", m.to_json())
        z.writestr(entry, "x")
    with pytest.raises(project.ProjectError):
        project.load(path)


def test_imported_geometry_files_travel_with_the_model():
    src_dir = tempfile.mkdtemp(prefix="es_asset_")
    src = os.path.join(src_dir, "part.step")
    with open(src, "w") as f:
        f.write("ISO-10303-21; dummy part\n")
    m = B.new_model("3D", ["ht"], "stationary")
    imp = m.create("geom.import", m.component.child("geometry"), index=0)
    imp.props["file"] = src
    path = os.path.join(tempfile.mkdtemp(prefix="es_projfile_"), "imported.mph")
    project.save(m, path, tempfile.mkdtemp(prefix="es_proj_"))
    m2, _ = project.load(path)
    assert m2.by_tag(imp.tag).props["file"] == src                      # the original is still there: kept
    shutil.rmtree(src_dir)                                              # the file opened on another computer
    m3, work3 = project.load(path)
    relinked = m3.by_tag(imp.tag).props["file"]
    assert relinked.startswith(os.path.join(work3, "_assets")) and m3.load_problems == []
    with open(relinked) as f:
        assert f.read() == "ISO-10303-21; dummy part\n"


# -------------------------------------------------------------------------------------- 4. interfaces collide
def _strip(m):
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="0.5", h="0.05")
    return build_geometry(m, display=False)


def test_a_second_interface_with_a_fixed_variable_is_refused():
    m = B.new_model("2D", ["ht", "tds", "pde"], "stationary")
    with pytest.raises(ValueError, match="already"):
        B.add_physics(m, "ht")
    tds2, pde2 = B.add_physics(m, "tds"), B.add_physics(m, "pde")
    assert tds2.get("species") == "c2" and pde2.get("depvar") == "u2"
    assert B.add_physics(m, "tds").get("species") == "c3"
    # a file from before the guard, with two Heat Transfer interfaces: refused when the SIF is built
    m = B.new_model("2D", ["ht"], "stationary")
    geo = _strip(m)
    B.add_material(m, "Copper", all_domains=True)
    m.create("physics.ht", m.component, index=len(m.component.children) - 2, tag="ht2", label="Heat Transfer 2")
    with pytest.raises(SifError, match="both solve the Elmer variable 'Temperature'"):
        _sif(m, geo)


def test_two_species_have_their_own_variables_and_keywords():
    m = B.new_model("2D", ["tds"], "stationary")
    geo = _strip(m)
    tds2 = B.add_physics(m, "tds")
    m.component.child("physics.tds").child("tds.tp").props["D"] = "1e-9[m^2/s]"
    tds2.child("tds.tp").props["D"] = "3e-9[m^2/s]"
    res = _sif(m, geo)
    text = res.text
    assert "Variable = Concentration_c2" in text and "Variable = Concentration\n" in text
    assert "Concentration Diffusivity = Real 1e-09" in text and "Concentration_c2 Diffusivity = Real 3e-09" in text
    assert "Diffusion Coefficient" not in text                          # ModelPDE's keyword: not written
    assert 'Flux Variable = String "Concentration_c2"' in text
    assert [e[0] for e in res.meta["catalog"]] == ["c2", f"{tds2.tag}.tflux", f"{tds2.tag}.tfluxMag",
                                                   f"{tds2.tag}.ntflux"]
    tds2.props["stab"] = False                                          # one Galerkin species next to one SUPG: fine
    _sif(m, geo)
    m.component.child("physics.tds").props["stab"] = False              # two ModelPDE species in one domain: not
    with pytest.raises(SifError, match="Transport of Diluted Species 2 and Transport of Diluted Species"):
        _sif(m, geo)
    tds2.child("tds.tp").props["D"] = "1e-9[m^2/s]"                    # even with equal coefficients
    with pytest.raises(SifError, match="ModelPDE"):
        _sif(m, geo)


def test_interfaces_that_set_one_keyword_differently_are_refused():
    m = B.new_model("2D", ["tds"], "stationary")
    geo = _strip(m)
    tds2 = B.add_physics(m, "tds")
    right = geo.select("boundary", xmin=0.5)
    B.add_feature(m, m.component.child("physics.tds"), "tds.flux", right, type="mt", kc="1e-5", cb="1")
    B.add_feature(m, tds2, "tds.flux", right, type="mt", kc="2e-5", cb="1")
    # AdvectionDiffusion reads 'Mass Transfer Coefficient' without the variable's prefix
    with pytest.raises(SifError, match="Mass Transfer Coefficient"):
        _sif(m, geo)


def test_two_species_solve_and_keep_their_inputs(elmer):
    L, h, D1, D2 = 0.5, 0.05, 1e-9, 2e-9
    m = B.new_model("2D", ["tds"], "stationary")
    m.by_tag("param").props["table"] = [{"name": "Dtwo", "expr": f"{D2}[m^2/s]"}]
    geo = _strip(m)
    left, right = geo.select("boundary", xmax=0), geo.select("boundary", xmin=L)
    t1 = m.component.child("physics.tds")
    t2 = B.add_physics(m, "tds")
    t1.child("tds.tp").props["D"] = f"{D1}[m^2/s]"
    t2.child("tds.tp").props["D"] = "Dtwo"
    B.add_feature(m, t1, "tds.conc", left, c0="0")
    B.add_feature(m, t1, "tds.conc", right, c0="1[mol/m^3]")
    B.add_feature(m, t2, "tds.conc", left, c0="3[mol/m^3]")
    B.add_feature(m, t2, "tds.conc", right, c0="0")
    st = m.studies()[0]
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_species_"), elmer).run()
    sol = Solution(folders[0])
    bind_results(m, sol, geo)
    i = sol.frame_index("last")
    x = sol.topo.nodes[:, 0] * sol.scale
    assert np.max(np.abs(sol.evaluate("c", i) - x / L)) < 1e-8
    assert np.max(np.abs(sol.evaluate("c2", i) - 3 * (1 - x / L))) < 1e-8
    flux2 = sol.integrate(f"{t2.tag}.ntflux", i, "boundary", right)
    assert abs(abs(flux2) - 3 * D2 * h / L) < 1e-3 * 3 * D2 * h / L, flux2
    # 5. provenance: editing the parameter afterwards does not change what the solution was computed with
    m.by_tag("param").props["table"] = [{"name": "Dtwo", "expr": "9e-9[m^2/s]"}]
    bind_results(m, sol, geo)
    assert sol.params["Dtwo"] == pytest.approx(D2) and sol.params_changed == ["Dtwo"]
    with open(os.path.join(folders[0], "model.json")) as f:
        assert '"Dtwo"' in f.read()
