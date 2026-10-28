"""Parallel (MPI) runs: why a study runs serially, finding mpiexec, the environment Elmer gets from a frozen build,
and solver settings that work on MPI processes (issue #1: parallel runs only on Windows).

An MPI run itself needs an MPI build of Elmer; set ELMERSTUDIO_TEST_MPI_HOME to its folder to run
test_tutorial_cell_on_two_mpi_processes."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elmerstudio.core import elmer as E  # noqa: E402
from elmerstudio.core.builder import study_steps  # noqa: E402
from elmerstudio.core.sif import build_sif  # noqa: E402
from test_electrochemistry import KINETICS, exact_current, strip  # noqa: E402


def _inst(tmp, mpi=True, mpiexec=True, grid=True):
    b = os.path.join(tmp, "bin")
    os.makedirs(b, exist_ok=True)
    return E.ElmerInstall(tmp, b, os.path.join(b, "ElmerSolver"), os.path.join(b, "ElmerGrid") if grid else None,
                          os.path.join(b, "ElmerSolver_mpi") if mpi else None,
                          os.path.join(tmp, "mpi", "mpiexec") if mpiexec else None)


def test_the_reason_for_a_serial_run_is_named(tmp_path):
    t = str(tmp_path)
    assert E.parallel_problem(_inst(t), 4) is None
    assert E.parallel_problem(_inst(t, mpi=False), 1) is None                 # one process needs nothing
    assert "no ElmerSolver_mpi" in E.parallel_problem(_inst(t, mpi=False), 4)
    assert "no mpiexec or mpirun" in E.parallel_problem(_inst(t, mpiexec=False), 4)
    assert "ElmerGrid" in E.parallel_problem(_inst(t, grid=False), 4)


def test_mpiexec_is_found_without_it_on_path(tmp_path, monkeypatch):
    """A macOS app started from the Finder has PATH=/usr/bin:/bin:/usr/sbin:/sbin: Homebrew's mpiexec is not on it."""
    d = tmp_path / "openmpi" / "bin"
    d.mkdir(parents=True)
    exe = d / ("mpiexec" + E.EXE)
    exe.write_text("")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("MSMPI_BIN", raising=False)
    monkeypatch.setitem(E.MPI_DIRS, E.platform.system(), [str(d)])
    assert E.find_mpiexec() == str(exe)
    beside = tmp_path / "elmer" / "bin"                                     # an Elmer bundled with its own MPI
    beside.mkdir(parents=True)
    (beside / ("mpiexec" + E.EXE)).write_text("")
    assert E.find_mpiexec(str(beside)) == str(beside / ("mpiexec" + E.EXE))
    monkeypatch.setitem(E.MPI_DIRS, E.platform.system(), [])
    assert E.find_mpiexec() is None                                         # control: nothing anywhere


def test_a_frozen_linux_build_does_not_pass_its_library_path_to_elmer(tmp_path, monkeypatch):
    monkeypatch.setattr(E.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/ElmerStudio/_internal")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/home/me/lib")
    env = _inst(str(tmp_path)).env()
    assert "_internal" not in env["LD_LIBRARY_PATH"] and "/home/me/lib" in env["LD_LIBRARY_PATH"]
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG")                 # the user had none: the bundle's is dropped
    assert "_internal" not in _inst(str(tmp_path)).env()["LD_LIBRARY_PATH"]
    monkeypatch.setattr(sys, "frozen", False)                  # control: from source the variable is kept
    assert "_internal" in _inst(str(tmp_path)).env()["LD_LIBRARY_PATH"]


def test_parallel_runs_get_solver_settings_that_work_on_mpi_processes():
    """UMFPACK is serial only (Elmer switches to MUMPS, which many MPI builds lack, and stops), and the Newton line
    search fails in parallel runs: on more than one process the SIF has neither."""
    m, geo, _an = strip("bv")
    study = m.studies()[0]
    step = study_steps(study)[0]
    serial = build_sif(m, study, step, geo, 300)
    assert "UMFPack" in serial.text and "Linesearch = Logical True" in serial.text     # control: serial unchanged
    par = build_sif(m, study, step, geo, 300, nproc=4)
    assert "UMFPack" not in par.text and "Linesearch" not in par.text
    assert par.text.count('Linear System Iterative Method = String "BiCGStab"') == 2
    warn = " ".join(par.meta["warnings"])
    assert "4 MPI processes" in warn and "line search off" in warn


def test_a_serial_elmer_says_why_it_runs_serially(elmer):
    from elmerstudio.core.study_runner import StudyRunner
    if elmer.solver_mpi and elmer.mpiexec:
        pytest.skip("this Elmer can run MPI processes")
    m, _geo, _an = strip("linear")
    m.studies()[0].props["np"] = 2
    seen = []
    StudyRunner(m, m.studies()[0], tempfile.mkdtemp(prefix="es_par_"), elmer,
                on_event=lambda k, p: seen.append((k, p))).run()
    warn = [p for k, p in seen if k == "warning" and "running serially" in str(p)]
    assert len(warn) == 1 and E.parallel_problem(elmer, 2) in warn[0], warn


@pytest.mark.skipif(not os.environ.get("ELMERSTUDIO_TEST_MPI_HOME"), reason="set ELMERSTUDIO_TEST_MPI_HOME")
def test_tutorial_cell_on_two_mpi_processes():
    from elmerstudio.core.nodes import bind_results
    from elmerstudio.core.results import Solution
    from elmerstudio.core.study_runner import StudyRunner
    inst = E.find_elmer(os.environ["ELMERSTUDIO_TEST_MPI_HOME"])
    assert E.parallel_problem(inst, 2) is None, E.parallel_problem(inst, 2)
    V, L, W, sigma = 0.1, 2e-4, 1e-4, 20.0
    m, geo, an = strip("bv", V=V, L=L, W=W, sigma=sigma)
    m.studies()[0].props["np"] = 2
    folder = StudyRunner(m, m.studies()[0], tempfile.mkdtemp(prefix="es_mpi_"), inst).run()[-1]
    assert os.path.isdir(os.path.join(folder, "mesh", "partitioning.2"))
    assert any(f.endswith(".pvtu") for f in os.listdir(os.path.join(folder, "results")))
    sol = bind_results(m, Solution(folder), geo)
    current = float(sol.evaluate(f"cd.{an.tag}.Itot", sol.frame_index("last"))[0])
    assert abs(current / (exact_current(KINETICS["bv"], V, L, sigma) * W) - 1) < 1e-8
