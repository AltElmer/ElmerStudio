import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*extract_surface.*")
# recovery checkpoints of the windows the tests open never reach the user's own recovery folder
if "ELMERSTUDIO_RECOVERY_DIR" not in os.environ:
    import tempfile as _tf
    os.environ["ELMERSTUDIO_RECOVERY_DIR"] = _tf.mkdtemp(prefix="es_recovery_")

# a script (drives the GUI event loop); run it directly:  python tests/test_gui_smoke.py
collect_ignore = ["test_gui_smoke.py"]


@pytest.fixture(scope="session", autouse=True)
def _solver_scratch(request):
    """Every case folder the tests create (tempfile.mkdtemp) goes into one directory, removed at the end when every
    test passed (a full run writes about 1 GB); kept after a failure so the case can be inspected (CI uploads it)."""
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="elmerstudio_tests_")
    old = tempfile.tempdir
    tempfile.tempdir = d
    yield d
    tempfile.tempdir = old
    if request.session.testsfailed == 0:
        shutil.rmtree(d, ignore_errors=True)


def _find_elmer():
    from elmerstudio.core.elmer import find_elmer
    return find_elmer()


def _find_openfoam():
    from elmerstudio.core.backends.openfoam.install import find_openfoam
    return find_openfoam()


def _find_nastran():
    from elmerstudio.core.backends.nastran.install import find_nastran
    return find_nastran()


# solver -> (how to find its installation, what to say when it is not there)
SOLVERS = {
    "elmer": (_find_elmer, "Elmer (ElmerSolver) not installed"),
    "openfoam": (_find_openfoam, "OpenFOAM not installed (set OPENFOAM_ROOT)"),
    "nastran": (_find_nastran, "NASTRAN-95 not installed (set NASTRAN_HOME or place nastran on PATH)"),
}


def missing_required(required: str, solvers=None) -> list[str]:
    """What is wrong with ELMERSTUDIO_TEST_REQUIRE (a comma-separated list of solvers): one line for each solver
    that is not known or not found."""
    solvers = SOLVERS if solvers is None else solvers
    out = []
    for name in (s.strip().lower() for s in required.split(",")):
        if not name:
            continue
        if name not in solvers:
            out.append(f"{name}: not a solver of the tests ({', '.join(sorted(solvers))})")
        elif solvers[name][0]() is None:
            out.append(f"{name}: {solvers[name][1]}")
    return out


def pytest_sessionstart(session):
    """Tests of a solver skip where the solver is not installed.  Where it is installed on purpose (CI names it in
    ELMERSTUDIO_TEST_REQUIRE) a solver that is not found stops the run: those tests must not pass by not running."""
    wrong = missing_required(os.environ.get("ELMERSTUDIO_TEST_REQUIRE", ""))
    if wrong:
        pytest.exit("ELMERSTUDIO_TEST_REQUIRE is not met:\n  " + "\n  ".join(wrong), returncode=1)


def _solver(name):
    find, hint = SOLVERS[name]
    inst = find()
    if inst is None:
        pytest.skip(hint)
    return inst


@pytest.fixture(scope="session")
def openfoam():
    return _solver("openfoam")


@pytest.fixture(scope="session")
def elmer():
    return _solver("elmer")


@pytest.fixture(scope="session")
def nastran():
    return _solver("nastran")