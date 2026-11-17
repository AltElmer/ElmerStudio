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


@pytest.fixture(scope="session")
def openfoam():
    from elmerstudio.core.backends.openfoam.install import find_openfoam
    inst = find_openfoam()
    if inst is None:
        pytest.skip("OpenFOAM not installed (set OPENFOAM_ROOT)")
    return inst


@pytest.fixture(scope="session")
def elmer():
    from elmerstudio.core.elmer import find_elmer
    inst = find_elmer()
    if inst is None:
        pytest.skip("Elmer (ElmerSolver) not installed")
    return inst


# Skip budget tracking
_skip_budget_count = 0


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    outcome.get_result().item = item


def pytest_runtest_logreport(report):
    """Track skipped tests for budget enforcement."""
    global _skip_budget_count
    if report.skipped:
        # Check if the test is exempt from skip budget
        item = report.item
        if not any(m.name == "skip_budget_exempt" for m in item.own_markers):
            _skip_budget_count += 1
            print(f"DEBUG: Skip counted, total={_skip_budget_count}", file=sys.stderr)


def pytest_sessionfinish(session, exitstatus):
    """Enforce skip budget: fail if too many tests are skipped.

    Budget is set via ELMERSTUDIO_SKIP_BUDGET environment variable (default: 5).
    Tests marked with @pytest.mark.skip_budget_exempt are not counted.
    """
    global _skip_budget_count
    if exitstatus != 0:
        return  # already failed, don't mask the real failure

    budget = int(os.environ.get("ELMERSTUDIO_SKIP_BUDGET", "5"))
    if _skip_budget_count > budget:
        print(f"SKIP BUDGET EXCEEDED: {_skip_budget_count} skipped (budget: {budget})")
        session.exitstatus = 1
    # Reset for next session (if any)
    _skip_budget_count = 0
