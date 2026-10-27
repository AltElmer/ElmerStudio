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
def elmer():
    from elmerstudio.core.elmer import find_elmer
    inst = find_elmer()
    if inst is None:
        pytest.skip("Elmer (ElmerSolver) not installed")
    return inst
