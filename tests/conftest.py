import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*extract_surface.*")

# a script (drives the GUI event loop); run it directly:  python tests/test_gui_smoke.py
collect_ignore = ["test_gui_smoke.py"]


@pytest.fixture(scope="session")
def elmer():
    from elmerstudio.core.elmer import find_elmer
    inst = find_elmer()
    if inst is None:
        pytest.skip("Elmer (ElmerSolver) not installed")
    return inst
