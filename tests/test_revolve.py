"""Revolution-2D plot of an axisymmetric solution produces a 3D scene with field data."""
from __future__ import annotations

import os
import sys
import tempfile
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core.elmer import find_elmer  # noqa: E402
from elmerstudio.core.examples import transient_sphere  # noqa: E402
from elmerstudio.core.results import Solution  # noqa: E402
from elmerstudio.core.study_runner import StudyRunner  # noqa: E402
from elmerstudio.ui.plots import build_scene  # noqa: E402


def test_revolve_plot(elmer):
    m = transient_sphere()
    st = m.studies()[0]
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_rev_"), elmer or find_elmer()).run()
    B.default_plots(m, st, "time")
    pg = next(c for c in m.results.children if c.kind == "pg2d")
    pg.props["revolve"] = True
    sol = Solution(folders[-1])
    sc = build_scene(pg, sol, 0, 2, axi=True)
    print("sdim", sc.sdim, "items", [(type(d).__name__, d.n_cells, "f" in d.point_data) for d, _ in sc.items],
          "bounds", [round(b, 4) for b in sc.items[0][0].bounds])
    assert sc.sdim == 3 and sc.items and "f" in sc.items[0][0].point_data
    b = sc.items[0][0].bounds
    assert b[4] < -0.015 and b[1] > 0.015, b      # swept into negative z (x-y plane is r-z)


if __name__ == "__main__":
    test_revolve_plot(find_elmer())
