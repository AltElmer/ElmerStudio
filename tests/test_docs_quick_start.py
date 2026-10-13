"""The documentation's Quick Start model (docs/modules/ROOT/pages/quick-start.adoc and the scripting tutorial):
a copper bar, T0 at x = 0, q0 into x = L, insulated elsewhere -> T(x) = T0 + q0*x/k, 45 degC at the hot end."""
import tempfile

import numpy as np

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner


def test_quick_start_bar(elmer):
    m = B.new_model("3D", ["ht"], "stationary")
    m.global_defs.child("params").props["table"] = [
        {"name": "L", "expr": "10[cm]", "descr": ""}, {"name": "W", "expr": "2[cm]", "descr": ""},
        {"name": "H", "expr": "1[cm]", "descr": ""}, {"name": "T0", "expr": "20[degC]", "descr": ""},
        {"name": "q0", "expr": "1e5[W/m^2]", "descr": ""}]
    m.create("geom.block", m.component.child("geometry"), index=0, w="L", d="W", h="H")
    geo = build_geometry(m, display=False)
    assert (geo.count("domain"), geo.count("boundary"), geo.count("edge"), geo.count("point")) == (1, 6, 12, 8)
    B.add_material(m, "Copper", all_domains=True)
    ht = m.component.child("physics.ht")
    B.add_feature(m, ht, "ht.temperature", geo.select("boundary", xmax=0), T0="T0")
    B.add_feature(m, ht, "ht.heatflux", geo.select("boundary", xmin=0.1), q0="q0")
    folders = StudyRunner(m, m.studies()[0], tempfile.mkdtemp(prefix="es_qs_"), elmer).run()
    sol = Solution(folders[-1])
    fi = sol.frame_index("last")
    T = sol.evaluate("T", fi)
    x = sol.topo.nodes[:, 0] * sol.scale
    assert abs(T.max() - 273.15 - 45.0) < 1e-6
    assert np.max(np.abs(T - (293.15 + 1e5 * x / 400.0))) < 1e-6      # linear profile, exact on quadratic elements
