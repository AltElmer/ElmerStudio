"""StudyRunner + Solution + project round-trip (needs Elmer)."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elmerstudio.core import builder as B, project  # noqa: E402
from elmerstudio.core.elmer import find_elmer  # noqa: E402
from elmerstudio.core.geometry import build_geometry  # noqa: E402
from elmerstudio.core.results import Solution  # noqa: E402
from elmerstudio.core.study_runner import StudyRunner  # noqa: E402


def main():
    inst = find_elmer()
    m = B.new_model("3D", ["ht"], "stationary")
    m.global_defs.child("params").props["table"] = [{"name": "Q", "expr": "1e6[W/m^3]", "descr": ""},
                                                      {"name": "L", "expr": "10[cm]", "descr": ""}]
    geom = m.component.child("geometry")
    geom.props["unit"] = "mm"
    m.create("geom.block", geom, index=0, w="L", d="20", h="20")
    B.add_material(m, "Structural steel", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    B.add_feature(m, ph, "ht.source", [1], Q0="Q")
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0) + geo.select("boundary", xmin=100), T0="0[K]")
    st = m.studies()[0]
    sw = m.create("study.sweep", st, index=0, pname="Q", pvalues="1e6 2e6", punit="W/m^3")
    work = tempfile.mkdtemp(prefix="es_run_")
    events = []
    r = StudyRunner(m, st, work, inst, on_event=lambda k, p: events.append(k))
    folders = r.run()
    print("folders", folders, "events", {k: events.count(k) for k in set(events)})
    for f in folders:
        sol = Solution(f)
        i = sol.frame_index("last")
        Tmax = sol.evaluate("T", i).max()
        vol = sol.integrate("1", i, "domain", [1])
        Tav = sol.integrate("T", i, "domain", [1], average=True)
        mx, mn = sol.maxmin("T-273.15", i)
        pe = sol.point_eval("T", i, [[0.05, 0.01, 0.01]])
        s, pts, line = sol.line_eval("T", i, (0, 0.01, 0.01), (0.1, 0.01, 0.01), 50)
        assert abs(s[-1] - 0.1) < 1e-12 and abs(np.nanmax(line) - Tmax) / Tmax < 1e-3, (s[-1], np.nanmax(line))
        print(f"{os.path.basename(f)}: Tmax={Tmax:.4f} K (exact {1e6 * 0.01 / (8 * 44.5) * (2 if f.endswith('1_0') else 1):.4f}), "
              f"vol={vol:.4e} m^3 (exact 4e-5), Tavg={Tav:.4f} (exact {2 / 3 * Tmax:.4f}), max-273.15={mx:.3f}, "
              f"T(center)={pe[0]:.4f}, line pts={len(s)}, vars={[a for a, *_ in sol.available()][:6]}")
    out = os.path.join(work, "roundtrip.esm")
    project.save(m, out, work)
    m2, w2 = project.load(out)
    print("roundtrip nodes:", sum(1 for _ in m.root.walk()), sum(1 for _ in m2.root.walk()),
          "solutions restored:", os.path.isdir(os.path.join(w2, st.tag)))


if __name__ == "__main__":
    main()
