"""Build and solve every example model headless; print one summary line each."""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elmerstudio.core.elmer import find_elmer  # noqa: E402
from elmerstudio.core.examples import EXAMPLES  # noqa: E402
from elmerstudio.core.results import Solution  # noqa: E402
from elmerstudio.core.study_runner import StudyRunner  # noqa: E402


def main(only=()):
    inst = find_elmer()
    fails = 0
    for key, (title, fn) in EXAMPLES.items():
        if only and key not in only:
            continue
        t0 = time.time()
        try:
            m = fn()
            st = m.studies()[0]
            work = tempfile.mkdtemp(prefix=f"es_ex_{key}_")
            folders = StudyRunner(m, st, work, inst).run()
            sol = Solution(folders[-1])
            av = [a for a, *_ in sol.available()]
            i = sol.frame_index("last")
            summary = {}
            for e in ("T", "V", "solid.disp", "spf.U", "mf.normB", "es.normE"):
                if e in av:
                    v = sol.evaluate(e, i)
                    summary[e] = (float(np.nanmin(v)), float(np.nanmax(v)))
            extra = f" eig={[round(f.value, 2) for f in sol.frames][:6]}" if sol.kind == "study.eigen" else ""
            print(f"OK   {key:14s} {time.time() - t0:6.1f}s frames={len(sol.frames)} {summary}{extra}")
        except Exception as exc:
            fails += 1
            print(f"FAIL {key:14s} {exc}")
            traceback.print_exc(limit=3)
    print("failures:", fails)


try:
    import pytest

    @pytest.mark.parametrize("key", list(EXAMPLES))
    def test_example_solves(key, elmer):
        m = EXAMPLES[key][1]()
        folders = StudyRunner(m, m.studies()[0], tempfile.mkdtemp(prefix=f"es_ex_{key}_"), elmer).run()
        sol = Solution(folders[-1])
        assert sol.frames and np.isfinite(sol.evaluate("x", sol.frame_index("last"))).all()
except ImportError:
    pass


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
