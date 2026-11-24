"""Mode shapes of an eigenfrequency study (issue #13: eigenvalues computed, every mode shape zero).

Elmer writes mode N of a field as "<field> EigenMode<N>". When its solver holds no eigenvectors at the time of the
output, it writes the field under its own name, as zeros. The first test solves the cantilever and checks the
shapes; the others damage the result file of that solve in the two ways such a failure shows, and check that it
is reported.
"""
import glob
import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from elmerstudio.core import study_runner
from elmerstudio.core.examples import cantilever_eigen
from elmerstudio.core.results import Solution, mode_shape_problems
from elmerstudio.core.study_runner import StudyRunner

MODES = 6


def solve(elmer):
    events = []
    model = cantilever_eigen()
    runner = StudyRunner(model, model.studies()[0], tempfile.mkdtemp(prefix="es_eigen_"), elmer,
                         on_event=lambda kind, payload: events.append((kind, payload)))
    return runner.run()[0], events


@pytest.fixture(scope="module")
def solved(elmer):
    return solve(elmer)


def damaged(folder, change):
    """a copy of the solution folder whose result file went through change(point_data)"""
    copy = os.path.join(tempfile.mkdtemp(prefix="es_eigen_damaged_"), "sol")
    shutil.copytree(folder, copy)
    import pyvista as pv
    file = sorted(glob.glob(os.path.join(copy, "results", "case_t*.vtu")))[-1]
    grid = pv.read(file)
    change(grid.point_data)
    grid.save(file)
    return copy


def mode_arrays(data, mode):
    return [k for k in data.keys() if k.lower().endswith(f" eigenmode{mode}")]


def test_every_mode_has_a_shape(solved):
    folder, events = solved
    sol = Solution(folder)
    assert sol.kind == "study.eigen" and len(sol.frames) == MODES
    assert mode_shape_problems(folder, MODES) == []
    assert not [p for kind, p in events if kind == "warning"]
    assert sol.meta["diagnostics"] == []
    x = sol.topo.nodes[:, 0]
    shapes, vectors = [], []
    for i in range(MODES):
        u = np.asarray(sol.evaluate("solid.disp", i), float)
        assert np.isfinite(u).all() and u.max() > 0, f"mode {i + 1} has no displacement"
        assert u[np.isclose(x, 0.0)].max() < 1e-9 * u.max(), f"mode {i + 1} moves at the fixed end"
        shapes.append(u / u.max())
        v = np.array(sol.arrays(i)["displacement"], float).ravel()
        vectors.append(v / np.linalg.norm(v))
    # the first mode is the first bending mode of a beam clamped at x = 0: largest at the free end, and growing
    assert np.isclose(x[np.argmax(shapes[0])], x.max())
    middle, tip = shapes[0][np.isclose(x, 0.5 * x.max(), atol=0.06 * x.max())], shapes[0][np.isclose(x, x.max())]
    assert 0.2 < middle.mean() < 0.5 and tip.min() > 0.95            # Euler-Bernoulli: 0.34 at mid-span
    # each frame shows its own mode, not one array six times: modes are orthogonal in the mass matrix, so nearly
    # orthogonal as vectors of nodal values (the two bending directions of the square section included)
    overlap = np.abs(np.array(vectors) @ np.array(vectors).T - np.eye(MODES))
    assert overlap.max() < 0.3


def test_a_file_without_mode_arrays_is_reported(solved):
    """what Elmer writes when it has no eigenvectors: the field under its own name, zeros"""
    def strip(data):
        zeros = np.zeros_like(np.asarray(data[mode_arrays(data, 1)[0]]))
        for n in range(1, MODES + 1):
            for k in mode_arrays(data, n):
                del data[k]
        data["displacement"] = zeros

    folder = damaged(solved[0], strip)
    assert float(np.abs(Solution(folder).evaluate("solid.disp", 0)).max()) == 0.0     # the plot the reporter saw
    problems = mode_shape_problems(folder, MODES)
    assert len(problems) == 1
    assert "no shape for mode 1, 2, 3, 4, 5, 6 of 6" in problems[0] and "displacement" in problems[0]


def test_a_mode_array_of_zeros_is_reported(solved):
    def zero(data):
        for k in mode_arrays(data, 2):
            data[k] = np.zeros_like(np.asarray(data[k]))

    folder = damaged(solved[0], zero)
    problems = mode_shape_problems(folder, MODES)
    assert len(problems) == 1 and "mode 2 of 6 has no non-zero value" in problems[0]
    sol = Solution(folder)
    assert float(np.abs(sol.evaluate("solid.disp", 1)).max()) == 0.0
    assert float(np.abs(sol.evaluate("solid.disp", 0)).max()) > 0.0


def test_a_mode_array_wins_over_a_plain_array_of_the_same_field(solved):
    def add_plain(data):
        data["displacement"] = np.zeros_like(np.asarray(data[mode_arrays(data, 1)[0]]))

    sol = Solution(damaged(solved[0], add_plain))
    assert float(np.abs(sol.evaluate("solid.disp", 0)).max()) > 0.0


def test_the_run_warns_and_records_missing_mode_shapes(elmer, monkeypatch):
    monkeypatch.setattr(study_runner, "mode_shape_problems", lambda folder, modes: [f"no shape, {modes} modes"])
    folder, events = solve(elmer)
    warnings = [p for kind, p in events if kind == "warning"]
    assert len(warnings) == 1 and warnings[0].startswith("NO MODE SHAPE: ") and "no shape, 6 modes" in warnings[0]
    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)
    assert meta["status"] == "converged" and len(meta["eigen"]) == MODES
    assert meta["diagnostics"] == [warnings[0][len("NO MODE SHAPE: "):]]
