"""Qualification of an OpenFOAM run: before a run counts as converged its log must show the expected equations,
finite residuals and the planned end time, and the written result fields must be finite.

The judge is exercised through its public boundary (FoamRunState.feed and judge_foam), so these tests need no
OpenFOAM of their own.

Run:  python -m pytest tests/test_foam_qualification.py -s
"""
from __future__ import annotations

from types import SimpleNamespace

from elmerstudio.core.backends.openfoam.case import FoamPlan
from elmerstudio.core.backends.openfoam.runner import FoamRunState, judge_foam


def heat_plan(transient=False, **kw) -> FoamPlan:
    """The plan of a heat case: the judge reads the solver settings and the time schedule from it."""
    return FoamPlan(physics="ht", app="laplacianFoam", transient=transient, params={}, **kw)


def feed(lines) -> FoamRunState:
    """A run state fed with the lines of a solver log (the solver exited cleanly)."""
    rs = FoamRunState(returncode=0)
    for ln in lines:
        rs.feed(ln)
    return rs


def judge(rs, P, outputs=((1.0, "output-folder"),), label="Heat"):
    return judge_foam(rs, P, list(outputs), label)


def solve_line(field="T", r0="0.1", r1="1e-12", it=1):
    return f"PCG: Solving for {field}, Initial residual = {r0}, Final residual = {r1}, No Iterations {it}"


def field_file(name="T", values="300\n350\n400"):
    """A written OpenFOAM field (the ascii form the case writes)."""
    return ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volScalarField;\n"
            f"    object      {name};\n}}\n\ndimensions      [0 0 0 1 0 0 0];\n\n"
            f"internalField   nonuniform List<scalar> 3\n(\n{values}\n)\n;\n\n"
            "boundaryField\n{\n    inlet\n    {\n        type            fixedValue;\n"
            "        value           uniform 300;\n    }\n}\n")


# =========================================================================== the controls of the review
def test_the_control_cases_of_the_review():
    """The four cases of the review, through the public boundary: a residual below the linear tolerance is a
    converged run, a large one is not, a non-finite one is not, and a log without a solve is not."""
    P = SimpleNamespace(app="laplacianFoam", transient=True, lin_tol=1e-10, lin_maxit=1000)
    cases = [("finite", "1e-12", "converged"), ("large", "0.01", "not converged"),
             ("nonfinite", "nan", "not converged"), ("no-solve", None, "not converged")]
    for name, residual, expected in cases:
        rs = FoamRunState(returncode=0)
        rs.feed("Time = 1")
        if residual is not None:
            rs.feed(solve_line(r1=residual))
        rs.feed("End")
        status, why = judge_foam(rs, P, [(1.0, "existing-output-folder")], "Heat")
        print(f"{name}: parsed solves {len(rs.solves)}, status {status!r} (expected {expected!r}), {why}")
        assert status == expected, (name, status, why)


def test_a_finite_residual_converges():
    """The control of the control: a residual well below the linear tolerance is a converged steady run."""
    rs = feed(["Time = 1", solve_line(r1="1e-12"), "End"])
    status, why = judge(rs, heat_plan())
    print(f"finite residual: {len(rs.solves)} solve(s), status {status!r}, {why}")
    assert len(rs.solves) == 1 and status == "converged", (status, why)


def test_a_residual_above_the_tolerance_is_not_converged():
    """The control of the control: a residual above the linear tolerance is not a converged run."""
    rs = feed(["Time = 1", solve_line(r1="0.01"), "End"])
    status, why = judge(rs, heat_plan())
    print(f"large residual: status {status!r}, {why}")
    assert status == "not converged" and why


# =========================================================================== the defects
def test_a_nonfinite_residual_is_not_converged():
    """A residual of nan or inf (in any case and with any sign) is a failed solve, not a converged one: the
    number must be read and rejected, not skipped."""
    for token in ("nan", "inf", "-inf", "-nan", "NaN", "INF", "+inf"):
        rs = feed(["Time = 1", solve_line(r1=token), "End"])
        status, why = judge(rs, heat_plan())
        print(f"residual {token!r}: parsed solves {len(rs.solves)}, status {status!r}, {why}")
        assert len(rs.solves) == 1, f"the solve line with {token!r} was not read: {rs.lines}"
        assert status == "not converged", (token, status, why)


def test_a_log_without_a_solve_is_not_converged():
    """A log that reports no solve at all (the expected equation was never solved) does not qualify."""
    rs = feed(["Time = 1", "End"])
    status, why = judge(rs, heat_plan())
    print(f"no solve: parsed solves {len(rs.solves)}, status {status!r}, {why}")
    assert len(rs.solves) == 0
    assert status == "not converged", (status, why)


def test_a_log_without_end_is_reported_as_incomplete():
    """A log that ends without the solver's 'End' is an unfinished run: it is failed, and the diagnostic says
    that the log is incomplete (not that the solver exited with code 0)."""
    rs = feed(["Time = 1", solve_line(r1="1e-12")])
    status, why = judge(rs, heat_plan())
    print(f"without End: status {status!r}, {why}")
    assert status == "failed", (status, why)
    assert why and "End" in why[0], why


def test_a_transient_run_that_stops_before_the_end_time_is_not_converged():
    """A transient run that stops at t = 0.5 s while the plan asks for 1 s (its output times are 0.1 ... 1 s)
    has not solved the planned time span: it does not qualify."""
    P = heat_plan(transient=True, t0=0.0, dt=0.1, nsteps=10, out_times=[0.1 * i for i in range(1, 11)])
    rs = feed(["Time = 0", solve_line(r1="1e-12"), "Time = 0.5", solve_line(r1="1e-12"), "End"])
    status, why = judge(rs, P, outputs=[(0.5, "output-folder")])
    print(f"short transient: status {status!r}, {why}")
    assert status == "not converged", (status, why)
    assert any("time" in d for d in why), why


def test_a_transient_run_that_reaches_the_end_time_converges():
    """The control of the transient check: the same run, with every planned output time written and the log at
    the planned end time, converges."""
    P = heat_plan(transient=True, t0=0.0, dt=0.5, nsteps=2, out_times=[0.5, 1.0])
    rs = feed(["Time = 0", solve_line(r1="1e-12"), "Time = 0.5", solve_line(r1="1e-12"), "Time = 1", solve_line(r1="1e-12"),
               "End"])
    status, why = judge(rs, P, outputs=[(0.5, "output-folder"), (1.0, "output-folder")])
    print(f"complete transient: status {status!r}, {why}")
    assert status == "converged", (status, why)


def test_a_transient_run_with_a_rounded_printed_time_converges():
    """OpenFOAM prints the time with the precision of its timePrecision setting: a complete run whose printed end
    time is rounded (0.123456 for the planned 0.1234564) has still reached the end of the plan."""
    dt = 0.01234564
    P = heat_plan(transient=True, t0=0.0, dt=dt, nsteps=10, out_times=[dt * i for i in range(1, 11)])
    rs = feed(["Time = 0.0123456", solve_line(r1="1e-12"), "Time = 0.123456", solve_line(r1="1e-12"), "End"])
    status, why = judge(rs, P, outputs=[(dt * i, "output-folder") for i in range(1, 11)])
    print(f"rounded printed time: status {status!r}, {why}")
    assert status == "converged", (status, why)


# =========================================================================== the written fields
def test_a_written_field_with_a_nonfinite_value_does_not_qualify(tmp_path):
    """A result field that holds nan (or inf) is not a result: the run does not qualify, however good its log."""
    folder = tmp_path / "1"
    folder.mkdir()
    (folder / "T").write_text(field_file("T", values="300\nnan\n400"))
    rs = feed(["Time = 1", solve_line(r1="1e-12"), "End"])
    status, why = judge(rs, heat_plan(), outputs=[(1.0, str(folder))])
    print(f"field with nan: status {status!r}, {why}")
    assert status == "not converged", (status, why)


def test_a_written_field_with_an_infinite_value_does_not_qualify(tmp_path):
    """The same for inf."""
    folder = tmp_path / "1"
    folder.mkdir()
    (folder / "T").write_text(field_file("T", values="300\ninf\n400"))
    rs = feed(["Time = 1", solve_line(r1="1e-12"), "End"])
    status, why = judge(rs, heat_plan(), outputs=[(1.0, str(folder))])
    print(f"field with inf: status {status!r}, {why}")
    assert status == "not converged", (status, why)


def test_a_written_finite_field_qualifies(tmp_path):
    """The control of the field check: a finite field and a good log qualify."""
    folder = tmp_path / "1"
    folder.mkdir()
    (folder / "T").write_text(field_file("T"))
    rs = feed(["Time = 1", solve_line(r1="1e-12"), "End"])
    status, why = judge(rs, heat_plan(), outputs=[(1.0, str(folder))])
    print(f"finite field: status {status!r}, {why}")
    assert status == "converged", (status, why)
