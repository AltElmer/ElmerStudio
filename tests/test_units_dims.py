"""Units (M2): a unit binds to its operand as in COMSOL, the unit of every expression is deduced and compared with
the setting's (a mismatch is reported, never silently dropped), deduced units do not depend on the order of the
parameter rows, and a model written in other units gives the same physical result."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from elmerstudio.core import builder as B  # noqa: E402
from elmerstudio.core import units as U  # noqa: E402
from elmerstudio.core.diagnostics import unit_report  # noqa: E402


def test_a_unit_binds_to_its_operand():
    assert U.evaluate("6/2[mm]") == pytest.approx(3000.0)            # 6 / (2 mm), not (6/2) mm
    assert U.evaluate("x/2[mm]", {"x": 1.0}) == pytest.approx(500.0)
    assert U.evaluate("2*3[mm]") == pytest.approx(0.006)
    assert U.evaluate("(1+1)[mm]") == pytest.approx(0.002)
    assert U.evaluate("sqrt(4)[mm]") == pytest.approx(0.002)
    assert U.evaluate("20[degC]") == pytest.approx(293.15)
    assert U.evaluate("-5[K]") == -5.0


def test_deduced_units():
    assert U.fmt_dim(U.unit_dims("kJ/(kg*K)")) == "J/(kg*K)"
    assert U.fmt_dim(U.unit_dims("mW/mm^2")) == "W/m^2"
    d, pure, w = U.expr_dims("q0*exp(-t/tau)", {"q0": (U.unit_dims("W/m^2"), False), "t": U.unit_dims("s"),
                                                 "tau": (U.unit_dims("s"), False)})
    assert U.fmt_dim(d) == "W/m^2" and not pure and not w
    assert U.unit_warnings("300", "K") == []                                 # a plain number: in the field's unit
    assert U.unit_warnings("k0*(1+0.002[1/K]*(T-293.15[K]))", "W/(m*K)",
                           {"k0": (U.unit_dims("W/(m*K)"), False), "T": U.unit_dims("K")}) == []


@pytest.mark.parametrize("expr,expected,says", [
    ("1[m]+1[s]", None, "inconsistent units in a sum"),
    ("2930[K]", "kg/m^3", "[K] deduced, [kg/m^3] expected"),      # COMSOL's own example: used, but reported
    ("sin(2[m])", "1", "sin() of a quantity with a unit"),
    ("2^(1[s])", None, "an exponent with a unit"),
    ("(T > 300[m])", None, "inconsistent units in a comparison"),
])
def test_dimensionally_invalid_expressions_are_reported(expr, expected, says):
    w = U.unit_warnings(expr, expected, {"T": U.unit_dims("K")})
    assert any(says in x for x in w), w


def test_deduced_units_do_not_depend_on_row_order():
    rows = [("L", "20[mm]"), ("A", "L^2"), ("q", "P/A"), ("P", "5[W]"), ("n", "3")]
    ref = U.parameter_dims(rows)
    for perm in ([4, 3, 2, 1, 0], [2, 0, 4, 1, 3], [1, 2, 3, 4, 0]):
        assert U.parameter_dims([rows[i] for i in perm]) == ref
    assert U.fmt_dim(ref["q"][0]) == "W/m^2" and ref["n"] == (U.DIMLESS, True)


def test_the_model_report_names_the_setting():
    m = B.new_model("2D", ["ht"], "stationary")
    mat = B.add_material(m, "Copper", all_domains=True)
    assert unit_report(m) == {}                                             # control: a correct model is clean
    mat.props["values"]["rho"] = "5[K]"
    f = B.add_feature(m, m.component.child("physics.ht"), "ht.temperature", [1], T0="300[K]+2[m]")
    rep = unit_report(m)
    assert any("Density" in w and "[K] deduced" in w for w in rep[mat.tag])
    assert any("inconsistent units in a sum" in w for w in rep[f.tag])


def _strip_model(geom_unit: str, L: str, q: str, k_unit_scale: str):
    """2D strip, temperature fixed at x=0, a heat flux in at x=L: T(L) = T0 + q L / k."""
    m = B.new_model("2D", ["ht"], "stationary")
    m.global_defs.child("params").props["table"] = [{"name": "L", "expr": L}, {"name": "q", "expr": q}]
    g = m.component.child("geometry")
    g.props["unit"] = geom_unit
    m.create("geom.rectangle", g, index=0, w="L", h="L/4")
    mat = B.add_material(m, "Copper", all_domains=True)
    mat.props["values"]["k"] = k_unit_scale
    from elmerstudio.core.geometry import build_geometry
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")
    s = geo.scale
    B.add_feature(m, ph, "ht.temperature", geo.select("boundary", xmax=0), T0="293.15[K]")
    B.add_feature(m, ph, "ht.heatflux", geo.select("boundary", xmin=float(U.evaluate(L)) / s), q0="q")
    return m


def test_identical_results_in_other_units(elmer):
    from test_pipeline import read, solve
    out = []
    for args in (("m", "0.02[m]", "5000[W/m^2]", "400[W/(m*K)]"),
                 ("mm", "20[mm]", "5[mW/mm^2]", "0.4[W/(mm*K)]")):
        m = _strip_model(*args)
        assert unit_report(m) == {}, unit_report(m)
        _mesh, _sif, _rs, files, _case = solve(m)
        out.append(float(np.max(read(files[-1]).point_data["temperature"])))
    exact = 293.15 + 5000 * 0.02 / 400
    assert out[0] == pytest.approx(exact, rel=1e-6) and out[1] == pytest.approx(out[0], rel=1e-9), out
