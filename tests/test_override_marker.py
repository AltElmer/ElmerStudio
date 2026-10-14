"""The Settings window's "(overridden)" marker follows build_sif: a later boundary feature replaces an earlier one, but
domain features merge key by key, so only a later domain feature of the same type overrides."""
from types import SimpleNamespace

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.ui.main_window import MainWindow


def test_override_marker_matches_the_sif_rules():
    m = B.new_model("2D", ["tds"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1", h="1")
    geo = build_geometry(m, display=False)
    tds = m.component.child("physics.tds")
    tp = tds.child("tds.tp")
    ctx = SimpleNamespace(geo=geo)
    over = lambda n: MainWindow.overridden_entities(ctx, n)          # noqa: E731
    B.add_feature(m, tds, "tds.reac", None, R0="1").selection.all = True
    assert over(tp) == set()                                           # Reactions adds a source, tp still applies
    B.add_feature(m, tds, "tds.tp", None, D="2e-9").selection.all = True
    assert over(tp) == {1}                                             # a second Transport Properties overrides
    left = geo.select("boundary", xmax=0)
    c1 = B.add_feature(m, tds, "tds.conc", left, c0="1")
    B.add_feature(m, tds, "tds.flux", left, N0="1")
    assert over(c1) == set(left)                                       # a later boundary condition replaces it
