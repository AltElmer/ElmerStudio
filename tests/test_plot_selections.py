"""Results on Box Selections and partial physics, as in COMSOL:

* a Line Graph on a Box Selection (made after solving) resolves it as each solution's geometry did, and with
  "All" on a Parametric Sweep draws one curve per parameter value;
* a 1D plot group's manual axis limits;
* the fields of a physics active on some domains only are undefined (NaN) elsewhere, where Elmer writes zeros.
"""
import tempfile

import numpy as np
from matplotlib.figure import Figure

from elmerstudio.core import builder as B
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.nodes import bind_results
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner
from elmerstudio.ui.plots import draw_1d


def two_blocks(elmer):
    m = B.new_model("2D", ["tds"], "stationary")
    m.global_defs.child("params").props["table"] = [{"name": "cin", "expr": "1[mol/m^3]", "descr": ""}]
    g = m.component.child("geometry")
    m.create("geom.rectangle", g, index=0, w="2", h="1")
    m.create("geom.rectangle", g, index=1, w="1", h="1", x="2")
    box = m.create("geom.boxsel", g, index=2, level="boundary", xmax="2", ymax="0")      # bottom of the left block
    geo = build_geometry(m, display=False)
    left = [d for d in geo.numbers["domain"] if geo.bbox_of["domain"][d][3] <= 2 + 1e-6]
    tds = m.component.child("physics.tds")
    tds.selection.all, tds.selection.entities = False, left
    B.add_feature(m, tds, "tds.conc", geo.select("boundary", xmax=0), c0="cin")
    B.add_feature(m, tds, "tds.conc", geo.select("boundary", xmin=2, xmax=2), c0="0")
    st = m.studies()[0]
    m.create("study.sweep", st, pname="cin", pvalues="1 3", punit="mol/m^3")
    folders = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_psel_"), elmer).run()
    sols = [bind_results(m, Solution(f), geo) for f in folders]
    return m, geo, st, box, left, sols


def test_box_selection_line_graph_sweep_axes_and_partial_physics(elmer):
    m, geo, st, box, left, sols = two_blocks(elmer)
    # partial physics: c is defined in the left block only
    s = sols[-1]
    c = s.evaluate("c", s.frame_index("last"))
    x = s.topo.nodes[:, 0] * s.scale
    assert np.isfinite(c[x <= 2 + 1e-9]).all() and np.isnan(c[x > 2 + 1e-9]).all()
    # a Line Graph of c on the Box Selection, over both sweep values, with manual axis limits
    ds = B.solution_dataset(m, st)
    pg = m.create("pg1d", m.results)
    pg.props.update(data=ds.tag, solnum="all", manual_axes=True, xmin="0", xmax="2", ymax="4")
    for ch in list(pg.children):
        pg.remove(ch)
    lg = m.create("plot.linegraph", pg)
    lg.props.update(data="fromparent", expr="c", xaxis="x")
    lg.selection.level, lg.selection.all, lg.selection.named = "boundary", False, box.tag     # no entities yet
    sweep = [(float(sl.meta["sweep_params"]["cin"]), sl) for sl in sols]
    fig = Figure()
    draw_1d(fig, pg, m, lambda d, sel: (sols[-1], sols[-1].frame_index("last"), sols[-1].scale), "all",
            get_sweep=lambda d: (sweep, "cin", "mol/m^3"))
    ax = fig.axes[0]
    assert len(ax.lines) == 2, [t.get_text() for t in ax.texts]
    tops = sorted(float(np.nanmax(ln.get_ydata())) for ln in ax.lines)
    assert np.allclose(tops, [1, 3], rtol=1e-6)                       # c = cin at the inlet: one curve per value
    assert ax.get_xlim() == (0.0, 2.0) and ax.get_ylim()[1] == 4.0
