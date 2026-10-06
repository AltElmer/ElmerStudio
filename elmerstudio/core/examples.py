"""Example models (the 'Application Libraries'), built with the scripting API."""
from __future__ import annotations

from . import builder as B
from .geometry import build_geometry


def _params(m, rows):
    m.global_defs.child("params").props["table"] = [{"name": n, "expr": e, "descr": d} for n, e, d in rows]


def busbar():
    """Joule heating of a copper busbar (electric currents + heat transfer)."""
    m = B.new_model("3D", ["@jh"], "stationary")
    m.root.label = "busbar"
    _params(m, [("L", "9[cm]", "Length"), ("tbb", "5[mm]", "Thickness"), ("wbb", "5[cm]", "Width"),
                ("rad_1", "6[mm]", "Bolt hole radius"), ("Vtot", "20[mV]", "Applied voltage"),
                ("htc", "5[W/(m^2*K)]", "Heat transfer coefficient")])
    g = m.component.child("geometry")
    g.props["unit"] = "mm"
    b1 = m.create("geom.block", g, index=0, w="L+2*tbb", d="wbb", h="tbb")
    b2 = m.create("geom.block", g, index=1, w="tbb", d="wbb", h="L", x="0", z="tbb")
    b3 = m.create("geom.block", g, index=2, w="tbb", d="wbb", h="L/2", x="L+tbb", z="-L/2+tbb")
    u = m.create("geom.union", g, index=3, input=[b1.tag, b2.tag, b3.tag], interior=False)
    c1 = m.create("geom.cylinder", g, index=4, r="rad_1", h="3*tbb", x="-tbb", y="wbb/2", z="L*0.75", axis="x")
    c2 = m.create("geom.cylinder", g, index=5, r="rad_1", h="3*tbb", x="L", y="wbb/2", z="-L/4", axis="x")
    m.create("geom.difference", g, index=6, input=[u.tag], tools=[c1.tag, c2.tag])
    B.add_material(m, "Copper", all_domains=True)
    geo = build_geometry(m, display=False)
    top = geo.select("boundary", zmin=90 + 5 - 1e-6)            # top end of the tall leg
    bottom = geo.select("boundary", zmax=-45 + 5 + 1e-6)        # bottom end of the short leg
    ec = m.component.child("physics.ec")
    B.add_feature(m, ec, "ec.pot", top, V0="Vtot")
    B.add_feature(m, ec, "ec.ground", bottom)
    ht = m.component.child("physics.ht")
    allb = geo.numbers["boundary"]
    B.add_feature(m, ht, "ht.heatflux", [b for b in allb if b not in top + bottom], type="convective", h="htc",
                  Text="293.15[K]")
    B.add_feature(m, ht, "ht.temperature", top + bottom, T0="293.15[K]")
    return m


def cantilever_eigen():
    m = B.new_model("3D", ["solid"], "eigen")
    m.root.label = "cantilever_eigen"
    _params(m, [("Lb", "1[m]", "Beam length"), ("hb", "5[cm]", "Height"), ("wb", "5[cm]", "Width")])
    g = m.component.child("geometry")
    m.create("geom.block", g, index=0, w="Lb", d="wb", h="hb")
    B.add_material(m, "Structural steel", all_domains=True)
    geo = build_geometry(m, display=False)
    B.add_feature(m, m.component.child("physics.solid"), "solid.fixed", geo.select("boundary", xmax=0))
    B.study_steps(m.studies()[0])[0].props["neig"] = 6
    return m


def heat_sink():
    m = B.new_model("3D", ["ht"], "stationary")
    m.root.label = "heat_sink"
    _params(m, [("P0", "10[W]", "Dissipated power"), ("nfin", "6", "Number of fins"),
                ("tf", "1.5[mm]", "Fin thickness"), ("pitch", "6[mm]", "Fin pitch"), ("hf", "25[mm]", "Fin height")])
    g = m.component.child("geometry")
    g.props["unit"] = "mm"
    base = m.create("geom.block", g, index=0, w="nfin*pitch", d="40", h="4")
    fin = m.create("geom.block", g, index=1, w="tf", d="40", h="hf", x="(pitch-tf)/2", z="4")
    arr = m.create("geom.array", g, index=2, input=[fin.tag], nx="6", ny="1", nz="1", dx="pitch", dy="0", dz="0")
    arr.props["nx"] = 6
    m.create("geom.union", g, index=3, input=[base.tag, arr.tag], interior=False)
    B.add_material(m, "Aluminum", all_domains=True)
    geo = build_geometry(m, display=False)
    ht = m.component.child("physics.ht")
    bottom = geo.select("boundary", zmax=0)
    B.add_feature(m, ht, "ht.heatflux", bottom, q0="P0/(nfin*pitch*40[mm])")
    others = [b for b in geo.numbers["boundary"] if b not in bottom]
    B.add_feature(m, ht, "ht.heatflux", others, type="convective", h="15[W/(m^2*K)]", Text="20[degC]")
    return m


def flow_cylinder():
    m = B.new_model("2D", ["spf"], "stationary")
    m.root.label = "flow_past_cylinder"
    _params(m, [("U0", "2[mm/s]", "Inlet velocity (Re = 20)"), ("D", "1[cm]", "Cylinder diameter")])
    g = m.component.child("geometry")
    r = m.create("geom.rectangle", g, index=0, w="0.22", h="0.041")
    c = m.create("geom.circle", g, index=1, r="D/2", x="0.05", y="0.02")
    m.create("geom.difference", g, index=2, input=[r.tag], tools=[c.tag])
    B.add_material(m, "Water, liquid", all_domains=True)
    geo = build_geometry(m, display=False)
    spf = m.component.child("physics.spf")
    B.add_feature(m, spf, "spf.inlet", geo.select("boundary", xmax=0), U0="U0")
    B.add_feature(m, spf, "spf.outlet", geo.select("boundary", xmin=0.22))
    m.component.child("mesh").props["size"] = "3"
    return m


def capacitor():
    m = B.new_model("2D", ["es"], "stationary")
    m.root.label = "capacitor_fringe"
    _params(m, [("V0", "1[kV]", "Plate voltage"), ("gap", "2[mm]", "Plate gap"), ("tp", "0.5[mm]", "Plate thickness"),
                ("wp", "10[mm]", "Plate width")])
    g = m.component.child("geometry")
    g.props["unit"] = "mm"
    box = m.create("geom.rectangle", g, index=0, w="40", h="30", base="center")
    p1 = m.create("geom.rectangle", g, index=1, w="wp", h="tp", base="center", y="(gap+tp)/2")
    p2 = m.create("geom.rectangle", g, index=2, w="wp", h="tp", base="center", y="-(gap+tp)/2")
    m.create("geom.difference", g, index=3, input=[box.tag], tools=[p1.tag, p2.tag])
    B.add_material(m, "Air", all_domains=True)
    geo = build_geometry(m, display=False)
    es = m.component.child("physics.es")
    upper = geo.select("boundary", xmin=-5, xmax=5, ymin=1, ymax=1.5)
    lower = geo.select("boundary", xmin=-5, xmax=5, ymin=-1.5, ymax=-1)
    B.add_feature(m, es, "es.pot", upper, V0="V0")
    B.add_feature(m, es, "es.ground", lower)
    return m


def coil_axi():
    """Axisymmetric copper coil around an iron core: magnetic flux density."""
    m = B.new_model("2Daxi", ["mf"], "stationary")
    m.root.label = "coil_axisymmetric"
    _params(m, [("J0", "2[A/mm^2]", "Coil current density")])
    g = m.component.child("geometry")
    g.props["unit"] = "mm"
    air = m.create("geom.rectangle", g, index=0, w="100", h="200", y="-100")
    core = m.create("geom.rectangle", g, index=1, w="10", h="60", y="-30")
    coil = m.create("geom.rectangle", g, index=2, w="10", h="40", x="15", y="-20")
    m.create("geom.union", g, index=3, input=[air.tag, core.tag, coil.tag])
    B.add_material(m, "Air", all_domains=True)
    geo = build_geometry(m, display=False)
    core_d = geo.select("domain", xmax=10, ymin=-30, ymax=30)
    coil_d = geo.select("domain", xmin=15, xmax=25, ymin=-20, ymax=20)
    B.add_material(m, "Iron (soft magnetic)", core_d)
    B.add_material(m, "Copper", coil_d)
    mf = m.component.child("physics.mf")
    B.add_feature(m, mf, "mf.ecd", coil_d, Jz="J0")
    # magnetic insulation on the outer boundaries only; the symmetry axis r=0 is natural in Elmer's axisymmetric A-phi form
    return m


def transient_sphere():
    """Quenching of a steel sphere (2D axisymmetric, time dependent)."""
    m = B.new_model("2Daxi", ["ht"], "time")
    m.root.label = "sphere_quench"
    _params(m, [("R0", "2[cm]", "Sphere radius"), ("Tinit", "600[degC]", "Initial temperature")])
    g = m.component.child("geometry")
    m.create("geom.circle", g, index=0, r="R0", sector="180", rot="-90", x="0", y="0")
    B.add_material(m, "Structural steel", all_domains=True)
    geo = build_geometry(m, display=False)
    ht = m.component.child("physics.ht")
    ht.child("ht.init").props["T0"] = "Tinit"
    surf = [b for b in geo.numbers["boundary"] if geo.bbox_of["boundary"][b][3] > 1e-6]
    B.add_feature(m, ht, "ht.heatflux", surf, type="convective", h="500[W/(m^2*K)]", Text="20[degC]")
    B.study_steps(m.studies()[0])[0].props["times"] = "range(0,10,300)"
    return m


EXAMPLES = {
    "busbar": ("Busbar - Joule heating (electric currents + heat transfer)", busbar),
    "heat_sink": ("Heat sink - conduction with convective cooling", heat_sink),
    "cantilever": ("Cantilever beam - eigenfrequencies", cantilever_eigen),
    "flow_cylinder": ("Flow past a cylinder - 2D laminar flow", flow_cylinder),
    "capacitor": ("Parallel-plate capacitor - fringing fields (2D electrostatics)", capacitor),
    "coil": ("Coil with iron core - 2D axisymmetric magnetic fields", coil_axi),
    "sphere": ("Sphere quenching - 2D axisymmetric transient heat transfer", transient_sphere),
}
