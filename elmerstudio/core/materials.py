"""Material properties, built-in material library and material nodes.

Library values are room-temperature (20 degC) handbook values; they are constants
unless an expression in T is given. Users can edit every value in the Material
Contents table or add their own materials.
"""
from __future__ import annotations

from .model import NodeType, Prop, register

# key: (label, symbol, unit, Elmer keyword, property group)
MATPROPS = {
    "rho": ("Density", "ρ", "kg/m^3", "Density", "Basic"),
    "k": ("Thermal conductivity", "k", "W/(m*K)", "Heat Conductivity", "Basic"),
    "Cp": ("Heat capacity at constant pressure", "Cₚ", "J/(kg*K)", "Heat Capacity", "Basic"),
    "E": ("Young's modulus", "E", "Pa", "Youngs Modulus", "Young's modulus and Poisson's ratio"),
    "nu": ("Poisson's ratio", "ν", "1", "Poisson Ratio", "Young's modulus and Poisson's ratio"),
    "alpha": ("Coefficient of thermal expansion", "α", "1/K", "Heat Expansion Coefficient", "Basic"),
    "sigma": ("Electrical conductivity", "σ", "S/m", "Electric Conductivity", "Basic"),
    "epsr": ("Relative permittivity", "εᵣ", "1", "Relative Permittivity", "Basic"),
    "mur": ("Relative permeability", "μᵣ", "1", "Relative Permeability", "Basic"),
    "mu": ("Dynamic viscosity", "μ", "Pa*s", "Viscosity", "Basic"),
    "c": ("Speed of sound", "c", "m/s", "Sound Speed", "Basic"),
    "eps": ("Surface emissivity", "ε", "1", "Emissivity", "Basic"),
}

# name: (category, values, rgb color, note)
LIBRARY = {
    "Air": ("Built-in", {"rho": "1.204", "k": "0.0257", "Cp": "1005", "mu": "1.814e-5", "c": "343.2",
                         "epsr": "1", "mur": "1", "sigma": "0", "alpha": "3.43e-3", "eps": "0"},
            (200, 225, 255), "Dry air at 20 degC and 1 atm (constant properties)."),
    "Water, liquid": ("Built-in", {"rho": "998.2", "k": "0.598", "Cp": "4182", "mu": "1.002e-3", "c": "1481",
                                   "epsr": "80.1", "mur": "1", "sigma": "5.5e-6", "alpha": "2.07e-4", "eps": "0.96"},
                      (90, 150, 230), "Pure water at 20 degC."),
    "Copper": ("Built-in", {"rho": "8960", "k": "400", "Cp": "385", "E": "110e9", "nu": "0.35", "alpha": "17e-6",
                            "sigma": "5.998e7", "epsr": "1", "mur": "1", "eps": "0.5"},
               (200, 110, 60), "Annealed copper."),
    "Aluminum": ("Built-in", {"rho": "2700", "k": "238", "Cp": "900", "E": "70e9", "nu": "0.33", "alpha": "23e-6",
                              "sigma": "3.774e7", "epsr": "1", "mur": "1", "eps": "0.1"},
                 (190, 195, 205), "Pure aluminum."),
    "Structural steel": ("Built-in", {"rho": "7850", "k": "44.5", "Cp": "475", "E": "200e9", "nu": "0.3",
                                      "alpha": "12.3e-6", "sigma": "4.032e6", "epsr": "1", "mur": "1", "eps": "0.28"},
                         (150, 155, 165), "Generic structural (carbon) steel."),
    "Stainless steel 304": ("Metals", {"rho": "7930", "k": "16.2", "Cp": "500", "E": "193e9", "nu": "0.29",
                                       "alpha": "17.3e-6", "sigma": "1.389e6", "epsr": "1", "mur": "1.02", "eps": "0.3"},
                            (170, 172, 180), "Austenitic stainless steel AISI 304."),
    "Iron (soft magnetic)": ("Metals", {"rho": "7870", "k": "80.2", "Cp": "449", "E": "200e9", "nu": "0.29",
                                        "alpha": "11.8e-6", "sigma": "1.12e7", "epsr": "1", "mur": "4000", "eps": "0.3"},
                             (110, 115, 125), "Soft iron with constant relative permeability (linear)."),
    "Titanium Ti-6Al-4V": ("Metals", {"rho": "4430", "k": "6.7", "Cp": "526", "E": "113.8e9", "nu": "0.342",
                                      "alpha": "8.6e-6", "sigma": "5.8e5", "epsr": "1", "mur": "1", "eps": "0.3"},
                           (175, 170, 160), "Titanium alloy, grade 5."),
    "Brass": ("Metals", {"rho": "8500", "k": "120", "Cp": "380", "E": "100e9", "nu": "0.33", "alpha": "19e-6",
                         "sigma": "1.5e7", "epsr": "1", "mur": "1", "eps": "0.3"},
              (205, 170, 80), "Yellow brass."),
    "Silver": ("Metals", {"rho": "10490", "k": "429", "Cp": "235", "E": "83e9", "nu": "0.37", "alpha": "18.9e-6",
                          "sigma": "6.3e7", "epsr": "1", "mur": "1", "eps": "0.03"},
               (215, 215, 220), "Pure silver."),
    "Gold": ("Metals", {"rho": "19300", "k": "317", "Cp": "129", "E": "79e9", "nu": "0.42", "alpha": "14.2e-6",
                        "sigma": "4.52e7", "epsr": "1", "mur": "1", "eps": "0.03"},
             (230, 190, 60), "Pure gold."),
    "Silicon": ("Semiconductors", {"rho": "2329", "k": "130", "Cp": "700", "E": "170e9", "nu": "0.28",
                                   "alpha": "2.6e-6", "sigma": "4.35e-4", "epsr": "11.7", "mur": "1", "eps": "0.6"},
                (100, 100, 130), "Single-crystal silicon, isotropic approximation, intrinsic conductivity."),
    "Silica glass": ("Built-in", {"rho": "2203", "k": "1.38", "Cp": "703", "E": "73.1e9", "nu": "0.17",
                                  "alpha": "0.55e-6", "sigma": "1e-14", "epsr": "3.75", "mur": "1", "c": "5900", "eps": "0.9"},
                     (210, 235, 240), "Fused silica (quartz glass)."),
    "FR4 (Circuit Board)": ("Built-in", {"rho": "1900", "k": "0.3", "Cp": "1369", "E": "22e9", "nu": "0.15",
                                         "alpha": "18e-6", "sigma": "4e-3", "epsr": "4.5", "mur": "1", "eps": "0.9"},
                            (80, 140, 60), "Glass-reinforced epoxy laminate (in-plane properties)."),
    "Concrete": ("Built-in", {"rho": "2300", "k": "1.8", "Cp": "880", "E": "25e9", "nu": "0.2", "alpha": "10e-6",
                              "eps": "0.9"},
                 (180, 180, 170), "Normal-weight concrete."),
    "Acrylic plastic": ("Polymers", {"rho": "1190", "k": "0.19", "Cp": "1420", "E": "3.2e9", "nu": "0.35",
                                     "alpha": "70e-6", "sigma": "1e-14", "epsr": "3.0", "mur": "1", "eps": "0.9"},
                        (225, 235, 245), "PMMA."),
    "Nylon": ("Polymers", {"rho": "1150", "k": "0.26", "Cp": "1700", "E": "2e9", "nu": "0.4", "alpha": "80e-6",
                           "sigma": "1e-12", "epsr": "4", "mur": "1", "eps": "0.9"},
              (240, 240, 225), "Polyamide 6,6 (dry)."),
    "Polyethylene (HDPE)": ("Polymers", {"rho": "950", "k": "0.48", "Cp": "1900", "E": "1.1e9", "nu": "0.42",
                                         "alpha": "150e-6", "sigma": "1e-15", "epsr": "2.3", "mur": "1", "eps": "0.9"},
                            (235, 235, 235), "High-density polyethylene."),
    "Engine oil": ("Fluids and Gases", {"rho": "888", "k": "0.145", "Cp": "1880", "mu": "0.799", "c": "1450",
                                        "epsr": "2.2", "mur": "1", "sigma": "1e-12", "alpha": "7e-4"},
                   (190, 150, 40), "Unused SAE 30 engine oil at 20 degC."),
    "Glycerol": ("Fluids and Gases", {"rho": "1261", "k": "0.286", "Cp": "2427", "mu": "1.412", "c": "1904",
                                      "epsr": "42.5", "mur": "1", "sigma": "6.4e-6", "alpha": "5e-4"},
                 (230, 230, 200), "Glycerol at 20 degC."),
}

register(NodeType("materials", "Materials", "materials", "materials", label="Materials", numbered=False,
                  deletable=False, can_disable=False, children=["material"]))

register(NodeType("material", "Material", "mat", "material", selection="domain", props=[
    Prop("values", "", "matprops", {}, section="Material Contents"),
    Prop("libname", "Library material:", "info", "", section="Material Properties"),
    Prop("color", "Color:", "color", [190, 190, 190], section="Appearance"),
]))


def library_categories():
    cats: dict[str, list[str]] = {}
    for name, (cat, *_rest) in LIBRARY.items():
        cats.setdefault(cat, []).append(name)
    return cats


def make_material(model, parent, name: str, domains=None, all_domains=False):
    cat, vals, color, note = LIBRARY[name]
    n = model.create("material", parent, label=name)
    n.props["values"] = dict(vals)
    n.props["libname"] = name
    n.props["color"] = list(color)
    n.meta["note"] = note
    if domains is not None:
        n.selection.entities = list(domains)
    n.selection.all = all_domains
    return n
