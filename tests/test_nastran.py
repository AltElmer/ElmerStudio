"""NASTRAN-95 backend: verification, closed-form checks, refusals, deck generation, and persistence.

Tests:
1. Discovery and install diagnostics.
2. Model validation and refusal rules (before geometry/meshing).
3. Input deck generation and preview (APP DISP/HEAT, SOL 1/3/9, Bulk Data cards).
4. Output parser for displacements, stresses, eigenvalues, temperatures, and fatal errors.
5. End-to-end static solve (axial tension / cantilever bending) compared with beam theory.
6. End-to-end normal modes (SOL 3) eigenfrequencies.
7. End-to-end steady heat transfer (HEAT 1) temperature distribution.
8. Persistence: project save and reload with NASTRAN solution.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile

import numpy as np
import pytest

from elmerstudio.core import builder as B
from elmerstudio.core import project
from elmerstudio.core.backends import study_backend
from elmerstudio.core.backends.nastran.deck import NastranRefusal, check_model
from elmerstudio.core.backends.nastran.install import find_nastran
from elmerstudio.core.backends.nastran.parser import parse_output
from elmerstudio.core.backends.nastran.runner import NastranStudy, preview_files
from elmerstudio.core.geometry import build_geometry
from elmerstudio.core.model import Model
from elmerstudio.core.results import Solution
from elmerstudio.core.study_runner import StudyRunner


def _run_study(m, st=None, events=None):
    st = st or m.studies()[0]
    st.props["backend"] = "nastran"
    wd = tempfile.mkdtemp(prefix="es_nastran_")
    cb = (lambda k, p: events.append((k, p))) if events is not None else None
    folders = StudyRunner(m, st, wd, None, on_event=cb).run()
    return [Solution(f) for f in folders]


# =========================================================================== 1. Discovery
def test_nastran_install_discovery():
    inst = find_nastran()
    assert inst is not None, "NASTRAN-95 must be found on this machine"
    
    # Verify the executable exists (handle WSL paths correctly)
    if inst.kind == "wsl":
        import subprocess
        wsl = shutil.which("wsl") or "wsl.exe"
        result = subprocess.run(
            [wsl, "-e", "bash", "-c", f"[ -x '{inst.exe}' ] && echo OK"],
            capture_output=True, text=True, timeout=10
        )
        assert "OK" in result.stdout, f"WSL executable {inst.exe} must exist"
        result = subprocess.run(
            [wsl, "-e", "bash", "-c", f"[ -d '{inst.rf_dir}' ] && echo OK"],
            capture_output=True, text=True, timeout=10
        )
        assert "OK" in result.stdout, f"WSL rf_dir {inst.rf_dir} must exist"
    else:
        assert os.path.isfile(inst.exe), f"Executable {inst.exe} must exist"
        assert os.path.isdir(inst.rf_dir), f"Rigid formats {inst.rf_dir} must exist"
    
    desc = inst.describe()
    assert "NASTRAN" in desc and inst.kind in ("windows", "linux", "macos", "wsl")


# =========================================================================== 2. Refusals
def test_refusals_without_geometry():
    # CFD (spf) is refused
    m = B.new_model("2D", ["spf"], "stationary")
    err = check_model(m, m.studies()[0])
    assert err is not None and "not supported" in err and "Elmer" in err

    # Acoustics (acpr) is refused
    m = B.new_model("2D", ["acpr"], "freq")
    err = check_model(m, m.studies()[0])
    assert err is not None and "not supported" in err

    # Axisymmetric components are refused
    m = B.new_model("2Daxi", ["solid"], "stationary")
    err = check_model(m, m.studies()[0])
    assert err is not None and "axisymmetric" in err

    # Multiple physics solved together are refused
    m = B.new_model("3D", ["solid", "ht"], "stationary")
    err = check_model(m, m.studies()[0])
    assert err is not None and "one at a time" in err

    # Supported models pass:
    m = B.new_model("2D", ["solid"], "stationary")
    assert check_model(m, m.studies()[0]) is None

    m = B.new_model("3D", ["solid"], "eigen")
    assert check_model(m, m.studies()[0]) is None

    m = B.new_model("3D", ["ht"], "stationary")
    assert check_model(m, m.studies()[0]) is None


def test_study_backend_selection():
    m = B.new_model("2D", ["solid"], "stationary")
    st = m.studies()[0]
    assert study_backend(st) == "elmer"
    st.props["backend"] = "nastran"
    assert study_backend(st) == "nastran"


# =========================================================================== 3. Deck Preview
def test_deck_preview_generation():
    m = B.new_model("2D", ["solid"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1.0", h="0.1")
    B.add_material(m, "Structural steel", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")
    B.add_feature(m, ph, "solid.fixed", geo.select("boundary", xmax=0))

    st = m.studies()[0]
    st.props["backend"] = "nastran"

    files = preview_files(m, st)
    assert len(files) == 1
    fname, content = files[0]
    assert fname == "case.bdf"
    assert "ID " in content
    assert "APP DISP" in content
    assert "SOL 1" in content
    assert "CEND" in content
    assert "BEGIN BULK" in content
    assert "GRID," in content
    assert "CQUAD4," in content or "CTRIA3," in content or "CTRIM6," in content
    assert "PSHELL," in content or "PTRIM6," in content
    assert "MAT1," in content
    assert "SPC1," in content
    assert "ENDDATA" in content


# =========================================================================== 4. Output Parser
def test_output_parser_extracts_data():
    sample_out = """
 NASTRAN EXECUTIVE CONTROL DECK
 APP DISP
 SOL 1
 CEND
 TITLE = Cantilever
 BEGIN BULK
 ENDDATA
               D I S P L A C E M E N T   V E C T O R
      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3
             1      G      0.000000E+00   0.000000E+00   0.000000E+00   0.000000E+00   0.000000E+00   0.000000E+00
             2      G      1.250000E-04  -3.500000E-03   0.000000E+00   0.000000E+00   0.000000E+00   1.200000E-03
             3      G      2.500000E-04  -7.000000E-03   0.000000E+00   0.000000E+00   0.000000E+00   2.400000E-03

               S T R E S S E S   I N   Q U A D R I L A T E R A L   E L E M E N T S   ( Q U A D 4 )
  ELEMENT-ID    GRID-ID    FIBER-DIST         NORMAL-X       NORMAL-Y          SHEAR-XY        VON MISES
           1          1   -5.000000E-02   1.500000E+07   2.000000E+06   1.000000E+06   1.450000E+07
                      2   -5.000000E-02   1.400000E+07   1.800000E+06   9.000000E+05   1.350000E+07

               R E A L   E I G E N V A L U E S
   ROOT    EXTRACTION    EIGENVALUE          RADIANS             CYCLES            GENERALIZED         GENERALIZED
    NO.       ORDER                                                                   MASS              STIFFNESS
      1         1        1.558231E+05      3.947444E+02      6.282561E+01      1.000000E+00      1.558231E+05
      2         2        6.142315E+06      2.478369E+03      3.944445E+02      1.000000E+00      6.142315E+06
 * * * END OF JOB * * *
"""
    res = parse_output(sample_out)
    assert res.success
    assert len(res.displacements) == 3
    assert res.displacements[2][0] == pytest.approx(1.25e-4)
    assert res.displacements[2][1] == pytest.approx(-3.5e-3)
    assert len(res.stresses) >= 1
    assert res.stresses[1] == pytest.approx(1.45e+7)
    assert len(res.eigenvalues) == 2
    assert res.eigenvalues[0]["mode"] == 1
    assert res.eigenvalues[0]["freq_hz"] == pytest.approx(62.82561)
    assert res.eigenvalues[1]["freq_hz"] == pytest.approx(394.4445)


def test_output_parser_fatal_error():
    fatal_out = """
 *** USER FATAL MESSAGE 9050 (SEKRRS)
     RUN TERMINATED DUE TO EXCESSIVE PIVOT RATIOS IN MATRIX KLL.
     USER ACTION: CHECK CONSTRAINTS.
"""
    res = parse_output(fatal_out)
    assert not res.success
    assert len(res.fatal_messages) > 0
    assert "USER FATAL MESSAGE 9050" in res.fatal_messages[0]


# =========================================================================== 5. End-to-End Statics (Axial Bar)
def test_axial_bar_closed_form(nastran):
    """Axial bar of length L = 1 m, height H = 0.05 m, thickness t = 0.01 m.
    Material: Steel with E = 2.0e11 Pa, nu = 0.3.
    Fixed at x = 0; Tension force P = 10,000 N applied at x = L.
    Cross-section: A = H * t = 0.05 * 0.01 = 0.0005 m^2.
    delta = (P * L) / (E * A) = 10000 * 1.0 / (2.0e11 * 0.0005) = 1.0e-4 m = 0.1 mm.
    Normal stress: sigma = P / A = 10000 / 0.0005 = 2.0e7 Pa = 20 MPa.
    """
    L, H_bar, t_bar = 1.0, 0.05, 0.01
    E_mod, P_force = 2.0e11, 10000.0
    A_cs = H_bar * t_bar
    delta_theory = (P_force * L) / (E_mod * A_cs)
    sigma_theory = P_force / A_cs

    m = B.new_model("2D", ["solid"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(H_bar))
    B.add_material(m, "Structural steel", all_domains=True)

    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.05", hmin="0.02")

    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")

    bnd_left = geo.select("boundary", xmax=0)
    bnd_right = geo.select("boundary", xmin=0.99 * L)
    assert bnd_left and bnd_right

    # Fixed constraint at x = 0
    B.add_feature(m, ph, "solid.fixed", bnd_left)

    # Tension load at x = L
    bload = B.add_feature(m, ph, "solid.bload", bnd_right)
    bload.props["Fx"] = f"{P_force}[N]"
    bload.props["Fy"] = "0[N]"

    # Solve
    sols = _run_study(m)
    assert len(sols) == 1
    sol = sols[0]
    assert sol.meta["status"] == "converged"
    assert sol.meta["backend"] == "nastran"
    assert sol.meta["nastran"]["sol"] == 1

    coords = sol.topo.nodes * sol.scale
    tip_mask = np.isclose(coords[:, 0], L, atol=1e-4)
    ux = sol.names(0)["u"][tip_mask]
    mean_ux = np.mean(ux)
    err_disp = abs(mean_ux - delta_theory) / delta_theory
    print(f"Axial bar elongation: measured {mean_ux:.4e} m, theory {delta_theory:.4e} m, relative error {err_disp:.2%}")
    assert err_disp < 0.05, f"Displacement error {err_disp:.2%} exceeds 5%"

    if "vonmises" in sol.names(0):
        vm = sol.names(0)["vonmises"]
        interior = (coords[:, 0] > 0.2 * L) & (coords[:, 0] < 0.8 * L)
        if np.any(interior):
            vm_int = vm[interior]
            mean_vm = np.mean(vm_int)
            err_stress = abs(mean_vm - sigma_theory) / sigma_theory
            vm_min, vm_max = np.min(vm_int), np.max(vm_int)
            err_min = abs(vm_min - sigma_theory) / sigma_theory
            err_max = abs(vm_max - sigma_theory) / sigma_theory
            print(f"Axial bar stress: measured {mean_vm:.3e} Pa, theory {sigma_theory:.3e} Pa, relative error {err_stress:.2%}")
            print(f"Axial bar stress min/max: {vm_min:.3e}/{vm_max:.3e} Pa, errors {err_min:.2%}/{err_max:.2%}")
            assert err_stress < 0.10, f"Stress error {err_stress:.2%} exceeds 10%"
            assert err_min < 0.10, f"Min stress error {err_min:.2%} exceeds 10%"
            # Max stress can exceed 10% due to load application stress concentration near x=L
            assert err_max < 0.20, f"Max stress error {err_max:.2%} exceeds 20%"


# =========================================================================== 5b. End-to-End 3D Statics (Axial Bar)
def test_axial_bar_3d_closed_form(nastran):
    """3D axial bar of length L = 1 m, cross-section 0.05 x 0.01 m (H x W).
    Material: Steel with E = 2.0e11 Pa, nu = 0.3.
    Fixed at x = 0; Tension force P = 10,000 N applied at x = L.
    Cross-section: A = H * W = 0.05 * 0.01 = 0.0005 m^2.
    delta = (P * L) / (E * A) = 10000 * 1.0 / (2.0e11 * 0.0005) = 1.0e-4 m = 0.1 mm.
    Normal stress: sigma = P / A = 10000 / 0.0005 = 2.0e7 Pa = 20 MPa.
    """
    L, H_bar, W_bar = 1.0, 0.05, 0.01
    E_mod, P_force = 2.0e11, 10000.0
    A_cs = H_bar * W_bar
    delta_theory = (P_force * L) / (E_mod * A_cs)
    sigma_theory = P_force / A_cs

    m = B.new_model("3D", ["solid"], "stationary")
    m.create("geom.block", m.component.child("geometry"), index=0, w=str(L), h=str(H_bar), d=str(W_bar))
    B.add_material(m, "Structural steel", all_domains=True)

    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.1", hmin="0.05")

    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")

    bnd_left = geo.select("boundary", xmax=0)
    bnd_right = geo.select("boundary", xmin=0.99 * L)
    assert bnd_left and bnd_right

    # Fixed constraint at x = 0
    B.add_feature(m, ph, "solid.fixed", bnd_left)

    # Tension load at x = L
    bload = B.add_feature(m, ph, "solid.bload", bnd_right)
    bload.props["Fx"] = f"{P_force}[N]"
    bload.props["Fy"] = "0[N]"
    bload.props["Fz"] = "0[N]"

    # Solve
    sols = _run_study(m)
    assert len(sols) == 1
    sol = sols[0]
    assert sol.meta["status"] == "converged"
    assert sol.meta["backend"] == "nastran"
    assert sol.meta["nastran"]["sol"] == 1

    coords = sol.topo.nodes * sol.scale
    tip_mask = np.isclose(coords[:, 0], L, atol=1e-4)
    ux = sol.names(0)["u"][tip_mask]
    mean_ux = np.mean(ux)
    err_disp = abs(mean_ux - delta_theory) / delta_theory
    print(f"3D axial bar elongation: measured {mean_ux:.4e} m, theory {delta_theory:.4e} m, relative error {err_disp:.2%}")
    assert err_disp < 0.05, f"Displacement error {err_disp:.2%} exceeds 5%"

    if "vonmises" in sol.names(0):
        vm = sol.names(0)["vonmises"]
        interior = (coords[:, 0] > 0.2 * L) & (coords[:, 0] < 0.8 * L)
        if np.any(interior):
            mean_vm = np.mean(vm[interior])
            err_stress = abs(mean_vm - sigma_theory) / sigma_theory
            print(f"3D axial bar stress: measured {mean_vm:.3e} Pa, theory {sigma_theory:.3e} Pa, relative error {err_stress:.2%}")
            assert err_stress < 0.10, f"Stress error {err_stress:.2%} exceeds 10%"


# =========================================================================== 5c. End-to-End 3D Eigenfrequencies (SOL 3)
def test_cantilever_eigenfrequencies_3d(nastran):
    """3D cantilever beam modal analysis (SOL 3 normal modes).
    L = 1.0 m, H = 0.05 m, W = 0.01 m.
    Material: Steel E=2e11 Pa, rho=7850 kg/m^3.
    Euler-Bernoulli f1 = 1.875104^2/(2*pi)*sqrt(E*H^2/(12*rho))/L^2 = 40.77 Hz.
    """
    import math
    L, H, W = 1.0, 0.05, 0.01
    E, rho = 2.0e11, 7850.0
    f1_theory = (1.875104**2 / (2 * math.pi)) * math.sqrt(E * H**2 / (12 * rho)) / L**2

    m = B.new_model("3D", ["solid"], "eigen")
    m.create("geom.block", m.component.child("geometry"), index=0, w=str(L), h=str(H), d=str(W))
    B.add_material(m, "Structural steel", all_domains=True)

    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.1", hmin="0.05")

    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")

    bnd_left = geo.select("boundary", xmax=0)
    assert bnd_left
    B.add_feature(m, ph, "solid.fixed", bnd_left)

    sols = _run_study(m)
    assert len(sols) == 1
    sol = sols[0]
    assert sol.meta["status"] == "converged"
    assert sol.meta["backend"] == "nastran"
    assert sol.meta["nastran"]["sol"] == 3

    freqs = sol.meta["freqs"]
    print(f"3D computed natural frequencies (Hz): {freqs}")
    assert len(freqs) == 6, f"Expected 6 extracted modes, got {len(freqs)}"
    assert all(freqs[i] < freqs[i+1] for i in range(len(freqs)-1)), f"Frequencies not strictly increasing: {freqs}"
    rel_err = abs(freqs[0] - f1_theory) / f1_theory
    print(f"3D first frequency: {freqs[0]:.2f} Hz, theory: {f1_theory:.2f} Hz, rel error: {rel_err:.2%}")
    # 3D thin beam (H=0.05, W=0.01) with coarse tet mesh: allow 30% tolerance
    assert rel_err < 0.30, f"Frequency error {rel_err:.2%} exceeds 30% (f1={freqs[0]:.2f} Hz, theory={f1_theory:.2f} Hz)"
    ratio_theory = (4.694091 / 1.875104) ** 2
    ratio_meas = freqs[1] / freqs[0]
    ratio_err = abs(ratio_meas - ratio_theory) / ratio_theory
    print(f"3D f2/f1 ratio: {ratio_meas:.4f}, theory: {ratio_theory:.4f}, rel error: {ratio_err:.2%}")
    assert ratio_err < 0.15, f"Ratio error {ratio_err:.2%} exceeds 15% (measured={ratio_meas:.4f}, theory={ratio_theory:.4f})"


# =========================================================================== 5d. End-to-End 3D Heat Conduction
def test_thermal_conduction_3d_closed_form(nastran):
    """Steady 3D heat conduction across block of length L = 1.0 m, H = 0.05 m, W = 0.01 m.
    Left face (x=0): T = 350 K.
    Right face (x=L): T = 300 K.
    Closed-form: linear gradient T(x) = T_left + (T_right - T_left) * (x / L).
    """
    L, H, W = 1.0, 0.05, 0.01
    T_left, T_right = 350.0, 300.0

    m = B.new_model("3D", ["ht"], "stationary")
    m.create("geom.block", m.component.child("geometry"), index=0, w=str(L), h=str(H), d=str(W))
    B.add_material(m, "Structural steel", all_domains=True)

    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.1", hmin="0.05")

    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")

    bnd_left = geo.select("boundary", xmax=0)
    bnd_right = geo.select("boundary", xmin=0.99 * L)
    assert bnd_left and bnd_right

    t1 = B.add_feature(m, ph, "ht.temperature", bnd_left)
    t1.props["T0"] = f"{T_left}[K]"

    t2 = B.add_feature(m, ph, "ht.temperature", bnd_right)
    t2.props["T0"] = f"{T_right}[K]"

    sols = _run_study(m)
    assert len(sols) == 1
    sol = sols[0]
    assert sol.meta["status"] == "converged"
    assert sol.meta["backend"] == "nastran"
    assert sol.meta["nastran"]["sol"] == 1

    coords = sol.topo.nodes * sol.scale
    T_field = sol.names(0)["T"]
    T_exact = T_left + (T_right - T_left) * (coords[:, 0] / L)
    max_err = np.max(np.abs(T_field - T_exact))
    print(f"3D heat conduction max temperature error: {max_err:.4e} K")
    assert max_err < 1.0, f"Max error {max_err:.3f} K exceeds 1 K tolerance"


# =========================================================================== 6. End-to-End Eigenfrequencies (SOL 3)
def test_cantilever_eigenfrequencies(nastran):
    """Cantilever beam modal analysis (SOL 3 normal modes).
    L = 1.0 m, H = 0.05 m.
    Material: Steel E=2e11 Pa, rho=7850 kg/m^3.
    Euler-Bernoulli f1 = 1.875104^2/(2*pi)*sqrt(E*H^2/(12*rho))/L^2 = 40.77 Hz.
    """
    import math
    L, H = 1.0, 0.05
    E, rho = 2.0e11, 7850.0
    f1_theory = (1.875104**2 / (2 * math.pi)) * math.sqrt(E * H**2 / (12 * rho)) / L**2

    m = B.new_model("2D", ["solid"], "eigen")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(H))
    B.add_material(m, "Structural steel", all_domains=True)

    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.05", hmin="0.02")

    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")

    bnd_left = geo.select("boundary", xmax=0)
    assert bnd_left
    B.add_feature(m, ph, "solid.fixed", bnd_left)

    sols = _run_study(m)
    assert len(sols) == 1
    sol = sols[0]
    assert sol.meta["status"] == "converged"
    assert sol.meta["backend"] == "nastran"
    assert sol.meta["nastran"]["sol"] == 3

    freqs = sol.meta["freqs"]
    print(f"Computed natural frequencies (Hz): {freqs}")
    # Exactly neig=6 extracted modes
    assert len(freqs) == 6, f"Expected 6 extracted modes, got {len(freqs)}"
    # Frequencies must be strictly increasing
    assert all(freqs[i] < freqs[i+1] for i in range(len(freqs)-1)), f"Frequencies not strictly increasing: {freqs}"
    # First bending frequency (Euler-Bernoulli): 40.77 Hz
    # Coarse plane-stress mesh: allow 15% relative tolerance
    rel_err = abs(freqs[0] - f1_theory) / f1_theory
    print(f"First frequency: {freqs[0]:.2f} Hz, theory: {f1_theory:.2f} Hz, rel error: {rel_err:.2%}")
    assert rel_err < 0.15, f"Frequency error {rel_err:.2%} exceeds 15% (f1={freqs[0]:.2f} Hz, theory={f1_theory:.2f} Hz)"
    # Second-to-first frequency ratio: Euler-Bernoulli (4.694091/1.875104)^2 = 6.267
    ratio_theory = (4.694091 / 1.875104) ** 2
    ratio_meas = freqs[1] / freqs[0]
    ratio_err = abs(ratio_meas - ratio_theory) / ratio_theory
    print(f"f2/f1 ratio: {ratio_meas:.4f}, theory: {ratio_theory:.4f}, rel error: {ratio_err:.2%}")
    assert ratio_err < 0.15, f"Ratio error {ratio_err:.2%} exceeds 15% (measured={ratio_meas:.4f}, theory={ratio_theory:.4f})"


# =========================================================================== 7. End-to-End Heat Transfer (HEAT 1)
def test_thermal_conduction_closed_form(nastran):
    """Steady 1D/2D heat conduction across plate of length L = 1.0 m, H = 0.2 m.
    Left boundary: T = 350 K.
    Right boundary: T = 300 K.
    Closed-form solution: linear gradient T(x) = T_left + (T_right - T_left) * (x / L).
    """
    L, H_plate = 1.0, 0.2
    T_left, T_right = 350.0, 300.0

    m = B.new_model("2D", ["ht"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(H_plate))
    B.add_material(m, "Structural steel", all_domains=True)

    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.1", hmin="0.05")

    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.ht")

    bnd_left = geo.select("boundary", xmax=0)
    bnd_right = geo.select("boundary", xmin=0.99 * L)
    assert bnd_left and bnd_right

    t1 = B.add_feature(m, ph, "ht.temperature", bnd_left)
    t1.props["T0"] = f"{T_left}[K]"

    t2 = B.add_feature(m, ph, "ht.temperature", bnd_right)
    t2.props["T0"] = f"{T_right}[K]"

    sols = _run_study(m)
    assert len(sols) == 1
    sol = sols[0]
    assert sol.meta["status"] == "converged"
    assert sol.meta["backend"] == "nastran"

    coords = sol.topo.nodes * sol.scale
    T_field = sol.names(0)["T"]
    T_exact = T_left + (T_right - T_left) * (coords[:, 0] / L)
    max_err = np.max(np.abs(T_field - T_exact))
    print(f"Heat conduction max temperature error: {max_err:.4e} K")
    assert max_err < 1.0, f"Max error {max_err:.3f} K exceeds 1 K tolerance"


# =========================================================================== 8. Persistence
def test_saved_nastran_solution_reopens_elsewhere(nastran):
    m = B.new_model("2D", ["solid"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="0.5", h="0.05")
    B.add_material(m, "Structural steel", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")
    B.add_feature(m, ph, "solid.fixed", geo.select("boundary", xmax=0))

    st = m.studies()[0]
    st.props["backend"] = "nastran"
    wd = project.new_workdir()
    folders = StudyRunner(m, st, wd, None).run()
    assert folders
    folder = folders[0]

    out_esm = os.path.join(tempfile.mkdtemp(prefix="es_saved_nas_"), "model.esm")
    project.save(m, out_esm, wd)
    project.discard_workdir(wd)
    shutil.rmtree(wd, ignore_errors=True)

    m2, wd2 = project.load(out_esm)
    sol2 = Solution(os.path.join(wd2, st.tag, "sol_0_0"))
    assert sol2.meta["backend"] == "nastran"
    assert sol2.meta["status"] == "converged"
