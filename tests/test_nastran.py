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
def test_nastran_install_discovery(nastran):
    inst = nastran                      # the fixture skips where NASTRAN-95 is not installed (CI runners)

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
    assert err is not None and "Frequency Domain is not a Stationary or Eigenfrequency step" in err

    # Axisymmetric components are refused
    m = B.new_model("2Daxi", ["solid"], "stationary")
    err = check_model(m, m.studies()[0])
    assert err is not None and "axisymmetric" in err

    # Multiple physics solved together are refused
    m = B.new_model("3D", ["solid", "ht"], "stationary")
    err = check_model(m, m.studies()[0])
    assert err is not None and "one at a time" in err

    # Time dependent studies have no NASTRAN deck yet
    m = B.new_model("2D", ["solid"], "time")
    err = check_model(m, m.studies()[0])
    assert err is not None and "Stationary or Eigenfrequency" in err

    # A heat flux boundary is not mapped (it would be dropped)
    m = B.new_model("2D", ["ht"], "stationary")
    m.create("ht.heatflux", m.component.child("physics.ht"))
    err = check_model(m, m.studies()[0])
    assert err is not None and "Heat Flux" in err

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
    assert "CTRIA3," in content and "PSHELL," in content      # each 6-node triangle as 4 CTRIA3
    assert content.count("CTRIA3,") % 4 == 0
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

                     S T R E S S E S   I N   G E N E R A L   T R I A N G U L A R   E L E M E N T S      ( C T R I A 3 )
  ELEMENT      FIBRE               STRESSES IN STRESS COORD. SYSTEM             PRINCIPAL STRESSES (ZERO SHEAR)            MAX
    ID.       DISTANCE           NORMAL-X       NORMAL-Y      SHEAR-XY       ANGLE         MAJOR           MINOR          SHEAR
0       1   -5.000000E-03      9.999999E+06   9.999919E+06   9.999978E+06    44.9999    1.999994E+07   -1.900000E+01   9.999978E+06
             5.000000E-03      9.999999E+06   9.999919E+06   9.999978E+06    44.9999    1.999994E+07   -1.900000E+01   9.999978E+06
0       4   -5.000000E-03      1.999997E+07   1.800000E+01  -1.567129E+01    -0.0000    1.999997E+07    1.800000E+01   9.999975E+06

               R E A L   E I G E N V A L U E S
   ROOT    EXTRACTION    EIGENVALUE          RADIANS             CYCLES            GENERALIZED         GENERALIZED
    NO.       ORDER                                                                   MASS              STIFFNESS
      1         1        1.558231E+05      3.947444E+02      6.282561E+01      1.000000E+00      1.558231E+05
      2         2        6.142315E+06      2.478369E+03      3.944445E+02      1.000000E+00      6.142315E+06
1     Study 1 - Step 1: Eigenfrequency                                      /    95 SUN SOLARIS NASTRAN    / OCT 30, ** / PAGE     8
0
      EIGENVALUE =  0.786656E+05    (CYCLIC FREQUENCY =  4.463882E+01 HZ)

                                         R E A L   E I G E N V E C T O R   N O .          1

      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3
             1      G      0.0            0.0            0.0            0.0            0.0            0.0
             2      G     -3.461398E-02  -1.007968E+00   0.0            0.0            0.0            0.0
1     Study 1 - Step 1: Eigenfrequency                                      /    95 SUN SOLARIS NASTRAN    / OCT 30, ** / PAGE     9
      EIGENVALUE =  0.786656E+05    (CYCLIC FREQUENCY =  4.463882E+01 HZ)

                                         R E A L   E I G E N V E C T O R   N O .          1

      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3
            47      G      3.444267E-02  -8.004698E-01   0.0            0.0            0.0            0.0
      EIGENVALUE =  0.213125E+09    (CYCLIC FREQUENCY =  2.323471E+03 HZ)

                                         R E A L   E I G E N V E C T O R   N O .          2

      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3
             2      G      1.079015E-01  -1.303593E-01   0.0            0.0            0.0            0.0
 * * * END OF JOB * * *
"""
    res = parse_output(sample_out)
    assert res.success
    assert len(res.displacements) == 3
    assert res.displacements[2][0] == pytest.approx(1.25e-4)
    assert res.displacements[2][1] == pytest.approx(-3.5e-3)
    # CTRIA3: von Mises from the principal stresses (the element's own axes are rotated in element 1)
    assert sorted(res.stresses) == [1, 4]
    assert res.stresses[1] == pytest.approx(1.999994e7, rel=1e-5)
    assert res.principal[4] == pytest.approx((1.999997e7, 18.0))
    # eigenvectors: a mode's table continues over pages; each mode is kept apart
    assert sorted(res.eigenvectors) == [1, 2]
    assert sorted(res.eigenvectors[1]) == [1, 2, 47]
    assert res.eigenvectors[1][2] == pytest.approx([-3.461398e-2, -1.007968, 0.0])
    assert res.eigenvectors[2][2] == pytest.approx([1.079015e-1, -1.303593e-1, 0.0])
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
    # a scratch file overflow ends the run without a fatal message (real output of a 3D eigen run)
    io_out = """
 I/O SUBSYSTEM ERROR NUMBER       105

 I/O ERROR #     0 ON FILE       67 NAME=KELM     UNIT=  48
                                        * * * END OF JOB * * *
"""
    res = parse_output(io_out)
    assert not res.success and "error 105 on file KELM" in res.fatal_messages[0]


# =========================================================================== 5. End-to-End Statics (Axial Bar)
def test_axial_bar_closed_form(nastran):
    """Axial bar of length L = 1 m, height H = 0.05 m, in plane stress (2D models are per unit depth).
    Material: Steel with E = 2.0e11 Pa, nu = 0.3.
    Fixed at x = 0; a boundary load of sigma = 2.0e7 Pa (force per unit area) pulls the end x = L.
    delta = sigma * L / E = 1.0e-4 m = 0.1 mm; normal stress sigma = 20 MPa.
    """
    L, H_bar = 1.0, 0.05
    E_mod = 2.0e11
    sigma_theory = 2.0e7
    delta_theory = sigma_theory * L / E_mod

    m = B.new_model("2D", ["solid"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w=str(L), h=str(H_bar))
    B.add_material(m, "Structural steel", all_domains=True)

    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.05", hmin="0.02")

    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")
    ph.props["plane"] = "stress"

    bnd_left = geo.select("boundary", xmax=0)
    bnd_right = geo.select("boundary", xmin=0.99 * L)
    assert bnd_left and bnd_right

    # Fixed constraint at x = 0
    B.add_feature(m, ph, "solid.fixed", bnd_left)

    # Tension load at x = L
    bload = B.add_feature(m, ph, "solid.bload", bnd_right)
    bload.props.update({"type": "force", "Fx": f"{sigma_theory}[N/m^2]", "Fy": "0[N/m^2]"})

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
    assert err_disp < 0.01, f"Displacement error {err_disp:.2%} exceeds 1%"

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
            assert err_stress < 0.01, f"Stress error {err_stress:.2%} exceeds 1%"
            assert err_min < 0.01 and err_max < 0.01, f"Stress not uniform: {vm_min:.4e} to {vm_max:.4e} Pa"


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

    # Tension load at x = L, given as the total force on the end face
    bload = B.add_feature(m, ph, "solid.bload", bnd_right)
    bload.props.update({"type": "total", "Ftx": f"{P_force}[N]", "Fty": "0[N]", "Ftz": "0[N]"})

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
    L = 1.0 m, square section H = W = 0.05 m, so the first two modes are the same bending pair.
    Material: Steel E=2e11 Pa, rho=7850 kg/m^3.
    Euler-Bernoulli f1 = f2 = 1.875104^2/(2*pi)*sqrt(E*H^2/(12*rho))/L^2 = 40.77 Hz.
    The split linear tetrahedra are stiff in bending: two 25 mm elements through the depth give about 11-15 % high.
    """
    import math
    L, H, W = 1.0, 0.05, 0.05
    E, rho = 2.0e11, 7850.0
    f1_theory = (1.875104**2 / (2 * math.pi)) * math.sqrt(E * H**2 / (12 * rho)) / L**2

    m = B.new_model("3D", ["solid"], "eigen")
    m.create("geom.block", m.component.child("geometry"), index=0, w=str(L), h=str(H), d=str(W))
    B.add_material(m, "Structural steel", all_domains=True)

    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.025", hmin="0.01")

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
    assert all(freqs[i] <= freqs[i+1] for i in range(len(freqs)-1)), f"Frequencies not increasing: {freqs}"
    for k in (0, 1):
        rel_err = (freqs[k] - f1_theory) / f1_theory
        print(f"3D bending frequency {k + 1}: {freqs[k]:.2f} Hz, theory: {f1_theory:.2f} Hz, rel error: {rel_err:.2%}")
        assert 0 < rel_err < 0.20, f"f{k + 1} = {freqs[k]:.2f} Hz is not within 0-20 % above {f1_theory:.2f} Hz"
    # the two bending directions of the square section
    assert abs(freqs[1] - freqs[0]) / freqs[0] < 0.10
    # second bending pair: (4.694091/1.875104)^2 = 6.267 times the first, beyond the pair
    assert freqs[2] / freqs[0] > 5.0
    v1, w1 = sol.names(0)["v"], sol.names(0)["w"]
    assert max(np.abs(v1).max(), np.abs(w1).max()) > 0          # mode shapes are there


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
    m.create("mesh.size", mn, mode="custom", hmax="0.0125", hmin="0.005")

    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")
    ph.props["plane"] = "stress"        # beam theory: E, not the plane strain modulus

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
    # First bending frequency (Euler-Bernoulli): 40.77 Hz; linear triangles converge from above
    rel_err = abs(freqs[0] - f1_theory) / f1_theory
    print(f"First frequency: {freqs[0]:.2f} Hz, theory: {f1_theory:.2f} Hz, rel error: {rel_err:.2%}")
    assert rel_err < 0.05, f"Frequency error {rel_err:.2%} exceeds 5% (f1={freqs[0]:.2f} Hz, theory={f1_theory:.2f} Hz)"
    # Second-to-first frequency ratio: Euler-Bernoulli (4.694091/1.875104)^2 = 6.267 (shear lowers it slightly)
    ratio_theory = (4.694091 / 1.875104) ** 2
    ratio_meas = freqs[1] / freqs[0]
    ratio_err = abs(ratio_meas - ratio_theory) / ratio_theory
    print(f"f2/f1 ratio: {ratio_meas:.4f}, theory: {ratio_theory:.4f}, rel error: {ratio_err:.2%}")
    assert ratio_err < 0.05, f"Ratio error {ratio_err:.2%} exceeds 5% (measured={ratio_meas:.4f}, theory={ratio_theory:.4f})"
    # Mode shapes: the first mode bends (transverse v dominates), and each mode has its own shape
    v1, u1 = sol.names(0)["v"], sol.names(0)["u"]
    v2 = sol.names(1)["v"]
    assert np.abs(v1).max() > 10 * np.abs(u1).max() > 0
    coords = sol.topo.nodes * sol.scale
    tip = np.isclose(coords[:, 0], L, atol=1e-6)
    assert np.all(np.abs(v1[tip]) > 0.9 * np.abs(v1).max())       # largest at the free end
    assert abs(np.corrcoef(v1, v2)[0, 1]) < 0.9


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


# =========================================================================== 7b. Same model, both backends
def _both(m, elmer):
    out = []
    for backend in ("elmer", "nastran"):
        st = m.studies()[0]
        st.props["backend"] = backend
        folder = StudyRunner(m, st, tempfile.mkdtemp(prefix="es_both_"), elmer).run()[0]
        out.append(Solution(folder))
    return out


def test_same_model_as_elmer(nastran, elmer):
    """NASTRAN-95 solves the model Elmer solves: Studio's load types, 2D plane strain per unit depth, a pressure
    along the inward normal, and element stresses mapped back to the right nodes."""
    # 2D bar in the default plane strain, a total force of 10 kN (per metre of depth) on the end
    m = B.new_model("2D", ["solid"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1.0", h="0.05")
    B.add_material(m, "Structural steel", all_domains=True)
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")
    B.add_feature(m, ph, "solid.fixed", geo.select("boundary", xmax=0))
    bl = B.add_feature(m, ph, "solid.bload", geo.select("boundary", xmin=0.99))
    bl.props.update({"type": "total", "Ftx": "1e4[N]", "Fty": "0[N]"})
    se, sn = _both(m, elmer)
    c = se.topo.nodes * se.scale
    tip, mid = np.isclose(c[:, 0], 1.0, atol=1e-6), (c[:, 0] > 0.2) & (c[:, 0] < 0.8)
    for name, mask, tol in (("u", tip, 0.005), ("solid.mises", mid, 0.005)):
        e, n = np.mean(np.asarray(se.evaluate(name, 0))[mask]), np.mean(np.asarray(sn.evaluate(name, 0))[mask])
        print(f"plane strain bar {name}: elmer {e:.5e}, nastran {n:.5e}")
        assert abs(n - e) / abs(e) < tol, (name, e, n)

    # 2D cantilever in plane stress under a pressure on its top face, 4 elements through the depth
    m = B.new_model("2D", ["solid"], "stationary")
    m.create("geom.rectangle", m.component.child("geometry"), index=0, w="1.0", h="0.05")
    B.add_material(m, "Structural steel", all_domains=True)
    mn = m.component.child("mesh")
    mn.props["sequence"] = "user"
    m.create("mesh.size", mn, mode="custom", hmax="0.0125", hmin="0.005")
    geo = build_geometry(m, display=False)
    ph = m.component.child("physics.solid")
    ph.props["plane"] = "stress"
    B.add_feature(m, ph, "solid.fixed", geo.select("boundary", xmax=0))
    bl = B.add_feature(m, ph, "solid.bload", geo.select("boundary", ymin=0.049))
    bl.props.update({"type": "pressure", "p": "1e5[Pa]"})
    se, sn = _both(m, elmer)
    c = se.topo.nodes * se.scale
    tip = np.isclose(c[:, 0], 1.0, atol=1e-6)
    ve, vn = np.mean(np.asarray(se.evaluate("v", 0))[tip]), np.mean(np.asarray(sn.evaluate("v", 0))[tip])
    print(f"cantilever tip v: elmer {ve:.5e}, nastran {vn:.5e} (beam theory -6.0e-3)")
    assert ve < 0 and vn < 0                                   # the pressure pushes down
    assert abs(vn - ve) / abs(ve) < 0.03                       # linear triangles, slightly stiffer
    me, mn_ = np.asarray(se.evaluate("solid.mises", 0)), np.asarray(sn.evaluate("solid.mises", 0))
    assert np.corrcoef(me, mn_)[0, 1] > 0.99                   # the same stress field, node by node


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
