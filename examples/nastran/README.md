# NASTRAN-95 Backend MWEs

These examples demonstrate the NASTRAN-95 solver backend in Elmer Studio.
Each script builds a model through the Studio API, solves with NASTRAN through Studio,
prints `measured … theory … error …%`, saves the `.esm` project, and writes an
offscreen PNG of the result field.

## Results

| Example | Physics | Dimension | Measured | Theory | Error |
|---------|---------|-----------|----------|--------|-------|
| `axial_bar_2d.py` | Solid statics | 2D | elongation 1.0019e-04 m | 1.0000e-04 m | 0.19% |
| `axial_bar_2d.py` | Solid statics | 2D | stress 2.012e+07 Pa | 2.000e+07 Pa | 0.61% |
| `cantilever_modes_2d.py` | Eigenfrequency | 2D | f1 44.64 Hz | 40.77 Hz | 9.49% |
| `cantilever_modes_2d.py` | Eigenfrequency | 2D | f2/f1 6.1938 | 6.2669 | 1.17% |
| `heat_2d.py` | Heat conduction | 2D | max error 2.03e-04 K | 0 K | <1 K |
| `axial_bar_3d.py` | Solid statics | 3D | elongation 1.0149e-04 m | 1.0000e-04 m | 1.49% |
| `axial_bar_3d.py` | Solid statics | 3D | stress 1.940e+07 Pa | 2.000e+07 Pa | 3.02% |

*Values above are from the last successful run on this machine. Run the scripts to reproduce.*

## Running

```bash
# From the ElmerStudio root directory
cd <path to ElmerStudio repo>

# Set NASTRAN environment (choose one)
export NASTRAN_WSL_HOME=/home/<user>/NASTRAN-95
# or
export NASTRAN_HOME=/path/to/native/nastran95

# Run examples with the venv python
<venv python> examples/nastran/axial_bar_2d.py
<venv python> examples/nastran/cantilever_modes_2d.py
<venv python> examples/nastran/heat_2d.py
<venv python> examples/nastran/axial_bar_3d.py
```

## Outputs

Each script produces:
- `<name>.esm` — saved Elmer Studio project (reloadable)
- `<name>.png` — offscreen render of the result field

## Notes

- The 2D axial bar uses CTRIM6 quadratic triangles (structural) with PSHELL/PTRIM6.
- The 2D eigenfrequency uses SOL 3 (normal modes) with EIGR method GIV + ND.
- The 2D heat conduction uses HEAT 1 with THERMAL = ALL output request.
- The 3D axial bar uses 10-node tet → 8 CTETRA split with PS=456 on GRID cards.
- All examples use WSL NASTRAN-95 by default; set `NASTRAN_HOME` for native Windows.