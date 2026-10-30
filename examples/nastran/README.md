# NASTRAN-95 examples

Four small models solved by the NASTRAN-95 backend, the models of the
[NASTRAN-95 tutorial](../../docs/modules/tutorials/pages/nastran.adoc). Each script builds the model through
the Python API, solves it with NASTRAN-95, prints the result next to the closed-form value, saves the `.esm`
project and writes a picture of the result.

| Script | Model | Closed form | NASTRAN-95 |
|--------|-------|-------------|------------|
| `axial_bar_2d.py` | 2D bar, plane stress, 20 MPa on the end | elongation σL/E = 1.000e-4 m, stress 20 MPa | 0.9989e-4 m, 20.00 MPa |
| `cantilever_modes_2d.py` | 2D cantilever, plane stress, 12.5 mm mesh | f1 = 40.77 Hz, f2/f1 = 6.267 | 41.13 Hz, 6.195 |
| `heat_2d.py` | 2D plate, 350 K and 300 K on its ends | linear temperature | within 2e-4 K |
| `axial_bar_3d.py` | 3D bar, total force 10 kN on the end | elongation PL/EA = 1.000e-4 m, stress 20 MPa | 1.014e-4 m, 20.1 MPa |

## Running

NASTRAN-95 must be found: set its folder in Preferences, `NASTRAN_HOME` to a native build, or on Windows
`NASTRAN_WSL_HOME` to the Linux path of a build inside WSL (the folder with `dist/bin/nastran` and `rf/`).
Then, from the repository root:

```bash
python examples/nastran/axial_bar_2d.py
python examples/nastran/cantilever_modes_2d.py
python examples/nastran/heat_2d.py
python examples/nastran/axial_bar_3d.py
```

Each script writes `<name>.esm` (reopen it in Elmer Studio) and `<name>.png` next to itself.

## Notes

- 2D models are per unit depth, as in Elmer; the loads are Studio's Boundary Load types (force per unit area,
  total force, pressure).
- NASTRAN-95's elements are linear: 6-node triangles become 4 CTRIA3 and 10-node tetrahedra 8 CTETRA. They are
  exact for uniform stress and need a finer mesh than Elmer in bending; see the accuracy table in the user guide
  page `docs/modules/user-guide/pages/nastran.adoc`.
