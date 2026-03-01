# step01_box — Uniform Electrode Baseline

Validates the Elmer FEM static-conduction solver against the known analytic
solution for a rectangular tissue block with uniform electrode coverage.

## Problem

| Parameter | Value |
|-----------|-------|
| Geometry | 4 × 4 × 2 cm tissue box |
| Conductivity σ | 0.2 S/m |
| Top BC | 1 V (uniform electrode) |
| Bottom BC | 0 V (ground) |
| Sides | Insulated (zero flux) |

**Analytic solution:**  V(z) = z / Lz,  |J| = σ · ΔV / Lz = **10 A/m²** (uniform everywhere)

## Files

| File | Purpose |
|------|---------|
| `box.geo` | Gmsh geometry (mesh size lc = 4 mm) |
| `case.sif` | Elmer solver input (StatCurrentSolve) |
| `setup_case.py` | Auto-detects boundary IDs, writes `case.sif` |
| `find_boundaries.py` | Reconstructs `mesh.boundary` from tet mesh |
| `test_step01_baseline.py` | **Automated validation test** |
| `visualize.py` | Generates summary figure |

## Run the pipeline from scratch

```bash
cd step01_box

gmsh -3 box.geo -o box.msh
ElmerGrid 14 2 box.msh -out elmer_mesh
python3 setup_case.py
mkdir -p results
ElmerSolver case.sif
```

## Run the validation test

```bash
cd step01_box
python3 test_step01_baseline.py
```

The test auto-runs the full pipeline if `results/case_t0001.vtu` is missing.

### Expected PASS output

```
============================================================
step01_box  baseline validation
============================================================
  Geometry            Lz = 2.0 cm
  Analytic |J|           = 10.0000 A/m²  (σ·ΔV/Lz)

  mean(|J|)              = 10.000000 A/m²
  std(|J|)               = ...e-...  A/m²
  CV = std/mean          = ...e-...      tol < 1e-02
  rel error vs analytic  = ...e-...      tol < 1e-03

  V(z) R²  (center col)  = 1.0000000   tol > 0.9999
  V(z) slope             = 50.0000 V/m  (analytic 50.0000)
  Φ range                = [0.0000, 1.0000] V

  Flux |J_z| at top      = 10.0000 A/m²
  Flux |J_z| at bottom   = 10.0000 A/m²
  Flux conservation err  = ...e-...      tol < 1e-02

RESULT:  PASS
```

### Pass criteria

| Metric | Tolerance | Physical meaning |
|--------|-----------|-----------------|
| rel error of mean\|J\| vs analytic | < 1e-3 | Solver accuracy |
| CV = std/mean of \|J\| | < 1e-2 | J is spatially uniform |
| R² of V(z) along center column | > 0.9999 | Potential is linear in z |
| Top/bottom flux mismatch | < 1e-2 | Current conservation |

## Generate figures

```bash
cd step01_box
python3 visualize.py
# → results/step01_summary.png
# → results/step01_3d.png
```
