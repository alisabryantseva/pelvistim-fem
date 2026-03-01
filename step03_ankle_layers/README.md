# step03_ankle_layers — Layered Ankle Slab Model

## What this model is

A 3-layer finite-element slab representing an ankle cross-section (skin / subcutaneous fat / muscle), with two bipolar surface electrodes on the top face:

| Electrode | Position | Boundary condition |
|-----------|----------|--------------------|
| Medial (active) | `x = medial_offset` | Potential = 1 V |
| Lateral (return) | `x = Lx − lateral_offset` | Potential = 0 V |

All other surfaces are insulated (zero normal current flux — the natural Neumann BC).

The solver computes the electrostatic potential φ and the current density vector **J** = σ∇φ in each tissue layer. Key outputs:

- **peak |J| at skin surface** — comfort proxy (higher = more sensation/discomfort risk)
- **mean |J| in ROI** — efficacy proxy (sphere at ~10 mm depth under active electrode, approximating tibial nerve location)
- **tradeoff = ROI |J| / peak skin |J|** — how much deep stimulation you get per unit surface discomfort

## What this model is NOT

- **Not anatomically accurate** — it is a rectangular slab, not a real ankle geometry
- **Not calibrated** — all conductivity values are PLACEHOLDERS (see below)
- **Not validated** — outputs are order-of-magnitude estimates only

## Directory layout

```
step03_ankle_layers/
├── params.yaml              # all parameters (geometry, layers, conductivities, electrodes)
├── run_layered_sweep.py     # mesh → solve → extract metrics for every (t_fat, r) pair
├── plot_layered_results.py  # generate figures from results/summary.json + VTU files
├── smoke_test.py            # quick pipeline validation (1 coarse case, ~2–5 min)
└── results/                 # auto-created by run_layered_sweep.py
    ├── summary.csv
    ├── summary.json
    ├── J_surface_maps.png
    ├── summary_metrics.png
    ├── representative_3d.png
    └── tfat<N>um_r<N>um/    # one sub-dir per case
        ├── mesh.msh
        ├── elmer_mesh/
        ├── case.sif
        └── results/case_t0001.vtu
```

## Quick start

All commands run from `step03_ankle_layers/`.

### 1. Smoke test — verifies the pipeline in ~2–5 min

```bash
python3 smoke_test.py
```

Runs one coarse case, then checks:
- VTU file created
- Potential in [0, 1] V, no NaN/Inf
- Current density finite
- Current conservation error < 5%
- ROI mean |J| is positive

### 2. Full parameter sweep — 9 cases (3 fat thicknesses × 3 electrode radii)

```bash
python3 run_layered_sweep.py
```

Typical runtime: 15–40 min on a laptop depending on mesh size and solver.
Progress is printed for each case.

### 3. Generate figures

```bash
python3 plot_layered_results.py
```

Requires `results/summary.json` from step 2 (or the smoke test).

Outputs:
- `results/J_surface_maps.png` — heatmaps of |J| on the skin surface
- `results/summary_metrics.png` — peak J, ROI J, and tradeoff vs electrode area
- `results/representative_3d.png` — 3D render of one case clipped at y = Ly/2

## Parameters (`params.yaml`)

### Geometry

| Key | Value | Meaning |
|-----|-------|---------|
| `Lx` | 0.12 m | Medial–lateral width (12 cm) |
| `Ly` | 0.09 m | Anterior–posterior depth (9 cm) |
| `Lz` | 0.040 m | Total slab depth (4 cm) |

### Layers

Layer order from bottom (z=0) to top (z=Lz): **muscle → fat → skin**

| Key | Default | Note |
|-----|---------|------|
| `t_skin` | 1.5 mm | Fixed across sweep |
| `t_fat` | 5 mm | Default (used in smoke test) |
| `t_fat_sweep` | [3, 5, 8] mm | Fat thicknesses in full sweep |
| t_muscle | computed | `Lz − t_skin − t_fat` |

### Conductivities — ALL PLACEHOLDERS

```yaml
sigma_skin:    0.0002  # S/m  — literature: ~0.0001–0.001
sigma_fat:     0.040   # S/m  — literature: ~0.01–0.06
sigma_muscle:  0.350   # S/m  — literature: ~0.1–0.5 (isotropic approx)
```

### How to plug in real parameters

1. Open `params.yaml`
2. Replace the three `conductivities` values with literature or measured values
3. Adjust `roi.z_target` to match actual tibial nerve depth in your subject population
4. Re-run the sweep and plots

Suggested references for conductivities:
- Gabriel et al. (1996), Physics in Medicine and Biology — tabulated values at multiple frequencies
- Hasgall et al., IT'IS Database for thermal and electromagnetic parameters of biological tissues

### Electrodes

| Key | Value | Meaning |
|-----|-------|---------|
| `shape` | `circle` | `circle` or `square` |
| `size_list` | [5, 10, 15] mm | Electrode radii swept |
| `medial_offset` | 25 mm | Center of active electrode from medial edge |
| `lateral_offset` | 25 mm | Center of return electrode from lateral edge |

## Output metrics (summary.csv columns)

| Column | Units | Description |
|--------|-------|-------------|
| `t_fat_mm` | mm | Subcutaneous fat thickness |
| `elec_r_mm` | mm | Electrode radius |
| `elec_area_cm2` | cm² | Electrode contact area |
| `peak_J_skin` | A/m² | Maximum |J| anywhere on skin surface — comfort proxy |
| `mean_J_roi` | A/m² | Mean |J| in ROI sphere — efficacy proxy |
| `tradeoff` | — | `mean_J_roi / peak_J_skin` (higher = more efficient) |
| `flux_err` | — | Fractional difference in normal flux between electrodes (current conservation check) |
| `roi_layer` | — | Which tissue layer the ROI center falls in |
| `roi_n_nodes` | — | Number of FEM nodes inside ROI sphere (< 5 means ROI is under-sampled) |

## Interpreting the results

- **Smaller electrodes** → higher `peak_J_skin` → more surface discomfort for the same total current
- **More fat** → attenuates current reaching the nerve → lower `mean_J_roi`
- **Tradeoff curve**: the goal is to find electrode sizes that maximize `tradeoff` — more efficacy per unit of skin discomfort

## Dependencies

```bash
pip install gmsh pyvista pyyaml numpy matplotlib
```

Requires `ElmerGrid` and `ElmerSolver` on your `PATH`.

## Known limitations

1. **Uniform isotropic conductivities** — real muscle is anisotropic (higher along fiber direction)
2. **Rectangular slab** — real ankle has curved layered geometry
3. **No frequency dependence** — electrostatics (DC); actual stimulation uses pulsed waveforms
4. **No bone** — for pelvic floor stimulation, bone may be relevant; not modeled here
5. **Coarse ROI** — the ROI sphere approximation is a point estimate, not a true nerve fascicle model
