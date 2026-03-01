# step03_ankle_layers — Layered Ankle Slab Model

## What this model is

A 3-layer finite-element slab representing an ankle cross-section (skin / subcutaneous fat / muscle), with two bipolar surface electrodes on the top face.

**Electrode model:** dry / reusable electrodes without added gel. `sigma_skin` is an **effective parameter** capturing both the epidermis conductivity and the contact impedance. It should remain LOW (≤ 0.005 S/m). See `sigma_skin_sweep` in `params.yaml` to quantify sensitivity.

### Stimulation control modes

| Mode | Active electrode BC | Return electrode BC | Use case |
|------|---------------------|---------------------|----------|
| `voltage` | `Potential = 1 V` | `Potential = 0 V` | Geometry exploration; normalize outputs by `total_current_A` for comparison |
| `current` | `Current Density = I/A` (uniform Neumann) | `Potential = 0 V` | Approximates TENS-like current-regulated stimulators; outputs directly comparable across cases |

**Voltage mode** (default): fixed potentials. The injected current depends strongly on contact impedance (`sigma_skin`). Use the `peak_J_skin_per_A` and `roi_mean_J_per_A` columns for cross-case comparison.

**Current mode**: uniform current density is applied over the active electrode area. The compliance voltage (potential at active electrode) is not constrained. Better approximation for devices with current-regulated output. Set `control.control_mode: current` and `control.injected_current_mA` in `params.yaml`.

> **Elmer implementation note (current mode):** `σ ∂φ/∂n = I/A` is set on the active electrode patch (outward normal = +z at top face), where I is the requested current and A is the electrode area. This is a uniform Neumann BC — it does not enforce a spatially varying density profile.

### Key outputs

- **`peak_J_skin`** — max |J| at skin surface (comfort proxy; higher = more sensation/discomfort risk)
- **`mean_J_roi`** — mean |J| in ROI sphere at ~10 mm depth under active electrode (efficacy proxy; never NaN — radius auto-expands if mesh is sparse)
- **`total_current_A`** — total injected current, computed by integrating J·dA over the active electrode patch
- **`peak_J_skin_per_A`** / **`roi_mean_J_per_A`** — above metrics normalised by `total_current_A` (comparable across conductivity sweeps and modes)
- **`tradeoff`** — `mean_J_roi / peak_J_skin` (higher = more deep stimulation per unit skin exposure)

## What this model is NOT

- **Not anatomically accurate** — rectangular slab, not a real ankle geometry
- **Not calibrated** — all conductivity values are PLACEHOLDERS
- **Not validated** — outputs are order-of-magnitude estimates only

## Directory layout

```
step03_ankle_layers/
├── params.yaml              # all parameters (geometry, layers, conductivities, electrodes, control)
├── run_layered_sweep.py     # mesh → solve → extract metrics for every (t_fat, r) pair
├── plot_layered_results.py  # figures from results/summary.json + VTU files
├── smoke_test.py            # quick pipeline validation (~2–5 min)
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

Checks: VTU created, potential in [0,1]V, J finite, E field present, current conservation < 5%, ROI J positive.

### 2. Full parameter sweep — 9 cases (3 fat thicknesses × 3 electrode radii)

```bash
python3 run_layered_sweep.py
```

Typical runtime: 15–40 min on a laptop. Progress is printed per case.

### 3. Generate figures

```bash
python3 plot_layered_results.py
```

Outputs:
- `results/J_surface_maps.png` — |J| heatmaps (global color scale: 0 → 99.5th percentile)
- `results/summary_metrics.png` — raw and normalised metrics vs electrode area (2×3 layout)
- `results/representative_3d.png` — 3D pyvista render

## Parameters (`params.yaml`)

### Geometry

| Key | Value | Meaning |
|-----|-------|---------|
| `Lx` | 0.12 m | Medial–lateral width (12 cm) |
| `Ly` | 0.09 m | Anterior–posterior depth (9 cm) |
| `Lz` | 0.040 m | Total slab depth (4 cm) |

### Layers

Layer order bottom → top: **muscle → fat → skin**

| Key | Default | Note |
|-----|---------|------|
| `t_skin` | 1.5 mm | Fixed across sweep |
| `t_fat` | 5 mm | Default (used in smoke test) |
| `t_fat_sweep` | [3, 5, 8] mm | Fat thicknesses in full sweep |
| t_muscle | computed | `Lz − t_skin − t_fat` |

### Conductivities — ALL PLACEHOLDERS

```yaml
sigma_skin:    0.001   # S/m  effective dry-electrode value — PLACEHOLDER
sigma_fat:     0.040   # S/m  literature: ~0.01–0.06
sigma_muscle:  0.350   # S/m  literature: ~0.1–0.5 (isotropic approx)

sigma_skin_sweep: [0.0002, 0.001, 0.005]  # explore sensitivity
```

**`sigma_skin` is an effective parameter** capturing epidermis + contact impedance for dry/reusable electrodes. Gel-coupled electrodes have higher effective conductivity (0.003–0.01 S/m range). For dry/reusable electrodes without gel, values below 0.001 S/m are physically appropriate. Use `sigma_skin_sweep` to characterise sensitivity:

```bash
# Manual sigma_skin sweep: edit params.yaml, re-run
for sig in 0.0002 0.001 0.005; do
    sed -i "s/sigma_skin:.*/sigma_skin: $sig/" params.yaml
    python3 run_layered_sweep.py
    cp results/summary.json results/summary_sigma${sig}.json
done
```

### How to plug in real parameters

1. Open `params.yaml`
2. Replace `conductivities` with literature or impedance-spectroscopy values
3. Adjust `roi.z_target` to match the actual tibial nerve depth in your subject population
4. If using current-regulated stimulator: set `control.control_mode: current` and `control.injected_current_mA`

Suggested references:
- Gabriel et al. (1996), *Physics in Medicine and Biology* — tabulated tissue conductivities
- Hasgall et al., IT'IS Database for tissue parameters
- Grimnes & Martinsen, *Bioimpedance and Bioelectricity Basics* — contact impedance models

### Electrodes

| Key | Value | Meaning |
|-----|-------|---------|
| `shape` | `circle` | `circle` or `square` |
| `size_list` | [5, 10, 15] mm | Electrode radii swept |
| `medial_offset` | 25 mm | Active electrode center from medial edge |
| `lateral_offset` | 25 mm | Return electrode center from lateral edge |

### Control

| Key | Default | Meaning |
|-----|---------|---------|
| `control_mode` | `voltage` | `"voltage"` or `"current"` |
| `injected_current_mA` | 5.0 | Applied current (current mode only) |

## Output metrics (summary.csv columns)

| Column | Units | Description |
|--------|-------|-------------|
| `t_fat_mm` | mm | Subcutaneous fat thickness |
| `elec_r_mm` | mm | Electrode radius |
| `elec_area_cm2` | cm² | Electrode contact area |
| `sigma_skin` | S/m | Skin/contact conductivity used |
| `control_mode` | — | `voltage` or `current` |
| `peak_J_skin` | A/m² | Max \|J\| at skin surface — comfort proxy |
| `mean_J_roi` | A/m² | Mean \|J\| in ROI sphere — efficacy proxy (never NaN) |
| `roi_mean_E` | V/m | Mean \|E\| in ROI sphere |
| `total_current_A` | A | Injected current from surface integral of J·dA |
| `peak_J_skin_per_A` | 1/m² | `peak_J_skin / total_current_A` — normalised comfort proxy |
| `roi_mean_J_per_A` | 1/m² | `mean_J_roi / total_current_A` — normalised efficacy proxy |
| `roi_mean_E_per_A` | V/m/A | `roi_mean_E / total_current_A` |
| `tradeoff` | — | `mean_J_roi / peak_J_skin` |
| `flux_err` | — | \|I_active − I_return\| / max(…) — current conservation error |
| `roi_layer` | — | Tissue layer containing the ROI center |
| `roi_n_cells` | — | FEM cells inside the ROI sphere (auto-expanded if < 4) |
| `roi_radius_used_mm` | mm | Actual ROI radius used (may exceed `roi.roi_radius`) |

## Interpreting the results

- **Smaller electrodes**: higher `peak_J_skin_per_A` → more skin discomfort per mA
- **More fat**: attenuates `roi_mean_J_per_A` → lower deep stimulation per mA
- **Tradeoff / normalised tradeoff**: find the electrode size that maximises deep J while keeping skin J acceptable
- **`sigma_skin` sweep**: understand how much the results depend on contact impedance assumptions — if the sweep shows large variation, the contact model is the dominant uncertainty

## Dependencies

```bash
pip install gmsh pyvista pyyaml numpy matplotlib
```

Requires `ElmerGrid` and `ElmerSolver` on your `PATH`.

## Known limitations

1. **Uniform isotropic conductivities** — real muscle is anisotropic (higher along fiber direction)
2. **Rectangular slab** — real ankle has curved layered geometry
3. **No frequency dependence** — electrostatics (DC); actual TENS uses pulsed waveforms
4. **Uniform Neumann BC in current mode** — real electrodes may have non-uniform current density depending on edge effects and skin impedance variation
5. **No bone** — for pelvic floor stimulation, bone may be relevant
6. **sigma_skin is isotropic and uniform** — real skin impedance varies spatially and with frequency
