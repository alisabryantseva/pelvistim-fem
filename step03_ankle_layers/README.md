# step03_ankle_layers — Ankle Cross-Section Layered Model

## What this model is

A finite-element model of an ankle cross-section for tibial nerve stimulation (PTNS-like).

**Geometry** (default): approximate ankle polygon (12-point outline, medial–lateral × anterior–posterior) extruded to depth `Lz`.  Three tissue layers stacked in depth: skin / subcutaneous fat / muscle.  An optional thin **contact layer** at each electrode position models the electrode–skin interface.

```
Top view (x = medial → lateral, y = anterior → posterior):

      *---*---*
    /           \
  *               *
  |  active (+I)  |      ← medial groove  (low x, mid y)
  *               *---*
    \               \
      *---*---*---*---*
                  ← posterior-lateral (high x, high y)
                     return (0V)
```

**Electrode placement**:
- Active: medial groove — between tendon and medial malleolus region
- Return: posterior-lateral — behind lateral malleolus, on the lateral/top surface

**Contact model**: thin contact-material volume on top of the skin layer at each electrode position.  `sigma_contact_Spm` captures the effective interface conductivity (low for dry/reusable electrodes, higher for gel).

### Stimulation control modes

| Mode | Active electrode BC | Return electrode BC | Use case |
|------|---------------------|---------------------|----------|
| `current` (**default**) | `Current Density = I/A` (uniform Neumann) | `Potential = 0 V` | Matches TENS-like current-regulated devices; compliance voltage reported |
| `voltage` | `Potential = 1 V` | `Potential = 0 V` | Geometry exploration; normalise outputs by `total_current_A` |

**Choosing a mode:**
- Use **current mode** (default) when you want to replicate a real device that delivers a fixed mA; outputs directly reflect clinical stimulation at a known current level.
- Use **voltage mode** for geometry sensitivity studies where you care about relative changes between cases, then divide all J/E outputs by `total_current_A` to normalise.

**Compliance voltage** (`compliance_V` in summary): mean potential at the active electrode minus mean at the return (= 0 V with Dirichlet BC). If `compliance_V > compliance_voltage_V`, the stimulator would clip — increase electrode size or reduce `injected_current_mA`.  The column `exceeded_compliance` is `true` when this limit is breached.

### Key outputs

| Metric | Units | Description |
|--------|-------|-------------|
| `peak_J_skin_no_elec` | A/m² | Max \|J\| at skin surface **outside** electrode footprint — comfort proxy |
| `peak_J_skin_with_elec` | A/m² | Max \|J\| at skin surface including under electrode |
| `roi_mean_E` | V/m | Mean \|E\| in ROI sphere at ~10 mm depth — efficacy proxy |
| `roi_mean_J` | A/m² | Mean \|J\| in same ROI sphere |
| `efficiency` | m | `roi_mean_E / peak_J_skin_no_elec` — deep field per unit skin exposure |
| `jn_used` | A/m² | Applied current density at active electrode (current mode only) |
| `compliance_V` | V | mean(V_active) − mean(V_return); warn if > `compliance_voltage_V` |
| `exceeded_compliance` | bool | True if compliance_V > limit |
| `total_current_A` | A | Abs surface integral of J·n at active electrode |
| `I_return_A` | A | Abs surface integral of J·n at return electrode |
| `flux_err` | — | \|I_active_signed + I_return_signed\| / max(…) — signed KCL check |

## What this model is NOT

- **Not anatomically accurate** — polygon approximation, not a real ankle geometry; no bone, no tendon, no blood vessels
- **Not calibrated** — all conductivity values are PLACEHOLDERS
- **Not frequency-dependent** — electrostatics (DC quasi-static); real TENS uses pulsed waveforms
- **Isotropic conductivities** — real muscle is anisotropic

## Directory layout

```
step03_ankle_layers/
├── params.yaml              # all parameters (geometry, layers, contact, placement, stim, …)
├── run_layered_sweep.py     # mesh → solve → extract metrics
├── plot_layered_results.py  # figures from results/summary.json + VTU files
├── smoke_test.py            # quick pipeline validation
└── results/                 # auto-created by run_layered_sweep.py
    ├── summary.csv
    ├── summary.json
    ├── J_surface_maps.png         ← linear scale (99.95th pct max)
    ├── J_surface_maps_log.png     ← log scale  (if log_norm: true)
    ├── J_surface_maps_masked.png  ← electrode footprints masked  (if make_masked: true)
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

### 1. Smoke test — verifies the pipeline (~3–8 min)

```bash
python3 smoke_test.py
```

Checks: VTU created, potential in expected range, J finite, ROI J positive, flux conservation.

### 2. Full parameter sweep — 9 cases (3 fat thicknesses × 3 electrode radii)

```bash
python3 run_layered_sweep.py
```

Progress is printed per case.  With contact layer and ankle geometry, typical runtime is 20–60 min on a laptop.

### 3. Generate figures

```bash
python3 plot_layered_results.py
```

All three J-surface maps are **always generated** (no flags needed):

| File | Scale | Best for |
|------|-------|----------|
| `results/J_surface_maps_linear.png` | Linear, 0 → 99.95th pct | Side-by-side absolute peak J comparison across cases |
| `results/J_surface_maps_log.png` | Log scale | Revealing low-J current spreading far from electrodes |
| `results/J_surface_maps_masked.png` | Linear, electrode footprints = NaN | Current spreading pattern **outside** electrode pads only |
| `results/summary_metrics.png` | — | Metrics vs electrode area: skin peak J, ROI E, efficiency, compliance V |
| `results/representative_3d.png` | — | 3D pyvista render of one case |

`vmax_percentile` under `plotting:` in `params.yaml` controls color scale ceiling.

## Parameters (`params.yaml`)

### Geometry

| Key | Default | Meaning |
|-----|---------|---------|
| `Lx` | 0.08 m | Medial–lateral bounding width (8 cm) |
| `Ly` | 0.06 m | Anterior–posterior bounding depth (6 cm) |
| `Lz` | 0.040 m | Total depth (4 cm) |
| `cross_section` | `"ankle"` | `"ankle"` (12-point polygon) or `"rect"` (rectangular slab) |

### Layers

| Key | Default | Note |
|-----|---------|------|
| `t_skin` | 1.5 mm | Fixed across sweep — PLACEHOLDER |
| `t_fat` | 5 mm | Default (used in smoke test) — PLACEHOLDER |
| `t_fat_sweep` | [3, 5, 8] mm | Fat thicknesses in full sweep |
| t_muscle | computed | `Lz − t_skin − t_fat` |

### Conductivities — ALL PLACEHOLDERS

```yaml
sigma_skin:    0.001   # S/m  effective skin conductivity (background)
sigma_fat:     0.040   # S/m  literature: ~0.01–0.06
sigma_muscle:  0.350   # S/m  isotropic; literature: ~0.1–0.5
```

### Contact layer — PLACEHOLDER

```yaml
contact:
  enabled: true
  model: "layer"             # thin contact-material volume at each electrode
  t_contact_mm: 0.5         # thickness (mm)
  sigma_contact_Spm: 0.005  # effective conductivity (S/m)
                             # dry/reusable: 0.001–0.01
                             # gel-coupled:  0.05–0.2
```

Setting `enabled: false` disables the contact layer; `sigma_skin` then acts as the sole contact-impedance proxy (backward-compatible mode).

### Electrode placement

```yaml
placement:
  active_xy: [0.020, 0.030]    # m: active electrode center (medial groove)
  return_xy:  [0.068, 0.046]   # m: return electrode center (posterior-lateral)
  electrode_shape: "circle"    # "circle" or "square"
  electrode_r_mm_list: [5, 10, 15]
```

Both positions are given as `[x_m, y_m]` in the bounding-box coordinate frame.

### Stimulation

```yaml
stim:
  control_mode:         "current"  # "current" or "voltage"
  injected_current_mA:  5.0
  compliance_voltage_V: 100.0      # flag if V_active exceeds this
```

### Mesh

```yaml
mesh:
  lc_global_mm:    3.0   # background element size
  lc_electrode_mm: 1.5   # fine mesh near electrode
  lc_skin_min:     0.5   # minimum size (mm) — resolves thin layers
```

### How to plug in real parameters

1. Replace `conductivities` with literature or impedance-spectroscopy values
2. Set `contact.sigma_contact_Spm` from measured electrode-skin impedance at your operating frequency
3. Adjust `roi.z_target` to match the actual tibial nerve depth (~7–12 mm)
4. Set `stim.control_mode: "current"` with your device's output current and compliance limit

Suggested references:
- Gabriel et al. (1996), *Physics in Medicine and Biology* — tabulated tissue conductivities
- Hasgall et al., IT'IS Database — updated tissue parameters
- Grimnes & Martinsen, *Bioimpedance and Bioelectricity Basics* — contact impedance models

## Output metrics (summary.csv columns)

| Column | Units | Description |
|--------|-------|-------------|
| `t_fat_mm` | mm | Subcutaneous fat thickness |
| `elec_r_mm` | mm | Electrode radius |
| `elec_area_cm2` | cm² | Electrode contact area |
| `elec_shape` | — | `circle` or `square` |
| `contact_enabled` | — | Whether contact layer was used |
| `sigma_skin` | S/m | Skin conductivity used |
| `control_mode` | — | `current` or `voltage` |
| `jn_used` | A/m² | Applied current density at active electrode (current mode only) |
| `peak_J_skin_with_elec` | A/m² | Max \|J\| at skin surface (including under electrode) |
| `peak_J_skin_no_elec` | A/m² | Max \|J\| at skin surface **outside** electrode footprints |
| `roi_mean_J` | A/m² | Mean \|J\| in ROI sphere |
| `roi_mean_E` | V/m | Mean \|E\| in ROI sphere |
| `efficiency` | m | `roi_mean_E / peak_J_skin_no_elec` |
| `compliance_V` | V | mean(V_active) − mean(V_return) — required drive voltage (current mode) |
| `exceeded_compliance` | bool | True if compliance_V > `compliance_voltage_V` limit |
| `total_current_A` | A | Abs surface integral of J·n at active electrode |
| `I_return_A` | A | Abs surface integral of J·n at return electrode |
| `peak_J_skin_per_A` | 1/m² | `peak_J_skin_no_elec / total_current_A` |
| `roi_mean_E_per_A` | V/m/A | `roi_mean_E / total_current_A` |
| `flux_err` | — | \|I_active_signed + I_return_signed\| / max(…) — signed KCL error |
| `roi_layer` | — | Tissue layer containing ROI center |
| `roi_n_cells` | — | FEM cells in ROI (auto-expanded if < 4) |
| `roi_radius_used_mm` | mm | Actual ROI radius used |

## Interpreting results

- **Smaller electrodes**: higher `peak_J_skin_no_elec` → more skin discomfort per mA
- **More fat**: attenuates `roi_mean_E` → lower deep field per mA; also tends to reduce efficiency
- **Efficiency**: find the electrode size maximising deep field per unit skin exposure
- **Compliance voltage**: in current mode, larger electrodes lower compliance voltage (lower impedance); if compliance_V > limit, reduce current or increase electrode size
- **Contact impedance sweep**: change `sigma_contact_Spm`; higher values lower compliance_V and reduce sensitivity to electrode placement

## Recommended sweeps

```bash
# Fat thickness × electrode size (default, 9 cases)
python3 run_layered_sweep.py

# Contact conductivity sensitivity
# Models dry electrode (low) vs gel-coupled electrode (high)
# Effect: higher sigma_contact → lower compliance_V, less sensitivity to placement
for sig in 0.001 0.005 0.02; do
    sed -i "s/sigma_contact_Spm:.*/sigma_contact_Spm: $sig/" params.yaml
    python3 run_layered_sweep.py
    cp results/summary.json results/summary_sigc${sig}.json
done

# Disable contact layer entirely (bare-skin BC, backward-compatible)
# Set contact.enabled: false in params.yaml

# Background skin conductivity sensitivity
# Captures inter-subject variability in skin hydration / stratum corneum thickness
for sig in 0.0002 0.001 0.005; do
    sed -i "s/sigma_skin:.*/sigma_skin: $sig/" params.yaml
    python3 run_layered_sweep.py
    cp results/summary.json results/summary_sigs${sig}.json
done
```

## Dependencies

```bash
pip install gmsh pyvista pyyaml numpy matplotlib
```

Requires `ElmerGrid` and `ElmerSolver` on your `PATH`.

## Known limitations

1. **Approximate cross-section** — 12-point polygon, not a real ankle geometry
2. **No bone** — tibia / fibula not included; would reduce current path in a real model
3. **No tendons or blood vessels** — simplified soft-tissue only
4. **Isotropic conductivities** — real muscle is anisotropic (higher along fibers)
5. **No frequency dependence** — electrostatics (DC); real TENS uses pulsed waveforms
6. **Uniform Neumann BC** — real electrodes may have non-uniform current density at edges
7. **No sweat/moisture model** — sigma_skin and sigma_contact are homogeneous over electrode
