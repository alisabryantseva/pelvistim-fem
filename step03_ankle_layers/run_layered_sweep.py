"""
run_layered_sweep.py — ankle-like layered slab, bipolar electrode sweep
=======================================================================
Geometry: 3-layer slab (skin / fat / muscle) with two surface electrodes.
Active electrode (medial, +1V or current-controlled) and return electrode
(lateral, 0V).

Usage (from step03_ankle_layers/):
    python3 run_layered_sweep.py             # full sweep from params.yaml
    python3 run_layered_sweep.py --smoke     # 1 coarse case, quick check

Outputs: results/<case>/  with VTU + case.sif
         results/summary.csv  + results/summary.json

Metrics in summary:
  peak_J_skin        — max |J| at skin surface (comfort proxy)
  mean_J_roi         — mean |J| in ROI sphere (efficacy proxy), NEVER NaN
  roi_mean_E         — mean |E| in ROI sphere (if electric field exported)
  total_current_A    — total injected current (surface integral of J·n)
  peak_J_skin_per_A  — peak_J_skin normalised by total_current_A  [A/m² per A]
  roi_mean_J_per_A   — mean_J_roi  normalised by total_current_A
  roi_mean_E_per_A   — roi_mean_E  normalised by total_current_A  [V/m per A]
  flux_err           — |I_active - I_return| / max(...)  (conservation check)
  roi_n_cells        — number of FEM cells inside the ROI sphere used
  roi_radius_used_mm — actual ROI radius (may be expanded from params default)
  sigma_skin         — skin conductivity used in this case
  control_mode       — "voltage" or "current"
"""

import sys
import argparse
import subprocess
import csv
import json
import yaml
import numpy as np
import pyvista as pv
import gmsh
from pathlib import Path

PARAMS_FILE = Path(__file__).parent / "params.yaml"
RESULTS_DIR = Path(__file__).parent / "results"


# ── Load parameters ───────────────────────────────────────────────────────────
def load_params():
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


# ── 1. Build gmsh mesh ────────────────────────────────────────────────────────
def build_mesh(p, t_fat, elec_r, run_dir, coarse=False):
    """
    Three-layer box (muscle / fat / skin) with two electrode patches on top.
    Physical groups:
        Volume 1 = muscle, 2 = fat, 3 = skin
        Surface 101 = active electrode, 102 = return electrode, 103 = insulated
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    g = p["geometry"]
    Lx, Ly, Lz = g["Lx"], g["Ly"], g["Lz"]
    t_skin   = p["layers"]["t_skin"]
    t_muscle = Lz - t_skin - t_fat
    if t_muscle <= 1e-4:
        raise ValueError(f"t_muscle = {t_muscle*1000:.2f} mm ≤ 0.1 mm — "
                         f"reduce t_skin + t_fat or increase Lz")

    ep        = p["electrodes"]
    shape     = ep["shape"]
    medial_x  = ep["medial_offset"]
    lateral_x = Lx - ep["lateral_offset"]
    elec_y    = Ly / 2.0

    # Validate electrode placement
    if medial_x - elec_r < 0 or lateral_x + elec_r > Lx:
        raise ValueError("Electrode extends outside box bounds — "
                         "reduce elec_r or adjust offsets")

    m = p["mesh"]
    scale = 2.0 if coarse else 1.0
    lc_elec = elec_r * m["lc_elec_factor"] * scale
    lc_bulk = min(elec_r * 4, m["lc_bulk_cap"]) * scale
    lc_min  = m["lc_skin_min"]   # keep fixed so skin layer is always resolved

    z0_fat  = t_muscle
    z0_skin = t_muscle + t_fat

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("ankle_layers")

    # Three stacked layer boxes
    vol_musc = gmsh.model.occ.addBox(0, 0, 0,      Lx, Ly, t_muscle)
    vol_fat  = gmsh.model.occ.addBox(0, 0, z0_fat,  Lx, Ly, t_fat)
    vol_skin = gmsh.model.occ.addBox(0, 0, z0_skin, Lx, Ly, t_skin)

    # Electrode patches on top face (z = Lz)
    if shape == "circle":
        e1 = gmsh.model.occ.addDisk(medial_x,  elec_y, Lz, elec_r, elec_r)
        e2 = gmsh.model.occ.addDisk(lateral_x, elec_y, Lz, elec_r, elec_r)
    else:
        e1 = gmsh.model.occ.addRectangle(
            medial_x - elec_r,  elec_y - elec_r, Lz, 2*elec_r, 2*elec_r)
        e2 = gmsh.model.occ.addRectangle(
            lateral_x - elec_r, elec_y - elec_r, Lz, 2*elec_r, 2*elec_r)

    # Fragment: merge layer interfaces + embed electrodes in skin top face
    gmsh.model.occ.fragment(
        [(3, vol_musc), (3, vol_fat), (3, vol_skin)],
        [(2, e1), (2, e2)]
    )
    gmsh.model.occ.synchronize()

    # ── Identify volumes by z-centroid ────────────────────────────────────────
    def bb(dim, tag):
        return gmsh.model.getBoundingBox(dim, tag)

    def z_mid(dim, tag):
        b = bb(dim, tag)
        return (b[2] + b[5]) / 2.0

    def xy_center(dim, tag):
        b = bb(dim, tag)
        return np.array([(b[0]+b[3])/2, (b[1]+b[4])/2, (b[2]+b[5])/2])

    vols = gmsh.model.getEntities(3)
    musc_v = min(vols, key=lambda dt: abs(z_mid(*dt) - t_muscle / 2))
    fat_v  = min(vols, key=lambda dt: abs(z_mid(*dt) - (z0_fat + t_fat / 2)))
    skin_v = min(vols, key=lambda dt: abs(z_mid(*dt) - (z0_skin + t_skin / 2)))

    # ── Identify electrode surfaces on top face ───────────────────────────────
    surfs = gmsh.model.getEntities(2)
    top_surfs = [dt for dt in surfs if abs(xy_center(*dt)[2] - Lz) < Lz * 1e-3]

    e1_pos = np.array([medial_x,  elec_y])
    e2_pos = np.array([lateral_x, elec_y])

    e1_s = min(top_surfs, key=lambda dt: np.linalg.norm(xy_center(*dt)[:2] - e1_pos))
    e2_s = min(top_surfs, key=lambda dt: np.linalg.norm(xy_center(*dt)[:2] - e2_pos))
    other_s = [dt for dt in surfs if dt not in (e1_s, e2_s)]

    # ── Physical groups ───────────────────────────────────────────────────────
    gmsh.model.addPhysicalGroup(3, [musc_v[1]], 1, name="muscle")
    gmsh.model.addPhysicalGroup(3, [fat_v[1]],  2, name="fat")
    gmsh.model.addPhysicalGroup(3, [skin_v[1]], 3, name="skin")
    gmsh.model.addPhysicalGroup(2, [e1_s[1]],   101, name="active_electrode")
    gmsh.model.addPhysicalGroup(2, [e2_s[1]],   102, name="return_electrode")
    gmsh.model.addPhysicalGroup(2, [s[1] for s in other_s], 103, name="insulated")

    # ── Mesh size field ───────────────────────────────────────────────────────
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", [e1_s[1], e2_s[1]])
    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "InField",  f_dist)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin",  lc_elec)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMax",  lc_bulk)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin",  elec_r)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax",  elec_r * 7)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)

    gmsh.model.mesh.generate(3)
    msh = run_dir / "mesh.msh"
    gmsh.write(str(msh))
    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()

    return n_nodes, np.array([medial_x, elec_y, Lz]), np.array([lateral_x, elec_y, Lz])


# ── 2. Detect Elmer boundary IDs for the two electrodes ──��───────────────────
def detect_elec_bc_ids(elmer_mesh_dir, e1_pos, e2_pos, Lz):
    nodes = {}
    with open(elmer_mesh_dir / "mesh.nodes") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                nid = int(parts[0])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                nodes[nid] = np.array([x, y, z])
            except ValueError:
                continue

    etype_nn = {202: 2, 303: 3, 306: 6, 404: 4}
    tol_z    = Lz * 0.01

    bc_centers = {}
    with open(elmer_mesh_dir / "mesh.boundary") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                bcid  = int(parts[1])
                etype = int(parts[4])
            except (ValueError, IndexError):
                continue
            nn = etype_nn.get(etype, 3)
            try:
                nids = [int(p) for p in parts[5:5+nn]]
            except ValueError:
                continue
            coords = [nodes[n] for n in nids if n in nodes]
            if len(coords) < 3:
                continue
            zvals = [c[2] for c in coords]
            if min(zvals) < Lz - tol_z:
                continue
            cxy = np.mean([c[:2] for c in coords], axis=0)
            bc_centers.setdefault(bcid, []).append(cxy)

    bc_mean = {bid: np.mean(pts, axis=0) for bid, pts in bc_centers.items() if pts}
    if len(bc_mean) < 2:
        raise RuntimeError(f"Expected ≥2 top-face BCs, found: {list(bc_mean.keys())}")

    e1_id = min(bc_mean, key=lambda b: np.linalg.norm(bc_mean[b] - e1_pos[:2]))
    e2_id = min(bc_mean, key=lambda b: np.linalg.norm(bc_mean[b] - e2_pos[:2]))
    return e1_id, e2_id


# ── 3. Write Elmer SIF (3-body, 3-material) ───────────────────────────────────
SIF_TEMPLATE = """\
Header
  CHECK KEYWORDS Warn
  Mesh DB "." "elmer_mesh"
  Include Path ""
  Results Directory "results"
End

Simulation
  Max Output Level = 3
  Coordinate System = Cartesian 3D
  Coordinate Mapping(3) = 1 2 3
  Simulation Type = Steady State
  Steady State Max Iterations = 1
  Output Intervals = 1
End

Constants
  Permittivity of Vacuum = 8.8542e-12
End

! ── Three tissue bodies (muscle / fat / skin) ────────────────────────────────
Body 1
  Name = "muscle"
  Target Bodies(1) = 1
  Equation = 1
  Material = 1
End

Body 2
  Name = "fat"
  Target Bodies(1) = 2
  Equation = 1
  Material = 2
End

Body 3
  Name = "skin"
  Target Bodies(1) = 3
  Equation = 1
  Material = 3
End

Equation 1
  Name = "Conduction"
  Active Solvers(2) = 1 2
End

Solver 1
  Equation = "Static Current Conduction"
  Procedure = "StatCurrentSolve" "StatCurrentSolver"
  Variable = "Potential"
  Variable DOFs = 1
  Calculate Volume Current = True
  Linear System Solver = Direct
  Linear System Direct Method = {lin_solver}
  Steady State Convergence Tolerance = {tol}
End

Solver 2
  Equation = "ResultOutput"
  Procedure = "ResultOutputSolve" "ResultOutputSolver"
  Output File Name = "case"
  Output Format = VTU
  VTU Format = Logical True
  Save Geometry IDs = Logical True
End

! ── PLACEHOLDER conductivities — replace with measured values ─────────────────
Material 1
  Name = "muscle"
  Electric Conductivity = {sigma_muscle}   ! PLACEHOLDER S/m
End

Material 2
  Name = "fat"
  Electric Conductivity = {sigma_fat}      ! PLACEHOLDER S/m
End

Material 3
  Name = "skin"
  Electric Conductivity = {sigma_skin}     ! PLACEHOLDER effective S/m (dry electrode)
End

Boundary Condition 1
  Name = "active_electrode"
  Target Boundaries = {e1_id}
{bc1_active}
End

Boundary Condition 2
  Name = "return_electrode"
  Target Boundaries = {e2_id}
  Potential = 0.0
End
"""


def write_sif(run_dir, e1_id, e2_id, p, elec_r, sigma_skin_override=None):
    c  = p["conductivities"]
    sv = p["solver"]
    ep = p["electrodes"]
    ctrl = p.get("control", {})
    mode = ctrl.get("control_mode", "voltage")

    sigma_skin = sigma_skin_override if sigma_skin_override is not None \
        else c["sigma_skin"]

    if mode == "voltage":
        bc1_active = "  Potential = 1.0"
    else:  # current
        I_A = ctrl.get("injected_current_mA", 5.0) * 1e-3
        shape = ep["shape"]
        area = np.pi * elec_r**2 if shape == "circle" else (2 * elec_r)**2
        # σ ∂φ/∂n = I/A > 0  (outward n = +z at top face → current into tissue)
        jn = I_A / area
        bc1_active = (f"  Current Density = {jn:.6e}  "
                      f"! I={I_A*1e3:.1f}mA uniform; A={area*1e4:.3f}cm²")

    sif = SIF_TEMPLATE.format(
        sigma_muscle=c["sigma_muscle"],
        sigma_fat=c["sigma_fat"],
        sigma_skin=sigma_skin,
        e1_id=e1_id,
        e2_id=e2_id,
        tol=sv["tolerance"],
        lin_solver=sv["linear_solver"],
        bc1_active=bc1_active,
    )
    (run_dir / "case.sif").write_text(sif)


# ── 4. Run shell commands ─────────────────────────────────────────────────────
def _run(cmd, cwd, label):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR in {label}:")
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        sys.exit(1)


# ── 5a. Compute total injected current (surface integration) ──────────────────
def compute_injected_current(mesh, e1_pos3d, e2_pos3d, elec_r, Lz):
    """
    Integrate J·dA over the active and return electrode patches on the top face.

    At the top face (z = Lz), the outward normal is +z, so J·n_outward = J_z.
    For current flowing INTO the tissue, J_z < 0, so
        I_injected = |∑ J_z * cell_area|  over active patch.

    Returns: (I_active_A, I_return_A, flux_err)
    """
    tol_z = Lz * 5e-3

    surface    = mesh.extract_surface(algorithm="dataset_surface")
    surface_cd = surface.point_data_to_cell_data()
    J_cells    = np.array(surface_cd.cell_data["volume current"])  # (N_surf, 3)
    cell_pts   = np.array(surface.cell_centers().points)           # (N_surf, 3)

    sizes = surface.compute_cell_sizes()
    areas = np.array(sizes.cell_data["Area"])                      # (N_surf,)

    active_mask = ((cell_pts[:, 2] > Lz - tol_z) &
                   (np.linalg.norm(cell_pts[:, :2] - e1_pos3d[:2], axis=1) < elec_r * 1.2))
    return_mask = ((cell_pts[:, 2] > Lz - tol_z) &
                   (np.linalg.norm(cell_pts[:, :2] - e2_pos3d[:2], axis=1) < elec_r * 1.2))

    if not active_mask.any() or not return_mask.any():
        return np.nan, np.nan, np.nan

    # J_z = J·n_outward at top face
    I_active = float(abs(np.sum(J_cells[active_mask, 2] * areas[active_mask])))
    I_return = float(abs(np.sum(J_cells[return_mask, 2] * areas[return_mask])))
    denom    = max(I_active, I_return)
    flux_err = float(abs(I_active - I_return) / denom) if denom > 0 else np.nan

    return I_active, I_return, flux_err


# ── 5b. Evaluate ROI using cell data (robust, never NaN) ─────────────────────
def eval_roi(mesh, roi_cen, roi_radius_init, min_cells=4):
    """
    Compute mean |J| and mean |E| inside a spherical ROI using cell-centroid data.

    If fewer than min_cells cells are found at the initial radius, the radius is
    expanded (×1.5, ×2, ×3) until enough cells are collected.  A warning is
    printed if expansion occurs; a value is always returned (never NaN from
    under-sampling).

    Returns: (mean_J, mean_E, n_cells, roi_radius_used, warning_str_or_None)
    """
    mesh_cd    = mesh.point_data_to_cell_data()
    J_cells    = np.array(mesh_cd.cell_data["volume current"])
    Jmag_cells = np.linalg.norm(J_cells, axis=1)

    # Compute E = -∇φ at cell centres via pyvista gradient (more reliable than
    # asking Elmer to export it — StatCurrentSolve doesn't always do so).
    try:
        grad_mesh  = mesh_cd.compute_derivative(scalars="potential")
        E_cells    = -np.array(grad_mesh.cell_data["gradient"])  # E = -∇φ
        Emag_cells = np.linalg.norm(E_cells, axis=1)
    except Exception:
        Emag_cells = None

    cell_pts = np.array(mesh_cd.cell_centers().points)
    dist     = np.linalg.norm(cell_pts - roi_cen, axis=1)

    warning          = None
    roi_radius_used  = roi_radius_init

    for mult in [1.0, 1.5, 2.0, 3.0]:
        r_test = roi_radius_init * mult
        mask   = dist < r_test
        n      = int(mask.sum())
        if n >= min_cells:
            roi_radius_used = r_test
            if mult > 1.0:
                warning = (f"ROI radius expanded {mult:.1f}x to "
                           f"{r_test*1000:.1f} mm  ({n} cells found)")
            break
    else:
        roi_radius_used = roi_radius_init * 3.0
        mask   = dist < roi_radius_used
        n      = int(mask.sum())
        warning = (f"ROI at 3x expansion ({roi_radius_used*1000:.1f} mm) "
                   f"has only {n} cells — results may be noisy")

    n = int(mask.sum())
    if n == 0:
        return (np.nan, np.nan, 0, roi_radius_used,
                "No cells in ROI even at 3x expansion")

    mean_J = float(Jmag_cells[mask].mean())
    mean_E = float(Emag_cells[mask].mean()) if Emag_cells is not None else np.nan

    return mean_J, mean_E, n, roi_radius_used, warning


# ── 5c. Extract all metrics from VTU ──────────────────────────────────────────
def extract_results(run_dir, p, t_fat, elec_r, e1_pos3d, e2_pos3d, elec_shape,
                    sigma_skin_used=None):
    vtu_path = run_dir / "results" / "case_t0001.vtu"
    if not vtu_path.exists():
        raise FileNotFoundError(f"VTU not found: {vtu_path}")

    mesh = pv.read(str(vtu_path))
    pts  = np.array(mesh.points)
    J    = np.array(mesh.point_data["volume current"])
    Jmag = np.linalg.norm(J, axis=1)

    g  = p["geometry"]
    Lx, Ly, Lz = g["Lx"], g["Ly"], g["Lz"]
    tol_z = Lz * 5e-3

    # ── Peak |J| at skin surface (node-based, z ≈ Lz) ────────────────────────
    skin_mask   = pts[:, 2] > Lz - tol_z
    peak_J_skin = float(Jmag[skin_mask].max()) if skin_mask.any() else np.nan

    # ── Total injected current (surface integration) ──────────────────────────
    I_active, I_return, flux_err = compute_injected_current(
        mesh, e1_pos3d, e2_pos3d, elec_r, Lz)
    print(f"    I_active={I_active:.4e} A  "
          f"I_return={I_return:.4e} A  "
          f"flux_err={flux_err:.2e}")

    # ── ROI (cell-based, auto-expanding) ──────────────────────────────────────
    r_cfg    = p["roi"]
    z_nerve  = Lz - r_cfg["z_target"]
    roi_cen  = np.array([e1_pos3d[0], e1_pos3d[1], z_nerve])

    mean_J_roi, mean_E_roi, roi_n_cells, roi_r_used, roi_warn = eval_roi(
        mesh, roi_cen, r_cfg["roi_radius"])

    if roi_warn:
        print(f"    ROI: {roi_warn}")

    # ── Layer at ROI depth ────────────────────────────────────────────────────
    t_skin    = p["layers"]["t_skin"]
    z_fat_bot = Lz - t_skin - t_fat
    roi_layer = ("skin"   if z_nerve > Lz - t_skin
                 else "fat"    if z_nerve > z_fat_bot
                 else "muscle")

    # ── Electrode area ────────────────────────────────────────────────────────
    area = np.pi * elec_r**2 if elec_shape == "circle" else (2 * elec_r)**2

    # ── Normalise by injected current ─────────────────────────────────────────
    I_ref = I_active if np.isfinite(I_active) and I_active > 0 else np.nan

    def _norm(val):
        v = float(val)
        return v / I_ref if np.isfinite(v) and np.isfinite(I_ref) else np.nan

    peak_J_skin_per_A = _norm(peak_J_skin)
    roi_mean_J_per_A  = _norm(mean_J_roi)
    roi_mean_E_per_A  = _norm(mean_E_roi)

    # ── Tradeoff (ROI / skin peak, raw values) ────────────────────────────────
    tradeoff = (float(mean_J_roi) / peak_J_skin
                if peak_J_skin > 0 and np.isfinite(mean_J_roi)
                else np.nan)

    # ── Which sigma_skin was used ─────────────────────────────────────────────
    c    = p["conductivities"]
    ctrl = p.get("control", {})
    sig  = sigma_skin_used if sigma_skin_used is not None else c["sigma_skin"]

    def _r(val, n):
        v = float(val)
        return round(v, n) if np.isfinite(v) else v   # returns float nan if nan

    return {
        "t_fat_mm":           _r(t_fat * 1000, 2),
        "elec_r_mm":          _r(elec_r * 1000, 2),
        "elec_area_cm2":      _r(area * 1e4, 4),
        "elec_shape":         elec_shape,
        "sigma_skin":         sig,
        "control_mode":       ctrl.get("control_mode", "voltage"),
        "peak_J_skin":        _r(peak_J_skin, 6),
        "mean_J_roi":         _r(mean_J_roi, 6),
        "roi_mean_E":         _r(mean_E_roi, 4),
        "total_current_A":    _r(I_active, 8),
        "peak_J_skin_per_A":  _r(peak_J_skin_per_A, 4),
        "roi_mean_J_per_A":   _r(roi_mean_J_per_A, 4),
        "roi_mean_E_per_A":   _r(roi_mean_E_per_A, 4),
        "tradeoff":           _r(tradeoff, 6),
        "flux_err":           _r(flux_err, 6),
        "roi_layer":          roi_layer,
        "roi_n_cells":        roi_n_cells,
        "roi_radius_used_mm": _r(roi_r_used * 1000, 2),
    }


# ── 6. Main sweep ─────────────────────────────────────────────────────────────
def run_sweep(p, t_fat_list, elec_r_list, coarse=False, sigma_skin_override=None):
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = []

    sigma_skin = (sigma_skin_override
                  if sigma_skin_override is not None
                  else p["conductivities"]["sigma_skin"])

    for t_fat in t_fat_list:
        for elec_r in elec_r_list:
            label = f"tfat{int(t_fat*1000):04d}um_r{int(elec_r*1000):04d}um"
            run_dir = RESULTS_DIR / label
            print(f"\n[{label}]  t_fat={t_fat*1000:.1f}mm  r={elec_r*1000:.1f}mm  "
                  f"sigma_skin={sigma_skin}")

            print("  meshing ...")
            n_nodes, e1_pos, e2_pos = build_mesh(p, t_fat, elec_r, run_dir,
                                                  coarse=coarse)
            print(f"    {n_nodes} nodes")

            print("  ElmerGrid ...")
            elmer_dir = run_dir / "elmer_mesh"
            elmer_dir.mkdir(exist_ok=True)
            _run(["ElmerGrid", "14", "2", "mesh.msh", "-out", "elmer_mesh"],
                 cwd=run_dir, label="ElmerGrid")

            print("  detecting electrode BCs ...")
            e1_id, e2_id = detect_elec_bc_ids(
                elmer_dir, e1_pos, e2_pos, p["geometry"]["Lz"])
            print(f"    active={e1_id}  return={e2_id}")

            write_sif(run_dir, e1_id, e2_id, p, elec_r,
                      sigma_skin_override=sigma_skin_override)
            (run_dir / "results").mkdir(exist_ok=True)

            print("  ElmerSolver ...")
            _run(["ElmerSolver", "case.sif"], cwd=run_dir, label="ElmerSolver")

            print("  extracting metrics ...")
            res = extract_results(run_dir, p, t_fat, elec_r, e1_pos, e2_pos,
                                  p["electrodes"]["shape"],
                                  sigma_skin_used=sigma_skin)
            print(f"    peak_J_skin={res['peak_J_skin']:.4f}  "
                  f"mean_J_roi={res['mean_J_roi']:.6f}  "
                  f"tradeoff={res['tradeoff']:.4f}  "
                  f"roi_n_cells={res['roi_n_cells']}")
            all_results.append(res)

    return all_results


def save_results(all_results):
    if not all_results:
        return
    # CSV
    csv_path = RESULTS_DIR / "summary.csv"
    keys = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved → {csv_path}")

    # JSON
    json_path = RESULTS_DIR / "summary.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: None if (
            isinstance(x, float) and np.isnan(x)) else x)
    print(f"Saved → {json_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ankle layered slab sweep")
    parser.add_argument("--smoke", action="store_true",
                        help="Single coarse case for quick pipeline check")
    args = parser.parse_args()

    p = load_params()

    if args.smoke:
        t_fat_list  = [p["layers"]["t_fat"]]
        elec_r_list = [p["electrodes"]["size_list"][1]]   # middle size
        coarse      = True
        print("=== SMOKE TEST (1 coarse case) ===")
    else:
        t_fat_list  = p["layers"]["t_fat_sweep"]
        elec_r_list = p["electrodes"]["size_list"]
        coarse      = False
        print(f"=== FULL SWEEP: {len(t_fat_list)} fat thicknesses × "
              f"{len(elec_r_list)} electrode sizes = "
              f"{len(t_fat_list)*len(elec_r_list)} cases ===")

    results = run_sweep(p, t_fat_list, elec_r_list, coarse=coarse)
    save_results(results)

    print("\nAll done. Run plot_layered_results.py to generate figures.")
