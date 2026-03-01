"""
run_layered_sweep.py — ankle-like layered slab, bipolar electrode sweep
=======================================================================
Geometry: 3-layer slab (skin / fat / muscle) with two surface electrodes.
Active electrode (medial, +1V) and return electrode (lateral, 0V).

Usage (from step03_ankle_layers/):
    python3 run_layered_sweep.py             # full sweep from params.yaml
    python3 run_layered_sweep.py --smoke     # 1 coarse case, quick check

Outputs: results/<case>/  with VTU + case.sif
         results/summary.csv  + results/summary.json
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
    lc_min  = m["lc_skin_min"]   # keep this fixed so skin is always resolved

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


# ── 2. Detect Elmer boundary IDs for the two electrodes ──────────────────────
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

! ── PLACEHOLDER conductivities — replace with literature values ───────────────
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
  Electric Conductivity = {sigma_skin}     ! PLACEHOLDER S/m
End

Boundary Condition 1
  Name = "active_electrode"
  Target Boundaries = {e1_id}
  Potential = 1.0
End

Boundary Condition 2
  Name = "return_electrode"
  Target Boundaries = {e2_id}
  Potential = 0.0
End
"""


def write_sif(run_dir, e1_id, e2_id, p):
    c  = p["conductivities"]
    sv = p["solver"]
    sif = SIF_TEMPLATE.format(
        sigma_muscle=c["sigma_muscle"],
        sigma_fat=c["sigma_fat"],
        sigma_skin=c["sigma_skin"],
        e1_id=e1_id, e2_id=e2_id,
        tol=sv["tolerance"],
        lin_solver=sv["linear_solver"],
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


# ── 5. Extract results from VTU ───────────────────────────────────────────────
def extract_results(run_dir, p, t_fat, elec_r, e1_pos3d, e2_pos3d, elec_shape):
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

    # ── Skin surface (z ≈ Lz) ─────────────────────────────────────────────────
    skin_mask = pts[:, 2] > Lz - tol_z

    # ── Electrode footprint (nodes within 1.2×r of electrode center on skin) ──
    r_to_e1 = np.linalg.norm(pts[:, :2] - e1_pos3d[:2], axis=1)
    r_to_e2 = np.linalg.norm(pts[:, :2] - e2_pos3d[:2], axis=1)
    active_mask = skin_mask & (r_to_e1 < elec_r * 1.2)
    return_mask = skin_mask & (r_to_e2 < elec_r * 1.2)

    peak_J_active = Jmag[active_mask].max() if active_mask.any() else np.nan
    peak_J_return = Jmag[return_mask].max() if return_mask.any() else np.nan
    peak_J_skin   = max(peak_J_active, peak_J_return)

    # ── Approx current conservation: flux through electrode patches ───────────
    flux_active = np.abs(J[active_mask, 2]).mean() if active_mask.any() else np.nan
    flux_return = np.abs(J[return_mask, 2]).mean() if return_mask.any() else np.nan
    if np.isfinite(flux_active) and np.isfinite(flux_return):
        flux_err = abs(flux_active - flux_return) / max(flux_active, flux_return)
    else:
        flux_err = np.nan

    # ── ROI under medial electrode (tibial nerve proxy) ───────────────────────
    r_cfg   = p["roi"]
    z_nerve = Lz - r_cfg["z_target"]
    roi_cen = np.array([e1_pos3d[0], e1_pos3d[1], z_nerve])
    dist    = np.linalg.norm(pts - roi_cen, axis=1)
    roi_mask = dist < r_cfg["roi_radius"]

    if roi_mask.sum() < 3:
        mean_J_roi = np.nan
    else:
        mean_J_roi = Jmag[roi_mask].mean()

    # ── Layer thickness at ROI depth — warn if not in muscle ─────────────────
    t_skin    = p["layers"]["t_skin"]
    z_fat_bot = Lz - t_skin - t_fat
    roi_layer = ("skin"   if z_nerve > Lz - t_skin
                 else "fat"    if z_nerve > z_fat_bot
                 else "muscle")

    if elec_shape == "circle":
        area = np.pi * elec_r**2
    else:
        area = (2 * elec_r)**2

    tradeoff = mean_J_roi / peak_J_skin if peak_J_skin > 0 else np.nan

    return {
        "t_fat_mm":       round(t_fat * 1000, 2),
        "elec_r_mm":      round(elec_r * 1000, 2),
        "elec_area_cm2":  round(area * 1e4, 4),
        "elec_shape":     elec_shape,
        "peak_J_skin":    round(float(peak_J_skin), 4),
        "mean_J_roi":     round(float(mean_J_roi), 4),
        "tradeoff":       round(float(tradeoff), 6),
        "flux_err":       round(float(flux_err), 6),
        "roi_layer":      roi_layer,
        "roi_n_nodes":    int(roi_mask.sum()),
    }


# ── 6. Main sweep ─────────────────────────────────────────────────────────────
def run_sweep(p, t_fat_list, elec_r_list, coarse=False):
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = []

    for t_fat in t_fat_list:
        for elec_r in elec_r_list:
            label = f"tfat{int(t_fat*1000):04d}um_r{int(elec_r*1000):04d}um"
            run_dir = RESULTS_DIR / label
            print(f"\n[{label}]  t_fat={t_fat*1000:.1f}mm  r={elec_r*1000:.1f}mm")

            print("  meshing ...")
            n_nodes, e1_pos, e2_pos = build_mesh(p, t_fat, elec_r, run_dir, coarse=coarse)
            print(f"    {n_nodes} nodes")

            print("  ElmerGrid ...")
            elmer_dir = run_dir / "elmer_mesh"
            elmer_dir.mkdir(exist_ok=True)
            _run(["ElmerGrid", "14", "2", "mesh.msh", "-out", "elmer_mesh"],
                 cwd=run_dir, label="ElmerGrid")

            print("  detecting electrode BCs ...")
            e1_id, e2_id = detect_elec_bc_ids(elmer_dir, e1_pos, e2_pos, p["geometry"]["Lz"])
            print(f"    active={e1_id}  return={e2_id}")

            write_sif(run_dir, e1_id, e2_id, p)
            (run_dir / "results").mkdir(exist_ok=True)

            print("  ElmerSolver ...")
            _run(["ElmerSolver", "case.sif"], cwd=run_dir, label="ElmerSolver")

            print("  extracting metrics ...")
            res = extract_results(run_dir, p, t_fat, elec_r, e1_pos, e2_pos,
                                  p["electrodes"]["shape"])
            print(f"    peak_J_skin={res['peak_J_skin']:.2f}  "
                  f"mean_J_roi={res['mean_J_roi']:.4f}  "
                  f"tradeoff={res['tradeoff']:.4f}  "
                  f"flux_err={res['flux_err']:.2e}")
            all_results.append(res)

    return all_results


def save_results(all_results):
    # CSV
    csv_path = RESULTS_DIR / "summary.csv"
    if all_results:
        keys = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved → {csv_path}")

    # JSON
    json_path = RESULTS_DIR / "summary.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved → {json_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ankle layered slab sweep")
    parser.add_argument("--smoke", action="store_true",
                        help="Single coarse case for quick pipeline check")
    args = parser.parse_args()

    p = load_params()

    if args.smoke:
        t_fat_list = [p["layers"]["t_fat"]]
        elec_r_list = [p["electrodes"]["size_list"][1]]   # middle size
        coarse = True
        print("=== SMOKE TEST (1 coarse case) ===")
    else:
        t_fat_list  = p["layers"]["t_fat_sweep"]
        elec_r_list = p["electrodes"]["size_list"]
        coarse = False
        print(f"=== FULL SWEEP: {len(t_fat_list)} fat thicknesses × "
              f"{len(elec_r_list)} electrode sizes = "
              f"{len(t_fat_list)*len(elec_r_list)} cases ===")

    results = run_sweep(p, t_fat_list, elec_r_list, coarse=coarse)
    save_results(results)

    print("\nAll done. Run plot_layered_results.py to generate figures.")
