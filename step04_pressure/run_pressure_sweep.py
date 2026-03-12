"""
run_pressure_sweep.py — pressure-dependent contact impedance sweep
==================================================================
Models the effect of electrode wrap tightness on:
  - Contact impedance (compliance voltage)
  - Charge density at the skin surface
  - E-field at the tibial nerve (ROI)

Sweep axis: sigma_contact (S/m), representing wrap pressure from loose to tight.
Fixed: t_fat=5mm, elec_r=10mm, I=5mA @ 10Hz 200µs.

Key efficiency: the mesh is built ONCE and reused across all pressure levels
(only sigma_contact changes between cases — no geometry changes).

Usage (from step04_pressure/):
    python3 run_pressure_sweep.py          # full sweep (5 pressure levels)
    python3 run_pressure_sweep.py --smoke  # 1 coarse case

Outputs:
    results/<pressure_label>/  with VTU + case.sif
    results/summary.csv  + results/summary.json
"""

import sys
import argparse
import subprocess
import csv
import json
import shutil
import yaml
import numpy as np
import pyvista as pv
import gmsh
from pathlib import Path

PARAMS_FILE = Path(__file__).parent / "params.yaml"
RESULTS_DIR = Path(__file__).parent / "results"


# ── Load parameters ────────────────────────────────────────────────────────────
def load_params():
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def _pl(p):
    return p.get("placement", p.get("electrodes", {}))


def _stim(p):
    return p.get("stim", p.get("control", {}))


# ── 1. Build gmsh mesh (rect slab, one contact layer per electrode) ────────────
def build_mesh(p, run_dir, coarse=False):
    """
    Build the ankle slab mesh with fixed t_fat and elec_r from params.
    Contact layer sigma is NOT embedded in the mesh — only geometry is built here.
    Reuse this mesh for all pressure levels by copying elmer_mesh/.

    Returns: (n_nodes, e1_pos3d, e2_pos3d, body_info)
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    g   = p["geometry"]
    Lx, Ly, Lz = g["Lx"], g["Ly"], g["Lz"]
    ls  = p["layers"]
    t_skin   = ls["t_skin"]
    t_fat    = ls["t_fat"]
    t_muscle = Lz - t_skin - t_fat
    if t_muscle <= 1e-4:
        raise ValueError(f"t_muscle={t_muscle*1000:.2f} mm <= 0.1 mm")

    pl     = _pl(p)
    shape  = pl.get("electrode_shape", "circle")
    e1x, e1y = float(pl["active_xy"][0]), float(pl["active_xy"][1])
    e2x, e2y = float(pl["return_xy"][0]),  float(pl["return_xy"][1])
    elec_r   = float(pl["electrode_r_mm"]) * 1e-3

    ct = p["contact"]
    t_contact = ct["t_contact_mm"] * 1e-3

    m       = p["mesh"]
    scale   = 2.0 if coarse else 1.0
    lc_elec = m["lc_electrode_mm"] * 1e-3 * scale
    lc_bulk = m["lc_global_mm"]    * 1e-3 * scale
    lc_min  = m["lc_skin_min"]     * 1e-3

    z0_fat     = t_muscle
    z0_skin    = t_muscle + t_fat
    z_skin_top = Lz
    z_e1_skin  = Lz   # rect: flat surface
    z_e2_skin  = Lz
    z_e1_top   = Lz + t_contact
    z_e2_top   = Lz + t_contact
    z_elec_top = z_e1_top

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("ankle_pressure")

    def _rect_vol(z_bot, height, lc):
        pts = [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]
        pt_tags = [gmsh.model.occ.addPoint(x, y, z_bot, lc) for x, y in pts]
        lines = [gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i+1) % 4])
                 for i in range(4)]
        loop = gmsh.model.occ.addCurveLoop(lines)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        result = gmsh.model.occ.extrude([(2, surf)], 0, 0, height)
        return next(dt[1] for dt in result if dt[0] == 3)

    def _contact_vol(xc, yc, z_bot, height, r, lc):
        if shape == "circle":
            base = gmsh.model.occ.addDisk(xc, yc, z_bot, r, r)
        else:
            base = gmsh.model.occ.addRectangle(xc - r, yc - r, z_bot, 2*r, 2*r)
        result = gmsh.model.occ.extrude([(2, base)], 0, 0, height)
        return next(dt[1] for dt in result if dt[0] == 3)

    vol_musc = _rect_vol(0,      t_muscle, lc_bulk)
    vol_fat  = _rect_vol(z0_fat, t_fat,    lc_bulk)
    vol_skin = _rect_vol(z0_skin, t_skin,  lc_min)
    vol_c1   = _contact_vol(e1x, e1y, z_skin_top, t_contact, elec_r, lc_elec)
    vol_c2   = _contact_vol(e2x, e2y, z_skin_top, t_contact, elec_r, lc_elec)

    all_vols = [(3, v) for v in (vol_musc, vol_fat, vol_skin, vol_c1, vol_c2)]
    gmsh.model.occ.fragment(all_vols, [])
    gmsh.model.occ.synchronize()

    def _z_mid(dim, tag):
        b = gmsh.model.getBoundingBox(dim, tag)
        return (b[2] + b[5]) / 2.0

    def _xy_cen(dim, tag):
        b = gmsh.model.getBoundingBox(dim, tag)
        return np.array([(b[0]+b[3])/2, (b[1]+b[4])/2])

    vols    = gmsh.model.getEntities(3)
    musc_v  = min(vols, key=lambda dt: abs(_z_mid(*dt) - t_muscle / 2))
    fat_v   = min(vols, key=lambda dt: abs(_z_mid(*dt) - (z0_fat + t_fat / 2)))
    skin_v  = min(vols, key=lambda dt: abs(_z_mid(*dt) - (z0_skin + t_skin / 2)))
    cands   = [v for v in vols if v not in (musc_v, fat_v, skin_v)]
    c1_v    = min(cands, key=lambda dt: np.linalg.norm(_xy_cen(*dt) - [e1x, e1y]))
    c2_v    = min([v for v in cands if v != c1_v],
                  key=lambda dt: np.linalg.norm(_xy_cen(*dt) - [e2x, e2y]))

    # Electrode BC surfaces: top of each contact disk
    surfs = gmsh.model.getEntities(2)

    def _surf_z_mid(dim, tag):
        b = gmsh.model.getBoundingBox(dim, tag)
        return (b[2] + b[5]) / 2.0

    def _surf_xy_cen(dim, tag):
        b = gmsh.model.getBoundingBox(dim, tag)
        return np.array([(b[0]+b[3])/2, (b[1]+b[4])/2])

    def _pick_elec_surf(candidates, pos2d, r_xy):
        near = [dt for dt in candidates
                if np.linalg.norm(_surf_xy_cen(*dt) - pos2d) < r_xy]
        pool = near if near else candidates
        return max(pool, key=lambda dt: _surf_z_mid(*dt))

    tol_z     = max(z_elec_top * 2e-2, 1e-4)
    top_surfs = [dt for dt in surfs if _surf_z_mid(*dt) >= z_elec_top - tol_z]
    if not top_surfs:
        top_surfs = sorted(surfs, key=lambda dt: -_surf_z_mid(*dt))[:4]

    e1_s = _pick_elec_surf(top_surfs, np.array([e1x, e1y]), elec_r * 2)
    e2_s = _pick_elec_surf([s for s in top_surfs if s != e1_s],
                           np.array([e2x, e2y]), elec_r * 2)
    other_s = [dt for dt in surfs if dt not in (e1_s, e2_s)]

    gmsh.model.addPhysicalGroup(3, [musc_v[1]], 1, name="muscle")
    gmsh.model.addPhysicalGroup(3, [fat_v[1]],  2, name="fat")
    gmsh.model.addPhysicalGroup(3, [skin_v[1]], 3, name="skin")
    gmsh.model.addPhysicalGroup(3, [c1_v[1]],   4, name="contact_active")
    gmsh.model.addPhysicalGroup(3, [c2_v[1]],   5, name="contact_return")
    gmsh.model.addPhysicalGroup(2, [e1_s[1]],            101, name="active_electrode")
    gmsh.model.addPhysicalGroup(2, [e2_s[1]],            102, name="return_electrode")
    gmsh.model.addPhysicalGroup(2, [s[1] for s in other_s], 103, name="insulated")

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", [e1_s[1], e2_s[1]])
    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "InField",  f_dist)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin",  lc_elec)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMax",  lc_bulk)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin",  elec_r)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax",  elec_r * 6)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)

    gmsh.model.mesh.generate(3)
    msh = run_dir / "mesh.msh"
    gmsh.write(str(msh))
    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()

    body_info = {
        "contact_enabled": True,
        "z_skin_top":    z_skin_top,
        "z_elec_top":    z_elec_top,
        "z_e1_skin":     z_e1_skin,
        "z_e2_skin":     z_e2_skin,
        "z_e1_elec_top": z_e1_top,
        "z_e2_elec_top": z_e2_top,
        "c1_body_id":    4,
        "c2_body_id":    5,
        "elec_shape":    shape,
    }
    return (n_nodes,
            np.array([e1x, e1y, z_e1_top]),
            np.array([e2x, e2y, z_e2_top]),
            body_info)


# ── 2. Detect Elmer BC IDs ─────────────────────────────────────────────────────
def detect_elec_bc_ids(elmer_mesh_dir, e1_pos, e2_pos, z_e1_top, z_e2_top):
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

    z_floor = min(z_e1_top, z_e2_top) - 5e-3

    def _elem_area(nids):
        coords = [nodes[n] for n in nids if n in nodes]
        if len(coords) < 3:
            return 0.0
        v0, v1, v2 = coords[0], coords[1], coords[2]
        a = 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))
        if len(coords) == 4:
            v3 = coords[3]
            a += 0.5 * float(np.linalg.norm(np.cross(v2 - v0, v3 - v0)))
        return a

    etype_nn = {202: 2, 303: 3, 306: 6, 404: 4}
    bc_centers   = {}
    bc_z_centers = {}
    bc_elem_nids = {}
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
            nn    = etype_nn.get(etype, 3)
            try:
                nids = [int(p) for p in parts[5:5+nn]]
            except ValueError:
                continue
            coords = [nodes[n] for n in nids if n in nodes]
            if len(coords) < 2:
                continue
            zvals = [c[2] for c in coords]
            if max(zvals) < z_floor:
                continue
            cxy = np.mean([c[:2] for c in coords], axis=0)
            cz  = float(np.mean(zvals))
            bc_centers.setdefault(bcid, []).append(cxy)
            bc_z_centers.setdefault(bcid, []).append(cz)
            bc_elem_nids.setdefault(bcid, []).append(nids)

    bc_mean   = {bid: np.mean(pts, axis=0) for bid, pts in bc_centers.items() if pts}
    bc_z_mean = {bid: float(np.mean(zs))   for bid, zs  in bc_z_centers.items() if zs}

    def _find_elec_bc(pos_e, z_e_top, exclude=None):
        tol_z_e = max(z_e_top * 2e-2, 5e-4)
        candidates = {bid: cxy for bid, cxy in bc_mean.items()
                      if bid != exclude
                      and abs(bc_z_mean.get(bid, 0) - z_e_top) < tol_z_e}
        if not candidates:
            candidates = {bid: cxy for bid, cxy in bc_mean.items() if bid != exclude}
        return min(candidates, key=lambda b: np.linalg.norm(candidates[b] - pos_e[:2]))

    e1_id = _find_elec_bc(e1_pos, z_e1_top)
    e2_id = _find_elec_bc(e2_pos, z_e2_top, exclude=e1_id)
    e1_area = sum(_elem_area(nids) for nids in bc_elem_nids.get(e1_id, []))
    e2_area = sum(_elem_area(nids) for nids in bc_elem_nids.get(e2_id, []))
    return e1_id, e2_id, e1_area, e2_area


# ── 3. Write Elmer SIF ─────────────────────────────────────────────────────────
_SIF_HEADER = """\
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
"""


def write_sif(run_dir, e1_id, e2_id, p, elec_r, body_info,
              sigma_contact_override, elec_area_mesh=None):
    c   = p["conductivities"]
    sv  = p["solver"]
    st  = _stim(p)
    I_A = st["injected_current_mA"] * 1e-3

    area_analytic = np.pi * elec_r**2
    area = elec_area_mesh if (elec_area_mesh and elec_area_mesh > 0) else area_analytic
    jn   = I_A / area

    bcs = f"""
Boundary Condition 1
  Name = "active_electrode"
  Target Boundaries = {e1_id}
  Current Density = {jn:.6e}   ! I={I_A*1e3:.1f}mA, A_mesh={area*1e4:.4f}cm²
End

Boundary Condition 2
  Name = "return_electrode"
  Target Boundaries = {e2_id}
  Potential = 0.0
End
"""

    bodies = f"""
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

Body 4
  Name = "contact_active"
  Target Bodies(1) = 4
  Equation = 1
  Material = 4
End

Body 5
  Name = "contact_return"
  Target Bodies(1) = 5
  Equation = 1
  Material = 4
End
"""

    materials = f"""
Material 1
  Name = "muscle"
  Electric Conductivity = {c["sigma_muscle"]}
End

Material 2
  Name = "fat"
  Electric Conductivity = {c["sigma_fat"]}
End

Material 3
  Name = "skin"
  Electric Conductivity = {c["sigma_skin"]}
End

Material 4
  Name = "contact"
  Electric Conductivity = {sigma_contact_override}   ! pressure-dependent — PLACEHOLDER
End
"""

    sif = (_SIF_HEADER.format(tol=sv["tolerance"], lin_solver=sv["linear_solver"])
           + bodies + materials + bcs)
    (run_dir / "case.sif").write_text(sif)
    return jn


# ── 4. Run shell commands ──────────────────────────────────────────────────────
def _run(cmd, cwd, label):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR in {label}:")
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        sys.exit(1)


# ── 5a. Surface flux integral ──────────────────────────────────────────────────
def compute_injected_current(mesh, e1_pos3d, e2_pos3d, elec_r,
                             z_e1_top, z_e2_top, elec_shape="circle"):
    ctypes  = mesh.celltypes
    bnd_ids = np.where(np.isin(ctypes, [5, 9]))[0]
    if len(bnd_ids) == 0:
        bnd_mesh = mesh.extract_surface(algorithm="dataset_surface")
    else:
        bnd_mesh = mesh.extract_cells(bnd_ids)

    bnd_cd  = bnd_mesh.point_data_to_cell_data()
    J_cells = np.array(bnd_cd.cell_data["volume current"])
    c_pts   = np.array(bnd_mesh.cell_centers().points)
    areas   = np.array(bnd_mesh.compute_cell_sizes().cell_data["Area"])

    tol = 0.2

    def _mask(pos, z_top):
        tol_z = max(z_top * 5e-3, 1e-5)
        m     = c_pts[:, 2] > z_top - tol_z
        dx    = c_pts[:, 0] - pos[0]
        dy    = c_pts[:, 1] - pos[1]
        return m & (np.sqrt(dx**2 + dy**2) < elec_r * (1 + tol))

    act_m = _mask(e1_pos3d, z_e1_top)
    ret_m = _mask(e2_pos3d, z_e2_top)
    if not act_m.any() or not ret_m.any():
        return np.nan, np.nan, np.nan, np.nan, np.nan

    I_act_s = float(np.sum(J_cells[act_m, 2] * areas[act_m]))
    I_ret_s = float(np.sum(J_cells[ret_m, 2] * areas[ret_m]))
    I_act   = abs(I_act_s)
    I_ret   = abs(I_ret_s)
    denom   = max(I_act, I_ret)
    flux_err = float(abs(I_act_s + I_ret_s) / denom) if denom > 0 else np.nan
    return I_act, I_ret, flux_err, I_act_s, I_ret_s


# ── 5b. ROI evaluation ────────────────────────────────────────────────────────
def eval_roi(mesh, roi_cen, roi_radius_init, min_cells=4):
    mesh_cd    = mesh.point_data_to_cell_data()
    J_cells    = np.array(mesh_cd.cell_data["volume current"])
    Jmag_cells = np.linalg.norm(J_cells, axis=1)

    try:
        scalar_name = next(
            (s for s in ("potential", "Potential") if s in mesh_cd.array_names), None)
        if scalar_name is None:
            raise KeyError("Potential not found")
        grad_mesh  = mesh_cd.compute_derivative(scalars=scalar_name)
        Emag_cells = np.linalg.norm(
            np.array(grad_mesh.cell_data["gradient"]), axis=1)
    except Exception:
        Emag_cells = None

    cell_pts = np.array(mesh_cd.cell_centers().points)
    dist     = np.linalg.norm(cell_pts - roi_cen, axis=1)

    roi_r = roi_radius_init
    warning = None
    for mult in [1.0, 1.5, 2.0, 3.0]:
        r_test = roi_radius_init * mult
        mask   = dist < r_test
        if mask.sum() >= min_cells:
            roi_r = r_test
            if mult > 1.0:
                warning = f"ROI radius expanded {mult:.1f}x to {r_test*1000:.1f} mm"
            break
    else:
        roi_r  = roi_radius_init * 3.0
        mask   = dist < roi_r
        warning = f"ROI at 3x only {mask.sum()} cells — noisy"

    n = int(mask.sum())
    if n == 0:
        return np.nan, np.nan, 0, roi_r, "No cells in ROI"

    mean_J = float(Jmag_cells[mask].mean())
    mean_E = float(Emag_cells[mask].mean()) if Emag_cells is not None else np.nan
    return mean_J, mean_E, n, roi_r, warning


# ── 5c. Extract metrics from VTU ──────────────────────────────────────────────
def extract_results(run_dir, p, sigma_contact, pressure_label,
                    e1_pos3d, e2_pos3d, body_info, jn_used, elec_area_mesh):
    vtu_path = run_dir / "results" / "case_t0001.vtu"
    if not vtu_path.exists():
        raise FileNotFoundError(f"VTU not found: {vtu_path}")

    mesh = pv.read(str(vtu_path))
    pts  = np.array(mesh.points)
    J    = np.array(mesh.point_data["volume current"])
    Jmag = np.linalg.norm(J, axis=1)

    g  = p["geometry"]
    ls = p["layers"]
    st = _stim(p)
    pl = _pl(p)
    elec_r    = float(pl["electrode_r_mm"]) * 1e-3
    elec_shape = body_info["elec_shape"]

    z_skin_top = body_info["z_skin_top"]
    z_e1_top   = body_info["z_e1_elec_top"]
    z_e2_top   = body_info["z_e2_elec_top"]
    t_skin     = ls["t_skin"]
    t_fat      = ls["t_fat"]
    z0_skin    = z_skin_top - t_skin

    # Peak |J| at skin surface
    skin_mask = pts[:, 2] > z0_skin + t_skin * 0.80
    peak_J_with = float(Jmag[skin_mask].max()) if skin_mask.any() else np.nan

    if skin_mask.any():
        xp = pts[skin_mask, 0]
        yp = pts[skin_mask, 1]
        Jm = Jmag[skin_mask]
        outside = ~((np.sqrt((xp - e1_pos3d[0])**2 + (yp - e1_pos3d[1])**2) < elec_r) |
                    (np.sqrt((xp - e2_pos3d[0])**2 + (yp - e2_pos3d[1])**2) < elec_r))
        peak_J_no = float(Jm[outside].max()) if outside.any() else peak_J_with
    else:
        peak_J_no = np.nan

    # Injected current
    (I_act, I_ret, flux_err,
     I_act_s, I_ret_s) = compute_injected_current(
        mesh, e1_pos3d, e2_pos3d, elec_r, z_e1_top, z_e2_top, elec_shape)
    print(f"    I_active={I_act:.4e} A  I_return={I_ret:.4e} A  flux_err={flux_err:.2e}")

    I_target = st["injected_current_mA"] * 1e-3
    if np.isfinite(I_act):
        dev = abs(I_act - I_target) / I_target
        if dev > 0.02:
            print(f"    *** CURRENT ERROR > 2%: {I_act*1e3:.3f} mA vs target "
                  f"{I_target*1e3:.1f} mA ({dev:.1%}) ***")

    # Compliance voltage
    phi_key = next((k for k in ("Potential", "potential")
                    if k in mesh.point_data), None)
    compliance_V = np.nan
    if phi_key:
        phi = np.array(mesh.point_data[phi_key])
        tol_z = max(z_e1_top * 5e-3, 1e-5)
        act_n = (pts[:, 2] > z_e1_top - tol_z) & (
            np.sqrt((pts[:, 0] - e1_pos3d[0])**2 +
                    (pts[:, 1] - e1_pos3d[1])**2) < elec_r * 1.5)
        ret_n = (pts[:, 2] > z_e2_top - tol_z) & (
            np.sqrt((pts[:, 0] - e2_pos3d[0])**2 +
                    (pts[:, 1] - e2_pos3d[1])**2) < elec_r * 1.5)
        if act_n.any():
            V_act = float(phi[act_n].mean())
            V_ret = float(phi[ret_n].mean()) if ret_n.any() else 0.0
            compliance_V = V_act - V_ret

    cmp_lim = st.get("compliance_voltage_V", 200.0)
    exceeded = bool(np.isfinite(compliance_V) and compliance_V > cmp_lim)
    if exceeded:
        print(f"    WARNING: compliance_V={compliance_V:.1f} V > limit {cmp_lim:.0f} V")

    # Contact impedance (ohm): V / I, represents total path from active → return
    contact_Z = float(compliance_V / I_act) if (
        np.isfinite(compliance_V) and np.isfinite(I_act) and I_act > 0) else np.nan

    # ROI
    r_cfg   = p["roi"]
    z_nerve = z_skin_top - r_cfg["z_target"]
    roi_cen = np.array([e1_pos3d[0], e1_pos3d[1], z_nerve])
    mean_J_roi, mean_E_roi, roi_n, roi_r, roi_warn = eval_roi(
        mesh, roi_cen, r_cfg["roi_radius"])
    if roi_warn:
        print(f"    ROI: {roi_warn}")

    # Charge density (mC/cm²) from peak J at skin (under electrode)
    # Q/A = J * pulse_width_s  [C/m²]  → convert to mC/cm²: * 1e-4 * 1e3 = * 1e-7 ... wait
    # C/m² → C/cm²: divide by 1e4   → mC/cm²: multiply by 1e3
    # net: * (1e3 / 1e4) = * 0.1
    pw_us = st.get("pulse_width_us", 200.0)
    pw_s  = pw_us * 1e-6
    # peak_J_with includes electrode footprint — that's the highest density point
    charge_density = float(peak_J_with * pw_s * 0.1) if np.isfinite(peak_J_with) else np.nan
    safety_limit   = p.get("safety", {}).get("charge_density_limit_mC_cm2", 1.0)
    exceeds_charge = bool(np.isfinite(charge_density) and charge_density > safety_limit)

    # Efficiency
    efficiency = (float(mean_E_roi) / peak_J_no
                  if (np.isfinite(mean_E_roi) and peak_J_no > 0) else np.nan)

    def _r(val, n):
        v = float(val)
        return round(v, n) if np.isfinite(v) else v

    return {
        "pressure_label":           pressure_label,
        "sigma_contact_Spm":        sigma_contact,
        "elec_r_mm":                float(pl["electrode_r_mm"]),
        "t_fat_mm":                 ls["t_fat"] * 1000,
        "compliance_V":             _r(compliance_V, 3),
        "contact_impedance_ohm":    _r(contact_Z, 1),
        "exceeded_compliance":      exceeded,
        "I_active_A":               _r(I_act, 8),
        "I_return_A":               _r(I_ret, 8),
        "I_active_signed_A":        _r(I_act_s, 8),
        "I_return_signed_A":        _r(I_ret_s, 8),
        "flux_err":                 _r(flux_err, 6),
        "jn_used_A_m2":             _r(jn_used, 6),
        "peak_J_skin_with_elec":    _r(peak_J_with, 4),
        "peak_J_skin_no_elec":      _r(peak_J_no,   4),
        "charge_density_mC_cm2":    _r(charge_density, 6),
        "exceeds_charge_limit":     exceeds_charge,
        "roi_mean_J":               _r(mean_J_roi, 6),
        "roi_mean_E":               _r(mean_E_roi, 4),
        "efficiency":               _r(efficiency, 6),
        "roi_n_cells":              roi_n,
        "roi_radius_used_mm":       _r(roi_r * 1000, 2),
        "pulse_width_us":           pw_us,
        "frequency_Hz":             st.get("frequency_Hz", 10.0),
    }


# ── 6. Main pressure sweep ─────────────────────────────────────────────────────
def run_pressure_sweep(p, sigma_contact_list, pressure_labels, coarse=False):
    """
    Build mesh ONCE then loop over sigma_contact levels (pressure).
    Reuses the same Elmer mesh — only the SIF changes between cases.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    pl     = _pl(p)
    st     = _stim(p)
    elec_r = float(pl["electrode_r_mm"]) * 1e-3
    I_mA   = st["injected_current_mA"]

    print(f"\n{'='*60}")
    print(f"  PRESSURE SWEEP — sigma_contact vs compliance/charge/ROI")
    print(f"  Fixed: t_fat={p['layers']['t_fat']*1000:.0f}mm  "
          f"r={pl['electrode_r_mm']:.0f}mm  I={I_mA:.1f}mA  "
          f"freq={st.get('frequency_Hz',10):.0f}Hz  "
          f"pw={st.get('pulse_width_us',200):.0f}µs")
    print(f"  {len(sigma_contact_list)} pressure level(s): "
          + ", ".join(f"{lbl}({s:.4f})" for s, lbl
                      in zip(sigma_contact_list, pressure_labels)))
    print(f"{'='*60}\n")

    # ── Build mesh ONCE ────────────────────────────────────────────────────────
    mesh_dir = RESULTS_DIR / "_mesh_base"
    print("  Building mesh (shared for all pressure levels)...")
    n_nodes, e1_pos, e2_pos, body_info = build_mesh(p, mesh_dir, coarse=coarse)
    print(f"    {n_nodes} nodes")

    print("  ElmerGrid ...")
    elmer_dir = mesh_dir / "elmer_mesh"
    elmer_dir.mkdir(exist_ok=True)
    _run(["ElmerGrid", "14", "2", "mesh.msh", "-out", "elmer_mesh"],
         cwd=mesh_dir, label="ElmerGrid")

    print("  Detecting electrode BCs ...")
    e1_id, e2_id, A_active, A_return = detect_elec_bc_ids(
        elmer_dir, e1_pos, e2_pos, e1_pos[2], e2_pos[2])
    area_analytic = np.pi * elec_r**2
    print(f"    active={e1_id}  return={e2_id}  "
          f"A_active={A_active*1e4:.4f}cm²  "
          f"A_analytic={area_analytic*1e4:.4f}cm²")

    # ── Loop over pressure levels ──────────────────────────────────────────────
    all_results = []
    for sigma_c, label in zip(sigma_contact_list, pressure_labels):
        print(f"\n[{label}]  sigma_contact={sigma_c:.4f} S/m")

        run_dir = RESULTS_DIR / label
        run_dir.mkdir(exist_ok=True)

        # Copy elmer mesh (SIF references it by relative path)
        run_elmer = run_dir / "elmer_mesh"
        if run_elmer.exists():
            shutil.rmtree(run_elmer)
        shutil.copytree(elmer_dir, run_elmer)

        jn_used = write_sif(run_dir, e1_id, e2_id, p, elec_r, body_info,
                            sigma_contact_override=sigma_c,
                            elec_area_mesh=A_active)
        (run_dir / "results").mkdir(exist_ok=True)

        print("  ElmerSolver ...")
        _run(["ElmerSolver", "case.sif"], cwd=run_dir, label="ElmerSolver")

        print("  Extracting metrics ...")
        res = extract_results(run_dir, p, sigma_c, label,
                              e1_pos, e2_pos, body_info, jn_used, A_active)

        print(f"    compliance_V={res['compliance_V']:.1f} V  "
              f"Z_contact={res['contact_impedance_ohm']:.0f} Ω  "
              f"charge={res['charge_density_mC_cm2']:.5f} mC/cm²  "
              f"roi_E={res['roi_mean_E']:.2f} V/m")

        all_results.append(res)

    return all_results


# ── 7. Save results ────────────────────────────────────────────────────────────
def save_results(all_results):
    if not all_results:
        return
    csv_path = RESULTS_DIR / "summary.csv"
    keys = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved → {csv_path}")

    json_path = RESULTS_DIR / "summary.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
    print(f"Saved → {json_path}")


def print_run_summary(results):
    print(f"\n{'='*60}")
    print("  RUN COMPLETE")
    print(f"{'='*60}")
    print(f"  results/summary.csv / summary.json")
    print(f"  {len(results)} pressure level(s) computed")
    print(f"\n  {'Label':<10} {'sigma_c':>10}  {'V_compliance':>14}  "
          f"{'Z(Ω)':>10}  {'Q(mC/cm²)':>12}  {'E_roi(V/m)':>12}")
    print("  " + "-" * 74)
    for r in results:
        flag = " [!]" if r.get("exceeds_charge_limit") else ""
        print(f"  {r['pressure_label']:<10} {r['sigma_contact_Spm']:>10.4f}  "
              f"{r['compliance_V']:>14.1f}  "
              f"{r['contact_impedance_ohm']:>10.0f}  "
              f"{r['charge_density_mC_cm2']:>12.5f}{flag}  "
              f"{r['roi_mean_E']:>12.2f}")
    print(f"{'='*60}")
    print("  Run plot_pressure_results.py to generate figures.\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pressure-dependent contact sweep")
    parser.add_argument("--smoke", action="store_true",
                        help="Single coarse case (middle pressure level)")
    args = parser.parse_args()

    p   = load_params()
    ps  = p["pressure_sweep"]
    sig_list = ps["sigma_contact_Spm"]
    lbl_list = ps["labels"]

    if args.smoke:
        mid = len(sig_list) // 2
        sig_list = [sig_list[mid]]
        lbl_list = [lbl_list[mid]]
        print("=== SMOKE TEST (1 coarse case) ===")

    results = run_pressure_sweep(p, sig_list, lbl_list, coarse=args.smoke)
    save_results(results)
    print_run_summary(results)
