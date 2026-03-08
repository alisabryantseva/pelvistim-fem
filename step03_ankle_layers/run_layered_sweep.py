"""
run_layered_sweep.py — ankle cross-section layered model, bipolar electrode sweep
==================================================================================
Geometry:
  Approximate ankle cross-section (12-point polygon or rectangle) extruded to
  depth Lz. Three tissue layers in z: skin / fat / muscle.
  Optional thin contact-material volumes at electrode positions model the
  electrode–skin interface for dry / reusable electrodes without gel.

Electrode placement:
  Active  — medial groove (low x, mid y): between tendon and medial malleolus.
  Return  — posterior-lateral (high x, upper y): behind lateral malleolus.

Usage (from step03_ankle_layers/):
    python3 run_layered_sweep.py             # full sweep
    python3 run_layered_sweep.py --smoke     # 1 coarse case

Outputs:
    results/<case>/  with VTU + case.sif
    results/summary.csv  + results/summary.json

Summary metrics:
  peak_J_skin_with_elec  — max |J| at skin surface including electrode footprint
  peak_J_skin_no_elec    — max |J| at skin surface EXCLUDING electrode footprint
  roi_mean_E             — mean |E| in ROI sphere  [V/m]
  roi_mean_J             — mean |J| in ROI sphere  [A/m²]
  efficiency             — roi_mean_E / peak_J_skin_no_elec  [m]
  total_current_A        — surface integral of J·n over active electrode
  jn_used                — current density applied at active electrode (current mode, A/m²)
  compliance_V           — mean(V_active) − mean(V_return)  (current mode)
  exceeded_compliance    — True if compliance_V > compliance_voltage_V limit
  I_return_A             — surface-integral |J| at return electrode
  flux_err               — |I_active + I_return| / max(|I_active|,|I_return|)  (signed KCL)
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


# ── Load parameters ────────────────────────────────────────────────────────────
def load_params():
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def _pl(p):
    """Return the placement sub-dict (supports old 'electrodes' key too)."""
    return p.get("placement", p.get("electrodes", {}))


def _stim(p):
    """Return the stim sub-dict (supports old 'control' key too)."""
    return p.get("stim", p.get("control", {}))


# ── 1a. Ankle cross-section outline ───────────────────────────────────────────
def ankle_outline_pts(Lx, Ly):
    """
    12-point polygon approximating an ankle cross-section.
    x = medial (0) → lateral (Lx).  y = anterior (0) → posterior (Ly).

    Active electrode target: medial groove  (low x, mid y  ~P10).
    Return electrode target: posterior-lateral             (~P5).
    """
    frac = [
        (0.25, 0.02),   # P0  anterior-medial
        (0.50, 0.00),   # P1  anterior center
        (0.75, 0.02),   # P2  anterior-lateral
        (0.97, 0.22),   # P3  lateral-anterior
        (1.00, 0.47),   # P4  lateral-mid
        (0.93, 0.72),   # P5  posterior-lateral  ← return electrode
        (0.75, 0.97),   # P6  posterior-lateral end
        (0.50, 1.00),   # P7  posterior center
        (0.25, 0.97),   # P8  posterior-medial end
        (0.07, 0.72),   # P9  medial-posterior
        (0.02, 0.47),   # P10 medial groove      ← active electrode
        (0.07, 0.22),   # P11 medial-anterior
    ]
    return [(fx * Lx, fy * Ly) for fx, fy in frac]


def _ankle_z_top(x, y, Lx, Ly, Lz):
    """
    Anatomical skin surface height z_top(x, y) for the ankle.
    Nominal height = Lz (flat baseline) plus smooth Gaussian bumps/dips:

      Medial groove  (P10, low x, mid y) : −4 mm  ← active electrode sits here
      Achilles       (P7,  mid x, high y): +3 mm  ← Achilles tendon protrudes
      Lateral malle. (P4,  high x, mid y): +2 mm  ← bony prominence
    """
    xn = x / Lx   # 0 = medial, 1 = lateral
    yn = y / Ly   # 0 = anterior, 1 = posterior
    h  = Lz

    # Medial groove: 4 mm dip at P10 (0.02, 0.47)
    r2 = ((xn - 0.02) / 0.12)**2 + ((yn - 0.47) / 0.18)**2
    h -= 0.004 * np.exp(-r2)

    # Achilles tendon: 3 mm bump at P7 (0.50, 0.97)
    r2 = ((xn - 0.50) / 0.18)**2 + ((yn - 0.97) / 0.07)**2
    h += 0.003 * np.exp(-r2)

    # Lateral malleolus: 2 mm bump at P4 (0.97, 0.47)
    r2 = ((xn - 0.97) / 0.07)**2 + ((yn - 0.47) / 0.18)**2
    h += 0.002 * np.exp(-r2)

    return float(h)


# ── 1b. Build gmsh mesh ────────────────────────────────────────────────────────
def build_mesh(p, t_fat, elec_r, run_dir, coarse=False):
    """
    Create ankle (or rect) cross-section mesh with 3 tissue layers.
    Optional contact-layer volumes at each electrode position.

    Physical groups:
        Vol  1 = muscle,  2 = fat,  3 = skin
        Vol  4 = contact-active,  5 = contact-return  (if contact.enabled)
        Surf 101 = active electrode BC surface
        Surf 102 = return electrode BC surface
        Surf 103 = insulated (all other outer surfaces)

    Returns:
        n_nodes   — total mesh nodes
        e1_pos3d  — active electrode center  [x, y, z_elec_top]
        e2_pos3d  — return electrode center  [x, y, z_elec_top]
        body_info — dict with geometry/contact metadata for downstream use
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    g  = p["geometry"]
    Lx, Ly, Lz = g["Lx"], g["Ly"], g["Lz"]
    ls = p["layers"]
    t_skin   = ls["t_skin"]
    t_muscle = Lz - t_skin - t_fat
    if t_muscle <= 1e-4:
        raise ValueError(
            f"t_muscle = {t_muscle*1000:.2f} mm ≤ 0.1 mm — "
            f"reduce t_fat + t_skin or increase Lz")

    pl    = _pl(p)
    shape = pl.get("electrode_shape", pl.get("shape", "circle"))
    active_xy = pl.get("active_xy",
                       [pl.get("medial_offset",  0.025), Ly / 2])
    return_xy  = pl.get("return_xy",
                        [Lx - pl.get("lateral_offset", 0.025), Ly / 2])
    e1x, e1y = float(active_xy[0]), float(active_xy[1])
    e2x, e2y = float(return_xy[0]),  float(return_xy[1])

    ct = p.get("contact", {})
    contact_enabled = ct.get("enabled", False)
    t_contact = ct.get("t_contact_mm", 0.5) * 1e-3 if contact_enabled else 0.0

    m = p.get("mesh", {})
    scale    = 2.0 if coarse else 1.0
    lc_elec  = m.get("lc_electrode_mm", elec_r * 300) * 1e-3 * scale
    lc_bulk  = m.get("lc_global_mm",    3.0)           * 1e-3 * scale
    lc_min   = m.get("lc_skin_min",     0.5)           * 1e-3

    z0_fat     = t_muscle
    z0_skin    = t_muscle + t_fat
    z_skin_top = Lz
    cross      = g.get("cross_section", "rect")

    # Per-electrode skin surface z (anatomical bumps/grooves for ankle cross-section)
    z_e1_skin     = _ankle_z_top(e1x, e1y, Lx, Ly, Lz) if cross == "ankle" else Lz
    z_e2_skin     = _ankle_z_top(e2x, e2y, Lx, Ly, Lz) if cross == "ankle" else Lz
    z_e1_elec_top = z_e1_skin + t_contact
    z_e2_elec_top = z_e2_skin + t_contact
    z_elec_top    = max(z_e1_elec_top, z_e2_elec_top)  # conservative max for BC search

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("ankle_layers")

    def _outline_surface(z_bot, lc_pts):
        """Create a planar ankle (or rect) surface at z = z_bot."""
        if cross == "ankle":
            pts_xy = ankle_outline_pts(Lx, Ly)
        else:
            pts_xy = [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]
        pt_tags = [gmsh.model.occ.addPoint(x, y, z_bot, lc_pts)
                   for x, y in pts_xy]
        n = len(pt_tags)
        lines = [gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i+1) % n])
                 for i in range(n)]
        loop = gmsh.model.occ.addCurveLoop(lines)
        return gmsh.model.occ.addPlaneSurface([loop])

    def _layer_vol(z_bot, height, lc_pts):
        surf   = _outline_surface(z_bot, lc_pts)
        result = gmsh.model.occ.extrude([(2, surf)], 0, 0, height)
        return next(dt[1] for dt in result if dt[0] == 3)

    vol_musc = _layer_vol(0,       t_muscle, lc_bulk)
    vol_fat  = _layer_vol(z0_fat,  t_fat,    lc_bulk)
    vol_skin = _layer_vol(z0_skin, t_skin,   lc_min)

    all_vols = [(3, vol_musc), (3, vol_fat), (3, vol_skin)]

    vol_c1 = vol_c2 = None
    if contact_enabled:
        def _contact_vol(xc, yc, z_bot, height, r, sh, lc):
            if sh == "circle":
                base = gmsh.model.occ.addDisk(xc, yc, z_bot, r, r)
            else:
                base = gmsh.model.occ.addRectangle(
                    xc - r, yc - r, z_bot, 2*r, 2*r)
            result = gmsh.model.occ.extrude([(2, base)], 0, 0, height)
            return next(dt[1] for dt in result if dt[0] == 3)

        vol_c1 = _contact_vol(e1x, e1y, z_e1_skin, t_contact,
                              elec_r, shape, lc_elec)
        vol_c2 = _contact_vol(e2x, e2y, z_e2_skin, t_contact,
                              elec_r, shape, lc_elec)
        all_vols += [(3, vol_c1), (3, vol_c2)]

    # ── Fragment to merge layer interfaces and embed electrodes ───────────────
    gmsh.model.occ.fragment(all_vols, [])
    gmsh.model.occ.synchronize()

    # ── Identify volumes by z-centroid ────────────────────────────────────────
    def _z_mid(dim, tag):
        b = gmsh.model.getBoundingBox(dim, tag)
        return (b[2] + b[5]) / 2.0

    def _xy_cen(dim, tag):
        b = gmsh.model.getBoundingBox(dim, tag)
        return np.array([(b[0]+b[3])/2, (b[1]+b[4])/2])

    vols = gmsh.model.getEntities(3)
    musc_v = min(vols, key=lambda dt: abs(_z_mid(*dt) - t_muscle/2))
    fat_v  = min(vols, key=lambda dt: abs(_z_mid(*dt) - (z0_fat + t_fat/2)))
    skin_v = min(vols, key=lambda dt: abs(_z_mid(*dt) - (z0_skin + t_skin/2)))

    c1_v = c2_v = None
    c1_body_id = c2_body_id = None
    if contact_enabled and vol_c1 is not None:
        contact_cands = [v for v in vols
                         if v not in (musc_v, fat_v, skin_v)]
        e1_pos2d = np.array([e1x, e1y])
        e2_pos2d = np.array([e2x, e2y])
        c1_v = min(contact_cands,
                   key=lambda dt: np.linalg.norm(_xy_cen(*dt) - e1_pos2d))
        c2_v = min([v for v in contact_cands if v != c1_v],
                   key=lambda dt: np.linalg.norm(_xy_cen(*dt) - e2_pos2d))
        c1_body_id = 4
        c2_body_id = 5

    # ── Identify electrode BC surfaces at z_elec_top ──────────────────────────
    surfs = gmsh.model.getEntities(2)

    def _surf_z_mid(dim, tag):
        b = gmsh.model.getBoundingBox(dim, tag)
        return (b[2] + b[5]) / 2.0

    # Accept surfaces at or near either electrode's top z (handles uneven surface)
    z_min_elec = min(z_e1_elec_top, z_e2_elec_top)
    tol_z      = max(z_elec_top * 2e-2, 1e-4)
    top_surfs  = [dt for dt in surfs
                  if _surf_z_mid(*dt) >= z_min_elec - tol_z]

    if not top_surfs:
        # Fall back: surfaces with highest z centroids
        top_surfs = sorted(surfs, key=lambda dt: -_surf_z_mid(*dt))[:4]

    e1_pos2d = np.array([e1x, e1y])
    e2_pos2d = np.array([e2x, e2y])

    def _surf_xy_cen(dim, tag):
        b = gmsh.model.getBoundingBox(dim, tag)
        return np.array([(b[0]+b[3])/2, (b[1]+b[4])/2])

    def _pick_elec_surf(candidates, pos2d, r_xy):
        """Surfaces near pos2d within r_xy; pick the one with the highest z centroid.
        Multiple surfaces can share the same xy centroid (e.g. contact top vs
        skin-contact interface), so we break ties by z rather than xy distance."""
        near = [dt for dt in candidates
                if np.linalg.norm(_surf_xy_cen(*dt) - pos2d) < r_xy]
        pool = near if near else candidates
        return max(pool, key=lambda dt: _surf_z_mid(*dt))

    e1_s = _pick_elec_surf(top_surfs, e1_pos2d, elec_r * 2)
    e2_s = _pick_elec_surf([s for s in top_surfs if s != e1_s],
                           e2_pos2d, elec_r * 2)
    other_s = [dt for dt in surfs if dt not in (e1_s, e2_s)]

    # ── Physical groups ───────────────────────────────────────────────────────
    gmsh.model.addPhysicalGroup(3, [musc_v[1]], 1, name="muscle")
    gmsh.model.addPhysicalGroup(3, [fat_v[1]],  2, name="fat")
    gmsh.model.addPhysicalGroup(3, [skin_v[1]], 3, name="skin")
    if contact_enabled and c1_v and c2_v:
        gmsh.model.addPhysicalGroup(3, [c1_v[1]], 4, name="contact_active")
        gmsh.model.addPhysicalGroup(3, [c2_v[1]], 5, name="contact_return")
    gmsh.model.addPhysicalGroup(2, [e1_s[1]],            101, name="active_electrode")
    gmsh.model.addPhysicalGroup(2, [e2_s[1]],            102, name="return_electrode")
    gmsh.model.addPhysicalGroup(2, [s[1] for s in other_s], 103, name="insulated")

    # ── Mesh size field ───────────────────────────────────────────────────────
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList",
                                     [e1_s[1], e2_s[1]])
    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "InField",  f_dist)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin",  lc_elec)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMax",  lc_bulk)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin",  elec_r)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax",  elec_r * 6)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)

    gmsh.model.mesh.generate(3)

    # ── Anatomical surface deformation (ankle cross-section only) ─────────────
    # Deform skin layer nodes so the top surface follows _ankle_z_top(x,y).
    # Flat mesh topology is preserved; only z-coordinates change.
    # Skin bottom (z = z0_skin) stays fixed; skin top nodes move to z_top(x,y).
    if cross == "ankle":
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        coords = ncoords.reshape(-1, 3).copy()
        for i in range(len(ntags)):
            x, y, z = coords[i]
            if z > z0_skin - 1e-6:          # node in skin layer (or contact above)
                t = (z - z0_skin) / t_skin  # 0 at skin bottom, 1 at skin top
                t = max(0.0, min(1.0, t))
                z_target = _ankle_z_top(x, y, Lx, Ly, Lz)
                coords[i, 2] = z + (z_target - Lz) * t
        for i, tag in enumerate(ntags):
            gmsh.model.mesh.setNode(int(tag), coords[i].tolist(), [])

    msh = run_dir / "mesh.msh"
    gmsh.write(str(msh))
    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()

    body_info = {
        "contact_enabled": contact_enabled,
        "z_skin_top":      z_skin_top,       # nominal flat skin bottom (= Lz)
        "z_elec_top":      z_elec_top,       # conservative max for fallback logic
        "z_e1_skin":       z_e1_skin,        # skin surface z at active electrode
        "z_e2_skin":       z_e2_skin,        # skin surface z at return electrode
        "z_e1_elec_top":   z_e1_elec_top,    # electrode top z at active site
        "z_e2_elec_top":   z_e2_elec_top,    # electrode top z at return site
        "c1_body_id":      c1_body_id,
        "c2_body_id":      c2_body_id,
        "elec_shape":      shape,
    }
    return (n_nodes,
            np.array([e1x, e1y, z_e1_elec_top]),
            np.array([e2x, e2y, z_e2_elec_top]),
            body_info)


# ── 2. Detect Elmer boundary IDs for the two electrodes ──────────────────────
def detect_elec_bc_ids(elmer_mesh_dir, e1_pos, e2_pos, z_e1_top, z_e2_top):
    """Scan mesh.nodes + mesh.boundary to find the BC IDs nearest to each electrode.
    z_e1_top / z_e2_top: expected z of each electrode's top surface (may differ for uneven geometry)."""
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

    # Wide floor filter: keep any BC element above the skin-fat interface level.
    # Per-electrode z matching happens when selecting e1_id / e2_id below.
    z_floor = min(z_e1_top, z_e2_top) - 5e-3   # 5 mm below lowest electrode top

    def _elem_area(nids):
        """Geometric area of a triangular or quad boundary element."""
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
    bc_centers   = {}    # bcid -> list of xy centroids
    bc_z_centers = {}    # bcid -> list of z centroids
    bc_elem_nids = {}    # bcid -> list of nid lists, for area computation
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
    if len(bc_mean) < 2:
        raise RuntimeError(
            f"Expected ≥2 top-face BCs, found: {list(bc_mean.keys())}")

    # For each electrode, pick the BC whose (xy centroid, z centroid) best matches
    # the electrode's expected position.  Weight z-mismatch heavily so the contact
    # top face is preferred over the skin-contact interface below it.
    def _find_elec_bc(pos_e, z_e_top, exclude=None):
        tol_z_e = max(z_e_top * 2e-2, 5e-4)
        # 1st pass: BCs within z tolerance of expected electrode top
        candidates = {bid: cxy for bid, cxy in bc_mean.items()
                      if bid != exclude
                      and abs(bc_z_mean.get(bid, 0) - z_e_top) < tol_z_e}
        if not candidates:
            # Fallback: all BCs (skip only excluded)
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
              sigma_skin_override=None, elec_area_mesh=None):
    c    = p["conductivities"]
    sv   = p["solver"]
    st   = _stim(p)
    ct   = p.get("contact", {})
    mode = st.get("control_mode", "voltage")

    sigma_skin = (sigma_skin_override
                  if sigma_skin_override is not None
                  else c["sigma_skin"])
    sigma_c = ct.get("sigma_contact_Spm", 0.005)
    contact  = body_info.get("contact_enabled", False)

    # ── Bodies ────────────────────────────────────────────────────────────────
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
"""
    if contact:
        c1 = body_info["c1_body_id"]
        c2 = body_info["c2_body_id"]
        bodies += f"""
Body 4
  Name = "contact_active"
  Target Bodies(1) = {c1}
  Equation = 1
  Material = 4
End

Body 5
  Name = "contact_return"
  Target Bodies(1) = {c2}
  Equation = 1
  Material = 4
End
"""

    # ── Materials ─────────────────────────────────────────────────────────────
    materials = f"""
! PLACEHOLDER conductivities — replace with measured values
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
  Electric Conductivity = {sigma_skin}
End
"""
    if contact:
        materials += f"""
Material 4
  Name = "contact"
  Electric Conductivity = {sigma_c}   ! effective contact conductivity — PLACEHOLDER
End
"""

    # ── BCs ───────────────────────────────────────────────────────────────────
    pl    = _pl(p)
    shape = body_info.get("elec_shape",
                          pl.get("electrode_shape", pl.get("shape", "circle")))

    jn_used = None
    if mode == "voltage":
        bc1_active = "  Potential = 1.0"
    else:  # current
        I_A           = st.get("injected_current_mA", 5.0) * 1e-3
        area_analytic = np.pi * elec_r**2 if shape == "circle" else (2 * elec_r)**2
        if elec_area_mesh is not None and elec_area_mesh > 0:
            area = elec_area_mesh
            rel_err = abs(area - area_analytic) / area_analytic
            if rel_err > 0.10:
                print(f"    WARNING: mesh electrode area ({area*1e4:.4f} cm²) "
                      f"differs {rel_err:.1%} from analytic ({area_analytic*1e4:.4f} cm²)")
        else:
            area = area_analytic
        jn   = I_A / area
        jn_used = jn
        bc1_active = (f"  Current Density = {jn:.6e}  "
                      f"! I={I_A*1e3:.1f}mA, A_mesh={area*1e4:.4f}cm²")

    bcs = f"""
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

    sif = (_SIF_HEADER.format(
               tol=sv["tolerance"],
               lin_solver=sv["linear_solver"])
           + bodies + materials + bcs)

    (run_dir / "case.sif").write_text(sif)
    return jn_used


# ── 4. Run shell commands ──────────────────────────────────────────────────────
def _run(cmd, cwd, label):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR in {label}:")
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        sys.exit(1)


# ── 4b. BC debug report (saved per case) ──────────────────────────────────────
def save_bc_debug_report(run_dir, label, e1_id, e2_id, A_active_mesh,
                         A_return_mesh, jn_used, p, body_info):
    """
    Save a text report documenting the Neumann BC setup for one case.
    Helps diagnose current-control bugs by showing what was applied and why.
    """
    st   = _stim(p)
    mode = st.get("control_mode", "voltage")
    I_mA = st.get("injected_current_mA", 5.0)
    I_A  = I_mA * 1e-3

    lines = [
        f"BC DEBUG REPORT — {label}",
        "=" * 60,
        f"  control_mode     : {mode}",
        f"  injected_current : {I_mA} mA  ({I_A:.4e} A)",
        "",
        f"  Elmer boundary ID — active  : {e1_id}",
        f"  Elmer boundary ID — return  : {e2_id}",
        "",
        f"  Mesh area — active electrode : {A_active_mesh*1e4:.4f} cm²",
        f"  Mesh area — return electrode : {A_return_mesh*1e4:.4f} cm²",
    ]
    if mode == "current" and jn_used is not None:
        expected = jn_used * A_active_mesh
        lines += [
            "",
            f"  Current density applied (Jn) : {jn_used:.6e} A/m²",
            f"  Expected current (Jn * A)    : {expected*1e3:.4f} mA",
            f"  Target current               : {I_mA:.4f} mA",
            f"  Pre-solve area error         : {abs(expected - I_A)/I_A*100:.2f}%",
            "",
            "  Elmer BC keyword used: 'Current Density = Jn'",
            "  Elmer interprets this as uniform normal J (A/m²) Neumann BC.",
            "  n_outward at top face = +z; current INTO tissue has J_z < 0.",
            "  This BC applies ONLY to the active electrode surface.",
            "  Return electrode is Dirichlet: Potential = 0.",
        ]

    contact = body_info.get("contact_enabled", False)
    lines += [
        "",
        f"  contact_enabled  : {contact}",
        f"  z_skin_top (nom) : {body_info['z_skin_top']*1000:.2f} mm",
        f"  z_e1_skin        : {body_info.get('z_e1_skin', body_info['z_skin_top'])*1000:.2f} mm  (active electrode skin surface)",
        f"  z_e2_skin        : {body_info.get('z_e2_skin', body_info['z_skin_top'])*1000:.2f} mm  (return electrode skin surface)",
        f"  z_e1_elec_top    : {body_info.get('z_e1_elec_top', body_info['z_elec_top'])*1000:.2f} mm",
        f"  z_e2_elec_top    : {body_info.get('z_e2_elec_top', body_info['z_elec_top'])*1000:.2f} mm",
    ]

    report = "\n".join(lines) + "\n"
    out    = run_dir / "bc_debug_report.txt"
    out.write_text(report)
    print(f"    BC debug → {out.relative_to(run_dir.parent.parent)}")


# ── 5a. Surface flux integral (electrode current) ──────────────────────────────
def compute_injected_current(mesh, e1_pos3d, e2_pos3d, elec_r,
                             z_e1_elec_top, z_e2_elec_top,
                             elec_shape="circle"):
    """
    Integrate J·n_outward over each electrode patch.
    Uses per-electrode z heights to handle uneven skin surface.
    Uses only embedded 2D boundary cells (VTK type 5/9) to avoid double-counting.

    Returns: (I_active_abs, I_return_abs, flux_err, I_active_signed, I_return_signed)
    """
    tolerance = 0.2

    # Extract only 2D boundary cells to avoid double-counting
    ctypes  = mesh.celltypes
    bnd_ids = np.where(np.isin(ctypes, [5, 9]))[0]

    if len(bnd_ids) == 0:
        print("    WARNING: No 2D boundary cells (type 5/9) found in VTU — "
              "falling back to extract_surface() [results may be 2× too large]")
        bnd_mesh = mesh.extract_surface(algorithm="dataset_surface")
    else:
        bnd_mesh = mesh.extract_cells(bnd_ids)

    bnd_cd   = bnd_mesh.point_data_to_cell_data()
    J_cells  = np.array(bnd_cd.cell_data["volume current"])
    cell_pts = np.array(bnd_mesh.cell_centers().points)

    sizes = bnd_mesh.compute_cell_sizes()
    areas = np.array(sizes.cell_data["Area"])

    def _mask(pos3d, z_top):
        tol_z  = max(z_top * 5e-3, 1e-5)
        top_m  = cell_pts[:, 2] > z_top - tol_z
        dx = cell_pts[:, 0] - pos3d[0]
        dy = cell_pts[:, 1] - pos3d[1]
        if elec_shape == "square":
            return (top_m
                    & (np.abs(dx) < elec_r * (1 + tolerance))
                    & (np.abs(dy) < elec_r * (1 + tolerance)))
        else:
            return top_m & (np.sqrt(dx**2 + dy**2) < elec_r * (1 + tolerance))

    active_mask = _mask(e1_pos3d, z_e1_elec_top)
    return_mask = _mask(e2_pos3d, z_e2_elec_top)

    if not active_mask.any() or not return_mask.any():
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Signed integrals: inward at active → negative; outward at return → positive
    I_active_signed = float(np.sum(J_cells[active_mask, 2] * areas[active_mask]))
    I_return_signed = float(np.sum(J_cells[return_mask, 2] * areas[return_mask]))
    I_active_abs    = abs(I_active_signed)
    I_return_abs    = abs(I_return_signed)
    denom           = max(I_active_abs, I_return_abs)
    # KCL: signed sum should ≈ 0 if current is conserved
    flux_err = float(abs(I_active_signed + I_return_signed) / denom) if denom > 0 else np.nan

    return I_active_abs, I_return_abs, flux_err, I_active_signed, I_return_signed


# ── 5b. ROI evaluation (cell-based, never NaN) ────────────────────────────────
def eval_roi(mesh, roi_cen, roi_radius_init, min_cells=4):
    """
    Mean |J| and mean |E| in a spherical ROI.  Auto-expands radius if sparse.
    E = -∇φ computed via pyvista gradient; scalar name detected automatically.

    Returns: (mean_J, mean_E, n_cells, roi_radius_used, warning_or_None)
    """
    mesh_cd    = mesh.point_data_to_cell_data()
    J_cells    = np.array(mesh_cd.cell_data["volume current"])
    Jmag_cells = np.linalg.norm(J_cells, axis=1)

    try:
        scalar_candidates = ["potential", "Potential"]
        scalar_name = next(
            (s for s in scalar_candidates if s in mesh_cd.array_names), None)
        if scalar_name is None:
            raise KeyError(
                f"Neither 'potential' nor 'Potential' found. "
                f"Available: {list(mesh_cd.array_names)}")
        grad_mesh  = mesh_cd.compute_derivative(scalars=scalar_name)
        E_cells    = -np.array(grad_mesh.cell_data["gradient"])
        Emag_cells = np.linalg.norm(E_cells, axis=1)
    except Exception:
        Emag_cells = None

    cell_pts = np.array(mesh_cd.cell_centers().points)
    dist     = np.linalg.norm(cell_pts - roi_cen, axis=1)

    warning         = None
    roi_radius_used = roi_radius_init

    for mult in [1.0, 1.5, 2.0, 3.0]:
        r_test = roi_radius_init * mult
        mask   = dist < r_test
        n      = int(mask.sum())
        if n >= min_cells:
            roi_radius_used = r_test
            if mult > 1.0:
                warning = (f"ROI radius expanded {mult:.1f}x to "
                           f"{r_test*1000:.1f} mm ({n} cells)")
            break
    else:
        roi_radius_used = roi_radius_init * 3.0
        mask   = dist < roi_radius_used
        n      = int(mask.sum())
        warning = (f"ROI at 3x ({roi_radius_used*1000:.1f} mm) "
                   f"has only {n} cells — noisy")

    n = int(mask.sum())
    if n == 0:
        return (np.nan, np.nan, 0, roi_radius_used,
                "No cells in ROI even at 3x expansion")

    mean_J = float(Jmag_cells[mask].mean())
    mean_E = (float(Emag_cells[mask].mean())
              if Emag_cells is not None else np.nan)

    return mean_J, mean_E, n, roi_radius_used, warning


# ── 5c. Extract all metrics from VTU ──────────────────────────────────────────
def extract_results(run_dir, p, t_fat, elec_r, e1_pos3d, e2_pos3d,
                    body_info, sigma_skin_used=None, jn_used=None,
                    elec_area_mesh=None, return_area_mesh=None,
                    e1_elmer_id=None, e2_elmer_id=None):
    vtu_path = run_dir / "results" / "case_t0001.vtu"
    if not vtu_path.exists():
        raise FileNotFoundError(f"VTU not found: {vtu_path}")

    mesh = pv.read(str(vtu_path))
    pts  = np.array(mesh.points)
    J    = np.array(mesh.point_data["volume current"])
    Jmag = np.linalg.norm(J, axis=1)

    g  = p["geometry"]
    Lx, Ly, Lz = g["Lx"], g["Ly"], g["Lz"]
    z_skin_top    = body_info["z_skin_top"]
    z_elec_top    = body_info["z_elec_top"]
    z_e1_elec_top = body_info.get("z_e1_elec_top", z_elec_top)
    z_e2_elec_top = body_info.get("z_e2_elec_top", z_elec_top)
    elec_shape    = body_info.get("elec_shape", "circle")
    ls_p          = p["layers"]
    z0_skin       = z_skin_top - ls_p["t_skin"]  # flat skin-fat interface

    # ── Peak |J| at skin surface (top 20% of skin layer; handles uneven top) ─
    skin_mask = pts[:, 2] > z0_skin + ls_p["t_skin"] * 0.80

    # with-electrode: include electrode footprint
    peak_J_skin_with = float(Jmag[skin_mask].max()) if skin_mask.any() else np.nan

    # no-electrode: exclude points under either electrode footprint
    if skin_mask.any():
        xp = pts[skin_mask, 0]
        yp = pts[skin_mask, 1]
        Jm = Jmag[skin_mask]

        def _in_elec(xc, yc):
            if elec_shape == "circle":
                return np.sqrt((xp - xc)**2 + (yp - yc)**2) < elec_r
            else:
                return (np.abs(xp - xc) < elec_r) & (np.abs(yp - yc) < elec_r)

        outside = ~(_in_elec(e1_pos3d[0], e1_pos3d[1]) |
                    _in_elec(e2_pos3d[0], e2_pos3d[1]))
        peak_J_skin_no = float(Jm[outside].max()) if outside.any() else peak_J_skin_with
    else:
        peak_J_skin_no = np.nan

    # ── Total injected current ────────────────────────────────────────────────
    (I_active, I_return, flux_err,
     I_active_signed, I_return_signed) = compute_injected_current(
        mesh, e1_pos3d, e2_pos3d, elec_r,
        z_e1_elec_top, z_e2_elec_top, elec_shape)
    print(f"    I_active={I_active:.4e} A  "
          f"I_return={I_return:.4e} A  "
          f"flux_err={flux_err:.2e}")

    # Hard 2% warning for current-mode deviations
    st_inner = _stim(p)
    if st_inner.get("control_mode", "voltage") == "current":
        I_target_inner = st_inner.get("injected_current_mA", 5.0) * 1e-3
        if np.isfinite(I_active) and I_target_inner > 0:
            dev = abs(I_active - I_target_inner) / I_target_inner
            if dev > 0.02:
                print(f"    *** CURRENT ERROR > 2%: measured {I_active*1e3:.3f} mA "
                      f"vs target {I_target_inner*1e3:.1f} mA "
                      f"({dev:.1%} deviation) ***")

    # ── Compliance voltage (current mode) ─────────────────────────────────────
    st   = _stim(p)
    mode = st.get("control_mode", "voltage")
    compliance_V = np.nan
    exceeded_compliance = False
    if mode == "current":
        def _node_mask_elec(pos3d, z_et):
            tol_z_e = max(z_et * 5e-3, 1e-5)
            m = pts[:, 2] > z_et - tol_z_e
            if elec_shape == "circle":
                m &= (np.sqrt((pts[:, 0] - pos3d[0])**2
                              + (pts[:, 1] - pos3d[1])**2) < elec_r * 1.5)
            else:
                m &= ((np.abs(pts[:, 0] - pos3d[0]) < elec_r * 1.5)
                      & (np.abs(pts[:, 1] - pos3d[1]) < elec_r * 1.5))
            return m

        active_mask_n = _node_mask_elec(e1_pos3d, z_e1_elec_top)
        return_mask_n = _node_mask_elec(e2_pos3d, z_e2_elec_top)

        phi_key = next((k for k in ("Potential", "potential")
                        if k in mesh.point_data), None)
        if phi_key and active_mask_n.any():
            phi = np.array(mesh.point_data[phi_key])
            V_active_mean = float(phi[active_mask_n].mean())
            V_return_mean = (float(phi[return_mask_n].mean())
                             if return_mask_n.any() else 0.0)
            compliance_V = V_active_mean - V_return_mean

        cmp_lim = st.get("compliance_voltage_V", 100.0)
        if np.isfinite(compliance_V):
            exceeded_compliance = bool(compliance_V > cmp_lim)
            if exceeded_compliance:
                print(f"    WARNING: compliance_V={compliance_V:.1f} V "
                      f"> limit {cmp_lim:.0f} V — consider reducing current "
                      f"or increasing electrode size")

    # ── ROI (cell-based, auto-expanding) ──────────────────────────────────────
    r_cfg   = p["roi"]
    z_nerve = z_skin_top - r_cfg["z_target"]
    roi_cen = np.array([e1_pos3d[0], e1_pos3d[1], z_nerve])

    mean_J_roi, mean_E_roi, roi_n_cells, roi_r_used, roi_warn = eval_roi(
        mesh, roi_cen, r_cfg["roi_radius"])
    if roi_warn:
        print(f"    ROI: {roi_warn}")

    # ── ROI layer metadata ────────────────────────────────────────────────────
    ls_p    = p["layers"]
    t_sk    = ls_p["t_skin"]
    z_fat_bot = z_skin_top - t_sk - t_fat          # fat–muscle interface z
    z_fat_top = z_skin_top - t_sk                  # skin–fat interface z
    dist_fat_muscle_mm = abs(z_nerve - z_fat_bot) * 1000.0

    # Fraction of ROI cells in each layer (using cell centers from eval_roi)
    mesh_cd_roi  = mesh.point_data_to_cell_data()
    cell_pts_all = np.array(mesh_cd_roi.cell_centers().points)
    dist_all     = np.linalg.norm(cell_pts_all - roi_cen, axis=1)
    roi_mask_all = dist_all < roi_r_used
    if roi_mask_all.any():
        z_roi = cell_pts_all[roi_mask_all, 2]
        n_roi = roi_mask_all.sum()
        frac_skin   = float((z_roi > z_fat_top).sum())   / n_roi
        frac_fat    = float(((z_roi > z_fat_bot) & (z_roi <= z_fat_top)).sum()) / n_roi
        frac_muscle = float((z_roi <= z_fat_bot).sum())  / n_roi
    else:
        frac_skin = frac_fat = frac_muscle = np.nan

    # ── Electrode area ────────────────────────────────────────────────────────
    area = np.pi * elec_r**2 if elec_shape == "circle" else (2 * elec_r)**2

    # ── Efficiency = roi_mean_E / peak_J_skin_no_elec ────────────────────────
    efficiency = (float(mean_E_roi) / peak_J_skin_no
                  if (np.isfinite(mean_E_roi) and peak_J_skin_no > 0)
                  else np.nan)

    # ── Normalise by injected current ─────────────────────────────────────────
    I_ref = I_active if np.isfinite(I_active) and I_active > 0 else np.nan

    def _norm(val):
        v = float(val)
        return v / I_ref if np.isfinite(v) and np.isfinite(I_ref) else np.nan

    # ── Layer at ROI depth ────────────────────────────────────────────────────
    ls    = p["layers"]
    t_sk  = ls["t_skin"]
    z_fat_bot = z_skin_top - t_sk - t_fat
    roi_layer = ("skin"   if z_nerve > z_skin_top - t_sk
                 else "fat"    if z_nerve > z_fat_bot
                 else "muscle")

    c   = p["conductivities"]
    sig = sigma_skin_used if sigma_skin_used is not None else c["sigma_skin"]

    def _r(val, n):
        v = float(val)
        return round(v, n) if np.isfinite(v) else v

    return {
        "t_fat_mm":              _r(t_fat * 1000, 2),
        "elec_r_mm":             _r(elec_r * 1000, 2),
        "elec_area_cm2":         _r(area * 1e4, 4),
        "elec_area_mesh_cm2":    _r(elec_area_mesh * 1e4, 4) if elec_area_mesh else None,
        "return_area_mesh_cm2":  _r(return_area_mesh * 1e4, 4) if return_area_mesh else None,
        "elec_shape":            elec_shape,
        "contact_enabled":       body_info.get("contact_enabled", False),
        "sigma_skin":            sig,
        "control_mode":          mode,
        "jn_used":               _r(jn_used, 4) if jn_used is not None else None,
        "peak_J_skin_with_elec": _r(peak_J_skin_with, 6),
        "peak_J_skin_no_elec":   _r(peak_J_skin_no,   6),
        "roi_mean_J":            _r(mean_J_roi,        6),
        "roi_mean_E":            _r(mean_E_roi,        4),
        "efficiency":            _r(efficiency,        6),
        "compliance_V":          _r(compliance_V,      3),
        "exceeded_compliance":   exceeded_compliance,
        "total_current_A":       _r(I_active,          8),
        "I_active_signed_A":     _r(I_active_signed,   8),
        "I_return_A":            _r(I_return,          8),
        "I_return_signed_A":     _r(I_return_signed,   8),
        "peak_J_skin_per_A":     _r(_norm(peak_J_skin_no), 4),
        "roi_mean_J_per_A":      _r(_norm(mean_J_roi),     4),
        "roi_mean_E_per_A":      _r(_norm(mean_E_roi),     4),
        "efficiency_per_A":      _r(efficiency, 6),  # = roi_mean_E_per_A / peak_J_skin_per_A
        "flux_err":              _r(flux_err,          6),
        "roi_layer":             roi_layer,
        "roi_n_cells":           roi_n_cells,
        "roi_radius_used_mm":    _r(roi_r_used * 1000, 2),
        # ── ROI layer metadata (PART 3) ──────────────────────────────────────
        "roi_center_z_mm":       _r(z_nerve * 1000, 3),
        "dist_fat_muscle_mm":    _r(dist_fat_muscle_mm, 3),
        "roi_frac_muscle":       _r(frac_muscle, 4),
        "roi_frac_fat":          _r(frac_fat,    4),
        "roi_frac_skin":         _r(frac_skin,   4),
        # ── BC boundary IDs (PART 2) ─────────────────────────────────────────
        "active_boundary_id_used": e1_elmer_id,
        "return_boundary_id_used": e2_elmer_id,
    }


# ── 6. Main sweep ──────────────────────────────────────────────────────────────
def run_sweep(p, t_fat_list, elec_r_list, coarse=False, sigma_skin_override=None):
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = []

    c  = p["conductivities"]
    pl = _pl(p)
    st = _stim(p)
    sigma_skin = (sigma_skin_override
                  if sigma_skin_override is not None
                  else c["sigma_skin"])
    elec_r_list_m = [r * 1e-3 for r in elec_r_list]  # mm → m

    # ── Control mode banner ───────────────────────────────────────────────────
    mode = st.get("control_mode", "voltage")
    print(f"\n{'='*60}")
    if mode == "current":
        I_mA    = st.get("injected_current_mA", 5.0)
        cmp_lim = st.get("compliance_voltage_V", 100.0)
        print(f"  CONTROL MODE : current")
        print(f"  Injected I   : {I_mA:.1f} mA  (per-case Neumann BC at active electrode)")
        print(f"  Compliance   : warn if V_active > {cmp_lim:.0f} V")
    else:
        print(f"  CONTROL MODE : voltage")
        print(f"  V_active = 1.0 V  |  V_return = 0 V  (Dirichlet BCs)")
        print(f"  Normalise outputs by total_current_A for cross-case comparison")
    print(f"{'='*60}\n")

    for t_fat in t_fat_list:
        for elec_r in elec_r_list_m:
            label = (f"tfat{int(t_fat*1000):04d}um"
                     f"_r{int(elec_r*1000):04d}um")
            run_dir = RESULTS_DIR / label
            print(f"\n[{label}]  t_fat={t_fat*1000:.1f}mm  "
                  f"r={elec_r*1000:.1f}mm  sigma_skin={sigma_skin}")

            print("  meshing ...")
            n_nodes, e1_pos, e2_pos, body_info = build_mesh(
                p, t_fat, elec_r, run_dir, coarse=coarse)
            print(f"    {n_nodes} nodes")

            print("  ElmerGrid ...")
            elmer_dir = run_dir / "elmer_mesh"
            elmer_dir.mkdir(exist_ok=True)
            _run(["ElmerGrid", "14", "2", "mesh.msh", "-out", "elmer_mesh"],
                 cwd=run_dir, label="ElmerGrid")

            print("  detecting electrode BCs + computing mesh areas ...")
            e1_id, e2_id, A_active_mesh, A_return_mesh = detect_elec_bc_ids(
                elmer_dir, e1_pos, e2_pos, e1_pos[2], e2_pos[2])
            area_analytic = (np.pi * elec_r**2 if pl.get("electrode_shape",
                             pl.get("shape", "circle")) == "circle"
                             else (2 * elec_r)**2)
            print(f"    active={e1_id}  return={e2_id}  "
                  f"A_active={A_active_mesh*1e4:.4f}cm²  "
                  f"A_analytic={area_analytic*1e4:.4f}cm²")

            jn_used = write_sif(run_dir, e1_id, e2_id, p, elec_r, body_info,
                                sigma_skin_override=sigma_skin_override,
                                elec_area_mesh=A_active_mesh)
            save_bc_debug_report(run_dir, label, e1_id, e2_id,
                                 A_active_mesh, A_return_mesh,
                                 jn_used, p, body_info)
            (run_dir / "results").mkdir(exist_ok=True)

            print("  ElmerSolver ...")
            _run(["ElmerSolver", "case.sif"], cwd=run_dir, label="ElmerSolver")

            print("  extracting metrics ...")
            res = extract_results(
                run_dir, p, t_fat, elec_r, e1_pos, e2_pos,
                body_info, sigma_skin_used=sigma_skin, jn_used=jn_used,
                elec_area_mesh=A_active_mesh, return_area_mesh=A_return_mesh,
                e1_elmer_id=e1_id, e2_elmer_id=e2_id)

            print(f"    peak_J_no_elec={res['peak_J_skin_no_elec']:.4f}  "
                  f"roi_mean_E={res['roi_mean_E']:.4f}  "
                  f"efficiency={res['efficiency']:.4e}  "
                  f"flux_err={res['flux_err']:.3e}")
            if res.get("control_mode") == "current":
                I_target = st.get("injected_current_mA", 5.0) * 1e-3
                I_actual = res.get("total_current_A", float("nan"))
                print(f"    compliance_V={res['compliance_V']:.2f} V  "
                      f"I_active={I_actual:.4e} A  I_return={res.get('I_return_A', float('nan')):.4e} A")
                if np.isfinite(I_actual) and I_target > 0:
                    dev = abs(I_actual - I_target) / I_target
                    if dev > 0.05:
                        note = " (coarse mesh — expected)" if coarse else " — check mesh/BC"
                        print(f"    WARNING: I_active ({I_actual*1e3:.2f} mA) deviates "
                              f"{dev:.1%} from target {I_target*1e3:.1f} mA{note}")

            all_results.append(res)

    return all_results


def print_run_summary(results, p):
    """Print a human-readable end-of-run summary with files created and example metrics."""
    st   = _stim(p)
    mode = st.get("control_mode", "voltage")
    print(f"\n{'='*60}")
    print("  RUN COMPLETE — OUTPUTS")
    print(f"{'='*60}")
    print(f"  results/summary.csv")
    print(f"  results/summary.json")
    print(f"  {len(results)} case(s) computed")
    if results:
        ex = results[len(results) // 2]   # middle case as example
        print(f"\n  Example case  "
              f"(fat={ex['t_fat_mm']:.1f} mm, r={ex['elec_r_mm']:.1f} mm):")
        print(f"    control_mode       : {ex.get('control_mode', '?')}")
        if mode == "current" and ex.get("jn_used") is not None:
            print(f"    jn_used            : {ex['jn_used']:.4f} A/m²")
        print(f"    I_active           : {ex.get('total_current_A', float('nan')):.4e} A")
        print(f"    I_return           : {ex.get('I_return_A', float('nan')):.4e} A")
        print(f"    flux_err           : {ex.get('flux_err', float('nan')):.3e}")
        if mode == "current":
            cV = ex.get("compliance_V", float("nan"))
            ec = ex.get("exceeded_compliance", False)
            print(f"    compliance_V       : {cV:.2f} V"
                  + ("  [EXCEEDED]" if ec else ""))
        print(f"    peak_J_no_elec     : {ex.get('peak_J_skin_no_elec', float('nan')):.4f} A/m²")
        print(f"    roi_mean_E         : {ex.get('roi_mean_E', float('nan')):.4f} V/m")
        print(f"    efficiency         : {ex.get('efficiency', float('nan')):.4e} m")
    print(f"{'='*60}")
    print("  Run plot_layered_results.py to generate figures.\n")


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
                  default=lambda x: None
                  if isinstance(x, float) and np.isnan(x) else x)
    print(f"Saved → {json_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ankle layered slab sweep")
    parser.add_argument("--smoke", action="store_true",
                        help="Single coarse case for quick pipeline check")
    args = parser.parse_args()

    p  = load_params()
    pl = _pl(p)

    if args.smoke:
        t_fat_list  = [p["layers"]["t_fat"]]
        r_list_mm   = [pl.get("electrode_r_mm_list",
                              pl.get("size_list", [10]))[1]]
        coarse      = True
        print("=== SMOKE TEST (1 coarse case) ===")
    else:
        t_fat_list  = p["layers"]["t_fat_sweep"]
        r_list_mm   = pl.get("electrode_r_mm_list",
                             pl.get("size_list", [5, 10, 15]))
        coarse      = False
        print(f"=== FULL SWEEP: {len(t_fat_list)} fat thicknesses × "
              f"{len(r_list_mm)} electrode sizes = "
              f"{len(t_fat_list)*len(r_list_mm)} cases ===")

    results = run_sweep(p, t_fat_list, r_list_mm, coarse=coarse)
    save_results(results)
    print_run_summary(results, p)
