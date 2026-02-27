"""
run_sweep.py  —  Bipolar electrode current density sweep
======================================================
Answers: "For a given electrode area, what is the current density
distribution across the skin surface?"  (Michelle's question)

Setup:
  - Tissue box 15×15×5 cm (large enough that edges don't matter)
  - Two electrode patches on the TOP surface (bipolar):
      active (+1V) centered at (cx - sep/2, cy)
      return  (0V) centered at (cx + sep/2, cy)
  - All other surfaces insulated (zero-flux natural BC)
  - Sweep: circle and square, 4 radii each  →  8 simulations

Run from step02_electrodes/:
    python3 run_sweep.py

Outputs in results/<shape>_r<mm>mm/:
    mesh.msh, elmer_mesh/, case.sif, results/case_t0001.vtu

Summary plots:
    results/sweep_J_maps.png   — 4×2 grid of |J| heatmaps at skin surface
    results/sweep_summary.png  — peak/mean J vs electrode area
"""

import subprocess, sys, shutil
import numpy as np
from pathlib import Path

import gmsh
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.tri as mtri
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, Normalize

# ── Parameters ────────────────────────────────────────────────────────────────
Lx, Ly, Lz = 0.15, 0.15, 0.05          # tissue box (m)
SEP         = 0.06                       # center-to-center separation (m)
SIGMA       = 0.2                        # tissue conductivity (S/m)

SHAPES = ["circle", "square"]
RADII  = [0.005, 0.010, 0.015, 0.020]   # electrode radius / half-side (m)
VOLTS  = (1.0, 0.0)                      # (active, return) voltage (V)

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

cx_box, cy_box = Lx / 2, Ly / 2
e1_pos = np.array([cx_box - SEP / 2, cy_box])   # active electrode center
e2_pos = np.array([cx_box + SEP / 2, cy_box])   # return electrode center

# ── 1. Build gmsh mesh ────────────────────────────────────────────────────────
def build_mesh(shape, r, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("electrodes")

    # Volume box
    box = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)

    # Electrode patches on top face (z = Lz)
    if shape == "circle":
        e1 = gmsh.model.occ.addDisk(e1_pos[0], e1_pos[1], Lz, r, r)
        e2 = gmsh.model.occ.addDisk(e2_pos[0], e2_pos[1], Lz, r, r)
        area = np.pi * r**2
    else:  # square — same half-side as r, area ≈ same order
        half = r
        e1 = gmsh.model.occ.addRectangle(
            e1_pos[0] - half, e1_pos[1] - half, Lz, 2*half, 2*half)
        e2 = gmsh.model.occ.addRectangle(
            e2_pos[0] - half, e2_pos[1] - half, Lz, 2*half, 2*half)
        area = (2*r)**2

    # Embed electrodes into box top face
    gmsh.model.occ.fragment([(3, box)], [(2, e1), (2, e2)])
    gmsh.model.occ.synchronize()

    # Identify surfaces by bounding-box center
    surfs = gmsh.model.getEntities(2)
    vols  = gmsh.model.getEntities(3)

    def bb_center(dim, tag):
        bb = gmsh.model.getBoundingBox(dim, tag)
        return np.array([(bb[0]+bb[3])/2, (bb[1]+bb[4])/2, (bb[2]+bb[5])/2])

    # Active electrode: surface on top face closest to e1_pos
    # Return electrode: surface on top face closest to e2_pos
    top_surfs = [(d, t) for d, t in surfs
                 if abs(bb_center(d, t)[2] - Lz) < Lz * 1e-3]

    def nearest(pos2d, candidates):
        return min(candidates,
                   key=lambda dt: np.linalg.norm(bb_center(*dt)[:2] - pos2d))

    e1_surf = nearest(e1_pos, top_surfs)
    e2_surf = nearest(e2_pos, top_surfs)
    other_surfs = [dt for dt in surfs if dt not in (e1_surf, e2_surf)]

    gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], 1, name="tissue")
    gmsh.model.addPhysicalGroup(2, [e1_surf[1]], 101, name="active")
    gmsh.model.addPhysicalGroup(2, [e2_surf[1]], 102, name="return")
    gmsh.model.addPhysicalGroup(2, [s[1] for s in other_surfs], 103, name="insulated")

    # Adaptive mesh: fine under electrodes, coarse elsewhere
    lc_elec = r / 3.5
    lc_bulk = min(r * 4, 0.012)

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList",
                                     [e1_surf[1], e2_surf[1]])
    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "InField",  f_dist)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin",  lc_elec)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMax",  lc_bulk)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin",  r)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax",  r * 7)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)

    gmsh.model.mesh.generate(3)

    msh_path = run_dir / "mesh.msh"
    gmsh.write(str(msh_path))
    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()
    print(f"    mesh: {n_nodes} nodes → {msh_path.name}")
    return area


# ── 2. Detect Elmer boundary IDs for electrodes ───────────────────────────────
def detect_elec_bc_ids(mesh_dir):
    """After ElmerGrid conversion, find which BC IDs are the two electrodes
    by locating which boundaries are small flat patches on the top face."""
    nodes = {}
    with open(mesh_dir / "mesh.nodes") as f:
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

    etype_nn = {202: 2, 203: 3, 303: 3, 306: 6, 404: 4}
    tol_z = Lz * 0.01

    bc_centers = {}   # bcid -> list of xy centroids of top-face elements
    with open(mesh_dir / "mesh.boundary") as f:
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
                node_ids = [int(p) for p in parts[5:5+nn]]
            except ValueError:
                continue
            coords = [nodes[n] for n in node_ids if n in nodes]
            if len(coords) < 3:
                continue
            zvals = [c[2] for c in coords]
            if min(zvals) < Lz - tol_z:   # not on top face
                continue
            centroid_xy = np.mean([c[:2] for c in coords], axis=0)
            bc_centers.setdefault(bcid, []).append(centroid_xy)

    # Aggregate: mean centroid per BC on top face
    bc_mean = {bid: np.mean(pts, axis=0)
               for bid, pts in bc_centers.items() if pts}

    if len(bc_mean) < 2:
        raise RuntimeError(
            f"Expected ≥2 top-face boundaries, found: {list(bc_mean.keys())}\n"
            "Check that Physical Surface 101/102 were meshed.")

    def nearest_bc(pos2d):
        return min(bc_mean,
                   key=lambda bid: np.linalg.norm(bc_mean[bid] - pos2d))

    e1_id = nearest_bc(e1_pos)
    e2_id = nearest_bc(e2_pos)
    return e1_id, e2_id


# ── 3. Write Elmer SIF ────────────────────────────────────────────────────────
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

Body 1
  Target Bodies(1) = 1
  Name = "tissue"
  Equation = 1
  Material = 1
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
  Linear System Direct Method = UMFPACK
  Steady State Convergence Tolerance = 1.0e-8
End

Solver 2
  Equation = "ResultOutput"
  Procedure = "ResultOutputSolve" "ResultOutputSolver"
  Output File Name = "case"
  Output Format = VTU
  VTU Format = Logical True
  Save Geometry IDs = Logical True
End

Material 1
  Name = "Tissue"
  Electric Conductivity = {sigma}
End

Boundary Condition 1
  Name = "active"
  Target Boundaries = {e1_id}
  Potential = {v_active}
End

Boundary Condition 2
  Name = "return"
  Target Boundaries = {e2_id}
  Potential = {v_return}
End
"""

def write_sif(run_dir, e1_id, e2_id):
    sif = SIF_TEMPLATE.format(
        sigma=SIGMA, e1_id=e1_id, e2_id=e2_id,
        v_active=VOLTS[0], v_return=VOLTS[1])
    (run_dir / "case.sif").write_text(sif)


# ── 4. Shell helpers ──────────────────────────────────────────────────────────
def run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR running {cmd[0]}:")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        sys.exit(1)


# ── 5. Extract results from VTU ───────────────────────────────────────────────
def extract_top_J(run_dir):
    """Return (x, y, |J|) arrays for all nodes on the top face (z ≈ Lz)."""
    vtu = run_dir / "results" / "case_t0001.vtu"
    if not vtu.exists():
        raise FileNotFoundError(f"VTU not found: {vtu}")
    mesh = pv.read(str(vtu))
    pts  = np.array(mesh.points)
    J    = np.linalg.norm(np.array(mesh.point_data["volume current"]), axis=1)
    mask = pts[:, 2] > Lz * 0.99
    return pts[mask, 0], pts[mask, 1], J[mask]


# ── 6. Main sweep loop ────────────────────────────────────────────────────────
sweep_results = []   # list of dicts

for shape in SHAPES:
    for r in RADII:
        label = f"{shape}_r{int(r*1000):02d}mm"
        run_dir = RESULTS / label
        print(f"\n[{label}]")

        # --- mesh ---
        print("  building mesh...")
        area = build_mesh(shape, r, run_dir)

        # --- ElmerGrid ---
        print("  converting mesh (ElmerGrid)...")
        elmer_mesh_dir = run_dir / "elmer_mesh"
        elmer_mesh_dir.mkdir(exist_ok=True)
        run(["ElmerGrid", "14", "2", "mesh.msh", "-out", "elmer_mesh"],
            cwd=run_dir)

        # --- detect electrode BC IDs ---
        print("  detecting electrode boundary IDs...")
        e1_id, e2_id = detect_elec_bc_ids(elmer_mesh_dir)
        print(f"    active BC={e1_id}, return BC={e2_id}")

        # --- write SIF and run Elmer ---
        write_sif(run_dir, e1_id, e2_id)
        (run_dir / "results").mkdir(exist_ok=True)
        print("  running ElmerSolver...")
        run(["ElmerSolver", "case.sif"], cwd=run_dir)

        # --- extract results ---
        print("  extracting J on skin surface...")
        x, y, J = extract_top_J(run_dir)
        peak_J = J.max()
        mean_J = J.mean()
        print(f"    peak|J|={peak_J:.2f}  mean|J|={mean_J:.2f} A/m²")

        sweep_results.append({
            "shape": shape, "r": r, "area": area,
            "label": label, "run_dir": run_dir,
            "x": x, "y": y, "J": J,
            "peak_J": peak_J, "mean_J": mean_J,
        })

print("\nAll simulations done. Generating plots...")


# ── 7. Plot: 4×2 grid of J heatmaps at skin surface ─────────────────────────
fig, axes = plt.subplots(
    len(RADII), len(SHAPES),
    figsize=(5 * len(SHAPES), 4.2 * len(RADII)),
    squeeze=False
)
fig.suptitle(
    "Current density |J| at skin surface (z = Lz)  —  bipolar electrodes\n"
    f"Tissue σ = {SIGMA} S/m  |  ΔV = {VOLTS[0]-VOLTS[1]:.0f} V  |  "
    f"Electrode separation = {SEP*100:.0f} cm  |  "
    "Insulated skin around electrodes",
    fontsize=11, y=0.995
)

# Shared color scale across all panels (log scale to show edge concentration)
all_J = np.concatenate([d["J"] for d in sweep_results])
vmin_log = np.percentile(all_J[all_J > 0], 5)
vmax_log = np.percentile(all_J, 99)

for row_i, r in enumerate(RADII):
    for col_j, shape in enumerate(SHAPES):
        ax = axes[row_i][col_j]
        d  = next(res for res in sweep_results
                  if res["shape"] == shape and res["r"] == r)

        # Triangulate top-face nodes and plot J
        tri = mtri.Triangulation(d["x"], d["y"])
        J_norm = np.clip(d["J"], vmin_log, vmax_log)
        tc = ax.tricontourf(tri, J_norm, levels=40, cmap="inferno",
                            vmin=vmin_log, vmax=vmax_log)
        plt.colorbar(tc, ax=ax, label="|J| (A/m²)", shrink=0.85)

        # Electrode footprint outlines
        if shape == "circle":
            for xc, yc, clr, lbl in [
                (e1_pos[0], e1_pos[1], "cyan",  "+1V"),
                (e2_pos[0], e2_pos[1], "lime",  "0V"),
            ]:
                circle = plt.Circle((xc, yc), r, fill=False,
                                    edgecolor=clr, linewidth=2, linestyle="--")
                ax.add_patch(circle)
                ax.text(xc, yc, lbl, ha="center", va="center",
                        color=clr, fontsize=7, fontweight="bold")
        else:
            for xc, yc, clr, lbl in [
                (e1_pos[0], e1_pos[1], "cyan",  "+1V"),
                (e2_pos[0], e2_pos[1], "lime",  "0V"),
            ]:
                rect = mpatches.Rectangle(
                    (xc - r, yc - r), 2*r, 2*r,
                    fill=False, edgecolor=clr, linewidth=2, linestyle="--")
                ax.add_patch(rect)
                ax.text(xc, yc, lbl, ha="center", va="center",
                        color=clr, fontsize=7, fontweight="bold")

        area_cm2 = d["area"] * 1e4
        ax.set_title(
            f"{shape.capitalize()}  r={int(r*1000)} mm  "
            f"(area={area_cm2:.2f} cm²)\n"
            f"peak|J|={d['peak_J']:.1f}  mean|J|={d['mean_J']:.2f} A/m²",
            fontsize=9
        )
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
        ax.set_aspect("equal")

plt.tight_layout(rect=[0, 0, 1, 0.99])
out_maps = RESULTS / "sweep_J_maps.png"
plt.savefig(out_maps, dpi=130, bbox_inches="tight")
print(f"Saved → {out_maps}")
plt.close()


# ── 8. Summary plot: peak & mean J vs electrode area ─────────────────────────
fig2, (ax_peak, ax_mean) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle(
    "Current density vs electrode size  —  key result for electrode design\n"
    f"Bipolar config, ΔV = {VOLTS[0]-VOLTS[1]:.0f} V, σ = {SIGMA} S/m, "
    f"tissue {Lx*100:.0f}×{Ly*100:.0f}×{Lz*100:.0f} cm",
    fontsize=11
)

colors = {"circle": "#e74c3c", "square": "#3498db"}
markers = {"circle": "o", "square": "s"}

for shape in SHAPES:
    sub = [d for d in sweep_results if d["shape"] == shape]
    areas_cm2 = np.array([d["area"] * 1e4 for d in sub])
    peak_Js   = np.array([d["peak_J"] for d in sub])
    mean_Js   = np.array([d["mean_J"] for d in sub])
    radii_mm  = np.array([d["r"] * 1000 for d in sub])

    for ax, y_vals, ylabel, title in [
        (ax_peak, peak_Js,  "Peak |J| (A/m²)",
         "Peak current density vs electrode area\n"
         "(smaller electrode → higher peak J → more discomfort risk)"),
        (ax_mean, mean_Js,  "Mean |J| under electrode (A/m²)",
         "Mean current density under electrode\n"
         "(relates to nerve activation threshold)"),
    ]:
        ax.plot(areas_cm2, y_vals, f"{markers[shape]}-",
                color=colors[shape], lw=2, ms=8, label=shape.capitalize())
        for a, j, rm in zip(areas_cm2, y_vals, radii_mm):
            ax.annotate(f"r={rm:.0f}mm", (a, j),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color=colors[shape])

for ax, ylabel, title in [
    (ax_peak, "Peak |J| (A/m²)",
     "Peak current density vs electrode area\n"
     "(smaller electrode → higher peak J → more discomfort risk)"),
    (ax_mean, "Mean |J| under electrode (A/m²)",
     "Mean current density under electrode\n"
     "(relates to nerve activation threshold)"),
]:
    ax.set_xlabel("Electrode area (cm²)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    ax.set_xscale("log"); ax.set_yscale("log")

plt.tight_layout()
out_summary = RESULTS / "sweep_summary.png"
plt.savefig(out_summary, dpi=150, bbox_inches="tight")
print(f"Saved → {out_summary}")
plt.close()

print("\nDone. Key outputs:")
print(f"  {out_maps}")
print(f"  {out_summary}")
print(f"\nInterpretation:")
print(f"  • Smaller electrodes → concentrated current under pad → higher peak J")
print(f"  • Larger electrodes  → spread current → lower peak J, more comfort")
print(f"  • Circle vs square: similar mean J, but squares concentrate more at corners")
