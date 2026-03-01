"""
test_step01_baseline.py — automated validation for step01_box
=============================================================
Checks that the FEM solution matches the known analytic result for
static current conduction in a rectangular box:

    V(z) = z / Lz,   |J| = sigma / Lz = 10 A/m^2  (uniform everywhere)

Usage (from step01_box/):
    python3 test_step01_baseline.py

Exit 0 = PASS, Exit 1 = FAIL.
"""

import sys
import subprocess
import numpy as np
import pyvista as pv
from pathlib import Path

# ── Tolerances ────────────────────────────────────────────────────────────────
TOL_J_REL   = 1e-3    # relative error: |mean(|J|) - analytic| / analytic
TOL_J_CV    = 1e-2    # uniformity:     std(|J|) / mean(|J|)
TOL_V_R2    = 0.9999  # linearity:      R^2 for V(z) along center column
TOL_FLUX    = 1e-2    # conservation:   |flux_top - flux_bot| / flux_top

# ── Physics ───────────────────────────────────────────────────────────────────
SIGMA = 0.2   # S/m
V_TOP = 1.0   # V
V_BOT = 0.0   # V

VTU = Path("results/case_t0001.vtu")


# ── Pipeline runner (only if VTU missing) ─────────────────────────────────────
def _run(cmd, label):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"ERROR: {label} failed\n{r.stdout[-1000:]}\n{r.stderr[-1000:]}")


def run_pipeline_if_needed():
    if VTU.exists():
        return
    print("VTU not found — running pipeline ...\n")
    if not Path("box.msh").exists():
        _run(["gmsh", "-3", "box.geo", "-o", "box.msh"], "gmsh")
    if not Path("elmer_mesh/mesh.nodes").exists():
        _run(["ElmerGrid", "14", "2", "box.msh", "-out", "elmer_mesh"], "ElmerGrid")
    _run([sys.executable, "setup_case.py"], "setup_case.py")
    Path("results").mkdir(exist_ok=True)
    _run(["ElmerSolver", "case.sif"], "ElmerSolver")
    if not VTU.exists():
        sys.exit("ERROR: pipeline ran but VTU still missing")
    print("Pipeline complete.\n")


# ── Metric computation ────────────────────────────────────────────────────────
def compute_metrics(vtu_path):
    mesh  = pv.read(str(vtu_path))
    pts   = np.array(mesh.points)
    phi   = np.array(mesh.point_data["potential"])
    J_vec = np.array(mesh.point_data["volume current"])
    J_mag = np.linalg.norm(J_vec, axis=1)

    Lx = pts[:, 0].max()
    Ly = pts[:, 1].max()
    Lz = pts[:, 2].max()
    J_an = SIGMA * (V_TOP - V_BOT) / Lz   # analytic |J|

    # 1. Volume-average |J|: arithmetic mean over nodes (valid for ~uniform mesh)
    mean_J = J_mag.mean()
    std_J  = J_mag.std(ddof=1)
    cv_J   = std_J / mean_J
    rel_J  = abs(mean_J - J_an) / J_an

    # 2. R^2 for V(z) linearity along the center column
    r_xy   = np.hypot(pts[:, 0] - Lx / 2, pts[:, 1] - Ly / 2)
    col    = r_xy < Lx * 0.08
    z_c    = pts[col, 2]
    phi_c  = phi[col]
    coeffs = np.polyfit(z_c, phi_c, 1)          # slope, intercept
    phi_fit = np.polyval(coeffs, z_c)
    ss_res = np.sum((phi_c - phi_fit) ** 2)
    ss_tot = np.sum((phi_c - phi_c.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    slope  = coeffs[0]                           # should equal 1/Lz

    # 3. Current conservation: mean |J_z| on top vs bottom surface
    #    J_z < 0 (current flows downward, top=1V → bot=0V)
    tol_z   = Lz * 1e-3
    top_Jz  = np.abs(J_vec[pts[:, 2] > Lz - tol_z, 2])
    bot_Jz  = np.abs(J_vec[pts[:, 2] < tol_z,       2])
    ft      = top_Jz.mean()
    fb      = bot_Jz.mean()
    flux_err = abs(ft - fb) / max(ft, fb)

    return dict(
        Lz=Lz, J_an=J_an,
        mean_J=mean_J, std_J=std_J, cv_J=cv_J, rel_J=rel_J,
        r2=r2, slope=slope,
        flux_top=ft, flux_bot=fb, flux_err=flux_err,
        phi_min=phi.min(), phi_max=phi.max(),
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    run_pipeline_if_needed()

    m = compute_metrics(VTU)

    W = 60
    print("=" * W)
    print("step01_box  baseline validation")
    print("=" * W)
    print(f"  Geometry            Lz = {m['Lz']*100:.1f} cm")
    print(f"  Analytic |J|           = {m['J_an']:.4f} A/m²  (σ·ΔV/Lz)")
    print()
    print(f"  mean(|J|)              = {m['mean_J']:.6f} A/m²")
    print(f"  std(|J|)               = {m['std_J']:.2e}  A/m²")
    print(f"  CV = std/mean          = {m['cv_J']:.2e}      tol < {TOL_J_CV:.0e}")
    print(f"  rel error vs analytic  = {m['rel_J']:.2e}      tol < {TOL_J_REL:.0e}")
    print()
    print(f"  V(z) R²  (center col)  = {m['r2']:.7f}   tol > {TOL_V_R2}")
    print(f"  V(z) slope             = {m['slope']:.4f} V/m  "
          f"(analytic {(V_TOP-V_BOT)/m['Lz']:.4f})")
    print(f"  Φ range                = [{m['phi_min']:.4f}, {m['phi_max']:.4f}] V")
    print()
    print(f"  Flux |J_z| at top      = {m['flux_top']:.4f} A/m²")
    print(f"  Flux |J_z| at bottom   = {m['flux_bot']:.4f} A/m²")
    print(f"  Flux conservation err  = {m['flux_err']:.2e}      tol < {TOL_FLUX:.0e}")
    print()

    failures = []
    if m["rel_J"]    >= TOL_J_REL:
        failures.append(f"  FAIL  rel error mean|J| = {m['rel_J']:.2e} (>= {TOL_J_REL:.0e})")
    if m["cv_J"]     >= TOL_J_CV:
        failures.append(f"  FAIL  CV std/mean|J|    = {m['cv_J']:.2e} (>= {TOL_J_CV:.0e})")
    if m["r2"]       <  TOL_V_R2:
        failures.append(f"  FAIL  R^2               = {m['r2']:.7f} (< {TOL_V_R2})")
    if m["flux_err"] >= TOL_FLUX:
        failures.append(f"  FAIL  flux conservation = {m['flux_err']:.2e} (>= {TOL_FLUX:.0e})")

    if failures:
        print("RESULT:  FAIL")
        for f in failures:
            print(f)
        sys.exit(1)

    print("RESULT:  PASS")
    print(f"  mean|J| = {m['mean_J']:.6f} A/m²  (analytic {m['J_an']:.6f})")
    print(f"  R²      = {m['r2']:.7f}")
    print(f"  CV      = {m['cv_J']:.2e}")
    print(f"  flux Δ  = {m['flux_err']:.2e}")
    print("=" * W)
    sys.exit(0)


if __name__ == "__main__":
    main()
