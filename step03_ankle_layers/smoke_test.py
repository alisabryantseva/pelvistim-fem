"""
smoke_test.py — quick end-to-end validation for step03_ankle_layers
=====================================================================
Runs a single coarse case (--smoke flag) and asserts:
  1. VTU output file exists
  2. Potential field: finite, range [0, 1] V (Dirichlet BCs)
  3. Current density field: finite, no NaNs
  4. Electric field present in VTU
  5. Current conservation at electrode patches (flux_err < 5%)
  6. total_current_A is positive and finite
  7. ROI mean |J| is positive and finite (cell-based, never NaN)

Usage (from step03_ankle_layers/):
    python3 smoke_test.py

Exit code 0 = all checks pass.  Non-zero = at least one failure (details printed).
"""

import subprocess
import sys
import json
import numpy as np
import pyvista as pv
import yaml
from pathlib import Path

HERE        = Path(__file__).parent
PARAMS_FILE = HERE / "params.yaml"
RESULTS_DIR = HERE / "results"

FLUX_TOL  = 0.05   # 5% tolerance for coarse mesh current conservation
ROI_MIN   = 1e-6   # ROI mean |J| must exceed this (sanity floor)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    line   = f"  [{status}]  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return condition


def run_smoke_case():
    print("Running run_layered_sweep.py --smoke ...")
    r = subprocess.run(
        [sys.executable, str(HERE / "run_layered_sweep.py"), "--smoke"],
        capture_output=False,        # stream stdout/stderr live
    )
    if r.returncode != 0:
        print(f"\n{FAIL}  run_layered_sweep.py exited with code {r.returncode}")
        sys.exit(r.returncode)
    print()


def find_smoke_vtu(p):
    """Locate the VTU produced by the smoke case (middle fat, middle electrode)."""
    t_fat   = p["layers"]["t_fat"]
    elec_r  = p["electrodes"]["size_list"][1]
    label   = f"tfat{int(t_fat*1000):04d}um_r{int(elec_r*1000):04d}um"
    vtu     = RESULTS_DIR / label / "results" / "case_t0001.vtu"
    return vtu, label, t_fat, elec_r


def main():
    with open(PARAMS_FILE) as f:
        p = yaml.safe_load(f)

    run_smoke_case()

    vtu_path, label, t_fat, elec_r = find_smoke_vtu(p)
    print(f"Checking case: {label}\n")

    passed = []

    # ── 1. VTU exists ─────────────────────────────────────────────────────────
    passed.append(check("VTU file exists", vtu_path.exists(), str(vtu_path)))
    if not vtu_path.exists():
        print("\nCannot continue — VTU not found.")
        sys.exit(1)

    # ── Load VTU ──────────────────────────────────────────────────────────────
    mesh = pv.read(str(vtu_path))
    pts  = np.array(mesh.points)

    # ── 2. Potential field ────────────────────────────────────────────────────
    has_phi = "potential" in mesh.point_data
    passed.append(check("Field 'potential' present", has_phi))
    if has_phi:
        phi    = np.array(mesh.point_data["potential"])
        finite = np.all(np.isfinite(phi))
        passed.append(check("Potential is finite (no NaN/Inf)", finite,
                            f"min={phi.min():.4f} max={phi.max():.4f} V"))
        in_range = (phi.min() >= -0.01) and (phi.max() <= 1.01)
        passed.append(check("Potential in [0, 1] V",   in_range,
                            f"min={phi.min():.4f} max={phi.max():.4f}"))

    # ── 3. Current density field ──────────────────────────────────────────────
    has_J = "volume current" in mesh.point_data
    passed.append(check("Field 'volume current' present", has_J))
    if has_J:
        J    = np.array(mesh.point_data["volume current"])
        Jmag = np.linalg.norm(J, axis=1)
        finite_J = np.all(np.isfinite(Jmag))
        passed.append(check("Current density is finite (no NaN/Inf)", finite_J,
                            f"max|J|={Jmag.max():.3f} A/m²"))

    # ── 4. Electric field computable from potential ───────────────────────────
    # Elmer StatCurrentSolve doesn't export E directly; we compute E = -∇φ
    # using pyvista's compute_derivative. Check it runs without error here.
    has_phi_check = "potential" in mesh.point_data
    if has_phi_check:
        try:
            mc = mesh.point_data_to_cell_data()
            gm = mc.compute_derivative(scalars="potential")
            E_cells = -np.array(gm.cell_data["gradient"])
            Emag_ok = np.all(np.isfinite(E_cells))
        except Exception as exc:
            Emag_ok = False
            print(f"    E gradient error: {exc}")
        passed.append(check("E = -∇φ computable and finite", Emag_ok))

    # ── 5. summary.json + quantitative checks ────────────────────────────────
    json_path = RESULTS_DIR / "summary.json"
    has_json  = json_path.exists()
    passed.append(check("summary.json exists", has_json))
    if has_json and has_J:
        with open(json_path) as f:
            results = json.load(f)
        row = next((r for r in results
                    if abs(r["t_fat_mm"] - t_fat*1000) < 0.1
                    and abs(r["elec_r_mm"] - elec_r*1000) < 0.1), None)

        if row is not None:
            # ── 5a. Current conservation ──────────────────────────────────────
            flux_err = row.get("flux_err", float("nan"))
            ok_flux  = np.isfinite(flux_err) and flux_err < FLUX_TOL
            passed.append(check(
                f"Current conservation (flux_err < {FLUX_TOL:.0%})",
                ok_flux,
                f"flux_err = {flux_err:.3%}"
            ))

            # ── 5b. Total injected current finite and positive ─────────────────
            I_total = row.get("total_current_A", float("nan"))
            ok_I = np.isfinite(I_total) and I_total > 0
            passed.append(check(
                "total_current_A is positive and finite",
                ok_I,
                f"total_current_A = {I_total:.4e} A"
            ))

            # ── 5c. ROI mean |J| — cell-based, never NaN ──────────────────────
            roi_J = row.get("mean_J_roi", float("nan"))
            roi_n = row.get("roi_n_cells", 0)
            roi_r = row.get("roi_radius_used_mm", "?")
            ok_roi = np.isfinite(roi_J) and roi_J > ROI_MIN
            passed.append(check(
                "ROI mean |J| is positive and finite",
                ok_roi,
                f"mean_J_roi={roi_J:.5f} A/m²  roi_n_cells={roi_n}  r_used={roi_r}mm"
            ))
        else:
            print("  [SKIP]  Could not find matching row in summary.json")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(passed)
    n_total = len(passed)
    print(f"\n{'='*50}")
    print(f"Result: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print(f"[{PASS}]  All checks passed — pipeline is working.")
    else:
        print(f"[{FAIL}]  {n_total - n_pass} check(s) failed.")
    print("="*50)

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
