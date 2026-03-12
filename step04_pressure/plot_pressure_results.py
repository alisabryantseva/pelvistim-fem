"""
plot_pressure_results.py — visualise pressure sweep results
============================================================
Reads results/summary.json and generates:
  results/pressure_results.png  — 4-panel figure
    Panel 1: Compliance voltage vs pressure
    Panel 2: Contact impedance (V/I) vs pressure
    Panel 3: Charge density at skin vs pressure (+ safety limit line)
    Panel 4: ROI E-field at tibial nerve vs pressure

Usage (from step04_pressure/):
    python3 plot_pressure_results.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from datetime import date

RESULTS_DIR = Path(__file__).parent / "results"
SUMMARY     = RESULTS_DIR / "summary.json"


def load():
    with open(SUMMARY) as f:
        return json.load(f)


def main():
    data = load()
    if not data:
        print("No data in summary.json")
        return

    labels   = [r["pressure_label"]       for r in data]
    sigma_c  = [r["sigma_contact_Spm"]    for r in data]
    comp_V   = [r["compliance_V"]          for r in data]
    Z_ohm    = [r["contact_impedance_ohm"] for r in data]
    charge   = [r["charge_density_mC_cm2"] for r in data]
    roi_E    = [r["roi_mean_E"]            for r in data]

    charge_limit = 1.0   # mC/cm² — irreversible damage threshold

    # x-axis: sigma_contact (log scale)
    x = sigma_c
    x_ticks = sigma_c

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("black")
    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.tick_params(colors="white", labelsize=10)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    marker_kw = dict(marker="o", markersize=7, linewidth=2)
    xlabel    = "Contact conductivity σ_c (S/m)  [loose → tight]"

    # ── Panel 1: Compliance voltage ───────────────────────────────────────────
    ax = axes[0, 0]
    ax.semilogx(x, comp_V, color="#ff6b35", **marker_kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Compliance voltage (V)")
    ax.set_title("Compliance Voltage vs Wrap Pressure")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{s}\n{lbl}" for s, lbl in zip(sigma_c, labels)],
                       fontsize=8)
    ax.axhline(100, color="#666666", linestyle="--", linewidth=1,
               label="100 V ref")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#111111",
              edgecolor="#444444")
    ax.grid(True, color="#333333", linestyle="--", alpha=0.5)

    # ── Panel 2: Contact impedance ────────────────────────────────────────────
    ax = axes[0, 1]
    ax.loglog(x, Z_ohm, color="#4ecdc4", **marker_kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Contact impedance (Ω)")
    ax.set_title("Contact Impedance vs Wrap Pressure")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{s}\n{lbl}" for s, lbl in zip(sigma_c, labels)],
                       fontsize=8)
    ax.grid(True, color="#333333", linestyle="--", alpha=0.5,
            which="both")

    # ── Panel 3: Charge density ───────────────────────────────────────────────
    ax = axes[1, 0]
    ax.semilogx(x, charge, color="#ffd166", **marker_kw)
    ax.axhline(charge_limit, color="#ff4444", linestyle="--", linewidth=1.5,
               label=f"Safety limit {charge_limit} mC/cm²")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Peak charge density (mC/cm²)")
    ax.set_title("Skin Charge Density vs Wrap Pressure")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{s}\n{lbl}" for s, lbl in zip(sigma_c, labels)],
                       fontsize=8)
    ax.legend(fontsize=8, labelcolor="white", facecolor="#111111",
              edgecolor="#444444")
    ax.grid(True, color="#333333", linestyle="--", alpha=0.5)

    # ── Panel 4: ROI E-field ──────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.semilogx(x, roi_E, color="#a8dadc", **marker_kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ROI mean |E| (V/m)")
    ax.set_title("Tibial Nerve E-field vs Wrap Pressure")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{s}\n{lbl}" for s, lbl in zip(sigma_c, labels)],
                       fontsize=8)
    ax.grid(True, color="#333333", linestyle="--", alpha=0.5)

    # ── Metadata ──────────────────────────────────────────────────────────────
    d0 = data[0]
    meta = (f"Fixed: t_fat={d0['t_fat_mm']:.0f}mm  r={d0['elec_r_mm']:.0f}mm  "
            f"I={d0['I_active_A']*1e3:.1f}mA  "
            f"freq={d0['frequency_Hz']:.0f}Hz  "
            f"pw={d0['pulse_width_us']:.0f}µs  |  "
            f"{date.today()}")
    fig.text(0.5, 0.01, meta, ha="center", va="bottom",
             fontsize=8, color="#888888")

    fig.suptitle("Step 04 — Pressure-Dependent Contact Impedance Sweep",
                 color="white", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    out = RESULTS_DIR / "pressure_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    print(f"Saved → {out}")

    # ── Print sanity check table ───────────────────────────────────────────────
    print("\n  RESULTS SANITY CHECK")
    print(f"  {'Label':<10} {'σ_c(S/m)':>10}  {'V_comp':>8}  "
          f"{'Z(Ω)':>8}  {'Q(mC/cm²)':>12}  {'E_roi(V/m)':>12}  "
          f"{'flux_err%':>10}")
    print("  " + "-" * 78)
    for r in data:
        flag = " [!]" if r.get("exceeds_charge_limit") else ""
        fe   = r.get("flux_err", float("nan"))
        print(f"  {r['pressure_label']:<10} {r['sigma_contact_Spm']:>10.4f}  "
              f"{r['compliance_V']:>8.1f}  "
              f"{r['contact_impedance_ohm']:>8.0f}  "
              f"{r['charge_density_mC_cm2']:>12.5f}{flag}  "
              f"{r['roi_mean_E']:>12.2f}  "
              f"{fe*100:>10.2f}")
    print()


if __name__ == "__main__":
    main()
