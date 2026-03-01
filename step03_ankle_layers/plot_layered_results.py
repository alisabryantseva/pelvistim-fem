"""
plot_layered_results.py — visualize sweep results for step03_ankle_layers
=========================================================================
Reads results/summary.csv  +  per-case VTU files.

Usage (from step03_ankle_layers/):
    python3 plot_layered_results.py

Outputs (in results/):
    J_surface_maps.png   — |J| heatmaps on skin surface, grid by fat × radius
    summary_metrics.png  — ROI J, skin peak J, and tradeoff vs electrode area
"""

import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patches as mpatches
import pyvista as pv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
PARAMS_FILE = Path(__file__).parent / "params.yaml"


def load_params():
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def load_summary():
    jp = RESULTS_DIR / "summary.json"
    if not jp.exists():
        raise SystemExit("results/summary.json not found — run run_layered_sweep.py first")
    with open(jp) as f:
        return json.load(f)


def load_vtu(t_fat_mm, elec_r_mm):
    label = f"tfat{int(t_fat_mm):04d}um_r{int(elec_r_mm):04d}um"
    vtu   = RESULTS_DIR / label / "results" / "case_t0001.vtu"
    if not vtu.exists():
        return None
    return pv.read(str(vtu))


# ── Plot 1: |J| surface heatmaps ─────────────────────────────────────────────
def plot_J_surface_maps(summary, p):
    t_fats  = sorted(set(r["t_fat_mm"]  for r in summary))
    radii   = sorted(set(r["elec_r_mm"] for r in summary))
    nrows, ncols = len(t_fats), len(radii)

    g   = p["geometry"]
    Lx, Ly, Lz = g["Lx"], g["Ly"], g["Lz"]
    ep  = p["electrodes"]
    e1x = ep["medial_offset"]
    e2x = Lx - ep["lateral_offset"]
    ey  = Ly / 2

    # Collect all J values across cases for a shared color scale
    all_J = []
    meshes = {}
    for row in summary:
        m = load_vtu(row["t_fat_mm"], row["elec_r_mm"])
        if m is None:
            continue
        pts  = np.array(m.points)
        Jmag = np.linalg.norm(np.array(m.point_data["volume current"]), axis=1)
        mask = pts[:, 2] > Lz * 0.99
        all_J.extend(Jmag[mask].tolist())
        meshes[(row["t_fat_mm"], row["elec_r_mm"])] = (pts, Jmag, mask)

    if not all_J:
        print("No VTU data found for surface maps.")
        return

    vmin = np.percentile(all_J, 2)
    vmax = np.percentile(all_J, 98)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 4 * nrows),
        constrained_layout=True
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        "Current density |J| at skin surface (z = Lz)  —  ankle layered model\n"
        f"σ: skin={p['conductivities']['sigma_skin']}, "
        f"fat={p['conductivities']['sigma_fat']}, "
        f"muscle={p['conductivities']['sigma_muscle']} S/m  [PLACEHOLDER values]",
        fontsize=10, fontweight="bold"
    )

    shape = ep["shape"]
    for ri, tfat in enumerate(t_fats):
        for ci, r_mm in enumerate(radii):
            ax = axes[ri][ci]
            key = (tfat, r_mm)
            row = next((x for x in summary
                        if x["t_fat_mm"] == tfat and x["elec_r_mm"] == r_mm), None)

            if key not in meshes or row is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center")
                ax.axis("off")
                continue

            pts, Jmag, mask = meshes[key]
            xp, yp = pts[mask, 0], pts[mask, 1]

            try:
                tri = mtri.Triangulation(xp, yp)
                tc  = ax.tricontourf(tri, Jmag[mask], levels=30, cmap="inferno",
                                     vmin=vmin, vmax=vmax)
            except Exception:
                sc = ax.scatter(xp, yp, c=Jmag[mask], cmap="inferno",
                                vmin=vmin, vmax=vmax, s=4)
                tc = sc

            r_m = r_mm / 1000.0
            if shape == "circle":
                for xc, lbl, clr in [(e1x, "+1V", "cyan"), (e2x, "0V", "lime")]:
                    ax.add_patch(plt.Circle((xc, ey), r_m, fill=False,
                                            edgecolor=clr, lw=1.8, ls="--"))
                    ax.text(xc, ey, lbl, ha="center", va="center",
                            color=clr, fontsize=7, fontweight="bold")
            else:
                for xc, lbl, clr in [(e1x, "+1V", "cyan"), (e2x, "0V", "lime")]:
                    ax.add_patch(mpatches.Rectangle(
                        (xc - r_m, ey - r_m), 2*r_m, 2*r_m,
                        fill=False, edgecolor=clr, lw=1.8, ls="--"))
                    ax.text(xc, ey, lbl, ha="center", va="center",
                            color=clr, fontsize=7, fontweight="bold")

            ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            ax.set_title(
                f"fat={tfat:.0f} mm  |  r={r_mm:.0f} mm\n"
                f"peak|J|={row['peak_J_skin']:.1f}  "
                f"ROI|J|={row['mean_J_roi']:.3f} A/m²",
                fontsize=8
            )

    # Single shared colorbar
    sm = plt.cm.ScalarMappable(cmap="inferno",
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="|J| (A/m²)", shrink=0.6, pad=0.01)

    out = RESULTS_DIR / "J_surface_maps.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ── Plot 2: summary metrics ───────────────────────────────────────────────────
def plot_summary_metrics(summary, p):
    t_fats = sorted(set(r["t_fat_mm"] for r in summary))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(t_fats)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle(
        "Electrode size effects — ankle layered model  [PLACEHOLDER conductivities]\n"
        "Each curve = one fat thickness.  Horizontal axis = electrode area.",
        fontsize=10, fontweight="bold"
    )

    titles = [
        ("Skin peak |J| (comfort proxy)\nSmaller electrode → higher peak → more discomfort risk",
         "peak_J_skin", "Peak |J| at skin (A/m²)"),
        (f"ROI mean |J| (efficacy proxy)\nROI: sphere r={p['roi']['roi_radius']*1000:.0f}mm "
         f"at {p['roi']['z_target']*1000:.0f}mm depth under active electrode",
         "mean_J_roi",  "Mean |J| in ROI (A/m²)"),
        ("Tradeoff: ROI |J| / skin peak |J|\nHigher = more efficient stimulation for given skin exposure",
         "tradeoff",    "Tradeoff ratio (dimensionless)"),
    ]

    for ax, (title, key, ylabel) in zip(axes, titles):
        for tfat, clr in zip(t_fats, colors):
            sub = sorted([r for r in summary if r["t_fat_mm"] == tfat],
                         key=lambda x: x["elec_area_cm2"])
            if not sub:
                continue
            areas = [r["elec_area_cm2"] for r in sub]
            vals  = [r[key] for r in sub]
            rmms  = [r["elec_r_mm"] for r in sub]

            ax.plot(areas, vals, "o-", color=clr, lw=2, ms=7,
                    label=f"fat={tfat:.0f} mm")
            for a, v, rmm in zip(areas, vals, rmms):
                ax.annotate(f"r={rmm:.0f}", (a, v),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=7, color=clr)

        ax.set_xlabel("Electrode area (cm²)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=8.5)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=8, framealpha=0.85)

    out = RESULTS_DIR / "summary_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ── 3D pyvista render of one representative case ─────────────────────────────
def plot_3d_representative(summary, p):
    if not summary:
        return
    # Pick the middle case (median t_fat, middle radius)
    t_fats = sorted(set(r["t_fat_mm"] for r in summary))
    radii  = sorted(set(r["elec_r_mm"] for r in summary))
    row    = next((r for r in summary
                   if r["t_fat_mm"] == t_fats[len(t_fats)//2]
                   and r["elec_r_mm"] == radii[len(radii)//2]), None)
    if row is None:
        return

    m = load_vtu(row["t_fat_mm"], row["elec_r_mm"])
    if m is None:
        return

    g  = p["geometry"]
    Ly = g["Ly"]

    pl = pv.Plotter(off_screen=True, window_size=(800, 580))
    pl.set_background("white")
    clipped = m.clip(normal="y", origin=[0, Ly * 0.5, 0])
    pl.add_mesh(clipped, scalars="volume current",
                component=None,
                scalar_bar_args={"title": "|J| (A/m²)", "width": 0.45,
                                 "position_x": 0.27, "position_y": 0.04,
                                 "title_font_size": 12, "label_font_size": 10},
                cmap="inferno", show_scalar_bar=True)
    pl.add_mesh(m.outline(), color="gray", line_width=1.5)
    pl.view_isometric()
    pl.camera.zoom(1.15)
    out = RESULTS_DIR / "representative_3d.png"
    pl.screenshot(str(out))
    pl.close()
    print(f"Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p       = load_params()
    summary = load_summary()

    print(f"Loaded {len(summary)} cases.")
    plot_J_surface_maps(summary, p)
    plot_summary_metrics(summary, p)
    plot_3d_representative(summary, p)
    print("Done.")
