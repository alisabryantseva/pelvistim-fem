"""
plot_layered_results.py — visualize sweep results for step03_ankle_layers
=========================================================================
Reads results/summary.json  +  per-case VTU files.

Usage (from step03_ankle_layers/):
    python3 plot_layered_results.py

Outputs (in results/):
    J_surface_maps_linear.png — |J| heatmaps, linear scale (vmax = vmax_percentile)
                                Best for: comparing absolute peak J across cases
    J_surface_maps_log.png    — same data, log color scale
                                Best for: seeing low-J spreading far from electrodes
    J_surface_maps_masked.png — linear scale with electrode footprints set to NaN
                                Best for: current spreading pattern outside electrodes
    summary_metrics.png       — Raw and normalised metrics vs electrode area
    representative_3d.png     — 3D pyvista render of one case

All three J maps are always generated regardless of params.yaml flags.
"""

import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    plot_cfg  = p.get("plotting", {})
    vmax_pct  = float(plot_cfg.get("vmax_percentile", 99.95))
    do_log    = bool(plot_cfg.get("log_norm", True))
    do_masked = bool(plot_cfg.get("make_masked", True))

    t_fats  = sorted(set(r["t_fat_mm"]  for r in summary))
    radii   = sorted(set(r["elec_r_mm"] for r in summary))
    nrows, ncols = len(t_fats), len(radii)

    g   = p["geometry"]
    Lx, Ly, Lz = g["Lx"], g["Ly"], g["Lz"]
    pl    = p.get("placement", p.get("electrodes", {}))
    shape = pl.get("electrode_shape", pl.get("shape", "circle"))
    e1x, e1y = pl.get("active_xy",
                       [pl.get("medial_offset",  0.025), Ly / 2])
    e2x, e2y = pl.get("return_xy",
                       [Lx - pl.get("lateral_offset", 0.025), Ly / 2])

    # ── Collect all skin-surface J values for a GLOBAL color scale ────────────
    all_J  = []
    meshes = {}
    for row in summary:
        m = load_vtu(row["t_fat_mm"], row["elec_r_mm"])
        if m is None:
            continue
        pts  = np.array(m.points)
        Jmag = np.linalg.norm(np.array(m.point_data["volume current"]), axis=1)
        # Skin surface is at z ≈ Lz regardless of contact layer (contact sits above it)
        mask = (pts[:, 2] > Lz * 0.99) & (pts[:, 2] < Lz * 1.02)
        all_J.extend(Jmag[mask].tolist())
        meshes[(row["t_fat_mm"], row["elec_r_mm"])] = (pts, Jmag, mask)

    if not all_J:
        print("No VTU data found for surface maps.")
        return

    vmin = 0.0
    vmax = float(np.percentile(all_J, vmax_pct))
    if vmax <= 0:
        vmax = float(np.max(all_J))

    sig_skin = summary[0].get("sigma_skin", p["conductivities"]["sigma_skin"])

    # Inferno colormap with black for NaN / out-of-domain
    _cmap = plt.cm.inferno.copy()
    _cmap.set_bad("black")
    _cmap.set_under("black")

    def _render_figure(norm, levels, out_name, title_suffix, mask_fn=None, footer=None):
        """Build and save one J-surface heatmap figure."""
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
            f"|J| at skin surface (z = Lz)  —  ankle layered model  {title_suffix}\n"
            f"σ_skin={sig_skin} S/m  fat={p['conductivities']['sigma_fat']} S/m  "
            f"muscle={p['conductivities']['sigma_muscle']} S/m  [PLACEHOLDER values]\n"
            f"Color scale: {vmin:.2g} – {vmax:.4f} A/m²  "
            f"({vmax_pct:.2f}th percentile max)   [black = outside domain or masked]",
            fontsize=9, fontweight="bold"
        )

        for ri, tfat in enumerate(t_fats):
            for ci, r_mm in enumerate(radii):
                ax  = axes[ri][ci]
                ax.set_facecolor("black")   # outside-domain regions appear black
                key = (tfat, r_mm)
                row = next((x for x in summary
                            if x["t_fat_mm"] == tfat and x["elec_r_mm"] == r_mm), None)

                if key not in meshes or row is None:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                            ha="center", va="center", color="white")
                    ax.axis("off")
                    continue

                pts, Jmag, smask = meshes[key]
                xp     = pts[smask, 0]
                yp     = pts[smask, 1]
                Jvals  = Jmag[smask].copy()

                if mask_fn is not None:
                    r_m   = r_mm / 1000.0
                    Jvals = mask_fn(Jvals, xp, yp, r_m)

                # Exclude non-finite and (for log) non-positive values
                # Excluded points leave black holes via the axes background
                valid = np.isfinite(Jvals)
                if isinstance(norm, mcolors.LogNorm):
                    valid = valid & (Jvals > 1e-12)

                if valid.sum() < 3:
                    ax.text(0.5, 0.5, "no valid data", transform=ax.transAxes,
                            ha="center", va="center", color="white")
                else:
                    try:
                        tri = mtri.Triangulation(xp[valid], yp[valid])
                        ax.tricontourf(tri, Jvals[valid], levels=levels,
                                       cmap=_cmap, norm=norm, extend="both")
                    except Exception:
                        ax.scatter(xp[valid], yp[valid], c=Jvals[valid],
                                   cmap=_cmap, norm=norm, s=4)

                r_m = r_mm / 1000.0
                for (xc, yc), lbl, clr in [
                        ((e1x, e1y), "+I", "cyan"),
                        ((e2x, e2y), "0V", "lime")]:
                    if shape == "circle":
                        ax.add_patch(plt.Circle((xc, yc), r_m, fill=False,
                                                edgecolor=clr, lw=1.8, ls="--"))
                    else:
                        ax.add_patch(mpatches.Rectangle(
                            (xc - r_m, yc - r_m), 2*r_m, 2*r_m,
                            fill=False, edgecolor=clr, lw=1.8, ls="--"))
                    ax.text(xc, yc, lbl, ha="center", va="center",
                            color=clr, fontsize=7, fontweight="bold")

                ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
                ax.set_aspect("equal")
                ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

                pk  = row.get("peak_J_skin_no_elec",
                              row.get("peak_J_skin", float("nan")))
                rj  = row.get("roi_mean_J", row.get("mean_J_roi", float("nan")))
                ax.set_title(
                    f"fat={tfat:.0f} mm  |  r={r_mm:.0f} mm\n"
                    f"σ_skin={row.get('sigma_skin', sig_skin)}\n"
                    f"peak|J|(no-elec)={pk:.4f}  ROI|J|={rj:.4f} A/m²",
                    fontsize=7.5
                )

        sm = plt.cm.ScalarMappable(cmap=_cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, label="|J| (A/m²)", shrink=0.6, pad=0.01)

        if footer:
            fig.text(0.5, -0.01, footer, ha="center", va="top",
                     fontsize=8, style="italic", color="gray")

        out = RESULTS_DIR / out_name
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.close(fig)

    # ── Linear plot (always generated) ────────────────────────────────────────
    lin_norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    lin_levels = np.linspace(vmin, vmax, 31)
    _render_figure(lin_norm, lin_levels, "J_surface_maps_linear.png",
                   f"(linear, {vmax_pct:.2f}th pct max)")

    # ── Log-scale plot (always generated) ────────────────────────────────────
    pos_J      = [j for j in all_J if j > 0]
    log_vmin   = max(1e-6, float(np.percentile(pos_J, 1))) if pos_J else 1e-6
    log_norm   = mcolors.LogNorm(vmin=log_vmin, vmax=vmax)
    log_levels = np.logspace(np.log10(log_vmin), np.log10(vmax), 30)
    _render_figure(log_norm, log_levels, "J_surface_maps_log.png",
                   "(log scale — reveals low-J spreading far from electrodes)")

    # ── Masked-electrode plot (always generated) ──────────────────────────────
    def _mask_electrodes(Jvals, xp, yp, r_m):
        Jout = Jvals.copy().astype(float)
        for ex, ey in [(e1x, e1y), (e2x, e2y)]:
            if shape == "circle":
                inside = np.sqrt((xp - ex)**2 + (yp - ey)**2) < r_m
            else:
                inside = (np.abs(xp - ex) < r_m) & (np.abs(yp - ey) < r_m)
            Jout[inside] = np.nan
        return Jout

    msk_norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    msk_levels = np.linspace(vmin, vmax, 31)
    _render_figure(msk_norm, msk_levels, "J_surface_maps_masked.png",
                   "(electrode footprints masked — shows spreading outside pads)",
                   mask_fn=_mask_electrodes,
                   footer="Gray regions under electrode outlines are masked (NaN). "
                          "Color shows J at skin surface outside electrode footprints only.")


# ── Plot 2: raw + normalised summary metrics ──────────────────────────────────
def plot_summary_metrics(summary, p):
    t_fats = sorted(set(r["t_fat_mm"] for r in summary))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(t_fats)))

    sig_skin = summary[0].get("sigma_skin", p["conductivities"]["sigma_skin"])
    mode     = summary[0].get("control_mode", "voltage")

    roi_r_mm = p["roi"]["roi_radius"] * 1000
    z_tgt_mm = p["roi"]["z_target"] * 1000

    st = p.get("stim", p.get("control", {}))
    compliance_lim  = st.get("compliance_voltage_V", 100.0)
    I_target_mA     = st.get("injected_current_mA", 5.0) if mode == "current" else None

    # ── Panel definitions ─────────────────────────────────────────────────────
    # Row 0: raw metrics
    raw_panels = [
        ("Skin peak |J| (no-electrode, comfort proxy)\n"
         "Smaller electrode → higher peak → more discomfort risk",
         "peak_J_skin_no_elec",
         "Peak |J| outside electrode (A/m²)"),
        (f"ROI mean |E|  (efficacy proxy)\n"
         f"Sphere r={roi_r_mm:.0f}mm at {z_tgt_mm:.0f}mm depth",
         "roi_mean_E",
         "Mean |E| in ROI (V/m)"),
        ("Efficiency = ROI |E| / skin peak |J|\n"
         "Higher = more deep field per unit skin exposure  [m]",
         "efficiency",
         "Efficiency  (V/m) / (A/m²) = m"),
    ]
    # Row 1: current mode — total_current check + compliance; voltage mode — normalised
    if mode == "current":
        norm_panels = [
            ("Injected current verification\n"
             "Should be flat at target I — deviations indicate mesh/BC issues",
             "total_current_A_mA",   # virtual key: we multiply A → mA in draw
             f"I_active  (mA)"),
            ("ROI mean |E| / I_injected\n"
             "Transfer function: deep E-field per unit injected current",
             "roi_mean_E_per_A",
             "ROI mean |E| / I  (V/m/A)"),
            ("Required electrode voltage  (compliance)\n"
             "Larger electrode → lower V; red line = compliance limit",
             "compliance_V",
             "V_active  (V)"),
        ]
    else:
        norm_panels = [
            ("Skin peak |J| / I_injected\n"
             "Comparable across voltage & current modes",
             "peak_J_skin_per_A",
             "Peak |J|(no-elec) / I  (1/m²)"),
            ("ROI mean |E| / I_injected\n"
             "Transfer function: deep E-field per unit injected current",
             "roi_mean_E_per_A",
             "ROI mean |E| / I  (V/m/A)"),
            ("Required electrode voltage  (fixed 1V in voltage mode)\n"
             "Run in current mode to see compliance variation",
             "compliance_V",
             "V_active  (V)"),
        ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        f"Electrode size effects — ankle cross-section model\n"
        f"σ_skin={sig_skin} S/m  |  mode={mode}  |  PLACEHOLDER conductivities\n"
        "Row 0: raw metrics.  Row 1: "
        + ("injected I check + normalised E + compliance." if mode == "current"
           else "normalised by I_injected + compliance."),
        fontsize=10, fontweight="bold"
    )

    def _draw_panel(ax, key, ylabel, title):
        for tfat, clr in zip(t_fats, colors):
            sub = sorted(
                [r for r in summary if r["t_fat_mm"] == tfat],
                key=lambda x: x["elec_area_cm2"])
            if not sub:
                continue
            areas, vals, rmms = [], [], []
            for r in sub:
                if key == "total_current_A_mA":
                    v_raw = r.get("total_current_A")
                    v = v_raw * 1e3 if (v_raw is not None and not (isinstance(v_raw, float) and np.isnan(v_raw))) else None
                else:
                    v = r.get(key)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                areas.append(r["elec_area_cm2"])
                vals.append(v)
                rmms.append(r["elec_r_mm"])
            if not areas:
                continue
            ax.plot(areas, vals, "o-", color=clr, lw=2, ms=7,
                    label=f"fat={tfat:.0f} mm")
            for a, v, rmm in zip(areas, vals, rmms):
                ax.annotate(f"r={rmm:.0f}", (a, v),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=7, color=clr)

        if key == "total_current_A_mA" and I_target_mA is not None:
            ax.axhline(I_target_mA, color="green", lw=1.5, ls="--",
                       label=f"target I = {I_target_mA:.1f} mA")
        if key == "compliance_V" and mode == "current":
            ax.axhline(compliance_lim, color="red", lw=1.2, ls="--",
                       label=f"compliance limit ({compliance_lim:.0f} V)")

        ax.set_xlabel("Electrode area (cm²)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=8.5, pad=6)
        ax.set_xscale("log")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=8, framealpha=0.85, loc="best")

    for row_idx, panels in enumerate([raw_panels, norm_panels]):
        for col_idx, (title, key, ylabel) in enumerate(panels):
            _draw_panel(axes[row_idx][col_idx], key, ylabel, title)

    out = RESULTS_DIR / "summary_metrics.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ── 3D pyvista render of one representative case ──────────────────────────────
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
    Lx, Ly, Lz = g["Lx"], g["Ly"], g["Lz"]
    pl_cfg = p.get("placement", p.get("electrodes", {}))
    shape  = pl_cfg.get("electrode_shape", "circle")
    e1x, e1y = pl_cfg.get("active_xy", [pl_cfg.get("medial_offset", 0.025), Ly/2])
    e2x, e2y = pl_cfg.get("return_xy", [Lx - pl_cfg.get("lateral_offset", 0.025), Ly/2])
    r_m = row["elec_r_mm"] / 1000.0

    # Compute |J| magnitude and attach to mesh
    J    = np.array(m.point_data["volume current"])
    Jmag = np.linalg.norm(J, axis=1)
    m.point_data["Jmag"] = Jmag

    # Extract top skin surface (z ≈ Lz); contact layer is slightly above, skin at Lz
    surf = m.extract_surface()
    cc_z = np.array(surf.cell_centers().points)[:, 2]
    top_ids = np.where((cc_z > Lz * 0.97) & (cc_z < Lz * 1.05))[0]
    skin_surf = surf.extract_cells(top_ids) if len(top_ids) > 10 else surf

    # Electrode outline polylines (circles or squares) at z = Lz + small offset
    z_elec = Lz + 1e-4
    def _elec_ring(xc, yc, r):
        theta = np.linspace(0, 2 * np.pi, 60)
        if shape == "circle":
            pts = np.column_stack([xc + r * np.cos(theta),
                                   yc + r * np.sin(theta),
                                   np.full(60, z_elec)])
        else:
            corners = np.array([
                [xc - r, yc - r, z_elec], [xc + r, yc - r, z_elec],
                [xc + r, yc + r, z_elec], [xc - r, yc + r, z_elec],
                [xc - r, yc - r, z_elec],
            ])
            return pv.Spline(corners, 20)
        return pv.Spline(pts, 60)

    Jmag_all  = skin_surf.point_data["Jmag"] if "Jmag" in skin_surf.point_data else None
    Jmag_vals = Jmag_all if Jmag_all is not None else np.zeros(skin_surf.n_points)
    vmax_3d   = float(np.percentile(Jmag_vals, 99.9)) if len(Jmag_vals) > 0 else 1.0

    pl_pv = pv.Plotter(off_screen=True, window_size=(900, 720))
    pl_pv.set_background("black")
    pl_pv.add_mesh(
        skin_surf, scalars="Jmag", cmap="inferno",
        clim=[0, vmax_3d],
        show_scalar_bar=True,
        scalar_bar_args={"title": "|J| (A/m²)", "width": 0.40,
                         "position_x": 0.30, "position_y": 0.04,
                         "title_font_size": 12, "label_font_size": 10,
                         "color": "white"},
    )
    pl_pv.add_mesh(_elec_ring(e1x, e1y, r_m), color="cyan",  line_width=4)
    pl_pv.add_mesh(_elec_ring(e2x, e2y, r_m), color="lime",  line_width=4)
    # Add text labels
    pl_pv.add_point_labels(
        np.array([[e1x, e1y, z_elec + 2e-3], [e2x, e2y, z_elec + 2e-3]]),
        ["+I (active)", "0V (return)"],
        font_size=12, text_color="white", shape_color="black",
        show_points=False, always_visible=True,
    )
    pl_pv.view_xy()          # top-down view so both pads are visible
    pl_pv.camera.zoom(1.05)

    out = RESULTS_DIR / "representative_3d.png"
    pl_pv.screenshot(str(out))
    pl_pv.close()
    print(f"Saved → {out}")


# ── Sanity check table ────────────────────────────────────────────────────────
def print_sanity_table(summary):
    """Print a compact per-case verification table to the console."""
    if not summary:
        return
    mode = summary[0].get("control_mode", "voltage")
    print(f"\n{'='*80}")
    print("  RESULTS SANITY CHECK")
    print(f"{'='*80}")
    hdr = (f"  {'r(mm)':>6}  {'fat(mm)':>7}  "
           f"{'I_active(mA)':>13}  {'I_return(mA)':>13}  {'flux_err(%)':>11}")
    if mode == "current":
        hdr += f"  {'compliance_V':>13}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in sorted(summary, key=lambda x: (x["elec_r_mm"], x["t_fat_mm"])):
        I_a = row.get("total_current_A", float("nan"))
        I_r = row.get("I_return_A",      float("nan"))
        fe  = row.get("flux_err",        float("nan"))
        line = (f"  {row['elec_r_mm']:>6.0f}  {row['t_fat_mm']:>7.1f}  "
                f"{I_a*1e3:>13.3f}  {I_r*1e3:>13.3f}  {fe*100:>10.2f}%")
        if mode == "current":
            cV = row.get("compliance_V", float("nan"))
            ec = row.get("exceeded_compliance", False)
            line += f"  {cV:>12.2f}V" + ("  [!]" if ec else "")
        print(line)
    print(f"{'='*80}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p       = load_params()
    summary = load_summary()

    print(f"Loaded {len(summary)} cases.")
    plot_J_surface_maps(summary, p)
    plot_summary_metrics(summary, p)
    plot_3d_representative(summary, p)
    print_sanity_table(summary)
    print("Done.")
