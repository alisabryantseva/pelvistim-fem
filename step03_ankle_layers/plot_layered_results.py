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
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import pyvista as pv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
PARAMS_FILE = Path(__file__).parent / "params.yaml"


# ── Ankle outline (mirrors run_layered_sweep.py) ──────────────────────────────
def ankle_outline_pts(Lx, Ly):
    """
    12-point ankle cross-section polygon (same as run_layered_sweep.py).
    x = medial (0) → lateral (Lx).   y = anterior (0) → posterior (Ly).
    P7 (0.50, 1.00) = posterior center ≈ Achilles tendon.
    P10 (0.02, 0.47) = medial groove  ← active electrode.
    P5  (0.93, 0.72) = posterior-lateral ← return electrode region.
    """
    frac = [
        (0.25, 0.02),   # P0  anterior-medial
        (0.50, 0.00),   # P1  anterior center
        (0.75, 0.02),   # P2  anterior-lateral
        (0.97, 0.22),   # P3  lateral-anterior
        (1.00, 0.47),   # P4  lateral-mid
        (0.93, 0.72),   # P5  posterior-lateral
        (0.75, 0.97),   # P6  posterior-lateral end
        (0.50, 1.00),   # P7  posterior center  ← Achilles tendon
        (0.25, 0.97),   # P8  posterior-medial end
        (0.07, 0.72),   # P9  medial-posterior
        (0.02, 0.47),   # P10 medial groove
        (0.07, 0.22),   # P11 medial-anterior
    ]
    return [(fx * Lx, fy * Ly) for fx, fy in frac]


def _ankle_mpl_path(Lx, Ly):
    """Return a closed matplotlib Path tracing the ankle polygon."""
    pts = ankle_outline_pts(Lx, Ly)
    pts_c = pts + [pts[0]]
    codes = ([MplPath.MOVETO]
             + [MplPath.LINETO] * (len(pts) - 1)
             + [MplPath.CLOSEPOLY])
    return MplPath(pts_c, codes)


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

    g     = p["geometry"]
    Lx, Ly, Lz = g["Lx"], g["Ly"], g["Lz"]
    cross = g.get("cross_section", "rect")
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

    def _render_figure(norm, levels, out_name, title_suffix, mask_fn=None,
                       footer=None, add_contours=False):
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

                # ── Build ankle clip path (for masking triangles + electrode outlines) ─
                a_path = _ankle_mpl_path(Lx, Ly) if cross == "ankle" else None

                if valid.sum() < 3:
                    ax.text(0.5, 0.5, "no valid data", transform=ax.transAxes,
                            ha="center", va="center", color="white")
                else:
                    try:
                        tri = mtri.Triangulation(xp[valid], yp[valid])
                        # Mask triangles whose centroid falls outside the ankle polygon
                        if a_path is not None:
                            cx_ = xp[valid][tri.triangles].mean(axis=1)
                            cy_ = yp[valid][tri.triangles].mean(axis=1)
                            outside = ~a_path.contains_points(
                                np.column_stack([cx_, cy_]))
                            tri.set_mask(outside)
                        ax.tricontourf(tri, Jvals[valid], levels=levels,
                                       cmap=_cmap, norm=norm, extend="both")
                        # Contour lines for masked map (shows spreading pattern)
                        if add_contours and isinstance(norm, mcolors.Normalize):
                            J_fin = Jvals[valid]
                            J_fin_pos = J_fin[np.isfinite(J_fin) & (J_fin > 0)]
                            if len(J_fin_pos) > 0:
                                vmax_c = float(np.nanmax(J_fin_pos))
                                for frac, ls_c in [(0.10, ":"), (0.25, "--"), (0.50, "-")]:
                                    lvl_c = vmax_c * frac
                                    if lvl_c > 0:
                                        ax.tricontour(tri, Jvals[valid],
                                                      levels=[lvl_c],
                                                      colors=["white"],
                                                      linewidths=[0.7],
                                                      linestyles=[ls_c],
                                                      alpha=0.55)
                    except Exception:
                        ax.scatter(xp[valid], yp[valid], c=Jvals[valid],
                                   cmap=_cmap, norm=norm, s=4)

                # ── Draw ankle outline ──────────────────────────────────────────────
                if a_path is not None:
                    apt = ankle_outline_pts(Lx, Ly)
                    xs_ = [q[0] for q in apt] + [apt[0][0]]
                    ys_ = [q[1] for q in apt] + [apt[0][1]]
                    ax.plot(xs_, ys_, color="white", lw=0.9, alpha=0.55, zorder=3)

                # ── Electrode outlines — arcs clipped to ankle polygon ──────────────
                r_m = r_mm / 1000.0

                for (xc, yc), lbl, clr in [
                        ((e1x, e1y), "+I", "cyan"),
                        ((e2x, e2y), "0V", "lime")]:
                    if shape == "circle":
                        # Parametric arc; insert NaN where outside ankle polygon
                        theta = np.linspace(0, 2 * np.pi, 721)
                        ax_pts = xc + r_m * np.cos(theta)
                        ay_pts = yc + r_m * np.sin(theta)
                        if a_path is not None:
                            inside = a_path.contains_points(
                                np.column_stack([ax_pts, ay_pts]))
                            ax_pts = np.where(inside, ax_pts, np.nan)
                            ay_pts = np.where(inside, ay_pts, np.nan)
                        ax.plot(ax_pts, ay_pts, color=clr, lw=1.8,
                                ls="--", zorder=4)
                    else:
                        patch = mpatches.Rectangle(
                            (xc - r_m, yc - r_m), 2*r_m, 2*r_m,
                            fill=False, edgecolor=clr, lw=1.8, ls="--", zorder=4)
                        ax.add_patch(patch)
                    ax.text(xc, yc, lbl, ha="center", va="center",
                            color=clr, fontsize=7, fontweight="bold", zorder=5)

                # ── Achilles tendon landmark ────────────────────────────────────────
                # P7 (0.50, 1.00) = posterior center = Achilles tendon
                at_x = Lx * 0.50
                at_y = Ly * 0.96   # just inside posterior boundary
                ax.plot(at_x, at_y, '^', color='white', ms=5,
                        mfc='white', mew=1.0, zorder=5)
                ax.text(at_x, at_y - Ly * 0.05, 'AT', ha='center', va='top',
                        color='white', fontsize=5, fontweight='bold', zorder=5)

                ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
                ax.set_aspect("equal")
                # Anatomical direction labels
                ax.set_xlabel("Medial → Lateral  (m)", fontsize=7)
                ax.set_ylabel("Anterior → Posterior  (m)", fontsize=7)

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
                   add_contours=True,
                   footer="Black under electrode outlines = masked (NaN). "
                          "White contour lines: 10/25/50% of local |J| max. "
                          "Color shows J at skin surface outside electrode footprints.")


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


# ── Plot 4: horizontal depth-slice |E| maps ───────────────────────────────────
def plot_depth_slice_E_maps(summary, p):
    """
    For each case: slice the 3D mesh at z = z_nerve (ROI depth) and plot |E|.
    E = -grad(Potential) computed by pyvista, then sliced at the target depth.

    Output: results/depth_slice_E_maps.png
    """
    t_fats = sorted(set(r["t_fat_mm"]  for r in summary))
    radii  = sorted(set(r["elec_r_mm"] for r in summary))
    nrows, ncols = len(t_fats), len(radii)

    g  = p["geometry"]
    Lx, Ly, Lz = g["Lx"], g["Ly"], g["Lz"]
    ls = p["layers"]
    r_cfg = p["roi"]
    pl_cfg = p.get("placement", p.get("electrodes", {}))
    shape  = pl_cfg.get("electrode_shape", "circle")
    e1x, e1y = pl_cfg.get("active_xy",  [0.015, 0.045])
    e2x, e2y = pl_cfg.get("return_xy",  [0.065, 0.045])
    t_skin = ls["t_skin"]
    z_target = r_cfg["z_target"]
    z_skin_top = Lz   # skin top (no contact layer offset for display)

    _cmap = plt.cm.inferno.copy()
    _cmap.set_bad("black")
    _cmap.set_under("black")

    # ── Collect |E| at slice depth for global color scale ─────────────────────
    all_E  = []
    slices = {}
    for row in summary:
        m = load_vtu(row["t_fat_mm"], row["elec_r_mm"])
        if m is None:
            continue
        phi_key = next((k for k in ("Potential", "potential")
                        if k in m.point_data), None)
        if phi_key is None:
            continue
        try:
            grad_m = m.compute_derivative(scalars=phi_key)
            E_pts  = -np.array(grad_m.point_data["gradient"])
            Emag   = np.linalg.norm(E_pts, axis=1)
            grad_m["Emag"] = Emag
        except Exception as exc:
            print(f"    depth_slice: grad failed for "
                  f"fat={row['t_fat_mm']} r={row['elec_r_mm']}: {exc}")
            continue

        # Slice at ROI depth. z_nerve is measured from skin top downward.
        z_nerve = z_skin_top - z_target
        try:
            slc = grad_m.slice(normal="z", origin=[Lx/2, Ly/2, z_nerve])
            if slc.n_points < 3:
                continue
        except Exception:
            continue

        xp  = np.array(slc.points[:, 0])
        yp  = np.array(slc.points[:, 1])
        Ep  = np.array(slc.point_data["Emag"])
        valid = np.isfinite(Ep) & (Ep > 0)
        all_E.extend(Ep[valid].tolist())
        slices[(row["t_fat_mm"], row["elec_r_mm"])] = (xp, yp, Ep, row)

    if not slices:
        print("  depth_slice_E_maps: no slice data found (no VTU or no Potential field).")
        return

    vmax_pct = float(p.get("plotting", {}).get("vmax_percentile", 99.95))
    vmax_E   = float(np.percentile(all_E, vmax_pct)) if all_E else 1.0
    vmin_E   = 0.0
    norm_E   = mcolors.Normalize(vmin=vmin_E, vmax=vmax_E)
    levels_E = np.linspace(vmin_E, vmax_E, 31)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 4 * nrows),
        constrained_layout=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    z_nerve_mm = (z_skin_top - z_target) * 1000
    fig.suptitle(
        f"|E| horizontal slice at z = {z_nerve_mm:.1f} mm  "
        f"(ROI depth: {z_target*1000:.0f} mm below skin top)\n"
        f"Color scale: 0 – {vmax_E:.2f} V/m  ({vmax_pct:.2f}th pct)   "
        f"[black = no data]",
        fontsize=9, fontweight="bold")

    for ri, tfat in enumerate(t_fats):
        for ci, r_mm in enumerate(radii):
            ax  = axes[ri][ci]
            ax.set_facecolor("black")
            key = (tfat, r_mm)
            if key not in slices:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="white")
                ax.axis("off")
                continue

            xp, yp, Ep, row = slices[key]
            valid = np.isfinite(Ep)

            if valid.sum() >= 3:
                try:
                    tri = mtri.Triangulation(xp[valid], yp[valid])
                    ax.tricontourf(tri, Ep[valid], levels=levels_E,
                                   cmap=_cmap, norm=norm_E, extend="both")
                    # Contour lines at 25%, 50%, 75% of vmax
                    for frac, ls_style in [(0.25, ":"), (0.50, "--"), (0.75, "-")]:
                        lvl = vmax_E * frac
                        if lvl > Ep[valid].min():
                            ax.tricontour(tri, Ep[valid], levels=[lvl],
                                          colors=["white"], linewidths=[0.8],
                                          linestyles=[ls_style], alpha=0.6)
                except Exception:
                    ax.scatter(xp[valid], yp[valid], c=Ep[valid],
                               cmap=_cmap, norm=norm_E, s=4)

            # Electrode outline circles
            r_m = r_mm / 1000.0
            for (xc, yc), lbl, clr in [
                    ((e1x, e1y), "+I", "cyan"),
                    ((e2x, e2y), "0V", "lime")]:
                if shape == "circle":
                    theta = np.linspace(0, 2 * np.pi, 361)
                    ax.plot(xc + r_m * np.cos(theta),
                            yc + r_m * np.sin(theta),
                            color=clr, lw=1.5, ls="--", zorder=4)
                else:
                    patch = mpatches.Rectangle(
                        (xc - r_m, yc - r_m), 2*r_m, 2*r_m,
                        fill=False, edgecolor=clr, lw=1.5, ls="--", zorder=4)
                    ax.add_patch(patch)
                ax.text(xc, yc, lbl, ha="center", va="center",
                        color=clr, fontsize=7, fontweight="bold", zorder=5)

            # ROI circle
            roi_r = r_cfg["roi_radius"]
            roi_theta = np.linspace(0, 2*np.pi, 181)
            ax.plot(e1x + roi_r * np.cos(roi_theta),
                    e1y + roi_r * np.sin(roi_theta),
                    color="yellow", lw=1.2, ls="-", alpha=0.8, zorder=4)
            ax.text(e1x, e1y - roi_r * 1.4, "ROI",
                    ha="center", va="top", color="yellow", fontsize=6, zorder=5)

            # Achilles tendon marker (same position as J maps)
            ax.plot(Lx * 0.50, Ly * 0.96, '^', color='white', ms=5,
                    mfc='white', mew=1.0, zorder=5)
            ax.text(Lx * 0.50, Ly * 0.96 - Ly * 0.05, 'AT',
                    ha='center', va='top', color='white',
                    fontsize=5, fontweight='bold', zorder=5)

            ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
            ax.set_aspect("equal")
            ax.set_xlabel("Medial → Lateral  (m)", fontsize=7)
            ax.set_ylabel("Anterior → Posterior  (m)", fontsize=7)
            re_val = row.get("roi_mean_E", float("nan"))
            ax.set_title(
                f"fat={tfat:.0f} mm  |  r={r_mm:.0f} mm\n"
                f"ROI mean|E|={re_val:.3f} V/m  "
                f"(white contours: 25/50/75% of vmax)",
                fontsize=7.5)

    sm = plt.cm.ScalarMappable(cmap=_cmap, norm=norm_E)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="|E| (V/m)", shrink=0.6, pad=0.01)

    out = RESULTS_DIR / "depth_slice_E_maps.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ── Plot 5: anatomical model diagram + current profile ────────────────────────
def plot_model_diagram(p, summary=None):
    """
    3-panel model illustration:
      Left   — anatomy side view (x–z): skin/fat/muscle layers, electrodes,
               current path arrows, ROI sphere, layer conductivities
      Middle — top view (x–y): skin surface with both electrode footprints,
               anatomical landmarks, current spreading arc
      Right  — |J| vs depth profile below active electrode (from simulation data
               if available, otherwise annotated schematic). Horizontal lines
               mark layer boundaries. Answers: "how much current at each depth?"

    Output: results/model_diagram.png
    """
    g  = p["geometry"]
    Lx, Ly, Lz = g["Lx"], g["Ly"], g["Lz"]
    ls = p["layers"]
    t_skin = ls["t_skin"]
    t_fat  = ls.get("t_fat", ls.get("t_fat_sweep", [0.005])[1])
    t_musc = Lz - t_skin - t_fat

    pl_cfg   = p.get("placement", p.get("electrodes", {}))
    shape    = pl_cfg.get("electrode_shape", "circle")
    e1x, e1y = pl_cfg.get("active_xy",  [0.015, 0.045])
    e2x, e2y = pl_cfg.get("return_xy",  [0.065, 0.045])
    r_list   = pl_cfg.get("electrode_r_mm_list", [10])
    r_mid_mm = r_list[len(r_list) // 2]
    r_m      = r_mid_mm / 1000.0

    ct        = p.get("contact", {})
    t_contact = ct.get("t_contact_mm", 0.5) * 1e-3 if ct.get("enabled") else 0.0

    r_cfg   = p["roi"]
    z_tgt   = r_cfg["z_target"]
    roi_r   = r_cfg["roi_radius"]
    z_skin_top = Lz
    z_nerve    = z_skin_top - z_tgt
    z_fat_bot  = z_skin_top - t_skin - t_fat   # fat–muscle interface

    c = p["conductivities"]

    LAYER_COLORS = {
        "muscle":  "#8B4513",
        "fat":     "#D4A800",
        "skin":    "#C68B59",
        "contact": "#8080FF",
    }
    BG = "#111111"
    TC = "white"

    # ── Try to load a representative VTU for data-driven J profile ────────────
    vtu_mesh = None
    vtu_label = None
    if summary:
        t_fats = sorted(set(r["t_fat_mm"] for r in summary))
        radii  = sorted(set(r["elec_r_mm"] for r in summary))
        mid_t  = t_fats[len(t_fats) // 2]
        mid_r  = radii[len(radii) // 2]
        vtu_mesh  = load_vtu(mid_t, mid_r)
        vtu_label = f"fat={mid_t:.0f}mm  r={mid_r:.0f}mm"

    # ── Figure layout: 3 panels, left widest ─────────────────────────────────
    fig = plt.figure(figsize=(18, 7), constrained_layout=True)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.8, 1.6])
    ax_side = fig.add_subplot(gs[0])
    ax_top  = fig.add_subplot(gs[1])
    ax_prof = fig.add_subplot(gs[2])

    for ax in (ax_side, ax_top, ax_prof):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TC, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # ─── Panel 1: Side view (x–z) ─────────────────────────────────────────────
    def _rect(ax, x0, z0, w, h, color, alpha=0.82, label=None, fs=9):
        patch = mpatches.FancyBboxPatch(
            (x0, z0), w, h, boxstyle="square,pad=0",
            facecolor=color, edgecolor="white", linewidth=0.7, alpha=alpha)
        ax.add_patch(patch)
        if label:
            ax.text(x0 + w/2, z0 + h/2, label, ha="center", va="center",
                    color="white", fontsize=fs, fontweight="bold",
                    multialignment="center")

    _rect(ax_side, 0, 0,          Lx, t_musc, LAYER_COLORS["muscle"],
          label=f"MUSCLE\nσ = {c['sigma_muscle']} S/m\n"
                f"({t_musc*1000:.1f} mm thick)")
    _rect(ax_side, 0, t_musc,     Lx, t_fat,  LAYER_COLORS["fat"],
          label=f"FAT  σ={c['sigma_fat']} S/m  ({t_fat*1000:.1f}mm)", fs=8)
    _rect(ax_side, 0, t_musc+t_fat, Lx, t_skin, LAYER_COLORS["skin"],
          label=f"SKIN  σ={c['sigma_skin']} S/m  ({t_skin*1000:.1f}mm)", fs=7.5)

    if t_contact > 0:
        for xc in (e1x, e2x):
            _rect(ax_side, xc - r_m, Lz, 2*r_m, t_contact,
                  LAYER_COLORS["contact"], alpha=0.75,
                  label=f"contact\nσ={c.get('sigma_contact_Spm', ct.get('sigma_contact_Spm', '?'))} S/m",
                  fs=6)

    # Electrodes
    z_elec_line = Lz + t_contact + 0.0008
    for xc, clr, lbl in [(e1x, "cyan",  "+I\nactive"),
                          (e2x, "lime",  "0V\nreturn")]:
        ax_side.plot([xc - r_m, xc + r_m], [z_elec_line, z_elec_line],
                     color=clr, lw=5, solid_capstyle="butt", zorder=5)
        ax_side.text(xc, z_elec_line + 0.0018, lbl, ha="center", va="bottom",
                     color=clr, fontsize=8, fontweight="bold", zorder=6,
                     multialignment="center")

    # Current path arrows (active: down into tissue; return: up out)
    ax_side.annotate("",
        xy=(e1x - 0.003, 0.002), xytext=(e1x - 0.003, Lz),
        arrowprops=dict(arrowstyle="-|>", color="cyan", lw=2.0,
                        connectionstyle="arc3,rad=0.0"))
    ax_side.annotate("",
        xy=(e2x + 0.003, Lz), xytext=(e2x + 0.003, 0.002),
        arrowprops=dict(arrowstyle="-|>", color="lime", lw=2.0,
                        connectionstyle="arc3,rad=0.0"))
    # Horizontal arc near bottom connecting the two paths
    ax_side.annotate("",
        xy=(e2x + 0.003, 0.004), xytext=(e1x - 0.003, 0.004),
        arrowprops=dict(arrowstyle="-", color="white", lw=1.2,
                        connectionstyle="arc3,rad=0.25", alpha=0.5))

    # ROI
    roi_c = plt.Circle((e1x, z_nerve), roi_r,
                        color="yellow", fill=False, lw=2.0, ls="-",
                        zorder=7, label=f"ROI sphere\nr={roi_r*1000:.0f}mm")
    ax_side.add_patch(roi_c)
    ax_side.text(e1x + roi_r + 0.001, z_nerve,
                 f"ROI\n(tibial nerve\n≈{z_tgt*1000:.0f}mm deep)",
                 ha="left", va="center", color="yellow", fontsize=7, zorder=8)

    # Layer boundary dashes + labels on right
    for zz, lbl in [(t_musc,         "fat | muscle"),
                    (t_musc + t_fat,  "skin | fat"),
                    (Lz,              "skin top")]:
        ax_side.axhline(zz, color="white", lw=0.7, ls="--", alpha=0.4, zorder=2)
        ax_side.text(Lx * 1.01, zz, lbl, color="white", fontsize=6.5,
                     va="center", alpha=0.8)

    ax_side.axhline(z_nerve, color="yellow", lw=1.0, ls=":", alpha=0.7, zorder=3)
    ax_side.text(Lx * 1.01, z_nerve,
                 f"z_nerve\n{z_nerve*1000:.0f}mm",
                 color="yellow", fontsize=6, va="center", alpha=0.9)

    ax_side.set_xlim(-0.004, Lx + 0.022)
    ax_side.set_ylim(-0.003, Lz + t_contact + 0.010)
    ax_side.set_xlabel("Medial → Lateral  (m)", color=TC, fontsize=9)
    ax_side.set_ylabel("Depth z  (m,  0=base → Lz=skin top)", color=TC, fontsize=9)
    ax_side.set_title(
        "ANATOMY (side view, x–z plane)\n"
        "Cyan/lime arrows = current IN / OUT of tissue",
        color=TC, fontsize=9, fontweight="bold")
    ax_side.legend(handles=[roi_c], loc="lower right",
                   facecolor="#222", edgecolor="white", labelcolor="white",
                   fontsize=7)

    # ─── Panel 2: Top view (x–y, skin surface) ────────────────────────────────
    domain = mpatches.Rectangle(
        (0, 0), Lx, Ly,
        facecolor=LAYER_COLORS["skin"], edgecolor="white", lw=1.0, alpha=0.35)
    ax_top.add_patch(domain)
    ax_top.text(Lx/2, Ly/2, "skin surface\n(z = Lz)",
                ha="center", va="center", color="white", fontsize=8, alpha=0.5)

    theta = np.linspace(0, 2*np.pi, 361)
    for (xc, yc), clr, lbl in [
            ((e1x, e1y), "cyan",
             f"+I active\n({e1x*1000:.0f}, {e1y*1000:.0f}) mm"),
            ((e2x, e2y), "lime",
             f"0V return\n({e2x*1000:.0f}, {e2y*1000:.0f}) mm")]:
        ax_top.fill(xc + r_m*np.cos(theta), yc + r_m*np.sin(theta),
                    color=clr, alpha=0.25)
        ax_top.plot(xc + r_m*np.cos(theta), yc + r_m*np.sin(theta),
                    color=clr, lw=2.0)
        ax_top.text(xc, yc, lbl, ha="center", va="center",
                    color=clr, fontsize=7, fontweight="bold",
                    multialignment="center")

    # Current spreading arcs
    for rad_mult, alpha in [(1.5, 0.5), (2.5, 0.3), (4.0, 0.15)]:
        ax_top.plot(e1x + r_m*rad_mult*np.cos(theta),
                    e1y + r_m*rad_mult*np.sin(theta),
                    color="cyan", lw=0.6, ls="--", alpha=alpha)

    # Arrows showing spreading
    for ang in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
        ax_top.annotate("",
            xy=(e1x + r_m*3.0*np.cos(ang), e1y + r_m*3.0*np.sin(ang)),
            xytext=(e1x + r_m*1.2*np.cos(ang), e1y + r_m*1.2*np.sin(ang)),
            arrowprops=dict(arrowstyle="-|>", color="cyan", lw=0.8, alpha=0.4))

    # Return electrode collection arcs
    for rad_mult, alpha in [(1.5, 0.4), (2.5, 0.25)]:
        ax_top.plot(e2x + r_m*rad_mult*np.cos(theta),
                    e2y + r_m*rad_mult*np.sin(theta),
                    color="lime", lw=0.6, ls="--", alpha=alpha)

    # Anatomy labels
    ax_top.text(0.003, Ly/2, "Medial\nbone", ha="left", va="center",
                color="white", fontsize=7, alpha=0.8)
    ax_top.text(Lx-0.003, Ly/2, "Lateral\nbone", ha="right", va="center",
                color="white", fontsize=7, alpha=0.8)
    ax_top.text(Lx/2, Ly-0.002, "Posterior\n(Achilles)", ha="center", va="top",
                color="white", fontsize=7, alpha=0.8)
    ax_top.text(Lx/2, 0.002, "Anterior", ha="center", va="bottom",
                color="white", fontsize=7, alpha=0.8)

    # Achilles tendon marker
    ax_top.plot(Lx*0.50, Ly*0.96, '^', color='white', ms=9,
                mfc='white', mew=1.5, zorder=5)
    ax_top.text(Lx*0.50, Ly*0.96 - Ly*0.05, 'AT',
                ha='center', va='top', color='white',
                fontsize=8, fontweight='bold', zorder=5)

    ax_top.set_xlim(-0.002, Lx+0.002)
    ax_top.set_ylim(-0.002, Ly+0.002)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("Medial → Lateral  (m)", color=TC, fontsize=9)
    ax_top.set_ylabel("Anterior → Posterior  (m)", color=TC, fontsize=9)
    ax_top.set_title(
        f"SKIN SURFACE (top view, z = {Lz*1000:.0f} mm)\n"
        f"Dashed circles = current spreading pattern  |  r = {r_mid_mm:.0f} mm",
        color=TC, fontsize=9, fontweight="bold")

    # ─── Panel 3: |J| vs depth below active electrode ─────────────────────────
    ax = ax_prof

    depth_data     = None   # mm from skin surface
    Jmag_data      = None
    profile_source = "schematic"

    if vtu_mesh is not None:
        try:
            pts  = np.array(vtu_mesh.points)
            J    = np.array(vtu_mesh.point_data["volume current"])
            Jmag = np.linalg.norm(J, axis=1)
            # Points within ±r_m/2 of active electrode center in xy
            tol_xy = max(r_m * 0.4, 0.003)
            near   = (np.abs(pts[:, 0] - e1x) < tol_xy) & \
                     (np.abs(pts[:, 1] - e1y) < tol_xy) & \
                     (pts[:, 2] <= Lz + t_contact + 1e-4)
            if near.sum() >= 4:
                z_near     = pts[near, 2]
                J_near     = Jmag[near]
                depth_mm   = (Lz - z_near) * 1000   # 0 = skin top, +ve = deeper
                # Bin into depth bins for a cleaner profile
                bins = np.linspace(depth_mm.min(), depth_mm.max(), 60)
                bin_idx = np.digitize(depth_mm, bins)
                bin_J   = [J_near[bin_idx == i].mean() if (bin_idx == i).any() else np.nan
                           for i in range(1, len(bins))]
                bin_d   = 0.5 * (bins[:-1] + bins[1:])
                valid   = np.isfinite(bin_J)
                if valid.sum() >= 3:
                    depth_data     = bin_d[valid]
                    Jmag_data      = np.array(bin_J)[valid]
                    profile_source = vtu_label
        except Exception as exc:
            print(f"  model_diagram: profile extraction failed: {exc}")

    if depth_data is not None and Jmag_data is not None:
        # Actual data profile
        ax.plot(Jmag_data, depth_data, color="cyan", lw=2.5, zorder=5,
                label="Simulated |J|")
        ax.fill_betweenx(depth_data, 0, Jmag_data, color="cyan", alpha=0.18)
        Jmax = float(Jmag_data.max()) if len(Jmag_data) > 0 else 1.0
    else:
        # Schematic profile (exponential-like decay as placeholder)
        d_sch  = np.linspace(0, Lz*1000, 200)
        J_sch  = np.exp(-d_sch / 12) * 5.0
        ax.plot(J_sch, d_sch, color="cyan", lw=2.0, ls="--", zorder=5,
                label="Schematic (no VTU)")
        ax.fill_betweenx(d_sch, 0, J_sch, color="cyan", alpha=0.12)
        Jmax = 5.5
        depth_data = d_sch

    ax.set_ylim(depth_data.max() + 1, -1)   # depth increases downward

    # Layer shading
    def _layer_band(ax, d0_mm, d1_mm, color, label):
        ax.axhspan(d0_mm, d1_mm, color=color, alpha=0.22, zorder=1)
        ax.text(Jmax * 1.02, (d0_mm + d1_mm) / 2, label,
                ha="left", va="center", color=color, fontsize=7.5,
                fontweight="bold")

    if t_contact > 0:
        _layer_band(ax, -t_contact*1000, 0, LAYER_COLORS["contact"], "contact")
    _layer_band(ax, 0,                t_skin*1000,           LAYER_COLORS["skin"],   "SKIN")
    _layer_band(ax, t_skin*1000,      (t_skin+t_fat)*1000,   LAYER_COLORS["fat"],    "FAT")
    _layer_band(ax, (t_skin+t_fat)*1000, Lz*1000,            LAYER_COLORS["muscle"], "MUSCLE")

    # Layer boundary lines
    for d_mm, lbl in [(0,                    "skin surface"),
                      (t_skin*1000,           "skin|fat"),
                      ((t_skin+t_fat)*1000,   "fat|muscle")]:
        ax.axhline(d_mm, color="white", lw=0.8, ls="--", alpha=0.5, zorder=3)
        ax.text(0, d_mm - 0.3, lbl, color="white", fontsize=6.0,
                va="bottom", ha="left", alpha=0.7)

    # ROI depth
    roi_depth_mm = z_tgt * 1000
    ax.axhline(roi_depth_mm, color="yellow", lw=1.5, ls="-.", alpha=0.85, zorder=4)
    ax.text(0, roi_depth_mm + 0.4,
            f"ROI (nerve)\n{roi_depth_mm:.0f} mm",
            color="yellow", fontsize=7, va="top", ha="left")

    # Annotate layer-average J if data available
    if depth_data is not None and Jmag_data is not None:
        def _layer_avg(d0, d1):
            m = (depth_data >= d0) & (depth_data < d1)
            return float(Jmag_data[m].mean()) if m.any() else np.nan

        for d0, d1, lname, clr in [
                (0,                  t_skin*1000,          "Skin",   LAYER_COLORS["skin"]),
                (t_skin*1000,        (t_skin+t_fat)*1000,  "Fat",    LAYER_COLORS["fat"]),
                ((t_skin+t_fat)*1000, Lz*1000,             "Muscle", LAYER_COLORS["muscle"])]:
            jav = _layer_avg(d0, d1)
            if np.isfinite(jav):
                dmid = (d0 + d1) / 2
                ax.annotate(f"avg={jav:.3f}\nA/m²",
                            xy=(jav, dmid),
                            xytext=(Jmax * 0.55, dmid),
                            fontsize=7, color=clr, ha="center", va="center",
                            arrowprops=dict(arrowstyle="->", color=clr,
                                           lw=0.7, alpha=0.6))

    ax.set_xlabel("|J|  (A/m²)", color=TC, fontsize=9)
    ax.set_ylabel("Depth below skin surface  (mm)", color=TC, fontsize=9)
    ax.set_title(
        f"|J| vs depth below active electrode\n"
        f"({profile_source if profile_source != 'schematic' else 'schematic — run sweep first'})",
        color=TC, fontsize=8.5, fontweight="bold")
    ax.legend(facecolor="#222", edgecolor="white", labelcolor="white", fontsize=8,
              loc="lower right")
    ax.set_xlim(left=0)

    fig.suptitle(
        "MODEL OVERVIEW — ankle 3-layer slab PTNS stimulation  "
        "(PLACEHOLDER conductivities — not validated)\n"
        f"Geometry: {Lx*100:.0f}×{Ly*100:.0f}×{Lz*100:.0f} cm  |  "
        f"skin {t_skin*1000:.1f}mm  fat {t_fat*1000:.1f}mm  "
        f"muscle {t_musc*1000:.1f}mm  |  "
        f"Active: ({e1x*1000:.0f},{e1y*1000:.0f})mm  "
        f"Return: ({e2x*1000:.0f},{e2y*1000:.0f})mm  r={r_mid_mm:.0f}mm",
        fontsize=9.5, fontweight="bold", color=TC)

    out = RESULTS_DIR / "model_diagram.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out}")
    plt.close(fig)


# ── Sanity check table ────────────────────────────────────────────────────────
def print_sanity_table(summary, p=None):
    """Print a comprehensive per-case verification table to the console."""
    if not summary:
        return
    mode = summary[0].get("control_mode", "voltage")
    st   = (p.get("stim", p.get("control", {})) if p else {})
    I_target_mA = st.get("injected_current_mA", 5.0) if mode == "current" else None

    width = 120 if mode == "current" else 100
    print(f"\n{'='*width}")
    print("  RESULTS SANITY CHECK — per-case current delivery & ROI")
    print(f"{'='*width}")

    # Header
    hdr = (f"  {'r(mm)':>6}  {'fat(mm)':>7}  "
           f"{'I_active(mA)':>13}  {'I_return(mA)':>13}  {'flux_err%':>10}")
    if mode == "current":
        hdr += f"  {'tgt(mA)':>8}  {'dev%':>6}  {'compV':>8}"
    hdr += (f"  {'BC_act':>6}  {'BC_ret':>6}"
            f"  {'roi_layer':>9}  {'frac_mu':>8}  {'frac_fa':>8}  {'frac_sk':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for row in sorted(summary, key=lambda x: (x["elec_r_mm"], x["t_fat_mm"])):
        I_a = row.get("total_current_A", float("nan"))
        I_r = row.get("I_return_A",      float("nan"))
        fe  = row.get("flux_err",        float("nan"))
        fm  = row.get("roi_frac_muscle", float("nan"))
        ff  = row.get("roi_frac_fat",    float("nan"))
        fs  = row.get("roi_frac_skin",   float("nan"))
        rl  = row.get("roi_layer",       "?")

        def _fmt(v, fmt):
            return fmt.format(v) if (v is not None and np.isfinite(float(v))) else "    N/A"

        line = (f"  {row['elec_r_mm']:>6.0f}  {row['t_fat_mm']:>7.1f}  "
                f"{_fmt(I_a*1e3 if np.isfinite(float(I_a)) else float('nan'), '{:>13.3f}'):>13}  "
                f"{_fmt(I_r*1e3 if np.isfinite(float(I_r)) else float('nan'), '{:>13.3f}'):>13}  "
                f"{_fmt(fe*100  if np.isfinite(float(fe))  else float('nan'), '{:>10.2f}'):>10}")
        if mode == "current" and I_target_mA is not None:
            dev = (abs(float(I_a) - I_target_mA * 1e-3) / (I_target_mA * 1e-3) * 100
                   if np.isfinite(float(I_a)) else float("nan"))
            cV = row.get("compliance_V", float("nan"))
            ec = row.get("exceeded_compliance", False)
            line += (f"  {I_target_mA:>8.1f}"
                     f"  {_fmt(dev, '{:>6.2f}'):>6}"
                     f"  {_fmt(cV, '{:>8.2f}'):>8}"
                     + ("  [!]" if ec else ""))
        bc_a = row.get("active_boundary_id_used")
        bc_r = row.get("return_boundary_id_used")
        line += (f"  {str(bc_a) if bc_a is not None else 'N/A':>6}"
                 f"  {str(bc_r) if bc_r is not None else 'N/A':>6}"
                 f"  {rl:>9}"
                 f"  {_fmt(fm, '{:>8.4f}'):>8}"
                 f"  {_fmt(ff, '{:>8.4f}'):>8}"
                 f"  {_fmt(fs, '{:>8.4f}'):>8}")
        print(line)

    if mode == "current" and I_target_mA is not None:
        print(f"\n  Target current: {I_target_mA:.1f} mA  |  dev% = |I_active - target| / target × 100")
        print(f"  BC_act/ret = Elmer boundary IDs used for active/return electrode")
        print(f"  frac_mu/fa/sk = fraction of ROI cells in muscle / fat / skin")
    print(f"{'='*width}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p       = load_params()
    summary = load_summary()

    print(f"Loaded {len(summary)} cases.")
    plot_J_surface_maps(summary, p)
    plot_summary_metrics(summary, p)
    plot_3d_representative(summary, p)
    plot_depth_slice_E_maps(summary, p)
    plot_model_diagram(p, summary=summary)
    print_sanity_table(summary, p)
    print("Done.")
