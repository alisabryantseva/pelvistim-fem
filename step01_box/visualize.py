"""
visualize.py  —  step01_box baseline figures
Run from step01_box/:   python3 visualize.py
Outputs: results/step01_summary.png, results/step01_3d.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pyvista as pv
from pathlib import Path

VTU = Path("results/case_t0001.vtu")
if not VTU.exists():
    raise SystemExit(f"ERROR: {VTU} not found. Run ElmerSolver first.")

# ── Load data ─────────────────────────────────────────────────────────────────
mesh  = pv.read(str(VTU))
pts   = np.array(mesh.points)
phi   = np.array(mesh.point_data["potential"])
J_vec = np.array(mesh.point_data["volume current"])
J_mag = np.linalg.norm(J_vec, axis=1)

Lx = pts[:, 0].max()
Ly = pts[:, 1].max()
Lz = pts[:, 2].max()
J_an = 0.2 / Lz                  # analytic |J| = σ·ΔV/Lz

# ── 3D pyvista render (saved separately, also embedded) ───────────────────────
Path("results").mkdir(exist_ok=True)
pl = pv.Plotter(off_screen=True, window_size=(700, 520))
pl.set_background("white")
pl.add_mesh(mesh.clip(normal="y", origin=[0, Ly * 0.5, 0]),
            scalars="potential", cmap="RdYlBu_r",
            show_scalar_bar=True,
            scalar_bar_args={"title": "V (V)", "width": 0.45,
                             "position_x": 0.27, "position_y": 0.04,
                             "title_font_size": 13, "label_font_size": 11})
pl.add_mesh(mesh.outline(), color="black", line_width=1.5)
pl.view_isometric()
pl.camera.zoom(1.2)
img3d = Path("results/step01_3d.png")
pl.screenshot(str(img3d))
pl.close()

# ── XZ mid-slice (y ≈ Ly/2) ───────────────────────────────────────────────────
mask_xz = np.abs(pts[:, 1] - Ly / 2) < Ly * 0.04
p_xz    = pts[mask_xz]
tri_xz  = mtri.Triangulation(p_xz[:, 0], p_xz[:, 2])
phi_xz  = phi[mask_xz]
Jvec_xz = J_vec[mask_xz]         # (N, 3) for the slice

# ── Center column (for 1-D profiles) ──────────────────────────────────────────
r_xy    = np.hypot(pts[:, 0] - Lx / 2, pts[:, 1] - Ly / 2)
col     = r_xy < Lx * 0.08
z_c     = pts[col, 2];  phi_c = phi[col];  J_c = J_mag[col]

# ── Validation metrics (printed in panel 6) ────────────────────────────────────
mean_J  = J_mag.mean()
cv_J    = J_mag.std(ddof=1) / mean_J
coeffs  = np.polyfit(z_c, phi_c, 1)
phi_fit = np.polyval(coeffs, z_c)
r2      = 1.0 - np.sum((phi_c - phi_fit)**2) / np.sum((phi_c - phi_c.mean())**2)
tol_z   = Lz * 1e-3
flux_top = np.abs(J_vec[pts[:, 2] > Lz - tol_z, 2]).mean()
flux_bot = np.abs(J_vec[pts[:, 2] < tol_z,       2]).mean()

# ── Figure: 2×3, constrained layout ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                         constrained_layout=True)

fig.suptitle(
    "step01_box — uniform electrode baseline  "
    r"($\sigma$ = 0.2 S/m, $\Delta V$ = 1 V, box 4×4×2 cm)",
    fontsize=12, fontweight="bold"
)

# ── [0,0] Potential — XZ cross-section ───────────────────────────────────────
ax = axes[0, 0]
cf = ax.tricontourf(tri_xz, phi_xz, levels=25, cmap="RdYlBu_r",
                    vmin=0, vmax=1)
ax.tricontour(tri_xz, phi_xz, levels=8, colors="k",
              linewidths=0.4, alpha=0.35)
fig.colorbar(cf, ax=ax, label="Potential (V)", fraction=0.046, pad=0.04)
ax.set_xlim(0, Lx); ax.set_ylim(0, Lz)
ax.set_aspect("equal")
ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
ax.set_title("Potential — XZ cross-section  (y = Ly/2)")
ax.text(0.02, 0.97, "top: 1 V", transform=ax.transAxes,
        va="top", fontsize=8, color="firebrick")
ax.text(0.02, 0.03, "bottom: 0 V", transform=ax.transAxes,
        va="bottom", fontsize=8, color="navy")

# ── [0,1] 3D view (embedded pyvista screenshot) ───────────────────────────────
ax = axes[0, 1]
ax.imshow(plt.imread(str(img3d)))
ax.axis("off")
ax.set_title("3D potential field  (clipped at y = Ly/2)")

# ── [0,2] J vectors — XZ cross-section ───────────────────────────────────────
ax = axes[0, 2]
# subsample: ~8×8 = 64 arrows on the slice, unit z-direction (J is uniform)
n_arrows = 64
idx_all  = np.where(mask_xz)[0]
step     = max(1, len(idx_all) // n_arrows)
idx      = idx_all[::step]
# normalise arrow length to a fixed fraction of Lz so they read cleanly
J_xz_norm = J_mag[idx]                  # magnitudes (all ~J_an)
ax.quiver(pts[idx, 0], pts[idx, 2],
          J_vec[idx, 0] / J_an,          # unit-normalised x
          J_vec[idx, 2] / J_an,          # unit-normalised z
          J_xz_norm,
          cmap="inferno", clim=(0, J_an * 1.05),
          pivot="mid", scale=30, width=0.005,
          headwidth=3, headlength=4)
sm = plt.cm.ScalarMappable(cmap="inferno",
                            norm=plt.Normalize(0, J_an * 1.05))
sm.set_array([])
fig.colorbar(sm, ax=ax, label="|J| (A/m²)", fraction=0.046, pad=0.04)
ax.set_xlim(0, Lx); ax.set_ylim(0, Lz)
ax.set_aspect("equal")
ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
ax.set_title("Current density vectors — XZ cross-section")
ax.text(0.5, 0.92,
        f"uniform  |J| = {J_an:.1f} A/m²",
        transform=ax.transAxes, ha="center", fontsize=8,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

# ── [1,0] Potential vs depth — center column ──────────────────────────────────
ax = axes[1, 0]
ax.scatter(phi_c, z_c * 100, s=7, alpha=0.55, color="steelblue",
           label="FEM nodes", zorder=3)
z_lin = np.linspace(0, Lz, 60)
ax.plot(z_lin / Lz, z_lin * 100, "r--", lw=1.8, label="Analytic V = z/Lz")
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.05 * Lz * 100, Lz * 100 * 1.05)
ax.set_xlabel("Potential (V)"); ax.set_ylabel("Depth z (cm)")
ax.set_title("Potential vs depth — center column")
ax.legend(fontsize=8, framealpha=0.8)
ax.grid(True, alpha=0.3, linewidth=0.5)

# ── [1,1] |J| vs depth — center column ───────────────────────────────────────
ax = axes[1, 1]
ax.scatter(J_c, z_c * 100, s=7, alpha=0.55, color="darkorange",
           label="FEM nodes", zorder=3)
ax.axvline(J_an, color="r", ls="--", lw=1.8,
           label=f"Analytic {J_an:.1f} A/m²")
J_spread = J_mag.max() - J_mag.min()
margin   = max(J_spread * 3, J_an * 0.05)
ax.set_xlim(J_an - margin, J_an + margin)
ax.set_ylim(-0.05 * Lz * 100, Lz * 100 * 1.05)
ax.set_xlabel("|J| (A/m²)"); ax.set_ylabel("Depth z (cm)")
ax.set_title("|J| vs depth — center column")
ax.legend(fontsize=8, framealpha=0.8)
ax.grid(True, alpha=0.3, linewidth=0.5)

# ── [1,2] Validation metrics table ───────────────────────────────────────────
ax = axes[1, 2]
ax.axis("off")

rows = [
    ("Analytic |J|",     f"{J_an:.4f} A/m²",     "σ·ΔV/Lz",           "—"),
    ("mean(|J|)",        f"{mean_J:.6f} A/m²",    "FEM volume avg",     "—"),
    ("rel error",        f"{abs(mean_J-J_an)/J_an:.2e}", "vs analytic", "< 1e-3"),
    ("CV std/mean(|J|)", f"{cv_J:.2e}",            "uniformity",        "< 1e-2"),
    ("R²  V(z)",         f"{r2:.7f}",              "linearity",         "> 0.9999"),
    ("slope  V(z)",      f"{coeffs[0]:.4f} V/m",  f"analytic {1/Lz:.4f}", "—"),
    ("flux top |J_z|",   f"{flux_top:.4f} A/m²",  "conservation",      "—"),
    ("flux bot |J_z|",   f"{flux_bot:.4f} A/m²",  "conservation",      "—"),
    ("Φ range",          f"[{phi.min():.3f}, {phi.max():.3f}] V", "—", "—"),
]

col_labels = ["Metric", "Value", "Note", "Tolerance"]
col_widths = [0.28, 0.28, 0.26, 0.18]
x_starts   = [0.0, 0.28, 0.56, 0.82]

# Header
for xi, lbl in zip(x_starts, col_labels):
    ax.text(xi, 0.97, lbl, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top",
            fontfamily="monospace")
ax.plot([0, 1], [0.93, 0.93], color="gray", linewidth=0.8,
        transform=ax.transAxes, clip_on=False)

# Rows
n = len(rows)
for i, row in enumerate(rows):
    y = 0.90 - i * (0.90 / n)
    bg = "whitesmoke" if i % 2 == 0 else "white"
    ax.add_patch(plt.Rectangle((0, y - 0.90/n/2), 1, 0.90/n,
                               facecolor=bg, alpha=0.6,
                               transform=ax.transAxes, clip_on=False))
    for xi, cell in zip(x_starts, row):
        ax.text(xi + 0.01, y, cell, transform=ax.transAxes,
                fontsize=7.5, va="center", fontfamily="monospace")

ax.set_title("Validation metrics", fontsize=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ── Save ─────────────────────────────────────────────────────────────────────
out = Path("results/step01_summary.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
print(f"Saved → {img3d}")
plt.close(fig)
