"""
visualize.py  —  step01_box baseline results
Run from step01_box/:   python3 visualize.py

step01 is the UNIFORM electrode baseline (whole top surface = 1V).
J is flat everywhere — this is the sanity check before step02
where we model finite electrode patches.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patches as mpatches
import pyvista as pv
from pathlib import Path

VTU = Path("results/case_t0001.vtu")
if not VTU.exists():
    raise SystemExit(f"ERROR: {VTU} not found. Run ElmerSolver first.")

mesh = pv.read(str(VTU))
pts   = np.array(mesh.points)
phi   = np.array(mesh.point_data["potential"])
J_vec = np.array(mesh.point_data["volume current"])
J_mag = np.linalg.norm(J_vec, axis=1)

Lx = pts[:, 0].max()
Ly = pts[:, 1].max()
Lz = pts[:, 2].max()
J_analytic = 0.2 * (1.0 / Lz)

# ── 3D pyvista render (save first, embed in figure) ───────────────────────────
Path("results").mkdir(exist_ok=True)
pl = pv.Plotter(off_screen=True, window_size=(800, 600))
pl.set_background("white")
clipped = mesh.clip(normal="y", origin=[0, Ly * 0.5, 0])
pl.add_mesh(clipped, scalars="potential", cmap="RdYlBu_r",
            show_scalar_bar=True,
            scalar_bar_args={"title": "Potential (V)", "width": 0.5,
                             "position_x": 0.25, "position_y": 0.02,
                             "title_font_size": 14, "label_font_size": 12})
pl.add_mesh(mesh.outline(), color="black", line_width=2)
pl.view_isometric()
pl.camera.zoom(1.15)
img3d_path = Path("results/step01_3d.png")
pl.screenshot(str(img3d_path))
pl.close()

# ── XZ mid-slice helpers ──────────────────────────────────────────────────────
mask_xz = np.abs(pts[:, 1] - Ly / 2) < Ly * 0.04
p_xz = pts[mask_xz]
tri_xz = mtri.Triangulation(p_xz[:, 0], p_xz[:, 2])
phi_xz = phi[mask_xz]
J_xz   = J_mag[mask_xz]
Jvec_xz = J_vec[mask_xz]

# center column for 1D profiles
r_xy = np.sqrt((pts[:, 0] - Lx/2)**2 + (pts[:, 1] - Ly/2)**2)
center_mask = r_xy < Lx * 0.08
z_c   = pts[center_mask, 2]
phi_c = phi[center_mask]
J_c   = J_mag[center_mask]
order = np.argsort(z_c)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 9))
fig.suptitle(
    "step01 — BASELINE: uniform electrode (entire top surface)\n"
    "4×4×2 cm tissue box  |  σ = 0.2 S/m  |  top = 1 V, bottom = 0 V  |  "
    "Current density J is flat everywhere — this is the sanity check",
    fontsize=11, y=0.98
)

gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

# ── Panel 1: Potential XZ cross-section ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
tc1 = ax1.tricontourf(tri_xz, phi_xz, levels=30, cmap="RdYlBu_r")
ax1.tricontour(tri_xz, phi_xz, levels=10, colors="k", linewidths=0.4, alpha=0.4)
fig.colorbar(tc1, ax=ax1, label="Potential (V)")
ax1.set_title("Potential — vertical cross-section\n(y = center)")
ax1.set_xlabel("x (m)"); ax1.set_ylabel("z (m)  ↑")
ax1.set_aspect("equal")
ax1.text(0.02, 0.97, "electrode 1V →", transform=ax1.transAxes,
         fontsize=7, va="top", color="darkred")
ax1.text(0.02, 0.03, "ground 0V →", transform=ax1.transAxes,
         fontsize=7, va="bottom", color="navy")

# ── Panel 2: 3D view (embedded) ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
img = plt.imread(str(img3d_path))
ax2.imshow(img)
ax2.axis("off")
ax2.set_title("3D potential field\n(clipped at y = center)")

# ── Panel 3: J arrows on XZ slice ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
step = max(1, mask_xz.sum() // 100)
idx = np.where(mask_xz)[0][::step]
q = ax3.quiver(pts[idx, 0], pts[idx, 2],
               J_vec[idx, 0], J_vec[idx, 2],
               J_mag[idx], cmap="inferno",
               pivot="mid", scale_units="xy",
               scale=J_mag.max() * 25, width=0.004)
fig.colorbar(q, ax=ax3, label="|J| (A/m²)")
ax3.set_title("Current density vectors\n(vertical cross-section)")
ax3.set_xlabel("x (m)"); ax3.set_ylabel("z (m)")
ax3.set_xlim(0, Lx); ax3.set_ylim(0, Lz)
ax3.set_aspect("equal")
ax3.text(0.5, 0.5, f"All arrows vertical\n|J| = {J_analytic:.1f} A/m²\nuniform",
         transform=ax3.transAxes, ha="center", va="center",
         fontsize=8, color="white",
         bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"))

# ── Panel 4: Potential vs depth ───────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(phi_c, z_c * 100, s=6, alpha=0.5, color="steelblue", label="FEM nodes")
z_lin = np.linspace(0, Lz, 80)
ax4.plot(z_lin / Lz, z_lin * 100, "r--", lw=2, label="Analytic (linear)")
ax4.set_xlabel("Potential (V)")
ax4.set_ylabel("Depth z (cm)")
ax4.set_title("Potential vs depth — center column")
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)
ax4.set_xlim(-0.02, 1.02)

# ── Panel 5: |J| vs depth ────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(J_c, z_c * 100, s=6, alpha=0.5, color="darkorange", label="FEM nodes")
ax5.axvline(J_analytic, color="r", ls="--", lw=2, label=f"Analytic: {J_analytic:.1f} A/m²")
ax5.set_xlabel("|J| (A/m²)")
ax5.set_ylabel("Depth z (cm)")
ax5.set_title("|J| vs depth — center column\n(flat = uniform electrode)")
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

# ── Panel 6: Story panel — what changes in step02 ────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")

# draw a schematic of finite electrode
from matplotlib.patches import FancyArrowPatch, Rectangle
# tissue box outline
box_rect = plt.Rectangle((0.1, 0.05), 0.8, 0.55, fill=False,
                          edgecolor="gray", linewidth=2)
ax6.add_patch(box_rect)
# finite electrode (small patch on top)
elec = plt.Rectangle((0.35, 0.60), 0.12, 0.04, color="red", zorder=5)
ax6.add_patch(elec)
# return electrode
ret = plt.Rectangle((0.62, 0.60), 0.12, 0.04, color="blue", zorder=5)
ax6.add_patch(ret)
# current concentration arrows under active electrode
for xi in [0.37, 0.40, 0.43]:
    ax6.annotate("", xy=(xi, 0.15), xytext=(xi, 0.58),
                 arrowprops=dict(arrowstyle="->", color="darkorange",
                                 lw=1.5))
# wide arrows elsewhere (low J)
for xi in [0.15, 0.75]:
    ax6.annotate("", xy=(xi, 0.15), xytext=(xi, 0.58),
                 arrowprops=dict(arrowstyle="->", color="wheat",
                                 lw=0.8))
ax6.text(0.5, 1.0,
         "step02: finite bipolar electrodes\n"
         "Key question: how does electrode size\n"
         "change current density distribution?",
         transform=ax6.transAxes, ha="center", va="top",
         fontsize=9, fontweight="bold",
         bbox=dict(facecolor="lightyellow", alpha=0.9, boxstyle="round"))
ax6.text(0.41, 0.67, "+1V\n(active)", fontsize=7, ha="center", color="darkred")
ax6.text(0.68, 0.67, "0V\n(return)", fontsize=7, ha="center", color="navy")
ax6.text(0.40, 0.10, "HIGH J\nunder\nelectrode", fontsize=7,
         ha="center", color="darkorange", fontweight="bold")
ax6.text(0.15, 0.10, "low J", fontsize=7, ha="center", color="gray")
ax6.set_xlim(0, 1); ax6.set_ylim(0, 1.05)
ax6.set_title("What step02 models", fontsize=10)

# ── Summary box ──────────────────────────────────────────────────────────────
fig.text(0.01, 0.01,
         f"step01 validation:  peak|J|={J_mag.max():.3f}  mean|J|={J_mag.mean():.3f}  "
         f"analytic={J_analytic:.3f} A/m²  |  error={abs(J_mag.mean()-J_analytic)/J_analytic*100:.2f}%  |  "
         f"R_eff={Lz/(0.2*Lx*Ly):.1f} Ω  |  Φ∈[{phi.min():.3f}, {phi.max():.3f}] V",
         fontsize=8, color="dimgray")

out = Path("results/step01_summary.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
