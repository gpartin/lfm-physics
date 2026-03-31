"""35 — Spin Precession (Bloch Sphere Survey)

A spin state can be visualised as a unit vector on the Bloch sphere.
Rotating the state with R_z(φ) precesses the Bloch vector around the
z-axis — exactly what happens to a magnetic moment in a magnetic field
(Larmor precession).

This example sweeps R_z(φ) over [0, 2π] and measures the three spin
expectation values ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩ at each angle.  Starting from
|+x⟩ = (|↑⟩ + |↓⟩)/√2:

    ⟨σ_z⟩ = 0         (unchanged — R_z preserves the equator)
    ⟨σ_x⟩ = cos(φ)    (precesses in x-y plane)
    ⟨σ_y⟩ = +sin(φ)   (90° ahead of σ_x in precession direction)

The Bloch sphere parametrisation for R_z(φ)|+x⟩:
    (n_x, n_y, n_z) = (cos φ, +sin φ, 0)

The second panel shows all 6 cardinal spin states on the sphere:
    |+z⟩, |−z⟩, |+x⟩, |−x⟩, |+y⟩, |−y⟩

Physics note
-----------
In LFM, GOV-01 is spin-blind: both spinor components see the same χ²,
so free time-evolution produces no spin precession.  Larmor precession
would require coupling the spin to the momentum current j via the ε_W
(helicity) term.  This example demonstrates the KINEMATIC structure of
spin precession using the SU(2) rotation operators directly.

Run::

    python examples/35_spin_precession.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import lfm
from lfm.fields.spinor import gaussian_spinor, apply_rotation_x, apply_rotation_z
from lfm.analysis.spinor import spinor_sigma_x, spinor_sigma_y, spinor_sigma_z
from _common import make_out_dir, parse_no_anim

_args = parse_no_anim()
_OUT  = make_out_dir("35_spin_precession")

N  = 32
cx = N // 2

print("35 — Spin Precession (Bloch Sphere Survey)")
print("=" * 55)
print()

# ── Build the 6 cardinal states ───────────────────────────────────────────

def make_state(spin_up: bool) -> tuple:
    return gaussian_spinor(N, (cx, cx, cx), amplitude=5.0, sigma=3.0,
                           spin_up=spin_up)

# |+z⟩ = |↑⟩
r_pz, i_pz = make_state(True)
# |−z⟩ = |↓⟩
r_mz, i_mz = make_state(False)
# |+y⟩ = R_x(−π/2)|↑⟩  — pivot north pole → +y
r_py, i_py = apply_rotation_x(r_pz, i_pz, -np.pi / 2)
# |−y⟩ = R_x(+π/2)|↑⟩  — pivot north pole → −y
r_my, i_my = apply_rotation_x(r_pz, i_pz,  np.pi / 2)
# |+x⟩ = R_z(−π/2)|+y⟩ — pivot +y → +x   (two-step: R_x then R_z)
r_px, i_px = apply_rotation_z(r_py, i_py, -np.pi / 2)
# |−x⟩ = R_z(+π/2)|+y⟩ — pivot +y → −x
r_mx, i_mx = apply_rotation_z(r_py, i_py,  np.pi / 2)

states = {
    "|+z⟩": (r_pz, i_pz, "expected: (σ_z=+1, σ_x= 0, σ_y= 0)"),
    "|−z⟩": (r_mz, i_mz, "expected: (σ_z=−1, σ_x= 0, σ_y= 0)"),
    "|+x⟩": (r_px, i_px, "expected: (σ_z= 0, σ_x=+1, σ_y= 0)"),
    "|−x⟩": (r_mx, i_mx, "expected: (σ_z= 0, σ_x=−1, σ_y= 0)"),
    "|+y⟩": (r_py, i_py, "expected: (σ_z= 0, σ_x= 0, σ_y=+1)"),
    "|−y⟩": (r_my, i_my, "expected: (σ_z= 0, σ_x= 0, σ_y=−1)"),
}

print("  Cardinal spin states on the Bloch sphere:")
print(f"  {'State':>5}   {'⟨σ_z⟩':>7} {'⟨σ_x⟩':>7} {'⟨σ_y⟩':>7}   Note")
print("  " + "-" * 64)

bloch_pts = []
for label, (r, i, note) in states.items():
    sz = spinor_sigma_z(r, i)
    sx = spinor_sigma_x(r, i)
    sy = spinor_sigma_y(r, i)
    bloch_pts.append((sx, sy, sz, label))
    print(f"  {label:>5}   {sz:+7.4f} {sx:+7.4f} {sy:+7.4f}   {note}")

print()

# ── Precession sweep: R_z(φ)|+x⟩ ─────────────────────────────────────────
n_pts   = 73
phi_d   = np.linspace(0.0, 360.0, n_pts)
phi_r   = np.radians(phi_d)

sz_arr, sx_arr, sy_arr = [], [], []

for phi in phi_r:
    r, i = apply_rotation_z(r_px, i_px, phi)
    sz_arr.append(spinor_sigma_z(r, i))
    sx_arr.append(spinor_sigma_x(r, i))
    sy_arr.append(spinor_sigma_y(r, i))

sz_arr = np.array(sz_arr)
sx_arr = np.array(sx_arr)
sy_arr = np.array(sy_arr)

# Check: σ_z should stay near 0, σ_x ≈ cos(φ), σ_y ≈ −sin(φ)
sz_rms = float(np.sqrt(np.mean(sz_arr**2)))
sx_err = float(np.max(np.abs(sx_arr - np.cos(phi_r))))
sy_err = float(np.max(np.abs(sy_arr - np.sin(phi_r))))

print("  R_z(φ)|+x⟩ precession sweep:")
print(f"    ⟨σ_z⟩ RMS (should be ~0) : {sz_rms:.6f}")
print(f"    max |⟨σ_x⟩ − cos(φ)|    : {sx_err:.6f}")
print(f"    max |⟨σ_y⟩ − sin(φ)|    : {sy_err:.6f}")
print()

if sz_rms < 0.01 and sx_err < 0.01 and sy_err < 0.01:
    print("  ✓ Precession (Bloch vector rotation) CONFIRMED")
else:
    print("  ✗ Check failed")
print()

# ── Plot ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5))

# Left: precession traces
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(phi_d, sx_arr, color="#2166ac", lw=2.0, label="⟨σ_x⟩  (≈ cos φ)")
ax1.plot(phi_d, sy_arr, color="#4dac26", lw=2.0, label="⟨σ_y⟩  (≈ sin φ)")
ax1.plot(phi_d, sz_arr, color="#d6604d", lw=1.5, ls="--", label="⟨σ_z⟩  (≈ 0)")
ax1.plot(phi_d, np.cos(phi_r),  color="#2166ac", lw=0.8, ls=":", alpha=0.5)
ax1.plot(phi_d, np.sin(phi_r), color="#4dac26", lw=0.8, ls=":", alpha=0.5,
         label="Exact theory (dotted)")
ax1.set_xlabel("Rotation angle φ (degrees)", fontsize=11)
ax1.set_ylabel("Spin expectation value", fontsize=11)
ax1.set_title("Larmor Precession: R$_z$(φ)|+x⟩", fontsize=12, fontweight="bold")
ax1.set_xlim(0, 360)
ax1.set_xticks([0, 90, 180, 270, 360])
ax1.set_ylim(-1.3, 1.3)
ax1.axhline(0, color="grey", lw=0.5)
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.25)

# Right: Bloch sphere with 6 cardinal points
ax2 = fig.add_subplot(1, 2, 2, projection="3d")

# Draw unit sphere skeleton
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 30)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax2.plot_surface(xs, ys, zs, alpha=0.05, color="lightblue")

# Draw axes
for d in [(1,0,0,"x"), (-1,0,0,"-x"), (0,1,0,"y"),
          (0,-1,0,"-y"), (0,0,1,"z"), (0,0,-1,"-z")]:
    ax2.plot([0, d[0]*1.2], [0, d[1]*1.2], [0, d[2]*1.2],
             color="grey", lw=0.6, alpha=0.4)

# Plot the 6 cardinal points
colours = {"z": "#d6604d", "x": "#2166ac", "y": "#4dac26"}
for (sx, sy, sz, lab) in bloch_pts:
    col = colours[lab[-2]]  # last axis letter
    ax2.scatter([sx], [sy], [sz], s=100, color=col, zorder=5)
    ax2.text(sx * 1.18, sy * 1.18, sz * 1.18, lab, fontsize=8,
             ha="center", color=col)

# Draw precession circle
ax2.plot(np.cos(phi_r), np.sin(phi_r), np.zeros_like(phi_r),
         color="#2166ac", lw=1.5, ls="-", alpha=0.6, label="Prec. circle")

ax2.set_title("Bloch Sphere — 6 Cardinal States", fontsize=12, fontweight="bold")
ax2.set_xlabel("σ_x"); ax2.set_ylabel("σ_y"); ax2.set_zlabel("σ_z")
ax2.set_xlim(-1.3, 1.3); ax2.set_ylim(-1.3, 1.3); ax2.set_zlim(-1.3, 1.3)
ax2.set_box_aspect([1, 1, 1])

fig.tight_layout()
out_path = _OUT / "35_spin_precession.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  Plot saved → {out_path}")
