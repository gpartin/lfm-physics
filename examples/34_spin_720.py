"""34 — Spin 720° Periodicity (Spinor Sign Flip)

Spin-1/2 particles require TWO full rotations (720°) to return to
their original quantum state.  After only ONE rotation (360°) the
wave-function acquires a sign flip:

    R_x(2π) |ψ⟩ = −|ψ⟩        (spinor — sign flip)
    e^{iφ}  |ψ⟩ = +|ψ⟩  at φ=2π   (scalar — no sign flip)

This is measurable via interferometry: combine the original state
with the rotated state and measure the total energy |ψ_ref + ψ_rot|².

    Same sign  → constructive  → energy = 4 × single-state energy
    Sign flip  → DESTRUCTIVE   → energy = 0

This is the LFM analog of neutron interferometry (Rauch et al. 1975,
Zeilinger et al. 1975) where a neutron beam was split, one path
rotated by a magnetic field, and the resulting interference showed
exactly this 720° period.

The plot shows:
  • Blue  (spinor): null at 360°, restored at 720°  — period 720°
  • Orange (scalar): null at 180°, restored at 360° — period 360°

No dynamics required — this is pure SU(2) field algebra.

Run::

    python examples/34_spin_720.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lfm
from lfm.fields.spinor import gaussian_spinor, apply_rotation_x
from lfm.analysis.spinor import spinor_interference_energy
from _common import make_out_dir, parse_no_anim

_args = parse_no_anim()
_OUT  = make_out_dir("34_spin_720")

# ── Setup ──────────────────────────────────────────────────────────────────
N   = 32
cx  = N // 2

print("34 — Spin 720° Periodicity (Spinor Sign Flip)")
print("=" * 55)
print()
print(f"  Grid: {N}³   Spinor: Gaussian amp=5, σ=3  at ({cx},{cx},{cx})")
print()

# Reference spin-up state
psi_r_ref, psi_i_ref = gaussian_spinor(
    N, (cx, cx, cx), amplitude=5.0, sigma=3.0, spin_up=True
)

# Normalisation constant: |ψ_ref|² summed over grid
norm_single = float(np.sum(psi_r_ref**2 + psi_i_ref**2))

# Scalar reference: treat real part of spin-up component as a real-valued
# scalar field and rotate it with a plain phase e^{iφ}
scalar_r0 = psi_r_ref[0].copy()
scalar_i0 = psi_i_ref[0].copy()

# Single-component norm (for scalar normalisation)
norm_scalar = float(np.sum(scalar_r0**2 + scalar_i0**2))

# ── Sweep φ from 0 to 4π ──────────────────────────────────────────────────
n_points = 73  # 0° to 720° in 10° steps
phi_vals  = np.linspace(0.0, 4.0 * np.pi, n_points)

spinor_vals = []
scalar_vals = []

for phi in phi_vals:
    # Spinor: SU(2) rotation R_x(φ)
    pr, pi = apply_rotation_x(psi_r_ref, psi_i_ref, phi)
    e_spin = spinor_interference_energy(psi_r_ref, psi_i_ref, pr, pi)
    spinor_vals.append(e_spin / (2.0 * norm_single))   # normalise to [0, 2]

    # Scalar: phase e^{iφ} applied to single component
    sr = scalar_r0 * np.cos(phi) - scalar_i0 * np.sin(phi)
    si = scalar_r0 * np.sin(phi) + scalar_i0 * np.cos(phi)
    e_scal = float(np.sum((scalar_r0 + sr)**2 + (scalar_i0 + si)**2))
    scalar_vals.append(e_scal / (2.0 * norm_scalar))

spinor_arr = np.array(spinor_vals)
scalar_arr = np.array(scalar_vals)
phi_deg = np.degrees(phi_vals)

# ── Report key values ──────────────────────────────────────────────────────
idx = {
    "  0°": 0,
    "180°": n_points // 4,
    "360°": n_points // 2,
    "540°": 3 * n_points // 4,
    "720°": n_points - 1,
}

print(f"  {'Angle':>5}   {'Spinor':>8}   {'Scalar':>8}")
print(f"  {'-----':>5}   {'------':>8}   {'------':>8}")
for label, i in idx.items():
    print(f"  {label}    {spinor_arr[i]:8.4f}   {scalar_arr[i]:8.4f}")
print()

# Key assertions
nulled  = spinor_arr[n_points // 2] < 0.05   # spinor null at 360°
notnull = scalar_arr[n_points // 2] > 1.9    # scalar constructive at 360°
restored = spinor_arr[-1] > 1.9               # spinor restored at 720°

print("  Spinor null at 360°  (sign flip) :", "YES ✓" if nulled   else "NO ✗")
print("  Scalar full  at 360° (no flip)   :", "YES ✓" if notnull  else "NO ✗")
print("  Spinor full  at 720° (restored)  :", "YES ✓" if restored else "NO ✗")
print()

if nulled and notnull and restored:
    print("  ✓ 720° spinor periodicity CONFIRMED")
else:
    print("  ✗ Check failed")
print()

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(phi_deg, spinor_arr, color="#2166ac", lw=2.5,
        label="Spinor  R$_x$(φ)|↑⟩ — period 720°")
ax.plot(phi_deg, scalar_arr, color="#d6604d", lw=2.0, ls="--",
        label="Scalar  e$^{iφ}$|ψ⟩ — period 360°")

# Annotations at 360°
ax.axvline(360, color="grey", lw=0.8, ls=":")
ax.annotate("360°\nspinor = 0\n(sign flip)",
            xy=(360, spinor_arr[n_points // 2]), xytext=(390, 0.5),
            fontsize=8, color="#2166ac",
            arrowprops=dict(arrowstyle="->", color="#2166ac", lw=0.8))
ax.annotate("360°\nscalar = 2\n(constructive)",
            xy=(360, scalar_arr[n_points // 2]), xytext=(390, 1.6),
            fontsize=8, color="#d6604d",
            arrowprops=dict(arrowstyle="->", color="#d6604d", lw=0.8))

ax.set_xlabel("Rotation angle φ (degrees)", fontsize=12)
ax.set_ylabel("Normalised interference energy", fontsize=12)
ax.set_title("Spin 720° Periodicity — LFM Spinor vs Scalar",
             fontsize=13, fontweight="bold")
ax.set_xlim(0, 720)
ax.set_xticks([0, 90, 180, 270, 360, 450, 540, 630, 720])
ax.set_xlim(0, 720)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)

fig.tight_layout()
out_path = _OUT / "34_spin_720_periodicity.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  Plot saved → {out_path}")
