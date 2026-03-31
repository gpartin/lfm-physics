"""36 — Spin Correlations (Entanglement Precursor)

When two spin-1/2 particles interact, their spins become correlated.
This example demonstrates two-particle spin correlations in LFM:

CONFIGURATIONS
--------------
A) Triplet |↑↑⟩         — both particles spin-up       (parallel)
B) Singlet-like |↑↓⟩    — antiparallel spins            (classical EPR)
C) Product |+x⟩|+x⟩     — both particles spin-right
D) Singlet superposition — (|↑↓⟩ − |↓↑⟩)/√2 field rep

KEY MEASUREMENTS
----------------
For each configuration:
  1. Local spin at each site: ⟨σ_z^A⟩, ⟨σ_z^B⟩
  2. χ well depth (identical for all — gravity is spin-blind)
  3. Joint correlation: C = ⟨σ_{n̂_A}^A⟩ × ⟨σ_{n̂_B}^B⟩
  4. CHSH Bell parameter S across 4 measurement angle pairs

CHSH BELL PARAMETER
-------------------
S = |E(a,b) − E(a,b') + E(a',b) + E(a',b')|
where E(θ_A, θ_B) = ⟨σ_{θ_A}^A⟩ × ⟨σ_{θ_B}^B⟩

Classical (local hidden variable) bound: S ≤ 2
Quantum entanglement (singlet) bound:    S ≤ 2√2 ≈ 2.828

Note on LFM and Bell inequalities
----------------------------------
LFM is a deterministic, local classical field theory.  It can produce
correlated two-body spin states, but the level of correlation depends
on the spin preparation.  Product states give S ≤ 2; maximally
correlated states approach S = 2 (the classical boundary).
For S > 2 in an LFM context, see Paper 085 (IT-05) which demonstrates
S = 2.829 via χ-geometry-mediated non-separable correlations when the
particles share a common χ well.  This example isolates the geometric
two-particle correlation structure as a foundation for that result.

Run::

    python examples/36_spin_entanglement.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lfm
from lfm.fields.spinor import gaussian_spinor, apply_rotation_x, apply_rotation_z
from lfm.analysis.spinor import (
    spinor_sigma_x, spinor_sigma_y, spinor_sigma_z, spinor_density
)
from lfm.fields.equilibrium import equilibrate_chi
from _common import make_out_dir, parse_no_anim

_args = parse_no_anim()
_OUT  = make_out_dir("36_spin_entanglement")

N  = 48
cx = N // 2
# Two particles well separated along x
xA = N // 4       # position 12
xB = 3 * N // 4   # position 36

print("36 — Spin Correlations (Entanglement Precursor)")
print("=" * 60)
print(f"  Grid: {N}³   Particle A at x={xA}, Particle B at x={xB}")
print(f"  Separation: {xB - xA} cells")
print()

# ── Helper: measure spin at a spatial subregion ──────────────────────────

def local_spin(psi_r, psi_i, x_center: int, half_width: int = 8):
    """Measure ⟨σ_z⟩, ⟨σ_x⟩, ⟨σ_y⟩ averaged over a slab around x_center."""
    x0 = max(0, x_center - half_width)
    x1 = min(N, x_center + half_width)
    sr = psi_r[:, x0:x1, :, :]
    si = psi_i[:, x0:x1, :, :]
    return (
        spinor_sigma_z(sr, si),
        spinor_sigma_x(sr, si),
        spinor_sigma_y(sr, si),
    )


def corr(psi_r, psi_i, theta_A: float, theta_B: float) -> float:
    """Joint spin correlation C(θ_A, θ_B) = ⟨σ_{θ_A}^A⟩ × ⟨σ_{θ_B}^B⟩.

    σ_θ = cos(θ) σ_z + sin(θ) σ_x  (measurement in z-x plane)
    """
    szA, sxA, _ = local_spin(psi_r, psi_i, xA)
    szB, sxB, _ = local_spin(psi_r, psi_i, xB)
    sA = np.cos(theta_A) * szA + np.sin(theta_A) * sxA
    sB = np.cos(theta_B) * szB + np.sin(theta_B) * sxB
    return sA * sB


def chsh(psi_r, psi_i) -> float:
    """Compute CHSH parameter S = |E(a,b)−E(a,b')+E(a',b)+E(a',b')|."""
    a,  b  = 0.0,          np.pi / 4
    a2, b2 = np.pi / 2,    3 * np.pi / 4
    Eab   = corr(psi_r, psi_i, a,  b)
    Eab2  = corr(psi_r, psi_i, a,  b2)
    Ea2b  = corr(psi_r, psi_i, a2, b)
    Ea2b2 = corr(psi_r, psi_i, a2, b2)
    return abs(Eab - Eab2 + Ea2b + Ea2b2)

# ── Build the 4 two-particle configurations ──────────────────────────────

def two_particle(spin_up_A: bool, spin_up_B: bool) -> tuple:
    """Two Gaussians at xA and xB, each in a pure eigenstate."""
    pr = np.zeros((2, N, N, N), dtype=np.float32)
    pi = np.zeros((2, N, N, N), dtype=np.float32)
    rA, iA = gaussian_spinor(N, (xA, cx, cx), amplitude=5.0, sigma=3.0,
                              spin_up=spin_up_A)
    rB, iB = gaussian_spinor(N, (xB, cx, cx), amplitude=5.0, sigma=3.0,
                              spin_up=spin_up_B)
    pr += rA + rB
    pi += iA + iB
    return pr, pi


def make_equatorial(x_pos: int, phi_equator: float) -> tuple:
    """Spinor at x_pos in state R_z(φ)|+y⟩ = R_z(φ) R_x(-π/2)|↑⟩.

    phi_equator=0  → |+y⟩   (0, +1, 0) on Bloch sphere
    phi_equator=-π/2 → |+x⟩  (+1, 0, 0)
    phi_equator=π/2  → |−x⟩  (-1, 0, 0)
    phi_equator=π    → |−y⟩  (0, -1, 0)
    """
    r0, i0 = gaussian_spinor(N, (x_pos, cx, cx), amplitude=5.0, sigma=3.0)
    r_py, i_py = apply_rotation_x(r0, i0, -np.pi / 2)  # → |+y⟩
    return apply_rotation_z(r_py, i_py, phi_equator)


def two_particle_x(phi_A: float, phi_B: float) -> tuple:
    """Both particles on the Bloch sphere equator.

    phi=−π/2 → |+x⟩, phi=+π/2 → |−x⟩, phi=0 → |+y⟩, phi=π → |−y⟩
    """
    pr = np.zeros((2, N, N, N), dtype=np.float32)
    pi = np.zeros((2, N, N, N), dtype=np.float32)
    rA, iA = make_equatorial(xA, phi_A)
    rB, iB = make_equatorial(xB, phi_B)
    pr += rA + rB
    pi += iA + iB
    return pr, pi


# A) Triplet |↑↑⟩
pr_tt, pi_tt = two_particle(True,  True)
# B) Singlet-like |↑↓⟩ (classical antiparallel product)
pr_td, pi_td = two_particle(True,  False)
# C) |+x⟩|+x⟩ — phi_equator = −π/2 → |+x⟩
pr_xx, pi_xx = two_particle_x(-np.pi/2, -np.pi/2)
# D) Singlet-like superposition: A→|+x⟩, B→|−x⟩  (anticorrelated in x-basis)
pr_sg, pi_sg = two_particle_x(-np.pi/2, +np.pi/2)

configs = {
    "A) Triplet  |↑↑⟩          ": (pr_tt, pi_tt),
    "B) Antipar. |↑↓⟩          ": (pr_td, pi_td),
    "C) Product  |+x⟩|+x⟩      ": (pr_xx, pi_xx),
    "D) Anticorr |+x⟩|−x⟩      ": (pr_sg, pi_sg),
}

# ── Measure everything ────────────────────────────────────────────────────
print("  Local spin observables and χ well depth:")
print(f"  {'Config':36} {'σ_z^A':>6} {'σ_z^B':>6}  "
      f"{'σ_x^A':>6} {'σ_x^B':>6}   {'χ_min':>7}   {'S':>6}")
print("  " + "-" * 88)

S_vals = []
chi_mins = []
sz_A_list, sz_B_list = [], []

for label, (pr, pi) in configs.items():
    szA, sxA, _ = local_spin(pr, pi, xA)
    szB, sxB, _ = local_spin(pr, pi, xB)
    density = spinor_density(pr, pi)
    chi = equilibrate_chi(density)
    chi_min = float(chi.min())
    S = chsh(pr, pi)
    S_vals.append(S)
    chi_mins.append(chi_min)
    sz_A_list.append(szA)
    sz_B_list.append(szB)
    print(f"  {label} {szA:+6.3f} {szB:+6.3f}  "
          f"{sxA:+6.3f} {sxB:+6.3f}   {chi_min:7.4f}   {S:6.4f}")

print()
print(f"  Classical CHSH bound: S ≤ 2.000")
print(f"  Quantum entanglement: S ≤ 2√2 ≈ {2*np.sqrt(2):.4f}")
print()

# χ spin-blindness check
chi_spread = max(chi_mins) - min(chi_mins)
print(f"  χ_min spread across all configs: {chi_spread:.6f} "
      f"({'spin-blind ✓' if chi_spread < 0.01 else 'NOT spin-blind ✗'})")
print()

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

cfg_labels = ["A\n|↑↑⟩", "B\n|↑↓⟩", "C\n|+x⟩|+x⟩", "D\n|+x⟩|−x⟩"]
x_pos = np.arange(len(cfg_labels))

# σ_z correlations
ax = axes[0]
ax.bar(x_pos - 0.2, sz_A_list, width=0.35, label="⟨σ_z^A⟩",
       color="#2166ac", alpha=0.85)
ax.bar(x_pos + 0.2, sz_B_list, width=0.35, label="⟨σ_z^B⟩",
       color="#d6604d", alpha=0.85)
ax.set_xticks(x_pos); ax.set_xticklabels(cfg_labels, fontsize=9)
ax.set_ylabel("⟨σ_z⟩", fontsize=11)
ax.set_title("Local z-Polarisation", fontsize=12, fontweight="bold")
ax.axhline(0, color="grey", lw=0.5)
ax.set_ylim(-1.3, 1.3)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.25)

# χ_min
ax = axes[1]
ax.bar(x_pos, chi_mins, color="#4dac26", alpha=0.85)
y_lo = min(chi_mins) - 0.5
y_hi = max(chi_mins) + 0.5
ax.axhline(lfm.CHI0, color="grey", lw=1.0, ls="--", label=f"χ₀={lfm.CHI0}")
ax.set_xticks(x_pos); ax.set_xticklabels(cfg_labels, fontsize=9)
ax.set_ylabel("χ_min", fontsize=11)
ax.set_title("Gravity is Spin-Blind\n(χ_min identical)", fontsize=12, fontweight="bold")
ax.set_ylim(y_lo, y_hi)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.25)

# CHSH S values
ax = axes[2]
colors = ["#d73027" if s > 2.0 else "#4dac26" for s in S_vals]
bars = ax.bar(x_pos, S_vals, color=colors, alpha=0.85)
ax.axhline(2.0,            color="#d73027", lw=1.5, ls="--",
           label="Classical bound (S=2)")
ax.axhline(2 * np.sqrt(2), color="#762a83", lw=1.0, ls=":",
           label=f"Quantum bound (S=2√2≈{2*np.sqrt(2):.3f})")
ax.set_xticks(x_pos); ax.set_xticklabels(cfg_labels, fontsize=9)
ax.set_ylabel("CHSH parameter S", fontsize=11)
ax.set_title("CHSH Bell Parameter", fontsize=12, fontweight="bold")
ax.set_ylim(0, 3.0)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.25)

fig.suptitle("Two-Particle Spin Correlations in LFM", fontsize=13, fontweight="bold")
fig.tight_layout()
out_path = _OUT / "36_spin_entanglement.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  Plot saved → {out_path}")
