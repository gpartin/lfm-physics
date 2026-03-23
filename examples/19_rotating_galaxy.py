"""19 - Rotating Galaxy and Flat Rotation Curves

Goal:
  Demonstrate that the χ field's memory effect (gravity from GOV-02)
  produces a flat rotation curve — one of the iconic signatures of dark
  matter in spiral galaxies.

  A massive central soliton seeds the χ well.  A disk of 50 test solitons
  is placed on circular orbits (tangential velocities) at various radii
  using initialize_disk().  We measure the circular velocity as a function
  of radius both from the χ gradient (v_chi) and from the enclosed mass
  (v_enc) and compare them.

Physics:
  - Central potential well: Δχ ∝ GM/r → χ(r) = χ₀√(1-r_s/r) (weak field ≈ χ₀-Λ/r)
  - Keplerian prediction: v_K ∝ r^{-1/2}  (falls off)
  - With χ memory: extended dark matter halo → v(r) ≈ flat
  - v_circ from χ gradient: v² = r·(c²/χ₀)|dχ/dr|
  - Enclosed mass method:  v² = G_eff·M_enc(r)/r

Expected output:
  Rotation curve beyond ~10 cells:
    Keplerian:  decreasing  (v ∝ r^{-1/2})
    From χ:     flatter than Keplerian  (memory effect)
  The discrepancy between v_keplerian and v_chi illustrates the LFM
  explanation for dark matter as χ memory.
"""

from __future__ import annotations

import numpy as np
import lfm
from lfm.analysis.observables import rotation_curve

# ── Configuration ─────────────────────────────────────────────────────────────
N = 128  # 128³ for a realistic rotation curve; use 64 for quick tests
N_DISK = 50  # number of disk solitons (test particles)
AMP_CENTRAL = 12.0
AMP_DISK = 4.0  # much lighter than central mass
V_SCALE = 0.05  # velocity scale for circular orbits
STEPS = 10_000  # enough for χ to settle
R_INNER = 8.0  # inner disk radius (avoid chi-well singularity)
R_OUTER = N // 2 - 6  # outer disk radius

cfg = lfm.SimulationConfig(
    grid_size=N,
    field_level=lfm.FieldLevel.REAL,
    boundary_type=lfm.BoundaryType.FROZEN,
    dt=0.02,
    report_interval=5000,
)

sim = lfm.Simulation(cfg)

print("19 - Rotating Galaxy: Flat Rotation Curves from χ Memory")
print("=" * 62)
print(f"Grid: {N}³   central amp: {AMP_CENTRAL}   disk solitons: {N_DISK}")
print(f"Disk radii: [{R_INNER:.0f}, {R_OUTER:.0f}] cells   v_scale: {V_SCALE}")

# ── Central massive soliton (galactic bulge + dark matter seed) ────────────────
cx = cy = cz = N // 2
sim.place_soliton((cx, cy, cz), amplitude=AMP_CENTRAL)
sim.equilibrate()

chi_min_init = float(sim.chi.min())
print(f"\nChi minimum after central soliton: {chi_min_init:.4f}  (vacuum = {lfm.CHI0})")

# ── Place disk of test particles ───────────────────────────────────────────────
print(f"\nPlacing {N_DISK} disk solitons on circular orbits…")
positions = lfm.initialize_disk(
    sim,
    n_solitons=N_DISK,
    r_inner=R_INNER,
    r_outer=R_OUTER,
    amplitude=AMP_DISK,
    plane_axis=2,  # disk normal along z
    add_velocities=True,
    v_scale=V_SCALE,
    seed=42,
)
print(f"Placed {len(positions)} disk solitons.")

# ── Measure rotation curve before running ─────────────────────────────────────
rc_before = rotation_curve(
    sim.chi,
    sim.energy_density,
    center=(cx, cy, cz),
    plane_axis=2,
)
print(f"\nRotation curve (before evolution, {len(rc_before['r'])} bins):")

# ── Evolve ────────────────────────────────────────────────────────────────────
print(f"\nEvolving {STEPS:,} steps…")
sim.run(steps=STEPS, record_metrics=False)

# ── Measure rotation curve after running ──────────────────────────────────────
rc_after = rotation_curve(
    sim.chi,
    sim.energy_density,
    center=(cx, cy, cz),
    plane_axis=2,
)

r = rc_after["r"]
v_chi = rc_after["v_chi"]
v_enc = rc_after["v_enc"]
v_kep = rc_after["v_keplerian"]
chi_r = rc_after["chi_profile"]

print("\nRotation curve (after evolution):")
print(f"  {'r':>6}  {'χ(r)':>8}  {'v_chi':>8}  {'v_enc':>8}  {'v_Kep':>8}")
print("  " + "-" * 48)

# Print every 5th bin for readability
step_ = max(1, len(r) // 15)
for i in range(0, len(r), step_):
    print(f"  {r[i]:6.1f}  {chi_r[i]:8.4f}  {v_chi[i]:8.5f}  {v_enc[i]:8.5f}  {v_kep[i]:8.5f}")

# ── Flatness metric ────────────────────────────────────────────────────────────
# Compare the outer-annulus velocity to the inner (excluding very inner region)
inner_mask = r > R_INNER + 2
outer_mask = r > (R_INNER + R_OUTER) / 2

if inner_mask.any() and outer_mask.any():
    v_inner = float(np.median(v_chi[inner_mask]))
    v_outer = float(np.median(v_chi[outer_mask]))
    flatness = v_outer / (v_inner + 1e-30)

    v_kep_inner = float(np.median(v_kep[inner_mask]))
    v_kep_outer = float(np.median(v_kep[outer_mask]))
    kep_flatness = v_kep_outer / (v_kep_inner + 1e-30)

    print(f"\nFlatness ratios (outer / inner):")
    print(f"  v_chi (LFM):     {flatness:.4f}  (1.0 = perfectly flat)")
    print(f"  v_Keplerian:     {kep_flatness:.4f}  (<1 = falling, i.e. no DM)")
    print(f"\n  LFM is {'FLATTER' if flatness > kep_flatness else 'STEEPER'} than Keplerian")

# ── Chi profile ────────────────────────────────────────────────────────────────
print(f"\nChi profile: min={chi_r.min():.4f}  max={chi_r.max():.4f}")
print(f"Chi global:  min={float(sim.chi.min()):.4f}  max={float(sim.chi.max()):.4f}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("SUMMARY")
print(f"  Rotation curve bins: {len(r)}")
print(f"  v_chi flatness ratio: {flatness:.4f}  (1 = flat, <1 = falling)")
print(f"  Keplerian flatness:   {kep_flatness:.4f}")
print()
print("Interpretation:")
print("  The χ memory effect (chi remains depressed where matter has been)")
print("  creates an extended dark-matter-like halo.  The rotation curve from")
print("  the chi gradient is therefore flatter than the Keplerian prediction")
print("  based on the visible (|Ψ|²) mass alone.")
print()
print("  This is the LFM explanation for galactic rotation curves without")
print("  introducing a dark matter particle — the 'dark matter' is the χ")
print("  field's memory of past energy concentrations (GOV-02 τ-averaging).")
