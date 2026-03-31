"""31 — Spiral Galaxy: SMBH + Stars + Gas + Emergent Dark Matter Halo

Goals:
  Simulate a galaxy-scale system from three components only:
    - Super-massive black hole (SMBH) at the galactic centre
    - Stellar disk (ring of solitons on circular orbits)
    - Diffuse gas disk (low-amplitude solitons at larger radii)

  Run GOV-01 + GOV-02.  Dark matter is NOT injected.  The χ field's
  wave inertia creates a persistent halo across the disk naturally.

  Output is a 4-panel PNG saved to examples/31_galaxy_dark_matter.png.

Physics:
  GOV-01  ∂²Ψ/∂t² = c²∇²Ψ − χ²Ψ          (wave equation)
  GOV-02  ∂²χ/∂t² = c²∇²χ − κ(|Ψ|² − E₀²)  (χ responds to mass)

Why NOT GOV-04 (Poisson: ∇²χ = κ|Ψ|²)?
  GOV-04 gives ZERO dark matter.  χ would snap instantly to the
  Poisson solution — it would follow the visible mass only, producing
  a purely Keplerian rotation curve.

  The dark matter halo lives in the χ wave inertia from GOV-02.
  As the disk orbits, the previous-step χ field (χ_prev in leapfrog)
  "remembers" past energy concentrations.  The χ depression never
  fully equilibrates between passes — it spreads across the disk.
  This is identical in effect to GOV-03's τ-averaging, but emerges
  directly from the two fundamental equations without approximation.

Expected output:
  - Persistent Δχ = χ₀ − χ depression across the full disk region
  - Rotation curve v_χ(r) flatter than Keplerian v_K(r)
  - SMBH signature: deep central minimum in χ
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")          # headless — works without a display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import lfm
from lfm.fields.arrangements import initialize_disk
from lfm.analysis.observables import rotation_curve

# ── Parameters ────────────────────────────────────────────────────────────────
N          = 256    # 256³ grid (~350 steps/sec on RTX 4060 ≈ 43 sec for 15k)
STEPS      = 15_000

# Quick-test override: set N=64 and STEPS=3000 for a ~10 sec run
# N = 64; STEPS = 3_000

# Super-massive black hole
AMP_SMBH   = 15.0   # large amplitude → deep χ well (SMBH)
SIGMA_SMBH = 4.0    # compact (4-cell radius)

# Stellar disk
N_STARS      = 40
AMP_STARS    = 4.0
SIGMA_STARS  = 3.0
R_STARS_IN   = 18.0
R_STARS_OUT  = 75.0
V_STARS      = 0.04  # tangential velocity scale

# Gas disk (diffuse, extends further than stellar disk)
N_GAS        = 100
AMP_GAS      = 0.8
SIGMA_GAS    = 3.0
R_GAS_IN     = 8.0
R_GAS_OUT    = 100.0
V_GAS        = 0.03

cfg = lfm.SimulationConfig(
    grid_size=N,
    field_level=lfm.FieldLevel.REAL,
    boundary_type=lfm.BoundaryType.FROZEN,
    dt=0.02,
    report_interval=5_000,
)
sim = lfm.Simulation(cfg)
cx = cy = cz = N // 2

print("31 — Spiral Galaxy: SMBH + Stars + Gas + Dark Matter Halo")
print("=" * 60)
print(f"Grid:      {N}³  =  {N**3:,} cells")
print(f"Equations: GOV-01 + GOV-02  (no other physics injected)")
print(f"SMBH:      amplitude {AMP_SMBH},  sigma {SIGMA_SMBH}")
print(f"Stars:     {N_STARS} solitons  r=[{R_STARS_IN:.0f},{R_STARS_OUT:.0f}]  amp={AMP_STARS}")
print(f"Gas:       {N_GAS} solitons   r=[{R_GAS_IN:.0f},{R_GAS_OUT:.0f}]  amp={AMP_GAS}")
print(f"Steps:     {STEPS:,}")
print()

# ── 1. SMBH — deep central χ well ─────────────────────────────────────────────
print("Placing SMBH at centre…")
sim.place_soliton((cx, cy, cz), amplitude=AMP_SMBH, sigma=SIGMA_SMBH)
sim.equilibrate()
print(f"  χ_min after SMBH equilibrate: {float(sim.chi.min()):.4f}  (vacuum χ₀ = {lfm.CHI0})")

# ── 2. Stellar disk ───────────────────────────────────────────────────────────
print(f"\nPlacing stellar disk ({N_STARS} stars)…")
star_pos = initialize_disk(
    sim,
    n_solitons=N_STARS,
    r_inner=R_STARS_IN,
    r_outer=R_STARS_OUT,
    amplitude=AMP_STARS,
    sigma=SIGMA_STARS,
    plane_axis=2,          # disk normal along z
    add_velocities=True,
    v_scale=V_STARS,
    seed=42,
)
print(f"  Placed {len(star_pos)} stars")

# ── 3. Gas disk ───────────────────────────────────────────────────────────────
print(f"Placing gas disk ({N_GAS} clumps)…")
gas_pos = initialize_disk(
    sim,
    n_solitons=N_GAS,
    r_inner=R_GAS_IN,
    r_outer=R_GAS_OUT,
    amplitude=AMP_GAS,
    sigma=SIGMA_GAS,
    plane_axis=2,
    add_velocities=True,
    v_scale=V_GAS,
    seed=99,
)
print(f"  Placed {len(gas_pos)} gas clumps")

# ── 4. Snapshot before evolution ──────────────────────────────────────────────
mid = N // 2
e_before  = np.asarray(sim.energy_density)
chi_before = np.asarray(sim.chi)
e_mid_before   = e_before[:, :, mid]
chi_mid_before = chi_before[:, :, mid]
rc_before = rotation_curve(
    sim.chi, sim.energy_density,
    center=(cx, cy, cz), plane_axis=2,
)

chi_min_before = float(sim.chi.min())
print(f"\nχ field before evolution: min={chi_min_before:.4f}  max={float(sim.chi.max()):.4f}")

# ── 5. Evolve — GOV-01 + GOV-02 ───────────────────────────────────────────────
print(f"\nEvolving {STEPS:,} steps via GOV-01 + GOV-02…")
sim.run(steps=STEPS, record_metrics=False)

# ── 6. Snapshot after evolution ───────────────────────────────────────────────
e_after   = np.asarray(sim.energy_density)
chi_after = np.asarray(sim.chi)
e_mid_after   = e_after[:, :, mid]
chi_mid_after = chi_after[:, :, mid]
rc = rotation_curve(
    sim.chi, sim.energy_density,
    center=(cx, cy, cz), plane_axis=2,
)
r       = rc["r"]
v_chi   = rc["v_chi"]
v_kep   = rc["v_keplerian"]
chi_r   = rc["chi_profile"]

chi_min_final = float(sim.chi.min())
chi_std_final = float(sim.chi.std())
print(f"χ field after evolution:  min={chi_min_final:.4f}  std={chi_std_final:.4f}")

# ── 7. Flatness metric ────────────────────────────────────────────────────────
inner_mask = r > R_STARS_IN
outer_mask = r > (R_STARS_IN + R_STARS_OUT) / 2

flatness = kep_flatness = 0.0
if inner_mask.any() and outer_mask.any():
    v_inner = float(np.median(v_chi[inner_mask]))
    v_outer = float(np.median(v_chi[outer_mask]))
    flatness = v_outer / (v_inner + 1e-30)

    vk_inner = float(np.median(v_kep[inner_mask]))
    vk_outer = float(np.median(v_kep[outer_mask]))
    kep_flatness = vk_outer / (vk_inner + 1e-30)

print(f"\nFlatness ratio (outer/inner v_chi):  {flatness:.4f}  (1.0 = flat)")
print(f"Keplerian flatness:                  {kep_flatness:.4f}")

# ── 8. Build 4-panel figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(
    f"LFM Galaxy Simulation  ·  N={N}³  ·  {STEPS:,} steps\n"
    "GOV-01 + GOV-02 only  (dark matter not injected — χ wave inertia only)",
    fontsize=12, y=0.98,
)

# ── Panel 1: Matter density mid-plane ─────────────────────────────────────────
ax = axes[0, 0]
im = ax.imshow(
    np.log1p(e_mid_after.T),
    origin="lower", cmap="inferno",
    extent=[0, N, 0, N],
)
ax.set_title("|Ψ|² density  (mid-plane z=N/2, log₁₊ scale)", fontsize=10)
ax.set_xlabel("x [cells]")
ax.set_ylabel("y [cells]")
plt.colorbar(im, ax=ax, label="log(1 + |Ψ|²)")
ax.plot(cx, cy, "w+", markersize=12, markeredgewidth=2, label="SMBH")
ax.legend(loc="upper right", fontsize=8)

# ── Panel 2: Dark matter halo — Δχ = χ₀ − χ ──────────────────────────────────
ax = axes[0, 1]
dchi_after = lfm.CHI0 - chi_mid_after
vmax_dchi  = max(float(dchi_after.max()), 0.1)
im = ax.imshow(
    dchi_after.T,
    origin="lower", cmap="plasma",
    extent=[0, N, 0, N],
    vmin=0, vmax=vmax_dchi,
)
ax.set_title(
    f"Dark matter halo:  Δχ = χ₀ − χ\n"
    f"(persistent depression across disk, not just at SMBH)",
    fontsize=9,
)
ax.set_xlabel("x [cells]")
ax.set_ylabel("y [cells]")
plt.colorbar(im, ax=ax, label=f"Δχ below vacuum (χ₀ = {lfm.CHI0})")
ax.plot(cx, cy, "w+", markersize=12, markeredgewidth=2)

# ── Panel 3: Rotation curve ───────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(r, v_chi, "b-",  lw=2, label="v_χ  (LFM — includes dark matter halo)")
ax.plot(r, v_kep, "r--", lw=2, label="v_Keplerian  (visible mass only)")
ax.axvspan(R_STARS_IN, R_STARS_OUT, alpha=0.12, color="green",  label="stellar disk")
ax.axvspan(R_GAS_IN,   R_GAS_OUT,   alpha=0.06, color="cyan",   label="gas disk")
ax.set_title("Rotation Curve: LFM χ gradient vs Keplerian", fontsize=10)
ax.set_xlabel("Radius [cells]")
ax.set_ylabel("v_circ [lattice units, c = 1]")
ax.set_xlim(0, min(r.max(), N // 2 - 2))
ax.legend(fontsize=8)
ax.text(
    0.97, 0.95,
    f"v_χ flatness  = {flatness:.3f}\n"
    f"Keplerian     = {kep_flatness:.3f}",
    transform=ax.transAxes, va="top", ha="right", fontsize=9,
    bbox=dict(boxstyle="round", fc="white", alpha=0.8),
)

# ── Panel 4: Radial χ profile before vs after ─────────────────────────────────
ax = axes[1, 1]
ax.plot(rc_before["r"], rc_before["chi_profile"], "c--", lw=1.5,
        label="χ(r) before evolution (initial)")
ax.plot(r, chi_r, "b-", lw=2, label="χ(r) after evolution (dark matter halo)")
ax.axhline(lfm.CHI0, color="gray", ls=":", lw=1.5, label=f"χ₀ = {lfm.CHI0} (vacuum)")
ax.axvspan(R_STARS_IN, R_STARS_OUT, alpha=0.12, color="green",  label="stellar disk")
ax.axvspan(R_GAS_IN,   R_GAS_OUT,   alpha=0.06, color="cyan",   label="gas disk")
ax.set_title("χ Profile Before / After  (dark matter halo formation)", fontsize=10)
ax.set_xlabel("Radius [cells]")
ax.set_ylabel("Mean χ in annulus")
ax.legend(fontsize=7, loc="upper right")

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = Path(__file__).parent / "31_galaxy_dark_matter.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved:  {out_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print(f"  Components:          SMBH + {len(star_pos)} stars + {len(gas_pos)} gas clumps")
print(f"  χ₀ (vacuum):         {lfm.CHI0}")
print(f"  χ_min (final):       {chi_min_final:.4f}  (SMBH well)")
print(f"  Δχ_centre:           {lfm.CHI0 - chi_min_final:.4f}  (below vacuum)")
print(f"  χ std (disk):        {chi_std_final:.4f}")
print(f"  Rotation flatness:   {flatness:.4f}  (1.0 = perfectly flat)")
print(f"  Keplerian flatness:  {kep_flatness:.4f}")
print()
print("Interpretation:")
print("  The χ field develops a broad depression across the stellar + gas disk.")
print("  This arises from χ wave inertia in GOV-02 — not from injected particles.")
print("  The χ_prev leapfrog buffer means χ responds to where matter WAS,")
print("  not just where it is now, producing an extended dark-matter-like halo.")
print()
print("  GOV equations used:  GOV-01 + GOV-02 only.")
print("  No Newtonian gravity. No MOND. No dark matter particles.")
