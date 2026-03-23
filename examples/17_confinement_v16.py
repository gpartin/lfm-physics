"""17 - Confinement and Flux Tubes: v16 S_a Auxiliary Fields

Goal:
  Demonstrate the v16 confinement mechanism.  Two coloured solitons are
  placed on a COLOR-level grid with the S_a auxiliary fields enabled
  (kappa_tube > 0).  The S_a fields diffuse from each soliton core and
  their smoothed colour variance (SCV) sources an extra term in GOV-02
  that deepens chi *between* the sources, forming a flux-tube-like channel.

  This is NOT full QCD confinement (no linear potential with string tension),
  but it demonstrates:
    - S_a allocation and diffusion equilibrium
    - SCV-driven chi deepening between coloured sources
    - flux_tube_profile: chi profile along the line between sources
    - string_tension: proxy tension from the slope of chi midpoint vs separation

Physics:
  GOV-02 (v16) adds:
      - kappa_tube * SCV     (S_a smoothed colour variance)
      - kappa_string * CCV   (colour current variance)
  where SCV = Σ_a S_a^2 - (1/3)(Σ_a S_a)^2

  S_a evolves via reaction-diffusion:
      dS_a/dt = D·∇²S_a + γ(|Ψ_a|² − S_a)

Expected output (varies with grid and amplitude):
  chi midpoint (before run): ~18.x
  chi midpoint (after run):  < 18.0  (deepened by flux tube)
  flux tube dip:             several tenths below background
  Peak SCV near soliton cores, dropping to near-zero in void
"""

from __future__ import annotations

import numpy as np
import lfm
from lfm.constants import CHI0, KAPPA, KAPPA_C, KAPPA_STRING, KAPPA_TUBE, LAMBDA_H

# ── Configuration ─────────────────────────────────────────────────────────────
N = 64
SEP = 20          # half-separation (solitons at N/2 ± SEP/2 along z)
AMP = 8.0         # soliton amplitude
STEPS = 15_000
KAPPA_TUBE_DEMO = 10.0 * KAPPA   # 10× default coupling (stronger signal on small grid)

cfg = lfm.SimulationConfig(
    grid_size=N,
    field_level=lfm.FieldLevel.COLOR,
    kappa_c=KAPPA_C,
    kappa_tube=KAPPA_TUBE_DEMO,
    kappa_string=KAPPA_STRING,
    lambda_self=LAMBDA_H,
    dt=0.02,
    report_interval=5000,
)

sim = lfm.Simulation(cfg)

print("17 - Confinement & Flux Tubes via v16 S_a Fields")
print("=" * 62)

# ── Place two solitons along the z axis ───────────────────────────────────────
cx = cy = N // 2
z1 = N // 2 - SEP // 2
z2 = N // 2 + SEP // 2

sim.place_soliton((cx, cy, z1), amplitude=AMP)
sim.place_soliton((cx, cy, z2), amplitude=AMP)
sim.equilibrate()

# Confirm SA allocated
sa = sim.sa_fields
print(f"S_a allocated:   {'YES' if sa is not None else 'NO'}")
print(f"S_a shape:       {sa.shape if sa is not None else 'N/A'}")
print(f"S_a max (init):  {sa.max():.4e}" if sa is not None else "")

# ── Measure before ─────────────────────────────────────────────────────────────
chi_mid_before = lfm.measure_chi_midpoint(
    sim.chi, (cx, cy, z1), (cx, cy, z2)
)
chi_global_min_before = float(sim.chi.min())
print(f"\nBEFORE run ({STEPS:,} steps):")
print(f"  chi midpoint:        {chi_mid_before:.4f}")
print(f"  chi global min:      {chi_global_min_before:.4f}")

# ── Evolve ────────────────────────────────────────────────────────────────────
sim.run(steps=STEPS, record_metrics=False)

# ── Measure after ─────────────────────────────────────────────────────────────
sa_after = sim.sa_fields
chi_mid_after = lfm.measure_chi_midpoint(
    sim.chi, (cx, cy, z1), (cx, cy, z2)
)
chi_global_min_after = float(sim.chi.min())

print(f"\nAFTER run:")
print(f"  chi midpoint:        {chi_mid_after:.4f}")
print(f"  chi global min:      {chi_global_min_after:.4f}")
print(f"  chi midpoint change: {chi_mid_after - chi_mid_before:+.4f}")
print(f"  S_a max (equilib):   {sa_after.max():.4e}" if sa_after is not None else "")

# ── Flux tube profile ─────────────────────────────────────────────────────────
profile = lfm.flux_tube_profile(
    sim.chi, sim.sa_fields, (cx, cy, z1), (cx, cy, z2)
)
print("\nFlux-tube profile along z axis (chi at each point):")
chi_along_z = profile["chi_profile"]
sa0_along_z = profile["sa_profile"][0] if "sa_profile" in profile else None
z_pts = profile.get("z_coords", np.arange(len(chi_along_z)))
print(f"  z range:   [{z_pts.min():.0f}, {z_pts.max():.0f}]")
print(f"  chi range: [{chi_along_z.min():.3f}, {chi_along_z.max():.3f}]")
if sa0_along_z is not None:
    print(f"  S_a[0] range: [{sa0_along_z.min():.4e}, {sa0_along_z.max():.4e}]")

# ── String-tension proxy ──────────────────────────────────────────────────────
tension = lfm.string_tension(sim)
print(f"\nString-tension proxy:  {tension:.6f}")
print(f"(non-zero ⟹ chi gradient between sources)")

# ── SCV statistics ────────────────────────────────────────────────────────────
if sa_after is not None:
    scv = lfm.smoothed_color_variance(sa_after)
    print(f"\nSCV statistics:")
    print(f"  mean SCV: {scv.mean():.4e}")
    print(f"  max  SCV: {scv.max():.4e}")
    print(f"  SCV near soliton 1 (z={z1}): {scv[cx, cy, z1]:.4e}")
    print(f"  SCV near soliton 2 (z={z2}): {scv[cx, cy, z2]:.4e}")
    print(f"  SCV at midpoint (z={N//2}):   {scv[cx, cy, N//2]:.4e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
deepened = chi_mid_after < chi_mid_before
print(f"Flux tube formed:        {'YES' if deepened else 'NO (check amplitude)'}")
print(f"Chi deepened at midpoint: {chi_mid_before:.3f} → {chi_mid_after:.3f}")
print(f"SA diffusion active:     {'YES' if (sa_after is not None and sa_after.max() > 1e-8) else 'NO'}")
print()
print("Interpretation:")
print("  The S_a fields diffuse from the coloured soliton cores and their")
print("  smoothed colour variance (SCV) deepens chi along the connecting")
print("  line.  This is the v16 flux-tube mechanism.")
print("  For full linear confinement, see v16 string_tension documentation.")
