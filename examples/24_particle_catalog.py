"""24 — Particle Catalog

All particles known to LFM, with their quantum numbers and grid parameters.

The LFM particle catalog maps every known fundamental particle to a set of
simulation parameters (amplitude, sigma) derived from its mass ratio via the
CALC-31/CALC-32 formulas.  Mass comes from angular momentum: m/m_e = l(l+1).
"""

import lfm

print("24 — LFM Particle Catalog")
print("=" * 65)
print()
print(f"{'Name':<18} {'mass_ratio':>12} {'l':>4} {'charge':>8} {'field':>8} {'stable':>7}")
print("-" * 65)

# Sort by mass ratio so lighter particles come first
particles = sorted(lfm.PARTICLES.values(), key=lambda p: p.mass_ratio)

for p in particles:
    field_name = {0: "real", 1: "complex", 2: "color"}.get(p.field_level, "?")
    print(
        f"{p.name:<18} {p.mass_ratio:>12.3f} {p.l:>4d} {p.charge:>8.1f}"
        f" {field_name:>8} {str(p.stable):>7}"
    )

print("-" * 65)
print(f"Total: {len(lfm.PARTICLES)} particles")
print()
print("Key parameter relationships:")
print(f"  chi0 = {lfm.CHI0}  (background chi in empty space)")
print(f"  kappa = {lfm.KAPPA:.5f}  (chi-energy coupling)")
print()
print("Mass from angular momentum: m/m_e = l*(l+1)")
print("  electron: l=0  -> mass_ratio = 0*1  = 1  (reference)")
print("  muon:     l=14 -> mass_ratio = 14*15 = 210  (measured: 206.768)")
print("  proton:   l=42 -> mass_ratio = 42*43 = 1806 (measured: 1836.15)")
print()
print("Grid parameters at N=64:")
print()

for p in particles[:6]:  # show first 6 as examples
    amp = lfm.amplitude_for_particle(p, 64)
    sig = lfm.sigma_for_particle(p, 64)
    print(f"  {p.name:<14}  amplitude={amp:.3f}  sigma={sig:.2f}")

print()
print("Use 'from lfm import get_particle' to look up any particle by name.")
print("Use 'from lfm import create_particle' to place one in a simulation.")
