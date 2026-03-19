"""03 — Measuring Gravity

In example 02 we saw χ drop below 19 around a soliton.  But is
this really gravity?  Let's measure the radial profile of χ and
see if it looks like Newton's 1/r potential.

We place a single massive soliton, equilibrate, evolve, then use
lfm.radial_profile() to extract χ(r) — the gravitational well shape.
"""

import lfm

config = lfm.SimulationConfig(grid_size=48)
sim = lfm.Simulation(config)

center = (24, 24, 24)
sim.place_soliton(center, amplitude=6.0, sigma=4.0)
sim.equilibrate()
sim.run(steps=1000)

print("03 — Measuring Gravity")
print("=" * 55)
print()

# Measure the radial chi profile.
prof = lfm.radial_profile(sim.chi, center=center, max_radius=18)

print("Radial χ profile (distance → chi value):")
print(f"  {'r':>4s}  {'χ(r)':>8s}  {'Δχ':>8s}  bar")
print(f"  {'----':>4s}  {'--------':>8s}  {'--------':>8s}  ---")

chi0 = lfm.CHI0
for r, chi_val in zip(prof["r"], prof["profile"]):
    if r < 1 or r > 16:
        continue
    delta = chi0 - chi_val
    bar = "█" * int(max(0, delta) * 4)
    print(f"  {r:4.0f}  {chi_val:8.3f}  {delta:8.3f}  {bar}")

print()
print(f"  χ at center = {prof['profile'][0]:.3f}  (deepest point)")
print(f"  χ at r=16   = {prof['profile'][16]:.3f}  (nearly vacuum)")
print()

# Check 1/r character: chi(r) should approach chi0 as 1/r
# At r=4 vs r=8, delta(4)/delta(8) should be ~2 if 1/r
d4 = chi0 - prof["profile"][4]
d8 = chi0 - prof["profile"][8]
if d8 > 0.001:
    ratio = d4 / d8
    print(f"  Δχ(r=4) / Δχ(r=8) = {ratio:.2f}  (expect ~2.0 for 1/r)")
else:
    print("  (well is shallow — increase amplitude for clearer 1/r)")

print()
print("The well is deepest at the center and drops off with distance,")
print("just like Newtonian gravity — but no F = -GM/r² was used.")
print()
print("Next: put TWO particles on the lattice (→ 04).")
