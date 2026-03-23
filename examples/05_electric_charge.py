"""05 — Electric Charge

So far we've used real-valued wave fields (Ψ ∈ ℝ), which give
gravity only.  Now we switch to complex fields (Ψ ∈ ℂ) and
discover that the PHASE of the wave acts as electric charge:

    phase θ = 0  → "electron"  (negative charge)
    phase θ = π  → "positron"  (positive charge)

Same phase  → constructive interference → energy UP  → REPEL
Opp. phase  → destructive interference → energy DOWN → ATTRACT

This is Coulomb's law, emerging from wave interference.  No
electromagnetic equations are added — just GOV-01 with complex Ψ.
"""

import numpy as np

import lfm

config = lfm.SimulationConfig(
    grid_size=48,
    field_level=lfm.FieldLevel.COMPLEX,
)

print("05 — Electric Charge")
print("=" * 55)
print()

# --- Experiment A: Same phase (both θ=0) → should REPEL ---
sim_same = lfm.Simulation(config)
sim_same.place_soliton((17, 24, 24), amplitude=5.0, sigma=3.5, phase=0.0)
sim_same.place_soliton((31, 24, 24), amplitude=5.0, sigma=3.5, phase=0.0)
sim_same.equilibrate()

psi_sq = sim_same.psi_real**2 + sim_same.psi_imag**2
sep_same_0 = lfm.measure_separation(psi_sq)

sim_same.run(steps=3000)
psi_sq = sim_same.psi_real**2 + sim_same.psi_imag**2
sep_same_f = lfm.measure_separation(psi_sq)

# --- Experiment B: Opposite phase (θ=0 and θ=π) → should ATTRACT ---
sim_opp = lfm.Simulation(config)
sim_opp.place_soliton((17, 24, 24), amplitude=5.0, sigma=3.5, phase=0.0)
sim_opp.place_soliton((31, 24, 24), amplitude=5.0, sigma=3.5, phase=np.pi)
sim_opp.equilibrate()

psi_sq = sim_opp.psi_real**2 + sim_opp.psi_imag**2
sep_opp_0 = lfm.measure_separation(psi_sq)

sim_opp.run(steps=3000)
psi_sq = sim_opp.psi_real**2 + sim_opp.psi_imag**2
sep_opp_f = lfm.measure_separation(psi_sq)

# --- Report ---
print("Same phase (both θ=0, same 'charge'):")
print(f"  Initial separation: {sep_same_0:.1f}")
print(f"  Final separation:   {sep_same_f:.1f}")
print(f"  Change: {sep_same_f - sep_same_0:+.1f} cells")
print()

print("Opposite phase (θ=0 and θ=π, opposite 'charges'):")
print(f"  Initial separation: {sep_opp_0:.1f}")
print(f"  Final separation:   {sep_opp_f:.1f}")
print(f"  Change: {sep_opp_f - sep_opp_0:+.1f} cells")
print()

delta_same = sep_same_f - sep_same_0
delta_opp = sep_opp_f - sep_opp_0

if delta_opp < delta_same:
    print("Opposite charges attract MORE than same charges →")
    print("Coulomb-like behavior from pure wave interference!")
else:
    print("(At this scale/amplitude, gravity dominates over EM.")
    print(" Try larger grids or smaller amplitudes to isolate EM.)")

print()
print("No Coulomb's law was injected.  Charge = phase.")
print()
print("Next: what is dark matter? (→ 06)")
