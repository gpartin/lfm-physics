"""07 — Matter Creation

Up to now we placed solitons by hand.  But where does matter come
from in the first place?

If χ oscillates at TWICE the natural frequency (Ω = 2χ₀ = 38),
GOV-01 becomes a Mathieu equation with unstable solutions.  Any
tiny Ψ perturbation — even machine epsilon — grows exponentially.

This is parametric resonance: the universe shakes the substrate and
matter crystallizes out of nothing.  No Ψ injected, no initial
energy — just a vibrating χ field.
"""

import numpy as np

import lfm
from _common import make_out_dir, parse_no_anim, run_and_save_3d_movie

_args = parse_no_anim()
_OUT  = make_out_dir("07_matter_creation")

N = 48
config = lfm.SimulationConfig(grid_size=N)
sim = lfm.Simulation(config)

print("07 — Matter Creation (Parametric Resonance)")
print("=" * 55)
print()

# --- Seed Ψ with machine-epsilon noise (NOT zero) ---
rng = np.random.default_rng(42)
noise = rng.normal(0, 1e-15, (N, N, N)).astype(np.float32)
sim.psi_real = noise

initial_energy = float(np.sum(sim.psi_real**2))
print(f"Initial |Ψ|² total = {initial_energy:.2e}  (essentially zero)")
print()

# --- Oscillate χ at 2χ₀ for parametric resonance ---
# We update χ every 10 steps (a full oscillation = 8 steps at dt=0.02,
# so this stays close to resonance even with coarse updates).
omega = 2 * lfm.CHI0  # resonance frequency = 38
amplitude = 3.0  # driving amplitude (χ₀ ± 3)
dt = config.dt
report_every = 500
total_steps = 5000

print(f"Driving χ at Ω = 2χ₀ = {omega}, amplitude ±{amplitude}")
print(f"Total steps: {total_steps}")
print()

energies = [initial_energy]
update_every = 10  # update χ every N steps

for step in range(0, total_steps, report_every):
    # Run in small chunks, refreshing the driven χ before each
    for sub in range(0, report_every, update_every):
        t = (sim.step) * dt
        chi_val = lfm.CHI0 + amplitude * np.sin(omega * t)
        chi_field = np.full((N, N, N), chi_val, dtype=np.float32)
        sim.chi = chi_field
        sim.run(steps=update_every)

    e = float(np.sum(sim.psi_real**2))
    energies.append(e)
    growth = e / initial_energy if initial_energy > 0 else float("inf")
    print(f"  Step {sim.step:5d}: |Ψ|² = {e:.4e}  (×{growth:.2e} from initial)")

final_energy = energies[-1]
growth_factor = final_energy / initial_energy if initial_energy > 0 else float("inf")

print()
if growth_factor > 100:
    print(f"RESULT: |Ψ|² grew by ×{growth_factor:.2e}!")
    print("  → Matter appeared from essentially nothing")
    print("  → Parametric resonance: χ oscillation → Ψ amplification")
elif growth_factor > 1.1:
    print(f"RESULT: |Ψ|² grew by ×{growth_factor:.1f}")
    print("  → Some amplification (more cycles would give more growth)")
else:
    print(f"RESULT: Growth factor = {growth_factor:.4f}")
    print("  → Weak resonance at this amplitude/frequency")

print()
print("In the early universe, χ oscillated violently after the")
print("Big Bang — and this resonance filled space with matter.")
print()
print("Next: put it all together and simulate a universe (→ 08)")

# 3-D movie of matter that was created by parametric resonance
snaps, _movie = run_and_save_3d_movie(
    sim, steps=500, out_dir=_OUT, stem="matter_creation",
    field="psi_real", snapshot_every=10, no_anim=_args.no_anim,
)
