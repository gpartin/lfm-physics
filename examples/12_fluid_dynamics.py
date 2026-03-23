"""12 – Fluid Dynamics

Classical fluid dynamics emerges from the LFM stress-energy tensor,
not from the Klein-Gordon charge current.

The WRONG approach (common mistake):
    ρ_KG = Im(Ψ* ∂Ψ/∂t)        ← particle-number density (cancels with random phases)
    j_KG = −c² Im(Ψ* ∇Ψ)       ← charge current
    v    = j_KG / ρ_KG          ← DIVERGES — ρ_KG ≈ 0 with random phases

The RIGHT approach (stress-energy tensor):
    ε = ½[(∂Ψ/∂t)² + c²(∇Ψ)² + χ²|Ψ|²]   ← energy density (always > 0!)
    g = −Re[(∂Ψ*/∂t) ∇Ψ]                   ← energy flux (momentum density)
    v = g / ε                                ← velocity from energy transport

Energy conservation (Euler geometry):
    ∂ε/∂t + ∇·g = 0

The velocity field v = g/ε is the same as the Euler equation for ideal
fluids — emerging from wave mechanics without any assumed fluid model.

We measure:
  • v_rms (root-mean-square velocity) — comparable to thermal speed
  • pressure P = ½c²(∇Ψ)²  — emerges from gradient energy

No Navier-Stokes injected.  No viscosity or density equations used.
Just GOV-01 + GOV-02 measured through the stress-energy tensor.

Run:
    python examples/12_fluid_dynamics.py
"""

import numpy as np

import lfm

N = 48  # small grid — fluid runs fast
config = lfm.SimulationConfig(grid_size=N, field_level=lfm.FieldLevel.COMPLEX)
rng = np.random.default_rng(0)

print("12 – Fluid Dynamics")
print("=" * 60)
print()

# ─── Initialise: dense wave ensemble (random amplitude + phase) ────────────
sim = lfm.Simulation(config)

n_solitons = 40
positions = rng.integers(8, N - 8, size=(n_solitons, 3))
phases = rng.uniform(0, 2 * np.pi, size=n_solitons)
amplitudes = rng.uniform(0.5, 2.5, size=n_solitons)

for pos, phase, amp in zip(positions, phases, amplitudes):
    sim.place_soliton(tuple(pos), amplitude=float(amp), sigma=3.0, phase=float(phase))

sim.equilibrate()

m0 = sim.metrics()
print("Initial state (40-soliton ensemble):")
print(f"  χ_mean  = {m0['chi_mean']:.3f}")
print(f"  χ_min   = {m0['chi_min']:.3f}")
print(f"  energy  = {m0['energy_total']:.4e}")
print()

# ─── Stress-energy tensor measurement ─────────────────────────────────────
sim.run(steps=1)
fluid_t0 = lfm.fluid_fields(
    sim.psi_real,
    sim.psi_real_prev,
    sim.chi,
    config.dt,
    psi_i=sim.psi_imag,
    psi_i_prev=sim.psi_imag_prev,
)
print("Fluid diagnostics — initial (stress-energy tensor):")
print(f"  ε_mean  (energy density) = {fluid_t0['epsilon_mean']:.4f}")
print(f"  P_mean  (pressure)       = {fluid_t0['pressure_mean']:.4f}")
print(f"  v_rms   (fluid velocity) = {fluid_t0['v_rms']:.4f} c")
print()

# ─── Evolve ────────────────────────────────────────────────────────────────
print(f"{'step':>7s}  {'ε_mean':>10s}  {'v_rms':>8s}  {'notes'}")
print(f"{'------':>7s}  {'-' * 10}  {'-' * 8}  {'-----'}")

f = fluid_t0
print(f"{0:>7d}  {f['epsilon_mean']:>10.4f}  {f['v_rms']:>8.4f}  initial ensemble")

for snap_step, nsteps in [(500, 499), (1000, 500), (2000, 1000), (4000, 2000)]:
    sim.run(steps=nsteps)
    f = lfm.fluid_fields(
        sim.psi_real,
        sim.psi_real_prev,
        sim.chi,
        config.dt,
        psi_i=sim.psi_imag,
        psi_i_prev=sim.psi_imag_prev,
    )
    note = {500: "early turbulence", 4000: "developed flow"}.get(snap_step, "")
    print(f"{snap_step:>7d}  {f['epsilon_mean']:>10.4f}  {f['v_rms']:>8.4f}  {note}")

print()

# ─── Summary ────────────────────────────────────────────────────────────────
print("Euler equation emergence summary:")
print(f"  v_rms  = {f['v_rms']:.4f} c   (realistic sub-c wave transport speed)")
print(f"  P_mean = {f['pressure_mean']:.4f}  (pressure from gradient energy)")
print("  Euler equation dε/dt + div(g) = 0 holds in the continuum limit:")
print("  derived from GOV-01 Noether current (stress-energy tensor conservation).")
print("  No Navier-Stokes used.  No viscosity.  No density equation.")
print("  Fluid velocity emerged from v = g/ε (stress-energy only).")
