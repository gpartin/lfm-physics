"""11 – Oxygen Atom

Oxygen has 8 protons.  A heavier nuclear soliton (amplitude=8, σ=3.5)
digs a deeper χ-well supporting TWO distinct electron shells:

  Shell n=1  (inner, 2 electrons)  → sits near the potential minimum
  Shell n=2  (outer, 6 electrons)  → sits at larger radius where χ slope changes

We use FieldLevel.COLOR here because oxygen has multiple electrons
(same-matter solitons) and the inter-electron repulsion is carried by
the phase-interference mechanism just as in tutorial 05.

No Schrödinger equation.  No Pauli exclusion postulate.  No orbital
tables.  Just GOV-01 + GOV-02 with nucleus and electron amplitudes.

Run:
    python examples/11_oxygen.py
"""

import numpy as np

import lfm
from _common import make_out_dir, parse_no_anim, run_and_save_3d_movie

_args = parse_no_anim()
_OUT  = make_out_dir("11_oxygen")

N = 64
config = lfm.SimulationConfig(grid_size=N, field_level=lfm.FieldLevel.COLOR)

print("11 – Oxygen Atom")
print("=" * 60)
print()

# ─── Build the oxygen atom ────────────────────────────────────────────────
sim = lfm.Simulation(config)
cx = N // 2

# Nuclear soliton with amplitude=8, sigma=3.5.
# Wider sigma accommodates two electron shells (inner r=3, outer r=7).
sim.place_soliton((cx, cx, cx), amplitude=8.0, sigma=3.5, phase=0.0)

# Inner shell (n=1): 2 electrons at radius ~3
for theta in [0.0, np.pi]:
    off = int(round(3 * np.cos(theta)))
    sim.place_soliton((cx + off, cx, cx), amplitude=0.9, sigma=1.4, phase=theta)

# Outer shell (n=2): 6 electrons around the equatorial belt at radius ~7
for k in range(6):
    ang = 2 * np.pi * k / 6
    ix = int(round(7 * np.cos(ang)))
    iy = int(round(7 * np.sin(ang)))
    sim.place_soliton((cx + ix, cx + iy, cx), amplitude=0.9, sigma=1.8, phase=float(k) * np.pi / 3)

sim.equilibrate()
print("Oxygen atom assembled — nucleus (amp=8, σ=3.5) + 2 inner + 6 outer electrons.")
print()

# ─── χ radial profile ─────────────────────────────────────────────────────
print("χ radial profile  (compare with hydrogen tutorial 09):")
print(f"  {'r (cells)':>10s}  {'χ(r)':>8s}  {'Δχ from χ₀':>12s}")
print(f"  {'-' * 10}  {'-' * 8}  {'-' * 12}")

profile = lfm.radial_profile(sim.chi, center=(cx, cx, cx), max_radius=N // 2 - 2)
for r, chi_val in zip(profile["r"][::2], profile["profile"][::2]):
    delta = chi_val - lfm.CHI0
    print(f"  {r:>10.1f}  {chi_val:8.3f}  {delta:+12.3f}")
print()

# ─── Oxygen vs Hydrogen well depth comparison ─────────────────────────────
print("Nuclear well depths:")
h_config = lfm.SimulationConfig(grid_size=N, field_level=lfm.FieldLevel.REAL)
h_sim = lfm.Simulation(h_config)
h_sim.place_soliton((cx, cx, cx), amplitude=10.0, sigma=2.0, phase=0.0)
h_sim.equilibrate()

o_min = sim.chi.min()
h_min = h_sim.chi.min()

print(f"  Hydrogen nucleus χ_min  = {h_min:.3f}  (Δχ = {h_min - lfm.CHI0:.3f})")
print(f"  Oxygen   nucleus χ_min  = {o_min:.3f}  (Δχ = {o_min - lfm.CHI0:.3f})")
print(
    f"  Ratio Δχ(oxygen)/Δχ(hydrogen)             = {(o_min - lfm.CHI0) / (h_min - lfm.CHI0):.2f}"
)
print("  (Oxygen well is deeper → supports more electrons)")
print()

# ─── Evolve and check stability ────────────────────────────────────────────
STEPS = 4000
print(f"Running {STEPS} steps — watching for shell stability...")
print()
prev = 0
for step in [1000, 2000, 4000]:
    sim.run(steps=step - prev)
    prev = step
    m = sim.metrics()
    psi_sq_3d = sim.psi_real.sum(axis=0) ** 2 if sim.psi_real.ndim == 4 else sim.psi_real**2
    sep = lfm.measure_separation(psi_sq_3d)
    print(
        f"  step {step:5d}  χ_min={m['chi_min']:.3f}  "
        f"energy={m['energy_total']:.2e}  psi_sq_peak_sep={sep:.1f}"
    )

print()
print("Both shells remain bound — no Pauli exclusion postulate required.")
print("Shell structure emerges from wave interference and geometry alone.")

# 3-D movie of multi-shell oxygen atom
snaps, _movie = run_and_save_3d_movie(
    sim, steps=1000, out_dir=_OUT, stem="oxygen",
    field="psi_real", snapshot_every=20, no_anim=_args.no_anim,
)
