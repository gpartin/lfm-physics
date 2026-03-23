"""15 – Visualisation & Analysis

The lfm.viz toolkit: plot slices, radial profiles, time-series
dashboards, power spectra, and parameter sweeps without writing
any matplotlib boilerplate.

Requires: pip install "lfm-physics[viz]"
"""

import lfm

# ── 1. Run a quick simulation ─────────────────────────────────────────

config = lfm.SimulationConfig(grid_size=32)
sim = lfm.Simulation(config)
sim.place_soliton((16, 16, 16), amplitude=6.0)
sim.equilibrate()
sim.run(steps=3000)

print("15 – Visualisation & Analysis")
print("=" * 55)
print()

m = sim.metrics()
print(f"  chi_min = {m['chi_min']:.2f}  (gravity well depth)")
print(f"  wells   = {m['well_fraction'] * 100:.1f}%")
print()

# ── 2. 2D slice through the chi field ─────────────────────────────────

from lfm.viz import plot_slice, plot_three_slices

fig, ax = plot_slice(sim.chi, axis=2, index=16, title="χ mid-plane (z=16)")
fig.savefig("tutorial_15_slice.png", dpi=120, bbox_inches="tight")
print("Saved: tutorial_15_slice.png")

# Three-panel overview (XY, XZ, YZ)
fig = plot_three_slices(sim.chi, title="χ field — three planes")
fig.savefig("tutorial_15_three_slices.png", dpi=120, bbox_inches="tight")
print("Saved: tutorial_15_three_slices.png")

# ── 3. Radial profile with 1/r reference ──────────────────────────────

from lfm.viz import plot_radial_profile

fig, ax = plot_radial_profile(sim.chi, center=(16, 16, 16), max_radius=12)
fig.savefig("tutorial_15_radial.png", dpi=120, bbox_inches="tight")
print("Saved: tutorial_15_radial.png")

# ── 4. Chi histogram ──────────────────────────────────────────────────

from lfm.viz import plot_chi_histogram

fig, ax = plot_chi_histogram(sim.chi, title="χ distribution after 3000 steps")
fig.savefig("tutorial_15_histogram.png", dpi=120, bbox_inches="tight")
print("Saved: tutorial_15_histogram.png")

# ── 5. Time-evolution dashboard ───────────────────────────────────────

from lfm.viz import plot_evolution

fig = plot_evolution(sim.history, title="Metric evolution")
fig.savefig("tutorial_15_evolution.png", dpi=120, bbox_inches="tight")
print("Saved: tutorial_15_evolution.png")

# ── 6. Fourier power spectrum ─────────────────────────────────────────

from lfm.viz import plot_power_spectrum

fig, ax = plot_power_spectrum(sim.chi, title="P(k) of χ field")
fig.savefig("tutorial_15_spectrum.png", dpi=120, bbox_inches="tight")
print("Saved: tutorial_15_spectrum.png")

# ── 7. Parameter sweep ───────────────────────────────────────────────

from lfm.viz import plot_sweep

sweep_cfg = lfm.SimulationConfig(grid_size=32)
results = lfm.sweep(
    sweep_cfg,
    param="amplitude",
    values=[2, 4, 6, 8],
    steps=2000,
    metric_names=["chi_min", "well_fraction"],
)
fig, ax = plot_sweep(
    results, x_param="amplitude", y_metric="chi_min", title="χ_min vs soliton amplitude"
)
fig.savefig("tutorial_15_sweep.png", dpi=120, bbox_inches="tight")
print("Saved: tutorial_15_sweep.png")

print()
print("All plots saved.  Open the PNG files to explore your simulation.")
print("Every lfm.viz function returns (fig, ax) so you can customise further.")
