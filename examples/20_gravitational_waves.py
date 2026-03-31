"""
Example 20 — Gravitational-Wave Inspiral
=========================================

Two massive solitons orbit each other and slowly spiral inward as they
radiate gravitational energy.  We extract the χ-field strain h(x,t) and
measure the gravitational-wave quadrupole power as a function of time.

Physics demonstrated
--------------------
* χ perturbations carry GW information (h = (χ − χ₀)/χ₀)
* Quadrupole radiation formula: L_{GW} = (1/5) Tr(d³I/dt³)²
* Binary inspiral shrinks orbital radius over time
* Chirp: GW frequency and amplitude both increase as separation decreases

Run (takes ~30 s on CPU for N=48, much faster on GPU)::

    python examples/20_gravitational_waves.py
"""

from __future__ import annotations

import numpy as np

import lfm
from _common import make_out_dir, parse_no_anim, run_and_save_3d_movie


def main() -> None:
    _args = parse_no_anim()
    _OUT  = make_out_dir("20_gravitational_waves")

    N = 48
    config = lfm.SimulationConfig(
        grid_size=N,
        e_amplitude=8.0,
        report_interval=200,
    )
    sim = lfm.Simulation(config)

    # Place two massive solitons in a close binary configuration
    cx = N // 2
    sep = 10  # initial separation (lattice units)
    sim.place_soliton((cx - sep // 2, cx, cx), amplitude=8.0)
    sim.place_soliton((cx + sep // 2, cx, cx), amplitude=8.0)
    sim.equilibrate()

    print("Example 20: Gravitational-Wave Inspiral")
    print(f"  Grid: {N}³,  initial separation: {sep}")
    print(f"  χ_min (initial): {sim.chi.min():.3f}")

    # Run with energy-density snapshots every 100 steps
    steps_total = 3000
    snaps, _movie = run_and_save_3d_movie(
        sim, steps=steps_total, out_dir=_OUT, stem="gravitational_waves",
        field="chi_deficit", snapshot_every=100, no_anim=_args.no_anim,
        extra_fields=["energy_density"],
    )
    print(f"  Collected {len(snaps)} snapshots")

    # ── Gravitational-wave strain analysis ────────────────────────────────
    # Mean peak separation over time (proxy for orbital radius)
    separations = []
    for snap in snaps:
        ed = snap["energy_density"]
        peaks = lfm.find_peaks(ed, n=2)
        if len(peaks) >= 2:
            p0, p1 = peaks[0], peaks[1]
            d = np.sqrt(sum((p0[k] - p1[k]) ** 2 for k in range(3)))
            separations.append(float(d))
        else:
            separations.append(float("nan"))

    # GW strain amplitude (rms of h field over interior)
    h_rms = []
    for snap in snaps:
        h = lfm.gravitational_wave_strain(snap["chi"])
        interior = lfm.interior_mask(N)
        h_rms.append(float(np.sqrt(np.mean(h[interior] ** 2))))

    # GW power via quadrupole formula
    gw = lfm.gw_power(snaps, field="energy_density", dt=100 * config.dt)

    print("\n  Step    Sep(grid)   h_rms       L_GW")
    print("  ------  ----------  ----------  ----------")
    for i in range(0, len(snaps), 5):
        step = snaps[i]["step"]
        sep_i = separations[i] if not np.isnan(separations[i]) else "-"
        lum = float(gw["luminosity"][i])
        if isinstance(sep_i, float):
            print(f"  {step:6d}  {sep_i:10.2f}  {h_rms[i]:10.4e}  {lum:10.4e}")
        else:
            print(f"  {step:6d}  {'?':10}  {h_rms[i]:10.4e}  {lum:10.4e}")

    # ── Visualisation ─────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        from lfm.viz import spacetime_diagram

        # Space-time diagram: χ(x, t) along x-axis through the binary
        fig, ax = spacetime_diagram(
            snaps,
            field="chi",
            axis=0,
            center=N // 2,
            dt=100 * config.dt,
            title="χ space–time (binary inspiral)",
        )
        fig.savefig("gw_spacetime.png", dpi=120, bbox_inches="tight")
        print("\n  Saved: gw_spacetime.png")

        # GW luminosity plot
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        t = gw["t"]
        lum = gw["luminosity"]
        ax2.semilogy(t, np.abs(lum) + 1e-30, color="navy")
        ax2.set_xlabel("time (simulation units)")
        ax2.set_ylabel("|L_GW|")
        ax2.set_title("Gravitational-wave power — LFM quadrupole formula")
        ax2.grid(True, alpha=0.3)
        fig2.savefig("gw_power.png", dpi=120, bbox_inches="tight")
        print("  Saved: gw_power.png")

        plt.close("all")
    except ImportError:
        print("  (matplotlib not available — skipping plots)")

    print("\nDone.")


if __name__ == "__main__":
    main()
