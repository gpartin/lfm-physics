"""
Example 22 — Soliton Normal Modes
===================================

An equilibrated soliton has a characteristic oscillation frequency set by
the curvature of the Mexican-hat potential:

    ω_H = √(8 · λ_H) · χ₀  ≈ 19.30 (lattice time units)

This example:
1. Creates and equilibrates a single soliton.
2. Applies a small radial perturbation to χ at the soliton core.
3. Runs ~12 000 steps, recording χ at the centre every step.
4. FFT-analyses the time series and identifies the dominant frequency.
5. Compares it to the theoretical Higgs / breathing-mode prediction.

Run::

    python examples/22_soliton_modes.py
"""

from __future__ import annotations

import numpy as np

import lfm
from lfm.constants import CHI0, LAMBDA_H


def _theory_omega() -> float:
    """Theoretical Higgs-breathing-mode frequency."""
    return float(np.sqrt(8.0 * LAMBDA_H) * CHI0)


def main() -> None:
    print("Example 22: Soliton Normal Modes")
    print(f"  Theory: ω_H = √(8·λ_H)·χ₀ = {_theory_omega():.4f} rad/step\n")

    N = 48
    DT = 0.02  # matches lfm default

    config = lfm.SimulationConfig(
        grid_size=N,
        field_level=lfm.FieldLevel.REAL,
        e_amplitude=6.0,
        sigma=2.5,
        report_interval=2000,
    )
    sim = lfm.Simulation(config)
    cx = N // 2
    sim.place_soliton((cx, cx, cx))
    sim.equilibrate()

    print("  Equilibration complete.  Applying small radial perturbation …")
    # Perturb χ at centre (+0.3) and immediate shell (-0.1) to excite breathing mode
    sim.chi[cx, cx, cx] += 0.3
    for di in (-1, 1):
        for dj in (-1, 1):
            for dk in (-1, 1):
                sim.chi[cx + di, cx + dj, cx + dk] -= 0.05

    # Record χ_centre at every step via callback
    chi_series: list[float] = []

    def record(s: lfm.Simulation, step: int) -> None:  # noqa: ARG001
        chi_series.append(float(s.chi[cx, cx, cx]))

    N_STEPS = 12_000
    # run() fires callback at report_interval boundaries, but we want every step.
    # Use run_with_snapshots with snapshot_every=1 for precise tracking.
    snaps = sim.run_with_snapshots(
        N_STEPS,
        snapshot_every=1,
        fields=["chi"],
        record_metrics=False,
    )
    chi_series = [float(sn["chi"][cx, cx, cx]) for sn in snaps]

    t = np.arange(len(chi_series)) * DT
    signal = np.array(chi_series)
    signal -= signal.mean()  # remove DC offset

    # ── FFT ──────────────────────────────────────────────────────────────
    freqs = np.fft.rfftfreq(len(signal), d=DT)  # cycles per lattice-time unit
    omegas = 2.0 * np.pi * freqs  # rad / lattice-time
    spectrum = np.abs(np.fft.rfft(signal))

    # Find dominant peak (skip DC bin 0)
    peak_idx = int(np.argmax(spectrum[1:])) + 1
    omega_measured = omegas[peak_idx]
    omega_theory = _theory_omega()
    error_pct = 100.0 * abs(omega_measured - omega_theory) / omega_theory

    print(f"\n  χ_centre oscillation analysis ({N_STEPS} steps):")
    print(f"    Measured ω = {omega_measured:.4f} rad/step")
    print(f"    Theory   ω = {omega_theory:.4f} rad/step")
    print(f"    Error      = {error_pct:.2f}%")

    if error_pct < 5.0:
        print("\n  ✓ PASS: Measured breathing frequency within 5% of theory")
    else:
        print(
            f"\n  NOTE: Difference is {error_pct:.1f}% — "
            "may need larger amplitude or longer run for clean mode isolation"
        )

    # ── Second-mode search: look for a sub-dominant peak ─────────────────
    # zero out the main peak ± 2 bins, find next
    spec2 = spectrum.copy()
    lo = max(0, peak_idx - 2)
    hi = min(len(spec2) - 1, peak_idx + 2)
    spec2[lo : hi + 1] = 0
    spec2[0] = 0
    peak2_idx = int(np.argmax(spec2))
    omega2 = omegas[peak2_idx]
    print(
        f"    Sub-dominant mode: ω = {omega2:.4f} rad/step "
        f"  (ratio = {omega2 / omega_measured:.3f})"
    )

    # ── Visualisation ─────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))

        # Time series
        axes[0].plot(t, signal + sim.chi.mean(), color="steelblue", lw=0.8)
        axes[0].set_xlabel("t (lattice time units)")
        axes[0].set_ylabel("χ centre")
        axes[0].set_title("Soliton Breathing Mode — χ at Core vs Time")
        axes[0].axhline(sim.chi.mean(), color="gray", ls="--", lw=0.8, label="mean")
        axes[0].legend()

        # Power spectrum
        axes[1].semilogy(omegas[1:], spectrum[1:], color="steelblue", lw=0.8)
        axes[1].axvline(
            omega_theory, color="red", ls="--", lw=1.5, label=f"ω_theory = {omega_theory:.2f}"
        )
        axes[1].axvline(
            omega_measured, color="orange", ls="-", lw=1.5, label=f"ω_meas = {omega_measured:.2f}"
        )
        axes[1].set_xlabel("ω (rad / lattice-time)")
        axes[1].set_ylabel("|FFT| (log scale)")
        axes[1].set_title("Power Spectrum of χ Core Oscillation")
        axes[1].legend()
        axes[1].set_xlim(0, 4 * omega_theory)

        fig.tight_layout()
        fig.savefig("soliton_modes.png", dpi=120, bbox_inches="tight")
        print("\n  Saved: soliton_modes.png")
        plt.close(fig)
    except ImportError:
        print("  (matplotlib not available — skipping plots)")

    print("\nDone.")


if __name__ == "__main__":
    main()
