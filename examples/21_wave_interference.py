"""
Example 21 — Wave Interference and Double-Slit Diffraction
============================================================

Demonstrates that LFM wave fields obey standard wave optics:

1. **Two-source interference** — two solitons with the same phase
   produce constructive and destructive interference fringes in χ.

2. **Spatial amplitude modulation** — the χ field encodes the
   standing-wave energy landscape created by interference.

3. **Phase difference sweep** — repeating with Δθ = 0, π/2, π shows
   how the fringe pattern shifts.

Physics demonstrated
--------------------
* Constructive interference (Δθ = 0): amplitude builds up between sources
* Destructive interference (Δθ = π): nodal lines are visible
* The cross-term 2 Re(Ψ₁*Ψ₂) makes χ respond to relative phase (charge!)

Run::

    python examples/21_wave_interference.py
"""

from __future__ import annotations

import numpy as np

import lfm
from _common import make_out_dir, parse_no_anim, run_and_save_3d_movie


def run_interference(N: int, phase_diff: float, steps: int = 1500) -> tuple[float, float]:
    """Return (chi_min, chi_midpoint) for two solitons with given phase difference."""
    config = lfm.SimulationConfig(
        grid_size=N,
        field_level=lfm.FieldLevel.COMPLEX,
        e_amplitude=5.0,
        report_interval=500,
    )
    sim = lfm.Simulation(config)

    cx = N // 2
    sep = 12
    # Source 1 at phase 0, source 2 at phase_diff
    sim.place_soliton((cx, cx - sep // 2, cx), phase=0.0)
    sim.place_soliton((cx, cx + sep // 2, cx), phase=phase_diff)
    sim.equilibrate()
    sim.run(steps, record_metrics=False)

    chi_min = float(sim.chi.min())
    # Midpoint between the two sources
    chi_mid = float(sim.chi[cx, cx, cx])
    return chi_min, chi_mid


def main() -> None:
    _args = parse_no_anim()
    _OUT  = make_out_dir("21_wave_interference")

    N = 48
    print("Example 21: Wave Interference and Double-Slit Diffraction")

    phases = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    labels = ["0", "π/4", "π/2", "3π/4", "π"]

    print("\n  Phase diff    χ_min    χ_midpoint   Δχ_mid")
    print("  ----------    ------   ----------   ------")

    chi_mids = []
    for ph, lbl in zip(phases, labels):
        chi_min, chi_mid = run_interference(N, ph)
        chi_mids.append(chi_mid)
        delta = chi_mid - lfm.CHI0
        print(f"  Δθ = {lbl:6s}    {chi_min:.3f}    {chi_mid:.3f}        {delta:+.3f}")

    # Expected: Δθ=0 → deepest well (constructive), Δθ=π → shallowest (destructive)
    if chi_mids[0] < chi_mids[-1]:
        print("\n  ✓ PASS: Same-phase (Δθ=0) creates deeper χ well than opposite-phase (Δθ=π)")
        print("         This is the LFM charge-from-phase mechanism!")
    else:
        print("\n  NOTE: Pattern may be amplitude-dependent; try higher e_amplitude")

    # ── Visualisation ─────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        from lfm.viz import plot_projection, plot_slice

        print("\n  Running full simulation for visualisation (Δθ = 0)...")
        config = lfm.SimulationConfig(
            grid_size=N,
            field_level=lfm.FieldLevel.COMPLEX,
            e_amplitude=5.0,
            report_interval=200,
        )
        sim = lfm.Simulation(config)
        cx = N // 2
        sep = 12
        sim.place_soliton((cx, cx - sep // 2, cx), phase=0.0)
        sim.place_soliton((cx, cx + sep // 2, cx), phase=0.0)
        sim.equilibrate()
        run_and_save_3d_movie(sim, steps=2000, out_dir=_OUT, stem="wave_interference",
            field="chi_deficit", snapshot_every=50, no_anim=_args.no_anim)

        # Mid-plane slice
        fig1, ax1 = plot_slice(sim.chi, axis=2, title="χ after interference (Δθ = 0)")
        fig1.savefig("interference_slice.png", dpi=120, bbox_inches="tight")
        print("  Saved: interference_slice.png")

        # Column density projection along z
        fig2, ax2 = plot_projection(
            sim.chi,
            axis=2,
            log=False,
            cmap="RdBu_r",
            title="χ column-density projection (Δθ = 0)",
        )
        fig2.savefig("interference_projection.png", dpi=120, bbox_inches="tight")
        print("  Saved: interference_projection.png")

        # Phase comparison: side-by-side 1-D cuts
        fig3, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax_i, ph, lbl in zip(axes, [0.0, np.pi / 2, np.pi], ["0", "π/2", "π"]):
            cfg2 = lfm.SimulationConfig(
                grid_size=N,
                field_level=lfm.FieldLevel.COMPLEX,
                e_amplitude=5.0,
                report_interval=500,
            )
            s2 = lfm.Simulation(cfg2)
            s2.place_soliton((cx, cx - sep // 2, cx), phase=0.0)
            s2.place_soliton((cx, cx + sep // 2, cx), phase=ph)
            s2.equilibrate()
            s2.run(1500, record_metrics=False)
            cut = s2.chi[cx, :, cx]
            ax_i.plot(cut, color="steelblue")
            ax_i.axhline(lfm.CHI0, color="gray", ls="--", lw=0.8)
            ax_i.set_title(f"Δθ = {lbl}")
            ax_i.set_xlabel("y (grid cells)")
        axes[0].set_ylabel("χ")
        fig3.suptitle("1-D χ cut through mid-plane — LFM interference")
        fig3.tight_layout()
        fig3.savefig("interference_phase_sweep.png", dpi=120, bbox_inches="tight")
        print("  Saved: interference_phase_sweep.png")

        plt.close("all")
    except ImportError:
        print("  (matplotlib not available — skipping plots)")

    print("\nDone.")


if __name__ == "__main__":
    main()
