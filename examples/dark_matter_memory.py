#!/usr/bin/env python3
"""
Dark Matter from χ Memory
==========================

Place a soliton, let it create a χ well via GOV-02, then remove the
soliton. The χ well persists — it remembers where matter was.

This is the LFM dark matter mechanism:
  1. Energy (|Ψ|²) exists at a location → GOV-02 reduces local χ
  2. Matter moves away → |Ψ|² drops to zero at that location
  3. χ well remains (GOV-02 has finite propagation speed)
  4. Other matter is attracted to the residual well
  → "Dark matter halo" with no actual matter present

Usage:
  python dark_matter_memory.py
"""

from __future__ import annotations

import numpy as np

import lfm


def main() -> None:
    N = 64
    center = N // 2

    config = lfm.SimulationConfig(
        grid_size=N,
        field_level=lfm.FieldLevel.REAL,
        boundary_type=lfm.BoundaryType.FROZEN,
        report_interval=500,
    )

    # Phase 1: Create matter and let χ respond
    print("Dark Matter from χ Memory")
    print("=" * 55)
    print()
    print("[Phase 1] Place soliton, let GOV-02 create χ well...")
    sim = lfm.Simulation(config)
    sim.place_soliton((center, center, center), amplitude=10.0, sigma=4.0)
    sim.equilibrate()

    m0 = sim.metrics()
    print(f"  χ_min = {m0['chi_min']:.2f}  (well depth: "
          f"{19.0 - m0['chi_min']:.2f} below vacuum)")

    sim.run(steps=2000)
    m1 = sim.metrics()
    print(f"  After 2000 steps: χ_min = {m1['chi_min']:.2f}")

    # Phase 2: Remove the soliton (set Ψ = 0 everywhere)
    print()
    print("[Phase 2] Remove all matter (set Ψ = 0)...")
    sim.psi_real = np.zeros((N, N, N), dtype=np.float32)

    chi_after_removal = sim.chi.copy()
    chi_center = float(chi_after_removal[center, center, center])
    print(f"  χ at center immediately after removal: {chi_center:.2f}")
    print(f"  |Ψ|² everywhere: {float(np.sum(sim.psi_real ** 2)):.2e}")

    # Phase 3: Watch the χ well persist
    print()
    print("[Phase 3] Evolve with no matter — does χ well persist?")

    for step_block in range(5):
        sim.run(steps=1000, record_metrics=False)
        chi = sim.chi
        chi_c = float(chi[center, center, center])
        chi_min = float(chi.min())
        print(f"  Step {(step_block + 1) * 1000:>5d}: "
              f"χ_center = {chi_c:.3f}, χ_min = {chi_min:.3f}")

    # Final assessment
    chi_final = sim.chi
    chi_center_final = float(chi_final[center, center, center])
    well_depth = 19.0 - chi_center_final

    print()
    print("-" * 55)
    if well_depth > 0.1:
        print(f"RESULT: χ well persists! Depth = {well_depth:.2f}")
        print("  → Gravitational well with NO matter present")
        print("  → This IS dark matter in LFM: substrate memory")
    else:
        print(f"RESULT: χ recovered to {chi_center_final:.2f} (well dissipated)")
        print("  (χ propagation refilled the well — try larger grid or deeper well)")

    print()
    print("Physics: dark matter is not a substance — it's the")
    print("substrate's memory of where matter used to be.")


if __name__ == "__main__":
    main()
