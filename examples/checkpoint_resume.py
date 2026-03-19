#!/usr/bin/env python3
"""
Checkpoint and Resume
=====================

Demonstrates saving and loading simulation state. This is essential
for long-running cosmological simulations that run for days.

  1. Create a simulation and run it partway
  2. Save a checkpoint (.npz file)
  3. Load the checkpoint into a fresh Simulation
  4. Continue running — metrics should be continuous

Usage:
  python checkpoint_resume.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import lfm


def main() -> None:
    print("Checkpoint / Resume Demo")
    print("=" * 50)

    # Phase 1: Run partway
    print("\n[1] Create simulation and run 2000 steps...")
    config = lfm.SimulationConfig(
        grid_size=32,
        field_level=lfm.FieldLevel.REAL,
        report_interval=1000,
    )
    sim = lfm.Simulation(config)
    sim.place_soliton((16, 16, 16), amplitude=8.0, sigma=3.0)
    sim.equilibrate()
    sim.run(steps=2000)

    m1 = sim.metrics()
    print(f"  Step {sim.step}: χ_min={m1['chi_min']:.3f}, "
          f"energy={m1['energy_total']:.4e}")

    # Phase 2: Save checkpoint
    ckpt_path = Path(tempfile.gettempdir()) / "lfm_demo_checkpoint.npz"
    print(f"\n[2] Saving checkpoint to {ckpt_path}...")
    sim.save_checkpoint(ckpt_path)
    print(f"  Saved ({ckpt_path.stat().st_size / 1024:.0f} KB)")

    # Phase 3: Load into fresh simulation
    print("\n[3] Loading checkpoint into fresh Simulation...")
    sim2 = lfm.Simulation.load_checkpoint(ckpt_path)
    m2 = sim2.metrics()
    print(f"  Restored at step {sim2.step}: χ_min={m2['chi_min']:.3f}, "
          f"energy={m2['energy_total']:.4e}")

    # Phase 4: Continue running
    print("\n[4] Continuing for 2000 more steps...")
    sim2.run(steps=2000)
    m3 = sim2.metrics()
    print(f"  Step {sim2.step}: χ_min={m3['chi_min']:.3f}, "
          f"energy={m3['energy_total']:.4e}")

    # Verify continuity
    drift = (abs(m3["energy_total"] - m1["energy_total"])
             / max(abs(m1["energy_total"]), 1e-30) * 100)
    print(f"\n  Energy drift across save/load: {drift:.3f}%")

    # Cleanup
    ckpt_path.unlink(missing_ok=True)

    print()
    print("-" * 50)
    print("Checkpoint stores: fields, step counter, config, history.")
    print("Use this for multi-day simulations (e.g. 256³ cosmic runs).")


if __name__ == "__main__":
    main()
