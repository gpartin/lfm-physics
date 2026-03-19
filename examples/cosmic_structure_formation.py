#!/usr/bin/env python3
"""
Cosmic Structure Formation — 256³ LFM Universe Simulation
==========================================================

Demonstrates how the LFM governing equations (GOV-01 + GOV-02) on a
discrete 3D lattice spontaneously produce cosmic structure: gravitational
wells where matter clumps and evacuated voids where χ rises toward χ₀ = 19.

This is a **simplified** version of the canonical universe simulator.
It sets up a 256³ grid (16.8 million cells) with random initial matter
fluctuations, evolves the coupled wave system, and tracks:

  - χ well formation (gravity → matter clumps)
  - Void formation (matter evacuation → χ rises)
  - Energy conservation

Physics
-------
Two equations, no external physics injected:

  GOV-01:  ∂²Ψ/∂t²  =  c²∇²Ψ  −  χ² Ψ
  GOV-02:  ∂²χ/∂t²   =  c²∇²χ  −  κ(|Ψ|² − E₀²)

All structure emerges from these dynamics alone.

Requirements
------------
  pip install lfm-physics          # CPU
  pip install lfm-physics[gpu]     # GPU (recommended for 256³)

Runtime
-------
  GPU (RTX 4060):  ~140 steps/sec  →  ~2.4 hrs for 1.2M steps
  CPU:             ~2 steps/sec    →  ~7 days (use smaller grid)

For a quick test, set GRID_SIZE = 64 below (runs in minutes on CPU).

Usage
-----
  python cosmic_structure_formation.py

License: MIT
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import lfm
from lfm import Simulation, SimulationConfig
from lfm.constants import CHI0, KAPPA, LAMBDA_H

# ═══════════════════════════════════════════════════════════════════
# Configuration — adjust these for your hardware
# ═══════════════════════════════════════════════════════════════════

GRID_SIZE = 256          # 256 for full cosmic sim, 64 for quick test
TOTAL_STEPS = 1_200_000  # 1.2M steps ≈ 30.6 Gyr of cosmic time
REPORT_EVERY = 10_000    # Print metrics every N steps
SNAPSHOT_EVERY = 50_000  # Save χ snapshot every N steps

# Physical box (for time conversion only, not used in equations)
BOX_SIZE_MPC = 100.0     # 100 Megaparsec box
MYR_PER_MPC = 3.262      # Myr per Mpc (1/H₀ conversion)

# Initial conditions
AMPLITUDE = 12.0         # Matter amplitude (sets initial energy)
SIGMA_FACTOR = 12.0      # σ = grid_size / sigma_factor

# Output directory
OUTPUT_DIR = Path("output_cosmic_sim")

# ═══════════════════════════════════════════════════════════════════
# Cosmic time conversion
# ═══════════════════════════════════════════════════════════════════

CELL_SIZE_MPC = BOX_SIZE_MPC / GRID_SIZE
TIME_PER_CODE_UNIT = CELL_SIZE_MPC * MYR_PER_MPC  # Myr per code unit
DT = 0.02  # timestep in code units
MYR_PER_STEP = DT * TIME_PER_CODE_UNIT
GYR_PER_STEP = MYR_PER_STEP / 1000.0

PRESENT_EPOCH_GYR = 13.8
PRESENT_STEP = int(PRESENT_EPOCH_GYR / GYR_PER_STEP)


def step_to_gyr(step: int) -> float:
    """Convert simulation step to cosmic time in Gyr."""
    return step * GYR_PER_STEP


# ═══════════════════════════════════════════════════════════════════
# Logging callback
# ═══════════════════════════════════════════════════════════════════

def on_report(sim: Simulation, step: int) -> None:
    """Called every report_interval steps to print progress."""
    t_gyr = step_to_gyr(step)
    m = sim.metrics()

    marker = " ← NOW" if abs(t_gyr - PRESENT_EPOCH_GYR) < GYR_PER_STEP * REPORT_EVERY else ""

    print(
        f"  Step {step:>9,d} | "
        f"t = {t_gyr:6.2f} Gyr | "
        f"χ_min = {m['chi_min']:6.2f} | "
        f"wells = {m['well_fraction']*100:5.1f}% | "
        f"voids = {m['void_fraction']*100:5.1f}% | "
        f"clusters = {m['n_clusters']:>5d}{marker}"
    )


# ═══════════════════════════════════════════════════════════════════
# Snapshot saver
# ═══════════════════════════════════════════════════════════════════

def save_snapshot(sim: Simulation, step: int) -> None:
    """Save a χ-field mid-plane slice + metrics to .npz file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_gyr = step_to_gyr(step)

    chi = sim.get_chi()
    N = chi.shape[0]
    chi_slice = chi[:, :, N // 2]  # z-midplane

    m = sim.metrics()

    fname = OUTPUT_DIR / f"snapshot_step{step:07d}_{t_gyr:.1f}Gyr.npz"
    np.savez_compressed(
        fname,
        chi_slice=chi_slice,
        step=step,
        cosmic_time_gyr=t_gyr,
        chi_min=m["chi_min"],
        chi_max=m["chi_max"],
        well_fraction=m["well_fraction"],
        void_fraction=m["void_fraction"],
        n_clusters=m["n_clusters"],
    )
    print(f"  >> Snapshot saved: {fname.name}")


# ═══════════════════════════════════════════════════════════════════
# Main simulation
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("LFM Cosmic Structure Formation Simulation")
    print("=" * 70)
    print(f"  Grid:       {GRID_SIZE}³ = {GRID_SIZE**3:,.0f} cells")
    print(f"  Steps:      {TOTAL_STEPS:,d}")
    print(f"  Cosmic time: 0 → {step_to_gyr(TOTAL_STEPS):.1f} Gyr")
    print(f"  Present epoch (13.8 Gyr): step ≈ {PRESENT_STEP:,d}")
    print(f"  Physics:    GOV-01 (wave) + GOV-02 (gravity-only)")
    print(f"  χ₀ = {CHI0}, κ = {KAPPA:.6f}")
    print(f"  GPU: {'available' if lfm.gpu_available() else 'not available (using CPU)'}")
    print("=" * 70)

    # ── 1. Configure ──────────────────────────────────────────────

    config = SimulationConfig(
        grid_size=GRID_SIZE,
        field_level=lfm.FieldLevel.REAL,  # Level 0: gravity only
        boundary_type=lfm.BoundaryType.FROZEN,  # χ = 19 at boundaries
        lambda_self=0.0,  # Gravity-only (Mexican hat negligible at cosmological scales)
        dt=DT,
        e_amplitude=AMPLITUDE,
        blob_sigma_factor=SIGMA_FACTOR,
        report_interval=REPORT_EVERY,
        random_seed=42,
    )

    # ── 2. Initialize ─────────────────────────────────────────────

    print("\n[1/3] Initializing fields...")
    sim = Simulation(config)

    # Random matter fluctuations (like primordial density perturbations)
    rng = np.random.default_rng(config.random_seed)
    N = config.grid_size
    noise = AMPLITUDE * rng.standard_normal((N, N, N)).astype(np.float32)
    sim.set_psi_real(noise)

    # Poisson-equilibrate χ from initial |Ψ|² distribution.
    # This gives self-consistent gravitational wells from step 0
    # (quasi-static limit of GOV-02: ∇²δχ = κ·|Ψ|²).
    sim.equilibrate()

    chi0 = sim.get_chi()
    print(f"  Initial χ range: [{chi0.min():.2f}, {chi0.max():.2f}]")
    print(f"  Initial |Ψ|² total: {np.sum(noise**2):.2e}")

    # Record E₀² = mean interior |Ψ|² so GOV-02 has the right vacuum level.
    # (Without this, the entire grid collapses.)
    mask = sim.get_interior_mask()
    e0_sq = float(np.mean(noise[mask] ** 2))
    config.e0_sq = e0_sq
    # Rebuild simulation with updated E₀²
    sim = Simulation(config)
    sim.set_psi_real(noise)
    sim.equilibrate()
    print(f"  E₀² = {e0_sq:.6f} (vacuum energy density)")

    # ── 3. Evolve ─────────────────────────────────────────────────

    print(f"\n[2/3] Evolving for {TOTAL_STEPS:,d} steps...")
    print(f"  (Reporting every {REPORT_EVERY:,d} steps)")
    print()

    t_start = time.perf_counter()
    steps_done = 0
    snapshot_counter = 0

    # Save initial snapshot
    save_snapshot(sim, 0)

    # Evolve in chunks so we can save snapshots at intervals
    while steps_done < TOTAL_STEPS:
        chunk = min(REPORT_EVERY, TOTAL_STEPS - steps_done)
        sim.run(chunk, callback=on_report, record_metrics=True)
        steps_done += chunk

        # Save periodic snapshots
        if steps_done % SNAPSHOT_EVERY == 0:
            save_snapshot(sim, steps_done)
            snapshot_counter += 1

    elapsed = time.perf_counter() - t_start
    rate = TOTAL_STEPS / elapsed if elapsed > 0 else 0

    # ── 4. Results ────────────────────────────────────────────────

    print()
    print("=" * 70)
    print("[3/3] SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Total steps:   {TOTAL_STEPS:,d}")
    print(f"  Wall time:     {elapsed:.1f} s ({elapsed/3600:.2f} hrs)")
    print(f"  Performance:   {rate:.1f} steps/sec")
    print(f"  Snapshots:     {snapshot_counter + 1} saved to {OUTPUT_DIR}/")

    # Final metrics
    m = sim.metrics()
    print()
    print("  Final state:")
    print(f"    χ_min        = {m['chi_min']:.2f}")
    print(f"    χ_max        = {m['chi_max']:.2f}")
    print(f"    Wells (χ<17) = {m['well_fraction']*100:.1f}%")
    print(f"    Voids (χ>18) = {m['void_fraction']*100:.1f}%")
    print(f"    Clusters     = {m['n_clusters']}")
    print(f"    Energy       = {m['energy_total']:.4e}")

    # Print cosmic milestones from history
    if sim.history:
        print()
        print("  Key cosmic milestones:")
        for snap in sim.history:
            t = step_to_gyr(int(snap["step"]))
            if t < 0.5 or abs(t - 3.0) < 0.5 or abs(t - 13.8) < 0.5 or abs(t - 20) < 0.5 or abs(t - 30) < 0.5:
                marker = " ★" if abs(t - 13.8) < 0.5 else ""
                print(
                    f"    t={t:5.1f} Gyr | "
                    f"wells={snap['well_fraction']*100:5.1f}% | "
                    f"voids={snap['void_fraction']*100:5.1f}% | "
                    f"χ_min={snap['chi_min']:6.2f}{marker}"
                )

    print()
    print("Done! Use the .npz snapshots to visualize structure evolution.")


if __name__ == "__main__":
    main()
