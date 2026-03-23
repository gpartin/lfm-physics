"""18 - Particle Collision with Energy Accounting

Goal:
  Demonstrate two solitons colliding head-on using the COMPLEX field level,
  with peak tracking, collision-event detection, and energy conservation
  accounting before and after the collision.

  Two solitons are given equal-and-opposite initial velocities along the
  x axis.  We track the energy-density peaks at regular intervals, detect
  the approach event, and compare total energy before and after impact.

Physics:
  - GOV-01 (complex Ψ): ∂²Ψ/∂t² = c²∇²Ψ − χ²Ψ
  - GOV-02: ∂²χ/∂t² = c²∇²χ − κ(|Ψ|² − E₀²)
  - Velocity boost encodes a phase gradient across the soliton:
      Ψ(x) → Ψ(x) · exp(i k⃗·x⃗)  with k = χ₀·v/c
  - At collision the wavefunctions overlap: interference + radiation
  - Energy is conserved in GOV-01+GOV-02 (to <0.1% drift)

Expected output (depends on grid and amplitude):
  Total energy before collision:  E₀ (baseline)
  Total energy after  collision:  E₁ ≈ E₀  (±O(1%) drift — physical)
  Approach event detected at separation < min_sep
  Collision event logged with step and separation
"""

from __future__ import annotations

import numpy as np
import lfm
from lfm.analysis.tracker import detect_collision_events, track_peaks
from lfm.analysis.energy import total_energy

# ── Configuration ─────────────────────────────────────────────────────────────
N = 64
AMP = 6.0
V = 0.10           # initial speed in lattice units (c = 1 → 10% of c)
TRACK_INTERVAL = 100
TOTAL_STEPS = 3_000
MIN_SEP = 5.0      # grid cells — "approach" event threshold

cfg = lfm.SimulationConfig(
    grid_size=N,
    field_level=lfm.FieldLevel.COMPLEX,
    boundary_type=lfm.BoundaryType.FROZEN,
    dt=0.02,
    report_interval=9999,
)

sim = lfm.Simulation(cfg)

print("18 - Particle Collision")
print("=" * 54)
print(f"Grid: {N}³   amplitude: {AMP}   speed: ±{V}c")

# ── Place solitons at ±separation from centre, heading toward each other ─────
cx = cy = cz = N // 2
sep = 16  # initial separation from centre

sim.place_soliton((cx - sep, cy, cz), amplitude=AMP, velocity=( V, 0.0, 0.0))
sim.place_soliton((cx + sep, cy, cz), amplitude=AMP, velocity=(-V, 0.0, 0.0))
sim.equilibrate()

# ── Energy before ─────────────────────────────────────────────────────────────
energy_before = total_energy(
    sim.psi_real, sim.psi_imag,
    sim.chi,
    dx=sim.config.dx, dt=sim.config.dt,
)
print(f"\nEnergy before: {energy_before:.4e}")

# ── Track peaks through the collision ─────────────────────────────────────────
print(f"\nRunning {TOTAL_STEPS:,} steps (tracking peaks every {TRACK_INTERVAL} steps)…")
trajectories = track_peaks(
    sim,
    steps=TOTAL_STEPS,
    interval=TRACK_INTERVAL,
    n_peaks=4,
    min_separation=4,
)

print(f"Snapshots recorded: {len(trajectories)}")
n_peaks_per_snap = [len(s) for s in trajectories]
print(f"Peak count range:   {min(n_peaks_per_snap)} – {max(n_peaks_per_snap)}")

# ── Detect events ─────────────────────────────────────────────────────────────
events = detect_collision_events(trajectories, min_sep=MIN_SEP)
approach_events = [e for e in events if e["type"] == "approach"]
merge_events    = [e for e in events if e["type"] == "merge"]

print(f"\nCollision events detected:")
print(f"  approach events: {len(approach_events)}")
print(f"  merge    events: {len(merge_events)}")

if approach_events:
    first = approach_events[0]
    print(f"\nFirst approach event:")
    print(f"  step = {first['step']:.0f},  sep = {first['sep']:.2f} cells")
    print(f"  peaks i={first['i']}, j={first['j']}")

if merge_events:
    m = merge_events[0]
    print(f"\nFirst merge event:  step = {m['step']:.0f}")

# ── Energy after ──────────────────────────────────────────────────────────────
energy_after = total_energy(
    sim.psi_real, sim.psi_imag,
    sim.chi,
    dx=sim.config.dx, dt=sim.config.dt,
)
drift_pct = 100.0 * abs(energy_after - energy_before) / (abs(energy_before) + 1e-30)
print(f"\nEnergy after:  {energy_after:.4e}")
print(f"Energy drift:  {drift_pct:.2f}%  (should be < ~5%)")

# ── Compute impact parameter using the first two flattened tracks ─────────────
if len(trajectories[0]) >= 2:
    from lfm.analysis.tracker import compute_impact_parameter, flatten_trajectories

    flat = flatten_trajectories(trajectories)
    # Split flat dict into two per-track dicts (track i=0 and i=1)
    # The flattened dict contains all peaks; we filter by initial x position
    # to get the two tracks separately.
    all_x = flat["x"]
    split = np.median(all_x)
    mask_left  = flat["x"] < split
    mask_right = flat["x"] >= split

    def _sub(d: dict, mask: np.ndarray) -> dict:
        return {k: d[k][mask] for k in d}

    traj_left  = _sub(flat, mask_left)
    traj_right = _sub(flat, mask_right)

    if len(traj_left["x"]) >= 2 and len(traj_right["x"]) >= 2:
        b = compute_impact_parameter(traj_left, traj_right)
        print(f"\nImpact parameter:  {b:.2f} cells  (head-on → b ≈ 0)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 54)
print("SUMMARY")
print(f"  Energy conserved:   {'YES (<5%)' if drift_pct < 5.0 else f'DRIFT={drift_pct:.1f}%'}")
print(f"  Approach detected:   {'YES' if approach_events else 'NO'}")
print(f"  Merge detected:      {'YES' if merge_events else 'NO'}")
print()
print("Interpretation:")
print("  Solitons approach, their wavefunctions overlap (constructive or")
print("  destructive, depending on phases), and radiation is emitted.  The")
print("  GOV-01+GOV-02 system conserves total field energy.")
