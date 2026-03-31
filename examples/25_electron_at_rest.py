"""25 — Electron at Rest

An electron is a standing wave that traps itself in its own chi-well.
This experiment:
  1. Solves for the self-consistent eigenmode (SCF solver).
  2. Verifies stability over 500 steps.
  3. Runs 2000 more steps recording psi_real every 4 steps.
  4. Saves a 3-panel slice movie (xy / xz / yz) showing the
     wave-function oscillating at frequency ω ≈ chi_local ≈ 19.

WHAT YOU SEE IN THE MOVIE
--------------------------
  psi_real(x, y, z, t) — a 2-D slice through the centre.
  The central Gaussian blob oscillates: positive → zero → negative →
  zero → positive.  One full oscillation ≈ 16 animation frames.

  The chi-well (gravity dip) is only ~0.01 units deep — the electron
  is the most lightly-bound soliton.  Compare: a proton sits in a
  well ~1.6 units deep.

Usage
-----
    python 25_electron_at_rest.py              # default N=48, animated MP4
    python 25_electron_at_rest.py --no-anim    # skip movie (fast)
    python 25_electron_at_rest.py --grid 32    # smaller/faster grid
    python 25_electron_at_rest.py --steps 4000 # longer movie

Next: 26_electron_traverse.py — electron moving through the lattice.
"""

from __future__ import annotations

import argparse

import numpy as np

import lfm
from _common import make_out_dir, run_and_save_3d_movie

# ── CLI ──────────────────────────────────────────────────────────────────────

_p = argparse.ArgumentParser(description="25 — Electron at rest oscillation")
_p.add_argument("--grid",    type=int,   default=128,  help="Grid size N (default 128)")
_p.add_argument("--steps",   type=int,   default=2000, help="Movie-run steps (default 2000)")
_p.add_argument("--no-anim", action="store_true",      help="Skip 3-D movie")
args = _p.parse_args()

N          = args.grid
MOVIE_STEPS = args.steps
NO_ANIM    = args.no_anim
# Period of psi_real oscillation ≈ 2π / chi_local ≈ 16 steps at dt=0.02.
# Capture every 4 steps → 4 frames per period → smooth animation.
SNAP_EVERY = 4

OUT_DIR = make_out_dir("25_electron_at_rest")

# ── STEP 1: eigenmode solve ───────────────────────────────────────────────────

print("25 — Electron at Rest")
print("=" * 55)
print(f"  Grid       : {N}³")
print(f"  Movie steps: {MOVIE_STEPS}  (snapshot every {SNAP_EVERY})")
print(f"  Animate    : {not NO_ANIM}  (MP4, zoomed to centre 24×24 cells)")
print()

print("Solving electron eigenmode (GOV-01 + GOV-02 SCF relaxation)…")
placed = lfm.create_particle("electron", N=N, use_eigenmode=True)
sim = placed.sim
sim.equilibrate()  # mark chi consistent; prevents auto-re-equilibrate on first run()

m0 = sim.metrics()
center0 = lfm.measure_center_of_energy(sim)

print(f"\nAfter eigenmode solve:")
print(f"  chi_min  = {m0['chi_min']:.4f}  (vacuum = 19.0, dip = {19.0 - m0['chi_min']:.4f})")
print(f"  energy   = {m0['energy_total']:.4e}")
print(f"  position = ({center0[0]:.1f}, {center0[1]:.1f}, {center0[2]:.1f})")

# ── STEP 2: stability check ───────────────────────────────────────────────────

print("\nStability check (500 steps)…")
sim.run(steps=500)
m1 = sim.metrics()
center1 = lfm.measure_center_of_energy(sim)

drift  = float(np.linalg.norm(center1 - center0))
# Note: energy_total oscillates for complex fields (phase rotation) so
# we only check position stability here.
stable = drift < 3.0

print(f"  position drift = {drift:.3f} cells   (target < 3.0)  {'PASS' if stable else 'FAIL'}")
print(f"  chi_min        = {m1['chi_min']:.4f}")
print(f"  Eigenmode stable: {stable}")

# ── STEP 3: movie run ─────────────────────────────────────────────────────────

print(f"\nRunning {MOVIE_STEPS} steps for 3-D movie (psi_real every {SNAP_EVERY} steps)…")
snaps, movie_path = run_and_save_3d_movie(
    sim,
    steps          = MOVIE_STEPS,
    out_dir        = OUT_DIR,
    stem           = "electron_at_rest",
    field          = "psi_real",     # renders as |Ψ| amplitude — no blinking
    snapshot_every = SNAP_EVERY,
    fps            = 20,
    no_anim        = NO_ANIM,
    intensity_floor = 0.001,         # shallow well → lower threshold
    camera_rotate  = False,
    title          = "Electron at Rest — Wave Amplitude |Ψ|",
    crop_radius    = N // 3,         # zoom: electron fills the view
)

# ── Summary ───────────────────────────────────────────────────────────────────

print()
print("=" * 55)
print("The electron soliton sits in its own chi-well and oscillates.")
print(f"  chi-well depth  = {19.0 - m0['chi_min']:.4f} lattice units")
print(f"  wave frequency  ≈ {m0['chi_min']:.3f} rad/step  (= local chi)")
osc_period_steps = round(6.2832 / m0["chi_min"] / sim.config.dt)
print(f"  oscillation period ≈ {osc_period_steps} steps  ({osc_period_steps * SNAP_EVERY / osc_period_steps:.0f} frames/period in movie)")
if movie_path is not None:
    print(f"  Movie            → {movie_path}")
print()
print("Next: 26_electron_traverse.py — electron moving through the lattice.")
