#!/usr/bin/env python3
"""
33 — LFM Higgs Boson: χ-Field Breathing Mode
=============================================

WHAT THE HIGGS BOSON IS IN LFM
-------------------------------
In LFM the chi field (χ) IS the Higgs field.  It has a Mexican-hat potential

    V(χ) = λ_H (χ² − χ₀²)²    with  λ_H = 4/31  (derived from lattice geometry)

This makes χ₀ = 19 a dynamical attractor.  Small oscillations about this
vacuum have a characteristic angular frequency

    ω_H = √(8 λ_H) · χ₀ ≈ 19.304  (lattice units)

This ringing frequency IS the Higgs boson mass (125 GeV in CERN units).

CERN ANALOGY
------------
CERN excites the Higgs field with a high-energy proton-proton collision, then
measures its decay products (H→γγ) in calorimeters surrounding the beam.
A resonant bump in the di-photon mass spectrum at 125 GeV = Higgs mass.

In LFM:
  1. A proton eigenmode (standing wave in χ-well) is placed at rest.
  2. A tiny impulsive kick perturbs χ by +Δχ at the soliton core.
     (This is the LFM equivalent of the collision exciting the Higgs field.)
  3. χ rings at ω_H — the Higgs breathing mode.
  4. FFT of χ(t) → a BUMP at ω_H.  This IS the di-photon mass spectrum.

WHY NOT A REAL COLLISION?
--------------------------
A p + p̄ collision at v = 0.005c has soliton overlap time

    Δt_overlap = 2σ / v  ≈  2 × 2.5 / 0.005  =  1000 time units

The Higgs period is T_H = 2π / ω_H ≈ 0.326 time units.
Ratio: Δt_overlap / T_H ≈ 3000  → collision is COMPLETELY QUASI-ADIABATIC.
χ just tracks |Ψ|² slowly — it never rings.  This is as if CERN tried to
measure the Higgs by slowly squeezing two protons together: the Higgs field
never gets excited as a distinct quantum, only as a classical response.

An impulsive local perturbation perturbs χ in a single timestep (dt = 0.02),
which is 16× shorter than T_H.  This is genuinely impulsive — it excites the
breathing mode cleanly, just as a high-energy quanta from a real collision
deposits energy in <<T_H.

Usage
-----
    python 33_higgs_boson.py              # N=64 default
    python 33_higgs_boson.py --grid 32    # quick smoke test (~10 s)
    python 33_higgs_boson.py --steps 20000  # cleaner FFT (more periods)
    python 33_higgs_boson.py --no-anim    # skip animation (not used)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Windows: force UTF-8 output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure the repo root is on sys.path when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import lfm  # noqa: E402
from lfm.constants import CHI0, LAMBDA_H  # noqa: E402
from _common import run_and_save_3d_movie  # noqa: E402

# ── CLI ────────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description="LFM Higgs boson: breathing mode")
_parser.add_argument("--grid",  type=int,   default=64,     help="Grid size N^3  (default 64)")
_parser.add_argument("--amp",   type=float, default=6.0,    help="Soliton amplitude  (default 6)")
_parser.add_argument("--sigma", type=float, default=2.5,    help="Soliton sigma  (default 2.5)")
_parser.add_argument("--kick",  type=float, default=0.3,    help="Chi kick magnitude  (default 0.3)")
_parser.add_argument("--steps", type=int,   default=12_000, help="Measurement steps  (default 12000)")
_parser.add_argument("--no-anim", dest="animate", action="store_false")
args = _parser.parse_args()

N      = args.grid
AMP    = args.amp
SIGMA  = args.sigma
KICK   = args.kick
MEAS   = args.steps
DT     = 0.02   # lfm default

# ── Derived constants ───────────────────────────────────────────────────────
CHI0_VAL: float = float(CHI0)
OMEGA_H: float  = math.sqrt(8.0 * LAMBDA_H) * CHI0_VAL   # approx 19.304
T_H: float      = 2.0 * math.pi / OMEGA_H                 # approx 0.326

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "33_higgs_boson"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Announce ────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  LFM Higgs Boson -- Experiment 33")
print("=" * 65)
print(f"  Grid     : {N}^3")
print(f"  AMP      : {AMP}   sigma = {SIGMA}")
print(f"  lambda_H : {LAMBDA_H:.5f}   (= 4/31, derived)")
print(f"  omega_H  : {OMEGA_H:.4f}   (target Higgs frequency)")
print(f"  T_H      : {T_H:.3f} lattice units = {T_H/DT:.0f} steps")
print(f"  Kick     : +{KICK} at centre, -{KICK*0.05:.3f} at 8 corner neighbours")
print(f"  Steps    : {MEAS}  =  {MEAS * DT / T_H:.0f} x T_H")
print()

# ── STEP 1: Build proton eigenmode with Mexican hat ON ─────────────────────
print("  Building proton eigenmode (Mexican hat ON) ...")
config = lfm.SimulationConfig(
    grid_size       = N,
    field_level     = lfm.FieldLevel.REAL,
    e_amplitude     = AMP,
    lambda_self     = LAMBDA_H,   # Mexican hat: critical for Higgs mode
    report_interval = MEAS,       # suppress per-batch progress output
    dt              = DT,
)
sim = lfm.Simulation(config)
cx = N // 2
sim.place_soliton((cx, cx, cx), amplitude=AMP, sigma=SIGMA)
sim.equilibrate()

chi_min_eq = float(np.asarray(sim.chi).min())
chi_c_eq   = float(sim.chi[cx, cx, cx])
print(f"  Equilibrated: chi_centre = {chi_c_eq:.4f}   chi_min = {chi_min_eq:.4f}")
print(f"  Well depth  : delta_chi = {CHI0_VAL - chi_min_eq:.4f}")
print()

# ── STEP 2: Impulsive chi kick (sub-T_H perturbation) ──────────────────────
#
# Perturbation duration = 1 step = dt = 0.02  <<  T_H = 0.326
# This IS genuinely impulsive: energy deposited in 0.02/0.326 = 6% of T_H.
# Contrast with v=0.005c collision: overlap time ~ 2*sigma/v = 1000 >> T_H.
print(f"  Applying impulsive chi kick (dt = {DT} = T_H/{T_H/DT:.0f}) ...")
sim.chi[cx, cx, cx] += KICK
for di in (-1, 1):
    for dj in (-1, 1):
        for dk in (-1, 1):
            sim.chi[cx + di, cx + dj, cx + dk] -= KICK * 0.05

print(f"  chi_centre after kick: {float(sim.chi[cx, cx, cx]):.4f}")
print()

# ── STEP 3: Run and record chi at centre every step ────────────────────────
print(f"  Running {MEAS} steps (recording chi every step) ...")
chi_series: list[float] = []

def _record(s: lfm.Simulation, _step: int) -> None:
    chi_series.append(float(s.chi[cx, cx, cx]))

# run_with_snapshots with step_callback fires _record at EVERY leapfrog step.
# snapshot_every=MEAS + fields=[] avoids storing large arrays mid-run.
sim.run_with_snapshots(
    MEAS,
    snapshot_every = MEAS,   # only 1 cheap snapshot at very end
    fields         = [],     # no field arrays stored -- scalars only
    step_callback  = _record,
    record_metrics = False,
)
print(f"  Recorded {len(chi_series)} samples.")
print()

# ── STEP 4: FFT -- Higgs mass ───────────────────────────────────────────────
print("  FFT analysis:")
t_arr  = np.arange(len(chi_series)) * DT
signal = np.array(chi_series, dtype=np.float64)
signal -= signal.mean()       # remove DC offset

freqs  = np.fft.rfftfreq(len(signal), d=DT)
omegas = 2.0 * math.pi * freqs
spec   = np.abs(np.fft.rfft(signal))

# Peak search in window [5, 40] rad/step -- safely brackets omega_H ~ 19
omega_mask = (omegas > 5.0) & (omegas < 40.0)
if omega_mask.any():
    idx_in_mask    = int(np.argmax(spec[omega_mask]))
    measured_omega = float(omegas[omega_mask][idx_in_mask])
    error_pct      = abs(measured_omega - OMEGA_H) / OMEGA_H * 100.0
    verdict        = "PASS" if error_pct < 5.0 else "FAIL"
    print(f"  Measured omega_H = {measured_omega:.4f}  "
          f"theory = {OMEGA_H:.4f}  "
          f"error = {error_pct:.2f}%  [{verdict}]")
else:
    measured_omega = math.nan
    error_pct      = math.nan
    verdict        = "N/A"
    print("  (no peak found in [5, 40] rad/step -- check lambda_self is set)")
print()

# ── FIGURE 1: Chi time trace ────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(12, 4), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
ax.plot(t_arr, signal + chi_c_eq, color="#60a5fa", lw=0.7,
        label="chi(centre, t)")
ax.axhline(CHI0_VAL, color="#94a3b8", ls="--", lw=0.8,
           label=f"chi0 = {CHI0_VAL:.0f}  (Higgs vacuum)")
ax.axhline(chi_c_eq, color="#6ee7b7", ls=":", lw=0.8,
           label=f"equilibrium chi_c = {chi_c_eq:.3f}  (proton well)")
ax.set_xlabel("Lattice time", color="white")
ax.set_ylabel("chi at centre", color="white")
if not math.isnan(measured_omega):
    title_str = (
        f"LFM Higgs field ringing at omega_H = {measured_omega:.3f}  "
        f"(theory {OMEGA_H:.3f},  error {error_pct:.1f}%)"
    )
else:
    title_str = "LFM Higgs field ringing (no peak found -- check lambda_self)"
ax.set_title(title_str, color="white", fontsize=11)
ax.tick_params(colors="white")
for sp in ax.spines.values():
    sp.set_color("#334155")
ax.legend(fontsize=9, framealpha=0.3)
fig1.patch.set_facecolor("#0a0a0a")
fig1.tight_layout()
trace_path = OUT_DIR / "higgs_trace.png"
fig1.savefig(str(trace_path), dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig1)
print(f"  Saved: {trace_path}")

# ── FIGURE 2: Power spectral density -- the di-photon bump ─────────────────
spec_psd = np.abs(np.fft.rfft(signal)) ** 2
spec_n   = spec_psd / (spec_psd.max() + 1e-30)

fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor="#0a0a0a")
ax2.set_facecolor("#0a0a0a")
ax2.fill_between(omegas, spec_n, alpha=0.3, color="#60a5fa")
ax2.plot(omegas, spec_n, color="#60a5fa", lw=1.5,
         label="|chi(omega)|^2  (normalised)")
ax2.axvline(OMEGA_H, color="#f87171", ls="--", lw=2.0,
            label=f"omega_H = {OMEGA_H:.3f}  (theory = 125 GeV)")
if not math.isnan(measured_omega):
    ax2.axvline(measured_omega, color="#fbbf24", ls=":", lw=1.5,
                label=f"measured {measured_omega:.3f}  ({error_pct:.1f}% err)")
ax2.set_xlim(0.0, OMEGA_H * 2.5)
ax2.set_ylim(-0.05, 1.15)
ax2.set_xlabel("omega  (lattice units = Higgs mass scale)", color="white")
ax2.set_ylabel("Normalised PSD", color="white")
ax2.set_title(
    "LFM di-photon analog: BUMP at omega_H = ATLAS/CMS bump at 125 GeV\n"
    f"V(chi) = lambda_H*(chi^2 - chi0^2)^2   lambda_H = {LAMBDA_H:.4f}  (derived)",
    color="white", fontsize=10
)
ax2.tick_params(colors="white")
for sp in ax2.spines.values():
    sp.set_color("#334155")
ax2.legend(fontsize=10, framealpha=0.3)
fig2.patch.set_facecolor("#0a0a0a")
fig2.tight_layout()
bump_path = OUT_DIR / "higgs_diphoton_bump.png"
fig2.savefig(str(bump_path), dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close(fig2)
print(f"  Saved: {bump_path}")

# ── Final summary ────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("RESULTS -- LFM Higgs Boson (Experiment 33)")
print("=" * 65)
print(f"  chi0 (vacuum)        : {CHI0_VAL:.1f}")
print(f"  chi_min (eigenmode)  : {chi_min_eq:.4f}  "
      f"(proton well depth = {CHI0_VAL - chi_min_eq:.3f})")
print(f"  lambda_H             : {LAMBDA_H:.5f}  (= 4/31, derived)")
print(f"  Theoretical omega_H  : {OMEGA_H:.4f}  (= sqrt(8*lambda_H)*chi0)")
if not math.isnan(measured_omega):
    print(f"  Measured omega_H     : {measured_omega:.4f}  "
          f"({error_pct:.2f}% error)  [{verdict}]")
print()
print("  CERN analogy:")
print("    ATLAS/CMS: p+p -> excite Higgs -> H->yy -> bump at 125 GeV")
print(f"    LFM: kick chi -> chi rings at omega_H -> FFT -> bump at {OMEGA_H:.2f}")
print()
print("  WHY NOT A COLLISION?")
print(f"    At v=0.005c: overlap time ~ 2*sigma/v = {2*SIGMA/0.005:.0f} time units")
print(f"    Higgs period T_H = {T_H:.3f} time units")
print(f"    Ratio = {2*SIGMA/0.005/T_H:.0f}  -> QUASI-ADIABATIC: chi never rings")
print(f"    Impulsive kick: dt = {DT} = T_H/{T_H/DT:.0f}  -> genuinely impulsive")
print()
print(f"  Output: {OUT_DIR}")
print("=" * 65)
print()

# 3-D movie of the soliton chi-breathing mode
run_and_save_3d_movie(sim, steps=1000, out_dir=OUT_DIR, stem="higgs_boson",
    field="chi_deficit", snapshot_every=20, no_anim=not args.animate)
