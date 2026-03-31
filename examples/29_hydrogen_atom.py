"""29 — Hydrogen Atom

Build an LFM hydrogen atom: a nuclear chi-well (proton potential) with an
electron eigenmode solved inside it using GOV-01 only.

This is Approach A from the Phase 4 project plan: the proton is too heavy to
simulate on the same grid as the electron (mass ratio ~1836:1), so we replace
it with an analytic Gaussian chi-well.  The electron's wave function is found
by running GOV-01 for 10 000 steps with chi frozen at the nuclear potential.
Unbound radiation radiates away; the residual psi is the bound s-state mode.
"""

import lfm
from _common import make_out_dir, parse_no_anim, run_and_save_3d_movie

_args = parse_no_anim()
_OUT  = make_out_dir("29_hydrogen_atom")

print("29 — LFM Hydrogen Atom")
print("=" * 55)
print()

print("Solving H atom (N=64, 10 000 GOV-01 steps)...")
print("  Nuclear well: depth=14 (chi_min=5), sigma=3 cells")
print()

atom = lfm.create_atom("H", N=64, steps=10_000)

print("Results:")
print(f"  Nuclear chi_min   = {atom.chi_nuclear.min():.2f}  (background chi0 = {lfm.CHI0})")
print(f"  Fraction near nucleus = {atom.fraction_near_nucleus:.3f}  (target >= 0.40)")
print(f"  Binding proxy     = {atom.binding_energy:.3f}  (positive = more confinement than free)")
print(f"  Bound             = {atom.bound}")
print()

if atom.bound:
    print("SUCCESS: Electron is confined to the nuclear chi-well.")
    print()
    print("Physical picture:")
    print("  1. Proton creates chi-well: chi drops from 19 -> 5 at centre")
    print("  2. GOV-01 modes with 5 < omega < 19 are evanescent at chi=19 edges")
    print("  3. Only trapped modes survive -> bound s-state analogue")
    print()
    print("This is the LFM equivalent of solving the hydrogen Schrodinger equation.")
    print("No external potential or QM rules were injected - binding emerged from")
    print("the wave equation with a spatially-varying chi field.")
else:
    print("WARNING: Electron not bound.  Try larger depth or more steps.")

print()
print("See also: 30_hydrogen_molecule.py")

# 3-D movie using the solved atom's simulation (electron psi in nuclear chi-well)
run_and_save_3d_movie(atom.sim, steps=500, out_dir=_OUT, stem="hydrogen_atom",
    field="psi_real", snapshot_every=10, intensity_floor=0.001, no_anim=_args.no_anim)
