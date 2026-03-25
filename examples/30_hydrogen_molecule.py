"""30 — Hydrogen Molecule (H2)

Place two proton chi-wells side by side on the grid and solve the shared
electron eigenmode.  The electron delocalises across both nuclear sites —
covalent bonding from nothing but GOV-01.

Acceptance criterion: psi^2 >= 2% near EACH proton after 10 000 steps.
"""

import lfm

print("30 — LFM Hydrogen Molecule (H2)")
print("=" * 55)
print()

BOND_LENGTH = 16.0   # proton separation in cells

print(f"Solving H2 molecule (N=64, bond_length={BOND_LENGTH:.0f} cells, 10 000 steps)...")
print()

mol = lfm.create_molecule("H2", N=64, bond_length=BOND_LENGTH, steps=10_000)

half = 64 // 2
x1 = half - BOND_LENGTH / 2
x2 = half + BOND_LENGTH / 2

print("Results:")
print(f"  Proton separation    = {mol.proton_separation:.1f} cells")
print(f"  Molecular chi_min    = {mol.chi_nuclear.min():.2f}")
print(f"  Electron energy (mol)= {mol.electron_energy:.1f}")
print(f"  Bond stable          = {mol.bond_stable}")
print()

if mol.bond_stable:
    print("SUCCESS: Electron density spans both nuclear sites.")
    print()
    print("Physical picture:")
    print(f"  Proton 1 at x = {x1:.1f}  creates chi-well")
    print(f"  Proton 2 at x = {x2:.1f}  creates chi-well")
    print("  Combined potential: chi_mol = chi0 - (chi0-chi1) - (chi0-chi2)")
    print("  Electron placed at midpoint, runs in frozen molecular potential")
    print("  Bound modes of the double-well extend to BOTH proton sites")
    print()
    print("This is covalent bonding from wave-equation interference alone.")
    print("No valence-bond theory or molecular orbital theory was injected.")
else:
    print("Bond not stable — try reducing bond_length or increasing nuclear depth.")

print()
print("Complete! The LFM particle physics ladder:")
print("  atoms -> molecules -> ... matter -> life")
