#!/usr/bin/env python3
"""
Particle Mass Spectrum
======================

Compute all particle masses from the l(l+1) angular momentum formula.
Every mass ratio m/m_e is predicted from χ₀ = 19 with algebraic offsets.

  Leptons:  l = gen·χ₀ + offset
  Quarks:   l = gen·χ₀ + offset  (separate up-type / down-type formulas)
  Proton:   l = 2χ₀ + 4 = 42  (or 3-generation sum: 11³ + 19² + 12²)

Usage:
  python particle_masses.py
"""

from __future__ import annotations

import lfm
from lfm.formulas.masses import (
    down_quark_l,
    down_quark_mass_ratio,
    lepton_l,
    lepton_mass_ratio,
    proton_l,
    proton_mass_ratio,
    proton_mass_ratio_3gen,
    up_quark_l,
    up_quark_mass_ratio,
)


def main() -> None:
    print("LFM Particle Mass Spectrum  (m/m_e from l(l+1), χ₀ = 19)")
    print("=" * 65)

    # Full mass table from the library
    table = lfm.mass_table()

    print(f"\n{'Particle':<12s} {'l':>5s} {'Predicted':>12s} "
          f"{'Measured':>12s} {'Error':>8s}")
    print("-" * 55)
    for row in table:
        l_str = f"{row['l']:.0f}" if row["l"] is not None else "—"
        print(f"  {row['particle']:<10s} {l_str:>5s} "
              f"{row['predicted']:>12.1f} {row['measured']:>12.1f} "
              f"{row['error_pct']:>7.2f}%")

    # Highlight the proton 3-generation decomposition
    print()
    print("Proton 3-generation decomposition:")
    chi0 = lfm.CHI0
    t1 = (chi0 - 8) ** 3
    t2 = chi0 ** 2
    t3 = (chi0 - 7) ** 2
    print(f"  (χ₀−8)³ + χ₀² + (χ₀−7)² = {t1:.0f} + {t2:.0f} + {t3:.0f} "
          f"= {t1 + t2 + t3:.0f}")
    print(f"  l(l+1) with l=42:  {proton_mass_ratio():.0f}")
    print(f"  Measured:          1836.15")

    # Show the l quantum numbers for each generation
    print()
    print("Angular momentum quantum numbers:")
    names = {
        "Leptons": [("e", 1), ("μ", 2), ("τ", 3)],
        "Up quarks": [("u", 1), ("c", 2), ("t", 3)],
        "Down quarks": [("d", 1), ("s", 2), ("b", 3)],
    }
    for group, particles in names.items():
        vals = []
        for name, gen in particles:
            if group == "Leptons":
                l = lepton_l(gen)
            elif group == "Up quarks":
                l = up_quark_l(gen)
            else:
                l = down_quark_l(gen)
            vals.append(f"{name}: l={l:.0f}")
        print(f"  {group:<14s}  {', '.join(vals)}")


if __name__ == "__main__":
    main()
