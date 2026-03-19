#!/usr/bin/env python3
"""
Predict 35+ Physics Constants from One Integer
================================================

The LFM framework derives all fundamental constants from χ₀ = 19
(the number of non-propagating modes on a 3D cubic lattice:
1 center + 6 faces + 12 edges = 19).

This example prints every prediction, its measured value, and the
percentage error. No simulation needed — pure algebra from one integer.

Usage:
  python predict_constants.py
"""

from __future__ import annotations

import lfm


def main() -> None:
    print("LFM Predictions from χ₀ = 19")
    print("=" * 72)
    print()

    catalog = lfm.predict_all()

    # Group by category
    groups = {
        "Couplings": ["alpha_em", "alpha_s", "sin2_theta_w", "lambda_Higgs"],
        "Lepton masses": ["m_mu/m_e", "m_tau/m_e", "m_tau/m_mu"],
        "Hadron masses": ["m_p/m_e", "m_p/m_e_3gen"],
        "Electroweak bosons": ["m_W/m_e", "m_Z/m_W", "m_H/m_W"],
        "Quark mass ratios": ["m_s/m_d", "m_c/m_s", "m_t/m_c", "m_t/m_b"],
        "CKM matrix": ["sin_theta_C", "CKM_A", "CKM_delta", "|V_ub|"],
        "PMNS matrix": [
            "sin2_theta12", "sin2_theta23", "sin2_theta13", "delta_CP_nu",
        ],
        "Anomalous moment": ["muon_g2"],
        "Cosmology": [
            "Omega_Lambda", "Omega_matter", "N_efoldings", "z_rec",
        ],
        "Particle structure": [
            "N_generations", "N_gluons", "beta0_QCD",
            "D_spacetime", "N_string_dim", "N_mtheory_dim",
        ],
    }

    total = 0
    within_2pct = 0

    for group_name, keys in groups.items():
        print(f"  {group_name}")
        print(f"  {'-' * 68}")
        for key in keys:
            if key not in catalog:
                continue
            e = catalog[key]
            total += 1
            if e["error_pct"] <= 2.0:
                within_2pct += 1
            marker = "✓" if e["error_pct"] <= 2.0 else "~"
            print(
                f"  {marker} {key:<20s}  "
                f"pred={e['predicted']:>14.6g}  "
                f"meas={e['measured']:>14.6g}  "
                f"err={e['error_pct']:>6.2f}%  "
                f"  {e['formula']}"
            )
        print()

    print("=" * 72)
    print(f"Total: {total} predictions, "
          f"{within_2pct} within 2% ({within_2pct/total*100:.0f}%)")
    print(f"\nAll from one integer: χ₀ = {lfm.CHI0:.0f}")


if __name__ == "__main__":
    main()
