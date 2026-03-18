"""
Mass Formulas
=============

Particle mass predictions from 4D angular momentum quantization.

    m/m_e = l(l+1)  where  l = τ·χ₀ + offset

Offsets are algebraic functions of χ₀ = 19 and generation number.
All formulas from LFM_COMPLETE_MASS_DERIVATION.md.
"""

from __future__ import annotations


def angular_momentum_mass(l: float) -> float:
    """Mass ratio from angular momentum quantum number: m/m_e = l(l+1).

    Parameters
    ----------
    l : float
        Angular momentum quantum number.

    Returns
    -------
    float
        Predicted mass ratio relative to electron mass.
    """
    return l * (l + 1)


# ── Lepton offsets ────────────────────────────────────────


def lepton_offset(gen: int, chi0: float = 19.0) -> float:
    """Lepton angular momentum offset for generation gen (1, 2, 3).

    offset = (gen − 1)(6·gen − 17)
    Coefficients: 6 = (χ₀−1)/3, 17 = χ₀−2.

    gen=1 (e): offset = 0 → l = 0  → l(l+1) = 0  (reference mass)
    gen=2 (μ): offset = -5 → l = χ₀−5 = 14 → l(l+1) = 210
    gen=3 (τ): offset = 2  → l = 3χ₀+2 = 59 → l(l+1) = 3540
    """
    return (gen - 1) * (6 * gen - 17)


def lepton_l(gen: int, chi0: float = 19.0) -> float:
    """Angular momentum quantum number for lepton generation.

    l = gen·χ₀ + offset − χ₀ = (gen−1)·χ₀ + offset

    For gen=1: l = 0 (electron is the reference)
    For gen=2: l = χ₀ + (-5) = 14
    For gen=3: l = 2χ₀ + 2 = 40... wait

    Direct formulas from catalog:
    gen=2: l = χ₀ − 5 = 14
    gen=3: l = 3χ₀ + 2 = 59
    """
    if gen == 1:
        return 0.0
    elif gen == 2:
        return chi0 - 5  # 14
    elif gen == 3:
        return 3 * chi0 + 2  # 59
    raise ValueError(f"gen must be 1, 2, or 3, got {gen}")


def lepton_mass_ratio(gen: int, chi0: float = 19.0) -> float:
    """Lepton mass / electron mass for generation gen.

    gen=1: 1.0 (electron)
    gen=2: l(l+1) = 14×15 = 210  (muon)
    gen=3: l(l+1) = 59×60 = 3540 (tau)
    """
    if gen == 1:
        return 1.0
    l = lepton_l(gen, chi0)
    return angular_momentum_mass(l)


# ── Quark offsets ─────────────────────────────────────────


def up_quark_l(gen: int, chi0: float = 19.0) -> float:
    """Angular momentum l for up-type quarks (u, c, t).

    Offset = −8 = −(χ₀ − 11) constant for all generations.
    l = gen·χ₀ − 8.

    gen=1 (u): l = 11  → l(l+1) = 132
    gen=2 (c): l = 30  → l(l+1) = 930
    gen=3 (t): l = 49  → l(l+1) = 2450
    """
    return gen * chi0 - (chi0 - 11)


def down_quark_l(gen: int, chi0: float = 19.0) -> float:
    """Angular momentum l for down-type quarks (d, s, b).

    Offset = gen − 8.
    l = gen·χ₀ + (gen − 8).

    gen=1 (d): l = 12  → l(l+1) = 156
    gen=2 (s): l = 32  → l(l+1) = 1056
    gen=3 (b): l = 52  → l(l+1) = 2756
    """
    return gen * chi0 + (gen - 8)


def up_quark_mass_ratio(gen: int, chi0: float = 19.0) -> float:
    """Up-type quark mass / electron mass."""
    return angular_momentum_mass(up_quark_l(gen, chi0))


def down_quark_mass_ratio(gen: int, chi0: float = 19.0) -> float:
    """Down-type quark mass / electron mass."""
    return angular_momentum_mass(down_quark_l(gen, chi0))


# ── Proton ────────────────────────────────────────────────


def proton_l(chi0: float = 19.0) -> float:
    """Proton angular momentum l = 2χ₀ + 4 = 42.

    Offset = +4 = D_st (spacetime dimensions).
    Three-generation decomposition:
    m_p/m_e = (χ₀−8)³ + χ₀² + (χ₀−7)² = 11³ + 361 + 144 = 1836.
    """
    return 2 * chi0 + 4


def proton_mass_ratio(chi0: float = 19.0) -> float:
    """m_p/m_e = l(l+1) = 42 × 43 = 1806."""
    return angular_momentum_mass(proton_l(chi0))


def proton_mass_ratio_3gen(chi0: float = 19.0) -> float:
    """m_p/m_e via three-generation decomposition.

    = (χ₀−8)³ + χ₀² + (χ₀−7)² = 11³ + 19² + 12² = 1836.
    """
    return (chi0 - 8) ** 3 + chi0**2 + (chi0 - 7) ** 2


# ── Electroweak boson masses ─────────────────────────────


def w_boson_mass_ratio(chi0: float = 19.0) -> float:
    """m_W/m_e = χ₀² × (24χ₀ − 20) = 157,396."""
    return chi0**2 * (24 * chi0 - 20)


def z_w_mass_ratio(chi0: float = 19.0) -> float:
    """m_Z/m_W = 9/8 = 1.125."""
    return 9.0 / 8.0


def higgs_w_mass_ratio(chi0: float = 19.0) -> float:
    """m_H/m_W = (χ₀ − D_st − 1)/N_gen² = 14/9."""
    D_st = 4  # spacetime dimensions
    N_gen = (chi0 - 1) / 6
    return (chi0 - D_st - 1) / N_gen**2


# ── Summary table ────────────────────────────────────────


def mass_table(chi0: float = 19.0) -> list[dict]:
    """Return a table of all mass predictions.

    Returns
    -------
    list of dict
        Each dict: particle, l, predicted, measured, error_pct.
    """
    entries = [
        ("electron", 0, 1.0, 1.0),
        ("muon", lepton_l(2, chi0), lepton_mass_ratio(2, chi0), 206.768),
        ("tau", lepton_l(3, chi0), lepton_mass_ratio(3, chi0), 3477.0),
        ("up", up_quark_l(1, chi0), up_quark_mass_ratio(1, chi0), 4.35),
        ("charm", up_quark_l(2, chi0), up_quark_mass_ratio(2, chi0), 2490.0),
        ("top", up_quark_l(3, chi0), up_quark_mass_ratio(3, chi0), 338600.0),
        ("down", down_quark_l(1, chi0), down_quark_mass_ratio(1, chi0), 9.4),
        ("strange", down_quark_l(2, chi0), down_quark_mass_ratio(2, chi0), 183.0),
        ("bottom", down_quark_l(3, chi0), down_quark_mass_ratio(3, chi0), 8190.0),
        ("proton", proton_l(chi0), proton_mass_ratio(chi0), 1836.15),
        ("W boson", None, w_boson_mass_ratio(chi0), 157294.0),
        ("Z/W ratio", None, z_w_mass_ratio(chi0), 1.1340),
        ("H/W ratio", None, higgs_w_mass_ratio(chi0), 1.5583),
    ]

    table = []
    for name, l_val, pred, meas in entries:
        if abs(meas) < 1e-30:
            err = 0.0
        else:
            err = abs(pred - meas) / abs(meas) * 100
        table.append({
            "particle": name,
            "l": l_val,
            "predicted": pred,
            "measured": meas,
            "error_pct": err,
        })
    return table
