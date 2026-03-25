"""
Particle Catalog for LFM Particle Physics Simulations
======================================================

Defines the ``Particle`` dataclass and the canonical particle table used
throughout Phases 1-5 of the particle physics project.

Usage::

    from lfm.particles.catalog import ELECTRON, MUON, PROTON, get_particle
    from lfm.particles.catalog import amplitude_for_particle, sigma_for_particle

All mass_ratio values are relative to the electron (m_e = 1).
Angular momentum quantum numbers l follow the LFM derivation:
    m/m_e = l(l+1)  with  l = tau * chi0 + offset

Phase 0 calibration results (c:\\Papers\\lfm-physics\\research\\p0_G_gonogo.md):
    N=64 best amplitude: 14-16  (NOT the default 6.0)
    chi_min must stay > 0 everywhere
    imaginary-time dtau <= 0.002
    max boost velocity ~0.08c
    l=14 (muon) rms_r ~ 4.1 cells on N=32 well
    l=42 (proton) rms_r ~ 6.8 cells on N=32 well
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lfm.constants import CHI0, E_AMPLITUDE_BY_GRID


@dataclass(frozen=True)
class Particle:
    """Specification of an LFM particle (soliton eigenmode).

    Parameters
    ----------
    name : str
        Lower-case canonical name (used as dict key).
    symbol : str
        ASCII-safe physics symbol (e.g. "e-", "mu-", "p+").
    mass_ratio : float
        Mass in units of electron mass (m_e = 1.0).
    charge : float
        Electric charge in units of e.  -1 = electron, +1 = positron, 0 = neutral.
    phase : float
        Initial wave phase in radians.  0 for negative charge, pi for positive.
    spin : float
        Spin quantum number.  0.5 for all fundamental fermions.
    color : str | None
        None for leptons/mesons/baryons; "r", "g", or "b" for quarks.
    field_level : int
        0 = real E (gravity only), 1 = complex Psi (gravity+EM),
        2 = 3-component color Psi (all forces).
    l : int
        Angular momentum quantum number for the eigenmode solver.
        From derivation: mass_ratio ~ l*(l+1).
    tau : int
        Temporal quantum number (generation marker in the l-formula).
    generation : int
        Particle generation: 1 (electron), 2 (muon), 3 (tau).
    category : str
        One of: "lepton", "quark", "baryon", "meson", "boson", "photon".
    stable : bool
        True if the particle is long-lived enough to simulate (no rapid decay).
    antiparticle : str | None
        Name (key) of the antiparticle, or None if self-conjugate.

    Properties
    ----------
    mass_eV : float
        Mass in eV (using m_e = 0.511 MeV).
    """

    name: str
    symbol: str
    mass_ratio: float
    charge: float
    phase: float
    spin: float
    color: str | None
    field_level: int
    l: int  # noqa: E741
    tau: int
    generation: int
    category: str
    stable: bool
    antiparticle: str | None

    @property
    def mass_eV(self) -> float:
        """Mass in eV."""
        return self.mass_ratio * 0.511e6

    @property
    def is_fermion(self) -> bool:
        return self.spin == 0.5

    @property
    def is_charged(self) -> bool:
        return self.charge != 0.0


# ---------------------------------------------------------------------------
# Tier 1: Leptons
# ---------------------------------------------------------------------------
# LFM derivation: l = tau * chi0 + offset
# Electron: l=0 (ground state, no angular momentum)
# Muon: l=14 -> m/m_e = 14*15 = 210  (measured: 206.768, 1.6% error)
# Tau: l=59 -> m/m_e = 59*60 = 3540  (measured: 3477, 1.8% error)

ELECTRON = Particle(
    name="electron",
    symbol="e-",
    mass_ratio=1.0,
    charge=-1.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=1,
    l=0,
    tau=0,
    generation=1,
    category="lepton",
    stable=True,
    antiparticle="positron",
)

POSITRON = Particle(
    name="positron",
    symbol="e+",
    mass_ratio=1.0,
    charge=+1.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=1,
    l=0,
    tau=0,
    generation=1,
    category="lepton",
    stable=True,
    antiparticle="electron",
)

MUON = Particle(
    name="muon",
    symbol="mu-",
    mass_ratio=206.768,
    charge=-1.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=1,
    l=14,
    tau=1,
    generation=2,
    category="lepton",
    stable=True,
    antiparticle="antimuon",
)

ANTIMUON = Particle(
    name="antimuon",
    symbol="mu+",
    mass_ratio=206.768,
    charge=+1.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=1,
    l=14,
    tau=1,
    generation=2,
    category="lepton",
    stable=True,
    antiparticle="muon",
)

TAU = Particle(
    name="tau",
    symbol="tau-",
    mass_ratio=3477.0,
    charge=-1.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=1,
    l=59,
    tau=3,
    generation=3,
    # tau lepton decays quickly but is stable enough for eigenmode calculation
    category="lepton",
    stable=False,
    antiparticle="antitau",
)

ANTITAU = Particle(
    name="antitau",
    symbol="tau+",
    mass_ratio=3477.0,
    charge=+1.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=1,
    l=59,
    tau=3,
    generation=3,
    category="lepton",
    stable=False,
    antiparticle="tau",
)

# ---------------------------------------------------------------------------
# Tier 2: Quarks (require field_level=2 for color)
# ---------------------------------------------------------------------------
# LFM angular momentum: up quark l offset = -(chi0-11) = -8 constant
# down quark l offset = generation - 8
# Quark mass ratios (vs electron):
#   up:    ~4 MeV / 0.511 MeV ~ 7.8
#   down:  ~8 MeV / 0.511 MeV ~ 15.7
#   strange: ~95 MeV / 0.511 MeV ~ 186.0
#   charm:  ~1270 MeV / 0.511 MeV ~ 2485.0

UP_QUARK = Particle(
    name="up",
    symbol="u",
    mass_ratio=7.8,
    charge=+2.0 / 3.0,
    phase=0.0,
    spin=0.5,
    color="r",
    field_level=2,
    l=2,
    tau=0,
    generation=1,
    category="quark",
    stable=False,
    antiparticle="anti-up",
)

DOWN_QUARK = Particle(
    name="down",
    symbol="d",
    mass_ratio=15.7,
    charge=-1.0 / 3.0,
    phase=0.0,
    spin=0.5,
    color="b",
    field_level=2,
    l=3,
    tau=0,
    generation=1,
    category="quark",
    stable=False,
    antiparticle="anti-down",
)

STRANGE_QUARK = Particle(
    name="strange",
    symbol="s",
    mass_ratio=186.0,
    charge=-1.0 / 3.0,
    phase=0.0,
    spin=0.5,
    color="g",
    field_level=2,
    l=14,
    tau=1,
    generation=2,
    category="quark",
    stable=False,
    antiparticle="anti-strange",
)

CHARM_QUARK = Particle(
    name="charm",
    symbol="c",
    mass_ratio=2485.0,
    charge=+2.0 / 3.0,
    phase=0.0,
    spin=0.5,
    color="r",
    field_level=2,
    l=50,
    tau=2,
    generation=2,
    category="quark",
    stable=False,
    antiparticle="anti-charm",
)

# ---------------------------------------------------------------------------
# Tier 3: Nucleons (composite — treated as effective solitons for Phase 4+)
# ---------------------------------------------------------------------------
# LFM derivation: m_p/m_e = l_p*(l_p+1), l_p=42 -> 42*43=1806 (measured:1836.15, 1.6%)
# Proton: l=42, tau=2
# Neutron: l=42, tau=2, charge=0 (composite of udd)

PROTON = Particle(
    name="proton",
    symbol="p+",
    mass_ratio=1836.15,
    charge=+1.0,
    phase=0.0,
    spin=0.5,
    # Use field_level=1 for gravity+EM; the color structure is internal
    color=None,
    field_level=1,
    l=42,
    tau=2,
    generation=1,
    category="baryon",
    stable=True,
    antiparticle="antiproton",
)

ANTIPROTON = Particle(
    name="antiproton",
    symbol="p-",
    mass_ratio=1836.15,
    charge=-1.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=1,
    l=42,
    tau=2,
    generation=1,
    category="baryon",
    stable=True,
    antiparticle="proton",
)

NEUTRON = Particle(
    name="neutron",
    symbol="n0",
    mass_ratio=1838.68,
    charge=0.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=0,  # neutral -> real field level
    l=42,
    tau=2,
    generation=1,
    category="baryon",
    stable=False,
    antiparticle="antineutron",
)

ANTINEUTRON = Particle(
    name="antineutron",
    symbol="n0bar",
    mass_ratio=1838.68,
    charge=0.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=0,
    l=42,
    tau=2,
    generation=1,
    category="baryon",
    stable=False,
    antiparticle="neutron",
)

# ---------------------------------------------------------------------------
# Tier 4: Photon
# ---------------------------------------------------------------------------
PHOTON = Particle(
    name="photon",
    symbol="y",
    mass_ratio=0.0,
    charge=0.0,
    phase=0.0,
    spin=1.0,
    color=None,
    field_level=1,
    l=0,
    tau=0,
    generation=1,
    category="photon",
    stable=True,
    antiparticle=None,  # self-conjugate
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PARTICLES: dict[str, Particle] = {
    p.name: p
    for p in [
        ELECTRON,
        POSITRON,
        MUON,
        ANTIMUON,
        TAU,
        ANTITAU,
        UP_QUARK,
        DOWN_QUARK,
        STRANGE_QUARK,
        CHARM_QUARK,
        PROTON,
        ANTIPROTON,
        NEUTRON,
        ANTINEUTRON,
        PHOTON,
    ]
}


def get_particle(name: str) -> Particle:
    """Look up a particle by name (case-insensitive).

    Parameters
    ----------
    name : str
        Particle name, e.g. "electron", "muon", "proton".

    Returns
    -------
    Particle

    Raises
    ------
    KeyError
        If the particle name is not in the catalog.
    """
    key = name.lower().strip()
    if key not in PARTICLES:
        available = sorted(PARTICLES.keys())
        raise KeyError(f"Unknown particle '{name}'. Available: {available}")
    return PARTICLES[key]


# ---------------------------------------------------------------------------
# Amplitude and sigma calibration
# ---------------------------------------------------------------------------
# Phase 0 calibration (research/p0_03_amplitude_scan.py, p0_05_scale_separation.py):
#
#   N=64 base amplitude for electron: 14.0  (NOT the library default of 6.0)
#   Scaling law for heavier particles: amp ~ sqrt(mass_ratio)
#   But capped to keep chi_min > 0 (Z2 vacuum flip prevention):
#     chi_min ~ chi0 - kappa * amp^2 * sigma^3 * (sqrt(2*pi))^3
#     For chi_min > 5 (safe margin): amp*sigma^1.5 < sqrt((chi0-5)/kappa) ~ sqrt(880) ~ 29.7
#
# The eigenmode solver fine-tunes amplitude during SCF iteration.
# These are SEED values only.

# Electron base amplitudes by grid size (from Phase 0 amplitude scan)
_ELECTRON_AMP: dict[int, float] = {
    32: 8.0,
    64: 14.0,
    128: 7.0,
    256: 3.5,
}


def amplitude_for_particle(particle: Particle, N: int) -> float:
    """Compute seed Gaussian amplitude for a particle on an N^3 grid.

    Heavier particles need deeper chi-wells: amplitude scales as
    the 4th root of mass_ratio (empirically more stable than sqrt).

    The returned value is a SEED for the eigenmode solver.  The SCF
    iteration will adjust it to achieve self-consistency.

    Parameters
    ----------
    particle : Particle
    N : int
        Grid size.

    Returns
    -------
    float
        Seed amplitude.
    """
    base = _ELECTRON_AMP.get(N, E_AMPLITUDE_BY_GRID.get(N, 14.0))
    if particle.mass_ratio <= 0:
        # Massless (photon) — use minimum detectable amplitude
        return base * 0.5
    # 4th-root scaling keeps chi_min > 0 for large mass_ratios
    scale = particle.mass_ratio**0.25
    amp = base * scale
    # Hard cap: prevent Z2 vacuum flip (chi_min must stay > 0)
    # Rule of thumb: amp <= 0.85 * chi0 for sigma=3
    cap = 0.85 * CHI0
    return float(min(amp, cap))


def sigma_for_particle(particle: Particle, N: int) -> float:
    """Compute seed Gaussian width (sigma) for a particle on an N^3 grid.

    Heavier particles are more compact: sigma ~ 1 / mass_ratio^(1/4).
    Minimum sigma = 2 cells (need ~4 cells across to resolve the mode).
    Maximum sigma = N/6 (wave must fit in box).

    Parameters
    ----------
    particle : Particle
    N : int
        Grid size.

    Returns
    -------
    float
        Seed sigma in lattice cells.
    """
    # Base sigma scales with box size
    base_sigma = max(3.0, N / 16.0)
    if particle.mass_ratio <= 0:
        return float(base_sigma * 2)
    # Lighter particles are wider; heavier are more compact
    scale = particle.mass_ratio ** (-0.25)
    sigma = base_sigma * scale
    return float(max(2.0, min(sigma, N / 6.0)))
