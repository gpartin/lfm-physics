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
    quark_content: tuple[str, ...] | None = None

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
    quark_content=("up", "up", "down"),
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
    quark_content=("anti-up", "anti-up", "anti-down"),
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
    quark_content=("up", "down", "down"),
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
    quark_content=("anti-up", "anti-down", "anti-down"),
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
# Tier 5: Antiquarks (CPT partners of Tier 2 quarks)
# ---------------------------------------------------------------------------

ANTI_UP_QUARK = Particle(
    name="anti-up",
    symbol="u~",
    mass_ratio=7.8,
    charge=-2.0 / 3.0,
    phase=math.pi,
    spin=0.5,
    color="r",
    field_level=2,
    l=2,
    tau=0,
    generation=1,
    category="quark",
    stable=False,
    antiparticle="up",
)

ANTI_DOWN_QUARK = Particle(
    name="anti-down",
    symbol="d~",
    mass_ratio=15.7,
    charge=+1.0 / 3.0,
    phase=math.pi,
    spin=0.5,
    color="b",
    field_level=2,
    l=3,
    tau=0,
    generation=1,
    category="quark",
    stable=False,
    antiparticle="down",
)

ANTI_STRANGE_QUARK = Particle(
    name="anti-strange",
    symbol="s~",
    mass_ratio=186.0,
    charge=+1.0 / 3.0,
    phase=math.pi,
    spin=0.5,
    color="g",
    field_level=2,
    l=14,
    tau=1,
    generation=2,
    category="quark",
    stable=False,
    antiparticle="strange",
)

ANTI_CHARM_QUARK = Particle(
    name="anti-charm",
    symbol="c~",
    mass_ratio=2485.0,
    charge=-2.0 / 3.0,
    phase=math.pi,
    spin=0.5,
    color="r",
    field_level=2,
    l=50,
    tau=2,
    generation=2,
    category="quark",
    stable=False,
    antiparticle="charm",
)

# ---------------------------------------------------------------------------
# Tier 6: Bottom and Top quarks + antiparticles
# ---------------------------------------------------------------------------
# Bottom: ~4.18 GeV / 0.511 MeV = 8180
# Top: ~173.0 GeV / 0.511 MeV = 338,600

BOTTOM_QUARK = Particle(
    name="bottom",
    symbol="b",
    mass_ratio=8180.0,
    charge=-1.0 / 3.0,
    phase=0.0,
    spin=0.5,
    color="g",
    field_level=2,
    l=90,
    tau=3,
    generation=3,
    category="quark",
    stable=False,
    antiparticle="anti-bottom",
)

ANTI_BOTTOM_QUARK = Particle(
    name="anti-bottom",
    symbol="b~",
    mass_ratio=8180.0,
    charge=+1.0 / 3.0,
    phase=math.pi,
    spin=0.5,
    color="g",
    field_level=2,
    l=90,
    tau=3,
    generation=3,
    category="quark",
    stable=False,
    antiparticle="bottom",
)

TOP_QUARK = Particle(
    name="top",
    symbol="t",
    mass_ratio=338600.0,
    charge=+2.0 / 3.0,
    phase=0.0,
    spin=0.5,
    color="r",
    field_level=2,
    l=582,
    tau=3,
    generation=3,
    category="quark",
    stable=False,
    antiparticle="anti-top",
)

ANTI_TOP_QUARK = Particle(
    name="anti-top",
    symbol="t~",
    mass_ratio=338600.0,
    charge=-2.0 / 3.0,
    phase=math.pi,
    spin=0.5,
    color="r",
    field_level=2,
    l=582,
    tau=3,
    generation=3,
    category="quark",
    stable=False,
    antiparticle="top",
)

# ---------------------------------------------------------------------------
# Tier 7: Neutrinos (nearly massless, neutral, REAL field level)
# ---------------------------------------------------------------------------
# Neutrino masses (upper bounds): nu_e < 0.8 eV, nu_mu < 0.17 MeV, nu_tau < 18.2 MeV
# Using best-fit squared mass differences:
#   nu_1 ~ 0, nu_2 ~ 0.0086 eV, nu_3 ~ 0.050 eV
# mass_ratio = m_nu / m_e: all < 0.0001
# For practical simulation: use small but nonzero mass_ratio.

ELECTRON_NEUTRINO = Particle(
    name="electron_neutrino",
    symbol="nu_e",
    mass_ratio=0.000004,  # ~2 eV upper bound / 511000 eV
    charge=0.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=0,
    l=0,
    tau=0,
    generation=1,
    category="lepton",
    stable=True,
    antiparticle="anti_electron_neutrino",
)

ANTI_ELECTRON_NEUTRINO = Particle(
    name="anti_electron_neutrino",
    symbol="nu_e~",
    mass_ratio=0.000004,
    charge=0.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=0,
    l=0,
    tau=0,
    generation=1,
    category="lepton",
    stable=True,
    antiparticle="electron_neutrino",
)

MUON_NEUTRINO = Particle(
    name="muon_neutrino",
    symbol="nu_mu",
    mass_ratio=0.000017,
    charge=0.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=0,
    l=0,
    tau=0,
    generation=2,
    category="lepton",
    stable=True,
    antiparticle="anti_muon_neutrino",
)

ANTI_MUON_NEUTRINO = Particle(
    name="anti_muon_neutrino",
    symbol="nu_mu~",
    mass_ratio=0.000017,
    charge=0.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=0,
    l=0,
    tau=0,
    generation=2,
    category="lepton",
    stable=True,
    antiparticle="muon_neutrino",
)

TAU_NEUTRINO = Particle(
    name="tau_neutrino",
    symbol="nu_tau",
    mass_ratio=0.000030,
    charge=0.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=0,
    l=0,
    tau=0,
    generation=3,
    category="lepton",
    stable=True,
    antiparticle="anti_tau_neutrino",
)

ANTI_TAU_NEUTRINO = Particle(
    name="anti_tau_neutrino",
    symbol="nu_tau~",
    mass_ratio=0.000030,
    charge=0.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=0,
    l=0,
    tau=0,
    generation=3,
    category="lepton",
    stable=True,
    antiparticle="tau_neutrino",
)

# ---------------------------------------------------------------------------
# Tier 8: Gauge Bosons (W+, W-, Z0, gluon)
# ---------------------------------------------------------------------------
# W boson: 80.379 GeV / 0.511 MeV = 157,298
# Z boson: 91.188 GeV / 0.511 MeV = 178,450

W_PLUS = Particle(
    name="w_plus",
    symbol="W+",
    mass_ratio=157298.0,
    charge=+1.0,
    phase=math.pi,
    spin=1.0,
    color=None,
    field_level=1,
    l=396,
    tau=0,
    generation=0,
    category="boson",
    stable=False,
    antiparticle="w_minus",
)

W_MINUS = Particle(
    name="w_minus",
    symbol="W-",
    mass_ratio=157298.0,
    charge=-1.0,
    phase=0.0,
    spin=1.0,
    color=None,
    field_level=1,
    l=396,
    tau=0,
    generation=0,
    category="boson",
    stable=False,
    antiparticle="w_plus",
)

Z_BOSON = Particle(
    name="z_boson",
    symbol="Z0",
    mass_ratio=178450.0,
    charge=0.0,
    phase=0.0,
    spin=1.0,
    color=None,
    field_level=0,
    l=422,
    tau=0,
    generation=0,
    category="boson",
    stable=False,
    antiparticle=None,  # self-conjugate
)

GLUON = Particle(
    name="gluon",
    symbol="g",
    mass_ratio=0.0,
    charge=0.0,
    phase=0.0,
    spin=1.0,
    color="r",
    field_level=2,
    l=0,
    tau=0,
    generation=0,
    category="boson",
    stable=True,
    antiparticle=None,  # self-conjugate (color/anticolor)
)

# ---------------------------------------------------------------------------
# Tier 9: Higgs Boson
# ---------------------------------------------------------------------------
# Higgs: 125.25 GeV / 0.511 MeV = 245,100
# In LFM: chi-field oscillation mode (Mexican hat), omega_H ~ 19.30

HIGGS = Particle(
    name="higgs",
    symbol="H0",
    mass_ratio=245100.0,
    charge=0.0,
    phase=0.0,
    spin=0.0,
    color=None,
    field_level=0,
    l=495,
    tau=0,
    generation=0,
    category="boson",
    stable=False,
    antiparticle=None,  # self-conjugate
)

# ---------------------------------------------------------------------------
# Tier 10: Light Mesons (quark-antiquark composites)
# ---------------------------------------------------------------------------
# Mesons are qq-bar bound states. quark_content specifies constituents.
# field_level = COMPLEX for charged, REAL for neutral.
# In LFM: these are two-soliton bound states in a shared chi-well.

PION_PLUS = Particle(
    name="pion_plus",
    symbol="pi+",
    mass_ratio=273.13,
    charge=+1.0,
    phase=math.pi,
    spin=0.0,
    color=None,
    field_level=1,
    l=16,
    tau=0,
    generation=1,
    category="meson",
    stable=False,
    antiparticle="pion_minus",
    quark_content=("up", "anti-down"),
)

PION_MINUS = Particle(
    name="pion_minus",
    symbol="pi-",
    mass_ratio=273.13,
    charge=-1.0,
    phase=0.0,
    spin=0.0,
    color=None,
    field_level=1,
    l=16,
    tau=0,
    generation=1,
    category="meson",
    stable=False,
    antiparticle="pion_plus",
    quark_content=("anti-up", "down"),
)

PION_ZERO = Particle(
    name="pion_zero",
    symbol="pi0",
    mass_ratio=263.89,
    charge=0.0,
    phase=0.0,
    spin=0.0,
    color=None,
    field_level=0,
    l=16,
    tau=0,
    generation=1,
    category="meson",
    stable=False,
    antiparticle=None,  # self-conjugate
    quark_content=("up", "anti-up"),  # simplified; actually (uu-bar - dd-bar)/sqrt(2)
)

KAON_PLUS = Particle(
    name="kaon_plus",
    symbol="K+",
    mass_ratio=966.12,
    charge=+1.0,
    phase=math.pi,
    spin=0.0,
    color=None,
    field_level=1,
    l=31,
    tau=0,
    generation=2,
    category="meson",
    stable=False,
    antiparticle="kaon_minus",
    quark_content=("up", "anti-strange"),
)

KAON_MINUS = Particle(
    name="kaon_minus",
    symbol="K-",
    mass_ratio=966.12,
    charge=-1.0,
    phase=0.0,
    spin=0.0,
    color=None,
    field_level=1,
    l=31,
    tau=0,
    generation=2,
    category="meson",
    stable=False,
    antiparticle="kaon_plus",
    quark_content=("anti-up", "strange"),
)

KAON_ZERO = Particle(
    name="kaon_zero",
    symbol="K0",
    mass_ratio=974.55,
    charge=0.0,
    phase=0.0,
    spin=0.0,
    color=None,
    field_level=0,
    l=31,
    tau=0,
    generation=2,
    category="meson",
    stable=False,
    antiparticle="anti_kaon_zero",
    quark_content=("down", "anti-strange"),
)

ANTI_KAON_ZERO = Particle(
    name="anti_kaon_zero",
    symbol="K0~",
    mass_ratio=974.55,
    charge=0.0,
    phase=math.pi,
    spin=0.0,
    color=None,
    field_level=0,
    l=31,
    tau=0,
    generation=2,
    category="meson",
    stable=False,
    antiparticle="kaon_zero",
    quark_content=("anti-down", "strange"),
)

ETA_MESON = Particle(
    name="eta",
    symbol="eta",
    mass_ratio=1073.2,
    charge=0.0,
    phase=0.0,
    spin=0.0,
    color=None,
    field_level=0,
    l=32,
    tau=0,
    generation=1,
    category="meson",
    stable=False,
    antiparticle=None,  # self-conjugate
    quark_content=("up", "anti-up"),  # simplified flavor singlet
)

RHO_MESON = Particle(
    name="rho",
    symbol="rho0",
    mass_ratio=1513.4,
    charge=0.0,
    phase=0.0,
    spin=1.0,
    color=None,
    field_level=0,
    l=38,
    tau=0,
    generation=1,
    category="meson",
    stable=False,
    antiparticle=None,  # self-conjugate
    quark_content=("up", "anti-up"),
)

# ---------------------------------------------------------------------------
# Tier 11: Charm Mesons (D mesons, J/psi)
# ---------------------------------------------------------------------------
# D+: cd-bar, 1869.65 MeV / 0.511 MeV = 3659
# D0: cu-bar, 1864.84 MeV / 0.511 MeV = 3650
# J/psi: cc-bar, 3096.9 MeV / 0.511 MeV = 6060

D_PLUS = Particle(
    name="d_plus",
    symbol="D+",
    mass_ratio=3659.0,
    charge=+1.0,
    phase=math.pi,
    spin=0.0,
    color=None,
    field_level=1,
    l=60,
    tau=0,
    generation=2,
    category="meson",
    stable=False,
    antiparticle="d_minus",
    quark_content=("charm", "anti-down"),
)

D_MINUS = Particle(
    name="d_minus",
    symbol="D-",
    mass_ratio=3659.0,
    charge=-1.0,
    phase=0.0,
    spin=0.0,
    color=None,
    field_level=1,
    l=60,
    tau=0,
    generation=2,
    category="meson",
    stable=False,
    antiparticle="d_plus",
    quark_content=("anti-charm", "down"),
)

D_ZERO = Particle(
    name="d_zero",
    symbol="D0",
    mass_ratio=3650.0,
    charge=0.0,
    phase=0.0,
    spin=0.0,
    color=None,
    field_level=0,
    l=60,
    tau=0,
    generation=2,
    category="meson",
    stable=False,
    antiparticle="anti_d_zero",
    quark_content=("charm", "anti-up"),
)

ANTI_D_ZERO = Particle(
    name="anti_d_zero",
    symbol="D0~",
    mass_ratio=3650.0,
    charge=0.0,
    phase=math.pi,
    spin=0.0,
    color=None,
    field_level=0,
    l=60,
    tau=0,
    generation=2,
    category="meson",
    stable=False,
    antiparticle="d_zero",
    quark_content=("anti-charm", "up"),
)

J_PSI = Particle(
    name="j_psi",
    symbol="J/psi",
    mass_ratio=6060.0,
    charge=0.0,
    phase=0.0,
    spin=1.0,
    color=None,
    field_level=0,
    l=77,
    tau=0,
    generation=2,
    category="meson",
    stable=False,
    antiparticle=None,  # self-conjugate
    quark_content=("charm", "anti-charm"),
)

# ---------------------------------------------------------------------------
# Tier 12: Bottom Mesons (B mesons, Upsilon)
# ---------------------------------------------------------------------------
# B+: ub-bar, 5279.34 MeV / 0.511 MeV = 10,332
# B0: db-bar, 5279.66 MeV / 0.511 MeV = 10,332
# Upsilon: bb-bar, 9460.3 MeV / 0.511 MeV = 18,514

B_PLUS = Particle(
    name="b_plus",
    symbol="B+",
    mass_ratio=10332.0,
    charge=+1.0,
    phase=math.pi,
    spin=0.0,
    color=None,
    field_level=1,
    l=101,
    tau=0,
    generation=3,
    category="meson",
    stable=False,
    antiparticle="b_minus",
    quark_content=("up", "anti-bottom"),
)

B_MINUS = Particle(
    name="b_minus",
    symbol="B-",
    mass_ratio=10332.0,
    charge=-1.0,
    phase=0.0,
    spin=0.0,
    color=None,
    field_level=1,
    l=101,
    tau=0,
    generation=3,
    category="meson",
    stable=False,
    antiparticle="b_plus",
    quark_content=("anti-up", "bottom"),
)

B_ZERO = Particle(
    name="b_zero",
    symbol="B0",
    mass_ratio=10332.0,
    charge=0.0,
    phase=0.0,
    spin=0.0,
    color=None,
    field_level=0,
    l=101,
    tau=0,
    generation=3,
    category="meson",
    stable=False,
    antiparticle="anti_b_zero",
    quark_content=("down", "anti-bottom"),
)

ANTI_B_ZERO = Particle(
    name="anti_b_zero",
    symbol="B0~",
    mass_ratio=10332.0,
    charge=0.0,
    phase=math.pi,
    spin=0.0,
    color=None,
    field_level=0,
    l=101,
    tau=0,
    generation=3,
    category="meson",
    stable=False,
    antiparticle="b_zero",
    quark_content=("anti-down", "bottom"),
)

UPSILON = Particle(
    name="upsilon",
    symbol="Y(1S)",
    mass_ratio=18514.0,
    charge=0.0,
    phase=0.0,
    spin=1.0,
    color=None,
    field_level=0,
    l=136,
    tau=0,
    generation=3,
    category="meson",
    stable=False,
    antiparticle=None,  # self-conjugate
    quark_content=("bottom", "anti-bottom"),
)

# ---------------------------------------------------------------------------
# Tier 13: Strange Baryons
# ---------------------------------------------------------------------------
# Lambda: uds, 1115.68 MeV / 0.511 MeV = 2183
# Sigma+: uus, 1189.37 MeV / 0.511 MeV = 2328
# Sigma-: dds, 1197.45 MeV / 0.511 MeV = 2343
# Xi-: dss, 1321.71 MeV / 0.511 MeV = 2587

LAMBDA_BARYON = Particle(
    name="lambda_baryon",
    symbol="Lambda0",
    mass_ratio=2183.0,
    charge=0.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=0,
    l=46,
    tau=1,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="anti_lambda_baryon",
    quark_content=("up", "down", "strange"),
)

ANTI_LAMBDA_BARYON = Particle(
    name="anti_lambda_baryon",
    symbol="Lambda0~",
    mass_ratio=2183.0,
    charge=0.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=0,
    l=46,
    tau=1,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="lambda_baryon",
    quark_content=("anti-up", "anti-down", "anti-strange"),
)

SIGMA_PLUS = Particle(
    name="sigma_plus",
    symbol="Sigma+",
    mass_ratio=2328.0,
    charge=+1.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=1,
    l=48,
    tau=1,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="anti_sigma_minus",
    quark_content=("up", "up", "strange"),
)

SIGMA_MINUS = Particle(
    name="sigma_minus",
    symbol="Sigma-",
    mass_ratio=2343.0,
    charge=-1.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=1,
    l=48,
    tau=1,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="anti_sigma_plus",
    quark_content=("down", "down", "strange"),
)

ANTI_SIGMA_PLUS = Particle(
    name="anti_sigma_plus",
    symbol="Sigma+~",
    mass_ratio=2343.0,
    charge=+1.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=1,
    l=48,
    tau=1,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="sigma_minus",
    quark_content=("anti-down", "anti-down", "anti-strange"),
)

ANTI_SIGMA_MINUS = Particle(
    name="anti_sigma_minus",
    symbol="Sigma-~",
    mass_ratio=2328.0,
    charge=-1.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=1,
    l=48,
    tau=1,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="sigma_plus",
    quark_content=("anti-up", "anti-up", "anti-strange"),
)

XI_MINUS = Particle(
    name="xi_minus",
    symbol="Xi-",
    mass_ratio=2587.0,
    charge=-1.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=1,
    l=50,
    tau=1,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="anti_xi_plus",
    quark_content=("down", "strange", "strange"),
)

ANTI_XI_PLUS = Particle(
    name="anti_xi_plus",
    symbol="Xi+~",
    mass_ratio=2587.0,
    charge=+1.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=1,
    l=50,
    tau=1,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="xi_minus",
    quark_content=("anti-down", "anti-strange", "anti-strange"),
)

OMEGA_BARYON = Particle(
    name="omega_baryon",
    symbol="Omega-",
    mass_ratio=3277.0,
    charge=-1.0,
    phase=0.0,
    spin=1.5,
    color=None,
    field_level=1,
    l=57,
    tau=1,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="anti_omega_baryon",
    quark_content=("strange", "strange", "strange"),
)

ANTI_OMEGA_BARYON = Particle(
    name="anti_omega_baryon",
    symbol="Omega+~",
    mass_ratio=3277.0,
    charge=+1.0,
    phase=math.pi,
    spin=1.5,
    color=None,
    field_level=1,
    l=57,
    tau=1,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="omega_baryon",
    quark_content=("anti-strange", "anti-strange", "anti-strange"),
)

# ---------------------------------------------------------------------------
# Tier 14: Heavy Baryons (charm, bottom)
# ---------------------------------------------------------------------------
# Lambda_c+: udc, 2286.46 MeV / 0.511 MeV = 4475
# Xi_c+: usc, 2467.87 MeV / 0.511 MeV = 4830
# Lambda_b0: udb, 5619.60 MeV / 0.511 MeV = 10,998

LAMBDA_C = Particle(
    name="lambda_c",
    symbol="Lambda_c+",
    mass_ratio=4475.0,
    charge=+1.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=1,
    l=66,
    tau=2,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="anti_lambda_c",
    quark_content=("up", "down", "charm"),
)

ANTI_LAMBDA_C = Particle(
    name="anti_lambda_c",
    symbol="Lambda_c-~",
    mass_ratio=4475.0,
    charge=-1.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=1,
    l=66,
    tau=2,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="lambda_c",
    quark_content=("anti-up", "anti-down", "anti-charm"),
)

XI_C_PLUS = Particle(
    name="xi_c_plus",
    symbol="Xi_c+",
    mass_ratio=4830.0,
    charge=+1.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=1,
    l=69,
    tau=2,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="anti_xi_c_minus",
    quark_content=("up", "strange", "charm"),
)

ANTI_XI_C_MINUS = Particle(
    name="anti_xi_c_minus",
    symbol="Xi_c-~",
    mass_ratio=4830.0,
    charge=-1.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=1,
    l=69,
    tau=2,
    generation=2,
    category="baryon",
    stable=False,
    antiparticle="xi_c_plus",
    quark_content=("anti-up", "anti-strange", "anti-charm"),
)

LAMBDA_B = Particle(
    name="lambda_b",
    symbol="Lambda_b0",
    mass_ratio=10998.0,
    charge=0.0,
    phase=0.0,
    spin=0.5,
    color=None,
    field_level=0,
    l=104,
    tau=3,
    generation=3,
    category="baryon",
    stable=False,
    antiparticle="anti_lambda_b",
    quark_content=("up", "down", "bottom"),
)

ANTI_LAMBDA_B = Particle(
    name="anti_lambda_b",
    symbol="Lambda_b0~",
    mass_ratio=10998.0,
    charge=0.0,
    phase=math.pi,
    spin=0.5,
    color=None,
    field_level=0,
    l=104,
    tau=3,
    generation=3,
    category="baryon",
    stable=False,
    antiparticle="lambda_b",
    quark_content=("anti-up", "anti-down", "anti-bottom"),
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PARTICLES: dict[str, Particle] = {
    p.name: p
    for p in [
        # Tier 1: Leptons
        ELECTRON,
        POSITRON,
        MUON,
        ANTIMUON,
        TAU,
        ANTITAU,
        # Tier 5: Antiquarks
        ANTI_UP_QUARK,
        ANTI_DOWN_QUARK,
        ANTI_STRANGE_QUARK,
        ANTI_CHARM_QUARK,
        # Tier 2: Quarks
        UP_QUARK,
        DOWN_QUARK,
        STRANGE_QUARK,
        CHARM_QUARK,
        # Tier 6: Bottom/Top quarks
        BOTTOM_QUARK,
        ANTI_BOTTOM_QUARK,
        TOP_QUARK,
        ANTI_TOP_QUARK,
        # Tier 7: Neutrinos
        ELECTRON_NEUTRINO,
        ANTI_ELECTRON_NEUTRINO,
        MUON_NEUTRINO,
        ANTI_MUON_NEUTRINO,
        TAU_NEUTRINO,
        ANTI_TAU_NEUTRINO,
        # Tier 3: Nucleons
        PROTON,
        ANTIPROTON,
        NEUTRON,
        ANTINEUTRON,
        # Tier 4+8: Bosons
        PHOTON,
        W_PLUS,
        W_MINUS,
        Z_BOSON,
        GLUON,
        HIGGS,
        # Tier 10: Light Mesons
        PION_PLUS,
        PION_MINUS,
        PION_ZERO,
        KAON_PLUS,
        KAON_MINUS,
        KAON_ZERO,
        ANTI_KAON_ZERO,
        ETA_MESON,
        RHO_MESON,
        # Tier 11: Charm Mesons
        D_PLUS,
        D_MINUS,
        D_ZERO,
        ANTI_D_ZERO,
        J_PSI,
        # Tier 12: Bottom Mesons
        B_PLUS,
        B_MINUS,
        B_ZERO,
        ANTI_B_ZERO,
        UPSILON,
        # Tier 13: Strange Baryons
        LAMBDA_BARYON,
        ANTI_LAMBDA_BARYON,
        SIGMA_PLUS,
        SIGMA_MINUS,
        ANTI_SIGMA_PLUS,
        ANTI_SIGMA_MINUS,
        XI_MINUS,
        ANTI_XI_PLUS,
        OMEGA_BARYON,
        ANTI_OMEGA_BARYON,
        # Tier 14: Heavy Baryons
        LAMBDA_C,
        ANTI_LAMBDA_C,
        XI_C_PLUS,
        ANTI_XI_C_MINUS,
        LAMBDA_B,
        ANTI_LAMBDA_B,
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
