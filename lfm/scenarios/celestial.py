"""Celestial body abstraction for multi-body LFM simulations.

Maps physical astronomical parameters to LFM soliton parameters:
  mass_solar → amplitude  (chi-well depth ∝ gravitational mass)
  body_type  → sigma      (compactness: BH < star < gas giant)

Key types and functions
-----------------------
``BodyType``
    Enum of supported astronomical object classes.
``CelestialBody``
    Dataclass storing physical properties; exposes derived LFM parameters
    as read-only properties.
``solar_system``, ``black_hole_system``, ``galaxy_core``
    Factory functions returning ready-to-use body lists.
``place_bodies(sim, bodies)``
    Place all bodies in the simulation, equilibrate chi, assign Keplerian
    orbital velocities, and return a ``{name: omega}`` dict for tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from lfm.config import FieldLevel
from lfm.constants import CHI0

# ══════════════════════════════════════════════════════════════════════════════
# Body type definitions
# ══════════════════════════════════════════════════════════════════════════════


class BodyType(str, Enum):
    """Astronomical object class.

    Inheriting from ``str`` means instances compare equal to their string
    value, e.g. ``BodyType.STAR == "star"`` is True.
    """

    ROCKY_PLANET = "rocky_planet"
    GAS_PLANET = "gas_planet"
    STAR = "star"
    NEUTRON_STAR = "neutron_star"
    BLACK_HOLE = "black_hole"
    SMBH = "smbh"


# Visual overlay style per type (no physics — only affects the rendered movie)
_VISUAL: dict[str, dict] = {
    BodyType.ROCKY_PLANET: dict(color="#CD853F", size=50, ring="#FF8C00"),
    BodyType.GAS_PLANET: dict(color="#4488FF", size=120, ring="#88CCFF"),
    BodyType.STAR: dict(color="#FFD700", size=200, ring="#FFFFFF"),
    BodyType.NEUTRON_STAR: dict(color="#00FFFF", size=60, ring="#00AAAA"),
    BodyType.BLACK_HOLE: dict(color="#330033", size=180, ring="#FF00FF"),
    BodyType.SMBH: dict(color="#1A0000", size=320, ring="#FF4400"),
}

# LFM soliton width (sigma, in lattice cells) by type.
# Compact objects (BH, NS) have small sigma; diffuse objects have large sigma.
_SIGMA: dict[str, float] = {
    BodyType.ROCKY_PLANET: 1.5,
    BodyType.GAS_PLANET: 2.5,
    BodyType.STAR: 2.5,
    BodyType.NEUTRON_STAR: 1.0,
    BodyType.BLACK_HOLE: 3.0,
    BodyType.SMBH: 4.0,
}

# Amplitude cap per type (prevents extreme chi excursions and Nyquist issues)
_AMP_MAX: dict[str, float] = {
    BodyType.ROCKY_PLANET: 2.0,
    BodyType.GAS_PLANET: 4.0,
    BodyType.STAR: 8.0,
    BodyType.NEUTRON_STAR: 10.0,
    BodyType.BLACK_HOLE: 13.0,
    BodyType.SMBH: 15.0,
}

_AMP_SCALE = 7.0  # 1 M☉ → amplitude 7.0   (calibrated so Sun fills grid nicely)
_AMP_EXPONENT = 0.45  # amplitude ∝ mass^0.45   (sub-linear to keep dynamic range sane)
_AMP_MIN = 0.3  # minimum visible amplitude


# ══════════════════════════════════════════════════════════════════════════════
# CelestialBody dataclass
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CelestialBody:
    """An astronomical object whose physical properties drive LFM soliton parameters.

    Parameters
    ----------
    name : str
        Label shown in the animated movie overlay.
    body_type : BodyType
        Controls compactness (sigma) and visual rendering.
    mass_solar : float
        Mass in solar masses.  Determines chi-well depth via amplitude.
        Typical values: Sun = 1.0, Earth ≈ 3 × 10⁻⁶, Jupiter ≈ 10⁻³,
        stellar BH ≈ 5–20, Milky Way SMBH ≈ 4 × 10⁶.
    orbital_radius : float
        Distance from grid centre in lattice cells.  Set to 0 for a fixed
        central body.
    orbital_phase : float
        Initial orbital angle in radians (0 = positive x-axis).  Ignored
        for central bodies.

    Derived LFM parameters (read-only properties)
    -----------------------------------------------
    amplitude : float
        Soliton amplitude = AMP_SCALE × mass^0.45, capped per type.
        Governs chi-well depth and hence gravitational binding energy.
    sigma : float
        Gaussian soliton width in lattice cells.  Compact bodies have
        smaller sigma.
    color, ring_color, dot_size
        Visual properties for the animated overlay.

    Physics
    -------
    The mapping between physical and LFM parameters is intentional:

    * ``amplitude²  × sigma³ ∝`` total soliton energy ∝ gravitational mass.
    * Black holes: narrow deep chi well (compact sigma + large amplitude).
    * Gas giants: broad shallow chi well (large sigma + small amplitude).
    * SMBH: wide AND deep chi well → extended flat rotation curve.
    """

    name: str
    body_type: BodyType
    mass_solar: float
    orbital_radius: float
    orbital_phase: float = 0.0

    @property
    def amplitude(self) -> float:
        raw = _AMP_SCALE * (self.mass_solar**_AMP_EXPONENT)
        return float(np.clip(raw, _AMP_MIN, _AMP_MAX[self.body_type]))

    @property
    def sigma(self) -> float:
        return _SIGMA[self.body_type]

    @property
    def color(self) -> str:
        return _VISUAL[self.body_type]["color"]

    @property
    def ring_color(self) -> str:
        return _VISUAL[self.body_type]["ring"]

    @property
    def dot_size(self) -> int:
        return _VISUAL[self.body_type]["size"]


# ══════════════════════════════════════════════════════════════════════════════
# Prebuilt scenario factories
# ══════════════════════════════════════════════════════════════════════════════


def solar_system() -> list[CelestialBody]:
    """Inner solar system: Sun + 4 rocky planets.

    The Sun mass (0.15 M\u2609) is calibrated so that the chi-well depth
    gives orbital speeds safely below the Nyquist limit
    (v_max \u2248 0.13 c) for all four planets.  Actual GOV-01/GOV-02
    orbital periods at dt=0.02:

      Mercury (r=12):  ~57 k steps
      Venus   (r=18):  ~105 k steps
      Earth   (r=25):  ~175 k steps
      Mars    (r=33):  ~275 k steps

    Planet masses are set to 0.05 M\u2609 (well above the amplitude floor
    _AMP_MIN=0.3) so each planet has amplitude \u22481.44 and sufficient
    psi\u00b2 signal to survive the full 120-frame / 90k-step movie.
    Real relative masses would fall below _AMP_MIN and the tracker
    would lose signal by frame ~15.

    Recommended grid: N = 128\u2013256.
    """
    return [
        CelestialBody("Sun", BodyType.STAR, 0.15, 0),
        CelestialBody("Mercury", BodyType.ROCKY_PLANET, 0.05, 12, 0.0),
        CelestialBody("Venus", BodyType.ROCKY_PLANET, 0.05, 18, 0.8),
        CelestialBody("Earth", BodyType.ROCKY_PLANET, 0.05, 25, 2.1),
        CelestialBody("Mars", BodyType.ROCKY_PLANET, 0.05, 33, 3.7),
    ]


def black_hole_system() -> list[CelestialBody]:
    """Stellar BH with companion stars and an inner debris ring.

    Recommended grid: N = 128.
    """
    return [
        CelestialBody("Black Hole", BodyType.BLACK_HOLE, 10.0, 0),
        CelestialBody("Star A", BodyType.STAR, 1.2, 38, 0.0),
        CelestialBody("Star B", BodyType.STAR, 0.9, 50, 1.7),
        CelestialBody("Debris 1", BodyType.ROCKY_PLANET, 0.05, 22, 3.0),
        CelestialBody("Debris 2", BodyType.ROCKY_PLANET, 0.05, 26, 5.2),
        CelestialBody("Debris 3", BodyType.ROCKY_PLANET, 0.05, 30, 1.0),
        CelestialBody("Debris 4", BodyType.ROCKY_PLANET, 0.05, 34, 4.3),
    ]


def galaxy_core(n_stars: int = 18) -> list[CelestialBody]:
    """SMBH surrounded by a stellar disk.

    Dark matter halo emerges automatically from chi wave inertia (the
    chi_prev buffer in the leapfrog integrator — no GOV-03/04 needed).

    Recommended grid: N = 256.
    """
    rng = np.random.default_rng(2026)
    bodies: list[CelestialBody] = [
        CelestialBody("Sgr A*", BodyType.SMBH, 4_000_000.0, 0),
    ]
    radii = np.linspace(15, 88, n_stars)
    phases = rng.uniform(0, 2 * np.pi, n_stars)
    masses = rng.uniform(0.5, 3.0, n_stars)
    for i, (r, theta, m) in enumerate(zip(radii, phases, masses, strict=False)):
        bodies.append(CelestialBody(f"S{i + 1}", BodyType.STAR, float(m), float(r), float(theta)))
    return bodies


# ══════════════════════════════════════════════════════════════════════════════
# Body placement helper
# ══════════════════════════════════════════════════════════════════════════════

#: Minimum v_chi  below which the rotation-curve value is treated as zero.
_VALID_V: float = 5e-4


def place_bodies(
    sim,
    bodies: list[CelestialBody],
    *,
    v_scale: float = 0.85,
    v_nyquist_fraction: float = 0.92,
    equilibrate: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    """Place all celestial bodies in *sim* and return angular velocities.

    Workflow
    --------
    1. Place the central body (orbital_radius == 0) and equilibrate chi.
    2. Measure the rotation curve (GOV-02 chi-gradient velocities).
    3. For radii beyond the chi wave-propagation front apply Keplerian
       1/√r extrapolation (physically correct for a point-mass potential at
       those scales, where no dark-matter halo has formed yet).
    4. Place each orbiting body with the appropriate tangential velocity,
       capped at the Nyquist limit to prevent place_soliton errors.
    5. Re-equilibrate chi with all bodies in place.

    Parameters
    ----------
    sim
        A fully configured ``lfm.Simulation`` instance (not yet run).
    bodies : list of CelestialBody
        Complete body list including the central body.
    v_scale : float
        Fraction of the circular velocity to assign (stability margin).
        The default 0.85 gives mildly elongated orbits, which is more
        stable than exactly circular for long runs.
    v_nyquist_fraction : float
        Hard cap as a fraction of the Nyquist velocity limit.  Bodies
        near a massive compact object will orbit slightly sub-circularly
        rather than failing placement.
    equilibrate : bool
        If False, skip the final re-equilibration (useful when you want
        to inspect the freshly-placed chi profile).
    verbose : bool
        Print placement diagnostics.

    Returns
    -------
    body_omega : dict[str, float]
        Angular velocity ω = v / r (rad / time unit) for each body.
        Use this dict together with the initial orbital_phase to track
        body positions analytically: θ(t) = orbital_phase + ω × t.
    """
    # Deferred import avoids circular dependency lfm → lfm.scenarios → lfm
    from lfm.analysis.observables import rotation_curve

    N = sim.chi.shape[0]
    cx = cy = cz = N // 2

    # ── 1. Place central body ─────────────────────────────────────────────────
    central = next((b for b in bodies if b.orbital_radius == 0), None)
    if central is not None:
        if verbose:
            print(
                f"Placing central body: {central.name}  "
                f"amp={central.amplitude:.3f}  σ={central.sigma:.1f}"
            )
        sim.place_soliton(
            (cx, cy, cz),
            amplitude=central.amplitude,
            sigma=central.sigma,
        )
        sim.equilibrate()
        if verbose:
            chi_min = float(np.array(sim.chi).min())
            print(f"  χ_min = {chi_min:.4f}  (background χ₀ = {CHI0})")

    # ── 2. Rotation curve + Keplerian extrapolation ───────────────────────────
    rc = rotation_curve(sim.chi, sim.energy_density, center=(cx, cy, cz), plane_axis=2)
    r_arr = np.asarray(rc["r"], dtype=np.float64)
    v_chi_raw = np.asarray(rc["v_chi"], dtype=np.float64)

    # chi propagates at c=1; after equilibrate(), the wave front is only at
    # r ≈ equilibrate_steps × dt ≈ a few cells.  Beyond that front, v_chi≈0.
    # Use Keplerian v ∝ 1/√r extrapolated from the last well-measured radius.
    valid = v_chi_raw > _VALID_V
    if valid.any():
        r_ref = float(r_arr[valid][-1])
        v_ref = float(v_chi_raw[valid][-1])
        v_arr = np.where(
            r_arr > r_ref,
            v_ref * np.sqrt(r_ref / np.maximum(r_arr, 0.1)),
            v_chi_raw,
        )
    else:
        v_arr = v_chi_raw

    # Nyquist velocity limit: |v| < 0.8π/χ₀  (from leapfrog stability)
    v_nyq = 0.8 * np.pi / CHI0 * v_nyquist_fraction  # ≈ 0.121c

    # ── 3. Place orbiting bodies ──────────────────────────────────────────────
    body_omega: dict[str, float] = {}
    if verbose:
        print("Placing orbiting bodies…")

    for b in bodies:
        if b.orbital_radius <= 0:
            body_omega[b.name] = 0.0
            continue

        r = b.orbital_radius
        theta = b.orbital_phase
        bx = cx + r * np.cos(theta)
        by = cy + r * np.sin(theta)

        v_circ = float(np.interp(r, r_arr, v_arr)) * v_scale
        v_circ = min(v_circ, v_nyq)

        vx = -np.sin(theta) * v_circ
        vy = np.cos(theta) * v_circ
        omega = v_circ / r if r > 0 else 0.0
        body_omega[b.name] = omega

        if verbose:
            # omega is rad/lattice-time-unit; divide by dt to convert to steps
            T_steps = int(2 * np.pi / max(omega, 1e-12) / sim.config.dt) if omega > 0 else 0
            print(f"  {b.name:12s}  r={r:5.1f}  v={v_circ:.5f}  T\u2248{T_steps:,} steps")

        try:
            sim.place_soliton(
                (bx, by, cz),
                amplitude=b.amplitude,
                sigma=b.sigma,
                velocity=(vx, vy, 0.0),
            )
        except ValueError as exc:
            if verbose:
                print(f"    ⚠ {b.name}: {exc}")
                print("    Placing without velocity (orbit will be non-circular).")
            sim.place_soliton((bx, by, cz), amplitude=b.amplitude, sigma=b.sigma)
            body_omega[b.name] = 0.0

    # ── 4. Re-equilibrate with all bodies present ─────────────────────────────
    if equilibrate:
        if verbose:
            print("Re-equilibrating chi with all bodies placed…")
        sim.equilibrate()
        if verbose:
            chi_min = float(np.array(sim.chi).min())
            print(f"  χ_min = {chi_min:.4f}")

    # ── 5. Phase-velocity stabilisation (COMPLEX / FULL fields only) ──────────
    # For real fields E = φ cos(ωt), E² oscillates at 2ω driving chi waves.
    # For complex fields Ψ = φ exp(−iωt), |Ψ|² = φ² = const → no chi waves.
    # Set Ψ_prev(t=−Δt) = Ψ(t=0) × exp(+iω Δt) so that the leapfrog propagates
    # the ENTIRE complex field as exp(−iωt), making |Ψ|² exactly constant.
    # Only the *prev* buffers are modified; the t=0 state is preserved unchanged.
    if sim.config.field_level != FieldLevel.REAL:
        dt = sim.config.dt
        psi_r = np.array(sim.psi_real, dtype=np.float64)
        pi_now = sim.psi_imag
        psi_i = np.array(pi_now, dtype=np.float64) if pi_now is not None else np.zeros_like(psi_r)
        chi_a = np.array(sim.chi, dtype=np.float64)
        # Local omega ≈ chi value (GOV-01 dispersion, k→0 limit; carrier k adds ~0.1% correction)
        cos_phi = np.cos(chi_a * dt)
        sin_phi = np.sin(chi_a * dt)
        # Ψ_prev = Ψ × exp(+iωΔt) — only overwrite prev buffers, NOT current
        psi_r_prev = (psi_r * cos_phi - psi_i * sin_phi).astype(np.float32)
        psi_i_prev = (psi_r * sin_phi + psi_i * cos_phi).astype(np.float32)
        sim.set_psi_real_prev(psi_r_prev)
        sim.set_psi_imag_prev(psi_i_prev)
        if verbose:
            print("  Phase velocity encoded → |Ψ|² stabilised (complex soliton mode)")

    return body_omega
