"""38 - Coupled GOV-01/GOV-02 Energy Exchange

Tester-facing demonstration of energy exchange in the action-closed LFM core.

This standalone script embeds the small canonical LFM core it needs: constants,
the 19-point stencil, Hamiltonian ledger, and velocity-Verlet update. It does
not introduce a new force law or continuum target law. The wave register is the
full R2 channel register represented as six real components:

    (Re Psi_1, Im Psi_1, Re Psi_2, Im Psi_2, Re Psi_3, Im Psi_3)

Discrete substrate equations used by this demo
----------------------------------------------

Let

    D_t2 f_i^n = (f_i^{n+1} - 2 f_i^n + f_i^{n-1}) / dt^2
    Delta19    = canonical 19-point face-plus-edge lattice Laplacian
    N_2        = sum_a |Psi_a|^2
    B          = chi0 / kappa

The lattice is a 3-D periodic cubic grid. Periodic boundaries are used because
they preserve the closed-system Hamiltonian ledger for this exchange demo.

GOV-01:

    D_t2 Psi_a^n = c^2 Delta19 Psi_a^n - (chi^n)^2 Psi_a^n

GOV-02, v38 action-closed causal core:

    D_t2 chi^n =
        c^2 Delta19 chi^n
        - (kappa / chi0) chi^n (N_2^n - E0_sq)
        - (8 lambda_H / chi0^4) chi^n ((chi^n)^2 - chi0^2)^3

Velocity-Verlet/leapfrog form used by ``step_bare_lfm``:

    p_{n+1/2} = p_n + 0.5 dt F(q_n)
    q_{n+1}   = q_n + dt M^{-1} p_{n+1/2}
    p_{n+1}   = p_{n+1/2} + 0.5 dt F(q_{n+1})

For the chi register, p_chi = B chi_dot. This is the standard staggered
leapfrog form written with canonical momenta.

Hamiltonian ledger measured by this demo
----------------------------------------

    H_total = integral [
        0.5 |Psi_dot|^2
        + 0.5 c^2 |grad19 Psi|^2
        + B/2 chi_dot^2
        + B c^2/2 |grad19 chi|^2
        + 0.5 chi^2 (N_2 - E0_sq)
        + B lambda_H / chi0^4 (chi^2 - chi0^2)^4
    ] d^3x

Optional weak-current, color-classifier, cross-color, and flux-tube extension
terms are not activated here. The canonical documents mark those terms as
outside this bare Hamiltonian ledger unless a separate interacting-action audit
is supplied. This script is therefore an energy-exchange diagnostic for the
full channel register of the action-closed coupled core, not a force-emergence
or continuum-closure claim.

Accounting convention: ``wave_sector`` is the sum of the six positive GOV-01
component ledgers and includes the onsite coupling term 0.5*chi^2*N_2.
``chi_sector`` is total minus ``wave_sector``, i.e. the chi kinetic, gradient,
and flat-octic self-potential ledger. The plotted exchange is an accounting
exchange inside one conserved Hamiltonian, not two independently conserved
subsystem energies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

D = 3
D_ST = D + 1
CHI0 = float(3**D - 2**D)
KAPPA = 1.0 / (4**D - 1)
LAMBDA_H = D_ST / (2 * D_ST**2 - 1)
C_DEFAULT = 1.0
STENCIL_FACE_WEIGHT = 1.0 / 3.0
STENCIL_EDGE_WEIGHT = 1.0 / 6.0
STENCIL_CENTER_WEIGHT = -4.0

EQUATIONS_TEXT = """
LFM action-closed coupled core used in this demo

Register:
  R2 wave register Psi_a in C, a=1,2,3
  Numeric representation: six real components
  N_2 = sum_a |Psi_a|^2
  chi is a real substrate field

Discrete operators:
  D_t2 f_i^n = (f_i^{n+1} - 2 f_i^n + f_i^{n-1}) / dt^2
  Delta19 = canonical 19-point face-plus-edge lattice Laplacian
  B = chi0 / kappa
  Boundary = periodic cubic lattice

GOV-01:
  D_t2 Psi_a^n = c^2 Delta19 Psi_a^n - (chi_i^n)^2 Psi_a^n

GOV-02, v38 action-closed causal core:
  D_t2 chi_i^n =
      c^2 Delta19 chi_i^n
      - (kappa / chi0) chi_i^n (N_2,i^n - E0_sq)
      - (8 lambda_H / chi0^4) chi_i^n ((chi_i^n)^2 - chi0^2)^3

Hamiltonian:
  H_total = integral [
      0.5 |Psi_dot|^2
      + 0.5 c^2 |grad19 Psi|^2
      + B/2 chi_dot^2
      + B c^2/2 |grad19 chi|^2
      + 0.5 chi^2 (N_2 - E0_sq)
      + B lambda_H / chi0^4 (chi^2 - chi0^2)^4
  ] d^3x

Numerical step:
  p_{n+1/2} = p_n + 0.5 dt F(q_n)
  q_{n+1}   = q_n + dt M^{-1} p_{n+1/2}
  p_{n+1}   = p_{n+1/2} + 0.5 dt F(q_{n+1})

Optional extension terms are off in this Hamiltonian demo:
  GOV-02 weak-current source epsilon_w * j
  GOV-02 color-classifier source kappa_c * f_c * N_2
  GOV-02 flux-tube source kappa_tube * SCV
  GOV-01 cross-color term epsilon_cc * chi^2 * (Psi_a - mean(Psi))

Energy accounting:
  wave_sector = six GOV-01 component ledgers, including 0.5 chi^2 N_2
  chi_sector = total - wave_sector
""".strip()


Offset = tuple[int, int, int]


@dataclass(frozen=True)
class BareLFMParameters:
    """Canonical parameters for the v38 bare GOV-01/GOV-02 core."""

    chi0: float = CHI0
    kappa: float = KAPPA
    lambda_h: float = LAMBDA_H
    wave_speed: float = C_DEFAULT
    background_norm_sq: float = 0.0
    spacing: float = 1.0

    @property
    def chi_inertia(self) -> float:
        """Return B=chi0/kappa, fixed by the action-closed chi source."""
        return self.chi0 / self.kappa

    def __post_init__(self) -> None:
        positive = (
            self.chi0,
            self.kappa,
            self.lambda_h,
            self.wave_speed,
            self.spacing,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("bare LFM parameters must be positive and finite")
        if not np.isfinite(self.background_norm_sq) or self.background_norm_sq < 0.0:
            raise ValueError("background_norm_sq must be finite and nonnegative")


@dataclass(frozen=True)
class BareHamiltonRates:
    """Hamiltonian vector field for the bare LFM registers."""

    wave: np.ndarray
    wave_momentum: np.ndarray
    chi: np.ndarray
    chi_momentum: np.ndarray


@dataclass(frozen=True)
class BareLFMState:
    """Coordinate and momentum registers for the bare LFM system."""

    wave: np.ndarray
    wave_momentum: np.ndarray
    chi: np.ndarray
    chi_momentum: np.ndarray


def stencil_19_links() -> tuple[tuple[Offset, float], ...]:
    """Return oriented face and edge links for the canonical 19-point stencil."""
    unique: tuple[tuple[Offset, float], ...] = (
        ((1, 0, 0), STENCIL_FACE_WEIGHT),
        ((0, 1, 0), STENCIL_FACE_WEIGHT),
        ((0, 0, 1), STENCIL_FACE_WEIGHT),
        ((1, 1, 0), STENCIL_EDGE_WEIGHT),
        ((1, -1, 0), STENCIL_EDGE_WEIGHT),
        ((1, 0, 1), STENCIL_EDGE_WEIGHT),
        ((1, 0, -1), STENCIL_EDGE_WEIGHT),
        ((0, 1, 1), STENCIL_EDGE_WEIGHT),
        ((0, 1, -1), STENCIL_EDGE_WEIGHT),
    )
    links: list[tuple[Offset, float]] = []
    for offset, weight in unique:
        links.append((offset, weight))
        links.append(((-offset[0], -offset[1], -offset[2]), weight))
    return tuple(links)


def shift_scalar(values: np.ndarray, offset: Offset) -> np.ndarray:
    """Periodic shift for a scalar 3-D lattice field."""
    return np.roll(values, shift=offset, axis=(0, 1, 2))


def shift_components(values: np.ndarray, offset: Offset) -> np.ndarray:
    """Periodic shift for a component field shaped (components, N, N, N)."""
    return np.roll(values, shift=offset, axis=(1, 2, 3))


def laplacian_19pt(field: np.ndarray) -> np.ndarray:
    """Canonical 19-point face-plus-edge Laplacian on a periodic cubic grid."""
    faces = (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        + np.roll(field, 1, axis=2)
        + np.roll(field, -1, axis=2)
    )
    edges = (
        np.roll(np.roll(field, 1, axis=0), 1, axis=1)
        + np.roll(np.roll(field, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(field, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(field, -1, axis=0), -1, axis=1)
        + np.roll(np.roll(field, 1, axis=0), 1, axis=2)
        + np.roll(np.roll(field, 1, axis=0), -1, axis=2)
        + np.roll(np.roll(field, -1, axis=0), 1, axis=2)
        + np.roll(np.roll(field, -1, axis=0), -1, axis=2)
        + np.roll(np.roll(field, 1, axis=1), 1, axis=2)
        + np.roll(np.roll(field, 1, axis=1), -1, axis=2)
        + np.roll(np.roll(field, -1, axis=1), 1, axis=2)
        + np.roll(np.roll(field, -1, axis=1), -1, axis=2)
    )
    return (
        STENCIL_FACE_WEIGHT * faces
        + STENCIL_EDGE_WEIGHT * edges
        + STENCIL_CENTER_WEIGHT * field
    )


def laplacian_19pt_components(values: np.ndarray) -> np.ndarray:
    """Apply the canonical 19-point Laplacian to every wave component."""
    return np.stack([laplacian_19pt(component) for component in values], axis=0)


def as_components(values: np.ndarray, name: str) -> np.ndarray:
    """Normalize wave registers to shape (components, N, N, N)."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3:
        return array[np.newaxis, ...]
    if array.ndim == 4:
        return array
    raise ValueError(f"{name} must have shape (N,N,N) or (components,N,N,N)")


def validated_registers(
    wave: np.ndarray,
    wave_momentum: np.ndarray,
    chi: np.ndarray,
    chi_momentum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize the LFM state arrays."""
    wave_values = as_components(wave, "wave")
    wave_p = as_components(wave_momentum, "wave_momentum")
    chi_values = np.asarray(chi, dtype=np.float64)
    chi_p = np.asarray(chi_momentum, dtype=np.float64)
    if wave_values.shape != wave_p.shape:
        raise ValueError("wave and wave_momentum shapes must match")
    if chi_values.ndim != 3 or chi_p.ndim != 3:
        raise ValueError("chi and chi_momentum must have shape (N,N,N)")
    if chi_values.shape != chi_p.shape:
        raise ValueError("chi and chi_momentum shapes must match")
    if wave_values.shape[1:] != chi_values.shape:
        raise ValueError("wave and chi spatial shapes must match")
    return wave_values, wave_p, chi_values, chi_p


def chi_potential_density(chi: np.ndarray, parameters: BareLFMParameters) -> np.ndarray:
    """Flat-octic v38 chi self-potential density."""
    displacement = chi**2 - parameters.chi0**2
    return (
        parameters.chi_inertia
        * parameters.lambda_h
        * displacement**4
        / parameters.chi0**4
    )


def chi_potential_momentum_force(
    chi: np.ndarray,
    parameters: BareLFMParameters,
) -> np.ndarray:
    """Negative derivative of the flat-octic v38 chi self-potential."""
    displacement = chi**2 - parameters.chi0**2
    return (
        -8.0
        * parameters.chi_inertia
        * parameters.lambda_h
        * chi
        * displacement**3
        / parameters.chi0**4
    )


def gradient_site_density(
    values: np.ndarray,
    *,
    coefficient: float,
    components: bool,
) -> np.ndarray:
    """Endpoint-split 19-point gradient energy density."""
    spatial_shape = values.shape[1:] if components else values.shape
    density = np.zeros(spatial_shape, dtype=np.float64)
    shift = shift_components if components else shift_scalar
    for offset, weight in stencil_19_links():
        difference = shift(values, offset) - values
        squared = np.sum(difference**2, axis=0) if components else difference**2
        density += 0.25 * coefficient * weight * squared
    return density


def bare_site_energy(
    wave: np.ndarray,
    wave_momentum: np.ndarray,
    chi: np.ndarray,
    chi_momentum: np.ndarray,
    parameters: BareLFMParameters,
) -> np.ndarray:
    """Return endpoint-split site energy for the canonical bare Hamiltonian."""
    wave_values, wave_p, chi_values, chi_p = validated_registers(
        wave,
        wave_momentum,
        chi,
        chi_momentum,
    )
    norm_sq = np.sum(wave_values**2, axis=0)
    onsite = (
        0.5 * np.sum(wave_p**2, axis=0)
        + chi_p**2 / (2.0 * parameters.chi_inertia)
        + 0.5 * chi_values**2 * (norm_sq - parameters.background_norm_sq)
        + chi_potential_density(chi_values, parameters)
    )
    wave_gradient = gradient_site_density(
        wave_values,
        coefficient=parameters.wave_speed**2 / parameters.spacing**2,
        components=True,
    )
    chi_gradient = gradient_site_density(
        chi_values,
        coefficient=parameters.chi_inertia * parameters.wave_speed**2 / parameters.spacing**2,
        components=False,
    )
    return onsite + wave_gradient + chi_gradient


def bare_total_energy(state: BareLFMState, parameters: BareLFMParameters) -> float:
    """Return the total canonical bare Hamiltonian."""
    density = bare_site_energy(
        state.wave,
        state.wave_momentum,
        state.chi,
        state.chi_momentum,
        parameters,
    )
    return float(np.sum(density) * parameters.spacing**3)


def bare_hamilton_rates(
    wave: np.ndarray,
    wave_momentum: np.ndarray,
    chi: np.ndarray,
    chi_momentum: np.ndarray,
    parameters: BareLFMParameters,
) -> BareHamiltonRates:
    """Return Hamilton's equations for canonical GOV-01/GOV-02."""
    wave_values, wave_p, chi_values, chi_p = validated_registers(
        wave,
        wave_momentum,
        chi,
        chi_momentum,
    )
    norm_sq = np.sum(wave_values**2, axis=0)
    wave_rate = wave_p
    wave_momentum_rate = (
        parameters.wave_speed**2
        * laplacian_19pt_components(wave_values)
        / parameters.spacing**2
        - chi_values[np.newaxis, ...] ** 2 * wave_values
    )
    chi_rate = chi_p / parameters.chi_inertia
    chi_momentum_rate = (
        parameters.chi_inertia
        * parameters.wave_speed**2
        * laplacian_19pt(chi_values)
        / parameters.spacing**2
        - chi_values * (norm_sq - parameters.background_norm_sq)
        + chi_potential_momentum_force(chi_values, parameters)
    )
    return BareHamiltonRates(
        wave=wave_rate,
        wave_momentum=wave_momentum_rate,
        chi=chi_rate,
        chi_momentum=chi_momentum_rate,
    )


def step_bare_lfm(
    state: BareLFMState,
    dt: float,
    parameters: BareLFMParameters,
) -> BareLFMState:
    """Advance canonical GOV-01/GOV-02 with one velocity-Verlet step."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    wave, wave_p, chi, chi_p = validated_registers(
        state.wave,
        state.wave_momentum,
        state.chi,
        state.chi_momentum,
    )
    rates_0 = bare_hamilton_rates(wave, wave_p, chi, chi_p, parameters)
    half_wave_p = wave_p + 0.5 * dt * rates_0.wave_momentum
    half_chi_p = chi_p + 0.5 * dt * rates_0.chi_momentum
    next_wave = wave + dt * half_wave_p
    next_chi = chi + dt * half_chi_p / parameters.chi_inertia
    rates_1 = bare_hamilton_rates(
        next_wave,
        half_wave_p,
        next_chi,
        half_chi_p,
        parameters,
    )
    next_wave_p = half_wave_p + 0.5 * dt * rates_1.wave_momentum
    next_chi_p = half_chi_p + 0.5 * dt * rates_1.chi_momentum
    return BareLFMState(
        wave=next_wave,
        wave_momentum=next_wave_p,
        chi=next_chi,
        chi_momentum=next_chi_p,
    )


def wave_component_site_energy(
    state: BareLFMState,
    component: int,
    parameters: BareLFMParameters,
) -> np.ndarray:
    """Return the positive energy ledger assigned to one real wave component."""
    wave, wave_p, chi, _ = validated_registers(
        state.wave,
        state.wave_momentum,
        state.chi,
        state.chi_momentum,
    )
    if component < 0 or component >= wave.shape[0]:
        raise IndexError("component is outside the GOV-01 register")
    values = wave[component]
    momentum = wave_p[component]
    onsite = 0.5 * momentum**2 + 0.5 * chi**2 * values**2
    gradient = gradient_site_density(
        values,
        coefficient=parameters.wave_speed**2 / parameters.spacing**2,
        components=False,
    )
    return onsite + gradient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a clean LFM GOV-01/GOV-02 energy-exchange demo.",
    )
    parser.add_argument("--grid", type=int, default=20, help="Cubic grid size.")
    parser.add_argument("--steps", type=int, default=800, help="Verlet steps to run.")
    parser.add_argument("--dt", type=float, default=0.002, help="Timestep.")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=5,
        help="Write one energy row every N steps.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.42,
        help="Initial wave-packet amplitude.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.2,
        help="Initial Gaussian packet width in cells.",
    )
    parser.add_argument(
        "--chi-kick",
        type=float,
        default=0.020,
        help="Initial chi velocity amplitude. p_chi is B times this velocity.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "38_coupled_energy_exchange",
        help="Directory for CSV, JSON, equations, and optional plot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG plotting even if matplotlib is installed.",
    )
    return parser.parse_args()


def periodic_delta(axis: np.ndarray, center: float, size: int) -> np.ndarray:
    """Return minimum-image coordinate differences on a periodic grid."""
    half = 0.5 * float(size)
    return (axis - float(center) + half) % float(size) - half


def gaussian_blob(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    center: tuple[float, float, float],
    sigma: float,
) -> np.ndarray:
    """Return a periodic Gaussian envelope."""
    n = x.shape[0]
    dx = periodic_delta(x, center[0], n)
    dy = periodic_delta(y, center[1], n)
    dz = periodic_delta(z, center[2], n)
    r2 = dx * dx + dy * dy + dz * dz
    return np.exp(-0.5 * r2 / (sigma * sigma))


def make_initial_state(
    grid: int,
    amplitude: float,
    sigma: float,
    parameters: BareLFMParameters,
    chi_kick: float,
) -> BareLFMState:
    """Build a full six-real-component R2 wave state coupled to chi."""
    if grid < 8:
        raise ValueError("grid must be at least 8")
    if amplitude <= 0.0 or not np.isfinite(amplitude):
        raise ValueError("amplitude must be positive and finite")
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("sigma must be positive and finite")
    if not np.isfinite(chi_kick):
        raise ValueError("chi_kick must be finite")

    axis = np.arange(grid, dtype=np.float64)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    two_pi = 2.0 * math.pi
    k1 = two_pi / float(grid)
    k2 = 2.0 * two_pi / float(grid)
    omega1 = math.sqrt(parameters.chi0**2 + parameters.wave_speed**2 * k1**2)
    omega2 = math.sqrt(parameters.chi0**2 + parameters.wave_speed**2 * k2**2)

    center_1 = (0.36 * grid, 0.50 * grid, 0.50 * grid)
    center_2 = (0.64 * grid, 0.50 * grid, 0.50 * grid)
    center_3 = (0.50 * grid, 0.62 * grid, 0.50 * grid)
    blob_1 = gaussian_blob(x, y, z, center_1, sigma)
    blob_2 = gaussian_blob(x, y, z, center_2, sigma)
    blob_3 = gaussian_blob(x, y, z, center_3, 1.25 * sigma)

    phase_1 = k1 * periodic_delta(x, center_1[0], grid)
    phase_2 = k1 * periodic_delta(y, center_2[1], grid)
    phase_3 = k2 * periodic_delta(z, center_3[2], grid)

    wave = np.zeros((6, grid, grid, grid), dtype=np.float64)
    wave_momentum = np.zeros_like(wave)

    amp_1 = amplitude
    amp_2 = 0.85 * amplitude
    amp_3 = 0.55 * amplitude

    wave[0] = amp_1 * blob_1 * np.cos(phase_1)
    wave[1] = amp_1 * blob_1 * np.sin(phase_1)
    wave_momentum[0] = -omega1 * amp_1 * blob_1 * np.sin(phase_1)
    wave_momentum[1] = omega1 * amp_1 * blob_1 * np.cos(phase_1)

    wave[2] = amp_2 * blob_2 * np.cos(phase_2)
    wave[3] = amp_2 * blob_2 * np.sin(phase_2)
    wave_momentum[2] = -omega1 * amp_2 * blob_2 * np.sin(phase_2)
    wave_momentum[3] = omega1 * amp_2 * blob_2 * np.cos(phase_2)

    wave[4] = amp_3 * blob_3 * np.cos(phase_3)
    wave[5] = amp_3 * blob_3 * np.sin(phase_3)
    wave_momentum[4] = -omega2 * amp_3 * blob_3 * np.sin(phase_3)
    wave_momentum[5] = omega2 * amp_3 * blob_3 * np.cos(phase_3)

    norm_sq = np.sum(wave * wave, axis=0)
    source = norm_sq - float(np.mean(norm_sq))
    source_scale = max(float(np.max(np.abs(source))), 1.0e-30)

    # Start chi slightly below chi0 where the wave energy is concentrated,
    # then give it a small canonical velocity kick so exchange is visible
    # within a short tester run.
    chi = parameters.chi0 - 0.018 * source / source_scale
    chi_velocity = chi_kick * (blob_1 - blob_2)
    chi_momentum = parameters.chi_inertia * chi_velocity

    return BareLFMState(
        wave=wave,
        wave_momentum=wave_momentum,
        chi=chi,
        chi_momentum=chi_momentum,
    )


def sector_energies(
    state: BareLFMState,
    parameters: BareLFMParameters,
) -> dict[str, float]:
    """Return total, wave-sector, and chi-sector Hamiltonian ledgers."""
    total = bare_total_energy(state, parameters)
    component_energies = [
        float(np.sum(wave_component_site_energy(state, index, parameters)) * parameters.spacing**3)
        for index in range(state.wave.shape[0])
    ]
    wave_sector = float(sum(component_energies))
    chi_sector = float(total - wave_sector)
    return {
        "total": float(total),
        "wave_sector": wave_sector,
        "chi_sector": chi_sector,
        "wave_component_0": component_energies[0],
        "wave_component_1": component_energies[1],
        "wave_component_2": component_energies[2],
        "wave_component_3": component_energies[3],
        "wave_component_4": component_energies[4],
        "wave_component_5": component_energies[5],
        "chi_min": float(np.min(state.chi)),
        "chi_max": float(np.max(state.chi)),
        "chi_mean": float(np.mean(state.chi)),
        "wave_norm_sq": float(np.sum(state.wave * state.wave) * parameters.spacing**3),
    }


def record_row(
    rows: list[dict[str, float]],
    step: int,
    state: BareLFMState,
    parameters: BareLFMParameters,
    dt: float,
) -> None:
    row = {"step": float(step), "time": float(step) * dt}
    row.update(sector_energies(state, parameters))
    rows.append(row)


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(path: Path, rows: list[dict[str, float]], enabled: bool) -> Path | None:
    if not enabled:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    steps = np.asarray([row["step"] for row in rows], dtype=np.float64)
    total = np.asarray([row["total"] for row in rows], dtype=np.float64)
    wave = np.asarray([row["wave_sector"] for row in rows], dtype=np.float64)
    chi = np.asarray([row["chi_sector"] for row in rows], dtype=np.float64)

    fig, (ax_energy, ax_drift) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_energy.plot(steps, wave, label="Wave plus interaction ledger")
    ax_energy.plot(steps, chi, label="Chi self ledger")
    ax_energy.set_ylabel("Hamiltonian sector energy")
    ax_energy.legend(loc="best")
    ax_energy.grid(True, alpha=0.3)

    drift = (total - total[0]) / max(abs(total[0]), 1.0)
    ax_drift.plot(steps, drift, color="black", label="relative total drift")
    ax_drift.set_xlabel("Verlet step")
    ax_drift.set_ylabel("relative total drift")
    ax_drift.legend(loc="best")
    ax_drift.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def summarize(rows: list[dict[str, float]], args: argparse.Namespace) -> dict[str, float | int]:
    total = np.asarray([row["total"] for row in rows], dtype=np.float64)
    wave = np.asarray([row["wave_sector"] for row in rows], dtype=np.float64)
    chi = np.asarray([row["chi_sector"] for row in rows], dtype=np.float64)
    total0 = float(total[0])
    total_drift = float(np.max(np.abs(total - total0)) / max(abs(total0), 1.0))
    wave_range = float(np.max(wave) - np.min(wave))
    chi_range = float(np.max(chi) - np.min(chi))
    exchange_floor = 1.0e-30
    if np.std(wave) > exchange_floor and np.std(chi) > exchange_floor:
        wave_chi_corr = float(np.corrcoef(wave, chi)[0, 1])
    else:
        wave_chi_corr = float("nan")
    return {
        "grid": int(args.grid),
        "steps": int(args.steps),
        "dt": float(args.dt),
        "sample_every": int(args.sample_every),
        "initial_total_energy": total0,
        "final_total_energy": float(total[-1]),
        "max_relative_total_drift": total_drift,
        "wave_sector_range": wave_range,
        "chi_sector_range": chi_range,
        "wave_sector_range_pct_of_total": 100.0 * wave_range / max(abs(total0), 1.0),
        "chi_sector_range_pct_of_total": 100.0 * chi_range / max(abs(total0), 1.0),
        "wave_chi_correlation": wave_chi_corr,
        "initial_chi_min": float(rows[0]["chi_min"]),
        "final_chi_min": float(rows[-1]["chi_min"]),
    }


def run(
    args: argparse.Namespace,
) -> tuple[list[dict[str, float]], dict[str, float | int], Path | None]:
    if args.steps < 0:
        raise ValueError("steps must be nonnegative")
    if args.sample_every <= 0:
        raise ValueError("sample-every must be positive")
    if args.dt <= 0.0 or not np.isfinite(args.dt):
        raise ValueError("dt must be positive and finite")

    parameters = BareLFMParameters(
        chi0=CHI0,
        kappa=KAPPA,
        lambda_h=LAMBDA_H,
        wave_speed=1.0,
        background_norm_sq=0.0,
        spacing=1.0,
    )

    state = make_initial_state(
        grid=args.grid,
        amplitude=args.amplitude,
        sigma=args.sigma,
        parameters=parameters,
        chi_kick=args.chi_kick,
    )

    rows: list[dict[str, float]] = []
    record_row(rows, 0, state, parameters, args.dt)
    for step in range(1, args.steps + 1):
        state = step_bare_lfm(state, args.dt, parameters)
        if step % args.sample_every == 0 or step == args.steps:
            record_row(rows, step, state, parameters, args.dt)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    equations_path = args.output_dir / "governing_equations.txt"
    csv_path = args.output_dir / "energy_exchange.csv"
    summary_path = args.output_dir / "summary.json"
    plot_path = args.output_dir / "energy_exchange.png"

    equations_path.write_text(EQUATIONS_TEXT + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    summary = summarize(rows, args)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plotted = maybe_plot(plot_path, rows, enabled=not args.no_plot)

    print("LFM coupled energy exchange demo")
    print("=" * 40)
    print("Discrete layer: six real wave components plus chi on a periodic 3-D cubic lattice.")
    print("Readout layer: Hamiltonian sector ledger in external grid units.")
    print("Continuum layer: no continuum force or metric claim is made by this demo.")
    print()
    print("GOV-01/GOV-02 equations written to:")
    print(f"  {equations_path}")
    print("Energy table written to:")
    print(f"  {csv_path}")
    print("Summary written to:")
    print(f"  {summary_path}")
    if plotted is not None:
        print("Plot written to:")
        print(f"  {plotted}")
    elif not args.no_plot:
        print("Plot skipped: matplotlib is not installed.")
    print()
    print("Energy exchange summary:")
    print(f"  initial total energy        = {summary['initial_total_energy']:.12e}")
    print(f"  final total energy          = {summary['final_total_energy']:.12e}")
    print(f"  max relative total drift    = {summary['max_relative_total_drift']:.6e}")
    print(f"  wave+interaction range      = {summary['wave_sector_range']:.12e}")
    print(f"  chi self-ledger range       = {summary['chi_sector_range']:.12e}")
    print(
        "  wave sector range / total   = "
        f"{summary['wave_sector_range_pct_of_total']:.6e}%"
    )
    print(
        "  chi sector range / total    = "
        f"{summary['chi_sector_range_pct_of_total']:.6e}%"
    )
    print(f"  wave/chi sector correlation = {summary['wave_chi_correlation']:.6f}")

    return rows, summary, plotted


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
