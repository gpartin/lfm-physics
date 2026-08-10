"""Reusable electromagnetic closure diagnostics for LFM experiments.

This module contains diagnostic readouts and small live probes. It does not
promote a canonical equation change and it does not insert a Coulomb, Maxwell,
Lorentz, Poisson, or nonlocal Green-function update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from lfm import BoundaryType, FieldLevel, Precision, Simulation, SimulationConfig
from lfm.analysis.phase import canonical_charge_density
from lfm.config import ChiPotentialModel
from lfm.constants import C_DEFAULT, CHI0, DT_DEFAULT, KAPPA, LAMBDA_H
from lfm.core.stencils import gradient_19pt


def _snapshot_array(snapshot: dict[str, object], key: str) -> np.ndarray:
    return cast("np.ndarray", snapshot[key])


def rms(values: np.ndarray) -> float:
    """Return root-mean-square magnitude."""

    arr = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


def l2_norm(values: np.ndarray) -> float:
    """Return L2 norm."""

    arr = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.sum(arr * arr)))


def fit_power_law(xs: list[float], ys: list[float]) -> dict[str, float | None]:
    """Fit y = A*x**slope on positive samples."""

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if len(x) < 3 or np.any(x <= 0.0) or np.any(y <= 0.0):
        return {"slope": None, "intercept": None, "r_squared": 0.0}
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    pred = slope * log_x + intercept
    residual = float(np.sum((log_y - pred) ** 2))
    total = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def divergence_19(vector: np.ndarray, *, dx: float = 1.0) -> np.ndarray:
    """Return site-centered divergence of a three-component vector field."""

    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape[0] != 3 or arr.ndim != 4:
        raise ValueError("vector must have shape (3,nx,ny,nz)")
    grad_x = gradient_19pt(arr[0], dx=dx)
    grad_y = gradient_19pt(arr[1], dx=dx)
    grad_z = gradient_19pt(arr[2], dx=dx)
    return grad_x[0] + grad_y[1] + grad_z[2]


def curl_19(vector: np.ndarray, *, dx: float = 1.0) -> np.ndarray:
    """Return site-centered curl of a three-component vector field."""

    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape[0] != 3 or arr.ndim != 4:
        raise ValueError("vector must have shape (3,nx,ny,nz)")
    grad_x = gradient_19pt(arr[0], dx=dx)
    grad_y = gradient_19pt(arr[1], dx=dx)
    grad_z = gradient_19pt(arr[2], dx=dx)
    out = np.empty_like(arr)
    out[0] = grad_z[1] - grad_y[2]
    out[1] = grad_x[2] - grad_z[0]
    out[2] = grad_y[0] - grad_x[1]
    return out


def clock_shear_acceleration(vector: np.ndarray, *, dx: float = 1.0) -> np.ndarray:
    """Return -curl(curl(A)) for the clock-shear carrier."""

    return -curl_19(curl_19(vector, dx=dx), dx=dx)


def clock_shear_energy_density(a_field: np.ndarray, e_field: np.ndarray) -> np.ndarray:
    """Return local clock-shear field energy density."""

    b_field = curl_19(a_field)
    return 0.5 * (np.sum(np.asarray(e_field) ** 2, axis=0) + np.sum(b_field * b_field, axis=0))


def total_clock_shear_energy(a_field: np.ndarray, e_field: np.ndarray) -> float:
    """Return total clock-shear field energy."""

    return float(np.sum(clock_shear_energy_density(a_field, e_field)))


def color_noether_charge_density(
    psi_real: np.ndarray,
    psi_imag: np.ndarray,
    psi_real_prev: np.ndarray,
    psi_imag_prev: np.ndarray,
    *,
    dt: float,
) -> np.ndarray:
    """Return summed temporal Noether charge density for color fields."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    real = np.asarray(psi_real, dtype=np.float64)
    imag = np.asarray(psi_imag, dtype=np.float64)
    real_prev = np.asarray(psi_real_prev, dtype=np.float64)
    imag_prev = np.asarray(psi_imag_prev, dtype=np.float64)
    if real.shape != imag.shape or real.shape != real_prev.shape:
        raise ValueError("color field arrays must have matching shapes")
    if real.ndim == 3:
        real = real[None, ...]
        imag = imag[None, ...]
        real_prev = real_prev[None, ...]
        imag_prev = imag_prev[None, ...]
    if real.ndim != 4:
        raise ValueError("color field arrays must have shape (components,nx,ny,nz)")
    momentum_real = (real - real_prev) / dt
    momentum_imag = (imag - imag_prev) / dt
    charge = np.zeros(real.shape[1:], dtype=np.float64)
    for component in range(real.shape[0]):
        charge += canonical_charge_density(
            real[component],
            imag[component],
            momentum_real[component],
            momentum_imag[component],
        )
    return charge


def color_noether_spatial_current(
    psi_real: np.ndarray,
    psi_imag: np.ndarray,
    *,
    c: float = C_DEFAULT,
    dx: float = 1.0,
) -> np.ndarray:
    """Return summed spatial Noether current for color fields."""

    real = np.asarray(psi_real, dtype=np.float64)
    imag = np.asarray(psi_imag, dtype=np.float64)
    if real.shape != imag.shape:
        raise ValueError("psi_real and psi_imag must have matching shapes")
    if real.ndim == 3:
        real = real[None, ...]
        imag = imag[None, ...]
    if real.ndim != 4:
        raise ValueError("color fields must have shape (components,nx,ny,nz)")
    current = np.zeros((3, *real.shape[1:]), dtype=np.float64)
    c2 = c * c
    for component in range(real.shape[0]):
        grad_real = gradient_19pt(real[component], dx=dx)
        grad_imag = gradient_19pt(imag[component], dx=dx)
        for axis in range(3):
            current[axis] += -c2 * (
                real[component] * grad_imag[axis] - imag[component] * grad_real[axis]
            )
    return current


@dataclass(frozen=True)
class PacketSpec:
    """Prepared charge packet specification."""

    center: tuple[float, float, float]
    charge_sign: int
    component: int = 0
    phase: float = 0.0


def periodic_displacement_grid(
    size: int,
    center: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return periodic displacement components and radius from center."""

    coords = np.indices((size, size, size), dtype=np.float64)
    displacements = []
    for axis, value in enumerate(center):
        raw = coords[axis] - float(value)
        raw = (raw + 0.5 * size) % size - 0.5 * size
        displacements.append(raw)
    radius_sq = sum(item * item for item in displacements)
    return displacements[0], displacements[1], displacements[2], np.sqrt(radius_sq)


def make_prepared_color_charge_state(
    *,
    size: int,
    packets: tuple[PacketSpec, ...],
    sigma: float,
    amplitude: float,
    omega: float,
    dt: float,
    n_colors: int = 3,
) -> dict[str, np.ndarray]:
    """Create a prepared COLOR wave state with signed temporal charge.

    The state is a prepared wave packet control, not an electron candidate.
    Charge sign is encoded only through the leapfrog phase rotation.
    """

    if sigma <= 0.0 or amplitude <= 0.0 or omega <= 0.0 or dt <= 0.0:
        raise ValueError("sigma, amplitude, omega, and dt must be positive")
    psi_real = np.zeros((n_colors, size, size, size), dtype=np.float64)
    psi_imag = np.zeros_like(psi_real)
    psi_real_prev = np.zeros_like(psi_real)
    psi_imag_prev = np.zeros_like(psi_real)
    for packet in packets:
        if packet.charge_sign not in (-1, 1):
            raise ValueError("charge_sign must be -1 or +1")
        if not 0 <= packet.component < n_colors:
            raise ValueError("packet component out of range")
        _dx, _dy, _dz, radius = periodic_displacement_grid(size, packet.center)
        envelope = amplitude * np.exp(-0.5 * (radius / sigma) ** 2)
        phase_now = packet.phase
        phase_prev = phase_now - packet.charge_sign * omega * dt
        psi_real[packet.component] += envelope * np.cos(phase_now)
        psi_imag[packet.component] += envelope * np.sin(phase_now)
        psi_real_prev[packet.component] += envelope * np.cos(phase_prev)
        psi_imag_prev[packet.component] += envelope * np.sin(phase_prev)
    return {
        "psi_real": psi_real,
        "psi_imag": psi_imag,
        "psi_real_prev": psi_real_prev,
        "psi_imag_prev": psi_imag_prev,
    }


def charge_centroid(
    charge_density: np.ndarray,
    center_hint: tuple[float, float, float],
    *,
    sign: int,
    radius: float,
) -> dict[str, Any]:
    """Return local signed-charge centroid near a center hint."""

    rho = np.asarray(charge_density, dtype=np.float64)
    dx_grid, dy_grid, dz_grid, dist = periodic_displacement_grid(rho.shape[0], center_hint)
    weight = np.maximum(rho, 0.0) if sign > 0 else np.maximum(-rho, 0.0)
    mask = dist <= radius
    weighted = weight * mask
    total = float(np.sum(weighted))
    if total <= 1.0e-300:
        return {
            "ok": False,
            "charge": 0.0,
            "center": list(center_hint),
            "local_displacement": [0.0, 0.0, 0.0],
        }
    disp = [
        float(np.sum(weighted * dx_grid) / total),
        float(np.sum(weighted * dy_grid) / total),
        float(np.sum(weighted * dz_grid) / total),
    ]
    center = [float((center_hint[axis] + disp[axis]) % rho.shape[0]) for axis in range(3)]
    signed_charge = total if sign > 0 else -total
    return {
        "ok": True,
        "charge": float(signed_charge),
        "center": center,
        "local_displacement": disp,
    }


def run_prepared_charge_pair_probe(
    *,
    size: int = 32,
    steps: int = 80,
    dt: float = DT_DEFAULT,
    sigma: float = 2.6,
    amplitude: float = 0.030,
    separation: float = 10.0,
    signs: tuple[int, int] = (1, 1),
    lambda_self: float = LAMBDA_H,
    kappa: float = KAPPA,
    epsilon_w: float = 0.0,
    use_gravity_recovery: bool = False,
) -> dict[str, Any]:
    """Run two prepared Noether-charge packets under full COLOR LFM.

    This diagnostic tests whether the current local equations make the signed
    charge channel dynamically active. It is not an electron gate.
    """

    if signs[0] not in (-1, 1) or signs[1] not in (-1, 1):
        raise ValueError("signs must contain only -1 or +1")
    center_a = (0.5 * size - 0.5 * separation, 0.5 * size, 0.5 * size)
    center_b = (0.5 * size + 0.5 * separation, 0.5 * size, 0.5 * size)
    config = SimulationConfig(
        grid_size=size,
        dt=dt,
        field_level=FieldLevel.COLOR,
        n_colors=3,
        boundary_type=BoundaryType.PERIODIC,
        precision=Precision.FLOAT64,
        lambda_self=lambda_self,
        kappa=kappa,
        epsilon_w=epsilon_w,
        enable_chi_floor=False,
        report_interval=max(1, steps + 1),
    )
    sim = Simulation(config, backend="cpu")
    state = make_prepared_color_charge_state(
        size=size,
        packets=(
            PacketSpec(center=center_a, charge_sign=signs[0], component=0),
            PacketSpec(center=center_b, charge_sign=signs[1], component=0),
        ),
        sigma=sigma,
        amplitude=amplitude,
        omega=CHI0,
        dt=dt,
    )
    sim.set_psi_real(state["psi_real"])
    sim.set_psi_imag(state["psi_imag"])
    sim.set_psi_real_prev(state["psi_real_prev"])
    sim.set_psi_imag_prev(state["psi_imag_prev"])
    chi = np.full((size, size, size), CHI0, dtype=np.float64)
    sim.set_chi(chi)
    sim.set_chi_prev(chi.copy())

    def snapshot(label: str) -> dict[str, Any]:
        snap = sim.phase_space_snapshot()
        psi_real = _snapshot_array(snap, "psi_real")
        psi_imag = _snapshot_array(snap, "psi_imag")
        psi_real_prev = _snapshot_array(snap, "psi_real_prev")
        psi_imag_prev = _snapshot_array(snap, "psi_imag_prev")
        chi_snapshot = _snapshot_array(snap, "chi")
        rho = color_noether_charge_density(
            psi_real,
            psi_imag,
            psi_real_prev,
            psi_imag_prev,
            dt=dt,
        )
        current = color_noether_spatial_current(
            psi_real,
            psi_imag,
            c=C_DEFAULT,
        )
        centroid_radius = min(2.0 * sigma, 0.40 * separation)
        ca = charge_centroid(rho, center_a, sign=signs[0], radius=centroid_radius)
        cb = charge_centroid(rho, center_b, sign=signs[1], radius=centroid_radius)
        sep_vec = [
            ((cb["center"][axis] - ca["center"][axis] + 0.5 * size) % size) - 0.5 * size
            for axis in range(3)
        ]
        separation_now = float(np.sqrt(sum(value * value for value in sep_vec)))
        return {
            "label": label,
            "step": int(sim.step),
            "charge_total": float(np.sum(rho)),
            "charge_abs": float(np.sum(np.abs(rho))),
            "current_rms": rms(current),
            "centroid_radius": float(centroid_radius),
            "packet_a": ca,
            "packet_b": cb,
            "separation": separation_now,
            "chi_min": float(np.min(chi_snapshot)),
            "chi_max": float(np.max(chi_snapshot)),
            "psi_norm": float(np.sqrt(np.sum(psi_real**2 + psi_imag**2))),
        }

    start = snapshot("start")
    if use_gravity_recovery:
        sim.run_gravity_recovery(
            steps,
            ChiPotentialModel.FLAT_OCTIC,
            freeze_psi=False,
        )
    else:
        sim.run(steps, record_metrics=False)
    end = snapshot("end")
    return {
        "settings": {
            "size": size,
            "steps": steps,
            "dt": dt,
            "sigma": sigma,
            "amplitude": amplitude,
            "separation": separation,
            "signs": list(signs),
            "lambda_self": lambda_self,
            "kappa": kappa,
            "epsilon_w": epsilon_w,
            "use_gravity_recovery": use_gravity_recovery,
            "field_level": "COLOR",
            "boundary_type": "PERIODIC",
        },
        "start": start,
        "end": end,
        "delta_separation": float(end["separation"] - start["separation"]),
        "charge_abs_retention": float(end["charge_abs"] / max(start["charge_abs"], 1.0e-300)),
    }


def run_signed_pair_comparison(**kwargs: Any) -> dict[str, Any]:
    """Compare same-charge and opposite-charge prepared pair histories."""

    same = run_prepared_charge_pair_probe(signs=(1, 1), **kwargs)
    opposite = run_prepared_charge_pair_probe(signs=(1, -1), **kwargs)
    same_delta = same["delta_separation"]
    opposite_delta = opposite["delta_separation"]
    signed_split = same_delta - opposite_delta
    scale = max(abs(same_delta), abs(opposite_delta), 1.0e-300)
    return {
        "same_charge": same,
        "opposite_charge": opposite,
        "signed_split": float(signed_split),
        "signed_split_relative_to_motion": float(abs(signed_split) / scale),
        "charge_abs_retention_min": float(
            min(same["charge_abs_retention"], opposite["charge_abs_retention"])
        ),
    }
