"""Two-body centre dynamics in the macroscopic LIMIT-02 LFM regime."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from lfm.constants import CHI0
from lfm.fields.macroscopic import (
    Limit02BodyProfile,
    limit02_acceleration_from_profile,
)


def _two_body_accelerations(
    heavy: Limit02BodyProfile,
    light: Limit02BodyProfile,
    heavy_position: np.ndarray,
    light_position: np.ndarray,
    *,
    chi0: float,
    c: float,
) -> tuple[np.ndarray, np.ndarray]:
    heavy_from_light = heavy_position - light_position
    light_from_heavy = light_position - heavy_position
    heavy_acceleration = limit02_acceleration_from_profile(
        light,
        heavy_from_light,
        chi0=chi0,
        c=c,
    )
    light_acceleration = limit02_acceleration_from_profile(
        heavy,
        light_from_heavy,
        chi0=chi0,
        c=c,
    )
    return heavy_acceleration, light_acceleration


def _orbit_row(
    step: int,
    dt: float,
    heavy: Limit02BodyProfile,
    light: Limit02BodyProfile,
    heavy_position: np.ndarray,
    light_position: np.ndarray,
    heavy_velocity: np.ndarray,
    light_velocity: np.ndarray,
    heavy_acceleration: np.ndarray,
    light_acceleration: np.ndarray,
) -> dict[str, float | int]:
    relative = light_position - heavy_position
    separation = float(np.linalg.norm(relative))
    unit = relative / max(separation, 1.0e-30)
    total_mass = heavy.mass + light.mass
    center = (heavy.mass * heavy_position + light.mass * light_position) / total_mass
    momentum = heavy.mass * heavy_velocity + light.mass * light_velocity
    return {
        "step": int(step),
        "time": float(step * dt),
        "heavy_x": float(heavy_position[0]),
        "heavy_y": float(heavy_position[1]),
        "heavy_z": float(heavy_position[2]),
        "light_x": float(light_position[0]),
        "light_y": float(light_position[1]),
        "light_z": float(light_position[2]),
        "heavy_vx": float(heavy_velocity[0]),
        "heavy_vy": float(heavy_velocity[1]),
        "heavy_vz": float(heavy_velocity[2]),
        "light_vx": float(light_velocity[0]),
        "light_vy": float(light_velocity[1]),
        "light_vz": float(light_velocity[2]),
        "heavy_ax": float(heavy_acceleration[0]),
        "heavy_ay": float(heavy_acceleration[1]),
        "heavy_az": float(heavy_acceleration[2]),
        "light_ax": float(light_acceleration[0]),
        "light_ay": float(light_acceleration[1]),
        "light_az": float(light_acceleration[2]),
        "heavy_inward_acceleration": float(np.dot(heavy_acceleration, unit)),
        "light_inward_acceleration": float(np.dot(light_acceleration, -unit)),
        "separation": separation,
        "bearing_rad": float(math.atan2(relative[1], relative[0])),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "momentum_x": float(momentum[0]),
        "momentum_y": float(momentum[1]),
        "momentum_z": float(momentum[2]),
    }


def integrate_limit02_two_body(
    heavy: Limit02BodyProfile,
    light: Limit02BodyProfile,
    *,
    initial_separation: float,
    light_tangential_speed: float,
    dt: float,
    steps: int,
    sample_every: int,
    domain_center: tuple[float, float, float] | None = None,
    chi0: float = CHI0,
    c: float = 1.0,
) -> list[dict[str, float | int]]:
    """Integrate two centres using only sampled LIMIT-02 chi geometry."""
    if heavy.grid_size != light.grid_size:
        raise ValueError("body profiles must use the same grid")
    if initial_separation <= heavy.radius + light.radius:
        raise ValueError("initial bodies must not overlap")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")
    if abs(light_tangential_speed) >= c:
        raise ValueError("tangential speed must remain below c")

    if domain_center is None:
        midpoint = float(heavy.grid_size // 2)
        domain_center = (midpoint, midpoint, midpoint)
    center = np.asarray(domain_center, dtype=np.float64)
    total_mass = heavy.mass + light.mass
    separation_vector = np.asarray(
        [initial_separation, 0.0, 0.0],
        dtype=np.float64,
    )
    heavy_position = center - (light.mass / total_mass) * separation_vector
    light_position = center + (heavy.mass / total_mass) * separation_vector
    heavy_velocity = np.asarray(
        [
            0.0,
            -light_tangential_speed * light.mass / heavy.mass,
            0.0,
        ],
        dtype=np.float64,
    )
    light_velocity = np.asarray(
        [0.0, light_tangential_speed, 0.0],
        dtype=np.float64,
    )

    heavy_acceleration, light_acceleration = _two_body_accelerations(
        heavy,
        light,
        heavy_position,
        light_position,
        chi0=chi0,
        c=c,
    )
    rows = [
        _orbit_row(
            0,
            dt,
            heavy,
            light,
            heavy_position,
            light_position,
            heavy_velocity,
            light_velocity,
            heavy_acceleration,
            light_acceleration,
        )
    ]

    dt_sq_half = 0.5 * dt * dt
    for step in range(1, steps + 1):
        heavy_position = heavy_position + dt * heavy_velocity + dt_sq_half * heavy_acceleration
        light_position = light_position + dt * light_velocity + dt_sq_half * light_acceleration
        new_heavy_acceleration, new_light_acceleration = _two_body_accelerations(
            heavy,
            light,
            heavy_position,
            light_position,
            chi0=chi0,
            c=c,
        )
        heavy_velocity = heavy_velocity + 0.5 * dt * (heavy_acceleration + new_heavy_acceleration)
        light_velocity = light_velocity + 0.5 * dt * (light_acceleration + new_light_acceleration)
        heavy_acceleration = new_heavy_acceleration
        light_acceleration = new_light_acceleration

        if step % sample_every == 0 or step == steps:
            rows.append(
                _orbit_row(
                    step,
                    dt,
                    heavy,
                    light,
                    heavy_position,
                    light_position,
                    heavy_velocity,
                    light_velocity,
                    heavy_acceleration,
                    light_acceleration,
                )
            )
    return rows


def summarize_limit02_orbit(
    rows: list[dict[str, float | int]],
) -> dict[str, Any]:
    """Summarize a continuous reduced-LFM two-body trajectory."""
    if len(rows) < 2:
        raise ValueError("at least two trajectory rows are required")
    separations = np.asarray(
        [float(row["separation"]) for row in rows],
        dtype=np.float64,
    )
    bearings = np.unwrap(
        np.asarray(
            [float(row["bearing_rad"]) for row in rows],
            dtype=np.float64,
        )
    )
    increments = np.diff(bearings)
    sweep_deg = float(np.degrees(bearings[-1] - bearings[0]))
    nonzero = increments[np.abs(increments) > 1.0e-12]
    if nonzero.size == 0 or abs(sweep_deg) < 1.0e-12:
        direction_fraction = 0.0
    else:
        net_sign = 1.0 if sweep_deg > 0.0 else -1.0
        direction_fraction = float(np.mean(np.sign(nonzero) == net_sign))

    centers = np.asarray(
        [[row["center_x"], row["center_y"], row["center_z"]] for row in rows],
        dtype=np.float64,
    )
    momenta = np.asarray(
        [[row["momentum_x"], row["momentum_y"], row["momentum_z"]] for row in rows],
        dtype=np.float64,
    )
    center_drift = np.linalg.norm(centers - centers[0], axis=1)
    momentum_drift = np.linalg.norm(momenta - momenta[0], axis=1)
    initial = float(separations[0])
    return {
        "samples": len(rows),
        "initial_separation": initial,
        "final_separation": float(separations[-1]),
        "separation_change": float(separations[-1] - initial),
        "final_separation_ratio": float(separations[-1] / initial),
        "minimum_separation": float(np.min(separations)),
        "maximum_separation": float(np.max(separations)),
        "separation_spread_ratio": float((np.max(separations) - np.min(separations)) / initial),
        "angular_sweep_deg": sweep_deg,
        "orbit_direction_fraction": direction_fraction,
        "max_center_drift": float(np.max(center_drift)),
        "max_momentum_drift": float(np.max(momentum_drift)),
        "initial_heavy_inward_acceleration": float(rows[0]["heavy_inward_acceleration"]),
        "initial_light_inward_acceleration": float(rows[0]["light_inward_acceleration"]),
    }


def sweep_limit02_orbits(
    heavy: Limit02BodyProfile,
    light: Limit02BodyProfile,
    speeds: list[float],
    **integrator_kwargs: Any,
) -> list[dict[str, Any]]:
    """Run a declared tangential-speed screen in the reduced LFM model."""
    cases = []
    for speed in speeds:
        rows = integrate_limit02_two_body(
            heavy,
            light,
            light_tangential_speed=float(speed),
            **integrator_kwargs,
        )
        cases.append(
            {
                "light_tangential_speed": float(speed),
                "rows": rows,
                "summary": summarize_limit02_orbit(rows),
            }
        )
    return cases


__all__ = [
    "integrate_limit02_two_body",
    "summarize_limit02_orbit",
    "sweep_limit02_orbits",
]
