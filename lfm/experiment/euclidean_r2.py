"""Local Euclidean diagnostic sampler for the unchanged full-R2 LFM action.

This module is an analysis entrance tool, not a replacement for live leapfrog
evolution. It samples the Euclidean continuation of the local GOV-01/GOV-02
Hamiltonian with the canonical 19-point spatial quadratic form. No gauge link,
Maxwell term, source, or nonlocal update is introduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from lfm.constants import CHI0, KAPPA, LAMBDA_H
from lfm.core.stencils import eigenvalue_19pt

EuclideanRestoringModel = Literal["canonical_quartic", "flat_octic"]


@dataclass(frozen=True)
class EuclideanR2Config:
    """Configuration for one finite periodic full-R2 Euclidean chain."""

    linear_size: int
    model: EuclideanRestoringModel
    seed: int
    psi_step: float = 0.055
    phi_step_quartic: float = 0.075
    phi_step_flat_octic: float = 0.55

    def __post_init__(self) -> None:
        if self.linear_size < 4 or self.linear_size % 2:
            raise ValueError("linear_size must be even and at least four")
        if self.model not in ("canonical_quartic", "flat_octic"):
            raise ValueError("unsupported restoring model")
        if self.psi_step <= 0.0:
            raise ValueError("psi_step must be positive")


def _spatial_offsets() -> tuple[tuple[tuple[int, int, int], float], ...]:
    values: list[tuple[tuple[int, int, int], float]] = []
    for axis in range(3):
        for sign in (-1, 1):
            offset = [0, 0, 0]
            offset[axis] = sign
            values.append((tuple(offset), 1.0 / 3.0))
    for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
        for sign_a in (-1, 1):
            for sign_b in (-1, 1):
                offset = [0, 0, 0]
                offset[axis_a] = sign_a
                offset[axis_b] = sign_b
                values.append((tuple(offset), 1.0 / 6.0))
    return tuple(values)


SPATIAL_OFFSETS = _spatial_offsets()
INCIDENT_WEIGHT = 6.0
B_CHI = CHI0 / KAPPA


def integrated_autocorrelation(values: np.ndarray) -> float:
    """Return a positive-window integrated autocorrelation estimate."""

    series = np.asarray(values, dtype=float)
    series = series - np.mean(series)
    variance = float(np.mean(series**2))
    if variance <= 0.0:
        return 0.5
    tau = 0.5
    maximum_lag = min(len(series) // 3, 50)
    for lag in range(1, maximum_lag + 1):
        correlation = float(np.mean(series[:-lag] * series[lag:]) / variance)
        if correlation <= 0.0:
            break
        tau += correlation
    return tau


def frozen_chi_gaussian_real_variance(size: int) -> float:
    """Exact finite-volume real-component variance with chi frozen at chi0."""

    points = 2.0 * math.pi * np.arange(size, dtype=float) / size
    p0, kx, ky, kz = np.meshgrid(points, points, points, points, indexing="ij")
    time_stiffness = 4.0 * np.sin(0.5 * p0) ** 2
    spatial_stiffness = -eigenvalue_19pt(kx, ky, kz)
    return float(np.mean(1.0 / (time_stiffness + spatial_stiffness + CHI0**2)))


class EuclideanR2Sampler:
    """Checkerboard Metropolis sampler for the local full-R2 Euclidean action."""

    def __init__(self, config: EuclideanR2Config) -> None:
        self.config = config
        size = config.linear_size
        self.rng = np.random.default_rng(config.seed)
        self.psi = np.zeros((size, size, size, size, 3), dtype=np.complex128)
        self.phi = np.zeros((size, size, size, size), dtype=np.float64)
        self.psi_step = float(config.psi_step)
        self.phi_step = float(
            config.phi_step_quartic
            if config.model == "canonical_quartic"
            else config.phi_step_flat_octic
        )
        coordinates = np.indices((size, size, size, size))
        masks = []
        for color in range(16):
            mask = np.ones((size, size, size, size), dtype=bool)
            for axis in range(4):
                mask &= (coordinates[axis] & 1) == ((color >> axis) & 1)
            masks.append(mask)
        self._masks = tuple(masks)

    @property
    def chi(self) -> np.ndarray:
        """Return the physical chi field represented by the rescaled sampler field."""

        return CHI0 + self.phi / math.sqrt(B_CHI)

    def _neighbor_sum(self, field: np.ndarray) -> np.ndarray:
        value = np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
        for offset, weight in SPATIAL_OFFSETS:
            value = value + weight * np.roll(
                field,
                shift=offset,
                axis=(1, 2, 3),
            )
        return value

    def _chi_potential(self, chi: np.ndarray) -> np.ndarray:
        delta = chi**2 - CHI0**2
        if self.config.model == "canonical_quartic":
            return B_CHI * LAMBDA_H * delta**2
        return B_CHI * LAMBDA_H * delta**4 / CHI0**4

    def _update_psi(self, mask: np.ndarray) -> tuple[int, int]:
        old = self.psi[mask]
        delta = self.psi_step * (
            self.rng.normal(size=old.shape)
            + 1j * self.rng.normal(size=old.shape)
        ) / math.sqrt(2.0)
        new = old + delta
        summed_neighbors = self._neighbor_sum(self.psi)[mask]
        old_norm = np.sum(np.abs(old) ** 2, axis=-1)
        new_norm = np.sum(np.abs(new) ** 2, axis=-1)
        link_delta = 0.5 * INCIDENT_WEIGHT * (new_norm - old_norm)
        link_delta -= np.real(np.sum(np.conj(delta) * summed_neighbors, axis=-1))
        chi = self.chi[mask]
        action_delta = link_delta + 0.5 * chi**2 * (new_norm - old_norm)
        accepted = np.log(self.rng.random(size=action_delta.shape)) < -action_delta
        old[accepted] = new[accepted]
        self.psi[mask] = old
        return int(np.sum(accepted)), int(accepted.size)

    def _update_phi(self, mask: np.ndarray) -> tuple[int, int]:
        old = self.phi[mask]
        new = old + self.phi_step * self.rng.normal(size=old.shape)
        delta = new - old
        summed_neighbors = self._neighbor_sum(self.phi)[mask]
        link_delta = 0.5 * INCIDENT_WEIGHT * (new**2 - old**2)
        link_delta -= delta * summed_neighbors
        density = np.sum(np.abs(self.psi[mask]) ** 2, axis=-1)
        old_chi = CHI0 + old / math.sqrt(B_CHI)
        new_chi = CHI0 + new / math.sqrt(B_CHI)
        action_delta = link_delta
        action_delta += 0.5 * density * (new_chi**2 - old_chi**2)
        action_delta += self._chi_potential(new_chi) - self._chi_potential(old_chi)
        accepted = np.log(self.rng.random(size=action_delta.shape)) < -action_delta
        old[accepted] = new[accepted]
        self.phi[mask] = old
        return int(np.sum(accepted)), int(accepted.size)

    def sweep(self) -> dict[str, int]:
        """Perform one local 16-color detailed-balance sweep."""

        totals = {
            "psi_accept": 0,
            "psi_total": 0,
            "phi_accept": 0,
            "phi_total": 0,
        }
        for mask in self._masks:
            accepted, total = self._update_psi(mask)
            totals["psi_accept"] += accepted
            totals["psi_total"] += total
            accepted, total = self._update_phi(mask)
            totals["phi_accept"] += accepted
            totals["phi_total"] += total
        return totals

    def warmup(self, sweeps: int, tune_every: int = 20) -> None:
        """Warm the chain while adapting proposal widths; production is untouched."""

        if sweeps < 1 or tune_every < 1:
            raise ValueError("warmup settings must be positive")
        block = {
            "psi_accept": 0,
            "psi_total": 0,
            "phi_accept": 0,
            "phi_total": 0,
        }
        for sweep_index in range(sweeps):
            result = self.sweep()
            for key, value in result.items():
                block[key] += value
            if (sweep_index + 1) % tune_every:
                continue
            psi_rate = block["psi_accept"] / block["psi_total"]
            phi_rate = block["phi_accept"] / block["phi_total"]
            if psi_rate > 0.58:
                self.psi_step *= 1.10
            elif psi_rate < 0.42:
                self.psi_step *= 0.90
            if phi_rate > 0.58:
                self.phi_step *= 1.10
            elif phi_rate < 0.42:
                self.phi_step *= 0.90
            for key in block:
                block[key] = 0

    def thinned_sweep(self, count: int) -> dict[str, int]:
        """Perform production sweeps and return combined acceptance counts."""

        if count < 1:
            raise ValueError("count must be positive")
        totals = {
            "psi_accept": 0,
            "psi_total": 0,
            "phi_accept": 0,
            "phi_total": 0,
        }
        for _ in range(count):
            result = self.sweep()
            for key, value in result.items():
                totals[key] += value
        return totals


__all__ = [
    "EuclideanR2Config",
    "EuclideanR2Sampler",
    "EuclideanRestoringModel",
    "frozen_chi_gaussian_real_variance",
    "integrated_autocorrelation",
]
