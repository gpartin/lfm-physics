"""Macroscopic density bodies in the quasi-static LIMIT-02 regime.

These helpers intentionally discard microscopic phase and color registers.
They represent rigid extended bodies by real density fields, solve the
19-point weak-field LIMIT-02 equation, and sample the resulting chi geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lfm.constants import CHI0, KAPPA
from lfm.core.stencils import gradient_19pt
from lfm.fields.equilibrium import equilibrate_chi_19pt

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Limit02BodyProfile:
    """A rigid spherical density and its isolated LIMIT-02 chi geometry."""

    grid_size: int
    center: tuple[float, float, float]
    radius: float
    mass: float
    density: NDArray[np.float64]
    chi_delta: NDArray[np.float64]
    gradient_x: NDArray[np.float64]
    gradient_y: NDArray[np.float64]
    gradient_z: NDArray[np.float64]


def smooth_spherical_density(
    grid_size: int,
    center: tuple[float, float, float],
    radius: float,
    mass: float,
) -> NDArray[np.float64]:
    """Return a compact smooth spherical density normalized to ``mass``.

    The unnormalized radial profile is ``(1 - r^2 / R^2)^2`` inside
    ``r < R`` and zero outside. Periodic minimum-image distances are used so
    the same helper can translate a body across a periodic LIMIT-02 domain.
    """
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if radius >= grid_size / 2.0:
        raise ValueError("radius must be smaller than half the grid")
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    if len(center) != 3:
        raise ValueError("center must contain three coordinates")

    axes = []
    for coordinate in center:
        delta = np.arange(grid_size, dtype=np.float64) - float(coordinate)
        delta = (delta + grid_size / 2.0) % grid_size - grid_size / 2.0
        axes.append(delta)
    dx, dy, dz = np.meshgrid(*axes, indexing="ij")
    radius_sq = dx * dx + dy * dy + dz * dz
    scaled = radius_sq / (radius * radius)
    density = np.where(scaled < 1.0, (1.0 - scaled) ** 2, 0.0)
    normalization = float(np.sum(density))
    if normalization <= 0.0:
        raise ValueError("radius is too small to resolve a nonzero body")
    density *= mass / normalization
    return density.astype(np.float64, copy=False)


def build_limit02_body_profile(
    grid_size: int,
    radius: float,
    mass: float,
    *,
    center: tuple[float, float, float] | None = None,
    chi0: float = CHI0,
    kappa: float = KAPPA,
    dx: float = 1.0,
) -> Limit02BodyProfile:
    """Build one spherical body's isolated 19-point LIMIT-02 field profile."""
    if center is None:
        midpoint = float(grid_size // 2)
        center = (midpoint, midpoint, midpoint)
    density = smooth_spherical_density(grid_size, center, radius, mass)
    chi = equilibrate_chi_19pt(
        density,
        chi0=chi0,
        kappa=kappa,
    ).astype(np.float64, copy=False)
    chi_delta = chi - float(chi0)
    gradient_x, gradient_y, gradient_z = gradient_19pt(
        chi_delta,
        dx=dx,
    )
    return Limit02BodyProfile(
        grid_size=grid_size,
        center=tuple(float(value) for value in center),
        radius=float(radius),
        mass=float(mass),
        density=density,
        chi_delta=chi_delta,
        gradient_x=np.asarray(gradient_x, dtype=np.float64),
        gradient_y=np.asarray(gradient_y, dtype=np.float64),
        gradient_z=np.asarray(gradient_z, dtype=np.float64),
    )


def periodic_trilinear_sample(
    field: NDArray[np.floating],
    point: tuple[float, float, float] | NDArray[np.floating],
) -> float:
    """Trilinearly sample a periodic 3-D scalar field."""
    values = np.asarray(field)
    if values.ndim != 3 or not (
        values.shape[0] == values.shape[1] == values.shape[2]
    ):
        raise ValueError("field must be a cubic 3-D array")
    coordinates = np.asarray(point, dtype=np.float64)
    if coordinates.shape != (3,):
        raise ValueError("point must contain three coordinates")

    size = values.shape[0]
    wrapped = np.mod(coordinates, float(size))
    lower = np.floor(wrapped).astype(int)
    fraction = wrapped - lower
    upper = (lower + 1) % size

    result = 0.0
    for bx in (0, 1):
        ix = lower[0] if bx == 0 else upper[0]
        wx = (1.0 - fraction[0]) if bx == 0 else fraction[0]
        for by in (0, 1):
            iy = lower[1] if by == 0 else upper[1]
            wy = (1.0 - fraction[1]) if by == 0 else fraction[1]
            for bz in (0, 1):
                iz = lower[2] if bz == 0 else upper[2]
                wz = (1.0 - fraction[2]) if bz == 0 else fraction[2]
                result += wx * wy * wz * float(values[ix, iy, iz])
    return float(result)


def limit02_acceleration_from_profile(
    source: Limit02BodyProfile,
    displacement_from_source: tuple[float, float, float] | NDArray[np.floating],
    *,
    chi0: float = CHI0,
    c: float = 1.0,
) -> NDArray[np.float64]:
    """Sample the weak-field WKB acceleration from a source's chi profile.

    ``displacement_from_source`` points from the source centre to the target.
    No analytic radial or inverse-square force is used.
    """
    if chi0 <= 0.0:
        raise ValueError("chi0 must be positive")
    displacement = np.asarray(displacement_from_source, dtype=np.float64)
    if displacement.shape != (3,):
        raise ValueError("displacement must contain three coordinates")
    sample_point = np.asarray(source.center, dtype=np.float64) + displacement
    gradient = np.asarray(
        [
            periodic_trilinear_sample(source.gradient_x, sample_point),
            periodic_trilinear_sample(source.gradient_y, sample_point),
            periodic_trilinear_sample(source.gradient_z, sample_point),
        ],
        dtype=np.float64,
    )
    return -(c * c / chi0) * gradient


__all__ = [
    "Limit02BodyProfile",
    "build_limit02_body_profile",
    "limit02_acceleration_from_profile",
    "periodic_trilinear_sample",
    "smooth_spherical_density",
]
