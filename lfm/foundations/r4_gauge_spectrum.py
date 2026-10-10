"""Fourier audits for the compact gauge-link loop complex used by R4."""

from __future__ import annotations

import numpy as np

from lfm.foundations.r3_link_frame_live import (
    _link_table,
    triangle_loops,
)

Offset = tuple[int, int, int]


def _offset3(values: tuple[int, ...]) -> Offset:
    return (values[0], values[1], values[2])


def _reverse(offset: Offset) -> Offset:
    return _offset3(tuple(-value for value in offset))


def oriented_cycle_fourier_coefficients(
    cycle: tuple[Offset, ...],
    wavevector: tuple[float, float, float],
    stencil: str,
) -> np.ndarray:
    """Return the linear holonomy coefficients of one translated cycle."""

    unique, table = _link_table(stencil)
    coefficients = np.zeros(len(unique), dtype=np.complex128)
    shift = (0, 0, 0)
    for offset in cycle:
        index, reverse = table[offset]
        if reverse:
            stored_base = _offset3(tuple(shift[axis] + offset[axis] for axis in range(3)))
            sign = -1.0
        else:
            stored_base = shift
            sign = 1.0
        phase = sum(wavevector[axis] * stored_base[axis] for axis in range(3))
        coefficients[index] += sign * np.exp(1.0j * phase)
        shift = _offset3(tuple(shift[axis] + offset[axis] for axis in range(3)))
    if shift != (0, 0, 0):
        raise ValueError("cycle offsets must close")
    return coefficients


def triangle_fourier_hessian(
    stencil: str,
    wavevector: tuple[float, float, float],
) -> np.ndarray:
    """Return the R4 triangle-loop magnetic Hessian at one momentum."""

    link_count = len(_link_table(stencil)[0])
    hessian = np.zeros(
        (link_count, link_count),
        dtype=np.complex128,
    )
    for first, second, third, weight in triangle_loops(stencil):
        coefficients = oriented_cycle_fourier_coefficients(
            (first, second, third),
            wavevector,
            stencil,
        )
        hessian += weight * np.outer(
            coefficients.conj(),
            coefficients,
        )
    return hessian


def face_square_cycles() -> tuple[tuple[Offset, ...], ...]:
    """Return the three positively oriented axial face squares."""

    axes: tuple[Offset, ...] = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    )
    result = []
    for first in range(3):
        for second in range(first + 1, 3):
            a = axes[first]
            b = axes[second]
            result.append((a, b, _reverse(a), _reverse(b)))
    return tuple(result)


def directional_link_inertia(stencil: str) -> float:
    """Return the isotropic squared link projection count."""

    unique, _ = _link_table(stencil)
    counts = [float(sum(offset[axis] ** 2 for offset, _ in unique)) for axis in range(3)]
    if max(counts) - min(counts) > 1.0e-12:
        raise RuntimeError("link inventory is not directionally isotropic")
    return counts[0]


def face_square_fourier_hessian(
    stencil: str,
    wavevector: tuple[float, float, float],
) -> np.ndarray:
    """Return the geometry-normalized axial face-square Hessian."""

    link_count = len(_link_table(stencil)[0])
    hessian = np.zeros(
        (link_count, link_count),
        dtype=np.complex128,
    )
    coefficient = directional_link_inertia(stencil)
    for cycle in face_square_cycles():
        coefficients = oriented_cycle_fourier_coefficients(
            cycle,
            wavevector,
            stencil,
        )
        hessian += coefficient * np.outer(
            coefficients.conj(),
            coefficients,
        )
    return hessian


def gauge_link_spectrum(
    stencil: str,
    wavevector: tuple[float, float, float],
    *,
    include_face_squares: bool,
) -> np.ndarray:
    """Return sorted magnetic eigenvalues for one compact-link momentum."""

    hessian = triangle_fourier_hessian(stencil, wavevector)
    if include_face_squares:
        hessian += face_square_fourier_hessian(stencil, wavevector)
    return np.linalg.eigvalsh(hessian).real


def transverse_mode_speeds(
    stencil: str,
    wavevector: tuple[float, float, float],
    *,
    include_face_squares: bool,
) -> tuple[float, float]:
    """Return the two lowest positive physical-mode speeds."""

    momentum = float(np.linalg.norm(wavevector))
    if momentum <= 0.0:
        raise ValueError("wavevector must be nonzero")
    eigenvalues = gauge_link_spectrum(
        stencil,
        wavevector,
        include_face_squares=include_face_squares,
    )
    tolerance = max(1.0e-12, momentum**2 * 1.0e-8)
    positive = eigenvalues[eigenvalues > tolerance]
    if positive.size < 2:
        return 0.0, 0.0
    return (
        float(np.sqrt(positive[0]) / momentum),
        float(np.sqrt(positive[1]) / momentum),
    )
