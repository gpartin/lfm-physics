"""Tests for discrete-to-continuum substrate observables."""

from __future__ import annotations

import numpy as np

from lfm.analysis.substrate_emergence import (
    composite_curvature_19pt,
    curvature_rms,
    normalize_internal_field,
    relational_wave_scaling_scan,
)


def _texture(size: int, epsilon: float = 0.1) -> np.ndarray:
    length = 2.0 * np.pi
    axis = np.arange(size, dtype=np.float64) * length / size
    x, y, _z = np.meshgrid(axis, axis, axis, indexing="ij")
    singlet = np.ones(3, dtype=np.complex128) / np.sqrt(3.0)
    relative = np.asarray((1.0, -1.0, 0.0), dtype=np.complex128) / np.sqrt(2.0)
    eta = np.sin(x) + 1j * np.sin(y)
    raw = singlet[:, None, None, None] + (epsilon * relative[:, None, None, None] * eta[None, ...])
    return normalize_internal_field(raw)


def test_relational_wave_scaling_converges() -> None:
    result = relational_wave_scaling_scan(
        (9.5, 19.0, 28.5),
        (0.25, 0.5, 1.0),
        (0.02, 0.01, 0.005, 0.0025),
        mass_factors=(1.0, np.sqrt(1.0 + 2.0 / 17.0)),
    )
    assert result["pass"] is True
    assert result["convergence"]["max_dispersion_error_slope"] > 1.8


def test_composite_curvature_is_global_unitary_invariant() -> None:
    z = _texture(20)
    rng = np.random.default_rng(20260725)
    matrix = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    unitary, _r = np.linalg.qr(matrix)
    rotated = np.einsum("ab,bxyz->axyz", unitary, z)
    dx = 2.0 * np.pi / z.shape[1]
    reference = composite_curvature_19pt(z, dx=dx)
    transformed = composite_curvature_19pt(rotated, dx=dx)
    error = max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(reference, transformed, strict=True)
    )
    assert curvature_rms(reference) > 1.0e-6
    assert error < 1.0e-12


def test_composite_curvature_changes_sign_under_conjugation() -> None:
    z = _texture(20)
    dx = 2.0 * np.pi / z.shape[1]
    positive = composite_curvature_19pt(z, dx=dx)
    negative = composite_curvature_19pt(np.conj(z), dx=dx)
    residual = max(
        float(np.max(np.abs(left + right))) for left, right in zip(positive, negative, strict=True)
    )
    assert residual < 1.0e-12
