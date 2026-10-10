"""Exact lattice U(1) charge-current tests."""

from __future__ import annotations

import numpy as np
import pytest

from lfm.analysis.phase import (
    bare_charge_continuity_residual,
    canonical_charge_density,
    oriented_charge_currents,
)


@pytest.mark.parametrize("stencil", ("19", "27"))
def test_exact_charge_continuity(stencil: str) -> None:
    rng = np.random.default_rng(20260724)
    shape = (7, 7, 7)
    fields = [rng.normal(size=shape) for _ in range(5)]
    residual = bare_charge_continuity_residual(
        *fields,
        wave_speed=0.73,
        stencil=stencil,
    )
    assert float(np.max(np.abs(residual))) < 1.0e-14


@pytest.mark.parametrize("stencil", ("19", "27"))
def test_oriented_link_current_is_antisymmetric(stencil: str) -> None:
    rng = np.random.default_rng(20260724)
    real = rng.normal(size=(6, 6, 6))
    imag = rng.normal(size=(6, 6, 6))
    currents = oriented_charge_currents(real, imag, stencil=stencil)
    for offset, current in currents.items():
        reverse = tuple(-value for value in offset)
        transported_reverse = np.roll(
            currents[reverse],
            shift=offset,
            axis=(0, 1, 2),
        )
        np.testing.assert_allclose(current, -transported_reverse, atol=1.0e-15)


def test_global_phase_rotation_preserves_charge() -> None:
    rng = np.random.default_rng(20260724)
    fields = [rng.normal(size=(5, 5, 5)) for _ in range(4)]
    real, imag, momentum_real, momentum_imag = fields
    angle = 0.731
    cosine = np.cos(angle)
    sine = np.sin(angle)
    real_rotated = cosine * real - sine * imag
    imag_rotated = sine * real + cosine * imag
    momentum_real_rotated = cosine * momentum_real - sine * momentum_imag
    momentum_imag_rotated = sine * momentum_real + cosine * momentum_imag
    expected = canonical_charge_density(
        real,
        imag,
        momentum_real,
        momentum_imag,
    )
    actual = canonical_charge_density(
        real_rotated,
        imag_rotated,
        momentum_real_rotated,
        momentum_imag_rotated,
    )
    np.testing.assert_allclose(actual, expected, atol=2.0e-15)
