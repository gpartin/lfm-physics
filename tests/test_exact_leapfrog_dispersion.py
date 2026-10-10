from __future__ import annotations

import math

import pytest

from lfm.experiment.dispersion import dispersion


def test_wavelength_path_uses_exact_leapfrog_phase() -> None:
    dt = 0.02
    chi0 = 19.0
    wavelength = 4.0
    k = 2.0 * math.pi / wavelength
    spatial_omega_sq = chi0 * chi0 + 2.0 * (1.0 - math.cos(k))
    expected = math.acos(1.0 - 0.5 * dt * dt * spatial_omega_sq) / dt

    observed = dispersion(wavelength=wavelength, chi0=chi0, dt=dt)

    assert observed.omega == pytest.approx(expected, rel=0.0, abs=1.0e-14)
    assert abs(observed.omega - math.sqrt(spatial_omega_sq)) > 0.1


def test_exact_leapfrog_dispersion_round_trip() -> None:
    original = dispersion(wavelength=68.0 / 17.0, chi0=19.0, dt=0.02)
    recovered = dispersion(omega=original.omega, chi0=19.0, dt=0.02)

    assert recovered.k_z == pytest.approx(original.k_z, rel=0.0, abs=1.0e-13)
    assert recovered.wavelength == pytest.approx(
        original.wavelength,
        rel=0.0,
        abs=1.0e-13,
    )


def test_discrete_mass_gap_is_rejected_as_nonpropagating() -> None:
    dt = 0.02
    chi0 = 19.0
    mass_gap = math.acos(1.0 - 0.5 * dt * dt * chi0 * chi0) / dt

    with pytest.raises(ValueError, match="discrete mass gap"):
        dispersion(omega=mass_gap, chi0=chi0, dt=dt)
