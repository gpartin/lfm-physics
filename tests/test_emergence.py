from __future__ import annotations

import numpy as np

from lfm.analysis.emergence import (
    aggregate_wave_density,
    localized_state_observables,
    plaquette_winding_summary,
    principal_internal_projection,
)


def test_aggregate_density_is_invariant_under_internal_unitary() -> None:
    rng = np.random.default_rng(17)
    psi = rng.normal(size=(3, 6, 6, 6)) + 1j * rng.normal(size=(3, 6, 6, 6))
    raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    unitary, _ = np.linalg.qr(raw)
    rotated = np.einsum("ab,bxyz->axyz", unitary, psi)
    before = aggregate_wave_density(psi.real, psi.imag)
    after = aggregate_wave_density(rotated.real, rotated.imag)
    np.testing.assert_allclose(before, after, atol=1.0e-11, rtol=1.0e-11)


def test_principal_projection_reports_rank_one_internal_field() -> None:
    grid = np.zeros((7, 7, 7), dtype=np.complex128)
    grid[3, 3, 3] = 2.0 + 1.0j
    internal = np.asarray([1.0, 2.0j, -0.5], dtype=np.complex128)
    psi = internal[:, None, None, None] * grid[None, ...]
    projected, gap = principal_internal_projection(psi.real, psi.imag)
    assert gap > 1.0 - 1.0e-12
    assert np.isclose(np.sum(np.abs(projected) ** 2), np.sum(np.abs(psi) ** 2))


def test_plaquette_winding_rejects_zero_amplitude_phase() -> None:
    field = np.zeros((8, 8, 8), dtype=np.complex128)
    summary = plaquette_winding_summary(field)
    assert summary == {"positive": 0, "negative": 0, "nonzero": 0, "valid": 0}


def test_localized_observables_identify_single_site_and_charge() -> None:
    shape = (3, 9, 9, 9)
    psi = np.zeros(shape, dtype=np.complex128)
    psi[:, 4, 4, 4] = np.asarray([1.0, 1.0j, -1.0])
    dt = 0.02
    omega = 3.0
    previous = psi * np.exp(-1j * omega * dt)
    chi = np.full(shape[1:], 19.0)
    out = localized_state_observables(
        psi.real,
        psi.imag,
        previous.real,
        previous.imag,
        chi,
        dt,
        19.0,
    )
    assert np.isclose(out["effective_sites"], 1.0)
    assert out["top7_c7_overlap"] == 1
    assert out["noether_charge"] > 0.0
    assert np.isclose(out["chi_drop"], 0.0)


def test_localized_observables_classify_nonfinite_without_diagonalization() -> None:
    psi = np.zeros((3, 8, 8, 8), dtype=np.float64)
    psi[0, 4, 4, 4] = np.nan
    chi = np.full((8, 8, 8), 19.0)
    out = localized_state_observables(psi, psi, psi, psi, chi, 0.02, 19.0)
    assert np.isnan(out["wave_norm"])
    assert out["winding_nonzero"] == 0
