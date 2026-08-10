"""Tests for the experiment-only local GOV-02 gravity recovery path."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from lfm.config import (
    BoundaryType,
    ChiPotentialModel,
    FieldLevel,
    Precision,
    SimulationConfig,
)
from lfm.config_presets import full_physics
from lfm.constants import CHI0, LAMBDA_H
from lfm.core.backends import gpu_available
from lfm.core.backends.kernel_source import GRAVITY_RECOVERY_REAL_KERNEL_SRC
from lfm.core.backends.numpy_backend import NumpyBackend
from lfm.experiment.gravity_recovery import (
    gravity_recovery_candidates,
    positive_frequency_previous_layers,
    potential_force,
    potential_second_derivative_at_vacuum,
)
from lfm.simulation import Simulation


def _config() -> SimulationConfig:
    return SimulationConfig(
        grid_size=8,
        dt=0.01,
        lambda_self=LAMBDA_H,
        field_level=FieldLevel.REAL,
        boundary_type=BoundaryType.PERIODIC,
        precision=Precision.FLOAT64,
        enable_chi_floor=False,
        report_interval=0,
    )


def _complex_config() -> SimulationConfig:
    return SimulationConfig(
        grid_size=8,
        dt=0.01,
        lambda_self=LAMBDA_H,
        field_level=FieldLevel.COMPLEX,
        boundary_type=BoundaryType.PERIODIC,
        precision=Precision.FLOAT64,
        enable_chi_floor=False,
        report_interval=0,
    )


def _color_config() -> SimulationConfig:
    return SimulationConfig(
        grid_size=8,
        dt=0.01,
        lambda_self=LAMBDA_H,
        field_level=FieldLevel.COLOR,
        boundary_type=BoundaryType.PERIODIC,
        precision=Precision.FLOAT64,
        enable_chi_floor=False,
        report_interval=0,
    )


def _state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(104729)
    shape = (8, 8, 8)
    psi = 0.02 * rng.standard_normal(shape)
    psi_prev = psi + 1.0e-4 * rng.standard_normal(shape)
    chi = CHI0 + 0.01 * rng.standard_normal(shape)
    chi_prev = chi + 1.0e-4 * rng.standard_normal(shape)
    return psi, psi_prev, chi, chi_prev


def _load(
    simulation: Simulation,
    state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    psi, psi_prev, chi, chi_prev = state
    simulation.psi_real = psi
    simulation._evolver.set_psi_real_prev(psi_prev)
    simulation.chi = chi
    simulation.chi_previous = chi_prev


def test_candidate_catalog_covers_required_families() -> None:
    families = {candidate.family for candidate in gravity_recovery_candidates()}
    assert families == set("ABCDEFGHIJKL")


def test_only_canonical_potential_has_vacuum_curvature() -> None:
    canonical = potential_second_derivative_at_vacuum(ChiPotentialModel.CANONICAL_QUARTIC)
    assert np.isclose(canonical, 8.0 * LAMBDA_H * CHI0**2, rtol=2.0e-8)
    for candidate in gravity_recovery_candidates():
        if candidate.model == ChiPotentialModel.CANONICAL_QUARTIC:
            continue
        curvature = potential_second_derivative_at_vacuum(candidate.model)
        assert abs(curvature) < 2.0e-4


def test_canonical_candidate_matches_production_real_step() -> None:
    state = _state()
    canonical = Simulation(_config(), backend="cpu")
    candidate = Simulation(_config(), backend="cpu")
    _load(canonical, state)
    _load(candidate, state)
    canonical.run(1, record_metrics=False)
    candidate.run_gravity_recovery(
        1,
        ChiPotentialModel.CANONICAL_QUARTIC,
        freeze_psi=False,
    )
    assert np.allclose(candidate.psi_real, canonical.psi_real, atol=1.0e-13)
    assert np.allclose(candidate.chi, canonical.chi, atol=1.0e-13)


def test_frozen_source_remains_exactly_frozen() -> None:
    state = _state()
    simulation = Simulation(_config(), backend="cpu")
    _load(simulation, state)
    source = simulation.psi_real.copy()
    simulation.run_gravity_recovery(
        4,
        ChiPotentialModel.FLAT_OCTIC,
        freeze_psi=True,
        relaxation_damping=0.5,
    )
    assert np.array_equal(simulation.psi_real, source)


def test_complex_flat_octic_matches_real_scalar_subspace() -> None:
    state = _state()
    real = Simulation(_config(), backend="cpu")
    complex_simulation = Simulation(_complex_config(), backend="cpu")
    _load(real, state)
    _load(complex_simulation, state)
    complex_simulation.psi_imag = np.zeros_like(state[0])
    complex_simulation.set_psi_imag_prev(np.zeros_like(state[0]))
    real.run_gravity_recovery(
        2,
        ChiPotentialModel.FLAT_OCTIC,
        freeze_psi=False,
    )
    complex_simulation.run_gravity_recovery(
        2,
        ChiPotentialModel.FLAT_OCTIC,
        freeze_psi=False,
    )
    assert np.allclose(
        complex_simulation.psi_real,
        real.psi_real,
        atol=1.0e-13,
    )
    assert np.allclose(complex_simulation.chi, real.chi, atol=1.0e-13)


def test_complex_flat_octic_is_quadrature_invariant() -> None:
    state = _state()
    real_quadrature = Simulation(_complex_config(), backend="cpu")
    imag_quadrature = Simulation(_complex_config(), backend="cpu")
    _load(real_quadrature, state)
    imag_state = (
        np.zeros_like(state[0]),
        np.zeros_like(state[1]),
        state[2],
        state[3],
    )
    _load(imag_quadrature, imag_state)
    real_quadrature.psi_imag = np.zeros_like(state[0])
    real_quadrature.set_psi_imag_prev(np.zeros_like(state[0]))
    imag_quadrature.psi_imag = state[0]
    imag_quadrature.set_psi_imag_prev(state[1])
    real_quadrature.run_gravity_recovery(
        3,
        ChiPotentialModel.FLAT_OCTIC,
        freeze_psi=False,
    )
    imag_quadrature.run_gravity_recovery(
        3,
        ChiPotentialModel.FLAT_OCTIC,
        freeze_psi=False,
    )
    assert np.allclose(
        imag_quadrature.psi_imag,
        real_quadrature.psi_real,
        atol=1.0e-13,
    )
    assert np.allclose(
        imag_quadrature.chi,
        real_quadrature.chi,
        atol=1.0e-13,
    )


def test_color_flat_octic_matches_complex_one_channel_subspace() -> None:
    state = _state()
    complex_simulation = Simulation(_complex_config(), backend="cpu")
    color_simulation = Simulation(_color_config(), backend="cpu")
    _load(complex_simulation, state)
    complex_simulation.psi_imag = np.zeros_like(state[0])
    complex_simulation.set_psi_imag_prev(np.zeros_like(state[0]))
    color_real = np.zeros((3,) + state[0].shape)
    color_real_prev = np.zeros_like(color_real)
    color_real[0] = state[0]
    color_real_prev[0] = state[1]
    color_simulation.psi_real = color_real
    color_simulation.set_psi_real_prev(color_real_prev)
    color_simulation.psi_imag = np.zeros_like(color_real)
    color_simulation.set_psi_imag_prev(np.zeros_like(color_real))
    color_simulation.chi = state[2]
    color_simulation.chi_previous = state[3]
    complex_simulation.run_gravity_recovery(
        2,
        ChiPotentialModel.FLAT_OCTIC,
        freeze_psi=False,
    )
    color_simulation.run_gravity_recovery(
        2,
        ChiPotentialModel.FLAT_OCTIC,
        freeze_psi=False,
    )
    assert np.allclose(
        color_simulation.psi_real[0],
        complex_simulation.psi_real,
        atol=1.0e-13,
    )
    assert np.allclose(
        color_simulation.chi,
        complex_simulation.chi,
        atol=1.0e-13,
    )


def test_positive_frequency_previous_layer_matches_uniform_mode() -> None:
    shape = (3, 8, 8, 8)
    real = np.zeros(shape)
    imag = np.zeros(shape)
    real[0] = 0.25
    chi = np.full(shape[-3:], CHI0)
    dt = 0.005
    previous_real, previous_imag, metadata = positive_frequency_previous_layers(
        real,
        imag,
        chi,
        dt=dt,
        polynomial_degree=12,
    )
    cosine = 1.0 - 0.5 * dt**2 * CHI0**2
    sine = np.sqrt(1.0 - cosine**2)
    assert np.allclose(previous_real[0], cosine * real[0], atol=1.0e-13)
    assert np.allclose(previous_imag[0], sine * real[0], atol=1.0e-13)
    assert np.array_equal(previous_real[1:], np.zeros_like(real[1:]))
    assert np.array_equal(previous_imag[1:], np.zeros_like(imag[1:]))
    assert metadata["local_stencil_radius_upper_bound"] == 12


def test_positive_frequency_initializer_contains_no_inverse_solver() -> None:
    source = inspect.getsource(positive_frequency_previous_layers).lower()
    for forbidden in (
        "np.fft",
        "rfftn",
        "irfftn",
        "green",
        "poisson",
        "inverse_square",
        "np.linalg.inv",
        "np.linalg.solve",
    ):
        assert forbidden not in source


@pytest.mark.skipif(not gpu_available(), reason="CuPy GPU backend unavailable")
def test_complex_flat_octic_cpu_gpu_parity() -> None:
    state = _state()
    cpu = Simulation(_complex_config(), backend="cpu")
    gpu = Simulation(_complex_config(), backend="gpu")
    _load(cpu, state)
    _load(gpu, state)
    imag = 0.75 * state[0]
    imag_prev = 0.75 * state[1]
    for simulation in (cpu, gpu):
        simulation.psi_imag = imag
        simulation.set_psi_imag_prev(imag_prev)
        simulation.run_gravity_recovery(
            2,
            ChiPotentialModel.FLAT_OCTIC,
            freeze_psi=False,
            relaxation_damping=0.25,
        )
    assert np.allclose(gpu.psi_real, cpu.psi_real, atol=2.0e-11)
    assert np.allclose(gpu.psi_imag, cpu.psi_imag, atol=2.0e-11)
    assert np.allclose(gpu.chi, cpu.chi, atol=2.0e-11)


@pytest.mark.skipif(not gpu_available(), reason="CuPy GPU backend unavailable")
def test_color_flat_octic_cpu_gpu_parity() -> None:
    state = _state()
    cpu = Simulation(_color_config(), backend="cpu")
    gpu = Simulation(_color_config(), backend="gpu")
    rng = np.random.default_rng(15485863)
    color_real = 0.02 * rng.standard_normal((3,) + state[0].shape)
    color_imag = 0.02 * rng.standard_normal((3,) + state[0].shape)
    color_real_prev = color_real + 1.0e-4 * rng.standard_normal(color_real.shape)
    color_imag_prev = color_imag + 1.0e-4 * rng.standard_normal(color_imag.shape)
    for simulation in (cpu, gpu):
        simulation.psi_real = color_real
        simulation.set_psi_real_prev(color_real_prev)
        simulation.psi_imag = color_imag
        simulation.set_psi_imag_prev(color_imag_prev)
        simulation.chi = state[2]
        simulation.chi_previous = state[3]
        simulation.run_gravity_recovery(
            2,
            ChiPotentialModel.FLAT_OCTIC,
            freeze_psi=False,
            relaxation_damping=0.25,
        )
    assert np.allclose(gpu.psi_real, cpu.psi_real, atol=2.0e-11)
    assert np.allclose(gpu.psi_imag, cpu.psi_imag, atol=2.0e-11)
    assert np.allclose(gpu.chi, cpu.chi, atol=2.0e-11)


@pytest.mark.skipif(not gpu_available(), reason="CuPy GPU backend unavailable")
@pytest.mark.parametrize(
    "model",
    [
        ChiPotentialModel.CANONICAL_QUARTIC,
        ChiPotentialModel.SMOOTH_EXPONENTIAL,
        ChiPotentialModel.NONLINEAR_GRADIENT,
        ChiPotentialModel.VARIABLE_INERTIA,
    ],
)
def test_cpu_gpu_candidate_step_parity(model: ChiPotentialModel) -> None:
    state = _state()
    cpu = Simulation(_config(), backend="cpu")
    gpu = Simulation(_config(), backend="gpu")
    _load(cpu, state)
    _load(gpu, state)
    cpu.run_gravity_recovery(
        2,
        model,
        freeze_psi=True,
        relaxation_damping=0.25,
    )
    gpu.run_gravity_recovery(
        2,
        model,
        freeze_psi=True,
        relaxation_damping=0.25,
    )
    assert np.allclose(gpu.chi, cpu.chi, atol=2.0e-11, rtol=2.0e-11)


def test_analysis_and_backend_force_laws_are_finite() -> None:
    chi = np.linspace(0.1 * CHI0, 1.5 * CHI0, 51)
    source = np.linspace(0.0, CHI0**2, 51)
    for candidate in gravity_recovery_candidates():
        force = potential_force(
            chi,
            candidate.model,
            source_density=source,
        )
        assert np.all(np.isfinite(force))


def test_gravity_recovery_evolution_contains_no_nonlocal_solver() -> None:
    cpu_source = inspect.getsource(NumpyBackend.step_real_gravity_recovery)
    combined = cpu_source.lower() + GRAVITY_RECOVERY_REAL_KERNEL_SRC.lower()
    for forbidden in (
        "np.fft",
        "rfftn",
        "irfftn",
        "green",
        "poisson",
        "inverse_square",
        "target_profile",
    ):
        assert forbidden not in combined


def test_flat_octic_force_matches_declared_equation() -> None:
    chi = np.linspace(0.5 * CHI0, 1.5 * CHI0, 41)
    measured = potential_force(
        chi,
        ChiPotentialModel.FLAT_OCTIC,
    )
    expected = -8.0 * LAMBDA_H * chi * (chi**2 - CHI0**2) ** 3 / CHI0**4
    assert np.allclose(measured, expected, atol=1.0e-10, rtol=1.0e-13)


def test_invalid_timestep_override_is_rejected() -> None:
    simulation = Simulation(_config(), backend="cpu")
    with pytest.raises(ValueError, match="dt_override"):
        simulation.run_gravity_recovery(
            1,
            ChiPotentialModel.FLAT_OCTIC,
            dt_override=0.0,
        )


@pytest.mark.parametrize("lambda_self", [0.0, LAMBDA_H])
def test_bare_color_candidate_is_charge_conjugation_blind(
    lambda_self: float,
) -> None:
    rng = np.random.default_rng(260725)
    shape = (3, 8, 8, 8)
    psi = 0.002 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    psi_prev = psi + 0.0001 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    chi = CHI0 + 0.002 * rng.standard_normal(shape[-3:])
    chi_prev = chi + 0.0001 * rng.standard_normal(chi.shape)
    simulations = []
    for field, previous in ((psi, psi_prev), (np.conj(psi), np.conj(psi_prev))):
        config = _color_config()
        config.lambda_self = lambda_self
        config.epsilon_w = 0.0
        simulation = Simulation(config, backend="cpu")
        simulation.psi_real = field.real
        simulation.psi_imag = field.imag
        simulation.set_psi_real_prev(previous.real)
        simulation.set_psi_imag_prev(previous.imag)
        simulation.chi = chi
        simulation.chi_previous = chi_prev
        simulation.run_gravity_recovery(
            5,
            ChiPotentialModel.FLAT_OCTIC,
            freeze_psi=False,
        )
        simulations.append(simulation)
    positive, negative = simulations
    assert np.array_equal(negative.chi, positive.chi)
    assert np.allclose(negative.psi_real, positive.psi_real, atol=1.0e-14)
    assert np.allclose(negative.psi_imag, -positive.psi_imag, atol=1.0e-14)


@pytest.mark.parametrize("lambda_self", [0.0, LAMBDA_H])
def test_bare_color_candidate_remains_parity_equivariant(
    lambda_self: float,
) -> None:
    rng = np.random.default_rng(32452843)
    shape = (3, 8, 8, 8)
    psi = 0.002 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    psi_prev = psi + 0.0001 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    chi = CHI0 + 0.002 * rng.standard_normal(shape[-3:])
    chi_prev = chi + 0.0001 * rng.standard_normal(chi.shape)
    simulations = []
    for field, previous, substrate, substrate_prev in (
        (psi, psi_prev, chi, chi_prev),
        (
            psi[:, ::-1, :, :],
            psi_prev[:, ::-1, :, :],
            chi[::-1, :, :],
            chi_prev[::-1, :, :],
        ),
    ):
        config = _color_config()
        config.lambda_self = lambda_self
        config.epsilon_w = 0.0
        simulation = Simulation(config, backend="cpu")
        simulation.psi_real = field.real
        simulation.psi_imag = field.imag
        simulation.set_psi_real_prev(previous.real)
        simulation.set_psi_imag_prev(previous.imag)
        simulation.chi = substrate
        simulation.chi_previous = substrate_prev
        simulation.run_gravity_recovery(
            5,
            ChiPotentialModel.FLAT_OCTIC,
            freeze_psi=False,
        )
        simulations.append(simulation)
    original, mirrored = simulations
    assert np.allclose(
        mirrored.psi_real,
        original.psi_real[:, ::-1, :, :],
        atol=1.0e-14,
    )
    assert np.allclose(
        mirrored.psi_imag,
        original.psi_imag[:, ::-1, :, :],
        atol=1.0e-14,
    )
    assert np.allclose(mirrored.chi, original.chi[::-1, :, :], atol=1.0e-13)


@pytest.mark.parametrize("lambda_self", [0.0, LAMBDA_H])
def test_bare_color_candidate_is_globally_su3_covariant(
    lambda_self: float,
) -> None:
    rng = np.random.default_rng(49979687)
    shape = (3, 8, 8, 8)
    psi = 0.001 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    psi_prev = psi + 0.00005 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    random_matrix = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    unitary, _ = np.linalg.qr(random_matrix)
    unitary = unitary / np.linalg.det(unitary) ** (1.0 / 3.0)
    rotated = np.einsum("ab,bijk->aijk", unitary, psi)
    rotated_prev = np.einsum("ab,bijk->aijk", unitary, psi_prev)
    chi = CHI0 + 0.001 * rng.standard_normal(shape[-3:])
    chi_prev = chi + 0.00005 * rng.standard_normal(chi.shape)
    simulations = []
    for field, previous in ((psi, psi_prev), (rotated, rotated_prev)):
        config = _color_config()
        config.lambda_self = lambda_self
        simulation = Simulation(config, backend="cpu")
        simulation.psi_real = field.real
        simulation.psi_imag = field.imag
        simulation.set_psi_real_prev(previous.real)
        simulation.set_psi_imag_prev(previous.imag)
        simulation.chi = chi
        simulation.chi_previous = chi_prev
        simulation.run_gravity_recovery(
            5,
            ChiPotentialModel.FLAT_OCTIC,
            freeze_psi=False,
        )
        simulations.append(simulation)
    original, transformed = simulations
    original_field = original.psi_real + 1j * original.psi_imag
    transformed_field = transformed.psi_real + 1j * transformed.psi_imag
    expected = np.einsum("ab,bijk->aijk", unitary, original_field)
    assert np.array_equal(transformed.chi, original.chi)
    assert np.allclose(transformed_field, expected, atol=1.0e-14)


def _full_color_config(lambda_self: float) -> SimulationConfig:
    return full_physics(
        grid_size=8,
        boundary_type=BoundaryType.PERIODIC,
        lambda_self=lambda_self,
        precision=Precision.FLOAT64,
        enable_chi_floor=False,
        use_stencil19_noether_current=True,
        report_interval=0,
    )


def _load_full_color_state(
    simulation: Simulation,
    *,
    seed: int = 67867967,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (3, 8, 8, 8)
    psi = 0.003 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    psi_prev = psi + 0.0001 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    chi = CHI0 + 0.003 * rng.standard_normal(shape[-3:])
    chi_prev = chi + 0.0001 * rng.standard_normal(chi.shape)
    simulation.psi_real = psi.real
    simulation.psi_imag = psi.imag
    simulation.set_psi_real_prev(psi_prev.real)
    simulation.set_psi_imag_prev(psi_prev.imag)
    simulation.chi = chi
    simulation.chi_previous = chi_prev
    return psi, psi_prev, chi, chi_prev


def test_full_color_zero_restoring_candidate_matches_production_step() -> None:
    production = Simulation(_full_color_config(0.0), backend="cpu")
    candidate = Simulation(_full_color_config(0.0), backend="cpu")
    _load_full_color_state(production)
    _load_full_color_state(candidate)
    production.run(1)
    candidate.run_gravity_recovery(
        1,
        ChiPotentialModel.FLAT_OCTIC,
        freeze_psi=False,
    )
    assert np.allclose(candidate.psi_real, production.psi_real, atol=1.0e-14)
    assert np.allclose(candidate.psi_imag, production.psi_imag, atol=1.0e-14)
    assert np.allclose(candidate.chi, production.chi, atol=1.0e-13)


def test_full_color_flat_octic_changes_only_restoring_force() -> None:
    production = Simulation(_full_color_config(0.0), backend="cpu")
    candidate = Simulation(_full_color_config(LAMBDA_H), backend="cpu")
    _, _, chi, _ = _load_full_color_state(production)
    _load_full_color_state(candidate)
    production.run(1)
    candidate.run_gravity_recovery(
        1,
        ChiPotentialModel.FLAT_OCTIC,
        freeze_psi=False,
    )
    expected_delta = production.config.dt**2 * potential_force(
        chi,
        ChiPotentialModel.FLAT_OCTIC,
    )
    assert np.allclose(candidate.psi_real, production.psi_real, atol=1.0e-14)
    assert np.allclose(candidate.psi_imag, production.psi_imag, atol=1.0e-14)
    assert np.allclose(
        candidate.chi - production.chi,
        expected_delta,
        atol=2.0e-13,
        rtol=1.0e-10,
    )


@pytest.mark.skipif(not gpu_available(), reason="CuPy GPU backend unavailable")
def test_full_color_flat_octic_cpu_gpu_parity() -> None:
    cpu = Simulation(_full_color_config(LAMBDA_H), backend="cpu")
    gpu = Simulation(_full_color_config(LAMBDA_H), backend="gpu")
    _load_full_color_state(cpu, seed=86028121)
    _load_full_color_state(gpu, seed=86028121)
    for simulation in (cpu, gpu):
        simulation.run_gravity_recovery(
            2,
            ChiPotentialModel.FLAT_OCTIC,
            freeze_psi=False,
        )
    assert np.allclose(gpu.psi_real, cpu.psi_real, atol=2.0e-10)
    assert np.allclose(gpu.psi_imag, cpu.psi_imag, atol=2.0e-10)
    assert np.allclose(gpu.chi, cpu.chi, atol=2.0e-10)
