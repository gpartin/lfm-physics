from __future__ import annotations

import numpy as np

from lfm import BoundaryType, FieldLevel, Simulation, SimulationConfig
from lfm.particles.noether import (
    cartesian_localization_metrics,
    cartesian_noether_charge,
    lift_radial_noether_state,
    make_radial_bag_guess,
    prolong_radial_fields,
    radial_fixed_charge_energy_and_gradient,
    radial_fixed_charge_hessian,
    radial_shell_geometry,
)


def test_radial_geometry_exactly_fills_domain() -> None:
    r, volumes, areas = radial_shell_geometry(2.0, 0.1)
    assert r.shape == (20,)
    assert volumes.shape == (20,)
    assert areas.shape == (21,)
    assert np.isclose(np.sum(volumes), 4.0 * np.pi * 2.0**3 / 3.0)
    assert areas[0] == 0.0


def test_bag_guess_has_requested_charge_at_guess_frequency() -> None:
    target_charge = 1200.0
    omega_guess = 10.0
    variables = make_radial_bag_guess(
        target_charge=target_charge,
        radius=3.0,
        dx=0.1,
        core_radius=0.7,
        omega_guess=omega_guess,
    )
    _, volumes, _ = radial_shell_geometry(3.0, 0.1)
    cells = volumes.size
    norm = float(np.dot(volumes, variables[:cells] ** 2))
    assert np.isclose(omega_guess * norm, target_charge, rtol=1.0e-13)


def test_cell_centered_prolongation_matches_declared_linear_interpolation() -> None:
    source_r, _, _ = radial_shell_geometry(2.0, 0.1)
    phi = np.exp(-source_r)
    chi = 19.0 - np.exp(-source_r)
    prolonged = prolong_radial_fields(
        radius=2.0,
        source_r=source_r,
        phi=phi,
        chi=chi,
        new_dx=0.05,
    )
    new_r, _, _ = radial_shell_geometry(2.0, 0.05)
    cells = new_r.size
    assert prolonged.shape == (2 * cells,)
    assert np.all(np.isfinite(prolonged))
    assert np.allclose(
        prolonged[:cells],
        np.interp(new_r, source_r, phi, left=phi[0], right=0.0),
    )
    assert np.allclose(
        prolonged[cells:],
        np.interp(new_r, source_r, chi, left=chi[0], right=19.0),
    )


def test_fixed_charge_analytic_gradient_matches_finite_difference() -> None:
    target_charge = 800.0
    radius = 1.2
    dx = 0.1
    variables = make_radial_bag_guess(
        target_charge=target_charge,
        radius=radius,
        dx=dx,
        core_radius=0.45,
        omega_guess=12.0,
        chi_depth_fraction=0.7,
    )
    ledger, gradient = radial_fixed_charge_energy_and_gradient(
        variables,
        target_charge=target_charge,
        radius=radius,
        dx=dx,
    )
    assert np.isfinite(ledger.total)
    probe_indices = [0, 3, 11, 12, 17, 23]
    # The total energy is O(1e7), so a 1e-6 step loses several digits to
    # subtraction. This step is inside the observed centered-difference
    # convergence window for both field blocks.
    epsilon = 3.0e-5
    for index in probe_indices:
        plus = variables.copy()
        minus = variables.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_energy, _ = radial_fixed_charge_energy_and_gradient(
            plus,
            target_charge=target_charge,
            radius=radius,
            dx=dx,
        )
        minus_energy, _ = radial_fixed_charge_energy_and_gradient(
            minus,
            target_charge=target_charge,
            radius=radius,
            dx=dx,
        )
        finite_difference = (plus_energy.total - minus_energy.total) / (2.0 * epsilon)
        assert np.isclose(
            gradient[index],
            finite_difference,
            rtol=5.0e-6,
            atol=3.0e-5,
        )


def test_fixed_charge_analytic_hessian_matches_gradient_difference() -> None:
    target_charge = 1200.0
    radius = 1.2
    dx = 0.1
    variables = make_radial_bag_guess(
        target_charge=target_charge,
        radius=radius,
        dx=dx,
        core_radius=0.45,
        omega_guess=10.0,
        chi_depth_fraction=0.8,
    )
    hessian = radial_fixed_charge_hessian(
        variables,
        target_charge=target_charge,
        radius=radius,
        dx=dx,
    )
    dense = hessian.toarray()
    assert np.allclose(dense, dense.T, rtol=0.0, atol=1.0e-12)

    epsilon = 1.0e-5
    for index in [0, 5, 12, 19, 23]:
        plus = variables.copy()
        minus = variables.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        _, plus_gradient = radial_fixed_charge_energy_and_gradient(
            plus,
            target_charge=target_charge,
            radius=radius,
            dx=dx,
        )
        _, minus_gradient = radial_fixed_charge_energy_and_gradient(
            minus,
            target_charge=target_charge,
            radius=radius,
            dx=dx,
        )
        finite_difference = (plus_gradient - minus_gradient) / (2.0 * epsilon)
        assert np.allclose(
            dense[:, index],
            finite_difference,
            rtol=2.0e-5,
            atol=3.0e-4,
        )


def test_radial_lift_initializes_the_declared_noether_rotation() -> None:
    source_r = (np.arange(40, dtype=np.float64) + 0.5) * 0.1
    phi = 12.0 * np.exp(-0.5 * (source_r / 0.7) ** 2)
    chi = 19.0 - 6.0 * np.exp(-0.5 * (source_r / 0.8) ** 2)
    omega = 1.7
    dt = 0.002
    state = lift_radial_noether_state(
        source_r=source_r,
        phi=phi,
        chi=chi,
        omega=omega,
        grid_size=32,
        dx=0.1,
        dt=dt,
        dtype=np.float64,
    )
    charge = cartesian_noether_charge(
        state.psi_real,
        state.psi_real_prev,
        state.psi_imag,
        state.psi_imag_prev,
        dt=dt,
        dx=0.1,
    )
    norm = float(np.sum(state.psi_real**2 + state.psi_imag**2)) * 0.1**3
    expected = np.sin(omega * dt) * norm / dt
    assert np.isclose(charge, expected, rtol=1.0e-13)
    metrics = cartesian_localization_metrics(
        state.psi_real,
        state.psi_imag,
        dx=0.1,
        core_radius=1.4,
    )
    assert metrics["rms_radius_cells"] > 5.0
    assert metrics["second_moment_anisotropy"] < 1.01


def test_conservative_mode_disables_historical_chi_floor() -> None:
    common = dict(
        grid_size=8,
        field_level=FieldLevel.COMPLEX,
        boundary_type=BoundaryType.PERIODIC,
        kappa=0.0,
        lambda_self=0.0,
        epsilon_w=0.0,
        dt=0.001,
        dx=0.5,
        report_interval=0,
    )
    unclipped = Simulation(
        SimulationConfig(**common, enable_chi_floor=False),
        backend="cpu",
    )
    below_floor = np.full((8, 8, 8), -20.0, dtype=np.float32)
    unclipped.set_chi(below_floor)
    unclipped.set_chi_prev(below_floor)
    unclipped.run(steps=1, record_metrics=False)
    assert np.allclose(unclipped.chi, -20.0)

    clipped = Simulation(
        SimulationConfig(**common, enable_chi_floor=True),
        backend="cpu",
    )
    clipped.set_chi(below_floor)
    clipped.set_chi_prev(below_floor)
    clipped.run(steps=1, record_metrics=False)
    assert np.allclose(clipped.chi, -19.0)
