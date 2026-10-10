"""Tests for the macroscopic LIMIT-02 two-body reduction."""

import numpy as np

import lfm


def test_smooth_spherical_density_normalizes_mass():
    density = lfm.smooth_spherical_density(
        32,
        center=(16.0, 16.0, 16.0),
        radius=4.5,
        mass=123.0,
    )
    assert density.shape == (32, 32, 32)
    assert np.all(density >= 0.0)
    np.testing.assert_allclose(
        np.sum(density),
        123.0,
        rtol=1e-13,
    )


def test_periodic_trilinear_sample_affine_interior():
    x, y, z = np.meshgrid(
        np.arange(8, dtype=np.float64),
        np.arange(8, dtype=np.float64),
        np.arange(8, dtype=np.float64),
        indexing="ij",
    )
    field = 2.0 * x - 3.0 * y + 0.5 * z
    point = (2.25, 3.5, 4.75)
    measured = lfm.periodic_trilinear_sample(field, point)
    expected = 2.0 * point[0] - 3.0 * point[1] + 0.5 * point[2]
    np.testing.assert_allclose(measured, expected, atol=1e-12)


def test_limit02_profile_accelerates_inward():
    source = lfm.build_limit02_body_profile(
        32,
        radius=4.0,
        mass=100.0,
    )
    acceleration = lfm.limit02_acceleration_from_profile(
        source,
        (10.0, 0.0, 0.0),
    )
    assert acceleration[0] < 0.0
    assert abs(acceleration[1]) < 1e-12
    assert abs(acceleration[2]) < 1e-12


def test_rest_release_reduces_separation():
    heavy = lfm.build_limit02_body_profile(
        32,
        radius=4.0,
        mass=100.0,
    )
    light = lfm.build_limit02_body_profile(
        32,
        radius=1.5,
        mass=2.0,
    )
    rows = lfm.integrate_limit02_two_body(
        heavy,
        light,
        initial_separation=10.0,
        light_tangential_speed=0.0,
        dt=1.0,
        steps=100,
        sample_every=10,
    )
    summary = lfm.summarize_limit02_orbit(rows)
    assert summary["initial_heavy_inward_acceleration"] > 0.0
    assert summary["initial_light_inward_acceleration"] > 0.0
    assert summary["final_separation"] < summary["initial_separation"]
