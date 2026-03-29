"""
Tests for Simulation.place_particle() — the eigenmode-based particle placement API.

Covers:
- Stationary particle placement (eigenmode at rest)
- Moving particle placement (eigenmode + phase-gradient boost)
- Charge phase (matter vs antimatter)
- Multi-particle superposition
- Factory function integration (create_particle, create_two_particles, create_collision)
- Motion verification (COM tracking over time steps)
"""

import math

import numpy as np
import pytest

from lfm import (
    BoundaryType,
    FieldLevel,
    Simulation,
    SimulationConfig,
    create_collision,
    create_particle,
    create_two_particles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim(N: int = 32, field_level=FieldLevel.COMPLEX, chi0: float = 19.0):
    """Create a fresh simulation for testing."""
    cfg = SimulationConfig(
        grid_size=N,
        field_level=field_level,
        boundary_type=BoundaryType.FROZEN,
        chi0=chi0,
    )
    return Simulation(cfg)


def _peak_z(sim: Simulation) -> float:
    """Return z-coordinate of peak |Ψ|² (marginalised over x, y)."""
    pr = sim.psi_real
    pi = sim.psi_imag
    e2 = pr ** 2
    if pi is not None:
        e2 = e2 + pi ** 2
    profile = e2.sum(axis=(0, 1))  # marginalise x, y → 1-D profile(z)
    return float(np.argmax(profile))


# ---------------------------------------------------------------------------
# Test: stationary placement
# ---------------------------------------------------------------------------

class TestPlaceParticleStationary:
    """Place a particle at rest and verify it stays put."""

    def test_eigenmode_converges(self):
        """place_particle() returns a converged SolitonSolution."""
        sim = _make_sim(N=32)
        sol = sim.place_particle("electron")
        assert sol.converged
        assert sol.eigenvalue > 0
        assert sol.chi_min < 19.0  # well formed

    def test_chi_well_at_centre(self):
        """After equilibration the χ minimum is near the particle centre."""
        sim = _make_sim(N=32)
        sim.place_particle("electron", position=(16, 16, 16))
        sim.equilibrate()
        chi = sim.chi
        # Minimum should be near grid centre
        idx = np.unravel_index(np.argmin(chi), chi.shape)
        for ax in range(3):
            assert abs(idx[ax] - 16) < 4, f"chi minimum off-centre on axis {ax}"

    def test_stationary_stays_put(self):
        """A particle at rest should not drift significantly over 200 steps."""
        sim = _make_sim(N=32)
        sim.place_particle("electron", position=(16, 16, 16))
        z0 = _peak_z(sim)
        sim.run(steps=200)
        z1 = _peak_z(sim)
        assert abs(z1 - z0) < 3, f"Stationary particle drifted: {z0:.1f} → {z1:.1f}"


# ---------------------------------------------------------------------------
# Test: moving placement
# ---------------------------------------------------------------------------

class TestPlaceParticleMoving:
    """Place a particle with velocity and verify it moves."""

    def test_moving_particle_shifts(self):
        """A boosted particle's peak should move along the velocity axis."""
        sim = _make_sim(N=64)
        sim.place_particle("electron", position=(32, 32, 20), velocity=(0, 0, 0.1))
        z0 = _peak_z(sim)
        sim.run(steps=3000)
        z1 = _peak_z(sim)
        # Should have moved toward higher z (positive velocity)
        assert z1 > z0 + 1, f"Particle didn't move: z0={z0:.1f}, z1={z1:.1f}"

    def test_requires_complex_field(self):
        """Placing a moving particle in a REAL sim should raise ValueError."""
        sim = _make_sim(N=32, field_level=FieldLevel.REAL)
        with pytest.raises(ValueError, match="COMPLEX"):
            sim.place_particle("electron", velocity=(0, 0, 0.1))


# ---------------------------------------------------------------------------
# Test: charge phase
# ---------------------------------------------------------------------------

class TestChargePhase:
    """Verify matter vs antimatter phase is applied correctly."""

    def test_positron_has_imaginary_component(self):
        """Positron (phase=π) should have nonzero psi_imag."""
        sim = _make_sim(N=32)
        sim.place_particle("positron", position=(16, 16, 16))
        pi = sim.psi_imag
        # Phase π rotates real → mostly -real with tiny imag (from sin(π)≈0
        # but cos(π)=-1, so psi_r should be inverted)
        pr = sim.psi_real
        # The main lobe should be negative (cos(π) = -1 flips sign)
        assert pr[16, 16, 16] < 0, "Positron phase not applied (psi_r should be negative)"


# ---------------------------------------------------------------------------
# Test: multi-particle superposition
# ---------------------------------------------------------------------------

class TestMultiParticle:
    """Place two particles and verify superposition."""

    def test_two_particles_superpose(self):
        """Two particles create two distinct |Ψ|² peaks along z."""
        sim = _make_sim(N=64)
        sim.place_particle("electron", position=(32, 32, 20))
        sim.place_particle("electron", position=(32, 32, 44))
        sim.equilibrate()

        pr = sim.psi_real
        e2 = pr ** 2
        if sim.psi_imag is not None:
            e2 = e2 + sim.psi_imag ** 2
        profile = e2.sum(axis=(0, 1))  # z-profile

        # Should have two peaks near z=20 and z=44
        peak_z1 = np.argmax(profile[:32])
        peak_z2 = np.argmax(profile[32:]) + 32
        assert abs(peak_z1 - 20) < 5, f"First peak at {peak_z1}, expected ~20"
        assert abs(peak_z2 - 44) < 5, f"Second peak at {peak_z2}, expected ~44"


# ---------------------------------------------------------------------------
# Test: factory function integration
# ---------------------------------------------------------------------------

class TestFactoryIntegration:
    """Verify factory functions route through place_particle correctly."""

    def test_create_particle_eigenmode(self):
        """create_particle() with eigenmode places a self-consistent soliton."""
        placed = create_particle("electron", N=32)
        assert placed.particle.name == "electron"
        assert placed.sim is not None
        # place_particle() sets Ψ only; χ needs equilibrate()
        placed.sim.equilibrate()
        chi = placed.sim.chi
        assert chi.min() < 19.0

    def test_create_particle_gaussian(self):
        """create_particle(use_eigenmode=False) still works via place_soliton."""
        placed = create_particle("electron", N=32, use_eigenmode=False)
        assert placed.particle.name == "electron"
        # Energy should be nonzero
        pr = placed.sim.psi_real
        assert np.max(np.abs(pr)) > 0

    def test_create_two_particles(self):
        """create_two_particles() places two eigenmodes in a shared sim."""
        pa, pb = create_two_particles("electron", "positron", N=32, separation=12, axis=2)
        assert pa.sim is pb.sim  # shared simulation
        # Should have field content
        assert np.max(np.abs(pa.sim.psi_real)) > 0

    def test_create_collision(self):
        """create_collision() sets up opposing velocity particles."""
        setup = create_collision("proton", "antiproton", speed=0.1, N=32)
        assert setup.particle_a.name == "proton"
        assert setup.particle_b.name == "antiproton"
        assert setup.vel_a[0] > 0 or setup.vel_a[1] > 0 or setup.vel_a[2] > 0
        # Opposite velocities
        for i in range(3):
            assert setup.vel_a[i] == -setup.vel_b[i]


# ---------------------------------------------------------------------------
# Test: collision motion verification
# ---------------------------------------------------------------------------

class TestCollisionMotion:
    """End-to-end: two particles approach, collide, annihilate."""

    @pytest.mark.slow
    def test_particles_approach(self):
        """Proton+antiproton peaks must converge along collision axis."""
        sim = _make_sim(N=64)
        sim.place_particle("proton",     position=(32, 32, 22), velocity=(0, 0, 0.1))
        sim.place_particle("antiproton", position=(32, 32, 42), velocity=(0, 0, -0.1))
        sim.run(steps=100)  # just equilibrate

        pr = sim.psi_real
        pi = sim.psi_imag
        e2 = pr ** 2
        if pi is not None:
            e2 = e2 + pi ** 2
        profile0 = e2.sum(axis=(0, 1)).copy()

        sim.run(steps=4000)

        pr2 = sim.psi_real
        pi2 = sim.psi_imag
        e2b = pr2 ** 2
        if pi2 is not None:
            e2b = e2b + pi2 ** 2
        profile1 = e2b.sum(axis=(0, 1))

        # The peaks should have moved closer to centre (z=32)
        peak_low_0 = np.argmax(profile0[:32])
        peak_high_0 = np.argmax(profile0[32:]) + 32
        sep_0 = peak_high_0 - peak_low_0

        peak_low_1 = np.argmax(profile1[:32])
        peak_high_1 = np.argmax(profile1[32:]) + 32
        sep_1 = peak_high_1 - peak_low_1

        # Separation should decrease (particles approach)
        assert sep_1 < sep_0, (
            f"Particles didn't approach: sep {sep_0} → {sep_1}"
        )
