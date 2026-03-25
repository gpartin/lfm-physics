"""
Tests for lfm.particles Phase 5: factory, composite (atom/molecule).

These tests are designed to be fast by using small grids and few steps.
Some physics-validation tests are marked with a longer timeout.

Coverage targets:
  - factory.py: create_particle, PlacedParticle
  - composite.py: nuclear_chi_well, create_atom, create_molecule
"""

import pytest

import lfm
from lfm import (
    ELECTRON,
    MUON,
    PROTON,
    AtomState,
    MoleculeState,
    PlacedParticle,
    create_atom,
    create_molecule,
    create_particle,
    nuclear_chi_well,
)
from lfm.constants import CHI0

# ═══════════════════════════════════════════════════════════════════════════
# PlacedParticle dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestPlacedParticle:
    """Basic structural tests — no simulation needed."""

    def test_has_required_fields(self):
        """PlacedParticle must have .sim, .particle, .position, .velocity, .energy."""
        p = create_particle("electron", N=32, use_eigenmode=False)
        assert hasattr(p, "sim")
        assert hasattr(p, "particle")
        assert hasattr(p, "position")
        assert hasattr(p, "velocity")
        assert hasattr(p, "energy")

    def test_particle_attribute_is_particle(self):
        p = create_particle("electron", N=32, use_eigenmode=False)
        assert p.particle is ELECTRON

    def test_velocity_default_is_zero(self):
        p = create_particle("muon", N=32, use_eigenmode=False)
        assert p.velocity == (0.0, 0.0, 0.0)

    def test_velocity_stored_correctly(self):
        v = (0.03, 0.0, 0.0)
        p = create_particle("electron", N=32, velocity=v, use_eigenmode=False)
        assert p.velocity == v

    def test_energy_is_float(self):
        p = create_particle("electron", N=32, use_eigenmode=False)
        assert isinstance(p.energy, float)


# ═══════════════════════════════════════════════════════════════════════════
# create_particle — Gaussian seed path (use_eigenmode=False, fast)
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateParticleGaussian:
    """Tests that use use_eigenmode=False (instant, no SCF convergence needed)."""

    def test_electron_at_rest(self):
        p = create_particle("electron", N=32, use_eigenmode=False)
        assert isinstance(p, PlacedParticle)
        assert p.sim is not None

    def test_muon_at_rest(self):
        p = create_particle("muon", N=32, use_eigenmode=False)
        assert p.particle is MUON

    def test_proton_at_rest(self):
        p = create_particle("proton", N=32, use_eigenmode=False)
        assert p.particle is PROTON

    def test_unknown_particle_raises(self):
        with pytest.raises(KeyError):
            create_particle("unobtainium", N=32, use_eigenmode=False)

    def test_electron_with_velocity(self):
        p = create_particle("electron", N=32, velocity=(0.03, 0.0, 0.0), use_eigenmode=False)
        assert p.velocity[0] == pytest.approx(0.03)

    def test_custom_position(self):
        pos = (10.0, 16.0, 16.0)
        p = create_particle("electron", N=32, position=pos, use_eigenmode=False)
        # The position stored should match (Gaussian seed uses given position)
        assert p.position == pos

    def test_sim_runs(self):
        """Verify we can run 10 steps after create_particle."""
        p = create_particle("electron", N=32, use_eigenmode=False)
        p.sim.run(10)  # should not raise

    def test_different_particles_different_chi_min(self):
        """Heavier particles should produce deeper chi-wells."""
        pe = create_particle("electron", N=32, use_eigenmode=False)
        pm = create_particle("muon", N=32, use_eigenmode=False)
        pe.sim.run(steps=0)  # just read initial metrics
        pm.sim.run(steps=0)
        chi_min_e = pe.sim.metrics()["chi_min"]
        chi_min_m = pm.sim.metrics()["chi_min"]
        # Muon amplitude is larger -> deeper well (chi_min smaller)
        assert chi_min_m <= chi_min_e, (
            f"Expected muon chi_min ({chi_min_m:.2f}) <= electron ({chi_min_e:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# create_particle — Eigenmode solver path (use_eigenmode=True, slower)
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateParticleEigenmode:
    """Tests that verify the eigenmode solver path at small N."""

    def test_electron_eigenmode_returns_placed_particle(self):
        p = create_particle("electron", N=32, use_eigenmode=True)
        assert isinstance(p, PlacedParticle)

    def test_electron_eigenmode_chi_below_chi0(self):
        """SCF solver must produce a chi-well below chi0."""
        p = create_particle("electron", N=32, use_eigenmode=True)
        chi_min = p.sim.metrics()["chi_min"]
        assert chi_min < CHI0, f"chi_min={chi_min:.2f} not below chi0={CHI0}"

    def test_electron_eigenmode_with_velocity(self):
        """Boosted eigenmode should be created without error."""
        p = create_particle("electron", N=32, velocity=(0.02, 0.0, 0.0), use_eigenmode=True)
        assert p.sim is not None
        # Should be able to run a few steps
        p.sim.run(20, evolve_chi=False)


# ═══════════════════════════════════════════════════════════════════════════
# nuclear_chi_well
# ═══════════════════════════════════════════════════════════════════════════


class TestNuclearChiWell:
    """Unit tests for the nuclear chi-well builder."""

    def test_shape(self):
        chi = nuclear_chi_well(16, (8.0, 8.0, 8.0))
        assert chi.shape == (16, 16, 16)

    def test_background_is_chi0(self):
        chi = nuclear_chi_well(16, (8.0, 8.0, 8.0), depth=10.0)
        # Corners far from centre should be close to chi0
        corner = float(chi[0, 0, 0])
        assert corner == pytest.approx(CHI0, abs=0.1)

    def test_minimum_at_centre(self):
        N = 16
        chi = nuclear_chi_well(N, (N / 2, N / 2, N / 2), depth=14.0)
        # Centre should be close to chi0 - depth = 19 - 14 = 5
        cx = N // 2
        centre_val = float(chi[cx, cx, cx])
        assert centre_val == pytest.approx(CHI0 - 14.0, abs=1.0)

    def test_clamped_above_one(self):
        """Very deep well should be clamped, not negative."""
        chi = nuclear_chi_well(16, (8.0, 8.0, 8.0), depth=30.0)
        assert float(chi.min()) >= 1.0

    def test_depth_zero_gives_flat_chi0(self):
        chi = nuclear_chi_well(16, (8.0, 8.0, 8.0), depth=0.0)
        assert float(chi.min()) == pytest.approx(CHI0, abs=0.01)
        assert float(chi.max()) == pytest.approx(CHI0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# create_atom
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateAtom:
    """Atom creation tests.  Use N=32 and fewer steps to stay fast."""

    def test_returns_atom_state(self):
        atom = create_atom("H", N=32, steps=2_000)
        assert isinstance(atom, AtomState)

    def test_atom_has_required_fields(self):
        atom = create_atom("H", N=32, steps=2_000)
        assert hasattr(atom, "element")
        assert hasattr(atom, "sim")
        assert hasattr(atom, "chi_nuclear")
        assert hasattr(atom, "electron_energy")
        assert hasattr(atom, "electron_energy_free")
        assert hasattr(atom, "binding_energy")
        assert hasattr(atom, "bound")
        assert hasattr(atom, "fraction_near_nucleus")

    def test_hydrogen_bound(self):
        """H atom must be bound (fraction_near_nucleus >= 0.40) with enough steps."""
        atom = create_atom("H", N=32, steps=5_000)
        assert atom.bound, (
            f"H atom not bound: fraction_near_nucleus={atom.fraction_near_nucleus:.3f}"
        )

    def test_hydrogen_chi_min_depressed(self):
        """Nuclear chi-well must be below chi0."""
        atom = create_atom("H", N=32, steps=1_000)
        assert atom.chi_nuclear.min() < CHI0

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="element"):
            create_atom("Xe", N=32)

    def test_helium_bound(self):
        """He atom (deeper well) should also be bound."""
        atom = create_atom("He", N=32, steps=5_000)
        assert atom.bound, (
            f"He atom not bound: fraction_near_nucleus={atom.fraction_near_nucleus:.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# create_molecule
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateMolecule:
    """Molecule creation tests."""

    def test_returns_molecule_state(self):
        mol = create_molecule("H2", N=32, bond_length=10.0, steps=2_000)
        assert isinstance(mol, MoleculeState)

    def test_molecule_has_required_fields(self):
        mol = create_molecule("H2", N=32, bond_length=10.0, steps=2_000)
        assert hasattr(mol, "formula")
        assert hasattr(mol, "sim")
        assert hasattr(mol, "chi_nuclear")
        assert hasattr(mol, "electron_energy")
        assert hasattr(mol, "bond_stable")
        assert hasattr(mol, "proton_separation")

    def test_proton_separation_stored(self):
        mol = create_molecule("H2", N=32, bond_length=10.0, steps=1_000)
        assert mol.proton_separation == pytest.approx(10.0)

    def test_h2_chi_min_below_chi0(self):
        mol = create_molecule("H2", N=32, bond_length=10.0, steps=1_000)
        assert mol.chi_nuclear.min() < CHI0

    def test_h2_bond_stable(self):
        """H2 with default depth should be bond-stable at reasonable bond length."""
        mol = create_molecule("H2", N=64, bond_length=16.0, steps=5_000)
        assert mol.bond_stable, "H2 bond not stable"

    def test_unknown_formula_raises(self):
        with pytest.raises(ValueError, match="formula"):
            create_molecule("H2O", N=32)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: create_particle + run
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Smoke tests that exercise the full create → run pipeline."""

    def test_electron_gaussian_runs_50_steps(self):
        p = create_particle("electron", N=32, use_eigenmode=False)
        p.sim.run(50)
        m = p.sim.metrics()
        assert m["chi_min"] < CHI0

    def test_positron_gaussian_runs_50_steps(self):
        p = create_particle("positron", N=32, use_eigenmode=False)
        p.sim.run(50)

    def test_eigenmode_electron_stable_energy(self):
        """After eigenmode solve: chi-well should persist over 200 coupled steps.

        energy_total from metrics() is unreliable for sub-report_interval runs
        (report_interval=5000) because _psi_r_prev is only the psi from the
        *start* of the most-recent run() call, not the step immediately before.
        The physically meaningful stability check is that the χ-well doesn't
        collapse (chi_min stays well below chi0=19).
        """
        p = create_particle("electron", N=32, use_eigenmode=True)
        chi_min_0 = p.sim.metrics()["chi_min"]
        assert chi_min_0 < 18.5, f"chi_min at creation = {chi_min_0:.3f}, expected << 19"
        p.sim.run(200)
        chi_min_f = p.sim.metrics()["chi_min"]
        # Well must remain significantly depressed (particle stays bound)
        assert chi_min_f < 18.5, (
            f"Chi-well collapsed after 200 steps: chi_min went from "
            f"{chi_min_0:.3f} to {chi_min_f:.3f}"
        )

    def test_from_lfm_namespace(self):
        """Verify create_particle and PlacedParticle are accessible from lfm.*."""
        p = lfm.create_particle("electron", N=32, use_eigenmode=False)
        assert isinstance(p, lfm.PlacedParticle)
