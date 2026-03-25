"""
Tests for lfm.particles catalog (Phase 1).

Verifies correct particle properties, amplitude/sigma functions,
and catalog lookup — independent of any GPU / CUDA dependency.
"""

import pytest

import lfm
from lfm.constants import CHI0

# ── Particle existence and basic properties ─────────────────────────────────


class TestParticleProperties:
    def test_electron_exists(self):
        p = lfm.ELECTRON
        assert p.name == "electron"
        assert p.mass_ratio == pytest.approx(1.0)
        assert p.charge == pytest.approx(-1.0)
        assert p.l == 0
        assert p.field_level == 1  # complex field
        assert p.stable is True

    def test_muon_mass_ratio(self):
        p = lfm.MUON
        assert p.mass_ratio == pytest.approx(206.768)
        assert p.l == 14

    def test_proton_mass_ratio(self):
        p = lfm.PROTON
        assert p.mass_ratio == pytest.approx(1836.15)
        assert p.l == 42

    def test_neutron_real_field(self):
        # Neutron is neutral -> FieldLevel.REAL
        p = lfm.NEUTRON
        assert p.charge == pytest.approx(0.0)
        assert p.field_level == 0  # real field

    def test_positron_is_antielectron(self):
        p = lfm.POSITRON
        assert p.charge == pytest.approx(+1.0)
        assert p.antiparticle == "electron"

    def test_photon_massless(self):
        p = lfm.PHOTON
        assert p.mass_ratio == pytest.approx(0.0)

    def test_all_particles_in_catalog(self):
        names = [p.name for p in lfm.PARTICLES.values()]
        for expected in [
            "electron",
            "muon",
            "proton",
            "neutron",
            "photon",
            "up",
            "down",
        ]:
            assert expected in names, f"Missing particle: {expected}"

    def test_get_particle_by_name(self):
        p = lfm.get_particle("electron")
        assert p is lfm.ELECTRON

    def test_get_particle_case_insensitive(self):
        p = lfm.get_particle("MUON")
        assert p is lfm.MUON

    def test_get_particle_unknown_raises(self):
        with pytest.raises(KeyError):
            lfm.get_particle("unobtainium")


# ── Amplitude / sigma scaling ────────────────────────────────────────────────


class TestAmplitudeSigma:
    def test_electron_amplitude_in_range(self):
        for N in (32, 64):
            amp = lfm.amplitude_for_particle(lfm.ELECTRON, N)
            assert 4.0 < amp < CHI0, f"N={N}: amp={amp} out of range"

    def test_amplitude_cap(self):
        # Heavy particles should be capped at 0.85 * CHI0
        cap = 0.85 * CHI0
        for p in [lfm.PROTON, lfm.MUON]:
            amp = lfm.amplitude_for_particle(p, 64)
            assert amp <= cap + 1e-6, f"{p.name}: amp={amp} exceeds cap {cap}"

    def test_heavier_not_lighter_than_electron(self):
        amp_e = lfm.amplitude_for_particle(lfm.ELECTRON, 64)
        amp_mu = lfm.amplitude_for_particle(lfm.MUON, 64)
        # Muon amplitude >= electron (or equal when both at cap)
        assert amp_mu >= amp_e - 1e-6

    def test_sigma_decreases_with_mass(self):
        sig_e = lfm.sigma_for_particle(lfm.ELECTRON, 64)
        sig_mu = lfm.sigma_for_particle(lfm.MUON, 64)
        assert sig_e >= sig_mu, "Heavier particle should have smaller sigma"

    def test_sigma_min_enforced(self):
        # Even the heaviest particle must have sigma >= 2.0
        for N in (32, 64):
            sig = lfm.sigma_for_particle(lfm.PROTON, N)
            assert sig >= 2.0, f"N={N}: sigma={sig} below minimum 2.0"

    def test_sigma_max_enforced(self):
        for N in (32, 64):
            sig = lfm.sigma_for_particle(lfm.ELECTRON, N)
            assert sig <= N / 6.0, f"N={N}: sigma={sig} exceeds N/6={N / 6}"

    def test_amplitude_photon_positive(self):
        # Photon has mass_ratio=0; amplitude should default to a positive value
        amp = lfm.amplitude_for_particle(lfm.PHOTON, 64)
        assert amp > 0.0
