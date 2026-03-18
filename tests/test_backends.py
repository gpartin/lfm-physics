"""Tests for lfm.core.backends — CPU backend and auto-detection."""

import numpy as np
import pytest

from lfm.core.backends import get_backend, gpu_available, NumpyBackend
from lfm.core.backends.protocol import Backend


class TestGetBackend:
    def test_cpu_returns_numpy(self):
        backend = get_backend("cpu")
        assert isinstance(backend, NumpyBackend)

    def test_auto_returns_backend(self):
        backend = get_backend("auto")
        assert isinstance(backend, Backend)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_backend("tpu")

    def test_gpu_available_is_bool(self):
        assert isinstance(gpu_available(), bool)


class TestNumpyBackend:
    @pytest.fixture
    def backend(self):
        return NumpyBackend()

    def test_name(self, backend):
        assert backend.name == "numpy"

    def test_boundary_mask_shape(self, backend):
        mask = backend.create_boundary_mask(16, 0.15)
        assert mask.shape == (16**3,)
        assert mask.dtype == np.float32

    def test_boundary_mask_center_is_zero(self, backend):
        N = 16
        mask = backend.create_boundary_mask(N, 0.15)
        # Center voxel index
        c = N // 2
        idx = c * N * N + c * N + c
        assert mask[idx] == 0.0, "Center should not be frozen"

    def test_boundary_mask_corner_is_one(self, backend):
        N = 16
        mask = backend.create_boundary_mask(N, 0.15)
        # Corner (0,0,0)
        assert mask[0] == 1.0, "Corner should be frozen"

    def test_to_from_numpy_roundtrip(self, backend):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = backend.to_numpy(backend.from_numpy(arr))
        np.testing.assert_array_equal(arr, out)

    def test_from_numpy_casts_dtype(self, backend):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        result = backend.from_numpy(arr)
        assert result.dtype == np.float32


class TestNumpyStepReal:
    """Test the real-field (Level 0) step."""

    @pytest.fixture
    def setup(self):
        N = 16
        total = N**3
        backend = NumpyBackend()
        mask = backend.create_boundary_mask(N, 0.15)

        # Initialize: chi=19, E=0
        chi0 = 19.0
        E_A = np.zeros(total, dtype=np.float32)
        E_prev_A = np.zeros(total, dtype=np.float32)
        chi_A = np.full(total, chi0, dtype=np.float32)
        chi_prev_A = np.full(total, chi0, dtype=np.float32)
        E_B = np.zeros(total, dtype=np.float32)
        E_prev_B = np.zeros(total, dtype=np.float32)
        chi_B = np.full(total, chi0, dtype=np.float32)
        chi_prev_B = np.full(total, chi0, dtype=np.float32)

        return {
            "backend": backend, "N": N, "mask": mask, "chi0": chi0,
            "E_A": E_A, "E_prev_A": E_prev_A,
            "chi_A": chi_A, "chi_prev_A": chi_prev_A,
            "E_B": E_B, "E_prev_B": E_prev_B,
            "chi_B": chi_B, "chi_prev_B": chi_prev_B,
        }

    def test_empty_universe_stable(self, setup):
        """Empty universe: E stays 0, chi stays chi0."""
        s = setup
        dt2 = 0.02**2
        for _ in range(10):
            s["backend"].step_real(
                s["E_A"], s["E_prev_A"], s["chi_A"], s["chi_prev_A"],
                s["mask"],
                s["E_B"], s["E_prev_B"], s["chi_B"], s["chi_prev_B"],
                s["N"], dt2, 1/63, 0.0, s["chi0"], 0.0,
            )
            # Swap A <-> B
            for k in ("E", "E_prev", "chi", "chi_prev"):
                s[f"{k}_A"], s[f"{k}_B"] = s[f"{k}_B"], s[f"{k}_A"]

        np.testing.assert_allclose(s["E_A"], 0.0, atol=1e-10)
        np.testing.assert_allclose(s["chi_A"], s["chi0"], atol=1e-5)

    def test_energy_reduces_chi(self, setup):
        """A blob of energy should cause chi to decrease."""
        s = setup
        N = s["N"]
        c = N // 2

        # Place energy at center (avoid boundary)
        idx = c * N * N + c * N + c
        s["E_A"][idx] = 5.0
        s["E_prev_A"][idx] = 5.0

        dt2 = 0.02**2
        for _ in range(50):
            s["backend"].step_real(
                s["E_A"], s["E_prev_A"], s["chi_A"], s["chi_prev_A"],
                s["mask"],
                s["E_B"], s["E_prev_B"], s["chi_B"], s["chi_prev_B"],
                s["N"], dt2, 1/63, 0.0, s["chi0"], 0.0,
            )
            s["E_A"], s["E_B"] = s["E_B"], s["E_A"]
            s["E_prev_A"], s["E_prev_B"] = s["E_prev_B"], s["E_prev_A"]
            s["chi_A"], s["chi_B"] = s["chi_B"], s["chi_A"]
            s["chi_prev_A"], s["chi_prev_B"] = s["chi_prev_B"], s["chi_prev_A"]

        # chi at center should be below chi0
        assert s["chi_A"][idx] < s["chi0"], "Energy should create χ well"


class TestNumpyStepComplex:
    """Test the complex-field (Level 1) step."""

    def test_empty_complex_stable(self):
        N = 16
        total = N**3
        backend = NumpyBackend()
        mask = backend.create_boundary_mask(N, 0.15)
        chi0 = 19.0
        dt2 = 0.02**2
        z = np.zeros(total, dtype=np.float32)
        chi = np.full(total, chi0, dtype=np.float32)

        Pr_A, Pr_prev_A = z.copy(), z.copy()
        Pi_A, Pi_prev_A = z.copy(), z.copy()
        chi_A, chi_prev_A = chi.copy(), chi.copy()
        Pr_B, Pr_prev_B = z.copy(), z.copy()
        Pi_B, Pi_prev_B = z.copy(), z.copy()
        chi_B, chi_prev_B = chi.copy(), chi.copy()

        for _ in range(10):
            backend.step_complex(
                Pr_A, Pr_prev_A, Pi_A, Pi_prev_A, chi_A, chi_prev_A,
                mask,
                Pr_B, Pr_prev_B, Pi_B, Pi_prev_B, chi_B, chi_prev_B,
                N, dt2, 1/63, 0.0, chi0, 0.0, 0.1,
            )
            Pr_A, Pr_B = Pr_B, Pr_A
            Pr_prev_A, Pr_prev_B = Pr_prev_B, Pr_prev_A
            Pi_A, Pi_B = Pi_B, Pi_A
            Pi_prev_A, Pi_prev_B = Pi_prev_B, Pi_prev_A
            chi_A, chi_B = chi_B, chi_A
            chi_prev_A, chi_prev_B = chi_prev_B, chi_prev_A

        np.testing.assert_allclose(Pr_A, 0.0, atol=1e-10)
        np.testing.assert_allclose(Pi_A, 0.0, atol=1e-10)
        np.testing.assert_allclose(chi_A, chi0, atol=1e-5)


class TestNumpyStepColor:
    """Test the 3-color (Level 2) step."""

    def test_empty_color_stable(self):
        N = 16
        total = N**3
        backend = NumpyBackend()
        mask = backend.create_boundary_mask(N, 0.15)
        chi0 = 19.0
        dt2 = 0.02**2

        z3 = np.zeros(3 * total, dtype=np.float32)
        chi = np.full(total, chi0, dtype=np.float32)

        Pr_A, Pr_prev_A = z3.copy(), z3.copy()
        Pi_A, Pi_prev_A = z3.copy(), z3.copy()
        chi_A, chi_prev_A = chi.copy(), chi.copy()
        Pr_B, Pr_prev_B = z3.copy(), z3.copy()
        Pi_B, Pi_prev_B = z3.copy(), z3.copy()
        chi_B, chi_prev_B = chi.copy(), chi.copy()

        for _ in range(10):
            backend.step_color(
                Pr_A, Pr_prev_A, Pi_A, Pi_prev_A, chi_A, chi_prev_A,
                mask,
                Pr_B, Pr_prev_B, Pi_B, Pi_prev_B, chi_B, chi_prev_B,
                N, dt2, 1/63, 0.0, chi0, 0.0, 0.1,
            )
            Pr_A, Pr_B = Pr_B, Pr_A
            Pr_prev_A, Pr_prev_B = Pr_prev_B, Pr_prev_A
            Pi_A, Pi_B = Pi_B, Pi_A
            Pi_prev_A, Pi_prev_B = Pi_prev_B, Pi_prev_A
            chi_A, chi_B = chi_B, chi_A
            chi_prev_A, chi_prev_B = chi_prev_B, chi_prev_A

        np.testing.assert_allclose(chi_A, chi0, atol=1e-5)
