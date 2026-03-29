"""GPU ↔ CPU parity tests.

Verify that CuPy (GPU) and NumPy (CPU) backends produce identical physics
for all three field levels: real, complex, and color.

These tests are marked ``@pytest.mark.gpu`` and auto-skip when CuPy is
not installed, so they never block CPU-only CI.  Run locally before
releases::

    pytest -m gpu -v
"""

from __future__ import annotations

import numpy as np
import pytest

from lfm.core.backends import NumpyBackend, gpu_available

# Skip entire module when no GPU is available
pytestmark = pytest.mark.gpu

try:
    from lfm.core.backends.cupy_backend import CupyBackend
except ImportError:
    CupyBackend = None  # type: ignore[misc,assignment]

_N = 16
_CHI0 = 19.0
_KAPPA = 1.0 / 63.0
_DT = 0.02
_DT2 = _DT**2
_LAMBDA_SELF = 0.0
_E0_SQ = 0.0
_EPSILON_W = 0.1
_STEPS = 50
_BOUNDARY_FRAC = 0.15

# Tolerance: float32 GPU kernels vs float32 NumPy — allow ~1e-4 drift
_ATOL = 2e-4
_RTOL = 1e-4


def _require_gpu():
    if not gpu_available() or CupyBackend is None:
        pytest.skip("CuPy/GPU not available")


def _seed_energy(total: int, N: int, rng: np.random.Generator) -> np.ndarray:
    """Small Gaussian blob at center, matching float32."""
    E = np.zeros(total, dtype=np.float32)
    c = N // 2
    for di in range(-2, 3):
        for dj in range(-2, 3):
            for dk in range(-2, 3):
                i, j, k = c + di, c + dj, c + dk
                r2 = di * di + dj * dj + dk * dk
                idx = i * N * N + j * N + k
                E[idx] = 3.0 * np.exp(-r2 / 2.0)
    return E


class TestRealFieldParity:
    """Level 0 (real E) — step_real parity."""

    def test_real_parity(self):
        _require_gpu()
        total = _N**3
        rng = np.random.default_rng(42)

        cpu = NumpyBackend()
        gpu = CupyBackend()

        mask_np = cpu.create_boundary_mask(_N, _BOUNDARY_FRAC)
        mask_gpu = gpu.from_numpy(mask_np)

        E_seed = _seed_energy(total, _N, rng)

        # CPU buffers
        cE_A = E_seed.copy()
        cE_prev_A = E_seed.copy()
        cChi_A = np.full(total, _CHI0, dtype=np.float32)
        cChi_prev_A = np.full(total, _CHI0, dtype=np.float32)
        cE_B = np.zeros(total, dtype=np.float32)
        cE_prev_B = np.zeros(total, dtype=np.float32)
        cChi_B = np.full(total, _CHI0, dtype=np.float32)
        cChi_prev_B = np.full(total, _CHI0, dtype=np.float32)

        # GPU buffers (copy same initial state)
        gE_A = gpu.from_numpy(E_seed.copy())
        gE_prev_A = gpu.from_numpy(E_seed.copy())
        gChi_A = gpu.from_numpy(np.full(total, _CHI0, dtype=np.float32))
        gChi_prev_A = gpu.from_numpy(np.full(total, _CHI0, dtype=np.float32))
        gE_B = gpu.from_numpy(np.zeros(total, dtype=np.float32))
        gE_prev_B = gpu.from_numpy(np.zeros(total, dtype=np.float32))
        gChi_B = gpu.from_numpy(np.full(total, _CHI0, dtype=np.float32))
        gChi_prev_B = gpu.from_numpy(np.full(total, _CHI0, dtype=np.float32))

        for _ in range(_STEPS):
            cpu.step_real(
                cE_A,
                cE_prev_A,
                cChi_A,
                cChi_prev_A,
                mask_np,
                cE_B,
                cE_prev_B,
                cChi_B,
                cChi_prev_B,
                _N,
                _DT2,
                _KAPPA,
                _LAMBDA_SELF,
                _CHI0,
                _E0_SQ,
            )
            gpu.step_real(
                gE_A,
                gE_prev_A,
                gChi_A,
                gChi_prev_A,
                mask_gpu,
                gE_B,
                gE_prev_B,
                gChi_B,
                gChi_prev_B,
                _N,
                _DT2,
                _KAPPA,
                _LAMBDA_SELF,
                _CHI0,
                _E0_SQ,
            )
            # Swap A↔B
            cE_A, cE_B = cE_B, cE_A
            cE_prev_A, cE_prev_B = cE_prev_B, cE_prev_A
            cChi_A, cChi_B = cChi_B, cChi_A
            cChi_prev_A, cChi_prev_B = cChi_prev_B, cChi_prev_A

            gE_A, gE_B = gE_B, gE_A
            gE_prev_A, gE_prev_B = gE_prev_B, gE_prev_A
            gChi_A, gChi_B = gChi_B, gChi_A
            gChi_prev_A, gChi_prev_B = gChi_prev_B, gChi_prev_A

        cpu_E = cE_A
        gpu_E = gpu.to_numpy(gE_A)
        cpu_chi = cChi_A
        gpu_chi = gpu.to_numpy(gChi_A)

        np.testing.assert_allclose(
            cpu_E, gpu_E, atol=_ATOL, rtol=_RTOL, err_msg="E field diverged (real)"
        )
        np.testing.assert_allclose(
            cpu_chi, gpu_chi, atol=_ATOL, rtol=_RTOL, err_msg="chi field diverged (real)"
        )


class TestComplexFieldParity:
    """Level 1 (complex Ψ) — step_complex parity."""

    def test_complex_parity(self):
        _require_gpu()
        total = _N**3
        rng = np.random.default_rng(42)

        cpu = NumpyBackend()
        gpu = CupyBackend()

        mask_np = cpu.create_boundary_mask(_N, _BOUNDARY_FRAC)
        mask_gpu = gpu.from_numpy(mask_np)

        Pr_seed = _seed_energy(total, _N, rng)
        Pi_seed = np.zeros(total, dtype=np.float32)  # pure real initially

        # CPU buffers
        cPr_A, cPr_prev_A = Pr_seed.copy(), Pr_seed.copy()
        cPi_A, cPi_prev_A = Pi_seed.copy(), Pi_seed.copy()
        cChi_A = np.full(total, _CHI0, dtype=np.float32)
        cChi_prev_A = cChi_A.copy()
        cPr_B, cPr_prev_B = np.zeros_like(Pr_seed), np.zeros_like(Pr_seed)
        cPi_B, cPi_prev_B = np.zeros_like(Pi_seed), np.zeros_like(Pi_seed)
        cChi_B, cChi_prev_B = cChi_A.copy(), cChi_A.copy()

        # GPU buffers
        gPr_A = gpu.from_numpy(Pr_seed.copy())
        gPr_prev_A = gpu.from_numpy(Pr_seed.copy())
        gPi_A = gpu.from_numpy(Pi_seed.copy())
        gPi_prev_A = gpu.from_numpy(Pi_seed.copy())
        gChi_A = gpu.from_numpy(np.full(total, _CHI0, dtype=np.float32))
        gChi_prev_A = gpu.from_numpy(np.full(total, _CHI0, dtype=np.float32))
        gPr_B = gpu.from_numpy(np.zeros(total, dtype=np.float32))
        gPr_prev_B = gpu.from_numpy(np.zeros(total, dtype=np.float32))
        gPi_B = gpu.from_numpy(np.zeros(total, dtype=np.float32))
        gPi_prev_B = gpu.from_numpy(np.zeros(total, dtype=np.float32))
        gChi_B = gpu.from_numpy(np.full(total, _CHI0, dtype=np.float32))
        gChi_prev_B = gpu.from_numpy(np.full(total, _CHI0, dtype=np.float32))

        for _ in range(_STEPS):
            cpu.step_complex(
                cPr_A,
                cPr_prev_A,
                cPi_A,
                cPi_prev_A,
                cChi_A,
                cChi_prev_A,
                mask_np,
                cPr_B,
                cPr_prev_B,
                cPi_B,
                cPi_prev_B,
                cChi_B,
                cChi_prev_B,
                _N,
                _DT2,
                _KAPPA,
                _LAMBDA_SELF,
                _CHI0,
                _E0_SQ,
                _EPSILON_W,
            )
            gpu.step_complex(
                gPr_A,
                gPr_prev_A,
                gPi_A,
                gPi_prev_A,
                gChi_A,
                gChi_prev_A,
                mask_gpu,
                gPr_B,
                gPr_prev_B,
                gPi_B,
                gPi_prev_B,
                gChi_B,
                gChi_prev_B,
                _N,
                _DT2,
                _KAPPA,
                _LAMBDA_SELF,
                _CHI0,
                _E0_SQ,
                _EPSILON_W,
            )
            # Swap
            cPr_A, cPr_B = cPr_B, cPr_A
            cPr_prev_A, cPr_prev_B = cPr_prev_B, cPr_prev_A
            cPi_A, cPi_B = cPi_B, cPi_A
            cPi_prev_A, cPi_prev_B = cPi_prev_B, cPi_prev_A
            cChi_A, cChi_B = cChi_B, cChi_A
            cChi_prev_A, cChi_prev_B = cChi_prev_B, cChi_prev_A

            gPr_A, gPr_B = gPr_B, gPr_A
            gPr_prev_A, gPr_prev_B = gPr_prev_B, gPr_prev_A
            gPi_A, gPi_B = gPi_B, gPi_A
            gPi_prev_A, gPi_prev_B = gPi_prev_B, gPi_prev_A
            gChi_A, gChi_B = gChi_B, gChi_A
            gChi_prev_A, gChi_prev_B = gChi_prev_B, gChi_prev_A

        np.testing.assert_allclose(
            cPr_A,
            gpu.to_numpy(gPr_A),
            atol=_ATOL,
            rtol=_RTOL,
            err_msg="Psi_real diverged (complex)",
        )
        np.testing.assert_allclose(
            cPi_A,
            gpu.to_numpy(gPi_A),
            atol=_ATOL,
            rtol=_RTOL,
            err_msg="Psi_imag diverged (complex)",
        )
        np.testing.assert_allclose(
            cChi_A,
            gpu.to_numpy(gChi_A),
            atol=_ATOL,
            rtol=_RTOL,
            err_msg="chi diverged (complex)",
        )


class TestColorFieldParity:
    """Level 2 (color Ψₐ) — step_color parity."""

    def test_color_parity(self):
        _require_gpu()
        total = _N**3
        rng = np.random.default_rng(42)

        cpu = NumpyBackend()
        gpu = CupyBackend()

        mask_np = cpu.create_boundary_mask(_N, _BOUNDARY_FRAC)
        mask_gpu = gpu.from_numpy(mask_np)

        # 3 colors × N³ for real and imag
        Pr_seed = np.zeros(3 * total, dtype=np.float32)
        Pi_seed = np.zeros(3 * total, dtype=np.float32)
        # Put energy in color 0 only (center blob)
        E_blob = _seed_energy(total, _N, rng)
        Pr_seed[:total] = E_blob

        # CPU buffers
        cPr_A, cPr_prev_A = Pr_seed.copy(), Pr_seed.copy()
        cPi_A, cPi_prev_A = Pi_seed.copy(), Pi_seed.copy()
        cChi_A = np.full(total, _CHI0, dtype=np.float32)
        cChi_prev_A = cChi_A.copy()
        cPr_B = np.zeros_like(Pr_seed)
        cPr_prev_B = np.zeros_like(Pr_seed)
        cPi_B = np.zeros_like(Pi_seed)
        cPi_prev_B = np.zeros_like(Pi_seed)
        cChi_B, cChi_prev_B = cChi_A.copy(), cChi_A.copy()

        # GPU buffers
        gPr_A = gpu.from_numpy(Pr_seed.copy())
        gPr_prev_A = gpu.from_numpy(Pr_seed.copy())
        gPi_A = gpu.from_numpy(Pi_seed.copy())
        gPi_prev_A = gpu.from_numpy(Pi_seed.copy())
        gChi_A = gpu.from_numpy(np.full(total, _CHI0, dtype=np.float32))
        gChi_prev_A = gpu.from_numpy(np.full(total, _CHI0, dtype=np.float32))
        gPr_B = gpu.from_numpy(np.zeros(3 * total, dtype=np.float32))
        gPr_prev_B = gpu.from_numpy(np.zeros(3 * total, dtype=np.float32))
        gPi_B = gpu.from_numpy(np.zeros(3 * total, dtype=np.float32))
        gPi_prev_B = gpu.from_numpy(np.zeros(3 * total, dtype=np.float32))
        gChi_B = gpu.from_numpy(np.full(total, _CHI0, dtype=np.float32))
        gChi_prev_B = gpu.from_numpy(np.full(total, _CHI0, dtype=np.float32))

        kappa_c = 1.0 / 189.0

        for _ in range(_STEPS):
            cpu.step_color(
                cPr_A,
                cPr_prev_A,
                cPi_A,
                cPi_prev_A,
                cChi_A,
                cChi_prev_A,
                mask_np,
                cPr_B,
                cPr_prev_B,
                cPi_B,
                cPi_prev_B,
                cChi_B,
                cChi_prev_B,
                _N,
                _DT2,
                _KAPPA,
                _LAMBDA_SELF,
                _CHI0,
                _E0_SQ,
                _EPSILON_W,
                kappa_c,
            )
            gpu.step_color(
                gPr_A,
                gPr_prev_A,
                gPi_A,
                gPi_prev_A,
                gChi_A,
                gChi_prev_A,
                mask_gpu,
                gPr_B,
                gPr_prev_B,
                gPi_B,
                gPi_prev_B,
                gChi_B,
                gChi_prev_B,
                _N,
                _DT2,
                _KAPPA,
                _LAMBDA_SELF,
                _CHI0,
                _E0_SQ,
                _EPSILON_W,
                kappa_c,
            )
            # Swap
            cPr_A, cPr_B = cPr_B, cPr_A
            cPr_prev_A, cPr_prev_B = cPr_prev_B, cPr_prev_A
            cPi_A, cPi_B = cPi_B, cPi_A
            cPi_prev_A, cPi_prev_B = cPi_prev_B, cPi_prev_A
            cChi_A, cChi_B = cChi_B, cChi_A
            cChi_prev_A, cChi_prev_B = cChi_prev_B, cChi_prev_A

            gPr_A, gPr_B = gPr_B, gPr_A
            gPr_prev_A, gPr_prev_B = gPr_prev_B, gPr_prev_A
            gPi_A, gPi_B = gPi_B, gPi_A
            gPi_prev_A, gPi_prev_B = gPi_prev_B, gPi_prev_A
            gChi_A, gChi_B = gChi_B, gChi_A
            gChi_prev_A, gChi_prev_B = gChi_prev_B, gChi_prev_A

        np.testing.assert_allclose(
            cPr_A,
            gpu.to_numpy(gPr_A),
            atol=_ATOL,
            rtol=_RTOL,
            err_msg="Psi_real diverged (color)",
        )
        np.testing.assert_allclose(
            cPi_A,
            gpu.to_numpy(gPi_A),
            atol=_ATOL,
            rtol=_RTOL,
            err_msg="Psi_imag diverged (color)",
        )
        np.testing.assert_allclose(
            cChi_A,
            gpu.to_numpy(gChi_A),
            atol=_ATOL,
            rtol=_RTOL,
            err_msg="chi diverged (color)",
        )


class TestBoundaryMaskParity:
    """Boundary mask: GPU and CPU produce identical masks."""

    def test_mask_values_match(self):
        _require_gpu()
        cpu = NumpyBackend()
        gpu = CupyBackend()

        cpu_mask = cpu.create_boundary_mask(_N, _BOUNDARY_FRAC)
        gpu_mask = gpu.to_numpy(gpu.create_boundary_mask(_N, _BOUNDARY_FRAC))

        np.testing.assert_array_equal(cpu_mask, gpu_mask, err_msg="Boundary masks differ")


class TestAllocateParity:
    """Allocate: GPU and CPU produce identical initial arrays."""

    def test_allocate_shapes_and_values(self):
        _require_gpu()
        cpu = NumpyBackend()
        gpu = CupyBackend()

        cpu_bufs = cpu.allocate(_N, 2, _CHI0)
        gpu_bufs = gpu.allocate(_N, 2, _CHI0)

        for key in cpu_bufs:
            cpu_arr = np.asarray(cpu_bufs[key])
            gpu_arr = gpu.to_numpy(gpu_bufs[key])
            assert cpu_arr.shape == gpu_arr.shape, f"Shape mismatch for {key}"
            np.testing.assert_array_equal(cpu_arr, gpu_arr, err_msg=f"Values differ for {key}")


class TestHighLevelSimulationParity:
    """End-to-end: Simulation with cpu vs gpu backend gives same chi_min."""

    def test_simulation_chi_parity(self):
        _require_gpu()
        from lfm import FieldLevel, Simulation, SimulationConfig

        cfg = SimulationConfig(
            grid_size=16,
            e_amplitude=6.0,
            field_level=FieldLevel.REAL,
        )

        sim_cpu = Simulation(cfg, backend="cpu")
        sim_gpu = Simulation(cfg, backend="gpu")

        center = (8.0, 8.0, 8.0)
        sim_cpu.place_soliton(center)
        sim_gpu.place_soliton(center)
        sim_cpu.equilibrate()
        sim_gpu.equilibrate()

        sim_cpu.run(steps=100)
        sim_gpu.run(steps=100)

        chi_cpu = sim_cpu.chi
        chi_gpu = sim_gpu.chi

        np.testing.assert_allclose(
            chi_cpu,
            chi_gpu,
            atol=5e-4,
            rtol=1e-3,
            err_msg="High-level Simulation chi diverged between CPU and GPU",
        )
