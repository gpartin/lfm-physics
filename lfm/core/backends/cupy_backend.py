"""
CuPy Backend (GPU)
==================

GPU-accelerated LFM evolution using CuPy RawKernels.
These are the production CUDA kernels from the canonical universe simulator.

Uses double-buffering: kernel reads from A-set and writes to B-set (or vice versa).
Boundary mask is a float32 array (1.0 = frozen, 0.0 = interior).
Block size = 256 threads, 1D grid — optimal for RTX 4060.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from lfm.core.backends.kernel_source import (
    EVOLUTION_COMPLEX_KERNEL_SRC,
    EVOLUTION_KERNEL_SRC,
    EVOLUTION_REAL_KERNEL_SRC,
    PHASE1_KERNEL_SRC,
)

# Block size — 256 is optimal for most NVIDIA GPUs
_BLOCK_SIZE = 256


def _grid_block(total: int) -> tuple[tuple[int], tuple[int]]:
    """Compute 1D CUDA grid and block dimensions."""
    blocks = (total + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    return (blocks,), (_BLOCK_SIZE,)


class CupyBackend:
    """GPU compute backend using CuPy + CUDA RawKernels.

    Raises ImportError if cupy is not installed.
    """

    def __init__(self) -> None:
        if not CUPY_AVAILABLE:
            raise ImportError(
                "CuPy is required for GPU backend. "
                "Install with: pip install lfm-physics[gpu]"
            )
        # Compile kernels (cached by CuPy after first call)
        self._kernel_real = cp.RawKernel(
            EVOLUTION_REAL_KERNEL_SRC, "evolve_real",
        )
        self._kernel_complex = cp.RawKernel(
            EVOLUTION_COMPLEX_KERNEL_SRC, "evolve_complex",
        )
        self._kernel_color = cp.RawKernel(
            EVOLUTION_KERNEL_SRC, "evolve_gov01_gov02",
        )
        self._kernel_phase1 = cp.RawKernel(
            PHASE1_KERNEL_SRC, "phase1_parametric",
        )

    @property
    def name(self) -> str:
        return "cupy"

    def allocate(
        self, N: int, n_psi_arrays: int, chi0: float,
    ) -> dict[str, cp.ndarray]:
        total = N**3
        psi_size = n_psi_arrays * total
        zero_psi = cp.zeros(psi_size, dtype=cp.float32)
        chi_init = cp.full(total, chi0, dtype=cp.float32)
        return {
            "psi_A": zero_psi.copy(), "psi_prev_A": zero_psi.copy(),
            "chi_A": chi_init.copy(), "chi_prev_A": chi_init.copy(),
            "psi_B": zero_psi.copy(), "psi_prev_B": zero_psi.copy(),
            "chi_B": chi_init.copy(), "chi_prev_B": chi_init.copy(),
        }

    def create_boundary_mask(
        self, N: int, boundary_fraction: float,
    ) -> cp.ndarray:
        center = N / 2.0
        r_max = N / 2.0
        r_freeze = (1.0 - boundary_fraction) * r_max
        coords = np.arange(N, dtype=np.float32) - center + 0.5
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        R = np.sqrt(X**2 + Y**2 + Z**2)
        mask = (R > r_freeze).astype(np.float32).ravel()
        return cp.asarray(mask)

    def step_real(
        self,
        psi_in, psi_prev_in,
        chi_in, chi_prev_in,
        boundary_mask,
        psi_out, psi_prev_out,
        chi_out, chi_prev_out,
        N: int, dt2: float, kappa: float,
        lambda_self: float, chi0: float, e0_sq: float,
    ) -> None:
        total = N**3
        grid, block = _grid_block(total)
        self._kernel_real(grid, block, (
            psi_in, psi_prev_in,
            chi_in, chi_prev_in,
            boundary_mask,
            psi_out, psi_prev_out,
            chi_out, chi_prev_out,
            np.int32(N), np.float32(dt2), np.float32(kappa),
            np.float32(lambda_self), np.float32(chi0), np.float32(e0_sq),
        ))
        cp.cuda.Stream.null.synchronize()

    def step_complex(
        self,
        psi_r_in, psi_r_prev_in,
        psi_i_in, psi_i_prev_in,
        chi_in, chi_prev_in,
        boundary_mask,
        psi_r_out, psi_r_prev_out,
        psi_i_out, psi_i_prev_out,
        chi_out, chi_prev_out,
        N: int, dt2: float, kappa: float,
        lambda_self: float, chi0: float, e0_sq: float,
        epsilon_w: float,
    ) -> None:
        total = N**3
        grid, block = _grid_block(total)
        self._kernel_complex(grid, block, (
            psi_r_in, psi_r_prev_in,
            psi_i_in, psi_i_prev_in,
            chi_in, chi_prev_in,
            boundary_mask,
            psi_r_out, psi_r_prev_out,
            psi_i_out, psi_i_prev_out,
            chi_out, chi_prev_out,
            np.int32(N), np.float32(dt2), np.float32(kappa),
            np.float32(lambda_self), np.float32(chi0), np.float32(e0_sq),
            np.float32(epsilon_w),
        ))
        cp.cuda.Stream.null.synchronize()

    def step_color(
        self,
        psi_r_in, psi_r_prev_in,
        psi_i_in, psi_i_prev_in,
        chi_in, chi_prev_in,
        boundary_mask,
        psi_r_out, psi_r_prev_out,
        psi_i_out, psi_i_prev_out,
        chi_out, chi_prev_out,
        N: int, dt2: float, kappa: float,
        lambda_self: float, chi0: float, e0_sq: float,
        epsilon_w: float,
    ) -> None:
        total = N**3
        grid, block = _grid_block(total)
        self._kernel_color(grid, block, (
            psi_r_in, psi_r_prev_in,
            psi_i_in, psi_i_prev_in,
            chi_in, chi_prev_in,
            boundary_mask,
            psi_r_out, psi_r_prev_out,
            psi_i_out, psi_i_prev_out,
            chi_out, chi_prev_out,
            np.int32(N), np.float32(dt2), np.float32(kappa),
            np.float32(lambda_self), np.float32(chi0), np.float32(e0_sq),
            np.float32(epsilon_w),
        ))
        cp.cuda.Stream.null.synchronize()

    def step_phase1(
        self,
        psi_r_in, psi_r_prev_in,
        psi_i_in, psi_i_prev_in,
        psi_r_out, psi_r_prev_out,
        psi_i_out, psi_i_prev_out,
        N: int, dt2: float, chi_sq: float,
    ) -> None:
        """Phase 1 parametric resonance step (uniform oscillating χ)."""
        total = N**3
        grid, block = _grid_block(total)
        self._kernel_phase1(grid, block, (
            psi_r_in, psi_r_prev_in,
            psi_i_in, psi_i_prev_in,
            psi_r_out, psi_r_prev_out,
            psi_i_out, psi_i_prev_out,
            np.int32(N), np.float32(dt2), np.float32(chi_sq),
        ))
        cp.cuda.Stream.null.synchronize()

    def to_numpy(self, arr) -> NDArray[np.float32]:
        return cp.asnumpy(arr)

    def from_numpy(self, arr: NDArray):
        return cp.asarray(arr, dtype=cp.float32)
