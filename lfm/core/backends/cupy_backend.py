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

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from typing import TYPE_CHECKING

from lfm.core.backends.kernel_source import (
    EVOLUTION_COMPLEX_KERNEL_SRC,
    EVOLUTION_KERNEL_SRC,
    EVOLUTION_REAL_KERNEL_SRC,
    PHASE1_KERNEL_SRC,
    SA_DIFFUSION_KERNEL_SRC,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
                "CuPy is required for GPU backend. Install with: pip install lfm-physics[gpu]"
            )
        # Compile kernels (cached by CuPy after first call)
        self._kernel_real = cp.RawKernel(
            EVOLUTION_REAL_KERNEL_SRC,
            "evolve_real",
        )
        self._kernel_complex = cp.RawKernel(
            EVOLUTION_COMPLEX_KERNEL_SRC,
            "evolve_complex",
        )
        self._kernel_color = cp.RawKernel(
            EVOLUTION_KERNEL_SRC,
            "evolve_gov01_gov02",
        )
        self._kernel_phase1 = cp.RawKernel(
            PHASE1_KERNEL_SRC,
            "phase1_parametric",
        )
        self._kernel_sa_diffusion = cp.RawKernel(
            SA_DIFFUSION_KERNEL_SRC,
            "evolve_sa_diffusion",
        )

    @property
    def name(self) -> str:
        return "cupy"

    def allocate(
        self,
        N: int,
        n_psi_arrays: int,
        chi0: float,
    ) -> dict[str, cp.ndarray]:
        total = N**3
        psi_size = n_psi_arrays * total
        zero_psi = cp.zeros(psi_size, dtype=cp.float32)
        chi_init = cp.full(total, chi0, dtype=cp.float32)
        return {
            "psi_A": zero_psi.copy(),
            "psi_prev_A": zero_psi.copy(),
            "chi_A": chi_init.copy(),
            "chi_prev_A": chi_init.copy(),
            "psi_B": zero_psi.copy(),
            "psi_prev_B": zero_psi.copy(),
            "chi_B": chi_init.copy(),
            "chi_prev_B": chi_init.copy(),
        }

    def create_boundary_mask(
        self,
        N: int,
        boundary_fraction: float,
    ) -> cp.ndarray:
        center = N / 2.0
        r_max = N / 2.0
        r_freeze = (1.0 - boundary_fraction) * r_max
        coords = np.arange(N, dtype=np.float32) - center + 0.5
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        R = np.sqrt(X**2 + Y**2 + Z**2)
        mask = (r_freeze < R).astype(np.float32).ravel()
        return cp.asarray(mask)

    def step_real(
        self,
        psi_in,
        psi_prev_in,
        chi_in,
        chi_prev_in,
        boundary_mask,
        psi_out,
        psi_prev_out,
        chi_out,
        chi_prev_out,
        N: int,
        dt2: float,
        kappa: float,
        lambda_self: float,
        chi0: float,
        e0_sq: float,
    ) -> None:
        total = N**3
        grid, block = _grid_block(total)
        self._kernel_real(
            grid,
            block,
            (
                psi_in,
                psi_prev_in,
                chi_in,
                chi_prev_in,
                boundary_mask,
                psi_out,
                psi_prev_out,
                chi_out,
                chi_prev_out,
                np.int32(N),
                np.float32(dt2),
                np.float32(kappa),
                np.float32(lambda_self),
                np.float32(chi0),
                np.float32(e0_sq),
            ),
        )
        # No synchronize() here — GPU ops are serialized on the default
        # stream and cp.asnumpy() / to_numpy() syncs when data is needed.

    def step_complex(
        self,
        psi_r_in,
        psi_r_prev_in,
        psi_i_in,
        psi_i_prev_in,
        chi_in,
        chi_prev_in,
        boundary_mask,
        psi_r_out,
        psi_r_prev_out,
        psi_i_out,
        psi_i_prev_out,
        chi_out,
        chi_prev_out,
        N: int,
        dt2: float,
        kappa: float,
        lambda_self: float,
        chi0: float,
        e0_sq: float,
        epsilon_w: float,
    ) -> None:
        total = N**3
        grid, block = _grid_block(total)
        self._kernel_complex(
            grid,
            block,
            (
                psi_r_in,
                psi_r_prev_in,
                psi_i_in,
                psi_i_prev_in,
                chi_in,
                chi_prev_in,
                boundary_mask,
                psi_r_out,
                psi_r_prev_out,
                psi_i_out,
                psi_i_prev_out,
                chi_out,
                chi_prev_out,
                np.int32(N),
                np.float32(dt2),
                np.float32(kappa),
                np.float32(lambda_self),
                np.float32(chi0),
                np.float32(e0_sq),
                np.float32(epsilon_w),
            ),
        )
        # No synchronize() here — syncs lazily on first CPU read.

    def step_color(
        self,
        psi_r_in,
        psi_r_prev_in,
        psi_i_in,
        psi_i_prev_in,
        chi_in,
        chi_prev_in,
        boundary_mask,
        psi_r_out,
        psi_r_prev_out,
        psi_i_out,
        psi_i_prev_out,
        chi_out,
        chi_prev_out,
        N: int,
        dt2: float,
        kappa: float,
        lambda_self: float,
        chi0: float,
        e0_sq: float,
        epsilon_w: float,
        kappa_c: float = 0.0,
        epsilon_cc: float = 0.0,
        # v16 S_a confinement fields
        kappa_string: float = 0.0,
        kappa_tube: float = 0.0,
        sa_fields_in=None,
        sa_fields_out=None,
        sa_gamma: float = 0.1,
        sa_d: float = 4.9,
        dt: float = 0.02,
    ) -> None:
        total = N**3
        grid, block = _grid_block(total)

        # v17: Helmholtz-smoothed S_a from |Ψ_a|² via FFT (replaces v16 Euler diffusion)
        if kappa_tube > 0.0 and sa_fields_in is not None:
            kx = cp.fft.fftfreq(N) * (2.0 * cp.pi)
            ky = cp.fft.fftfreq(N) * (2.0 * cp.pi)
            kz = cp.fft.rfftfreq(N) * (2.0 * cp.pi)
            k_sq = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
            h_filter = cp.float64(sa_gamma) / (cp.float64(sa_gamma) + cp.float64(sa_d) * k_sq)

            _sa_in = cp.zeros(3 * total, dtype=cp.float32)
            for a in range(3):
                s = slice(a * total, (a + 1) * total)
                psi_sq_a = psi_r_in[s] * psi_r_in[s] + psi_i_in[s] * psi_i_in[s]
                psi_sq_hat = cp.fft.rfftn(psi_sq_a.reshape(N, N, N))
                sa_3d = cp.fft.irfftn(h_filter * psi_sq_hat, s=(N, N, N))
                cp.clip(sa_3d, 0.0, None, out=sa_3d)
                _sa_in[s] = sa_3d.ravel().astype(cp.float32)
        else:
            _sa_in = (
                sa_fields_in if sa_fields_in is not None else cp.zeros(3 * total, dtype=cp.float32)
            )

        self._kernel_color(
            grid,
            block,
            (
                psi_r_in,
                psi_r_prev_in,
                psi_i_in,
                psi_i_prev_in,
                chi_in,
                chi_prev_in,
                boundary_mask,
                psi_r_out,
                psi_r_prev_out,
                psi_i_out,
                psi_i_prev_out,
                chi_out,
                chi_prev_out,
                np.int32(N),
                np.float32(dt2),
                np.float32(kappa),
                np.float32(lambda_self),
                np.float32(chi0),
                np.float32(e0_sq),
                np.float32(epsilon_w),
                np.float32(kappa_c),
                np.float32(epsilon_cc),
                _sa_in,
                np.float32(kappa_string),
                np.float32(kappa_tube),
            ),
        )

        # v17: copy Helmholtz-smoothed S_a to output buffer
        if kappa_tube > 0.0 and sa_fields_out is not None:
            cp.copyto(sa_fields_out, _sa_in)

        # No synchronize() here — syncs lazily on first CPU read.

    def step_phase1(
        self,
        psi_r_in,
        psi_r_prev_in,
        psi_i_in,
        psi_i_prev_in,
        psi_r_out,
        psi_r_prev_out,
        psi_i_out,
        psi_i_prev_out,
        N: int,
        dt2: float,
        chi_sq: float,
    ) -> None:
        """Phase 1 parametric resonance step (uniform oscillating χ)."""
        total = N**3
        grid, block = _grid_block(total)
        self._kernel_phase1(
            grid,
            block,
            (
                psi_r_in,
                psi_r_prev_in,
                psi_i_in,
                psi_i_prev_in,
                psi_r_out,
                psi_r_prev_out,
                psi_i_out,
                psi_i_prev_out,
                np.int32(N),
                np.float32(dt2),
                np.float32(chi_sq),
            ),
        )
        # No synchronize() here — syncs lazily on first CPU read.

    def synchronize(self) -> None:
        """Explicitly synchronise the GPU stream (use before timing or profiling)."""
        cp.cuda.Stream.null.synchronize()

    def to_numpy(self, arr) -> NDArray[np.float32]:
        return cp.asnumpy(arr)

    def from_numpy(self, arr: NDArray):
        return cp.asarray(arr, dtype=cp.float32)
