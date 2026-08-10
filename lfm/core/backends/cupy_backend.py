"""
CuPy Backend (GPU)
==================

GPU-accelerated LFM evolution using CuPy RawKernels.
These are the production CUDA kernels from the canonical universe simulator.

Uses double-buffering: kernel reads from A-set and writes to B-set (or vice versa).
Boundary mask uses the configured state dtype (1.0 = frozen, 0.0 = interior).
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

from lfm.config import Precision
from lfm.core.backends.kernel_source import (
    EVOLUTION_COMPLEX_KERNEL_SRC,
    EVOLUTION_KERNEL_SRC,
    EVOLUTION_REAL_KERNEL_SRC,
    GRAVITY_RECOVERY_REAL_KERNEL_SRC,
    PHASE1_KERNEL_SRC,
    SA_DIFFUSION_KERNEL_SRC,
    kernel_source_for_precision,
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

    def __init__(self, precision: Precision | str = Precision.FLOAT32) -> None:
        if not CUPY_AVAILABLE:
            raise ImportError(
                "CuPy is required for GPU backend. Install with: pip install lfm-physics[gpu]"
            )
        self.precision = Precision(precision)
        self.dtype = np.dtype(self.precision.value)
        self._array_dtype = cp.float32 if self.precision == Precision.FLOAT32 else cp.float64
        self._scalar_type = np.float32 if self.precision == Precision.FLOAT32 else np.float64

        def _source(source: str) -> str:
            return kernel_source_for_precision(source, self.precision.value)

        # Compile kernels (cached by CuPy after first call). The float32
        # source is returned unchanged to preserve the canonical baseline.
        self._kernel_real = cp.RawKernel(
            _source(EVOLUTION_REAL_KERNEL_SRC),
            "evolve_real",
        )
        self._kernel_real_gravity_recovery = cp.RawKernel(
            _source(GRAVITY_RECOVERY_REAL_KERNEL_SRC),
            "evolve_real_gravity_recovery",
        )
        self._kernel_complex = cp.RawKernel(
            _source(EVOLUTION_COMPLEX_KERNEL_SRC),
            "evolve_complex",
        )
        self._kernel_color = cp.RawKernel(
            _source(EVOLUTION_KERNEL_SRC),
            "evolve_gov01_gov02",
        )
        self._kernel_phase1 = cp.RawKernel(
            _source(PHASE1_KERNEL_SRC),
            "phase1_parametric",
        )
        self._kernel_sa_diffusion = cp.RawKernel(
            _source(SA_DIFFUSION_KERNEL_SRC),
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
        zero_psi = cp.zeros(psi_size, dtype=self._array_dtype)
        chi_init = cp.full(total, chi0, dtype=self._array_dtype)
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
        """Return a smooth cos² absorption mask in [0, 1].

        0 = fully transparent (interior), 1 = fully absorbed (boundary).
        A cosine taper prevents the leapfrog reflection of a hard cutoff.
        """
        center = N / 2.0
        r_max = N / 2.0
        r_freeze = (1.0 - boundary_fraction) * r_max
        coords = np.arange(N, dtype=self.dtype) - center + 0.5
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        R = np.sqrt(X**2 + Y**2 + Z**2)
        # Smooth cosine taper: 0 at r_freeze, 1 at r_max
        t = np.clip((R - r_freeze) / (r_max - r_freeze), 0.0, 1.0)
        mask = (np.sin(0.5 * np.pi * t) ** 2).astype(self.dtype).ravel()
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
        *,
        inv_dx2: float = 1.0,
        enable_chi_floor: bool = True,
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
                self._scalar_type(dt2),
                self._scalar_type(kappa),
                self._scalar_type(lambda_self),
                self._scalar_type(chi0),
                self._scalar_type(e0_sq),
                self._scalar_type(inv_dx2),
                np.int32(enable_chi_floor),
            ),
        )
        # No synchronize() here — GPU ops are serialized on the default
        # stream and cp.asnumpy() / to_numpy() syncs when data is needed.

    def step_real_gravity_recovery(
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
        dt: float,
        kappa: float,
        lambda_self: float,
        chi0: float,
        e0_sq: float,
        potential_model: int,
        freeze_psi: bool,
        relaxation_damping: float,
        *,
        inv_dx2: float = 1.0,
        enable_chi_floor: bool = False,
    ) -> None:
        """Advance one experiment-only local GOV-02 candidate step."""

        total = N**3
        grid, block = _grid_block(total)
        self._kernel_real_gravity_recovery(
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
                self._scalar_type(dt),
                self._scalar_type(kappa),
                self._scalar_type(lambda_self),
                self._scalar_type(chi0),
                self._scalar_type(e0_sq),
                np.int32(potential_model),
                np.int32(freeze_psi),
                self._scalar_type(relaxation_damping),
                self._scalar_type(inv_dx2),
                np.int32(enable_chi_floor),
            ),
        )

    def step_complex_gravity_recovery(
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
        dt: float,
        kappa: float,
        lambda_self: float,
        chi0: float,
        e0_sq: float,
        potential_model: int,
        freeze_psi: bool,
        relaxation_damping: float,
        *,
        inv_dx2: float = 1.0,
        enable_chi_floor: bool = False,
    ) -> None:
        """Advance the local complex flat-octic candidate system."""

        from lfm.config import ChiPotentialModel

        model = ChiPotentialModel(potential_model)
        if model != ChiPotentialModel.FLAT_OCTIC:
            raise ValueError("complex gravity recovery currently supports FLAT_OCTIC")
        self.step_complex(
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
            N,
            dt**2,
            kappa,
            0.0,
            chi0,
            e0_sq,
            0.0,
            use_stencil19_noether_current=False,
            inv_dx2=inv_dx2,
            enable_chi_floor=False,
        )
        absorb = 1.0 - boundary_mask
        safe_absorb = cp.where(absorb > 0.0, absorb, 1.0)
        undamped_chi = (chi_out - boundary_mask * chi0) / safe_absorb
        y = (chi_in * chi_in - chi0 * chi0) / (chi0 * chi0)
        flat_force = -8.0 * lambda_self * chi0**2 * chi_in * y**3
        damping_half_step = 0.5 * relaxation_damping * dt
        corrected = (undamped_chi + damping_half_step * chi_prev_in + dt**2 * flat_force) / (
            1.0 + damping_half_step
        )
        if enable_chi_floor:
            cp.maximum(corrected, -chi0, out=corrected)
        chi_out[...] = boundary_mask * chi0 + absorb * corrected
        if freeze_psi:
            psi_r_out[...] = absorb * psi_r_in
            psi_r_prev_out[...] = psi_r_in
            psi_i_out[...] = absorb * psi_i_in
            psi_i_prev_out[...] = psi_i_in

    def step_color_gravity_recovery(
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
        dt: float,
        kappa: float,
        lambda_self: float,
        chi0: float,
        e0_sq: float,
        potential_model: int,
        freeze_psi: bool,
        relaxation_damping: float,
        *,
        epsilon_w: float = 0.0,
        kappa_c: float = 0.0,
        epsilon_cc: float = 0.0,
        kappa_string: float = 0.0,
        kappa_tube: float = 0.0,
        sa_fields_in=None,
        sa_fields_out=None,
        sa_gamma: float = 0.1,
        sa_d: float = 4.9,
        use_stencil19_noether_current: bool = False,
        inv_dx2: float = 1.0,
        enable_chi_floor: bool = False,
    ) -> None:
        """Advance the full three-channel flat-octic candidate system."""

        from lfm.config import ChiPotentialModel

        model = ChiPotentialModel(potential_model)
        if model != ChiPotentialModel.FLAT_OCTIC:
            raise ValueError("color gravity recovery currently supports FLAT_OCTIC")
        self.step_color(
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
            N,
            dt**2,
            kappa,
            0.0,
            chi0,
            e0_sq,
            epsilon_w,
            kappa_c,
            epsilon_cc,
            kappa_string,
            kappa_tube,
            sa_fields_in,
            sa_fields_out,
            sa_gamma,
            sa_d,
            dt=dt,
            use_stencil19_noether_current=use_stencil19_noether_current,
            inv_dx2=inv_dx2,
            enable_chi_floor=False,
        )
        absorb = 1.0 - boundary_mask
        safe_absorb = cp.where(absorb > 0.0, absorb, 1.0)
        undamped_chi = (chi_out - boundary_mask * chi0) / safe_absorb
        y = (chi_in * chi_in - chi0 * chi0) / (chi0 * chi0)
        flat_force = -8.0 * lambda_self * chi0**2 * chi_in * y**3
        damping_half_step = 0.5 * relaxation_damping * dt
        corrected = (undamped_chi + damping_half_step * chi_prev_in + dt**2 * flat_force) / (
            1.0 + damping_half_step
        )
        if enable_chi_floor:
            cp.maximum(corrected, -chi0, out=corrected)
        chi_out[...] = boundary_mask * chi0 + absorb * corrected
        if freeze_psi:
            absorb3 = cp.tile(absorb, 3)
            psi_r_out[...] = absorb3 * psi_r_in
            psi_r_prev_out[...] = psi_r_in
            psi_i_out[...] = absorb3 * psi_i_in
            psi_i_prev_out[...] = psi_i_in

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
        *,
        use_stencil19_noether_current: bool = False,
        inv_dx2: float = 1.0,
        enable_chi_floor: bool = True,
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
                self._scalar_type(dt2),
                self._scalar_type(kappa),
                self._scalar_type(lambda_self),
                self._scalar_type(chi0),
                self._scalar_type(e0_sq),
                self._scalar_type(epsilon_w),
                np.int32(use_stencil19_noether_current),
                self._scalar_type(inv_dx2),
                np.int32(enable_chi_floor),
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
        *,
        use_stencil19_noether_current: bool = False,
        inv_dx2: float = 1.0,
        enable_chi_floor: bool = True,
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

            _sa_in = cp.zeros(3 * total, dtype=self._array_dtype)
            for a in range(3):
                s = slice(a * total, (a + 1) * total)
                psi_sq_a = psi_r_in[s] * psi_r_in[s] + psi_i_in[s] * psi_i_in[s]
                psi_sq_hat = cp.fft.rfftn(psi_sq_a.reshape(N, N, N))
                sa_3d = cp.fft.irfftn(h_filter * psi_sq_hat, s=(N, N, N))
                cp.clip(sa_3d, 0.0, None, out=sa_3d)
                _sa_in[s] = sa_3d.ravel().astype(self._array_dtype)
        else:
            _sa_in = (
                sa_fields_in
                if sa_fields_in is not None
                else cp.zeros(3 * total, dtype=self._array_dtype)
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
                self._scalar_type(dt2),
                self._scalar_type(kappa),
                self._scalar_type(lambda_self),
                self._scalar_type(chi0),
                self._scalar_type(e0_sq),
                self._scalar_type(epsilon_w),
                self._scalar_type(kappa_c),
                self._scalar_type(epsilon_cc),
                _sa_in,
                self._scalar_type(kappa_string),
                self._scalar_type(kappa_tube),
                np.int32(use_stencil19_noether_current),
                self._scalar_type(inv_dx2),
                np.int32(enable_chi_floor),
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
                self._scalar_type(dt2),
                self._scalar_type(chi_sq),
            ),
        )
        # No synchronize() here — syncs lazily on first CPU read.

    def synchronize(self) -> None:
        """Explicitly synchronise the GPU stream (use before timing or profiling)."""
        cp.cuda.Stream.null.synchronize()

    def to_numpy(self, arr) -> NDArray[np.floating]:
        return cp.asnumpy(arr)

    def from_numpy(self, arr: NDArray):
        return cp.asarray(arr, dtype=self._array_dtype)

    def apply_complex_phase_map(
        self,
        psi_r,
        psi_i,
        phase_cos,
        phase_sin,
    ) -> None:
        """Rotate a complex field in place by a local phase map."""

        real = psi_r.copy()
        imag = psi_i.copy()
        psi_r[:] = real * phase_cos - imag * phase_sin
        psi_i[:] = real * phase_sin + imag * phase_cos
