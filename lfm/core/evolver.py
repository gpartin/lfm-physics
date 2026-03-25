"""
Backend-Powered Evolver
=======================

High-performance evolution loop using the backend system.
Matches the production double-buffering pattern from the canonical
universe simulator. Supports CPU (NumPy) and GPU (CuPy) backends.

For simple use, the existing ``step_leapfrog`` in integrator.py is
still available. This module is for performance-critical loops.

Usage::

    from lfm.config import SimulationConfig
    from lfm.core.evolver import Evolver

    config = SimulationConfig(grid_size=128)
    evolver = Evolver(config)              # auto-detect GPU
    evolver = Evolver(config, backend="cpu")  # force CPU

    # Run 10000 steps
    evolver.evolve(10000)

    # Get numpy arrays for analysis
    chi = evolver.get_chi()       # shape (N, N, N)
    psi = evolver.get_psi()       # shape depends on field_level
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lfm.config import FieldLevel, SimulationConfig
from lfm.core.backends import get_backend


class Evolver:
    """Double-buffered LFM evolver using the backend system.

    This class manages GPU/CPU arrays, boundary masks, and the
    A/B buffer toggle — matching production code exactly.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration.
    backend : str
        Backend preference: 'auto', 'cpu', or 'gpu'.
    """

    def __init__(
        self,
        config: SimulationConfig,
        backend: str = "auto",
    ) -> None:
        self.config = config
        self.backend = get_backend(backend)
        self.N = config.grid_size
        self.total = self.N**3
        self.step = 0
        self._use_buffer_A = True

        # Determine number of psi component arrays
        if config.field_level == FieldLevel.REAL:
            self._n_psi = 1  # Just E
        elif config.field_level == FieldLevel.COMPLEX:
            self._n_psi = 1  # Single complex (Pr and Pi separate)
        else:
            self._n_psi = config.n_colors  # 3 colors

        # For complex/color: we have separate real and imaginary arrays
        # For real: psi_r is E, psi_i is unused (zero)
        self._has_imag = config.field_level != FieldLevel.REAL

        # Pre-compute scalar parameters
        self._dt2 = config.dt**2
        self._N = config.grid_size

        # Allocate arrays via backend
        self._init_arrays()

    def _init_arrays(self) -> None:
        """Allocate double-buffered arrays."""
        N = self.N
        total = self.total
        cfg = self.config

        if cfg.field_level == FieldLevel.REAL:
            psi_size = total
        elif cfg.field_level == FieldLevel.COMPLEX:
            psi_size = total
        else:  # COLOR
            psi_size = cfg.n_colors * total

        xp = self.backend

        # Psi real part — A and B buffers
        self.psi_r_A = xp.from_numpy(np.zeros(psi_size, dtype=np.float32))
        self.psi_r_prev_A = xp.from_numpy(np.zeros(psi_size, dtype=np.float32))
        self.psi_r_B = xp.from_numpy(np.zeros(psi_size, dtype=np.float32))
        self.psi_r_prev_B = xp.from_numpy(np.zeros(psi_size, dtype=np.float32))

        # Psi imaginary part (zero for real field, but allocated for uniform API)
        if self._has_imag:
            self.psi_i_A = xp.from_numpy(np.zeros(psi_size, dtype=np.float32))
            self.psi_i_prev_A = xp.from_numpy(np.zeros(psi_size, dtype=np.float32))
            self.psi_i_B = xp.from_numpy(np.zeros(psi_size, dtype=np.float32))
            self.psi_i_prev_B = xp.from_numpy(np.zeros(psi_size, dtype=np.float32))
        else:
            self.psi_i_A = None
            self.psi_i_prev_A = None
            self.psi_i_B = None
            self.psi_i_prev_B = None

        # Chi — A and B buffers
        chi_init = np.full(total, cfg.chi0, dtype=np.float32)
        self.chi_A = xp.from_numpy(chi_init.copy())
        self.chi_prev_A = xp.from_numpy(chi_init.copy())
        self.chi_B = xp.from_numpy(chi_init.copy())
        self.chi_prev_B = xp.from_numpy(chi_init.copy())

        # Boundary mask
        self.boundary_mask = xp.create_boundary_mask(N, cfg.boundary_fraction)

        # S_a auxiliary fields for v16 flux-tube confinement (COLOR field level only)
        if cfg.sa_enabled and cfg.field_level == FieldLevel.COLOR:
            sa_init = np.zeros(cfg.n_colors * total, dtype=np.float32)
            self.sa_A = xp.from_numpy(sa_init.copy())
            self.sa_B = xp.from_numpy(sa_init.copy())
        else:
            self.sa_A = None
            self.sa_B = None

    def evolve(self, steps: int, callback=None, freeze_chi: bool = False) -> None:
        """Run the evolution loop for a given number of steps.

        Parameters
        ----------
        steps : int
            Number of leapfrog steps.
        callback : callable, optional
            Called as callback(evolver, step) every report_interval steps.
        freeze_chi : bool
            If True, chi is held frozen at its value at the start of
            this call.  Only GOV-01 (Psi update) is applied; the chi
            arrays are restored to the frozen snapshot after every step.
            Used by the eigenmode SCF solver.
        """
        cfg = self.config
        report = cfg.report_interval
        chi_frozen = self.get_chi().copy() if freeze_chi else None

        for i in range(steps):
            self._step()
            if freeze_chi:
                self.set_chi(chi_frozen)
            self.step += 1

            if callback is not None and report > 0 and self.step % report == 0:
                callback(self, self.step)

    def _step(self) -> None:
        """Execute one leapfrog step with double-buffer toggle."""
        cfg = self.config

        if self._use_buffer_A:
            r_in, rp_in = self.psi_r_A, self.psi_r_prev_A
            r_out, rp_out = self.psi_r_B, self.psi_r_prev_B
            i_in, ip_in = self.psi_i_A, self.psi_i_prev_A
            i_out, ip_out = self.psi_i_B, self.psi_i_prev_B
            c_in, cp_in = self.chi_A, self.chi_prev_A
            c_out, cp_out = self.chi_B, self.chi_prev_B
        else:
            r_in, rp_in = self.psi_r_B, self.psi_r_prev_B
            r_out, rp_out = self.psi_r_A, self.psi_r_prev_A
            i_in, ip_in = self.psi_i_B, self.psi_i_prev_B
            i_out, ip_out = self.psi_i_A, self.psi_i_prev_A
            c_in, cp_in = self.chi_B, self.chi_prev_B
            c_out, cp_out = self.chi_A, self.chi_prev_A

        if cfg.field_level == FieldLevel.REAL:
            self.backend.step_real(
                r_in,
                rp_in,
                c_in,
                cp_in,
                self.boundary_mask,
                r_out,
                rp_out,
                c_out,
                cp_out,
                self._N,
                self._dt2,
                cfg.kappa,
                cfg.lambda_self,
                cfg.chi0,
                cfg.e0_sq,
            )
        elif cfg.field_level == FieldLevel.COMPLEX:
            self.backend.step_complex(
                r_in,
                rp_in,
                i_in,
                ip_in,
                c_in,
                cp_in,
                self.boundary_mask,
                r_out,
                rp_out,
                i_out,
                ip_out,
                c_out,
                cp_out,
                self._N,
                self._dt2,
                cfg.kappa,
                cfg.lambda_self,
                cfg.chi0,
                cfg.e0_sq,
                cfg.epsilon_w,
            )
        else:  # COLOR
            sa_in = (
                (self.sa_A if self._use_buffer_A else self.sa_B) if self.sa_A is not None else None
            )
            sa_out = (
                (self.sa_B if self._use_buffer_A else self.sa_A) if self.sa_A is not None else None
            )
            self.backend.step_color(
                r_in,
                rp_in,
                i_in,
                ip_in,
                c_in,
                cp_in,
                self.boundary_mask,
                r_out,
                rp_out,
                i_out,
                ip_out,
                c_out,
                cp_out,
                self._N,
                self._dt2,
                cfg.kappa,
                cfg.lambda_self,
                cfg.chi0,
                cfg.e0_sq,
                cfg.epsilon_w,
                cfg.kappa_c,
                cfg.epsilon_cc,
                kappa_string=cfg.kappa_string,
                kappa_tube=cfg.kappa_tube,
                sa_fields_in=sa_in,
                sa_fields_out=sa_out,
                sa_gamma=cfg.sa_gamma,
                sa_d=cfg.sa_d,
                dt=cfg.dt,
            )

        self._use_buffer_A = not self._use_buffer_A

    # --- Field accessors (return numpy arrays) ---

    def _current_buf(self) -> str:
        """Which buffer holds the most recent result."""
        return "A" if self._use_buffer_A else "B"

    def get_chi(self) -> NDArray[np.float32]:
        """Get current χ field as numpy array, shape (N, N, N)."""
        if self._use_buffer_A:
            flat = self.backend.to_numpy(self.chi_A)
        else:
            flat = self.backend.to_numpy(self.chi_B)
        return flat.reshape(self.N, self.N, self.N)

    def get_psi_real(self) -> NDArray[np.float32]:
        """Get real part of Ψ as numpy array.

        Shape: (N,N,N) for REAL/COMPLEX, (n_colors,N,N,N) for COLOR.
        """
        if self._use_buffer_A:
            flat = self.backend.to_numpy(self.psi_r_A)
        else:
            flat = self.backend.to_numpy(self.psi_r_B)

        if self.config.field_level == FieldLevel.COLOR:
            return flat.reshape(self.config.n_colors, self.N, self.N, self.N)
        return flat.reshape(self.N, self.N, self.N)

    def get_psi_imag(self) -> NDArray[np.float32] | None:
        """Get imaginary part of Ψ. None for REAL field level."""
        if not self._has_imag:
            return None
        if self._use_buffer_A:
            flat = self.backend.to_numpy(self.psi_i_A)
        else:
            flat = self.backend.to_numpy(self.psi_i_B)

        if self.config.field_level == FieldLevel.COLOR:
            return flat.reshape(self.config.n_colors, self.N, self.N, self.N)
        return flat.reshape(self.N, self.N, self.N)

    def get_energy_density(self) -> NDArray[np.float32]:
        """Compute |Ψ|² = Σₐ(Pr² + Pi²), shape (N, N, N)."""
        pr = self.get_psi_real()
        e2 = np.sum(pr**2, axis=0) if pr.ndim == 4 else pr**2
        pi = self.get_psi_imag()
        if pi is not None:
            e2 += np.sum(pi**2, axis=0) if pi.ndim == 4 else pi**2
        return e2

    def set_psi_real(self, arr: NDArray) -> None:
        """Set the real part of Ψ on both buffers.

        Parameters
        ----------
        arr : ndarray
            Shape (N,N,N) for REAL/COMPLEX, (n_colors,N,N,N) for COLOR.
        """
        flat = arr.astype(np.float32).ravel()
        data = self.backend.from_numpy(flat)
        # Set on current buffer (both current and prev for clean start)
        for buf in [self.psi_r_A, self.psi_r_B]:
            if hasattr(buf, "copy_"):
                buf[:] = data  # CuPy
            else:
                np.copyto(buf, data)  # NumPy
        for buf in [self.psi_r_prev_A, self.psi_r_prev_B]:
            if hasattr(buf, "copy_"):
                buf[:] = data
            else:
                np.copyto(buf, data)

    def set_psi_imag(self, arr: NDArray) -> None:
        """Set the imaginary part of Ψ on both buffers."""
        if not self._has_imag:
            raise ValueError("Cannot set imaginary part for REAL field level")
        flat = arr.astype(np.float32).ravel()
        data = self.backend.from_numpy(flat)
        for buf in [self.psi_i_A, self.psi_i_B]:
            if hasattr(buf, "copy_"):
                buf[:] = data
            else:
                np.copyto(buf, data)
        for buf in [self.psi_i_prev_A, self.psi_i_prev_B]:
            if hasattr(buf, "copy_"):
                buf[:] = data
            else:
                np.copyto(buf, data)

    def set_psi_real_prev(self, arr: NDArray) -> None:
        """Set *only* the previous-timestep Ψ_real buffers.

        Call after :meth:`set_psi_real` to override the previous-step
        buffers independently.  This is essential for proper traveling-wave
        initialisation: set the current buffers to Ψ(t=0) with
        :meth:`set_psi_real`, then set the previous buffers to Ψ(t=−Δt) here
        so that the leapfrog computes dΨ/dt ≠ 0 on the first step.

        Parameters
        ----------
        arr : ndarray
            Shape (N,N,N) for REAL/COMPLEX, (n_colors,N,N,N) for COLOR.
        """
        flat = arr.astype(np.float32).ravel()
        data = self.backend.from_numpy(flat)
        for buf in [self.psi_r_prev_A, self.psi_r_prev_B]:
            if hasattr(buf, "copy_"):
                buf[:] = data
            else:
                np.copyto(buf, data)

    def set_psi_imag_prev(self, arr: NDArray) -> None:
        """Set *only* the previous-timestep Ψ_imag buffers.

        See :meth:`set_psi_real_prev` for the intended usage pattern.
        """
        if not self._has_imag:
            raise ValueError("Cannot set imaginary part for REAL field level")
        flat = arr.astype(np.float32).ravel()
        data = self.backend.from_numpy(flat)
        for buf in [self.psi_i_prev_A, self.psi_i_prev_B]:
            if hasattr(buf, "copy_"):
                buf[:] = data
            else:
                np.copyto(buf, data)

    def set_psi_real_current(self, arr: NDArray) -> None:
        """Set *only* the active current-timestep Ψ_real buffer.

        Unlike :meth:`set_psi_real`, this does **not** touch the prev
        buffers at all, making it safe to call from a step callback that
        drives a continuous-wave source without resetting field velocities
        elsewhere on the grid.

        Parameters
        ----------
        arr : ndarray
            Shape (N,N,N) for REAL/COMPLEX, (n_colors,N,N,N) for COLOR.
        """
        flat = arr.astype(np.float32).ravel()
        data = self.backend.from_numpy(flat)
        buf = self.psi_r_A if self._use_buffer_A else self.psi_r_B
        if hasattr(buf, "copy_"):
            buf[:] = data
        else:
            np.copyto(buf, data)

    def set_psi_imag_current(self, arr: NDArray) -> None:
        """Set *only* the active current-timestep Ψ_imag buffer.

        See :meth:`set_psi_real_current` for the intended usage pattern.
        """
        if not self._has_imag:
            raise ValueError("Cannot set imaginary part for REAL field level")
        flat = arr.astype(np.float32).ravel()
        data = self.backend.from_numpy(flat)
        buf = self.psi_i_A if self._use_buffer_A else self.psi_i_B
        if hasattr(buf, "copy_"):
            buf[:] = data
        else:
            np.copyto(buf, data)

    def set_chi(self, arr: NDArray) -> None:
        """Set χ field on both buffers. Shape (N, N, N)."""
        flat = arr.astype(np.float32).ravel()
        data = self.backend.from_numpy(flat)
        for buf in [self.chi_A, self.chi_B]:
            if hasattr(buf, "copy_"):
                buf[:] = data
            else:
                np.copyto(buf, data)
        for buf in [self.chi_prev_A, self.chi_prev_B]:
            if hasattr(buf, "copy_"):
                buf[:] = data
            else:
                np.copyto(buf, data)

    def get_sa_fields(self) -> "NDArray[np.float32] | None":
        """Get S_a auxiliary fields as numpy array, shape (3, N, N, N).

        Returns None if SA confinement is not enabled (kappa_tube == 0).
        """
        if self.sa_A is None:
            return None
        flat = self.backend.to_numpy(self.sa_A if self._use_buffer_A else self.sa_B)
        return flat.reshape(self.config.n_colors, self.N, self.N, self.N)

    def set_sa_fields(self, arr: NDArray) -> None:
        """Set S_a auxiliary fields on both buffers.

        Parameters
        ----------
        arr : ndarray
            Shape (3, N, N, N) or (3*N^3,). Values must be ≥ 0.
        """
        if self.sa_A is None:
            raise ValueError(
                "SA fields not allocated — set kappa_tube > 0 in SimulationConfig"
                " before creating the Evolver."
            )
        flat = arr.astype(np.float32).ravel()
        data = self.backend.from_numpy(flat)
        for buf in [self.sa_A, self.sa_B]:
            if hasattr(buf, "copy_"):
                buf[:] = data
            else:
                np.copyto(buf, data)
