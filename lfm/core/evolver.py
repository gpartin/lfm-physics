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

from typing import TYPE_CHECKING

import numpy as np

from lfm.config import (
    BoundaryType,
    ChiPotentialModel,
    FieldLevel,
    SimulationConfig,
)
from lfm.core.backends import get_backend

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
        if backend.lower() == "remote":
            raise NotImplementedError(
                "Simulation does not support the remote backend protocol; "
                "use get_backend('remote').run_job() or run_steps() for "
                "direct float32 remote jobs"
            )
        self.config = config
        self.backend = get_backend(backend, precision=config.precision)
        self.dtype = self.backend.dtype
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
        self._local_phase_clock_cos: NDArray | None = None
        self._local_phase_clock_sin: NDArray | None = None

    def _init_arrays(self) -> None:
        """Allocate double-buffered arrays."""
        N = self.N
        total = self.total
        cfg = self.config

        if cfg.field_level == FieldLevel.REAL or cfg.field_level == FieldLevel.COMPLEX:
            psi_size = total
        else:  # COLOR
            psi_size = cfg.n_colors * total

        xp = self.backend

        # Psi real part — A and B buffers
        self.psi_r_A = xp.from_numpy(np.zeros(psi_size, dtype=self.dtype))
        self.psi_r_prev_A = xp.from_numpy(np.zeros(psi_size, dtype=self.dtype))
        self.psi_r_B = xp.from_numpy(np.zeros(psi_size, dtype=self.dtype))
        self.psi_r_prev_B = xp.from_numpy(np.zeros(psi_size, dtype=self.dtype))

        # Psi imaginary part (zero for real field, but allocated for uniform API)
        if self._has_imag:
            self.psi_i_A = xp.from_numpy(np.zeros(psi_size, dtype=self.dtype))
            self.psi_i_prev_A = xp.from_numpy(np.zeros(psi_size, dtype=self.dtype))
            self.psi_i_B = xp.from_numpy(np.zeros(psi_size, dtype=self.dtype))
            self.psi_i_prev_B = xp.from_numpy(np.zeros(psi_size, dtype=self.dtype))
        else:
            self.psi_i_A = None  # type: ignore[assignment]
            self.psi_i_prev_A = None  # type: ignore[assignment]
            self.psi_i_B = None  # type: ignore[assignment]
            self.psi_i_prev_B = None  # type: ignore[assignment]

        # Chi — A and B buffers
        chi_init = np.full(total, cfg.chi0, dtype=self.dtype)
        self.chi_A = xp.from_numpy(chi_init.copy())
        self.chi_prev_A = xp.from_numpy(chi_init.copy())
        self.chi_B = xp.from_numpy(chi_init.copy())
        self.chi_prev_B = xp.from_numpy(chi_init.copy())

        # Boundary mask
        if cfg.boundary_type == BoundaryType.PERIODIC:
            self.boundary_mask = xp.from_numpy(np.zeros(total, dtype=self.dtype))
        else:
            self.boundary_mask = xp.create_boundary_mask(N, cfg.boundary_fraction)

        # S_a auxiliary fields for v16 flux-tube confinement (COLOR field level only)
        if cfg.sa_enabled and cfg.field_level == FieldLevel.COLOR:
            sa_init = np.zeros(cfg.n_colors * total, dtype=self.dtype)
            self.sa_A = xp.from_numpy(sa_init.copy())
            self.sa_B = xp.from_numpy(sa_init.copy())
        else:
            self.sa_A = None  # type: ignore[assignment]
            self.sa_B = None  # type: ignore[assignment]

    def evolve(self, steps: int, callback=None, freeze_chi: bool = False) -> None:
        """Run the evolution loop for a given number of steps.

        Parameters
        ----------
        steps : int
            Number of leapfrog steps.
        callback : callable, optional
            Called as callback(evolver, step) every report_interval steps.
        freeze_chi : bool
            If True, χ is held frozen at its value at the start of this
            call.  After each kernel step the four χ double-buffers are
            restored to the snapshot so χ never drifts.  The snapshot is
            kept **device-resident** (CuPy array on GPU, or NumPy array on
            CPU) so no PCIe round-trips occur per step.
            Used by the eigenmode SCF solver and wave-optics experiments.
        """
        cfg = self.config
        report = cfg.report_interval

        # Build device-resident frozen copies of all four χ double-buffers so
        # that each restore is a fast device→device memcpy rather than a
        # CPU→GPU upload (which would add ~4 ms/step at N=256 on PCIe 3.0).
        if freeze_chi:
            _cf = [
                self.chi_A.copy(),
                self.chi_B.copy(),
                self.chi_prev_A.copy(),
                self.chi_prev_B.copy(),
            ]
        else:
            _cf = None

        for _i in range(steps):
            self._step()
            if _cf is not None:
                self.chi_A[:] = _cf[0]
                self.chi_B[:] = _cf[1]
                self.chi_prev_A[:] = _cf[2]
                self.chi_prev_B[:] = _cf[3]
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
                inv_dx2=1.0 / (cfg.dx * cfg.dx),
                enable_chi_floor=cfg.enable_chi_floor,
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
                use_stencil19_noether_current=cfg.use_stencil19_noether_current,
                inv_dx2=1.0 / (cfg.dx * cfg.dx),
                enable_chi_floor=cfg.enable_chi_floor,
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
                use_stencil19_noether_current=cfg.use_stencil19_noether_current,
                inv_dx2=1.0 / (cfg.dx * cfg.dx),
                enable_chi_floor=cfg.enable_chi_floor,
            )

        self._use_buffer_A = not self._use_buffer_A

    def evolve_gravity_recovery(
        self,
        steps: int,
        potential_model: ChiPotentialModel,
        *,
        freeze_psi: bool = True,
        relaxation_damping: float = 0.0,
        dt_override: float | None = None,
        callback=None,
    ) -> None:
        """Run the experiment-only local GOV-02 candidate evolution.

        This path is deliberately separate from :meth:`evolve`; canonical
        production behavior is unchanged. The full candidate catalog is
        available for the real register. The complex and three-channel
        registers support the selected flat-octic candidate needed for
        phase-stable, noninterfering live packets. Every path uses the
        configured 19-point backend update.
        """

        if self.config.field_level not in {
            FieldLevel.REAL,
            FieldLevel.COMPLEX,
            FieldLevel.COLOR,
        }:
            raise ValueError("gravity-recovery candidates require an LFM field register")
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if not np.isfinite(relaxation_damping) or relaxation_damping < 0.0:
            raise ValueError("relaxation_damping must be finite and nonnegative")
        model = ChiPotentialModel(potential_model)
        if (
            self.config.field_level in {FieldLevel.COMPLEX, FieldLevel.COLOR}
            and model != ChiPotentialModel.FLAT_OCTIC
        ):
            raise ValueError("multiquadrature gravity recovery supports FLAT_OCTIC")
        dt = self.config.dt if dt_override is None else float(dt_override)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_override must be positive and finite")
        report = self.config.report_interval
        for _index in range(steps):
            self._step_gravity_recovery(
                model,
                freeze_psi=freeze_psi,
                relaxation_damping=relaxation_damping,
                dt=dt,
            )
            self.step += 1
            if callback is not None and report > 0 and self.step % report == 0:
                callback(self, self.step)

    def _step_gravity_recovery(
        self,
        potential_model: ChiPotentialModel,
        *,
        freeze_psi: bool,
        relaxation_damping: float,
        dt: float,
    ) -> None:
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
            self.backend.step_real_gravity_recovery(
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
                dt,
                cfg.kappa,
                cfg.lambda_self,
                cfg.chi0,
                cfg.e0_sq,
                int(potential_model),
                freeze_psi,
                relaxation_damping,
                inv_dx2=1.0 / (cfg.dx * cfg.dx),
                enable_chi_floor=cfg.enable_chi_floor,
            )
        elif cfg.field_level == FieldLevel.COMPLEX:
            if any(value is None for value in (i_in, ip_in, i_out, ip_out)):
                raise RuntimeError("complex field buffers are missing")
            self.backend.step_complex_gravity_recovery(
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
                dt,
                cfg.kappa,
                cfg.lambda_self,
                cfg.chi0,
                cfg.e0_sq,
                int(potential_model),
                freeze_psi,
                relaxation_damping,
                inv_dx2=1.0 / (cfg.dx * cfg.dx),
                enable_chi_floor=cfg.enable_chi_floor,
            )
        else:
            if any(value is None for value in (i_in, ip_in, i_out, ip_out)):
                raise RuntimeError("color field buffers are missing")
            sa_in = (
                (self.sa_A if self._use_buffer_A else self.sa_B) if self.sa_A is not None else None
            )
            sa_out = (
                (self.sa_B if self._use_buffer_A else self.sa_A) if self.sa_A is not None else None
            )
            self.backend.step_color_gravity_recovery(
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
                dt,
                cfg.kappa,
                cfg.lambda_self,
                cfg.chi0,
                cfg.e0_sq,
                int(potential_model),
                freeze_psi,
                relaxation_damping,
                epsilon_w=cfg.epsilon_w,
                kappa_c=cfg.kappa_c,
                epsilon_cc=cfg.epsilon_cc,
                kappa_string=cfg.kappa_string,
                kappa_tube=cfg.kappa_tube,
                sa_fields_in=sa_in,
                sa_fields_out=sa_out,
                sa_gamma=cfg.sa_gamma,
                sa_d=cfg.sa_d,
                use_stencil19_noether_current=(cfg.use_stencil19_noether_current),
                inv_dx2=1.0 / (cfg.dx * cfg.dx),
                enable_chi_floor=cfg.enable_chi_floor,
            )
        self._use_buffer_A = not self._use_buffer_A

    # --- Field accessors (return numpy arrays) ---

    def _current_buf(self) -> str:
        """Which buffer holds the most recent result."""
        return "A" if self._use_buffer_A else "B"

    def get_chi(self) -> NDArray[np.floating]:
        """Get current χ field as numpy array, shape (N, N, N)."""
        if self._use_buffer_A:
            flat = self.backend.to_numpy(self.chi_A)
        else:
            flat = self.backend.to_numpy(self.chi_B)
        return flat.reshape(self.N, self.N, self.N)

    def get_psi_real(self) -> NDArray[np.floating]:
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

    def get_psi_imag(self) -> NDArray[np.floating] | None:
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

    def get_psi_real_prev(self) -> NDArray[np.floating]:
        """Get previous-timestep Ψ_real as numpy, same shape as get_psi_real."""
        if self._use_buffer_A:
            flat = self.backend.to_numpy(self.psi_r_prev_A)
        else:
            flat = self.backend.to_numpy(self.psi_r_prev_B)
        if self.config.field_level == FieldLevel.COLOR:
            return flat.reshape(self.config.n_colors, self.N, self.N, self.N)
        return flat.reshape(self.N, self.N, self.N)

    def get_psi_imag_prev(self) -> NDArray[np.floating] | None:
        """Get previous-timestep Ψ_imag. None for REAL field level."""
        if not self._has_imag:
            return None
        if self._use_buffer_A:
            flat = self.backend.to_numpy(self.psi_i_prev_A)
        else:
            flat = self.backend.to_numpy(self.psi_i_prev_B)
        if self.config.field_level == FieldLevel.COLOR:
            return flat.reshape(self.config.n_colors, self.N, self.N, self.N)
        return flat.reshape(self.N, self.N, self.N)

    def get_energy_density(self) -> NDArray[np.floating]:
        """Compute |Ψ|² = Σₐ(Pr² + Pi²), shape (N, N, N)."""
        pr = self.get_psi_real()
        e2 = np.sum(pr**2, axis=0) if pr.ndim == 4 else pr**2
        pi = self.get_psi_imag()
        if pi is not None:
            e2 += np.sum(pi**2, axis=0) if pi.ndim == 4 else pi**2
        return e2

    def get_chi_prev(self) -> NDArray[np.floating]:
        """Get previous-timestep chi as a NumPy array."""
        if self._use_buffer_A:
            flat = self.backend.to_numpy(self.chi_prev_A)
        else:
            flat = self.backend.to_numpy(self.chi_prev_B)
        return flat.reshape(self.N, self.N, self.N)

    def get_boundary_mask(self) -> NDArray[np.floating]:
        """Get the fixed absorption/freeze mask as a NumPy array."""
        flat = self.backend.to_numpy(self.boundary_mask)
        return flat.reshape(self.N, self.N, self.N)

    def set_boundary_mask(self, arr: NDArray) -> None:
        """Set an input-independent boundary mask before evolution starts.

        Mask values use the production-kernel convention: zero is an active
        cell and one is a fully frozen/absorbing cell. Fractional values in
        ``[0, 1]`` are allowed for graded absorption. The geometry is frozen
        after the first evolution step so it cannot act as a live controller.
        """
        if self.step != 0:
            raise RuntimeError("boundary geometry can only be set before evolution")
        host = np.asarray(arr, dtype=self.dtype)
        if host.shape != (self.N, self.N, self.N):
            raise ValueError(
                f"boundary mask must have shape ({self.N}, {self.N}, {self.N}), got {host.shape}"
            )
        if not np.isfinite(host).all():
            raise ValueError("boundary mask must contain only finite values")
        if np.any(host < 0.0) or np.any(host > 1.0):
            raise ValueError("boundary mask values must lie in [0, 1]")
        data = self.backend.from_numpy(host.ravel())
        self.boundary_mask[:] = data

    def set_local_phase_clock_map(
        self,
        dwell_steps: NDArray,
        unit_phase_rad: float,
        enable_mask: NDArray | None = None,
    ) -> None:
        """Set a precomputed local phase-clock map.

        ``dwell_steps`` declares the local Noether dwell class for each active
        complex field cell.  A 3D array applies to every complex/color
        component; a full field-shaped array can address components separately.
        The map is loaded once and stored in backend-native arrays.
        """

        if not self._has_imag:
            raise ValueError("local phase clocks require a complex field level")
        if self.step != 0:
            raise RuntimeError("local phase clock maps must be set before evolution")
        if not np.isfinite(unit_phase_rad):
            raise ValueError("unit_phase_rad must be finite")

        dwell = np.asarray(dwell_steps, dtype=np.float64)
        expected_3d = (self.N, self.N, self.N)
        expected_full = (
            (self.config.n_colors, self.N, self.N, self.N)
            if self.config.field_level == FieldLevel.COLOR
            else expected_3d
        )
        if dwell.shape == expected_3d and self.config.field_level == FieldLevel.COLOR:
            dwell = np.broadcast_to(dwell[None, :, :, :], expected_full).copy()
        elif dwell.shape != expected_full:
            raise ValueError(
                f"dwell_steps must have shape {expected_3d} or {expected_full}, got {dwell.shape}"
            )
        if not np.isfinite(dwell).all():
            raise ValueError("dwell_steps must contain only finite values")

        if enable_mask is not None:
            mask = np.asarray(enable_mask, dtype=np.float64)
            if mask.shape == expected_3d and self.config.field_level == FieldLevel.COLOR:
                mask = np.broadcast_to(mask[None, :, :, :], expected_full).copy()
            elif mask.shape != expected_full:
                raise ValueError(
                    "enable_mask must have shape "
                    f"{expected_3d} or {expected_full}, got {mask.shape}"
                )
            if not np.isfinite(mask).all():
                raise ValueError("enable_mask must contain only finite values")
            if np.any(mask < 0.0) or np.any(mask > 1.0):
                raise ValueError("enable_mask values must lie in [0, 1]")
            dwell = dwell * mask

        phase = dwell * float(unit_phase_rad)
        phase_cos = np.cos(phase).astype(self.dtype, copy=False).ravel()
        phase_sin = np.sin(phase).astype(self.dtype, copy=False).ravel()
        self._local_phase_clock_cos = self.backend.from_numpy(phase_cos)
        self._local_phase_clock_sin = self.backend.from_numpy(phase_sin)

    def apply_local_phase_clock_map(self) -> None:
        """Apply the stored local phase-clock map to complex phase space."""

        if not self._has_imag:
            raise ValueError("local phase clocks require a complex field level")
        if self._local_phase_clock_cos is None or self._local_phase_clock_sin is None:
            raise RuntimeError("local phase clock map has not been set")

        for real, imag in (
            (self.psi_r_A, self.psi_i_A),
            (self.psi_r_B, self.psi_i_B),
            (self.psi_r_prev_A, self.psi_i_prev_A),
            (self.psi_r_prev_B, self.psi_i_prev_B),
        ):
            if imag is None:
                raise RuntimeError("imaginary buffer missing for complex field")
            self.backend.apply_complex_phase_map(
                real,
                imag,
                self._local_phase_clock_cos,
                self._local_phase_clock_sin,
            )

    def set_psi_real(self, arr: NDArray) -> None:
        """Set the real part of Ψ on both buffers.

        Parameters
        ----------
        arr : ndarray
            Shape (N,N,N) for REAL/COMPLEX, (n_colors,N,N,N) for COLOR.
        """
        flat = arr.astype(self.dtype).ravel()
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
        flat = arr.astype(self.dtype).ravel()
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
        flat = arr.astype(self.dtype).ravel()
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
        flat = arr.astype(self.dtype).ravel()
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
        flat = arr.astype(self.dtype).ravel()
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
        flat = arr.astype(self.dtype).ravel()
        data = self.backend.from_numpy(flat)
        buf = self.psi_i_A if self._use_buffer_A else self.psi_i_B
        if hasattr(buf, "copy_"):
            buf[:] = data
        else:
            np.copyto(buf, data)

    def set_chi(self, arr: NDArray) -> None:
        """Set χ field on all four buffers (current + prev). Shape (N, N, N)."""
        flat = arr.astype(self.dtype).ravel()
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

    def set_chi_current(self, arr: NDArray) -> None:
        """Set *only* the current-timestep χ buffers."""
        flat = arr.astype(self.dtype).ravel()
        data = self.backend.from_numpy(flat)
        for buf in [self.chi_A, self.chi_B]:
            if hasattr(buf, "copy_"):
                buf[:] = data
            else:
                np.copyto(buf, data)

    def set_chi_prev(self, arr: NDArray) -> None:
        """Set *only* the previous-timestep χ buffers.

        Call after :meth:`set_chi_current` to give the χ field a nonzero
        time derivative (dχ/dt ≠ 0), essential for moving the χ-well of
        a velocity-boosted soliton.
        """
        flat = arr.astype(self.dtype).ravel()
        data = self.backend.from_numpy(flat)
        for buf in [self.chi_prev_A, self.chi_prev_B]:
            if hasattr(buf, "copy_"):
                buf[:] = data
            else:
                np.copyto(buf, data)

    def get_sa_fields(self) -> NDArray[np.floating] | None:
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
        flat = arr.astype(self.dtype).ravel()
        data = self.backend.from_numpy(flat)
        for buf in [self.sa_A, self.sa_B]:
            if hasattr(buf, "copy_"):
                buf[:] = data
            else:
                np.copyto(buf, data)
