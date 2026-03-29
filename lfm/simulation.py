"""
Simulation Facade
=================

High-level ``Simulation`` class that ties together Config, Fields,
Evolver, and Analysis into a single convenient API.

Usage::

    from lfm import Simulation, SimulationConfig

    sim = Simulation(SimulationConfig(grid_size=64))
    sim.place_soliton((32, 32, 32), amplitude=6.0)
    sim.equilibrate()
    sim.run(steps=10_000)
    print(sim.metrics())
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from lfm.experiment.barrier import Barrier
    from lfm.experiment.detector import DetectorScreen
    from lfm.experiment.source import ContinuousSource
    from lfm.particles.catalog import Particle
    from lfm.particles.solver import SolitonSolution

import numpy as np

from lfm.analysis.energy import total_energy
from lfm.analysis.metrics import compute_metrics
from lfm.analysis.structure import interior_mask as make_interior_mask
from lfm.config import BoundaryType, FieldLevel, SimulationConfig
from lfm.core.evolver import Evolver
from lfm.fields.equilibrium import equilibrate_from_fields
from lfm.fields.soliton import gaussian_soliton, place_solitons


class Simulation:
    """End-to-end LFM simulation facade.

    Wraps ``Evolver`` with field initialization and analysis.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration.
    backend : str
        Backend preference: 'auto', 'cpu', or 'gpu'.
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        backend: str = "auto",
    ) -> None:
        if config is None:
            config = SimulationConfig()
        self.config = config
        self._evolver = Evolver(config, backend=backend)
        self._interior_mask: NDArray[np.bool_] | None = None
        self._history: list[dict[str, float]] = []

        # Cache for previous-step fields (for energy calculation)
        self._psi_r_prev: NDArray[np.float32] | None = None
        self._psi_i_prev: NDArray[np.float32] | None = None

        # State tracking — prevents misuse like equilibrating before placing
        # solitons, or forgetting to equilibrate entirely.
        self._solitons_placed: bool = False
        self._equilibrated: bool = False

        # Velocity boosts registered by place_soliton().
        # Each entry is (velocity, envelope²_field) so equilibrate() can
        # compute the advective shift dχ/dt = −v·∇χ for moving wells.
        self._velocity_boosts: list[tuple[tuple[float, float, float], NDArray]] = []

    @property
    def step(self) -> int:
        """Current simulation step."""
        return self._evolver.step

    @property
    def history(self) -> list[dict[str, float]]:
        """List of metric snapshots collected during run()."""
        return self._history

    # ── Field access ──────────────────────────────────────

    @property
    def chi(self) -> NDArray[np.float32]:
        """Current χ field, shape (N, N, N)."""
        return self._evolver.get_chi()

    @chi.setter
    def chi(self, value: NDArray[np.float32]) -> None:
        self._evolver.set_chi(value)

    @property
    def psi_real(self) -> NDArray[np.float32]:
        """Real part of Ψ, shape (N, N, N)."""
        return self._evolver.get_psi_real()

    @psi_real.setter
    def psi_real(self, value: NDArray[np.float32]) -> None:
        self._evolver.set_psi_real(value)

    @property
    def psi_imag(self) -> NDArray[np.float32] | None:
        """Imaginary part of Ψ (None for real field level)."""
        return self._evolver.get_psi_imag()

    @psi_imag.setter
    def psi_imag(self, value: NDArray[np.float32]) -> None:
        self._evolver.set_psi_imag(value)

    @property
    def psi_real_prev(self) -> NDArray[np.float32] | None:
        """Real part of Ψ from the step *before* the last ``run()`` call.

        Automatically updated after every :meth:`run` and :meth:`run_driven`
        call.  Use with :func:`lfm.fluid_fields` to compute ∂Ψ/∂t without
        manually snapshotting state::

            sim.run(steps=1)
            f = lfm.fluid_fields(
                sim.psi_real, sim.psi_real_prev, sim.chi, config.dt,
                psi_i=sim.psi_imag, psi_i_prev=sim.psi_imag_prev,
            )
        """
        return self._psi_r_prev

    @property
    def psi_imag_prev(self) -> NDArray[np.float32] | None:
        """Imaginary part of Ψ from the step *before* the last ``run()`` call.

        See :attr:`psi_real_prev` for usage.
        """
        return self._psi_i_prev

    @property
    def energy_density(self) -> NDArray[np.float32]:
        """Energy density |Ψ|², shape (N, N, N)."""
        return self._evolver.get_energy_density()

    def get_chi(self) -> NDArray[np.float32]:
        """Get current χ field, shape (N, N, N)."""
        return self._evolver.get_chi()

    def get_psi_real(self) -> NDArray[np.float32]:
        """Get real part of Ψ."""
        return self._evolver.get_psi_real()

    def get_psi_imag(self) -> NDArray[np.float32] | None:
        """Get imaginary part of Ψ (None for real field level)."""
        return self._evolver.get_psi_imag()

    def get_energy_density(self) -> NDArray[np.float32]:
        """Get |Ψ|² energy density, shape (N, N, N)."""
        return self._evolver.get_energy_density()

    def set_psi_real(self, value: NDArray[np.float32]) -> None:
        """Set real part of Ψ."""
        self._evolver.set_psi_real(value)

    def set_psi_imag(self, value: NDArray[np.float32]) -> None:
        """Set imaginary part of Ψ."""
        self._evolver.set_psi_imag(value)

    def set_psi_real_prev(self, value: NDArray[np.float32]) -> None:
        """Override the previous-timestep Ψ_real for traveling-wave init.

        Call *after* :meth:`set_psi_real` to set Ψ(t=−Δt) independently,
        so that the leapfrog starts with a non-zero dΨ/dt.
        """
        self._evolver.set_psi_real_prev(value)

    def set_psi_imag_prev(self, value: NDArray[np.float32]) -> None:
        """Override the previous-timestep Ψ_imag for traveling-wave init."""
        self._evolver.set_psi_imag_prev(value)

    def set_psi_real_current(self, value: NDArray[np.float32]) -> None:
        """Set only the active current-timestep Ψ_real buffer.

        Safe to call from a step callback: does **not** touch the prev
        buffers, so field velocities elsewhere on the grid are preserved.
        Use this for driven continuous-wave sources.
        """
        self._evolver.set_psi_real_current(value)

    def set_psi_imag_current(self, value: NDArray[np.float32]) -> None:
        """Set only the active current-timestep Ψ_imag buffer.

        See :meth:`set_psi_real_current` for the intended usage pattern.
        """
        self._evolver.set_psi_imag_current(value)

    def set_chi(self, value: NDArray[np.float32]) -> None:
        """Set χ field."""
        self._evolver.set_chi(value)

    # ── zero-copy native buffer access (avoids GPU↔CPU roundtrips) ─────

    def _native_psi_real(self):
        """Active psi_real buffer as a 3D view on the native backend.

        Returns a *view* (cupy on GPU, numpy on CPU) — no data is copied.
        Modifications are immediately visible to the evolver.
        """
        ev = self._evolver
        buf = ev.psi_r_A if ev._use_buffer_A else ev.psi_r_B
        N = ev.N
        return buf.reshape(N, N, N)

    def _native_psi_imag(self):
        """Active psi_imag buffer as a 3D view (None for real field)."""
        ev = self._evolver
        if not ev._has_imag:
            return None
        buf = ev.psi_i_A if ev._use_buffer_A else ev.psi_i_B
        N = ev.N
        return buf.reshape(N, N, N)

    def _native_chi(self):
        """Active chi buffer as a 3D view on the native backend."""
        ev = self._evolver
        buf = ev.chi_A if ev._use_buffer_A else ev.chi_B
        N = ev.N
        return buf.reshape(N, N, N)

    def _native_chi_pair(self):
        """All four chi buffers as 3D views (for barrier enforcement).

        Returns (chi_A, chi_B, chi_prev_A, chi_prev_B).  Modify all four
        so the barrier persists across leapfrog double-buffer swaps.
        """
        ev = self._evolver
        N = ev.N
        return (
            ev.chi_A.reshape(N, N, N),
            ev.chi_B.reshape(N, N, N),
            ev.chi_prev_A.reshape(N, N, N),
            ev.chi_prev_B.reshape(N, N, N),
        )

    def _to_device(self, arr):
        """Convert a numpy array to the backend's native format."""
        return self._evolver.backend.from_numpy(arr.ravel().astype(np.float32))

    @property
    def sa_fields(self) -> NDArray[np.float32] | None:
        """S_a auxiliary confinement fields, shape (3, N, N, N).

        Returns ``None`` when ``config.kappa_tube == 0`` (SA disabled).
        Initialised to zero; the evolver updates them each step via the
        diffusion equation dS_a/dt = D∇²S_a + γ(|Ψ_a|² − S_a).
        """
        return self._evolver.get_sa_fields()

    @sa_fields.setter
    def sa_fields(self, value: NDArray[np.float32]) -> None:
        """Set S_a auxiliary confinement fields."""
        self._evolver.set_sa_fields(value)

    # ── Field initialization ──────────────────────────────

    def place_particle(
        self,
        name: "str | Particle",
        position: tuple[float, float, float] | None = None,
        velocity: tuple[float, float, float] | None = None,
        amplitude: float | None = None,
        sigma: float | None = None,
    ) -> "SolitonSolution":
        """Place a particle with proper eigenmode physics and optional motion.

        This is the recommended way to add particles to a simulation.
        It handles everything automatically:

        1. **Eigenmode relaxation** — solves for a self-consistent bound
           state (standing wave in its own χ-well) using alternating
           Poisson / imaginary-time relaxation.
        2. **Phase-gradient boost** — encodes momentum as a spatial phase
           gradient Ψ → Ψ·exp(ik·x), giving the soliton a nonzero
           group velocity.
        3. **Charge phase** — applies the particle's intrinsic phase
           (θ = 0 for matter, θ = π for antimatter).
        4. **Superposition** — adds to any existing field, so you can
           place multiple particles by calling this method repeatedly.
        5. **χ equilibration** is deferred to :meth:`equilibrate` (called
           automatically by :meth:`run`), which Poisson-solves χ from
           the combined |Ψ|² of all placed particles.

        Parameters
        ----------
        name : str or Particle
            Particle name (``"electron"``, ``"proton"``, ``"positron"``,
            etc.) or a :class:`~lfm.particles.catalog.Particle` instance.
        position : (x, y, z) or None
            Centre in grid coordinates. Defaults to grid centre.
        velocity : (vx, vy, vz) or None
            Velocity in units of c (e.g. ``(0, 0, 0.1)`` for 0.1c along z).
            If ``None`` the particle is placed at rest.
        amplitude : float or None
            Override the eigenmode seed amplitude. If ``None``, chosen
            automatically: shallow well (3.0) for moving particles,
            catalog default for stationary ones.
        sigma : float or None
            Override the eigenmode seed width. If ``None``, chosen
            automatically: ≥ 5 cells for moving particles to avoid
            lattice pinning.

        Returns
        -------
        SolitonSolution
            The converged eigenmode (before boost/roll). Useful for
            inspecting χ_min, eigenvalue ω, convergence, etc.

        Examples
        --------
        Electron at rest in the centre:

        >>> sim.place_particle("electron")

        Proton–antiproton collision:

        >>> sim.place_particle("proton",   (32, 32, 24), velocity=(0, 0, 0.1))
        >>> sim.place_particle("antiproton", (32, 32, 40), velocity=(0, 0, -0.1))
        >>> sim.run(steps=10_000)
        """
        import math as _math

        from lfm.particles.catalog import (
            Particle,
            amplitude_for_particle,
            get_particle,
            sigma_for_particle,
        )
        from lfm.particles.solver import boost_fields, relax_eigenmode

        # --- Resolve particle ---
        if isinstance(name, str):
            particle: Particle = get_particle(name)
        else:
            particle = name

        N = self.config.grid_size
        center = N // 2
        chi0 = self.config.chi0
        kappa = self.config.kappa
        dt = self.config.dt

        if position is None:
            position = (float(center), float(center), float(center))

        has_velocity = (
            velocity is not None
            and sum(v**2 for v in velocity) > 1e-20
        )

        # --- Choose amplitude and sigma ---
        # Moving particles need shallow wells to avoid Peierls-Nabarro
        # lattice pinning.  Stationary particles use deeper catalog defaults.
        _MOTION_AMP = 0.5
        _MOTION_SIGMA_MIN = 12.0

        if amplitude is None:
            if has_velocity:
                amplitude = _MOTION_AMP
            else:
                amplitude = amplitude_for_particle(particle, N)

        if sigma is None:
            if has_velocity:
                sigma = max(_MOTION_SIGMA_MIN, N / 5.3)
            else:
                sigma = sigma_for_particle(particle, N)
                sigma = max(sigma, 5.0)

        # --- Validate timestep for motion ---
        if has_velocity:
            from lfm.constants import DT_MOTION

            if dt > DT_MOTION * 1.01:
                import warnings

                warnings.warn(
                    f"Moving particles need a fine timestep for accurate "
                    f"motion.  Current dt={dt} but recommended "
                    f"dt={DT_MOTION}.  Create the simulation with "
                    f"SimulationConfig(dt={DT_MOTION}) for reliable "
                    f"velocity retention (>80%).",
                    stacklevel=2,
                )

        # --- Ensure complex field level for moving or charged particles ---
        needs_complex = has_velocity or abs(particle.phase) > 1e-10
        if needs_complex and self.config.field_level == FieldLevel.REAL:
            raise ValueError(
                "Moving or charged particles require field_level=COMPLEX. "
                "Create the simulation with "
                "SimulationConfig(field_level=FieldLevel.COMPLEX)."
            )

        # --- Step 1: Relax eigenmode at grid centre ---
        sol = relax_eigenmode(
            N=N,
            amplitude=amplitude,
            sigma=sigma,
            chi0=chi0,
            kappa=kappa,
        )

        # --- Step 2: Roll eigenmode to target position ---
        E = sol.psi_r.copy()
        for ax in range(3):
            shift = int(position[ax]) - center
            if shift != 0:
                E = np.roll(E, shift, axis=ax)

        chi_local = np.full_like(E, chi0)
        dchi = sol.chi - np.float32(chi0)
        for ax in range(3):
            shift = int(position[ax]) - center
            if shift != 0:
                dchi = np.roll(dchi, shift, axis=ax)
        chi_local = np.float32(chi0) + dchi

        # --- Step 3: Boost if moving ---
        if has_velocity:
            assert velocity is not None  # mypy narrowing
            pr_c, pi_c, pr_p, pi_p, _ = boost_fields(
                E, chi_local, velocity,
                dt=dt, omega=sol.eigenvalue, chi0=chi0,
            )
        else:
            pr_c = E
            pi_c = np.zeros_like(E)
            # Stationary eigenmode: ψ(−Δt) = ψ(0)·cos(ωΔt)
            if sol.eigenvalue and sol.eigenvalue > 0:
                cos_wdt = float(_math.cos(sol.eigenvalue * dt))
                pr_p = (E * cos_wdt).astype(np.float32)
            else:
                pr_p = E.copy()
            pi_p = np.zeros_like(E)

        # --- Step 4: Apply charge phase ---
        phase = float(particle.phase)
        if abs(phase) > 1e-10:
            cos_p = _math.cos(phase)
            sin_p = _math.sin(phase)
            pr_c2 = (pr_c * cos_p - pi_c * sin_p).astype(np.float32)
            pi_c2 = (pr_c * sin_p + pi_c * cos_p).astype(np.float32)
            pr_p2 = (pr_p * cos_p - pi_p * sin_p).astype(np.float32)
            pi_p2 = (pr_p * sin_p + pi_p * cos_p).astype(np.float32)
            pr_c, pi_c, pr_p, pi_p = pr_c2, pi_c2, pr_p2, pi_p2

        # --- Step 5: Superpose onto existing fields ---
        # Current buffers (t = 0)
        old_r = self._evolver.get_psi_real()
        self._evolver.set_psi_real_current(old_r + pr_c)

        old_r_prev = self._evolver.get_psi_real_prev()
        self._evolver.set_psi_real_prev(old_r_prev + pr_p)

        if self.config.field_level != FieldLevel.REAL:
            old_i = self._evolver.get_psi_imag()
            if old_i is not None:
                self._evolver.set_psi_imag_current(old_i + pi_c)
            old_i_prev = self._evolver.get_psi_imag_prev()
            if old_i_prev is not None:
                self._evolver.set_psi_imag_prev(old_i_prev + pi_p)

        # --- Step 6: Record state for equilibrate() ---
        self._solitons_placed = True
        self._equilibrated = False

        if has_velocity:
            assert velocity is not None  # mypy narrowing
            envelope_sq = pr_c ** 2 + pi_c ** 2
            self._velocity_boosts.append((velocity, envelope_sq))

        return sol

    def place_soliton(
        self,
        position: tuple[float, float, float],
        amplitude: float | None = None,
        sigma: float | None = None,
        phase: float = 0.0,
        velocity: tuple[float, float, float] | None = None,
    ) -> None:
        """Place a single Gaussian soliton on the grid.

        Parameters
        ----------
        position : (x, y, z)
            Center in grid coordinates.
        amplitude : float or None
            Peak amplitude (default from config).
        sigma : float or None
            Width (default from config).
        phase : float
            Complex phase (0 = electron, π = positron).
        velocity : (vx, vy, vz) or None
            Initial velocity in lattice units (|v| < c = 1).  Adds a
            spatial phase gradient k·(∂r−r₀) where k = χ₀·v/c,
            creating a boosted soliton with net momentum.

        Examples
        --------
        Single neutral soliton in the centre of a 64³ grid:

        >>> sim.place_soliton((32, 32, 32), amplitude=6.0)

        Electron–positron pair (opposite phases) separated by 20 cells:

        >>> sim.place_soliton((32, 22, 32), amplitude=4.0, phase=0.0)     # e⁻
        >>> sim.place_soliton((32, 42, 32), amplitude=4.0, phase=3.1416)  # e⁺

        Moving soliton with velocity 0.1c along the x-axis:

        >>> sim.place_soliton((32, 32, 32), amplitude=5.0,
        ...                   velocity=(0.1, 0.0, 0.0))
        """
        N = self.config.grid_size
        amp = amplitude if amplitude is not None else self.config.e_amplitude
        sig = sigma if sigma is not None else self.config.sigma

        if velocity is not None:
            import math as _math

            vx, vy, vz = velocity
            chi0 = self.config.chi0
            c = self.config.c
            dt = self.config.dt

            # Nyquist guard: carrier k must not exceed 80% of lattice Nyquist
            speed = _math.sqrt(vx**2 + vy**2 + vz**2)
            k_carrier = chi0 * speed / c
            k_nyquist = _math.pi
            if k_carrier > 0.8 * k_nyquist:
                v_max = 0.8 * k_nyquist * c / chi0
                raise ValueError(
                    f"Velocity |v|={speed:.4f}c gives carrier "
                    f"k={k_carrier:.2f} rad/cell, exceeding 80% of Nyquist "
                    f"limit ({0.8 * k_nyquist:.2f} rad/cell). "
                    f"Maximum safe speed is {v_max:.4f}c."
                )

            kx = chi0 * vx / c
            ky = chi0 * vy / c
            kz = chi0 * vz / c

            x = np.arange(N, dtype=np.float32)
            X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
            px, py, pz = position

            # --- t = 0: envelope centred at r₀ ---
            r2 = (X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2
            envelope = (amp * np.exp(-r2 / (2.0 * sig**2))).astype(np.float32)
            phase_grid = (phase + kx * (X - px) + ky * (Y - py) + kz * (Z - pz)).astype(np.float32)
            pr = (envelope * np.cos(phase_grid)).astype(np.float32)
            pi = (envelope * np.sin(phase_grid)).astype(np.float32)

            # --- t = −Δt: envelope shifted back by v·Δt ---
            # Both the Gaussian centre AND the phase reference point move
            # back by one timestep so that |Ψ_prev|² is centred at
            # r₀ − v·Δt.  This gives the χ-well a nonzero time derivative
            # after equilibration (the well tracks the soliton).
            px_prev = px - vx * dt
            py_prev = py - vy * dt
            pz_prev = pz - vz * dt
            r2_prev = (X - px_prev) ** 2 + (Y - py_prev) ** 2 + (Z - pz_prev) ** 2
            envelope_prev = (amp * np.exp(-r2_prev / (2.0 * sig**2))).astype(np.float32)
            omega = float(np.sqrt(kx**2 + ky**2 + kz**2 + chi0**2))
            phase_prev = (
                phase + kx * (X - px_prev) + ky * (Y - py_prev) + kz * (Z - pz_prev) + omega * dt
            ).astype(np.float32)
            pr_prev = (envelope_prev * np.cos(phase_prev)).astype(np.float32)
            pi_prev = (envelope_prev * np.sin(phase_prev)).astype(np.float32)
        else:
            pr, pi = gaussian_soliton(N, position, amp, sig, phase)
            pr_prev = pr
            pi_prev = pi

        # Add to existing field
        current_r = self._evolver.get_psi_real()
        new_r = current_r + pr
        self._evolver.set_psi_real(new_r)
        # For velocity-boosted soliton, override prev buffers with the
        # time-shifted version so the leapfrog computes dΨ/dt ≠ 0.
        if velocity is not None:
            prev_r = self._evolver.get_psi_real()  # = new_r (already set above)
            # Reconstruct: prev = (existing field contribution, same both steps) + pr_prev
            prev_r = current_r + pr_prev  # existing field assumed stationary
            self._evolver.set_psi_real_prev(prev_r)

        if self.config.field_level != FieldLevel.REAL:
            current_i = self._evolver.get_psi_imag()
            if current_i is not None:
                new_i = current_i + pi
                self._evolver.set_psi_imag(new_i)
                if velocity is not None:
                    prev_i = current_i + pi_prev
                    self._evolver.set_psi_imag_prev(prev_i)

        self._solitons_placed = True
        self._equilibrated = False  # needs re-equilibration after new soliton

        # Record velocity boost so equilibrate() can initialise dχ/dt ≠ 0.
        if velocity is not None:
            # envelope² is the |Ψ|² contribution of this soliton alone
            # (used as weight to localise the velocity field).
            self._velocity_boosts.append(((vx, vy, vz), envelope ** 2))

    def place_solitons(
        self,
        positions: list[tuple[float, float, float]],
        amplitude: float | None = None,
        sigma: float | None = None,
        phases: list[float] | None = None,
        colors: list[int] | None = None,
    ) -> None:
        """Place multiple solitons (for Level 2 color fields).

        Parameters
        ----------
        positions : list of (x, y, z)
            Centers in grid coordinates.
        amplitude, sigma : float or None
            Default from config.
        phases : list of float or None
            Per-soliton phases.
        colors : list of int or None
            Per-soliton color indices (round-robin if None).
        """
        N = self.config.grid_size
        amp = amplitude if amplitude is not None else self.config.e_amplitude
        sig = sigma if sigma is not None else self.config.sigma

        if self.config.field_level == FieldLevel.COLOR:
            pr, pi = place_solitons(
                N,
                positions,
                amp,
                sig,
                phases=phases,
                colors=colors,
                n_colors=self.config.n_colors,
            )
        else:
            # Single-component: sum all solitons
            pr_total = np.zeros((N, N, N), dtype=np.float32)
            pi_total = np.zeros((N, N, N), dtype=np.float32)
            if phases is None:
                phases = [0.0] * len(positions)
            for pos, ph in zip(positions, phases, strict=False):
                pr, pi = gaussian_soliton(N, pos, amp, sig, ph)
                pr_total += pr
                pi_total += pi
            pr, pi = pr_total, pi_total

        current_r = self._evolver.get_psi_real()
        self._evolver.set_psi_real(current_r + pr)

        if self.config.field_level != FieldLevel.REAL:
            current_i = self._evolver.get_psi_imag()
            if current_i is not None:
                self._evolver.set_psi_imag(current_i + pi)

        self._solitons_placed = True
        self._equilibrated = False

    def place_plane_wave(
        self,
        axis: int = 2,
        amplitude: float = 0.3,
        velocity: float = 0.5,
        z_max: int | None = None,
        phase: float = 0.0,
        beam_waist: float | None = None,
    ) -> None:
        """Initialize a forward-propagating Gaussian beam (coherent wave source).

        Fills the domain (or the region ``coord < z_max`` along *axis*) with a
        monochromatic Gaussian beam at ``t = 0`` **and** ``t = −Δt``, so the
        leapfrog integrator starts with the correct field velocity and the wave
        propagates in the ``+axis`` direction immediately.

        Unlike a flat plane wave, the beam is shaped with a Gaussian transverse
        envelope so its energy stays within the active (non-damped) interior,
        avoiding the edge-absorption problem of ABSORBING boundaries.

        This is the recommended source for double-slit and diffraction
        experiments: the wave is already near the barrier at ``t = 0``, so
        the experiment only needs barrier → detector transit time (not source
        → barrier → detector).

        Parameters
        ----------
        axis : int
            Propagation direction (0 = x, 1 = y, 2 = z).  Default 2.
        amplitude : float
            Peak wave amplitude.  Use small values (0.2–0.5) to keep Δχ
            negligible over the run.
        velocity : float
            Phase speed in lattice units (c = 1).  Higher values (~0.5)
            reduce required step count.  Group velocity
            ``v_g = χ₀·v / √((χ₀v)² + χ₀²) ≈ v`` for ``v ≪ 1``.
        z_max : int or None
            If given, zero the wave for ``coord ≥ z_max`` along *axis*.
            Set to ``barrier_position`` to pre-fill only the source side.
        phase : float
            Initial phase offset in radians (0 = cosine wave).
        beam_waist : float or None
            1/e² Gaussian half-width in the transverse plane (grid cells).
            Defaults to half of the frozen-boundary radius so the beam
            stays well inside the active region:
            ``w0 = 0.5 × boundary_fraction × N / 2``.
            Set larger to illuminate wider slit separations.
        """
        N = self.config.grid_size
        chi0 = self.config.chi0
        c = self.config.c
        dt = self.config.dt

        k = chi0 * velocity / c
        omega = float(np.sqrt(k**2 + chi0**2))

        # Beam waist: default to half the active-sphere radius so the beam
        # decays to ~2% before the absorbing/frozen boundary.
        if beam_waist is None:
            beam_waist = self.config.boundary_fraction * N / 2

        # ── Transverse Gaussian envelope (centred on grid) ──────────────────
        centre = N / 2.0
        trans_axes = [i for i in range(3) if i != axis]
        gauss = np.ones((N, N, N), dtype=np.float32)
        for ta in trans_axes:
            c_t = np.arange(N, dtype=np.float32) - centre
            shape = [1, 1, 1]
            shape[ta] = N
            gauss *= np.exp(-(c_t.reshape(shape) ** 2) / (2.0 * beam_waist**2))

        # ── Propagating cosine along the propagation axis ────────────────────
        prop_coord = np.arange(N, dtype=np.float32)  # 0 … N-1
        cos_cur = (amplitude * np.cos(k * prop_coord + phase)).astype(np.float32)
        cos_prev = (amplitude * np.cos(k * prop_coord + phase + omega * dt)).astype(np.float32)

        shape_p = [1, 1, 1]
        shape_p[axis] = N
        psi_3d = (gauss * cos_cur.reshape(shape_p)).astype(np.float32)
        psi_3d_prev = (gauss * cos_prev.reshape(shape_p)).astype(np.float32)

        # ── Truncate at z_max (don't pre-fill past the barrier) ─────────────
        if z_max is not None:
            slc: list = [slice(None), slice(None), slice(None)]
            slc[axis] = slice(z_max, None)
            psi_3d[tuple(slc)] = 0.0
            psi_3d_prev[tuple(slc)] = 0.0

        # ── Write into leapfrog buffers ───────────────────────────────────────
        # set_psi_real writes all four buffers to the same value; then
        # set_psi_real_prev overrides only the prev buffers so
        # dΨ/dt = (psi_current − psi_prev) / dt ≠ 0 → wave propagates.
        self._evolver.set_psi_real(psi_3d)
        self._evolver.set_psi_real_prev(psi_3d_prev)

    def equilibrate(self) -> None:
        """Poisson-equilibrate χ from current Ψ field.

        Solves the GOV-04 quasi-static limit via FFT:

        .. math::

            \\nabla^2 \\delta\\chi = \\kappa(|\\Psi|^2 - E_0^2)

        The Fourier-space solution is applied in-place to the χ buffers
        so that the very first leapfrog step starts from a self-consistent
        gravitational configuration rather than a cold uniform χ = χ₀
        background.

        Call :meth:`equilibrate` once after all initial solitons have
        been placed and before the first :meth:`run`.  If you forget,
        :meth:`run` and :func:`~lfm.experiment.common.gpu_snapshot_loop`
        will call it automatically.

        Examples
        --------
        >>> import lfm
        >>> sim = lfm.Simulation(lfm.SimulationConfig(grid_size=64))
        >>> sim.place_soliton((32, 32, 32), amplitude=6.0)
        >>> sim.equilibrate()           # χ-well forms around the soliton
        >>> print(sim.chi.min())        # should be < 19 (chi0)
        """
        import warnings

        if not self._solitons_placed:
            warnings.warn(
                "equilibrate() called before any solitons were placed — "
                "χ will stay flat at χ₀.  Call place_soliton() first.",
                stacklevel=2,
            )

        pr = self._evolver.get_psi_real()
        pi = self._evolver.get_psi_imag()

        # Build boundary mask if frozen
        bmask = None
        if self.config.boundary_type == BoundaryType.FROZEN:
            bmask = ~make_interior_mask(self.config.grid_size, self.config.boundary_fraction)

        chi_eq = equilibrate_from_fields(
            pr,
            pi,
            chi0=self.config.chi0,
            kappa=self.config.kappa,
            e0_sq=self.config.e0_sq,
            boundary_mask=bmask,
        )

        if self._velocity_boosts:
            # Moving solitons: set chi_current = chi_eq, then compute
            # chi_prev via advective shift dχ/dt = −v·∇χ  so the well
            # co-moves with the soliton from the very first leapfrog step.
            self._evolver.set_chi_current(chi_eq)

            # Gradient of the equilibrium chi field
            grad_x = np.gradient(chi_eq, axis=0).astype(np.float32)
            grad_y = np.gradient(chi_eq, axis=1).astype(np.float32)
            grad_z = np.gradient(chi_eq, axis=2).astype(np.float32)

            # Build density-weighted velocity field from all boosted solitons:
            #   v(r) = Σ_i v_i·ρ_i(r) / Σ_i ρ_i(r)
            N = self.config.grid_size
            v_dot_grad = np.zeros((N, N, N), dtype=np.float32)
            for (vx, vy, vz), rho in self._velocity_boosts:
                v_dot_grad += rho * (vx * grad_x + vy * grad_y + vz * grad_z)
            total_rho: np.ndarray = sum(rho for _, rho in self._velocity_boosts)  # type: ignore[assignment]
            mask = total_rho > 1e-10 * total_rho.max()
            v_dot_grad[mask] /= total_rho[mask]
            v_dot_grad[~mask] = 0.0

            # chi_prev = chi_eq + dt·(v·∇χ)  (note sign: prev is one step
            # BEFORE current, so the well was shifted backward)
            dt = self.config.dt
            chi_prev = chi_eq + dt * v_dot_grad
            self._evolver.set_chi_prev(chi_prev)

            self._velocity_boosts.clear()
        else:
            # Static solitons: old behaviour, chi_prev = chi_current.
            self._evolver.set_chi(chi_eq)

        self._equilibrated = True

    def _auto_equilibrate(self) -> None:
        """Run equilibration automatically if solitons were placed but
        ``equilibrate()`` was never called.

        Called by :meth:`run` and
        :func:`~lfm.experiment.common.gpu_snapshot_loop` so that
        experiments cannot accidentally evolve a flat-χ grid.
        """
        if self._solitons_placed and not self._equilibrated:
            import warnings

            warnings.warn(
                "Solitons were placed but equilibrate() was never called — "
                "auto-equilibrating χ now.  Call sim.equilibrate() explicitly "
                "to silence this warning.",
                stacklevel=3,
            )
            self.equilibrate()

    def place_barrier(
        self,
        axis: int = 2,
        position: int | None = None,
        height: float | None = None,
        thickness: int = 2,
        slits=None,
        slit_axis: int | None = None,
        absorb: bool = True,
        transit_steps: int = 10,
    ) -> Barrier:
        """Place a χ-potential barrier with configurable slit openings.

        Convenience method that creates and applies a
        :class:`~lfm.experiment.Barrier` attached to this simulation.

        Parameters
        ----------
        axis : int
            Propagation axis perpendicular to the barrier (default 2 = z).
        position : int or None
            Grid index along *axis* for the barrier centre.
        height : float or None
            χ value inside solid barrier cells (default ``CHI0 + 50``).
        thickness : int
            Barrier depth in cells.
        slits : list of :class:`~lfm.experiment.Slit` or None
            Slit specifications.  Two symmetric slits by default.
        slit_axis : int or None
            Transverse axis along which slit centres are positioned.
        absorb : bool
            Also zero Ψ inside the solid barrier each step.

        Returns
        -------
        barrier : :class:`~lfm.experiment.Barrier`
            The barrier object (register its ``step_callback`` with
            :meth:`run` or :meth:`run_with_snapshots`).
        """
        from lfm.experiment.barrier import Barrier

        return Barrier(
            self,
            axis=axis,
            position=position,
            height=height,
            thickness=thickness,
            slits=slits,
            slit_axis=slit_axis,
            absorb=absorb,
            transit_steps=transit_steps,
        )

    def add_detector(
        self,
        axis: int = 2,
        position: int | None = None,
        field: str = "energy_density",
    ) -> DetectorScreen:
        """Add a detector screen that records field intensity at a plane.

        Creates a :class:`~lfm.experiment.DetectorScreen` attached to
        this simulation.

        Parameters
        ----------
        axis : int
            Axis perpendicular to the detector plane (same as barrier axis).
        position : int or None
            Grid index along *axis*.  Defaults to 80 % of grid size.
        field : str
            Which field to record (``"energy_density"``, ``"psi_real"``,
            etc.).

        Returns
        -------
        screen : :class:`~lfm.experiment.DetectorScreen`
            Record frames by calling ``screen.step_callback`` or
            ``screen.record()`` at each step.
        """
        from lfm.experiment.detector import DetectorScreen

        return DetectorScreen(self, axis=axis, position=position, field=field)

    def add_source(
        self,
        *,
        axis: int = 2,
        position: float = 0.25,
        omega: float = 2.0,
        amplitude: float = 2.0,
        envelope_sigma: float = 0.25,
        phase: float = 0.0,
        boost: float = 10.0,
    ) -> ContinuousSource:
        """Add a CW monochromatic source plane (Paper-055 technique).

        Creates a :class:`~lfm.experiment.ContinuousSource` attached
        to this simulation.  All position/size parameters accept
        fractional (< 1.0 → fraction of N) or absolute cell values,
        so the same specification auto-scales with grid size.

        Parameters
        ----------
        axis : int
            Propagation axis (default 2 = z).
        position : float
            Source plane location.  Fractional (< 1) or absolute cell.
        omega : float
            Drive angular frequency.  Must exceed ``chi0`` for
            propagation.
        amplitude : float
            Peak injection amplitude.
        envelope_sigma : float
            Transverse Gaussian 1/e half-width (fractional or absolute).
        phase : float
            Complex phase of source (0 = electron, π = positron).
        boost : float
            Extra multiplicative factor per step (default 10.0).

        Returns
        -------
        source : ContinuousSource
        """
        from lfm.experiment.source import ContinuousSource

        return ContinuousSource(
            self,
            axis=axis,
            position=position,
            omega=omega,
            amplitude=amplitude,
            envelope_sigma=envelope_sigma,
            phase=phase,
            boost=boost,
        )

    # ── Evolution ─────────────────────────────────────────

    def run(
        self,
        steps: int,
        callback: Callable[[Simulation, int], None] | None = None,
        record_metrics: bool = True,
        evolve_chi: bool = True,
    ) -> None:
        """Run the simulation for a number of steps.

        Advances both fields (GOV-01 for Ψ, GOV-02 for χ) using the
        double-buffer leapfrog integrator.  Metric snapshots are appended
        to :attr:`history` at every ``config.report_interval`` steps.

        Parameters
        ----------
        steps : int
            Number of leapfrog steps to advance.
        callback : callable or None
            If provided, called as ``callback(sim, step)`` once per
            ``config.report_interval`` steps.  Use this for progress
            reporting or live visualisation.
        record_metrics : bool
            If ``True`` (default), append a :meth:`metrics` snapshot to
            :attr:`history` every ``report_interval`` steps.
        evolve_chi : bool
            If ``True`` (default), evolve both Ψ and χ via GOV-01+GOV-02.
            Set to ``False`` to freeze χ and evolve only Ψ (GOV-01 only).
            Used by the self-consistent-field eigenmode solver.

        Examples
        --------
        Basic run:

        >>> sim.run(steps=5_000)
        >>> print(sim.metrics())

        With a progress callback:

        >>> def progress(s, step):
        ...     print(f"step {step}  chi_min={s.chi.min():.3f}")
        >>> sim.run(10_000, callback=progress)
        """
        self._auto_equilibrate()

        # Snapshot previous state for energy calculations
        self._psi_r_prev = self._evolver.get_psi_real().copy()
        pi = self._evolver.get_psi_imag()
        self._psi_i_prev = pi.copy() if pi is not None else None

        def _internal_callback(evolver: Evolver, step: int) -> None:
            if record_metrics:
                m = self.metrics()
                m["step"] = float(step)
                self._history.append(m)
            if callback is not None:
                callback(self, step)
            # Update prev snapshot
            self._psi_r_prev = evolver.get_psi_real().copy()
            pi = evolver.get_psi_imag()
            self._psi_i_prev = pi.copy() if pi is not None else None

        self._evolver.evolve(
            steps,
            callback=_internal_callback,
            freeze_chi=(not evolve_chi),
        )

    def run_driven(
        self,
        steps: int,
        chi_forcing: Callable[[float], NDArray[np.float32]],
        record_metrics: bool = False,
    ) -> None:
        """Run with χ forced at *every* leapfrog step by an external function.

        At each step, χ is overwritten by ``chi_forcing(t)`` before the
        leapfrog update.  This is the correct way to study parametric
        resonance: the forcing is applied at the native ``dt`` rate rather
        than once per N-step block.

        Parameters
        ----------
        steps : int
            Number of leapfrog steps.
        chi_forcing : callable(t) -> ndarray
            Function of simulation time ``t`` (float) returning either a
            scalar or a (N,N,N) float32 array.  Use default-argument capture
            to close over loop variables correctly in sweeps::

                sim.run_driven(
                    1000,
                    chi_forcing=lambda t, A=3.0, w=omega:
                        np.full((N, N, N),
                                lfm.CHI0 + A * np.sin(w * t),
                                dtype=np.float32),
                )
        record_metrics : bool
            If True, append :meth:`metrics` to :attr:`history` every
            ``config.report_interval`` steps.
        """
        evolver = self._evolver
        dt = self.config.dt
        base_step = self.step

        # run_driven overwrites chi externally; equilibration is irrelevant
        # but we still mark it to avoid a stale warning on a later run()
        self._equilibrated = True

        # Snapshot state before this driven run
        self._psi_r_prev = evolver.get_psi_real().copy()
        pi = evolver.get_psi_imag()
        self._psi_i_prev = pi.copy() if pi is not None else None

        for s in range(steps):
            t = (base_step + s) * dt
            chi_raw = chi_forcing(t)
            chi_arr = np.asarray(chi_raw, dtype=np.float32)
            if chi_arr.ndim == 0:
                N = self.config.grid_size
                chi_arr = np.full((N, N, N), float(chi_arr), dtype=np.float32)
            evolver.set_chi(chi_arr)
            evolver.evolve(1)
            self._psi_r_prev = evolver.get_psi_real().copy()
            pi = evolver.get_psi_imag()
            self._psi_i_prev = pi.copy() if pi is not None else None
            if record_metrics and ((base_step + s + 1) % self.config.report_interval == 0):
                m = self.metrics()
                m["step"] = float(base_step + s + 1)
                self._history.append(m)

    def run_with_snapshots(
        self,
        steps: int,
        snapshot_every: int = 100,
        fields: list[str] | None = None,
        callback: Callable[[Simulation, int], None] | None = None,
        step_callback: Callable[[Simulation, int], None] | None = None,
        record_metrics: bool = True,
        evolve_chi: bool = True,
    ) -> list[dict[str, NDArray[np.float32]]]:
        """Run and accumulate field snapshots at regular intervals.

        Snapshots are stored in memory as copies of the requested fields.
        Use these with :func:`lfm.viz.animate_slice` to create animations.

        Parameters
        ----------
        step_callback : callable or None
            If provided, called at **every** leapfrog step.  Use this for
            things that must be re-enforced each step (e.g.
            ``barrier.step_callback``).  The ``callback`` parameter is
            called once per ``snapshot_every`` batch instead.

        Parameters
        ----------
        steps : int
            Number of leapfrog steps.
        snapshot_every : int
            Save a snapshot every this many steps (aligned to
            ``config.report_interval``).
        fields : list of str or None
            Which fields to capture: any subset of
            ``["chi", "psi_real", "psi_imag", "energy_density"]``.
            Defaults to ``["chi"]``.
        callback : callable or None
            Optional callback as in :meth:`run`.
        record_metrics : bool
            Also append metrics to :attr:`history` every report_interval.
        evolve_chi : bool
            If ``False``, freeze χ and evolve only Ψ (GOV-01 only).
            Useful for wave-optics demos where κ≈0 so χ barely moves —
            skipping GOV-02 gives ~40 % speedup with no physics change.

        Returns
        -------
        list of dict
            Each entry is ``{"step": int, "chi": array, ...}`` for each
            snapshot taken.

        Examples
        --------
        >>> snaps = sim.run_with_snapshots(5000, snapshot_every=200)
        >>> lfm.animate_slice(snaps, save_path="chi_evolution.gif")
        """
        self._auto_equilibrate()

        if fields is None:
            fields = ["chi"]

        snapshots: list[dict] = []
        n_full, remainder = divmod(steps, snapshot_every)

        def _take_snap() -> dict:
            snap: dict = {"step": self.step}
            if "chi" in fields:
                snap["chi"] = self.chi.copy()
            if "psi_real" in fields:
                snap["psi_real"] = self.psi_real.copy()
            if "psi_imag" in fields and self.psi_imag is not None:
                snap["psi_imag"] = self.psi_imag.copy()
            if "energy_density" in fields:
                snap["energy_density"] = self.energy_density.copy()
            return snap

        # Wrap step_callback so it fires through run()'s report_interval
        # mechanism.  We temporarily override report_interval to 1 so it
        # fires every step.  When step_callback is None, run without it.
        if step_callback is not None:
            orig_interval = self.config.report_interval
            self.config.report_interval = 1
            try:
                for _ in range(n_full):
                    # Pass record_metrics=False here: report_interval is
                    # temporarily 1, which would fire metrics() on EVERY step
                    # (thousands of GPU->CPU copies). We record metrics once
                    # per snapshot boundary below instead.
                    self.run(
                        snapshot_every,
                        callback=step_callback,
                        record_metrics=False,
                        evolve_chi=evolve_chi,
                    )
                    if callback is not None:
                        callback(self, self.step)
                    snapshots.append(_take_snap())
                    if record_metrics:
                        m = self.metrics()
                        m["step"] = float(self.step)
                        self._history.append(m)

                if remainder > 0:
                    self.run(
                        remainder,
                        callback=step_callback,
                        record_metrics=False,
                        evolve_chi=evolve_chi,
                    )
                    if callback is not None:
                        callback(self, self.step)
                    snapshots.append(_take_snap())
                    if record_metrics:
                        m = self.metrics()
                        m["step"] = float(self.step)
                        self._history.append(m)
            finally:
                self.config.report_interval = orig_interval
        else:
            for _ in range(n_full):
                # run() fires callback at report_interval, but we always
                # fire the snapshot callback explicitly after each batch so
                # it occurs every snapshot_every steps regardless.
                self.run(
                    snapshot_every,
                    callback=None,
                    record_metrics=record_metrics,
                    evolve_chi=evolve_chi,
                )
                if callback is not None:
                    callback(self, self.step)
                snapshots.append(_take_snap())

            if remainder > 0:
                self.run(
                    remainder,
                    callback=None,
                    record_metrics=record_metrics,
                    evolve_chi=evolve_chi,
                )
                if callback is not None:
                    callback(self, self.step)
                snapshots.append(_take_snap())

        return snapshots

    # ── Analysis ──────────────────────────────────────────

    def get_interior_mask(self) -> NDArray[np.bool_]:
        """Get (cached) interior mask for analysis."""
        if self._interior_mask is None:
            self._interior_mask = make_interior_mask(
                self.config.grid_size, self.config.boundary_fraction
            )
        return self._interior_mask

    def metrics(self) -> dict[str, float]:
        """Compute snapshot metrics for the current simulation state.

        Returns
        -------
        dict[str, float]
            A flat dictionary with the following keys:

            ``e_kinetic``, ``e_gradient``, ``e_potential``
                Energy decomposition (Hamiltonian components).
            ``e_total``
                Sum of all three energy components.
            ``chi_min``, ``chi_max``, ``chi_mean``, ``chi_std``
                Spatial statistics of the χ field (interior masked).
            ``well_fraction``
                Fraction of interior cells with χ < ``chi0 − 2``.
            ``void_fraction``
                Fraction of interior cells with χ > ``chi0 − 0.5``.
            ``psi_norm``
                L² norm of |Ψ|² over the interior.
            ``step``
                Present timestep counter (added by :meth:`run`).

        Examples
        --------
        >>> m = sim.metrics()
        >>> print(f"E_total={m['e_total']:.4f}  chi_min={m['chi_min']:.3f}")
        """
        pr = self._evolver.get_psi_real()
        pi = self._evolver.get_psi_imag()

        # Use previous step if available, else same step (kinetic = 0)
        pr_prev = self._psi_r_prev if self._psi_r_prev is not None else pr
        pi_prev = self._psi_i_prev if self._psi_i_prev is not None else pi

        mask = self.get_interior_mask()

        return compute_metrics(
            psi_r=pr,
            psi_r_prev=pr_prev,
            chi=self._evolver.get_chi(),
            dt=self.config.dt,
            c=self.config.c,
            psi_i=pi,
            psi_i_prev=pi_prev,
            interior_mask=mask,
        )

    def total_energy(self) -> float:
        """Compute total integrated energy at current step."""
        pr = self._evolver.get_psi_real()
        pi = self._evolver.get_psi_imag()
        pr_prev = self._psi_r_prev if self._psi_r_prev is not None else pr
        pi_prev = self._psi_i_prev if self._psi_i_prev is not None else pi

        return total_energy(
            pr,
            pr_prev,
            self._evolver.get_chi(),
            self.config.dt,
            self.config.c,
            pi,
            pi_prev,
        )

    # ── Checkpoint / Resume ───────────────────────────────

    def save_checkpoint(self, path: str | Path) -> None:
        """Save simulation state to a .npz file for later resumption.

        Saves fields, step counter, config, and metric history.

        Parameters
        ----------
        path : str or Path
            Output file path (should end in .npz).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, object] = {
            "step": np.int64(self.step),
            "chi": self.get_chi(),
            "psi_real": self.get_psi_real(),
        }
        pi = self.get_psi_imag()
        if pi is not None:
            data["psi_imag"] = pi
        if self._psi_r_prev is not None:
            data["psi_real_prev"] = self._psi_r_prev
        if self._psi_i_prev is not None:
            data["psi_imag_prev"] = self._psi_i_prev

        # S_a auxiliary fields (v16 confinement)
        sa = self._evolver.get_sa_fields()
        if sa is not None:
            data["sa_fields"] = sa

        # Serialize config + history as JSON strings
        cfg_dict = {k: v for k, v in vars(self.config).items() if not k.startswith("_")}
        # Convert enums to their values
        for k, v in cfg_dict.items():
            if hasattr(v, "value"):
                cfg_dict[k] = v.value
        data["config_json"] = np.array(json.dumps(cfg_dict))
        data["history_json"] = np.array(json.dumps(self._history))

        np.savez_compressed(str(path), **data)  # type: ignore[arg-type]

    @classmethod
    def load_checkpoint(cls, path: str | Path, backend: str = "auto") -> Simulation:
        """Load a simulation from a checkpoint file.

        Parameters
        ----------
        path : str or Path
            Path to .npz checkpoint file.
        backend : str
            Backend preference: 'auto', 'cpu', or 'gpu'.

        Returns
        -------
        Simulation
            Restored simulation ready to continue with run().
        """
        path = Path(path)
        data = np.load(str(path), allow_pickle=False)

        # Restore config (exclude derived fields that are computed in __post_init__)
        cfg_dict = json.loads(str(data["config_json"]))
        cfg_dict["field_level"] = FieldLevel(cfg_dict["field_level"])
        cfg_dict["boundary_type"] = BoundaryType(cfg_dict["boundary_type"])
        for key in ("dx", "sigma"):
            cfg_dict.pop(key, None)
        config = SimulationConfig(**cfg_dict)

        sim = cls(config, backend=backend)

        # Restore fields
        sim._evolver.set_psi_real(data["psi_real"])
        if "psi_imag" in data:
            sim._evolver.set_psi_imag(data["psi_imag"])
        sim._evolver.set_chi(data["chi"])
        sim._evolver.step = int(data["step"])

        # Restore prev fields for energy calculation
        if "psi_real_prev" in data:
            sim._psi_r_prev = data["psi_real_prev"]
        if "psi_imag_prev" in data:
            sim._psi_i_prev = data["psi_imag_prev"]

        # Restore S_a fields if present
        if "sa_fields" in data and sim._evolver.sa_A is not None:
            sim._evolver.set_sa_fields(data["sa_fields"])

        # Restore history
        sim._history = json.loads(str(data["history_json"]))

        return sim
