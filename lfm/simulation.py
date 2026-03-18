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

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from lfm.analysis.energy import energy_components, total_energy
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

    @property
    def step(self) -> int:
        """Current simulation step."""
        return self._evolver.step

    @property
    def history(self) -> list[dict[str, float]]:
        """List of metric snapshots collected during run()."""
        return self._history

    # ── Field access ──────────────────────────────────────

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

    # ── Field initialization ──────────────────────────────

    def place_soliton(
        self,
        position: tuple[float, float, float],
        amplitude: float | None = None,
        sigma: float | None = None,
        phase: float = 0.0,
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
        """
        N = self.config.grid_size
        amp = amplitude if amplitude is not None else self.config.e_amplitude
        sig = sigma if sigma is not None else self.config.sigma

        pr, pi = gaussian_soliton(N, position, amp, sig, phase)

        # Add to existing field
        current_r = self._evolver.get_psi_real()
        new_r = current_r + pr
        self._evolver.set_psi_real(new_r)

        if self.config.field_level != FieldLevel.REAL:
            current_i = self._evolver.get_psi_imag()
            if current_i is not None:
                new_i = current_i + pi
                self._evolver.set_psi_imag(new_i)

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
                N, positions, amp, sig,
                phases=phases, colors=colors,
                n_colors=self.config.n_colors,
            )
        else:
            # Single-component: sum all solitons
            pr_total = np.zeros((N, N, N), dtype=np.float32)
            pi_total = np.zeros((N, N, N), dtype=np.float32)
            if phases is None:
                phases = [0.0] * len(positions)
            for pos, ph in zip(positions, phases):
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

    def equilibrate(self) -> None:
        """Poisson-equilibrate χ from current Ψ field.

        Solves GOV-04 quasi-static limit: ∇²δχ = κ(|Ψ|² − E₀²).
        """
        pr = self._evolver.get_psi_real()
        pi = self._evolver.get_psi_imag()

        # Build boundary mask if frozen
        bmask = None
        if self.config.boundary_type == BoundaryType.FROZEN:
            bmask = ~make_interior_mask(
                self.config.grid_size, self.config.boundary_fraction
            )

        chi = equilibrate_from_fields(
            pr, pi,
            chi0=self.config.chi0,
            kappa=self.config.kappa,
            e0_sq=self.config.e0_sq,
            boundary_mask=bmask,
        )
        self._evolver.set_chi(chi)

    # ── Evolution ─────────────────────────────────────────

    def run(
        self,
        steps: int,
        callback: Callable[[Simulation, int], None] | None = None,
        record_metrics: bool = True,
    ) -> None:
        """Run the simulation for a number of steps.

        Parameters
        ----------
        steps : int
            Number of leapfrog steps.
        callback : callable or None
            Called as callback(sim, step) every report_interval.
        record_metrics : bool
            If True, record metrics every report_interval.
        """
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

        self._evolver.evolve(steps, callback=_internal_callback)

    # ── Analysis ──────────────────────────────────────────

    def get_interior_mask(self) -> NDArray[np.bool_]:
        """Get (cached) interior mask for analysis."""
        if self._interior_mask is None:
            self._interior_mask = make_interior_mask(
                self.config.grid_size, self.config.boundary_fraction
            )
        return self._interior_mask

    def metrics(self) -> dict[str, float]:
        """Compute snapshot metrics for the current state.

        Returns
        -------
        dict[str, float]
            Energy components, chi statistics, structure metrics.
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
            pr, pr_prev,
            self._evolver.get_chi(),
            self.config.dt, self.config.c,
            pi, pi_prev,
        )
