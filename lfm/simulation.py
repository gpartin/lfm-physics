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
from typing import Callable

import numpy as np
from numpy.typing import NDArray

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

    def set_chi(self, value: NDArray[np.float32]) -> None:
        """Set χ field."""
        self._evolver.set_chi(value)

    @property
    def sa_fields(self) -> "NDArray[np.float32] | None":
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
        """
        N = self.config.grid_size
        amp = amplitude if amplitude is not None else self.config.e_amplitude
        sig = sigma if sigma is not None else self.config.sigma

        if velocity is not None:
            # Velocity boost: spatial phase k·(∂r − r₀), k = χ₀·v/c
            vx, vy, vz = velocity
            chi0 = self.config.chi0
            c = self.config.c
            kx = chi0 * vx / c
            ky = chi0 * vy / c
            kz = chi0 * vz / c

            x = np.arange(N, dtype=np.float32)
            X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
            px, py, pz = position
            r2 = (X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2
            envelope = (amp * np.exp(-r2 / (2.0 * sig**2))).astype(np.float32)
            phase_grid = (phase + kx * (X - px) + ky * (Y - py) + kz * (Z - pz)).astype(np.float32)
            pr = (envelope * np.cos(phase_grid)).astype(np.float32)
            pi = (envelope * np.sin(phase_grid)).astype(np.float32)
        else:
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
            bmask = ~make_interior_mask(self.config.grid_size, self.config.boundary_fraction)

        chi = equilibrate_from_fields(
            pr,
            pi,
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
        record_metrics: bool = True,
    ) -> list[dict[str, NDArray[np.float32]]]:
        """Run and accumulate field snapshots at regular intervals.

        Snapshots are stored in memory as copies of the requested fields.
        Use these with :func:`lfm.viz.animate_slice` to create animations.

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

        for _ in range(n_full):
            self.run(snapshot_every, callback=callback, record_metrics=record_metrics)
            snapshots.append(_take_snap())

        if remainder > 0:
            self.run(remainder, callback=callback, record_metrics=record_metrics)
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

        np.savez_compressed(str(path), **data)

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
