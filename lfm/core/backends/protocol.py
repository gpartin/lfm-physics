"""
Backend Protocol
================

Defines the interface that all compute backends must implement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@runtime_checkable
class Backend(Protocol):
    """Interface for LFM compute backends (CPU or GPU).

    Every backend operates on flattened arrays using the configured precision
    and double-buffering.
    The step method reads from one set of buffers and writes to another,
    then the caller toggles which set is "current".
    """

    @property
    def name(self) -> str:
        """Human-readable backend name, e.g. 'numpy' or 'cupy'."""
        ...

    @property
    def dtype(self) -> object:
        """NumPy dtype used for persistent state and scalar parameters."""
        ...

    def allocate(
        self,
        N: int,
        n_psi_arrays: int,
        chi0: float,
    ) -> dict[str, object]:
        """Allocate all arrays needed for double-buffered evolution.

        Parameters
        ----------
        N : int
            Grid points per axis.
        n_psi_arrays : int
            Number of psi component arrays. For real: 1, complex: 2, color: 6.
        chi0 : float
            Background chi value for initialization.

        Returns
        -------
        dict
            Keys: 'psi_A', 'psi_prev_A', 'chi_A', 'chi_prev_A',
                  'psi_B', 'psi_prev_B', 'chi_B', 'chi_prev_B',
                  'boundary_mask'.  Values are backend-native arrays.
        """
        ...

    def create_boundary_mask(
        self,
        N: int,
        boundary_fraction: float,
    ) -> object:
        """Create spherical frozen boundary mask using the state dtype."""
        ...

    def step_real(
        self,
        psi_in: object,
        psi_prev_in: object,
        chi_in: object,
        chi_prev_in: object,
        boundary_mask: object,
        psi_out: object,
        psi_prev_out: object,
        chi_out: object,
        chi_prev_out: object,
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
        """One leapfrog step for real E field (Level 0)."""
        ...

    def step_real_gravity_recovery(
        self,
        psi_in: object,
        psi_prev_in: object,
        chi_in: object,
        chi_prev_in: object,
        boundary_mask: object,
        psi_out: object,
        psi_prev_out: object,
        chi_out: object,
        chi_prev_out: object,
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
        """One experiment-only local GOV-02 gravity-recovery step."""
        ...

    def step_complex_gravity_recovery(
        self,
        psi_r_in: object,
        psi_r_prev_in: object,
        psi_i_in: object,
        psi_i_prev_in: object,
        chi_in: object,
        chi_prev_in: object,
        boundary_mask: object,
        psi_r_out: object,
        psi_r_prev_out: object,
        psi_i_out: object,
        psi_i_prev_out: object,
        chi_out: object,
        chi_prev_out: object,
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
        """One complex flat-octic GOV-02 gravity-recovery step."""
        ...

    def step_color_gravity_recovery(
        self,
        psi_r_in: object,
        psi_r_prev_in: object,
        psi_i_in: object,
        psi_i_prev_in: object,
        chi_in: object,
        chi_prev_in: object,
        boundary_mask: object,
        psi_r_out: object,
        psi_r_prev_out: object,
        psi_i_out: object,
        psi_i_prev_out: object,
        chi_out: object,
        chi_prev_out: object,
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
        sa_fields_in: object | None = None,
        sa_fields_out: object | None = None,
        sa_gamma: float = 0.1,
        sa_d: float = 4.9,
        use_stencil19_noether_current: bool = False,
        inv_dx2: float = 1.0,
        enable_chi_floor: bool = False,
    ) -> None:
        """One full three-channel flat-octic GOV-02 candidate step."""
        ...

    def step_complex(
        self,
        psi_r_in: object,
        psi_r_prev_in: object,
        psi_i_in: object,
        psi_i_prev_in: object,
        chi_in: object,
        chi_prev_in: object,
        boundary_mask: object,
        psi_r_out: object,
        psi_r_prev_out: object,
        psi_i_out: object,
        psi_i_prev_out: object,
        chi_out: object,
        chi_prev_out: object,
        N: int,
        dt2: float,
        kappa: float,
        lambda_self: float,
        chi0: float,
        e0_sq: float,
        epsilon_w: float,
        kappa_c: float = 0.0,
        epsilon_cc: float = 0.0,
        kappa_string: float = 0.0,
        kappa_tube: float = 0.0,
        sa_fields_in: object | None = None,
        sa_fields_out: object | None = None,
        sa_gamma: float = 0.1,
        sa_d: float = 4.9,
        dt: float = 0.02,
        *,
        use_stencil19_noether_current: bool = False,
        inv_dx2: float = 1.0,
        enable_chi_floor: bool = True,
    ) -> None:
        """One leapfrog step for complex Ψ field (Level 1)."""
        ...

    def step_color(
        self,
        psi_r_in: object,
        psi_r_prev_in: object,
        psi_i_in: object,
        psi_i_prev_in: object,
        chi_in: object,
        chi_prev_in: object,
        boundary_mask: object,
        psi_r_out: object,
        psi_r_prev_out: object,
        psi_i_out: object,
        psi_i_prev_out: object,
        chi_out: object,
        chi_prev_out: object,
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
        """One leapfrog step for 3-color complex Ψₐ (Level 2)."""
        ...

    def to_numpy(self, arr: object) -> NDArray[np.floating]:
        """Convert backend array to numpy (no-op for NumPy backend)."""
        ...

    def from_numpy(self, arr: NDArray) -> object:
        """Convert numpy array to backend-native format."""
        ...

    def apply_complex_phase_map(
        self,
        psi_r: object,
        psi_i: object,
        phase_cos: object,
        phase_sin: object,
    ) -> None:
        """Rotate a flattened complex field in place by a local phase map."""
        ...
