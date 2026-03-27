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

    Every backend operates on flattened float32 arrays using double-buffering.
    The step method reads from one set of buffers and writes to another,
    then the caller toggles which set is "current".
    """

    @property
    def name(self) -> str:
        """Human-readable backend name, e.g. 'numpy' or 'cupy'."""
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
        """Create spherical frozen boundary mask (flattened N³ float32)."""
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
    ) -> None:
        """One leapfrog step for real E field (Level 0)."""
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
    ) -> None:
        """One leapfrog step for 3-color complex Ψₐ (Level 2)."""
        ...

    def to_numpy(self, arr: object) -> NDArray[np.float32]:
        """Convert backend array to numpy (no-op for NumPy backend)."""
        ...

    def from_numpy(self, arr: NDArray) -> object:
        """Convert numpy array to backend-native format."""
        ...
