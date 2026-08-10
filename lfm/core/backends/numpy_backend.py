"""
NumPy Backend (CPU)
===================

Pure-NumPy implementation of LFM leapfrog evolution.
Uses the 19-point isotropic stencil via np.roll.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lfm.config import ChiPotentialModel, Precision
from lfm.core.chi_potentials import potential_force, variable_inertia
from lfm.core.stencils import laplacian_19pt, noether_current_19pt_raw

if TYPE_CHECKING:
    from numpy.typing import NDArray


_GRAVITY_RECOVERY_LINKS = (
    ((1, 0, 0), 1.0 / 3.0),
    ((-1, 0, 0), 1.0 / 3.0),
    ((0, 1, 0), 1.0 / 3.0),
    ((0, -1, 0), 1.0 / 3.0),
    ((0, 0, 1), 1.0 / 3.0),
    ((0, 0, -1), 1.0 / 3.0),
    *tuple(((sx, sy, 0), 1.0 / 6.0) for sx in (-1, 1) for sy in (-1, 1)),
    *tuple(((sx, 0, sz), 1.0 / 6.0) for sx in (-1, 1) for sz in (-1, 1)),
    *tuple(((0, sy, sz), 1.0 / 6.0) for sy in (-1, 1) for sz in (-1, 1)),
)


class NumpyBackend:
    """CPU compute backend using NumPy."""

    def __init__(self, precision: Precision | str = Precision.FLOAT32) -> None:
        self.precision = Precision(precision)
        self.dtype = np.dtype(self.precision.value)

    @property
    def name(self) -> str:
        return "numpy"

    def allocate(
        self,
        N: int,
        n_psi_arrays: int,
        chi0: float,
    ) -> dict[str, NDArray[np.floating]]:
        total = N**3
        psi_size = n_psi_arrays * total
        zero_psi = np.zeros(psi_size, dtype=self.dtype)
        chi_init = np.full(total, chi0, dtype=self.dtype)
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
    ) -> NDArray[np.floating]:
        """Return a smooth cos² absorption mask in [0, 1].

        0 = fully transparent (interior), 1 = fully absorbed (boundary).
        A cosine taper from r_freeze to r_max prevents the leapfrog
        reflection that a hard (binary) cutoff would cause.
        """
        center = N / 2.0
        r_max = N / 2.0
        r_freeze = (1.0 - boundary_fraction) * r_max
        coords = np.arange(N, dtype=self.dtype) - center + 0.5
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        R = np.sqrt(X**2 + Y**2 + Z**2)
        # Smooth cosine taper: 0 at r_freeze, 1 at r_max
        t = np.clip((R - r_freeze) / (r_max - r_freeze), 0.0, 1.0)
        mask = (np.sin(0.5 * np.pi * t) ** 2).astype(self.dtype)
        return mask.ravel()

    def _laplacian_3d(self, flat: NDArray[np.floating], N: int) -> NDArray[np.floating]:
        """19-point Laplacian on a flat (N³,) or (K*N³,) array.

        Reshapes to 3D, computes, and flattens back.
        """
        field = flat.reshape(N, N, N)
        result = laplacian_19pt(field)
        return result.ravel()

    def step_real(
        self,
        psi_in: NDArray,
        psi_prev_in: NDArray,
        chi_in: NDArray,
        chi_prev_in: NDArray,
        boundary_mask: NDArray,
        psi_out: NDArray,
        psi_prev_out: NDArray,
        chi_out: NDArray,
        chi_prev_out: NDArray,
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
        E = psi_in
        E_prev = psi_prev_in
        chi = chi_in
        chi_prev = chi_prev_in

        lap_E = inv_dx2 * self._laplacian_3d(E, N)
        lap_chi = inv_dx2 * self._laplacian_3d(chi, N)
        chi_sq = chi * chi

        # GOV-01
        E_new = 2.0 * E - E_prev + dt2 * (lap_E - chi_sq * E)

        # GOV-02 v28.0
        chi_source = (kappa / chi0) * chi * (E * E - e0_sq)
        chi_accel = lap_chi - chi_source
        if lambda_self > 0:
            chi_accel -= 4.0 * lambda_self * chi * (chi_sq - chi0 * chi0)
        chi_new = 2.0 * chi - chi_prev + dt2 * chi_accel

        if enable_chi_floor:
            np.clip(chi_new, -chi0, None, out=chi_new)

        # Absorbing boundary — damp both new field AND prev field so the
        # leapfrog sees no energy at the boundary on the next step.
        absorb = 1.0 - boundary_mask
        E_new *= absorb
        chi_new = boundary_mask * chi0 + absorb * chi_new

        # Write to output buffers (double-buffer swap)
        np.copyto(psi_out, E_new)
        np.copyto(psi_prev_out, E * absorb)  # damp prev too — prevents reflection
        np.copyto(chi_out, chi_new)
        np.copyto(chi_prev_out, chi)

    def step_real_gravity_recovery(
        self,
        psi_in: NDArray,
        psi_prev_in: NDArray,
        chi_in: NDArray,
        chi_prev_in: NDArray,
        boundary_mask: NDArray,
        psi_out: NDArray,
        psi_prev_out: NDArray,
        chi_out: NDArray,
        chi_prev_out: NDArray,
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
        """Advance the experiment-only local GOV-02 candidate system."""

        model = ChiPotentialModel(potential_model)
        E = psi_in
        E_prev = psi_prev_in
        chi = chi_in
        chi_prev = chi_prev_in
        lap_E = inv_dx2 * self._laplacian_3d(E, N)
        lap_chi = inv_dx2 * self._laplacian_3d(chi, N)
        chi_sq = chi * chi
        E_new = E.copy() if freeze_psi else 2.0 * E - E_prev + dt**2 * (lap_E - chi_sq * E)
        source_density = E * E - e0_sq
        chi_accel = (
            lap_chi
            - (kappa / chi0) * chi * source_density
            + potential_force(
                chi,
                model,
                chi0=chi0,
                lambda_h=lambda_self,
                source_density=source_density,
            )
        )
        if model == ChiPotentialModel.NONLINEAR_GRADIENT:
            field = chi.reshape(N, N, N)
            nonlinear = np.zeros_like(field)
            for offset, weight in _GRAVITY_RECOVERY_LINKS:
                neighbor = np.roll(
                    field,
                    shift=tuple(-value for value in offset),
                    axis=(0, 1, 2),
                )
                nonlinear += weight * (neighbor - field) ** 3 / chi0**2
            chi_accel += inv_dx2 * nonlinear.ravel()
        velocity = (chi - chi_prev) / dt
        if model == ChiPotentialModel.VARIABLE_INERTIA:
            inertia = variable_inertia(chi, chi0=chi0)
            y = (chi_sq - chi0**2) / chi0**2
            inertia_derivative = 4.0 * chi * y / chi0**2
            chi_accel = (chi_accel - 0.5 * inertia_derivative * velocity**2) / inertia
        damping_half_step = 0.5 * relaxation_damping * dt
        chi_new = (2.0 * chi - (1.0 - damping_half_step) * chi_prev + dt**2 * chi_accel) / (
            1.0 + damping_half_step
        )
        if enable_chi_floor:
            np.clip(chi_new, -chi0, None, out=chi_new)
        absorb = 1.0 - boundary_mask
        E_new *= absorb
        chi_new = boundary_mask * chi0 + absorb * chi_new
        np.copyto(psi_out, E_new)
        np.copyto(
            psi_prev_out,
            E if freeze_psi else E * absorb,
        )
        np.copyto(chi_out, chi_new)
        np.copyto(chi_prev_out, chi)

    def step_complex_gravity_recovery(
        self,
        psi_r_in: NDArray,
        psi_r_prev_in: NDArray,
        psi_i_in: NDArray,
        psi_i_prev_in: NDArray,
        chi_in: NDArray,
        chi_prev_in: NDArray,
        boundary_mask: NDArray,
        psi_r_out: NDArray,
        psi_r_prev_out: NDArray,
        psi_i_out: NDArray,
        psi_i_prev_out: NDArray,
        chi_out: NDArray,
        chi_prev_out: NDArray,
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

        model = ChiPotentialModel(potential_model)
        if model != ChiPotentialModel.FLAT_OCTIC:
            raise ValueError("complex gravity recovery currently supports FLAT_OCTIC")
        psi_r = psi_r_in
        psi_i = psi_i_in
        chi = chi_in
        chi_prev = chi_prev_in
        lap_r = inv_dx2 * self._laplacian_3d(psi_r, N)
        lap_i = inv_dx2 * self._laplacian_3d(psi_i, N)
        lap_chi = inv_dx2 * self._laplacian_3d(chi, N)
        chi_sq = chi * chi
        if freeze_psi:
            psi_r_new = psi_r.copy()
            psi_i_new = psi_i.copy()
        else:
            psi_r_new = 2.0 * psi_r - psi_r_prev_in + dt**2 * (lap_r - chi_sq * psi_r)
            psi_i_new = 2.0 * psi_i - psi_i_prev_in + dt**2 * (lap_i - chi_sq * psi_i)
        source_density = psi_r**2 + psi_i**2 - e0_sq
        chi_accel = (
            lap_chi
            - (kappa / chi0) * chi * source_density
            + potential_force(
                chi,
                model,
                chi0=chi0,
                lambda_h=lambda_self,
                source_density=source_density,
            )
        )
        damping_half_step = 0.5 * relaxation_damping * dt
        chi_new = (2.0 * chi - (1.0 - damping_half_step) * chi_prev + dt**2 * chi_accel) / (
            1.0 + damping_half_step
        )
        if enable_chi_floor:
            np.clip(chi_new, -chi0, None, out=chi_new)
        absorb = 1.0 - boundary_mask
        psi_r_new *= absorb
        psi_i_new *= absorb
        chi_new = boundary_mask * chi0 + absorb * chi_new
        np.copyto(psi_r_out, psi_r_new)
        np.copyto(
            psi_r_prev_out,
            psi_r if freeze_psi else psi_r * absorb,
        )
        np.copyto(psi_i_out, psi_i_new)
        np.copyto(
            psi_i_prev_out,
            psi_i if freeze_psi else psi_i * absorb,
        )
        np.copyto(chi_out, chi_new)
        np.copyto(chi_prev_out, chi)

    def step_color_gravity_recovery(
        self,
        psi_r_in: NDArray,
        psi_r_prev_in: NDArray,
        psi_i_in: NDArray,
        psi_i_prev_in: NDArray,
        chi_in: NDArray,
        chi_prev_in: NDArray,
        boundary_mask: NDArray,
        psi_r_out: NDArray,
        psi_r_prev_out: NDArray,
        psi_i_out: NDArray,
        psi_i_prev_out: NDArray,
        chi_out: NDArray,
        chi_prev_out: NDArray,
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
        sa_fields_in: NDArray | None = None,
        sa_fields_out: NDArray | None = None,
        sa_gamma: float = 0.1,
        sa_d: float = 4.9,
        use_stencil19_noether_current: bool = False,
        inv_dx2: float = 1.0,
        enable_chi_floor: bool = False,
    ) -> None:
        """Advance the full three-channel flat-octic candidate system."""

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
        safe_absorb = np.where(absorb > 0.0, absorb, 1.0)
        undamped_chi = (chi_out - boundary_mask * chi0) / safe_absorb
        source_density = np.zeros_like(chi_in)
        total = N**3
        for channel in range(3):
            selected = slice(channel * total, (channel + 1) * total)
            source_density += psi_r_in[selected] ** 2 + psi_i_in[selected] ** 2
        source_density -= e0_sq
        flat_force = potential_force(
            chi_in,
            model,
            chi0=chi0,
            lambda_h=lambda_self,
            source_density=source_density,
        )
        damping_half_step = 0.5 * relaxation_damping * dt
        corrected = (undamped_chi + damping_half_step * chi_prev_in + dt**2 * flat_force) / (
            1.0 + damping_half_step
        )
        if enable_chi_floor:
            np.clip(corrected, -chi0, None, out=corrected)
        np.copyto(
            chi_out,
            boundary_mask * chi0 + absorb * corrected,
        )
        if freeze_psi:
            absorb3 = np.tile(absorb, 3)
            np.copyto(psi_r_out, absorb3 * psi_r_in)
            np.copyto(psi_r_prev_out, psi_r_in)
            np.copyto(psi_i_out, absorb3 * psi_i_in)
            np.copyto(psi_i_prev_out, psi_i_in)

    def step_complex(
        self,
        psi_r_in: NDArray,
        psi_r_prev_in: NDArray,
        psi_i_in: NDArray,
        psi_i_prev_in: NDArray,
        chi_in: NDArray,
        chi_prev_in: NDArray,
        boundary_mask: NDArray,
        psi_r_out: NDArray,
        psi_r_prev_out: NDArray,
        psi_i_out: NDArray,
        psi_i_prev_out: NDArray,
        chi_out: NDArray,
        chi_prev_out: NDArray,
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
        Pr, Pi = psi_r_in, psi_i_in
        chi, chi_prev = chi_in, chi_prev_in
        chi_sq = chi * chi

        lap_Pr = inv_dx2 * self._laplacian_3d(Pr, N)
        lap_Pi = inv_dx2 * self._laplacian_3d(Pi, N)
        lap_chi = inv_dx2 * self._laplacian_3d(chi, N)

        # GOV-01
        Pr_new = 2.0 * Pr - psi_r_prev_in + dt2 * (lap_Pr - chi_sq * Pr)
        Pi_new = 2.0 * Pi - psi_i_prev_in + dt2 * (lap_Pi - chi_sq * Pi)

        # |Ψ|² and momentum density
        psi_sq = Pr * Pr + Pi * Pi

        # j = Im(Ψ*·∇Ψ) via central differences on 3D grid
        Pr3 = Pr.reshape(N, N, N)
        Pi3 = Pi.reshape(N, N, N)
        if use_stencil19_noether_current:
            j_x_3d, j_y_3d, j_z_3d = noether_current_19pt_raw(Pr3, Pi3)
            j_x = j_x_3d.ravel()
            j_y = j_y_3d.ravel()
            j_z = j_z_3d.ravel()
        else:
            # Historical face-only central differences.
            dPr_dx = np.roll(Pr3, -1, 0) - np.roll(Pr3, 1, 0)
            dPr_dy = np.roll(Pr3, -1, 1) - np.roll(Pr3, 1, 1)
            dPr_dz = np.roll(Pr3, -1, 2) - np.roll(Pr3, 1, 2)
            dPi_dx = np.roll(Pi3, -1, 0) - np.roll(Pi3, 1, 0)
            dPi_dy = np.roll(Pi3, -1, 1) - np.roll(Pi3, 1, 1)
            dPi_dz = np.roll(Pi3, -1, 2) - np.roll(Pi3, 1, 2)
            j_x = (Pr3 * dPi_dx - Pi3 * dPr_dx).ravel()
            j_y = (Pr3 * dPi_dy - Pi3 * dPr_dy).ravel()
            j_z = (Pr3 * dPi_dz - Pi3 * dPr_dz).ravel()
        inv_dx = float(np.sqrt(inv_dx2))
        j_x *= inv_dx
        j_y *= inv_dx
        j_z *= inv_dx
        j_total = 0.5 * (j_x + j_y + j_z)

        # GOV-02 v28.0
        chi_source = (kappa / chi0) * chi * (psi_sq + epsilon_w * j_total - e0_sq)
        chi_accel = lap_chi - chi_source
        if lambda_self > 0:
            chi_accel -= 4.0 * lambda_self * chi * (chi_sq - chi0 * chi0)
        chi_new = 2.0 * chi - chi_prev + dt2 * chi_accel

        if enable_chi_floor:
            np.clip(chi_new, -chi0, None, out=chi_new)

        # Absorbing boundary — damp both new and prev to prevent reflection.
        absorb = 1.0 - boundary_mask
        Pr_new *= absorb
        Pi_new *= absorb
        chi_new = boundary_mask * chi0 + absorb * chi_new

        np.copyto(psi_r_out, Pr_new)
        np.copyto(psi_r_prev_out, Pr * absorb)  # damp prev — prevents leapfrog reflection
        np.copyto(psi_i_out, Pi_new)
        np.copyto(psi_i_prev_out, Pi * absorb)  # damp prev
        np.copyto(chi_out, chi_new)
        np.copyto(chi_prev_out, chi)

    def step_color(
        self,
        psi_r_in: NDArray,
        psi_r_prev_in: NDArray,
        psi_i_in: NDArray,
        psi_i_prev_in: NDArray,
        chi_in: NDArray,
        chi_prev_in: NDArray,
        boundary_mask: NDArray,
        psi_r_out: NDArray,
        psi_r_prev_out: NDArray,
        psi_i_out: NDArray,
        psi_i_prev_out: NDArray,
        chi_out: NDArray,
        chi_prev_out: NDArray,
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
        sa_fields_in: NDArray | None = None,
        sa_fields_out: NDArray | None = None,
        sa_gamma: float = 0.1,
        sa_d: float = 4.9,
        dt: float = 0.02,
        *,
        use_stencil19_noether_current: bool = False,
        inv_dx2: float = 1.0,
        enable_chi_floor: bool = True,
    ) -> None:
        total = N**3
        n_colors = 3
        chi, chi_prev = chi_in, chi_prev_in
        chi_sq = chi * chi

        psi_sq_total = np.zeros(total, dtype=self.dtype)
        j_total_acc = np.zeros(total, dtype=self.dtype)
        color_energy = np.zeros((n_colors, total), dtype=self.dtype)

        # For CCV (v15 GOV-02): store per-color per-direction currents
        need_ccv = kappa_string > 0
        if need_ccv:
            j_per_color = np.zeros((n_colors, 3, total), dtype=self.dtype)

        # v15: precompute color average for cross-color coupling
        if epsilon_cc > 0:
            Pr_avg = np.zeros(total, dtype=self.dtype)
            Pi_avg = np.zeros(total, dtype=self.dtype)
            for a in range(n_colors):
                s = slice(a * total, (a + 1) * total)
                Pr_avg += psi_r_in[s]
                Pi_avg += psi_i_in[s]
            Pr_avg /= n_colors
            Pi_avg /= n_colors

        for a in range(n_colors):
            off = a * total
            s = slice(off, off + total)

            Pr = psi_r_in[s]
            Pi = psi_i_in[s]

            lap_Pr = inv_dx2 * self._laplacian_3d(Pr, N)
            lap_Pi = inv_dx2 * self._laplacian_3d(Pi, N)

            # GOV-01
            Pr_new = 2.0 * Pr - psi_r_prev_in[s] + dt2 * (lap_Pr - chi_sq * Pr)
            Pi_new = 2.0 * Pi - psi_i_prev_in[s] + dt2 * (lap_Pi - chi_sq * Pi)

            # v15: cross-color coupling -eps_cc * chi^2 * (Psi_a - Psi_bar)
            if epsilon_cc > 0:
                Pr_new -= dt2 * epsilon_cc * chi_sq * (Pr - Pr_avg)
                Pi_new -= dt2 * epsilon_cc * chi_sq * (Pi - Pi_avg)

            # Boundary is applied below; store un-damped values first
            np.copyto(psi_r_out[s], Pr_new)
            np.copyto(psi_r_prev_out[s], Pr)
            np.copyto(psi_i_out[s], Pi_new)
            np.copyto(psi_i_prev_out[s], Pi)

            ea = Pr * Pr + Pi * Pi
            color_energy[a] = ea
            psi_sq_total += ea

            # per-color momentum currents j_{a,d} = Pr * dPi/dd - Pi * dPr/dd
            Pr3 = Pr.reshape(N, N, N)
            Pi3 = Pi.reshape(N, N, N)
            if use_stencil19_noether_current:
                j_x_3d, j_y_3d, j_z_3d = noether_current_19pt_raw(Pr3, Pi3)
                j_x = j_x_3d.ravel()
                j_y = j_y_3d.ravel()
                j_z = j_z_3d.ravel()
            else:
                dPr_dx = np.roll(Pr3, -1, 0) - np.roll(Pr3, 1, 0)
                dPr_dy = np.roll(Pr3, -1, 1) - np.roll(Pr3, 1, 1)
                dPr_dz = np.roll(Pr3, -1, 2) - np.roll(Pr3, 1, 2)
                dPi_dx = np.roll(Pi3, -1, 0) - np.roll(Pi3, 1, 0)
                dPi_dy = np.roll(Pi3, -1, 1) - np.roll(Pi3, 1, 1)
                dPi_dz = np.roll(Pi3, -1, 2) - np.roll(Pi3, 1, 2)
                j_x = (Pr3 * dPi_dx - Pi3 * dPr_dx).ravel()
                j_y = (Pr3 * dPi_dy - Pi3 * dPr_dy).ravel()
                j_z = (Pr3 * dPi_dz - Pi3 * dPr_dz).ravel()
            inv_dx = float(np.sqrt(inv_dx2))
            j_x *= inv_dx
            j_y *= inv_dx
            j_z *= inv_dx
            j_total_acc += 0.5 * (j_x + j_y + j_z)

            if need_ccv:
                j_per_color[a, 0] = j_x
                j_per_color[a, 1] = j_y
                j_per_color[a, 2] = j_z

        # v14: normalized color variance f_c → color_var_term
        color_var_term = np.zeros(total, dtype=self.dtype)
        if kappa_c > 0:
            sum_sq = np.sum(color_energy**2, axis=0)
            total_sq = psi_sq_total * psi_sq_total
            safe = total_sq > 1e-30
            ratio = np.where(
                safe,
                np.divide(
                    sum_sq, total_sq, where=safe, out=np.zeros_like(sum_sq, dtype=np.float64)
                ),
                0.0,
            )
            f_c = ((ratio - 1.0 / n_colors) * safe).astype(self.dtype)
            color_var_term = ((kappa_c / chi0) * chi * f_c * psi_sq_total).astype(self.dtype)

        # v15 GOV-02: color current variance (CCV)
        # CCV = Σ_d [ Σ_a j²_{a,d} - (1/N_c)(Σ_a j_{a,d})² ]
        ccv_term = np.zeros(total, dtype=self.dtype)
        if need_ccv:
            for d in range(3):
                j_d = j_per_color[:, d, :]  # shape (n_colors, total)
                sum_j_sq = np.sum(j_d**2, axis=0)
                sum_j = np.sum(j_d, axis=0)
                ccv_term += sum_j_sq - (1.0 / n_colors) * sum_j**2

        # v17: Helmholtz-smoothed S_a from |Ψ_a|² (replaces v16 Euler diffusion)
        # S̃_a(k) = γ/(γ + D·k²) · FT[|Ψ_a|²](k)  — quasi-static, unconditionally stable
        scv_term = np.zeros(total, dtype=self.dtype)
        if kappa_tube > 0 and sa_fields_out is not None:
            kx = np.fft.fftfreq(N) * (2.0 * np.pi)
            ky = np.fft.fftfreq(N) * (2.0 * np.pi)
            kz = np.fft.rfftfreq(N) * (2.0 * np.pi)
            k_sq = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
            h_filter = np.float64(sa_gamma) / (np.float64(sa_gamma) + np.float64(sa_d) * k_sq)

            sa_sum = np.zeros(total, dtype=self.dtype)
            sa_sq_sum = np.zeros(total, dtype=self.dtype)
            for a in range(n_colors):
                psi_sq_hat = np.fft.rfftn(color_energy[a].reshape(N, N, N))
                sa_3d = np.fft.irfftn(h_filter * psi_sq_hat, s=(N, N, N), axes=(0, 1, 2))
                sa_flat = np.clip(sa_3d, 0.0, None).astype(self.dtype).ravel()
                np.copyto(sa_fields_out[a * total : (a + 1) * total], sa_flat)
                sa_sum += sa_flat
                sa_sq_sum += sa_flat * sa_flat
            # SCV = Σ_a S_a² - (1/N_c)(Σ_a S_a)²
            scv_term = sa_sq_sum - (1.0 / n_colors) * sa_sum**2

        # GOV-02 v28.0
        lap_chi = inv_dx2 * self._laplacian_3d(chi, N)
        chi_source = (kappa / chi0) * chi * (psi_sq_total + epsilon_w * j_total_acc - e0_sq)
        chi_accel = (
            lap_chi - chi_source - color_var_term - kappa_string * ccv_term - kappa_tube * scv_term
        )
        if lambda_self > 0:
            chi_accel -= 4.0 * lambda_self * chi * (chi_sq - chi0 * chi0)
        chi_new = 2.0 * chi - chi_prev + dt2 * chi_accel

        if enable_chi_floor:
            np.clip(chi_new, -chi0, None, out=chi_new)

        # Absorbing boundary — damp both new and prev to prevent reflection.
        absorb3 = 1.0 - np.tile(boundary_mask, 3)
        absorb1 = 1.0 - boundary_mask
        psi_r_out *= absorb3
        psi_i_out *= absorb3
        psi_r_prev_out *= absorb3  # damp prev too
        psi_i_prev_out *= absorb3  # damp prev too
        chi_new = boundary_mask * chi0 + absorb1 * chi_new

        np.copyto(chi_out, chi_new)
        np.copyto(chi_prev_out, chi)

    def to_numpy(self, arr: NDArray[np.floating]) -> NDArray[np.floating]:
        return arr

    def from_numpy(self, arr: NDArray) -> NDArray[np.floating]:
        return arr.astype(self.dtype) if arr.dtype != self.dtype else arr

    def apply_complex_phase_map(
        self,
        psi_r: NDArray,
        psi_i: NDArray,
        phase_cos: NDArray,
        phase_sin: NDArray,
    ) -> None:
        """Rotate a complex field in place by a local phase map."""

        real = np.asarray(psi_r)
        imag = np.asarray(psi_i)
        cosv = np.asarray(phase_cos, dtype=self.dtype)
        sinv = np.asarray(phase_sin, dtype=self.dtype)
        new_real = real * cosv - imag * sinv
        new_imag = real * sinv + imag * cosv
        np.copyto(real, new_real)
        np.copyto(imag, new_imag)
