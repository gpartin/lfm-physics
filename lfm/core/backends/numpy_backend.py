"""
NumPy Backend (CPU)
===================

Pure-NumPy implementation of LFM leapfrog evolution.
Uses the 19-point isotropic stencil via np.roll.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lfm.core.stencils import laplacian_19pt


class NumpyBackend:
    """CPU compute backend using NumPy."""

    @property
    def name(self) -> str:
        return "numpy"

    def allocate(
        self,
        N: int,
        n_psi_arrays: int,
        chi0: float,
    ) -> dict[str, NDArray[np.float32]]:
        total = N**3
        psi_size = n_psi_arrays * total
        zero_psi = np.zeros(psi_size, dtype=np.float32)
        chi_init = np.full(total, chi0, dtype=np.float32)
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
    ) -> NDArray[np.float32]:
        center = N / 2.0
        r_max = N / 2.0
        r_freeze = (1.0 - boundary_fraction) * r_max
        coords = np.arange(N, dtype=np.float32) - center + 0.5
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        R = np.sqrt(X**2 + Y**2 + Z**2)
        mask = (R > r_freeze).astype(np.float32).ravel()
        return mask

    def _laplacian_3d(self, flat: NDArray[np.float32], N: int) -> NDArray[np.float32]:
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
    ) -> None:
        E = psi_in
        E_prev = psi_prev_in
        chi = chi_in
        chi_prev = chi_prev_in

        lap_E = self._laplacian_3d(E, N)
        lap_chi = self._laplacian_3d(chi, N)
        chi_sq = chi * chi

        # GOV-01
        E_new = 2.0 * E - E_prev + dt2 * (lap_E - chi_sq * E)

        # GOV-02
        chi_source = kappa * (E * E - e0_sq)
        chi_accel = lap_chi - chi_source
        if lambda_self > 0:
            chi_accel -= 4.0 * lambda_self * chi * (chi_sq - chi0 * chi0)
        chi_new = 2.0 * chi - chi_prev + dt2 * chi_accel

        # BH excision
        np.clip(chi_new, -chi0, None, out=chi_new)

        # Frozen boundary
        E_new *= 1.0 - boundary_mask
        chi_new = boundary_mask * chi0 + (1.0 - boundary_mask) * chi_new

        # Write to output buffers (double-buffer swap)
        np.copyto(psi_out, E_new)
        np.copyto(psi_prev_out, E)
        np.copyto(chi_out, chi_new)
        np.copyto(chi_prev_out, chi)

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
    ) -> None:
        Pr, Pi = psi_r_in, psi_i_in
        chi, chi_prev = chi_in, chi_prev_in
        chi_sq = chi * chi

        lap_Pr = self._laplacian_3d(Pr, N)
        lap_Pi = self._laplacian_3d(Pi, N)
        lap_chi = self._laplacian_3d(chi, N)

        # GOV-01
        Pr_new = 2.0 * Pr - psi_r_prev_in + dt2 * (lap_Pr - chi_sq * Pr)
        Pi_new = 2.0 * Pi - psi_i_prev_in + dt2 * (lap_Pi - chi_sq * Pi)

        # |Ψ|² and momentum density
        psi_sq = Pr * Pr + Pi * Pi

        # j = Im(Ψ*·∇Ψ) via central differences on 3D grid
        Pr3 = Pr.reshape(N, N, N)
        Pi3 = Pi.reshape(N, N, N)
        # Face-neighbor central differences
        dPr_dx = np.roll(Pr3, -1, 0) - np.roll(Pr3, 1, 0)
        dPr_dy = np.roll(Pr3, -1, 1) - np.roll(Pr3, 1, 1)
        dPr_dz = np.roll(Pr3, -1, 2) - np.roll(Pr3, 1, 2)
        dPi_dx = np.roll(Pi3, -1, 0) - np.roll(Pi3, 1, 0)
        dPi_dy = np.roll(Pi3, -1, 1) - np.roll(Pi3, 1, 1)
        dPi_dz = np.roll(Pi3, -1, 2) - np.roll(Pi3, 1, 2)
        j_x = (Pr3 * dPi_dx - Pi3 * dPr_dx).ravel()
        j_y = (Pr3 * dPi_dy - Pi3 * dPr_dy).ravel()
        j_z = (Pr3 * dPi_dz - Pi3 * dPr_dz).ravel()
        j_total = 0.5 * (j_x + j_y + j_z)

        # GOV-02
        chi_source = kappa * (psi_sq + epsilon_w * j_total - e0_sq)
        chi_accel = lap_chi - chi_source
        if lambda_self > 0:
            chi_accel -= 4.0 * lambda_self * chi * (chi_sq - chi0 * chi0)
        chi_new = 2.0 * chi - chi_prev + dt2 * chi_accel

        np.clip(chi_new, -chi0, None, out=chi_new)

        # Frozen boundary
        Pr_new *= 1.0 - boundary_mask
        Pi_new *= 1.0 - boundary_mask
        chi_new = boundary_mask * chi0 + (1.0 - boundary_mask) * chi_new

        np.copyto(psi_r_out, Pr_new)
        np.copyto(psi_r_prev_out, Pr)
        np.copyto(psi_i_out, Pi_new)
        np.copyto(psi_i_prev_out, Pi)
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
    ) -> None:
        total = N**3
        n_colors = 3
        chi, chi_prev = chi_in, chi_prev_in
        chi_sq = chi * chi

        psi_sq_total = np.zeros(total, dtype=np.float32)
        j_total_acc = np.zeros(total, dtype=np.float32)
        color_energy = np.zeros((n_colors, total), dtype=np.float32)

        # For CCV (v15 GOV-02): store per-color per-direction currents
        need_ccv = kappa_string > 0
        if need_ccv:
            j_per_color = np.zeros((n_colors, 3, total), dtype=np.float32)

        # v15: precompute color average for cross-color coupling
        if epsilon_cc > 0:
            Pr_avg = np.zeros(total, dtype=np.float32)
            Pi_avg = np.zeros(total, dtype=np.float32)
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

            lap_Pr = self._laplacian_3d(Pr, N)
            lap_Pi = self._laplacian_3d(Pi, N)

            # GOV-01
            Pr_new = 2.0 * Pr - psi_r_prev_in[s] + dt2 * (lap_Pr - chi_sq * Pr)
            Pi_new = 2.0 * Pi - psi_i_prev_in[s] + dt2 * (lap_Pi - chi_sq * Pi)

            # v15: cross-color coupling -eps_cc * chi^2 * (Psi_a - Psi_bar)
            if epsilon_cc > 0:
                Pr_new -= dt2 * epsilon_cc * chi_sq * (Pr - Pr_avg)
                Pi_new -= dt2 * epsilon_cc * chi_sq * (Pi - Pi_avg)

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
            dPr_dx = np.roll(Pr3, -1, 0) - np.roll(Pr3, 1, 0)
            dPr_dy = np.roll(Pr3, -1, 1) - np.roll(Pr3, 1, 1)
            dPr_dz = np.roll(Pr3, -1, 2) - np.roll(Pr3, 1, 2)
            dPi_dx = np.roll(Pi3, -1, 0) - np.roll(Pi3, 1, 0)
            dPi_dy = np.roll(Pi3, -1, 1) - np.roll(Pi3, 1, 1)
            dPi_dz = np.roll(Pi3, -1, 2) - np.roll(Pi3, 1, 2)
            j_x = (Pr3 * dPi_dx - Pi3 * dPr_dx).ravel()
            j_y = (Pr3 * dPi_dy - Pi3 * dPr_dy).ravel()
            j_z = (Pr3 * dPi_dz - Pi3 * dPr_dz).ravel()
            j_total_acc += 0.5 * (j_x + j_y + j_z)

            if need_ccv:
                j_per_color[a, 0] = j_x
                j_per_color[a, 1] = j_y
                j_per_color[a, 2] = j_z

        # v14: normalized color variance f_c → color_var_term
        color_var_term = np.zeros(total, dtype=np.float32)
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
            f_c = (ratio - 1.0 / n_colors) * safe
            color_var_term = kappa_c * f_c * psi_sq_total

        # v15 GOV-02: color current variance (CCV)
        # CCV = Σ_d [ Σ_a j²_{a,d} - (1/N_c)(Σ_a j_{a,d})² ]
        ccv_term = np.zeros(total, dtype=np.float32)
        if need_ccv:
            for d in range(3):
                j_d = j_per_color[:, d, :]  # shape (n_colors, total)
                sum_j_sq = np.sum(j_d**2, axis=0)
                sum_j = np.sum(j_d, axis=0)
                ccv_term += sum_j_sq - (1.0 / n_colors) * sum_j**2

        # v16: smoothed color variance (SCV) from S_a fields
        # SCV = Σ_a S_a² - (1/N_c)(Σ_a S_a)²
        scv_term = np.zeros(total, dtype=np.float32)
        if kappa_tube > 0 and sa_fields_in is not None:
            sa_sum = np.zeros(total, dtype=np.float32)
            sa_sq_sum = np.zeros(total, dtype=np.float32)
            for a in range(n_colors):
                s_a = sa_fields_in[a * total : (a + 1) * total]
                sa_sum += s_a
                sa_sq_sum += s_a * s_a
            scv_term = sa_sq_sum - (1.0 / n_colors) * sa_sum**2

        # GOV-02
        lap_chi = self._laplacian_3d(chi, N)
        chi_source = kappa * (psi_sq_total + epsilon_w * j_total_acc - e0_sq)
        chi_accel = (
            lap_chi - chi_source - color_var_term - kappa_string * ccv_term - kappa_tube * scv_term
        )
        if lambda_self > 0:
            chi_accel -= 4.0 * lambda_self * chi * (chi_sq - chi0 * chi0)
        chi_new = 2.0 * chi - chi_prev + dt2 * chi_accel

        np.clip(chi_new, -chi0, None, out=chi_new)

        # v16: Euler update for S_a fields
        # dS_a/dt = D·∇²S_a + γ(|Ψ_a|² − S_a)   [γ-normalized source: equilibrium S_a → |Ψ_a|²]
        if kappa_tube > 0 and sa_fields_in is not None and sa_fields_out is not None:
            for a in range(n_colors):
                s_a = sa_fields_in[a * total : (a + 1) * total]
                psi_sq_a = color_energy[a]
                lap_sa = self._laplacian_3d(s_a, N)
                sa_new = s_a + dt * (sa_d * lap_sa + sa_gamma * (psi_sq_a - s_a))
                np.clip(sa_new, 0.0, None, out=sa_new)
                np.copyto(sa_fields_out[a * total : (a + 1) * total], sa_new)

        # Frozen boundary
        psi_r_out *= 1.0 - np.tile(boundary_mask, 3)
        psi_i_out *= 1.0 - np.tile(boundary_mask, 3)
        chi_new = boundary_mask * chi0 + (1.0 - boundary_mask) * chi_new

        np.copyto(chi_out, chi_new)
        np.copyto(chi_prev_out, chi)

    def to_numpy(self, arr: NDArray[np.float32]) -> NDArray[np.float32]:
        return arr

    def from_numpy(self, arr: NDArray) -> NDArray[np.float32]:
        return arr.astype(np.float32) if arr.dtype != np.float32 else arr
