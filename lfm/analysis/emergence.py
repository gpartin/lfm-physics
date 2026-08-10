"""Basis-independent diagnostics for localized-state emergence.

These functions inspect the native LFM wave register without introducing a
spinor, gauge link, particle catalog entry, or preferred color component.
They are intended for discovery experiments in which the object being
measured is not known in advance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lfm.analysis.phase import charge_density

if TYPE_CHECKING:
    from numpy.typing import NDArray


def aggregate_wave_density(
    psi_r: NDArray,
    psi_i: NDArray | None = None,
) -> NDArray:
    """Return the internal-basis-invariant density sum_a |Psi_a|^2."""
    pr = np.asarray(psi_r, dtype=np.float64)
    pi = np.zeros_like(pr) if psi_i is None else np.asarray(psi_i, dtype=np.float64)
    density = pr * pr + pi * pi
    if density.ndim == 4:
        density = np.sum(density, axis=0)
    if density.ndim != 3:
        raise ValueError(f"wave field must be 3-D or component-first 4-D, got {pr.shape}")
    return density


def principal_internal_projection(
    psi_r: NDArray,
    psi_i: NDArray,
) -> tuple[NDArray, float]:
    """Project a multicomponent field onto its leading covariance direction.

    The leading eigendirection transforms covariantly under a global unitary
    change of internal basis. The returned eigengap fraction reports whether
    that direction is well separated; a value near zero means the projection
    is not unique and phase-topology measurements should be treated cautiously.
    """
    psi = np.asarray(psi_r, dtype=np.float64) + 1j * np.asarray(psi_i, dtype=np.float64)
    if psi.ndim == 3:
        return psi.astype(np.complex128, copy=False), 1.0
    if psi.ndim != 4:
        raise ValueError(f"wave field must be 3-D or component-first 4-D, got {psi.shape}")

    flat = psi.reshape(psi.shape[0], -1)
    covariance = flat @ flat.conj().T
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values.real)
    lead = vectors[:, order[-1]]
    projection = np.einsum("a,axyz->xyz", lead.conj(), psi)
    largest = float(max(values[order[-1]].real, 0.0))
    second = float(max(values[order[-2]].real, 0.0)) if len(order) > 1 else 0.0
    gap = (largest - second) / largest if largest > 0.0 else 0.0
    return projection, float(gap)


def plaquette_winding_summary(
    field: NDArray,
    amplitude_fraction: float = 1.0e-3,
) -> dict[str, int]:
    """Count resolved integer branch windings on elementary plaquettes.

    A plaquette is included only when all four corner amplitudes exceed the
    supplied fraction of the peak amplitude. This excludes arbitrary phase at
    numerical zeros. The result is a diagnostic of the projected complex
    field, not a compact-U(1) gauge curvature.
    """
    if not 0.0 <= amplitude_fraction < 1.0:
        raise ValueError("amplitude_fraction must lie in [0, 1)")
    psi = np.asarray(field, dtype=np.complex128)
    if psi.ndim != 3:
        raise ValueError(f"projected field must be 3-D, got {psi.shape}")
    amplitude = np.abs(psi)
    peak = float(np.max(amplitude))
    if peak == 0.0:
        return {"positive": 0, "negative": 0, "nonzero": 0, "valid": 0}

    theta = np.angle(psi)

    def wrap(delta: NDArray) -> NDArray:
        return (delta + np.pi) % (2.0 * np.pi) - np.pi

    positive = 0
    negative = 0
    valid_total = 0
    threshold = amplitude_fraction * peak
    for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
        theta_a = np.roll(theta, -1, axis=axis_a)
        theta_b = np.roll(theta, -1, axis=axis_b)
        theta_ab = np.roll(theta_a, -1, axis=axis_b)
        amp_a = np.roll(amplitude, -1, axis=axis_a)
        amp_b = np.roll(amplitude, -1, axis=axis_b)
        amp_ab = np.roll(amp_a, -1, axis=axis_b)
        valid = (
            (amplitude > threshold)
            & (amp_a > threshold)
            & (amp_b > threshold)
            & (amp_ab > threshold)
        )
        circulation = (
            wrap(theta_a - theta)
            + wrap(theta_ab - theta_a)
            + wrap(theta_b - theta_ab)
            + wrap(theta - theta_b)
        )
        winding = np.rint(circulation / (2.0 * np.pi)).astype(np.int8)
        positive += int(np.count_nonzero((winding > 0) & valid))
        negative += int(np.count_nonzero((winding < 0) & valid))
        valid_total += int(np.count_nonzero(valid))
    return {
        "positive": positive,
        "negative": negative,
        "nonzero": positive + negative,
        "valid": valid_total,
    }


def localized_state_observables(
    psi_r: NDArray,
    psi_i: NDArray | None,
    psi_r_prev: NDArray,
    psi_i_prev: NDArray | None,
    chi: NDArray,
    dt: float,
    chi0: float,
) -> dict[str, float | int]:
    """Measure localization, Noether charge, topology, and medium response."""
    raw_arrays = [np.asarray(psi_r), np.asarray(psi_r_prev), np.asarray(chi)]
    if psi_i is not None:
        raw_arrays.append(np.asarray(psi_i))
    if psi_i_prev is not None:
        raw_arrays.append(np.asarray(psi_i_prev))
    if not all(np.all(np.isfinite(arr)) for arr in raw_arrays):
        nan = float("nan")
        return {
            "wave_norm": nan,
            "effective_sites": nan,
            "effective_volume_fraction": nan,
            "rms_radius": nan,
            "radius3_fraction": nan,
            "radius5_fraction": nan,
            "c7_density_fraction": nan,
            "c19_density_fraction": nan,
            "top7_c7_overlap": 0,
            "top19_c19_overlap": 0,
            "peak_density": nan,
            "peak_x": 0,
            "peak_y": 0,
            "peak_z": 0,
            "boundary_fraction": nan,
            "noether_charge": nan,
            "charge_per_norm": nan,
            "chi_min": nan,
            "chi_at_peak": nan,
            "chi_drop": nan,
            "chi_density_peak_distance": nan,
            "internal_principal_gap": nan,
            "winding_positive": 0,
            "winding_negative": 0,
            "winding_nonzero": 0,
            "winding_valid_plaquettes": 0,
            "support_sites_10pct": 0,
            "support_sites_50pct": 0,
        }
    density = aggregate_wave_density(psi_r, psi_i)
    norm = float(np.sum(density))
    shape = density.shape
    total_sites = int(density.size)
    if norm <= 0.0:
        return {
            "wave_norm": 0.0,
            "effective_sites": 0.0,
            "effective_volume_fraction": 0.0,
            "rms_radius": 0.0,
            "radius3_fraction": 0.0,
            "radius5_fraction": 0.0,
            "c7_density_fraction": 0.0,
            "c19_density_fraction": 0.0,
            "top7_c7_overlap": 0,
            "top19_c19_overlap": 0,
            "peak_density": 0.0,
            "peak_x": 0,
            "peak_y": 0,
            "peak_z": 0,
            "boundary_fraction": 0.0,
            "noether_charge": 0.0,
            "charge_per_norm": 0.0,
            "chi_min": float(np.min(chi)),
            "chi_at_peak": float(chi.flat[0]),
            "chi_drop": float(chi0 - np.min(chi)),
            "chi_density_peak_distance": 0.0,
            "internal_principal_gap": 0.0,
            "winding_positive": 0,
            "winding_negative": 0,
            "winding_nonzero": 0,
            "winding_valid_plaquettes": 0,
            "support_sites_10pct": 0,
            "support_sites_50pct": 0,
        }

    probability = density / norm
    effective_sites = 1.0 / float(np.sum(probability * probability))
    coords = np.indices(shape, dtype=np.float64)
    center = np.array([float(np.sum(coords[a] * probability)) for a in range(3)])
    radius_sq = sum((coords[a] - center[a]) ** 2 for a in range(3))
    rms_radius = float(np.sqrt(np.sum(radius_sq * probability)))
    peak = tuple(int(v) for v in np.unravel_index(int(np.argmax(density)), shape))
    peak_radius_sq = sum((coords[a] - peak[a]) ** 2 for a in range(3))

    c7_offsets = (
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    c19_offsets = c7_offsets + tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if abs(dx) + abs(dy) + abs(dz) == 2
    )

    def support_indices(offsets: tuple[tuple[int, int, int], ...]) -> set[tuple[int, int, int]]:
        return {
            ((peak[0] + dx) % shape[0], (peak[1] + dy) % shape[1], (peak[2] + dz) % shape[2])
            for dx, dy, dz in offsets
        }

    c7_indices = support_indices(c7_offsets)
    c19_indices = support_indices(c19_offsets)
    flat_order = np.argsort(density.ravel())[::-1]
    top7 = {tuple(int(v) for v in np.unravel_index(int(i), shape)) for i in flat_order[:7]}
    top19 = {tuple(int(v) for v in np.unravel_index(int(i), shape)) for i in flat_order[:19]}
    c7_fraction = float(sum(density[p] for p in c7_indices) / norm)
    c19_fraction = float(sum(density[p] for p in c19_indices) / norm)

    edge_distance = np.minimum.reduce(
        [coords[0], coords[1], coords[2], shape[0] - 1 - coords[0], shape[1] - 1 - coords[1], shape[2] - 1 - coords[2]]
    )
    boundary_fraction = float(np.sum(probability[edge_distance < 2.0]))

    pr = np.asarray(psi_r)
    pr_prev = np.asarray(psi_r_prev)
    if psi_i is None or psi_i_prev is None:
        noether = 0.0
        projected = pr[0] if pr.ndim == 4 else pr
        principal_gap = 0.0
        winding = {"positive": 0, "negative": 0, "nonzero": 0, "valid": 0}
    else:
        pi = np.asarray(psi_i)
        pi_prev = np.asarray(psi_i_prev)
        rho = charge_density(pr, pi, dt=dt, psi_r_prev=pr_prev, psi_i_prev=pi_prev)
        noether = float(np.sum(rho))
        projected, principal_gap = principal_internal_projection(pr, pi)
        winding = plaquette_winding_summary(projected)

    chi_arr = np.asarray(chi, dtype=np.float64)
    chi_min_index = tuple(int(v) for v in np.unravel_index(int(np.argmin(chi_arr)), shape))
    alignment = float(np.linalg.norm(np.asarray(peak, dtype=np.float64) - np.asarray(chi_min_index, dtype=np.float64)))
    peak_density = float(density[peak])
    return {
        "wave_norm": norm,
        "effective_sites": effective_sites,
        "effective_volume_fraction": effective_sites / total_sites,
        "rms_radius": rms_radius,
        "radius3_fraction": float(np.sum(probability[peak_radius_sq <= 9.0])),
        "radius5_fraction": float(np.sum(probability[peak_radius_sq <= 25.0])),
        "c7_density_fraction": c7_fraction,
        "c19_density_fraction": c19_fraction,
        "top7_c7_overlap": len(top7 & c7_indices),
        "top19_c19_overlap": len(top19 & c19_indices),
        "peak_density": peak_density,
        "peak_x": peak[0],
        "peak_y": peak[1],
        "peak_z": peak[2],
        "boundary_fraction": boundary_fraction,
        "noether_charge": noether,
        "charge_per_norm": noether / norm,
        "chi_min": float(np.min(chi_arr)),
        "chi_at_peak": float(chi_arr[peak]),
        "chi_drop": float(chi0 - np.min(chi_arr)),
        "chi_density_peak_distance": alignment,
        "internal_principal_gap": principal_gap,
        "winding_positive": winding["positive"],
        "winding_negative": winding["negative"],
        "winding_nonzero": winding["nonzero"],
        "winding_valid_plaquettes": winding["valid"],
        "support_sites_10pct": int(np.count_nonzero(density >= 0.1 * peak_density)),
        "support_sites_50pct": int(np.count_nonzero(density >= 0.5 * peak_density)),
    }
