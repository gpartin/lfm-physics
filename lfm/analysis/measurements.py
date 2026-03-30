"""
Measurement Toolkit
===================

Convenience functions that extract physics from a :class:`~lfm.Simulation`
snapshot.  Every function operates on the current simulation state (no
external physics injected) and delegates to the lower-level
:mod:`lfm.analysis` modules.

For time-series measurements that need the simulation to evolve, the
function runs ``sim.run()`` internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lfm.simulation import Simulation


def _psi_imag_or_zeros(sim: Simulation) -> NDArray[np.floating]:
    """Return ``sim.psi_imag`` or a zero array of matching shape."""
    pi = sim.psi_imag
    if pi is not None:
        return pi
    return np.zeros_like(sim.psi_real)


# ── Snapshot measurements ──────────────────────────────────────────────


def measure_binding_energy(sim: Simulation) -> float:
    """Total field energy (kinetic + gradient + potential).

    To obtain *binding* energy, subtract the sum of isolated-particle
    energies (obtained by measuring each particle alone).

    Requires at least one ``sim.run()`` call so that ``psi_real_prev``
    is available for the time-derivative estimate.
    """
    from lfm.analysis.energy import total_energy

    psi_r_prev = sim.psi_real_prev
    if psi_r_prev is None:
        raise RuntimeError(
            "Need at least one sim.run() call before measuring energy "
            "(psi_real_prev is not yet set)."
        )
    return float(
        total_energy(
            sim.psi_real,
            psi_r_prev,
            sim.chi,
            dt=sim.config.dt,
        )
    )


def measure_color_fraction(
    sim: Simulation,
    point: tuple[int, int, int] | None = None,
) -> float:
    """Normalized color variance *f_c* at a point (or peak-averaged).

    Returns 0 for a balanced singlet, 2/3 for a single-color state.
    """
    from lfm.analysis.color import color_variance

    psi_i = _psi_imag_or_zeros(sim)
    result = color_variance(sim.psi_real, psi_i)
    fc: NDArray = result["f_c"]  # type: ignore[assignment]
    if point is not None:
        return float(fc[point])
    # Average over region where |Ψ|² is significant
    psi_sq = sim.psi_real**2 + psi_i**2
    if psi_sq.ndim == 4:
        psi_sq = psi_sq.sum(axis=0)
    mask = psi_sq > 0.01 * psi_sq.max()
    if mask.any():
        return float(fc[mask].mean())
    return 0.0


def measure_phase_winding(
    sim: Simulation,
    center: tuple[int, int, int],
    radius: int = 5,
    plane: str = "xy",
) -> float:
    """Phase winding number around *center* in a given plane.

    For a charge-quantized soliton the result should be near an integer.
    """
    from lfm.analysis.phase import phase_field

    theta = phase_field(sim.psi_real, _psi_imag_or_zeros(sim))
    if theta.ndim == 4:
        theta = theta[0]  # first color component
    cx, cy, cz = center
    # Sample phase around a circle
    n_samples = max(32, 4 * radius)
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    phase_vals = []
    for a in angles:
        if plane == "xy":
            xi = int(round(cx + radius * np.cos(a))) % sim.config.grid_size
            yi = int(round(cy + radius * np.sin(a))) % sim.config.grid_size
            zi = cz
        elif plane == "xz":
            xi = int(round(cx + radius * np.cos(a))) % sim.config.grid_size
            yi = cy
            zi = int(round(cz + radius * np.sin(a))) % sim.config.grid_size
        else:
            xi = cx
            yi = int(round(cy + radius * np.cos(a))) % sim.config.grid_size
            zi = int(round(cz + radius * np.sin(a))) % sim.config.grid_size
        phase_vals.append(theta[xi, yi, zi])

    # Sum of phase differences (unwrapped)
    phase_arr = np.array(phase_vals)
    diffs = np.diff(phase_arr)
    # Wrap to (-π, π)
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    winding = float(np.sum(diffs) / (2 * np.pi))
    return winding


def measure_chi_at_peak(sim: Simulation) -> float:
    """χ value at the brightest |Ψ|² peak (effective mass of the soliton)."""
    from lfm.analysis.observables import find_peaks

    psi_i = _psi_imag_or_zeros(sim)
    psi_sq = sim.psi_real**2 + psi_i**2
    if psi_sq.ndim == 4:
        psi_sq = psi_sq.sum(axis=0)
    peaks = find_peaks(psi_sq, n=1)
    if not peaks:
        return float(sim.config.chi0)
    return float(sim.chi[peaks[0]])


# ── Time-series measurements ──────────────────────────────────────────


def measure_oscillation_frequency(
    sim: Simulation,
    steps: int,
    probe: tuple[int, int, int],
    *,
    sample_interval: int = 1,
) -> tuple[float, NDArray[np.floating]]:
    """Run *sim* for *steps* and record ψ_real at *probe*. FFT → dominant ω.

    Returns ``(dominant_omega, power_spectrum)``.
    Compare *ω* to ``χ_local`` for mass verification: ``m ≈ ℏω/c²  ~ χ``.
    """
    i, j, k = probe
    time_series: list[float] = []
    for _ in range(steps // max(sample_interval, 1)):
        sim.run(steps=sample_interval, record_metrics=False)
        psi_r = sim.psi_real
        val = float(psi_r[(0, i, j, k) if psi_r.ndim == 4 else (i, j, k)])
        time_series.append(val)

    ts = np.asarray(time_series, dtype=np.float64)
    ts -= ts.mean()
    dt_eff = sim.config.dt * sample_interval
    spectrum = np.abs(np.fft.rfft(ts)) ** 2
    freqs = np.fft.rfftfreq(len(ts), d=dt_eff)

    if len(spectrum) > 1:
        idx = int(np.argmax(spectrum[1:])) + 1
        dominant_omega = float(freqs[idx]) * 2.0 * np.pi
    else:
        dominant_omega = 0.0

    return dominant_omega, spectrum


def measure_lifetime(
    sim: Simulation,
    steps: int,
    probe: tuple[int, int, int],
    *,
    threshold_fraction: float = 0.1,
    sample_interval: int = 100,
) -> int | None:
    """Run *sim* and return the step at which |Ψ|² at *probe* drops below
    *threshold_fraction* of its initial value.

    Returns ``None`` if the particle survives (stable).
    """
    i, j, k = probe
    psi_r = sim.psi_real
    psi_i = _psi_imag_or_zeros(sim)
    idx = (0, i, j, k) if psi_r.ndim == 4 else (i, j, k)
    initial = float(psi_r[idx] ** 2 + psi_i[idx] ** 2)
    if initial < 1e-30:
        return 0

    threshold = initial * threshold_fraction
    total_run = 0
    while total_run < steps:
        batch = min(sample_interval, steps - total_run)
        sim.run(steps=batch, record_metrics=False)
        total_run += batch
        psi_i = _psi_imag_or_zeros(sim)
        current = float(sim.psi_real[idx] ** 2 + psi_i[idx] ** 2)
        if current < threshold:
            return total_run

    return None  # stable


def measure_scattering_angle(
    sim: Simulation,
    steps: int,
    *,
    sample_interval: int = 100,
) -> dict[str, float]:
    """Track two brightest peaks over *steps* and return asymptotic
    deflection angle and impact parameter.
    """
    from lfm.analysis.observables import find_peaks

    positions: list[list[tuple[int, int, int]]] = []
    for _ in range(steps // max(sample_interval, 1)):
        sim.run(steps=sample_interval, record_metrics=False)
        psi_i = _psi_imag_or_zeros(sim)
        psi_sq = sim.psi_real**2 + psi_i**2
        if psi_sq.ndim == 4:
            psi_sq = psi_sq.sum(axis=0)
        peaks = find_peaks(psi_sq, n=2, min_separation=3)
        positions.append(peaks)

    if len(positions) < 2 or len(positions[0]) < 2 or len(positions[-1]) < 2:
        return {"angle_deg": 0.0, "impact_param": 0.0}

    # Initial and final separation vectors
    p0 = np.array(positions[0][0], dtype=float) - np.array(positions[0][1], dtype=float)
    pf = np.array(positions[-1][0], dtype=float) - np.array(positions[-1][1], dtype=float)
    cos_angle = float(np.dot(p0, pf) / (np.linalg.norm(p0) * np.linalg.norm(pf) + 1e-30))
    angle = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
    impact = float(np.linalg.norm(np.cross(p0, pf)) / (np.linalg.norm(pf) + 1e-30))
    return {"angle_deg": angle, "impact_param": impact}
