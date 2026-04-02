"""Ringdown analysis helpers.

Utilities in this module are extraction tools only. They do not encode any
external force law or observational target values.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def find_local_peaks(y: np.ndarray) -> np.ndarray:
    """Return indices of strict local maxima in a 1D signal."""
    if y.ndim != 1 or len(y) < 3:
        return np.array([], dtype=int)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1


def fit_ringdown_series(
    t: np.ndarray,
    signal: np.ndarray,
    *,
    start_frac: float = 0.2,
    min_peaks: int = 8,
) -> dict[str, float | int | bool]:
    """Fit dominant frequency and exponential envelope damping for 1D data.

    Returns dict with keys: ``omega``, ``gamma``, ``n_peaks``, ``valid``.
    """
    y = np.asarray(signal, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64)
    if y.ndim != 1 or tt.ndim != 1 or len(y) != len(tt) or len(y) < 64:
        return {"omega": float("nan"), "gamma": float("nan"), "n_peaks": 0, "valid": False}

    y = y - np.mean(y)
    i0 = int(max(0, min(len(y) - 1, round(start_frac * len(y)))))
    y2 = y[i0:]
    t2 = tt[i0:]
    if len(y2) < 64:
        return {"omega": float("nan"), "gamma": float("nan"), "n_peaks": 0, "valid": False}

    dt = float(t2[1] - t2[0])
    spec = np.fft.rfft(y2)
    f = np.fft.rfftfreq(len(y2), d=dt)
    mag = np.abs(spec)
    if len(mag) == 0:
        return {"omega": float("nan"), "gamma": float("nan"), "n_peaks": 0, "valid": False}
    mag[0] = 0.0
    omega = float(2.0 * np.pi * f[int(np.argmax(mag))])

    idx = find_local_peaks(np.abs(y2))
    if len(idx) < min_peaks:
        return {"omega": omega, "gamma": float("nan"), "n_peaks": int(len(idx)), "valid": False}

    pk_t = t2[idx]
    pk_a = np.abs(y2[idx])
    ok = pk_a > 1e-12
    pk_t = pk_t[ok]
    pk_a = pk_a[ok]
    if len(pk_t) < min_peaks:
        return {"omega": omega, "gamma": float("nan"), "n_peaks": int(len(idx)), "valid": False}

    A = np.column_stack([pk_t, np.ones_like(pk_t)])
    coeff, *_ = np.linalg.lstsq(A, np.log(pk_a), rcond=None)
    gamma = max(0.0, -float(coeff[0]))
    return {"omega": omega, "gamma": gamma, "n_peaks": int(len(idx)), "valid": True}


def relative_spread(values: list[float] | np.ndarray) -> float:
    """Return fractional spread ``(max-min)/mean_abs`` for finite values."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    mean_abs = float(np.mean(np.abs(x)))
    denom: float = max(mean_abs, 1e-12)
    return float((np.max(x) - np.min(x)) / denom)


def split_frequency_bands(
    omegas: list[float] | np.ndarray,
    *,
    min_target: float = 1.0,
) -> dict[str, list[float]]:
    """Split frequencies into target and slow bands.

    ``target`` contains values >= ``min_target``; ``slow`` contains values below.
    """
    x = np.asarray(omegas, dtype=np.float64)
    x = x[np.isfinite(x)]
    target = [float(v) for v in x if v >= min_target]
    slow = [float(v) for v in x if v < min_target]
    return {"target": target, "slow": slow}


def project_field_onto_modes(
    field: np.ndarray,
    modes: list[tuple[int, int, int]],
    *,
    subtract_mean: bool = True,
    center_shift: tuple[int, int, int] | None = None,
) -> dict[str, complex]:
    """Project a 3D field onto periodic Fourier basis modes.

    Parameters
    ----------
    field
        Real 3D field array.
    modes
        Lattice wave index tuples ``(kx, ky, kz)``.
    subtract_mean
        Remove k=0 monopole before projection.
    center_shift
        Optional integer roll ``(sx, sy, sz)`` applied before projection.
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("field must be a 3D array")

    if center_shift is not None:
        arr = np.roll(arr, shift=tuple(int(v) for v in center_shift), axis=(0, 1, 2))

    if subtract_mean:
        arr = arr - np.mean(arr)

    nx, ny, nz = arr.shape
    if not (nx == ny == nz):
        raise ValueError("field must be cubic (N,N,N)")
    n = nx

    ii, jj, kk = np.indices(arr.shape, dtype=np.float64)
    out: dict[str, complex] = {}
    for kx, ky, kz in modes:
        phase = -2.0 * np.pi * (kx * ii + ky * jj + kz * kk) / n
        phi = np.exp(1j * phase)
        coeff = np.sum(arr * phi) / arr.size
        out[f"({kx},{ky},{kz})"] = complex(coeff)
    return out


def target_band_summary(
    mode_fits: list[dict[str, Any]],
    *,
    min_target: float = 1.0,
) -> dict[str, Any]:
    """Summarize valid mode fits by target/slow frequency bands."""
    omegas = [float(r["omega"]) for r in mode_fits if bool(r.get("valid", False))]
    bands = split_frequency_bands(omegas, min_target=min_target)
    target = bands["target"]
    return {
        "valid_mode_count": len(omegas),
        "target_band": target,
        "slow_band": bands["slow"],
        "target_center": float(np.mean(target)) if target else float("nan"),
        "target_spread": relative_spread(target),
    }
