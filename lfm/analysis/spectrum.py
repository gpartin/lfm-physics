"""Fourier power-spectrum analysis of lattice fields."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def power_spectrum(
    field: NDArray,
    bins: int = 50,
) -> dict[str, NDArray]:
    """Compute the radially-averaged power spectrum of a 3-D field.

    Parameters
    ----------
    field : ndarray (N, N, N)
        Scalar field (e.g. ``sim.chi`` or ``sim.energy_density``).
    bins : int
        Number of radial k-bins.

    Returns
    -------
    dict
        ``k`` — bin centres, ``power`` — P(k) in each bin,
        ``counts`` — number of modes per bin.
    """
    if field.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {field.shape}")

    N = field.shape[0]
    fft = np.fft.fftn(field)
    pk = np.abs(fft) ** 2 / field.size

    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    kz = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    k_max = N // 2
    bin_edges = np.linspace(0.5, k_max + 0.5, bins + 1)
    k_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    power = np.zeros(bins)
    counts = np.zeros(bins, dtype=int)

    k_flat = K.ravel()
    pk_flat = pk.ravel()
    idx = np.digitize(k_flat, bin_edges) - 1

    for i in range(bins):
        mask = idx == i
        counts[i] = mask.sum()
        if counts[i] > 0:
            power[i] = pk_flat[mask].mean()

    return {"k": k_centres, "power": power, "counts": counts}
