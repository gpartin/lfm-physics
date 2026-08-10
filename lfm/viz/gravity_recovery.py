"""Plots for the experiment-only GOV-02 gravity recovery study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(figure: plt.Figure, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_profile_comparison(
    radius: np.ndarray,
    profile: np.ndarray,
    acceleration_radius: np.ndarray,
    acceleration: np.ndarray,
    *,
    title: str,
    path: str | Path,
) -> Path:
    """Plot radial substrate displacement and acceleration proxy."""

    radius = np.asarray(radius, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    acceleration_radius = np.asarray(acceleration_radius, dtype=np.float64)
    acceleration = np.asarray(acceleration, dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(radius, profile, color="#155e75", linewidth=2.0)
    axes[0].axhline(0.0, color="#64748b", linewidth=0.8)
    axes[0].set_xlabel("radius")
    axes[0].set_ylabel("chi - chi0")
    axes[0].set_title("Exterior substrate profile")
    axes[0].grid(alpha=0.25)
    positive = (acceleration_radius > 0.0) & (acceleration > 0.0)
    axes[1].loglog(
        acceleration_radius[positive],
        acceleration[positive],
        color="#c2410c",
        linewidth=2.0,
    )
    axes[1].set_xlabel("radius")
    axes[1].set_ylabel("|grad chi|")
    axes[1].set_title("Acceleration proxy")
    axes[1].grid(alpha=0.25, which="both")
    figure.suptitle(title)
    figure.tight_layout()
    return _save(figure, path)


def plot_energy_history(
    steps: np.ndarray,
    total_energy: np.ndarray,
    *,
    title: str,
    path: str | Path,
) -> Path:
    """Plot relative Hamiltonian drift."""

    steps = np.asarray(steps, dtype=np.float64)
    total_energy = np.asarray(total_energy, dtype=np.float64)
    scale = max(abs(float(total_energy[0])), 1.0e-30)
    drift = (total_energy - total_energy[0]) / scale
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.plot(steps, drift, color="#7c3aed", linewidth=1.8)
    axis.axhline(0.0, color="#64748b", linewidth=0.8)
    axis.set_xlabel("leapfrog step")
    axis.set_ylabel("relative Hamiltonian drift")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return _save(figure, path)


def plot_stability_sweep(
    timesteps: np.ndarray,
    maximum_amplitudes: np.ndarray,
    stable: np.ndarray,
    *,
    title: str,
    path: str | Path,
) -> Path:
    """Plot bounded and unstable points in a timestep sweep."""

    timesteps = np.asarray(timesteps, dtype=np.float64)
    maximum_amplitudes = np.asarray(maximum_amplitudes, dtype=np.float64)
    stable = np.asarray(stable, dtype=bool)
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.semilogy(
        timesteps[stable],
        maximum_amplitudes[stable],
        "o",
        color="#15803d",
        label="bounded",
    )
    axis.semilogy(
        timesteps[~stable],
        maximum_amplitudes[~stable],
        "x",
        color="#b91c1c",
        markersize=8,
        label="unstable",
    )
    axis.set_xlabel("timestep")
    axis.set_ylabel("max |chi - chi0|")
    axis.set_title(title)
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    return _save(figure, path)
