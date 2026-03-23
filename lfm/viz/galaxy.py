"""
Galaxy summary plot — LFM rotation-curve vs SPARC overlay.
===========================================================

Provides a two-panel figure comparing the simulated chi-field radial profile
and circular-velocity curve with observed SPARC data.

Requires ``matplotlib``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lfm.viz._util import _require_matplotlib

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


def galaxy_summary_plot(
    sim: Any,
    sparc_row: dict,
    tau: float | None = None,
    ax: Any | None = None,
) -> tuple["matplotlib.figure.Figure", list["matplotlib.axes.Axes"]]:
    """Two-panel figure: chi-profile slice and rotation-curve comparison.

    The left panel shows the radially-averaged chi profile from the
    simulation.  The right panel overlays the LFM circular-velocity curve
    with the SPARC observed data.

    Parameters
    ----------
    sim : Simulation
        A run (or equilibrated) LFM simulation.  Must have ``.chi`` and
        ``.energy_density`` attributes.
    sparc_row : dict
        A single-galaxy dict as returned by :func:`lfm.sparc_load`, with
        keys ``name``, ``r_kpc``, ``v_obs_kms``,  and optionally
        ``v_err_kms``.
    tau : float or None
        If given, pass through to :func:`lfm.rotation_curve_fit` as-is
        (skips the grid search).  Otherwise the best-fit tau is found
        automatically.
    ax : ignored
        Reserved for future subplot injection; currently unused.

    Returns
    -------
    (Figure, axes_list)
        ``axes_list`` has two :class:`matplotlib.axes.Axes`:
        ``axes_list[0]`` = chi radial profile,
        ``axes_list[1]`` = rotation curve comparison.

    Examples
    --------
    >>> row = lfm.sparc_load("NGC6503")
    >>> fig, axes = lfm.viz.galaxy_summary_plot(sim, row)
    >>> fig.savefig("galaxy.png", dpi=150)
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    from lfm.analysis.observables import radial_profile, rotation_curve

    # ------------------------------------------------------------------
    # 1. Radially-averaged chi profile
    # ------------------------------------------------------------------
    N = sim.chi.shape[0]
    center = (N / 2, N / 2, N / 2)
    chi_prof = radial_profile(sim.chi, center=center)
    r_cells = np.arange(len(chi_prof["mean"]))

    # ------------------------------------------------------------------
    # 2. LFM rotation curve
    # ------------------------------------------------------------------
    rc = rotation_curve(sim.energy_density, sim.chi)
    r_lfm = np.asarray(rc["r"], dtype=np.float64)
    v_lfm = np.asarray(rc["v_circ"], dtype=np.float64)

    # ------------------------------------------------------------------
    # 3. Normalise to SPARC scale (simple peak-velocity normalisation)
    # ------------------------------------------------------------------
    obs_r = np.asarray(sparc_row["r_kpc"], dtype=np.float64)
    obs_v = np.asarray(sparc_row["v_obs_kms"], dtype=np.float64)
    obs_err = np.asarray(sparc_row.get("v_err_kms", np.full_like(obs_v, 5.0)), dtype=np.float64)

    v_obs_peak = float(np.max(obs_v)) if len(obs_v) > 0 else 1.0
    v_lfm_peak = (
        float(np.max(np.abs(v_lfm))) if len(v_lfm) > 0 and np.any(np.isfinite(v_lfm)) else 1.0
    )
    if v_lfm_peak < 1e-30:
        v_lfm_peak = 1.0

    # Scale LFM radii so max matches SPARC max
    r_lfm_scaled = (
        r_lfm * (float(np.max(obs_r)) / float(np.max(r_lfm))) if float(np.max(r_lfm)) > 0 else r_lfm
    )
    # Scale LFM velocities to km/s
    v_lfm_kms = v_lfm * (v_obs_peak / v_lfm_peak)

    # ------------------------------------------------------------------
    # 4. Build figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel 1: chi radial profile ---
    ax0 = axes[0]
    ax0.plot(
        r_cells, chi_prof["mean"], color="#1f77b4", linewidth=2, label=r"$\langle\chi\rangle(r)$"
    )
    if "std" in chi_prof:
        std = np.asarray(chi_prof["std"])
        mean = np.asarray(chi_prof["mean"])
        ax0.fill_between(r_cells, mean - std, mean + std, alpha=0.25, color="#1f77b4")
    ax0.axhline(
        getattr(sim.config, "chi0", 19.0),
        color="grey",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=r"$\chi_0 = 19$",
    )
    ax0.set_xlabel("Radius (lattice cells)")
    ax0.set_ylabel(r"$\chi$")
    ax0.set_title(r"$\chi$ radial profile")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    # --- Panel 2: rotation curve ---
    ax1 = axes[1]
    galaxy_name = sparc_row.get("name", "SPARC")

    ax1.errorbar(
        obs_r,
        obs_v,
        yerr=obs_err,
        fmt="o",
        markersize=4,
        color="#d62728",
        elinewidth=1,
        capsize=3,
        label=f"{galaxy_name} (obs)",
        zorder=5,
    )
    ax1.plot(
        r_lfm_scaled,
        v_lfm_kms,
        color="#2ca02c",
        linewidth=2,
        label="LFM (normalised)",
        zorder=4,
    )
    ax1.set_xlabel("Radius (kpc)")
    ax1.set_ylabel(r"$v_\mathrm{circ}$ (km s$^{-1}$)")
    ax1.set_title(f"Rotation curve — {galaxy_name}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    fig.suptitle(
        f"LFM galaxy simulation vs {galaxy_name}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig, list(axes)
