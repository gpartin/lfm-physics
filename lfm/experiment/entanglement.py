"""
Spin Entanglement Experiment
============================

Two spin-1/2 particles (spinor solitons) initialized in correlated spin
states on a 3-D lattice.  Physics from GOV-01 + GOV-02 alone — no
quantum postulates injected.

Demonstrates
------------
* **Spin-blind gravity** — both χ wells have the same depth regardless of
  the spin configuration, because |Ψ|² = |ψ_↑|² + |ψ_↓|² is spin-blind.
* **Preserved spin expectation values** — ⟨σ_z⟩ and ⟨σ_x⟩ remain stable
  over many oscillation periods once equilibrated.
* **CHSH Bell parameter** — product states give S ≤ 2 (classical bound);
  maximally correlated states approach S = 2 (local-deterministic limit).
* **Spinor density evolution** — the 3-D movie shows two bright χ wells
  (spinor solitons) whose chifield structure is spin-configuration–
  independent.

Quickstart
----------
>>> from lfm.experiment import entanglement
>>> r = entanglement(N=64)                          # antiparallel (default)
>>> r = entanglement(config="triplet", N=64)
>>> r = entanglement(config="all", N=48)            # runs all 4 configs
>>> r.save("spin_entanglement", directory="outputs/spin_entanglement/")

Spin configurations
-------------------
``"triplet"``      — |↑↑⟩            both spin-up (parallel)
``"antiparallel"`` — |↑↓⟩            particle A up, B down
``"product_x"``    — |+x⟩|+x⟩        both spin-right
``"singlet"``      — |+x⟩|−x⟩        anticorrelated in x-basis
``"all"``          — run all four; ``entanglement()`` returns
                     ``EntanglementSuiteResult``

Reference: LFM-PAPER-048 (Spinor Representation in the Lattice Field
Medium).  See also Paper 085 (IT-05) for χ-mediated S > 2 correlations
when both solitons share one deep χ well.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lfm.analysis.spinor import (
    spinor_density,
    spinor_sigma_x,
    spinor_sigma_y,
    spinor_sigma_z,
)
from lfm.constants import CHI0, DT_DEFAULT, KAPPA
from lfm.experiment.common import ExperimentResult, gpu_snapshot_loop, midplane_slice
from lfm.fields.equilibrium import equilibrate_chi

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    import lfm as _lfm_t

__all__ = [
    "entanglement",
    "EntanglementResult",
    "EntanglementSuiteResult",
    "SPIN_CONFIGS",
]

# ── Available configurations ───────────────────────────────────────────────

SPIN_CONFIGS: tuple[str, ...] = ("triplet", "antiparallel", "product_x", "singlet")

# ── Defaults ───────────────────────────────────────────────────────────────

_CHI0: float = 19.0
_AMPLITUDE: float = 3.0   # shallow chi-wells → matches collision.py default
_SIGMA: float = 0.0        # 0 = auto (max(3.0, N*0.04)); explicit value overrides
_DT: float = DT_DEFAULT
_TOTAL_STEPS: int = 4000
_SNAP_EVERY: int = 100
_MOVIE_EVERY: int = 40
_METRICS_EVERY: int = 20


# ── Spinor state builders ─────────────────────────────────────────────────


def _build_eigenmode_spinors(
    N: int,
    config: str,
    E_A: np.ndarray,
    E_B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed relaxed scalar eigenmodes into a two-component spinor state.

    Parameters
    ----------
    E_A, E_B : ndarray shape (N, N, N)
        Relaxed real eigenmode profiles centered at particle A and B
        positions respectively.

    Returns
    -------
    psi_r, psi_i : ndarray shape (3, N, N, N)
        Three-color spinor arrays: colors 0 and 1 carry the spin-up/down
        amplitudes; color 2 is always zero (padding for the 3-color kernel).
        Imaginary part is zero (handled via leapfrog prev-buffers for
        constant-density init).
    """
    s2 = math.sqrt(2.0)
    psi_r = np.zeros((3, N, N, N), dtype=np.float32)
    psi_i = np.zeros((3, N, N, N), dtype=np.float32)

    if config == "triplet":
        # |↑↑⟩ — both particles spin-up (color 0)
        psi_r[0] = E_A + E_B
    elif config == "antiparallel":
        # |↑↓⟩ — A in color 0, B in color 1
        psi_r[0] = E_A
        psi_r[1] = E_B
    elif config == "product_x":
        # |+x,A⟩|+x,B⟩ = each particle in (|↑⟩+|↓⟩)/√2
        psi_r[0] = (E_A + E_B) / s2
        psi_r[1] = (E_A + E_B) / s2
    elif config == "singlet":
        # |+x,A⟩|−x,B⟩ — anticorrelated x-basis (product-state approx)
        psi_r[0] = (E_A + E_B) / s2
        psi_r[1] = (E_A - E_B) / s2
    else:
        raise ValueError(
            f"Unknown spin config {config!r}.  Choose from {SPIN_CONFIGS}."
        )
    return psi_r, psi_i


# ── Spin measurement helpers ───────────────────────────────────────────────


def _local_spin(
    psi_r: np.ndarray,
    psi_i: np.ndarray,
    x_center: int,
    N: int,
    half_width: int = 8,
) -> tuple[float, float, float]:
    """Return (⟨σ_z⟩, ⟨σ_x⟩, ⟨σ_y⟩) for a slab around *x_center*."""
    x0 = max(0, x_center - half_width)
    x1 = min(N, x_center + half_width)
    sr = psi_r[:, x0:x1, :, :]
    si = psi_i[:, x0:x1, :, :]
    return (
        spinor_sigma_z(sr, si),
        spinor_sigma_x(sr, si),
        spinor_sigma_y(sr, si),
    )


def _corr(
    psi_r: np.ndarray,
    psi_i: np.ndarray,
    theta_A: float,
    theta_B: float,
    xA: int,
    xB: int,
    N: int,
) -> float:
    """Joint correlation C(θ_A, θ_B) = ⟨σ_{θ_A}^A⟩ · ⟨σ_{θ_B}^B⟩."""
    szA, sxA, _ = _local_spin(psi_r, psi_i, xA, N)
    szB, sxB, _ = _local_spin(psi_r, psi_i, xB, N)
    sA = math.cos(theta_A) * szA + math.sin(theta_A) * sxA
    sB = math.cos(theta_B) * szB + math.sin(theta_B) * sxB
    return sA * sB


def _chsh(
    psi_r: np.ndarray,
    psi_i: np.ndarray,
    xA: int,
    xB: int,
    N: int,
) -> float:
    """CHSH parameter S = |E(a,b) − E(a,b') + E(a',b) + E(a',b')|."""
    a, b = 0.0, math.pi / 4
    a2, b2 = math.pi / 2, 3 * math.pi / 4
    Eab = _corr(psi_r, psi_i, a, b, xA, xB, N)
    Eab2 = _corr(psi_r, psi_i, a, b2, xA, xB, N)
    Ea2b = _corr(psi_r, psi_i, a2, b, xA, xB, N)
    Ea2b2 = _corr(psi_r, psi_i, a2, b2, xA, xB, N)
    return abs(Eab - Eab2 + Ea2b + Ea2b2)


def _to_numpy(arr: np.ndarray) -> np.ndarray:
    """Move CuPy array to NumPy if needed."""
    try:
        import cupy
        if isinstance(arr, cupy.ndarray):
            return cupy.asnumpy(arr)
    except ImportError:
        pass
    return np.asarray(arr)


# ── Result class ───────────────────────────────────────────────────────────


class EntanglementResult(ExperimentResult):
    """Result of a single :func:`entanglement` experiment run.

    Extends :class:`~lfm.experiment.common.ExperimentResult` with
    spin-measurement time-series and CHSH analysis.

    Attributes
    ----------
    config_name : str
        Spin configuration name (``"triplet"``, ``"antiparallel"``, …).
    xA, xB : int
        Grid x-coordinate of particle A and B wells.
    spin_history : list[dict]
        Per-snapshot spin measurements with keys
        ``step``, ``sz_A``, ``sx_A``, ``sy_A``, ``sz_B``, ``sx_B``, ``sy_B``.
    chsh_initial, chsh_final : float
        CHSH parameter S at beginning and end of evolution.
    chi_min_initial, chi_min_final : float
        χ well depth at beginning and end.
    """

    def __init__(
        self,
        *,
        snapshots: list[dict],
        movie_snapshots: list[dict],
        metrics: list[dict],
        spin_history: list[dict],
        label: str,
        N: int,
        config_name: str,
        xA: int,
        xB: int,
        chsh_initial: float,
        chsh_final: float,
        chi_min_initial: float,
    ) -> None:
        super().__init__(
            snapshots=snapshots,
            movie_snapshots=movie_snapshots,
            metrics=metrics,
            label=label,
            N=N,
        )
        self.config_name = config_name
        self.xA = xA
        self.xB = xB
        self.spin_history = spin_history
        self.chsh_initial = chsh_initial
        self.chsh_final = chsh_final
        self.chi_min_initial = chi_min_initial

    @property
    def chi_min_history(self) -> list[float]:
        """χ minimum at each metrics step."""
        return [m["chi_min"] for m in self.metrics]

    @property
    def energy_history(self) -> list[float]:
        """Total energy at each metrics step."""
        return [m["energy_total"] for m in self.metrics]

    @property
    def chsh_final(self) -> float:
        """CHSH parameter computed from the last snapshot."""
        return self._chsh_final

    @chsh_final.setter
    def chsh_final(self, v: float) -> None:
        self._chsh_final = v

    def plot(
        self,
        *,
        figsize: tuple[float, float] = (16, 10),
        title: str | None = None,
    ) -> "Figure":
        """Summary figure: spinor density slices, spin time-series, χ well."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=figsize, facecolor="white")
        fig.suptitle(
            title or self.label,
            fontsize=14,
            fontweight="bold",
        )
        N = self.N

        # ── Pick representative snapshots ──────────────────────────
        n = len(self.snapshots)
        if n >= 4:
            indices = [0, n // 3, 2 * n // 3, n - 1]
        elif n >= 2:
            indices = [0, n - 1]
        else:
            indices = list(range(n))

        n_cols = max(len(indices), 3)
        gs = GridSpec(2, n_cols, figure=fig, hspace=0.40, wspace=0.35,
                      top=0.91, bottom=0.08, left=0.06, right=0.97)

        # ── Row 1: spinor density slices (x–z mid-plane) ───────────
        for i, idx in enumerate(indices):
            ax = fig.add_subplot(gs[0, i])
            snap = self.snapshots[idx]
            psi_r = snap.get("psi_real")
            psi_i = snap.get("psi_imag")
            if psi_r is not None:
                pr = _to_numpy(psi_r)
                pi = _to_numpy(psi_i) if psi_i is not None else np.zeros_like(pr)
                # Sum over spinor components → total density
                dens = pr[0] ** 2 + pi[0] ** 2 + pr[1] ** 2 + pi[1] ** 2
                plane = dens[:, N // 2, :]  # x–z slice
                ax.imshow(plane.T, origin="lower", cmap="hot", aspect="equal")
            step = snap.get("step", idx)
            ax.set_title(f"step {step}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_ylabel("|ψ|² (x–z slice)")

        # ── Row 2: three diagnostic panels ─────────────────────────
        span = n_cols / 3

        def _bot(k):
            c0, c1 = round(k * span), round((k + 1) * span)
            return fig.add_subplot(gs[1, c0:c1])

        # Panel 0 — spin z over time
        if self.spin_history:
            steps_s = [h["step"] for h in self.spin_history]
            szA = [h["sz_A"] for h in self.spin_history]
            szB = [h["sz_B"] for h in self.spin_history]
            ax_sz = _bot(0)
            ax_sz.plot(steps_s, szA, color="#2166ac", lw=1.5, label="⟨σ_z^A⟩")
            ax_sz.plot(steps_s, szB, color="#d6604d", lw=1.5, label="⟨σ_z^B⟩", ls="--")
            ax_sz.set_ylim(-1.2, 1.2)
            ax_sz.axhline(0, color="grey", lw=0.5)
            ax_sz.set_xlabel("Step"); ax_sz.set_ylabel("⟨σ_z⟩")
            ax_sz.set_title("Spin-z Expectation", fontsize=10)
            ax_sz.legend(fontsize=8)

        # Panel 1 — χ_min over time
        if self.metrics:
            steps_m = [m["step"] for m in self.metrics]
            ax_chi = _bot(1)
            ax_chi.plot(steps_m, self.chi_min_history, color="#4dac26", lw=1.5)
            ax_chi.axhline(_CHI0, color="grey", ls="--", lw=0.8, label=f"χ₀={_CHI0}")
            ax_chi.set_xlabel("Step"); ax_chi.set_ylabel("χ_min")
            ax_chi.set_title("χ Well Depth (Spin-Blind)", fontsize=10)
            ax_chi.legend(fontsize=8)

        # Panel 2 — experiment summary text
        ax_info = _bot(2)
        ax_info.axis("off")
        chi_fin = self.chi_min_history[-1] if self.chi_min_history else float("nan")
        lines = [
            f"Config     : {self.config_name}",
            f"Grid       : {N}³",
            f"χ_min init : {self.chi_min_initial:.4f}",
            f"χ_min final: {chi_fin:.4f}",
            f"CHSH S₀    : {self.chsh_initial:.4f}",
            f"CHSH S_f   : {self.chsh_final:.4f}",
            f"S ≤ 2 (classical)",
        ]
        ax_info.text(0.05, 0.5, "\n".join(lines), transform=ax_info.transAxes,
                     fontsize=10, family="monospace", va="center")
        ax_info.set_title("Experiment Summary", fontsize=10)

        return fig

    def save(
        self,
        stem: str,
        *,
        directory: str | Path | None = None,
        dpi: int = 150,
        save_movie: bool = True,
        save_snapshots_npz: bool = True,
    ) -> dict[str, Path]:
        """Write PNG summary + 3-D MP4 movie + snapshots NPZ.

        Returns a dict mapping output type → Path written.
        """
        import matplotlib.pyplot as plt
        from lfm.viz.collision import animate_collision_3d

        out = Path(directory) if directory else Path(".")
        out.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        # ── Summary PNG ────────────────────────────────────────────
        fig = self.plot()
        p_png = out / f"{stem}.png"
        fig.savefig(str(p_png), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written["summary"] = p_png

        # ── Snapshots NPZ ──────────────────────────────────────────
        if save_snapshots_npz and self.snapshots:
            p_npz = out / f"{stem}_snapshots.npz"
            self.save_snapshots_npz(p_npz)
            written["snapshots"] = p_npz

        # ── 3-D Movie MP4 ──────────────────────────────────────────
        if save_movie and self.movie_snapshots:
            p_mp4 = out / f"{stem}_3d_movie.mp4"
            try:
                # Build chi_deficit virtual field for each movie snapshot
                _chi0_val = float(CHI0)
                movie_snaps = []
                for snap in self.movie_snapshots:
                    s = dict(snap)
                    chi = snap.get("chi")
                    if chi is not None:
                        chi_np = _to_numpy(chi)
                        s["chi_deficit"] = np.abs(_chi0_val - chi_np.astype(np.float32))
                    movie_snaps.append(s)

                animate_collision_3d(
                    movie_snaps,
                    field="chi_deficit",
                    colormap="inferno",
                    fps=20,
                    intensity_floor=0.001,
                    max_frames=150,
                    title=self.label,
                    pos_a=(self.xA, self.N // 2, self.N // 2),
                    pos_b=(self.xB, self.N // 2, self.N // 2),
                    save_path=str(p_mp4),
                )
                written["movie"] = p_mp4
            except Exception as exc:
                print(f"    (3-D movie skipped: {exc})")

        return written


# ── Suite result (returned when config="all") ──────────────────────────────


class EntanglementSuiteResult:
    """Container for all four spin-configuration results.

    Attributes
    ----------
    results : dict[str, EntanglementResult]
        One :class:`EntanglementResult` per :data:`SPIN_CONFIGS` key.
    """

    def __init__(self, results: dict[str, EntanglementResult]) -> None:
        self.results = results

    def __getitem__(self, key: str) -> EntanglementResult:
        return self.results[key]

    def chsh_table(self) -> str:
        """Return a formatted table of CHSH values for all configs."""
        header = f"{'Config':18} {'S_initial':>10} {'S_final':>10}  {'S≤2?':>8}"
        rows = [header, "-" * len(header)]
        for name, res in self.results.items():
            bound = "✓" if res.chsh_final <= 2.0 + 1e-6 else "✗"
            rows.append(
                f"{name:18} {res.chsh_initial:10.4f} {res.chsh_final:10.4f}  {bound:>8}"
            )
        return "\n".join(rows)

    def plot(self, **kwargs) -> "Figure":
        """Comparison figure showing CHSH, χ_min, and spin-z for all configs."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=kwargs.get("figsize", (15, 5)))
        fig.suptitle("Spin Entanglement — All Configurations", fontsize=13,
                     fontweight="bold")

        cfg_names = list(self.results.keys())
        labels = [n.replace("_", "\n") for n in cfg_names]
        x = np.arange(len(cfg_names))

        chsh_vals = [self.results[n].chsh_final for n in cfg_names]
        chi_mins = [self.results[n].chi_min_initial for n in cfg_names]
        sz_A_vals = []
        sz_B_vals = []
        for n in cfg_names:
            h = self.results[n].spin_history
            if h:
                sz_A_vals.append(h[0]["sz_A"])
                sz_B_vals.append(h[0]["sz_B"])
            else:
                sz_A_vals.append(0.0)
                sz_B_vals.append(0.0)

        # ── Spin-z ────────────────────────────────────────────────
        ax = axes[0]
        ax.bar(x - 0.2, sz_A_vals, width=0.35, label="⟨σ_z^A⟩",
               color="#2166ac", alpha=0.85)
        ax.bar(x + 0.2, sz_B_vals, width=0.35, label="⟨σ_z^B⟩",
               color="#d6604d", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(-1.3, 1.3)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title("z-Polarisation (initial)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

        # ── χ_min spin-blindness ──────────────────────────────────
        ax = axes[1]
        ax.bar(x, chi_mins, color="#4dac26", alpha=0.85)
        ax.axhline(_CHI0, color="grey", ls="--", lw=1.0,
                   label=f"χ₀={_CHI0}")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title("χ_min (Spin-Blind Gravity)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

        # ── CHSH ─────────────────────────────────────────────────
        ax = axes[2]
        colors = ["#d73027" if s > 2.0 + 1e-6 else "#4dac26" for s in chsh_vals]
        ax.bar(x, chsh_vals, color=colors, alpha=0.85)
        ax.axhline(2.0, color="#d73027", lw=1.5, ls="--",
                   label="Classical bound (S=2)")
        ax.axhline(2.0 * math.sqrt(2.0), color="#762a83", lw=1.0, ls=":",
                   label=f"Quantum bound (S≈{2*math.sqrt(2):.3f})")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 3.0)
        ax.set_title("CHSH Bell Parameter", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        return fig

    def save(
        self,
        stem: str,
        *,
        directory: str | Path | None = None,
        dpi: int = 150,
        save_movie: bool = True,
        save_snapshots_npz: bool = True,
    ) -> dict[str, dict[str, Path]]:
        """Save all four results.  Returns ``{config_name: {type: path}}``."""
        import matplotlib.pyplot as plt

        out = Path(directory) if directory else Path(".")
        out.mkdir(parents=True, exist_ok=True)
        all_written: dict[str, dict[str, Path]] = {}

        for name, res in self.results.items():
            config_stem = f"{stem}_{name}"
            written = res.save(
                config_stem,
                directory=out,
                dpi=dpi,
                save_movie=save_movie,
                save_snapshots_npz=save_snapshots_npz,
            )
            all_written[name] = written

        # Suite comparison figure
        fig = self.plot()
        p_suite = out / f"{stem}_comparison.png"
        fig.savefig(str(p_suite), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        all_written["_suite"] = {"comparison": p_suite}

        return all_written


# ── Simulation builder ────────────────────────────────────────────────────


def _build_entanglement_sim(
    N: int,
    config: str,
    amplitude: float,
    sigma: float,
    chi0: float,
    verbose: bool,
) -> tuple["_lfm_t.Simulation", np.ndarray, np.ndarray, float, float, float, float]:
    """Create and populate a spinor entanglement simulation.

    Uses eigenmode relaxation (same algorithm as collision.py) for stable
    soliton initial conditions.  Complex-phase prev-buffers give constant
    spinor density so the χ wells are steady in the 3-D movie.

    Returns
    -------
    sim : Simulation
        Fully initialised simulation (ready to run).
    psi_r0, psi_i0 : ndarray
        Initial spinor fields (before time evolution).
    chi_min_initial : float
    chsh_initial : float
    xA, xB : float
    """
    import warnings
    import lfm
    from lfm.particles.solver import relax_eigenmode

    xA = N // 4
    xB = 3 * N // 4
    center = N // 2

    # Auto-choose sigma if not set
    if sigma <= 0.0:
        sigma = max(3.0, N * 0.04)

    if verbose:
        print(
            f"  Relaxing eigenmode: config={config!r}  N={N}  "
            f"amplitude={amplitude}  sigma={sigma:.1f}",
            flush=True,
        )

    # ── Step 1: relax a stable scalar eigenmode at grid centre ────
    sol = relax_eigenmode(
        N=N,
        amplitude=amplitude,
        sigma=sigma,
        chi0=chi0,
        verbose=verbose,
    )
    if not sol.converged:
        warnings.warn(
            f"Eigenmode relaxation did not converge "
            f"(chi_min={sol.chi_min:.3f}).  Using best result.",
            stacklevel=3,
        )
    if verbose:
        print(
            f"  Eigenmode: chi_min={sol.chi_min:.4f}  omega={sol.eigenvalue:.4f}",
            flush=True,
        )

    E_template = sol.psi_r  # (N,N,N) real scalar eigenmode
    omega = sol.eigenvalue   # oscillation frequency

    # ── Step 2: roll eigenmode to particle positions ──────────────
    def _roll_to_x(arr: np.ndarray, x_target: int) -> np.ndarray:
        shift = x_target - center
        return np.roll(arr, shift, axis=0) if shift != 0 else arr.copy()

    E_A = _roll_to_x(E_template, xA)
    E_B = _roll_to_x(E_template, xB)

    # ── Step 3: embed into spinor structure ───────────────────────
    psi_r, psi_i = _build_eigenmode_spinors(N, config, E_A, E_B)

    # ── Step 4: Poisson-solve combined χ ─────────────────────────
    density = psi_r[0] ** 2 + psi_i[0] ** 2 + psi_r[1] ** 2 + psi_i[1] ** 2
    chi_eq = equilibrate_chi(density, chi0=chi0, kappa=KAPPA)
    chi_min_initial = float(chi_eq.min())

    if verbose:
        print(f"  Combined χ_min = {chi_min_initial:.4f}", flush=True)

    # ── Step 5: measure initial CHSH ─────────────────────────────
    chsh_initial = _chsh(psi_r, psi_i, xA, xB, N)

    # ── Step 6: build simulation ──────────────────────────────────
    sim = lfm.Simulation(
        lfm.SimulationConfig(
            grid_size=N,
            field_level=lfm.FieldLevel.COLOR,
            n_colors=3,  # step_color kernel requires exactly 3 colors
            chi0=chi0,
            boundary_type=lfm.BoundaryType.ABSORBING,
            boundary_fraction=0.15,
        )
    )

    # ── Step 7: inject fields ─────────────────────────────────────
    # Complex-phase prev-buffers: Ψ(t=−dt) = psi_r·cos(ω·dt) − i·psi_r·sin(ω·dt)
    # This initialises a rotating complex eigenmode with constant |Ψ|²,
    # keeping the χ wells at steady depth throughout the movie.
    omega_dt = float(omega) * float(_DT)
    cos_dt = float(np.cos(omega_dt))
    sin_dt = float(np.sin(omega_dt))
    psi_r_prev = (psi_r * cos_dt).astype(np.float32)
    psi_i_prev = (-psi_r * sin_dt).astype(np.float32)

    sim.set_psi_real(psi_r)
    sim.set_psi_imag(psi_i)               # zero at t=0
    sim.set_psi_real_prev(psi_r_prev)
    sim.set_psi_imag_prev(psi_i_prev)
    sim.set_chi(chi_eq)                   # sets both current + prev chi buffers
    sim._equilibrated = True

    return sim, psi_r, psi_i, chi_min_initial, chsh_initial, float(xA), float(xB)


# ── Main experiment function ───────────────────────────────────────────────


def entanglement(
    config: str = "antiparallel",
    *,
    N: int = 64,
    amplitude: float = _AMPLITUDE,
    sigma: float = _SIGMA,
    chi0: float = _CHI0,
    total_steps: int = _TOTAL_STEPS,
    snap_every: int = _SNAP_EVERY,
    movie_every: int = _MOVIE_EVERY,
    metrics_every: int = _METRICS_EVERY,
    animate: bool = True,
    verbose: bool = False,
) -> "EntanglementResult | EntanglementSuiteResult":
    """Run the spin entanglement experiment.

    Two eigenmode spinor solitons (GOV-01 COLOR level with n_colors=3,
    are Poisson-equilibrated and evolved under GOV-01 + GOV-02.

    Parameters
    ----------
    config : str
        Spin configuration: ``"triplet"``, ``"antiparallel"``,
        ``"product_x"``, ``"singlet"``, or ``"all"``.
    N : int
        Grid size per axis.  Default 64.
    amplitude : float
        Peak spinor amplitude.
    sigma : float
        Gaussian soliton width in lattice cells.
    chi0 : float
        Vacuum χ value.
    total_steps : int
        Number of leapfrog steps.
    snap_every : int
        Capture a full-field snapshot every this many steps.
    movie_every : int
        Capture a movie frame every this many steps.
    metrics_every : int
        Record lightweight metrics every this many steps.
    animate : bool
        If False, skip movie capture (faster; no MP4).
    verbose : bool
        Print step-rate progress.

    Returns
    -------
    EntanglementResult
        When *config* is a single configuration name.
    EntanglementSuiteResult
        When *config* is ``"all"``.
    """
    if config == "all":
        results: dict[str, EntanglementResult] = {}
        for cfg in SPIN_CONFIGS:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Running config: {cfg}")
                print(f"{'='*50}")
            r = entanglement(
                cfg,
                N=N,
                amplitude=amplitude,
                sigma=sigma,
                chi0=chi0,
                total_steps=total_steps,
                snap_every=snap_every,
                movie_every=movie_every,
                metrics_every=metrics_every,
                animate=animate,
                verbose=verbose,
            )
            results[cfg] = r  # type: ignore[assignment]
        return EntanglementSuiteResult(results)

    # ── Single configuration ──────────────────────────────────────
    if config not in SPIN_CONFIGS:
        raise ValueError(
            f"Unknown config {config!r}.  Choose from {SPIN_CONFIGS} or 'all'."
        )

    t0 = time.perf_counter()

    sim, psi_r0, psi_i0, chi_min_init, chsh_init, xA, xB = _build_entanglement_sim(
        N, config, amplitude, sigma, chi0, verbose
    )
    xA_int = int(xA)
    xB_int = int(xB)

    label = f"LFM Spin Entanglement: {config}  N={N}"

    if verbose:
        print(f"  Starting evolution: {total_steps} steps", flush=True)

    # ── Evolution loop ────────────────────────────────────────────
    _movie_every = movie_every if animate else None

    snapshots, metrics, movie_snaps = gpu_snapshot_loop(
        sim,
        total_steps=total_steps,
        snap_every=snap_every,
        fields=["psi_real", "psi_imag", "chi"],   # full snapshots for spin analysis
        movie_every=_movie_every,
        movie_fields=["chi"],                       # only chi needed for chi_deficit movie
        metrics_every=metrics_every,
        verbose=verbose,
        label=config,
    )

    # ── Compute spin history from snapshots ───────────────────────
    spin_history: list[dict] = []
    for snap in snapshots:
        pr_s = snap.get("psi_real")
        pi_s = snap.get("psi_imag")
        if pr_s is None:
            continue
        pr_np = _to_numpy(pr_s)
        pi_np = _to_numpy(pi_s) if pi_s is not None else np.zeros_like(pr_np)
        szA, sxA, syA = _local_spin(pr_np, pi_np, xA_int, N)
        szB, sxB, syB = _local_spin(pr_np, pi_np, xB_int, N)
        spin_history.append({
            "step": snap.get("step", 0),
            "sz_A": szA, "sx_A": sxA, "sy_A": syA,
            "sz_B": szB, "sx_B": sxB, "sy_B": syB,
        })

    # ── Final CHSH ─────────────────────────────────────────────────
    chsh_final = chsh_init  # default if no snapshots
    if snapshots:
        last = snapshots[-1]
        pr_f = _to_numpy(last.get("psi_real", psi_r0))
        pi_f = last.get("psi_imag")
        pi_f = _to_numpy(pi_f) if pi_f is not None else np.zeros_like(pr_f)
        chsh_final = _chsh(pr_f, pi_f, xA_int, xB_int, N)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  Done in {elapsed:.1f}s", flush=True)
        print(f"  CHSH: S₀={chsh_init:.4f}  S_f={chsh_final:.4f}", flush=True)

    return EntanglementResult(
        snapshots=snapshots,
        movie_snapshots=movie_snaps,
        metrics=metrics,
        spin_history=spin_history,
        label=label,
        N=N,
        config_name=config,
        xA=xA_int,
        xB=xB_int,
        chsh_initial=chsh_init,
        chsh_final=chsh_final,
        chi_min_initial=chi_min_init,
    )
