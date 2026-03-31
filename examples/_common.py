"""Shared helpers used by all lfm example scripts.

STANDARD PATTERN
----------------
There are TWO movie helpers.  Use run_and_save_3d_movie for everything
that involves particles — it produces the same true volumetric 3-D
scatter movie (rotating camera) that 18_collision.py and 23_double_slit.py
use.  run_and_save_movie is the flat 3-panel slice fallback.

  run_and_save_3d_movie  ← USE THIS for particles / solitons
  run_and_save_movie     ← flat 2-D slice panels (gravity / chi fields)

Usage in an experiment script::

    from _common import run_and_save_3d_movie, make_out_dir, parse_no_anim

    args = parse_no_anim()
    out  = make_out_dir("my_experiment")

    snaps, movie = run_and_save_3d_movie(
        sim, steps=4000, out_dir=out, stem="my_experiment",
        field="psi_real", snapshot_every=8, no_anim=args.no_anim,
    )
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lfm
from lfm.viz import animate_three_slices
from lfm.viz.collision import animate_collision_3d


# ---------------------------------------------------------------------------
# Directory helper
# ---------------------------------------------------------------------------

def make_out_dir(stem: str) -> Path:
    """Return ``examples/outputs/<stem>/``, creating it if needed."""
    out = Path(__file__).resolve().parent / "outputs" / stem
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Standard CLI flag
# ---------------------------------------------------------------------------

def parse_no_anim(parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    """Add ``--no-anim`` to *parser* (or a fresh parser) and parse args.

    --no-anim is the ONLY flag here; each experiment adds its own on top.
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-anim",
        action="store_true",
        help="Skip 3-D GIF movie (faster run, no Pillow needed)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

def run_and_save_movie(
    sim: lfm.Simulation,
    steps: int,
    out_dir: Path,
    stem: str,
    *,
    field: str = "psi_real",
    snapshot_every: int = 20,
    fps: int = 15,
    no_anim: bool = False,
    crop: int | None = None,
    fmt: str = "gif",
) -> tuple[list[dict], Path | None]:
    """Run *sim* for *steps* and save a 3-panel mid-plane slice movie.

    This is the standard pattern for ALL lfm example scripts.

    Three orthogonal mid-plane slices (xy / xz / yz) are animated side-by-side
    and saved as a GIF.  The movie is ON by default; disable with
    ``no_anim=True``.

    Parameters
    ----------
    sim : lfm.Simulation
        Simulation to evolve (must already be initialised / equilibrated).
    steps : int
        Number of leapfrog steps to run.
    out_dir : Path
        Output directory (created if needed).
    stem : str
        Base filename without extension, e.g. ``"electron_at_rest"``.
    field : str
        Scalar field to capture and animate.
        Choose from ``"psi_real"``, ``"chi"``, ``"energy_density"``.
        ``"psi_real"`` shows the wave-function oscillation for particles.
        ``"chi"`` shows the gravity-well structure.
    snapshot_every : int
        Capture a field snapshot every this many steps.
        Rule of thumb: set to ≈ (oscillation period in steps) / 4
        so the animation shows smooth motion.
        Electron at ω≈19, dt=0.02 → period ≈ 16 steps → use 4.
        Proton soliton → use 20–50.
    fps : int
        Frames per second in the saved GIF / MP4.
    no_anim : bool
        If True, run_with_snapshots still runs (for metrics / analysis)
        but the movie is not written.
    crop : int or None
        If given, crop each snapshot field to a cube of side 2*crop
        centred on the grid midpoint.  Use this to zoom into a small
        soliton (e.g. crop=12 → 24×24×24 view for a sigma≈3 electron).
    fmt : str
        Output format: ``"gif"`` (default) or ``"mp4"``.

    Returns
    -------
    snapshots : list[dict]
        Raw snapshot list (always returned, even when no_anim=True).
    save_path : Path or None
        Path to the saved movie, or None if no_anim=True.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always capture chi for metrics; add the requested visualization field.
    fields = list({"chi", field})

    snaps = sim.run_with_snapshots(
        steps,
        snapshot_every=snapshot_every,
        fields=fields,
    )

    save_path: Path | None = None
    if not no_anim:
        ext = fmt.lstrip(".")
        save_path = out_dir / f"{stem}_{field}.{ext}"
        display_snaps = _crop_snaps(snaps, field, crop) if crop is not None else snaps
        animate_three_slices(
            display_snaps,
            field=field,
            save_path=str(save_path),
            fps=fps,
        )
        sz_kb = save_path.stat().st_size // 1024
        print(f"  Movie saved  → {save_path.name}  ({sz_kb} KB)")

    return snaps, save_path


def run_and_save_3d_movie(
    sim: lfm.Simulation,
    steps: int,
    out_dir: Path,
    stem: str,
    *,
    field: str = "psi_real",
    extra_fields: list[str] | None = None,
    snapshot_every: int = 20,
    fps: int = 20,
    no_anim: bool = False,
    intensity_floor: float = 0.002,
    camera_rotate: bool = False,
    max_frames: int = 150,
    title: str = "LFM Simulation",
    crop_radius: int | None = None,
) -> tuple[list[dict], Path | None]:
    """Run *sim* and save a TRUE volumetric 3-D scatter movie.

    Identical renderer to ``18_collision.py`` and ``23_double_slit.py``:
    dark background, rotating camera, scatter points coloured and sized by
    field value.  USE THIS for all particle / soliton experiments.

    Parameters
    ----------
    sim : lfm.Simulation
        Fully initialised and equilibrated simulation.
    steps : int
        Steps to evolve.
    out_dir : Path
        Output directory (created if needed).
    stem : str
        Base filename (no extension), e.g. ``"electron_at_rest"``.
    field : str
        Field to scatter-render.  ``"psi_real"`` (particle wave function),
        ``"chi_deficit"`` (gravitational well structure — wells appear as
        bright spots on a black background; preferred over raw ``"chi"``),
        or ``"energy_density"``.  Raw ``"chi"`` is also accepted but renders
        the vacuum background as bright fog, hiding wells.
    snapshot_every : int
        Capture a 3-D snapshot every this many steps.
    fps : int
        Frames per second in the saved MP4.
    no_anim : bool
        If True, run still executes but no movie is written.
    intensity_floor : float
        Fraction of the global field peak below which scatter points are
        invisible.  Lowering this reveals shallow structures (use 0.001 for
        electrons).
    camera_rotate : bool
        If True the camera rotates around the scene.
    max_frames : int
        Maximum number of frames rendered (subsampled if there are more
        snapshots).

    Returns
    -------
    snapshots : list[dict]
        Raw snapshot list (always returned).
    save_path : Path or None
        Path to the saved ``.mp4``, or None when ``no_anim=True``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # chi_deficit is a virtual field: capture "chi" then compute CHI0-chi.
    _raw_field = "chi" if field == "chi_deficit" else field
    capture = {"chi", _raw_field}
    if _raw_field == "psi_real":
        capture.add("psi_imag")  # needed for amplitude mode (no blinking)
    if extra_fields:
        capture.update(extra_fields)
    fields = list(capture)
    snaps = sim.run_with_snapshots(
        steps,
        snapshot_every=snapshot_every,
        fields=fields,
    )
    # Materialise the chi_deficit virtual field: wells appear as bright spots.
    if field == "chi_deficit":
        import numpy as _np
        _chi0 = float(lfm.CHI0)
        for _snap in snaps:
            _chi = _snap.get("chi")
            if _chi is not None:
                try:
                    import cupy as _cp
                    _chi = _cp.asnumpy(_chi) if isinstance(_chi, _cp.ndarray) else _np.asarray(_chi)
                except ImportError:
                    _chi = _np.asarray(_chi)
                _snap["chi_deficit"] = _np.abs(_chi0 - _chi.astype(_np.float32))
    save_path: Path | None = None
    snaps_3d = _maybe_reduce_color(snaps, field)  # collapse COLOR level (3,N,N,N) → (N,N,N)
    render_snaps = _crop_snaps(snaps_3d, field, crop_radius) if crop_radius is not None else snaps_3d
    if not no_anim:
        save_path = out_dir / f"{stem}_3d.mp4"
        animate_collision_3d(
            render_snaps,
            field=field,
            fps=fps,
            intensity_floor=intensity_floor,
            camera_rotate=camera_rotate,
            max_frames=max_frames,
            title=title,
            show_phase_labels=False,
            save_path=str(save_path),
        )
        sz_kb = save_path.stat().st_size // 1024
        print(f"  3-D movie saved → {save_path.name}  ({sz_kb} KB)")
    return snaps, save_path


def _maybe_reduce_color(snaps: list[dict], field: str) -> list[dict]:
    """If *field* is 4-D (COLOR level, shape (3,N,N,N)), collapse to 3-D amplitude.

    Computes ``|Ψ| = sqrt(Σ_a (real_a² + imag_a²))`` when psi_imag is also
    4-D, otherwise ``sqrt(Σ_a real_a²)``.  The resulting (N,N,N) field works
    with the scatter renderer and the blink-free amplitude mode.
    """
    if not snaps:
        return snaps
    sample = snaps[0].get(field)
    if sample is None or getattr(sample, "ndim", 0) != 4:
        return snaps
    result = []
    for snap in snaps:
        s = dict(snap)
        real = s.get(field)
        imag = s.get("psi_imag")
        if real is not None and real.ndim == 4:
            sq = real ** 2
            if imag is not None and imag.ndim == 4:
                sq = sq + imag ** 2
            amp = sq.sum(axis=0) ** 0.5
            s[field] = amp
            if "psi_imag" in s:
                s["psi_imag"] = amp * 0  # zero → amplitude mode yields amp as-is
        result.append(s)
    return result


def _crop_snaps(snaps: list[dict], field: str, crop: int) -> list[dict]:
    """Return a copy of *snaps* with field (and chi) cropped to a
    ``2*crop``-wide cube centred on the grid midpoint."""
    result = []
    for snap in snaps:
        arr = snap[field]
        N = arr.shape[0]
        c = N // 2
        lo, hi = max(0, c - crop), min(N, c + crop)
        s = slice(lo, hi)
        new_snap = dict(snap)
        new_snap[field] = arr[s, s, s]
        if "chi" in snap and snap["chi"].ndim == 3:
            new_snap["chi"] = snap["chi"][s, s, s]
        if "psi_imag" in snap and snap["psi_imag"].ndim == 3:
            new_snap["psi_imag"] = snap["psi_imag"][s, s, s]
        result.append(new_snap)
    return result
