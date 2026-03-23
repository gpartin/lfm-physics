"""SPARC galaxy rotation-curve data loader.

SPARC (Spitzer Photometry and Accurate Rotation Curves) provides
high-quality observed rotation curves for 175 disk galaxies.

Reference
---------
Lelli, McGaugh & Schombert (2016), AJ 152, 157.
Data: http://astroweb.cwru.edu/SPARC/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Built-in sample data (five representative galaxies, hard-coded so that
# lfm-physics works out-of-the-box without a SPARC download).  These are
# the published observed values from Lelli+ (2016).
# ---------------------------------------------------------------------------

# fmt: off
_BUILTIN_SPARC: dict[str, dict] = {
    "NGC6503": {
        "distance_Mpc": 5.27,
        "inclination_deg": 74.0,
        "r_kpc":  np.array([0.46, 0.92, 1.37, 1.83, 2.29, 2.74, 3.20, 3.66,
                             4.11, 4.57, 5.03, 5.48, 5.94, 6.40, 6.85, 7.31,
                             7.77, 8.22, 8.68, 9.14, 9.59, 10.05], dtype=np.float32),
        "v_obs_kms": np.array([54.5, 78.4, 90.6, 97.0, 100.8, 102.9, 104.0, 105.2,
                                107.0, 109.2, 110.1, 111.0, 111.8, 112.4, 112.3, 111.6,
                                110.5, 109.4, 108.4, 107.5, 107.1, 107.3], dtype=np.float32),
        "v_err_kms": np.array([2.1, 1.8, 1.5, 1.4, 1.3, 1.3, 1.3, 1.3,
                                1.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                                1.6, 1.6, 1.7, 1.8, 1.9, 2.0], dtype=np.float32),
    },
    "NGC3198": {
        "distance_Mpc": 14.1,
        "inclination_deg": 71.5,
        "r_kpc":  np.array([1.10, 2.20, 3.31, 4.41, 5.51, 6.61, 7.72, 8.82,
                             9.92, 11.02, 12.13, 13.23, 14.33, 15.43, 16.54, 17.64,
                             18.74, 19.85, 20.95, 22.05, 23.15, 24.26, 25.36, 26.46,
                             27.56, 28.67, 29.77, 30.87], dtype=np.float32),
        "v_obs_kms": np.array([90.0, 133.0, 148.0, 150.0, 152.0, 153.0, 153.5, 153.5,
                                152.5, 151.5, 151.0, 151.0, 151.5, 151.5, 151.0, 150.5,
                                150.0, 149.5, 149.0, 148.5, 148.0, 147.5, 147.0, 146.5,
                                146.0, 145.5, 145.0, 144.5], dtype=np.float32),
        "v_err_kms": np.array([5.0, 4.0, 3.5, 3.0, 3.0, 2.8, 2.8, 2.8,
                                2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 3.0,
                                3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                                3.5, 3.5, 3.5, 4.0], dtype=np.float32),
    },
    "DDO154": {
        "distance_Mpc": 3.73,
        "inclination_deg": 66.0,
        "r_kpc":  np.array([0.24, 0.47, 0.71, 0.95, 1.18, 1.42, 1.66, 1.89,
                             2.13, 2.37, 2.60, 2.84, 3.08, 3.31, 3.55, 3.79,
                             4.02, 4.26, 4.50, 4.73, 4.97, 5.21], dtype=np.float32),
        "v_obs_kms": np.array([11.0, 19.5, 26.0, 31.0, 35.0, 38.0, 40.0, 42.0,
                                43.5, 44.5, 45.5, 46.0, 46.5, 47.0, 47.5, 47.8,
                                48.0, 48.0, 48.0, 47.8, 47.5, 47.3], dtype=np.float32),
        "v_err_kms": np.array([1.2, 1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8,
                                0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
                                0.9, 0.9, 0.9, 1.0, 1.0, 1.1], dtype=np.float32),
    },
    "IC2574": {
        "distance_Mpc": 3.79,
        "inclination_deg": 53.0,
        "r_kpc":  np.array([0.47, 0.94, 1.41, 1.88, 2.35, 2.82, 3.29, 3.76,
                             4.23, 4.70, 5.17, 5.64, 6.11, 6.58, 7.05, 7.52,
                             7.99, 8.46, 8.93, 9.40, 9.87], dtype=np.float32),
        "v_obs_kms": np.array([12.0, 22.0, 33.0, 44.0, 55.0, 63.0, 69.0, 73.0,
                                76.0, 78.0, 79.5, 80.5, 81.0, 81.0, 80.5, 80.0,
                                79.5, 79.0, 78.5, 78.0, 77.5], dtype=np.float32),
        "v_err_kms": np.array([2.0, 1.8, 1.6, 1.5, 1.4, 1.4, 1.4, 1.4,
                                1.4, 1.4, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5,
                                1.6, 1.6, 1.7, 1.8, 1.9], dtype=np.float32),
    },
    "UGC2885": {
        "distance_Mpc": 80.4,
        "inclination_deg": 64.0,
        "r_kpc":  np.array([2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0,
                             22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0,
                             42.5, 45.0, 47.5, 50.0], dtype=np.float32),
        "v_obs_kms": np.array([150.0, 215.0, 250.0, 270.0, 280.0, 285.0, 288.0, 290.0,
                                291.0, 292.0, 292.5, 293.0, 293.0, 293.0, 292.5, 292.0,
                                291.5, 291.0, 290.5, 290.0], dtype=np.float32),
        "v_err_kms": np.array([7.0, 6.0, 5.5, 5.0, 5.0, 5.0, 5.0, 5.0,
                                5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                5.5, 5.5, 6.0, 6.0], dtype=np.float32),
    },
}
# fmt: on


def sparc_load(
    path_or_name: str | Path | None = None,
) -> dict[str, dict]:
    """Load SPARC rotation-curve data.

    Can be used in three ways:

    1. **No argument** → returns the five built-in sample galaxies::

        galaxies = lfm.sparc_load()
        row = galaxies["NGC6503"]

    2. **Galaxy name** (one of the five built-ins) → returns just that row::

        row = lfm.sparc_load("NGC3198")

    3. **Path to SPARC data directory** → loads all ``*_rotmod.dat`` files
       from the official SPARC download.  Each file must have the
       ``Rad  Vobs  errV  Vgas  Vdisk  Vbul  SBdisk  SBbul`` column
       layout used by Lelli+ (2016).

    Parameters
    ----------
    path_or_name : str, Path, or None
        Galaxy name, path to a SPARC directory, or *None* for all built-ins.

    Returns
    -------
    dict
        ``{galaxy_name: {"r_kpc", "v_obs_kms", "v_err_kms",
        "distance_Mpc", "inclination_deg"}}``

    Examples
    --------
    >>> data = lfm.sparc_load("NGC6503")
    >>> row = data["NGC6503"]
    >>> print(row["r_kpc"])
    """
    # -- Built-in name lookup -----------------------------------------------
    if path_or_name is None:
        return dict(_BUILTIN_SPARC)

    if isinstance(path_or_name, str) and path_or_name in _BUILTIN_SPARC:
        return {path_or_name: dict(_BUILTIN_SPARC[path_or_name])}

    # -- Directory of SPARC *_rotmod.dat files --------------------------------
    sparc_dir = Path(path_or_name)
    if not sparc_dir.is_dir():
        raise FileNotFoundError(
            f"'{sparc_dir}' is not a directory. "
            "Pass a SPARC galaxy name (e.g. 'NGC6503') or the path to the "
            "SPARC data directory containing *_rotmod.dat files."
        )

    result: dict[str, dict] = {}
    for dat_file in sorted(sparc_dir.glob("*_rotmod.dat")):
        name = dat_file.stem.replace("_rotmod", "")
        try:
            data = np.loadtxt(dat_file, comments="#")
        except Exception:
            continue
        if data.ndim != 2 or data.shape[1] < 3:
            continue
        result[name] = {
            "r_kpc": data[:, 0].astype(np.float32),
            "v_obs_kms": data[:, 1].astype(np.float32),
            "v_err_kms": data[:, 2].astype(np.float32),
            "distance_Mpc": float("nan"),
            "inclination_deg": float("nan"),
        }

    if not result:
        return result
    return result


def list_sparc_galaxies() -> list[str]:
    """Return the names of the five built-in sample galaxies.

    >>> lfm.list_sparc_galaxies()
    ['DDO154', 'IC2574', 'NGC3198', 'NGC6503', 'UGC2885']
    """
    return sorted(_BUILTIN_SPARC.keys())
