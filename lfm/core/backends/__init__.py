"""
Compute Backends
================

Auto-detect and provide the best available backend.

Usage::

    from lfm.core.backends import get_backend

    backend = get_backend()       # Auto-detect (GPU if available, else CPU)
    backend = get_backend("cpu")  # Force CPU
    backend = get_backend("gpu")  # Force GPU (raises if unavailable)
"""

from __future__ import annotations

from lfm.config import Precision
from lfm.core.backends.numpy_backend import NumpyBackend

# Check GPU availability at import time (but don't fail)
try:
    from lfm.core.backends.cupy_backend import CUPY_AVAILABLE, CupyBackend
except ImportError:
    CupyBackend = None  # type: ignore[misc, assignment]
    CUPY_AVAILABLE = False


def get_backend(
    preference: str = "auto",
    precision: Precision | str = Precision.FLOAT32,
) -> NumpyBackend:
    """Get a compute backend instance.

    Parameters
    ----------
    preference : str
        One of 'auto', 'cpu', 'gpu'.
        - 'auto': Use GPU if CuPy is available, else CPU.
        - 'cpu': Always use NumPy (CPU).
        - 'gpu': Use CuPy (GPU). Raises ImportError if unavailable.
    precision : Precision or str
        Persistent state precision. Defaults to the canonical float32 path.

    Returns
    -------
    Backend
        A NumpyBackend or CupyBackend instance.

    Raises
    ------
    ImportError
        If preference='gpu' but CuPy is not installed.
    ValueError
        If preference is not recognized.
    """
    preference = preference.lower()
    precision = Precision(precision)

    if preference == "cpu":
        return NumpyBackend(precision=precision)

    if preference == "gpu":
        if not CUPY_AVAILABLE or CupyBackend is None:
            raise ImportError("CuPy not available. Install with: pip install lfm-physics[gpu]")
        return CupyBackend(precision=precision)  # type: ignore[return-value]

    if preference == "auto":
        if CUPY_AVAILABLE and CupyBackend is not None:
            return CupyBackend(precision=precision)  # type: ignore[return-value]
        return NumpyBackend(precision=precision)

    if preference == "remote":
        if precision != Precision.FLOAT32:
            raise NotImplementedError(
                "remote backend supports only float32 jobs; use cpu or gpu for float64"
            )
        from lfm.core.backends.remote_backend import RemoteBackend

        return RemoteBackend()  # type: ignore[return-value]

    raise ValueError(
        f"Unknown backend preference '{preference}'. Use 'auto', 'cpu', 'gpu', or 'remote'."
    )


def gpu_available() -> bool:
    """Check whether the GPU (CuPy) backend is available."""
    return CUPY_AVAILABLE


__all__ = ["get_backend", "gpu_available", "NumpyBackend"]
