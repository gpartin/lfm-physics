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

from lfm.core.backends.numpy_backend import NumpyBackend

# Check GPU availability at import time (but don't fail)
try:
    from lfm.core.backends.cupy_backend import CupyBackend, CUPY_AVAILABLE
except ImportError:
    CupyBackend = None  # type: ignore[misc, assignment]
    CUPY_AVAILABLE = False


def get_backend(preference: str = "auto") -> NumpyBackend:
    """Get a compute backend instance.

    Parameters
    ----------
    preference : str
        One of 'auto', 'cpu', 'gpu'.
        - 'auto': Use GPU if CuPy is available, else CPU.
        - 'cpu': Always use NumPy (CPU).
        - 'gpu': Use CuPy (GPU). Raises ImportError if unavailable.

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

    if preference == "cpu":
        return NumpyBackend()

    if preference == "gpu":
        if not CUPY_AVAILABLE or CupyBackend is None:
            raise ImportError(
                "CuPy not available. Install with: pip install lfm-physics[gpu]"
            )
        return CupyBackend()  # type: ignore[return-value]

    if preference == "auto":
        if CUPY_AVAILABLE and CupyBackend is not None:
            return CupyBackend()  # type: ignore[return-value]
        return NumpyBackend()

    raise ValueError(
        f"Unknown backend preference '{preference}'. Use 'auto', 'cpu', or 'gpu'."
    )


def gpu_available() -> bool:
    """Check whether the GPU (CuPy) backend is available."""
    return CUPY_AVAILABLE


__all__ = ["get_backend", "gpu_available", "NumpyBackend"]
