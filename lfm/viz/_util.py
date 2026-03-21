"""Helpers shared across viz submodules."""

from __future__ import annotations


def _require_matplotlib():
    """Raise a helpful error if matplotlib is not installed."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError(
            "Visualization requires matplotlib.  Install it with:\n"
            "    pip install 'lfm-physics[viz]'"
        ) from None
