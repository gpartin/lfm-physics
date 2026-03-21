"""3-D field rendering (isosurface / voxel plot)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfm.viz._util import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_isosurface(
    field: NDArray,
    threshold: float = 17.0,
    *,
    above: bool = False,
    title: str | None = None,
    color: str = "#2563eb",
    alpha: float = 0.3,
    figsize: tuple[float, float] = (8, 8),
) -> Figure:
    """Voxel render of cells where *field* crosses *threshold*.

    By default shows cells **below** the threshold (χ-wells).
    Set ``above=True`` to show cells **above** the threshold (voids).

    Parameters
    ----------
    field : ndarray (N, N, N)
        Scalar field.
    threshold : float
        Isosurface value.
    above : bool
        If False, render cells where field < threshold.
    title : str or None
        Plot title.
    color : str
        Voxel face colour.
    alpha : float
        Voxel transparency.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib Figure
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if field.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {field.shape}")

    mask = field > threshold if above else field < threshold

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Downsample if the grid is large to keep rendering fast
    N = field.shape[0]
    step = max(1, N // 64)
    small = mask[::step, ::step, ::step]

    ax.voxels(small, facecolors=color, edgecolor=color, alpha=alpha,
              linewidth=0.1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title is None:
        op = ">" if above else "<"
        title = f"Cells where field {op} {threshold}"
    ax.set_title(title)

    fig.tight_layout()
    return fig
