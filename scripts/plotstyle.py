"""Shared look for the visual tests.

One place to change how every figure reads. Fields are drawn on a single-hue
sequential ramp running from the page surface (zero, which recedes) to dark blue
(one), so magnitude is carried by lightness alone and stays legible in greyscale and
under colour-vision deficiency. Line series take categorical slots in fixed order.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "FIELD_CMAP",
    "FIGURES",
    "INK",
    "INK_MUTED",
    "INK_SECONDARY",
    "SERIES",
    "SURFACE",
    "field_axes",
    "save",
    "show_field",
    "use_style",
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

#: Categorical slots, in the order they must be assigned -- never cycled.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

#: Sequential ramp for scalar fields: surface at zero, dark blue at one.
FIELD_CMAP = LinearSegmentedColormap.from_list("field_blue", [SURFACE, *_BLUE])

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def use_style() -> None:
    """Apply the shared rcParams. Call once at the top of a script."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "text.color": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def show_field(ax, field, grid, vmin=0.0, vmax=1.0, cmap=None):
    """Draw a ``(ny, nx)`` field in physical coordinates.

    ``origin="lower"`` with an explicit extent keeps the picture in the same
    orientation as the maths -- y increasing upwards, and array index ``[j, i]`` at
    position ``(i*dx, j*dy)``.
    """
    return ax.imshow(
        field,
        origin="lower",
        extent=(0.0, grid.box.Lx, 0.0, grid.box.Ly),
        cmap=FIELD_CMAP if cmap is None else cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )


def field_axes(ax, grid, title=None):
    """Square axes covering the box exactly, with recessive chrome."""
    ax.set_xlim(0.0, grid.box.Lx)
    ax.set_ylim(0.0, grid.box.Ly)
    ax.set_aspect("equal")
    ax.set_xticks([0.0, grid.box.Lx])
    ax.set_yticks([0.0, grid.box.Ly])
    ax.set_xticklabels(["0", f"{grid.box.Lx:.2f}"])
    ax.set_yticklabels(["0", f"{grid.box.Ly:.2f}"])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BASELINE)
    if title:
        ax.set_title(title, loc="left", pad=8)
    return ax


def save(fig, name: str, directory: Path | str | None = None) -> Path:
    """Write a figure as PNG into ``directory``, creating it if needed.

    Defaults to the repository's ``figures/``. A script that keeps a run's ``.npz``
    somewhere else passes that directory, so every output of a run sits together.
    """
    directory = FIGURES if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path)
    return path
