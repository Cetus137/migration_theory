"""Drawing a :class:`~migration_theory.simulate.Trajectory`.

Everything here consumes a finished trajectory and produces a figure or an animation.
Nothing here simulates, and nothing in ``migration_theory`` imports this -- so a run can
be done headless, on a cluster, with no plotting library present, and rendered later
from the result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import plotstyle

__all__ = [
    "field_image",
    "colour_limits",
    "outlines",
    "energy_axes",
    "energy_figure",
    "make_animation",
    "observables_figure",
    "pair_correlation_figure",
    "sweep_figure",
    "save_animation",
]

#: Neighbour distances of a hexagonal lattice, in units of the spacing: where g(r)
#: would peak if the tissue were a crystal.
HEXAGONAL_SHELLS = (1.0, np.sqrt(3.0), 2.0, np.sqrt(7.0), 3.0)


def pair_correlation_figure(curves, spacing_label="cell spacings"):
    """``g(r)`` for several runs on one pair of axes, with the hexagonal shells marked.

    ``curves`` is a list of ``(name, x, g)`` with ``x`` in cell spacings. Ordered
    values of one parameter, so the lines take the sequential ramp, light to dark.
    The dotted verticals are the neighbour distances of a hexagonal lattice: peaks on
    them mean crystalline order, a broad second peak between them a glass, no second
    peak a fluid. ``g = 1`` is no structure.
    """
    plotstyle.use_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ramp = plotstyle.FIELD_CMAP(np.linspace(0.3, 1.0, max(len(curves), 2)))
    for shell in HEXAGONAL_SHELLS:
        ax.axvline(shell, color=plotstyle.INK_MUTED, lw=0.7, ls=":")
    ax.axhline(1.0, color=plotstyle.BASELINE, lw=0.8)
    for colour, (name, x, g) in zip(ramp, curves):
        ax.plot(x, g, color=colour, label=name)
    ax.set_xlim(0.0, max(float(np.max(x)) for _, x, _ in curves))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(f"separation  ({spacing_label})")
    ax.set_ylabel("$g(r)$")
    ax.set_title("Pair correlation of cell centres", loc="left", pad=8)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def field_image(ax, field, grid, vmin=0.0, vmax=None, colorbar=None,
                label="$\\sum_i \\phi_i^2$"):
    """A scalar field over the box, with optional colour bar."""
    plotstyle.field_axes(ax, grid)
    image = plotstyle.show_field(ax, field, grid, vmin=vmin,
                                 vmax=max(1.0, float(np.max(field)))
                                 if vmax is None else vmax)
    if colorbar is not None:
        bar = colorbar.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0, labelsize=8)
        bar.set_label(label, fontsize=8)
    return image


def outlines(ax, tissue, colour=None, width=0.8):
    """The ``phi = 0.5`` contour of every cell.

    A single neutral ink, not one colour per cell: with this many cells hue cannot
    carry identity, and here it does not need to.
    """
    X, Y = tissue.grid.coordinates
    for cell in tissue.fields:
        ax.contour(X, Y, cell, levels=[0.5],
                   colors=[plotstyle.INK if colour is None else colour], linewidths=width)


def energy_axes(ax, trajectory, colour=None, label=None, relative=False):
    """Free energy against time. ``relative`` plots ``F - F_final`` on a log axis."""
    times, energies = trajectory.times, trajectory.energies
    if relative:
        ax.plot(times[1:], energies[1:] - energies[-1] + 1e-15,
                color=colour or plotstyle.SERIES[0], label=label)
        ax.set_yscale("log")
        ax.set_ylabel("$F - F_{\\infty}$")
    else:
        ax.plot(times, energies, color=colour or plotstyle.SERIES[0], label=label)
        ax.set_ylabel("free energy $F$")
    ax.set_xlabel("time")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    return ax


def energy_figure(trajectory, transient=0.2):
    """A static four-panel account of how the run behaved over time.

    Separated from the animation because none of it benefits from being animated: it
    is a curve against time, and a curve against time is easiest to read all at once.

    The top row shows the total free energy twice -- once over the whole run, once
    scaled to the last ``1 - transient`` of it. A relaxing tissue sheds most of its
    energy in the first few percent, so a single axis scaled to the full range is a
    cliff followed by a flat line and tells you nothing about whether the run is still
    evolving.
    """
    plotstyle.use_style()
    times, energies = trajectory.times, trajectory.energies
    model = trajectory.model

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))

    ax = axes[0, 0]
    ax.plot(times, energies, color=plotstyle.SERIES[0])
    ax.set_title("Free energy, whole run", loc="left", pad=8)
    ax.set_xlabel("time")
    ax.set_ylabel("$F$")

    ax = axes[0, 1]
    start = min(max(1, int(transient * len(energies))), len(energies) - 1)
    ax.plot(times[start:], energies[start:], color=plotstyle.SERIES[0])
    ax.set_title(f"Free energy, $t$ > {times[start]:.3g}", loc="left", pad=8)
    ax.set_xlabel("time")
    ax.set_ylabel("$F$")

    ax = axes[1, 0]
    for index, name in enumerate(trajectory.snapshots[0].breakdown):
        ax.plot(times, [s.breakdown[name] for s in trajectory.snapshots],
                color=plotstyle.SERIES[index % 4], label=name)
    ax.set_yscale("log")   # the terms differ by orders of magnitude
    ax.set_title("By term", loc="left", pad=8)
    ax.set_xlabel("time")
    ax.set_ylabel("energy")
    ax.legend(loc="best")

    ax = axes[1, 1]
    for index, (key, name) in enumerate((("area_ratio", "$A/A_0$"),
                                         ("shape_index", "$q = P/\\sqrt{A}$"),
                                         ("occupancy", "max $\\sum_i\\phi_i$"),
                                         ("confluence", "confluence error"))):
        ax.plot(times, [getattr(s, key) for s in trajectory.snapshots],
                color=plotstyle.SERIES[index % 4], label=name)
    ax.axhline(1.0, color=plotstyle.INK_MUTED, lw=0.8, ls=":")
    ax.set_title("Diagnostics", loc="left", pad=8)
    ax.set_xlabel("time")
    ax.legend(loc="best")

    for ax in axes.ravel():
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    fig.suptitle(
        f"{model.n_cells} cells   $\\alpha$={model.alpha:g} $K$={model.K:g} "
        f"$\\epsilon$={model.epsilon:g} $\\lambda$={model.area_lambda:g} "
        f"$\\gamma$={model.friction:g} $v_0$={model.speed:g} $R$={model.cell_radius:g} um   |   "
        f"$t$ {trajectory.duration:.0f}   d$F$/d$t$ {trajectory.drift:+.2e}",
        x=0.01, ha="left", fontsize=10, fontweight="semibold", color=plotstyle.INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def colour_limits(snapshots, transient=0.2, span=(1.0, 99.9)):
    """Colour range for a field animation, ignoring the initial relaxation.

    Two problems with scaling to the whole run. The seeded configuration is the most
    extreme frame there is, so it sets a ceiling nothing else reaches and the rest of
    the video uses only part of the ramp. And a handful of outlying pixels can stretch
    the range so that the bulk of the tissue -- most of the image -- crowds into a
    narrow band of colour and reads as flat.

    So: drop the first ``transient`` of the run, then take percentiles rather than the
    extremes. Frames or pixels outside the result clip, which is the honest trade.
    """
    start = min(max(1, int(transient * len(snapshots))), len(snapshots) - 1)
    values = np.concatenate([s.field.ravel() for s in snapshots[start:]])
    low, high = np.percentile(values, span)
    if high <= low:
        high = low + max(abs(low), 1.0) * 1e-6
    return float(low), float(high)


def make_animation(trajectory, fps=20, transient=0.2, vmin=None, vmax=None):
    """The tissue over time. Just the tissue -- see :func:`energy_figure` for the rest."""
    snapshots = [s for s in trajectory.snapshots if s.field is not None]
    if not snapshots:
        raise ValueError("trajectory has no stored fields; simulate with keep_fields=True")

    plotstyle.use_style()
    model, grid = trajectory.model, trajectory.tissue.grid
    fig, left = plt.subplots(figsize=(6.2, 5.6))

    auto_low, auto_high = colour_limits(snapshots, transient)
    floor = auto_low if vmin is None else vmin
    ceiling = auto_high if vmax is None else vmax
    image = field_image(left, snapshots[0].field, grid, vmin=floor, vmax=ceiling,
                        colorbar=fig)
    markers, = left.plot([], [], ".", color=plotstyle.INK, ms=3)

    title = fig.suptitle("", x=0.01, ha="left", fontsize=9,
                         fontweight="semibold", color=plotstyle.INK)
    header = (
        f"{model.n_cells} cells   $\\alpha$={model.alpha:g} $K$={model.K:g} "
        f"$\\epsilon$={model.epsilon:g} $\\lambda$={model.area_lambda:g} "
        f"$\\gamma$={model.friction:g} $v_0$={model.speed:g}"
    )

    def draw(index):
        snapshot = snapshots[index]
        image.set_data(snapshot.field)
        markers.set_data(snapshot.centres[:, 0], snapshot.centres[:, 1])
        left.set_title(f"$t$ = {snapshot.time:.1f}", loc="left", pad=8)
        title.set_text(
            f"{header}   |   $A/A_0$ {snapshot.area_ratio:.3f}   "
            f"$q$ {snapshot.shape_index:.2f}"
        )
        return image, markers

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return animation.FuncAnimation(fig, draw, frames=len(snapshots),
                                   interval=1000 / fps, blit=False)


def observables_figure(series, label, reported):
    """Each observable against the swept parameter, one small panel apiece.

    Small multiples rather than shared axes: the quantities differ by orders of
    magnitude -- a T1 rate near zero beside a shape index near 4 -- and forcing them
    onto one pair of axes would flatten whichever is smaller into the baseline.

    ``series`` is a list of dicts, one line per entry, each with ``values`` (the x
    axis), ``states`` (one observables dict per value), an optional ``spreads`` (one
    dict per value, drawn as error bars -- the standard deviation over seeds) and an
    optional ``name`` for the legend. A one-parameter sweep is a single unnamed entry;
    a two-parameter grid is one entry per value of the second parameter, so that
    parameter is carried by colour and the first by the x axis.
    """
    plotstyle.use_style()
    columns = 4
    rows = int(np.ceil(len(reported) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.7 * rows),
                             squeeze=False)
    single = len(series) == 1
    # One series takes a categorical colour per panel. Several series are ordered
    # values of a second parameter, so they take a sequential ramp: light for the
    # smallest, dark for the largest, and the order is legible however many there are.
    ramp = plotstyle.FIELD_CMAP(np.linspace(0.3, 1.0, max(len(series), 2)))

    for index, (key, title) in enumerate(reported):
        ax = axes[index // columns][index % columns]
        for order, entry in enumerate(series):
            colour = (plotstyle.SERIES[index % len(plotstyle.SERIES)] if single
                      else ramp[order])
            values = entry["values"]
            heights = [state[key] for state in entry["states"]]
            if entry.get("spreads") is None:
                ax.plot(values, heights, "o-", color=colour, ms=5, label=entry.get("name"))
            else:
                ax.errorbar(values, heights, yerr=[s[key] for s in entry["spreads"]],
                            fmt="o-", color=colour, ms=5, capsize=3, label=entry.get("name"))
            if key.startswith("velocity_correlation_length") and "velocity_correlation_bound" in entry["states"][0]:
                # Separations stop at half the box, so a length sitting on this line
                # is only a lower bound, not a measurement.
                ax.plot(values, [s["velocity_correlation_bound"] for s in entry["states"]],
                        color=plotstyle.INK_MUTED, lw=0.8, ls=":")
        ax.set_title(title, loc="left", pad=6, fontsize=9)
        ax.set_xlabel(label)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        if key == "msd_exponent":
            # 1 is diffusive, 2 ballistic -- the reference lines make the panel readable
            for level, style in ((1.0, ":"), (2.0, ":")):
                ax.axhline(level, color=plotstyle.INK_MUTED, lw=0.8, ls=style)
        if key == "occupancy":
            ax.axhline(1.4, color=plotstyle.SERIES[1], lw=0.8, ls=":")
        if index == 0 and not single:
            ax.legend(loc="best")

    for spare in range(len(reported), rows * columns):
        axes[spare // columns][spare % columns].set_visible(False)

    fig.suptitle(f"Tissue observables against {label}", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold", color=plotstyle.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def sweep_figure(results, label, log_x=False, tolerance=1e-3):
    """A row of final states, with convergence and trends underneath.

    ``results`` is a sequence of ``(value, trajectory)``.
    """
    plotstyle.use_style()
    n = len(results)
    # At least two columns, so the bottom row can always be split in two -- with a
    # single result the right-hand slice would otherwise be empty and refuse to draw.
    columns = max(n, 2)
    fig = plt.figure(figsize=(3.3 * columns, 7.4))
    cells = fig.add_gridspec(2, columns, height_ratios=[1.35, 1.0], hspace=0.34,
                             wspace=0.18)

    for column, (value, trajectory) in enumerate(results):
        ax = fig.add_subplot(cells[0, column])
        tissue = trajectory.tissue
        field_image(ax, tissue.fields.occupancy, tissue.grid)
        outlines(ax, tissue)
        final = trajectory.final
        ax.set_title(
            f"{label} = {value:g}\n"
            f"$A/A_0$ {final.area_ratio:.3f}   $q$ {final.shape_index:.2f}   "
            f"{'settled' if trajectory.settled(tolerance) else 'STILL MOVING'}",
            loc="left", pad=8, fontsize=9,
        )

    split = max(1, columns // 2)
    ax = fig.add_subplot(cells[1, :split])
    for index, (value, trajectory) in enumerate(results):
        energy_axes(ax, trajectory, colour=plotstyle.SERIES[index % 4],
                    label=f"{label} = {value:g}", relative=True)
    ax.set_title("Approach to steady state", loc="left", pad=8)
    ax.legend(loc="upper right")

    ax = fig.add_subplot(cells[1, split:])
    values = [value for value, _ in results]
    for index, (key, name) in enumerate((("area_ratio", "$A/A_0$"),
                                         ("confluence", "confluence error"),
                                         ("occupancy", "max $\\sum_i\\phi_i$"))):
        ax.plot(values, [getattr(t.final, key) for _, t in results], "o-",
                color=plotstyle.SERIES[index % 4], ms=5, label=name)
    ax.axhline(1.0, color=plotstyle.INK_MUTED, lw=0.8, ls=":")
    ax.set_xlabel(label)
    ax.set_title("Steady-state measures", loc="left", pad=8)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    if log_x:
        ax.set_xscale("log")

    model = results[0][1].model
    fig.suptitle(
        f"Steady states, sweeping {label}   |   "
        f"$\\alpha$={model.alpha:g}  $\\epsilon$={model.epsilon:g}  $K$={model.K:g}  "
        f"$\\lambda$={model.area_lambda:g}  $\\gamma$={model.friction:g}   |   "
        f"width {model.interface_width:.3f}, $\\sigma$ {model.surface_tension:.4f}",
        x=0.01, ha="left", fontsize=11, fontweight="semibold", color=plotstyle.INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def save_animation(anim, stem: str, fps: int = 20, prefer: str = "mp4",
                   directory: Path | str | None = None, dpi: float | None = None) -> Path:
    """Write the animation, as mp4 when ffmpeg is present and gif otherwise.

    ``directory`` defaults to the repository's ``figures/``; pass the run's own output
    directory to keep the video next to its ``.npz``.

    ``dpi`` sets the frame resolution and so, for a gif, the file size almost
    directly. ``None`` takes the style's ``savefig.dpi`` of 200, which is meant for
    print and makes a 200-frame gif of an 85^2 field about 10 MB; 80 is plenty for an
    on-screen preview at a sixth of that.
    """
    directory = plotstyle.FIGURES if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if prefer == "mp4" and animation.writers.is_available("ffmpeg"):
        path = directory / f"{stem}.mp4"
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
        return path
    if prefer == "mp4":
        print("  ffmpeg not found -- writing a gif instead")
    path = directory / f"{stem}.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    return path
