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
    "sweep_figure",
    "save_animation",
]


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


def observables_figure(values, states, label, reported):
    """Each observable against the swept parameter, one small panel apiece.

    Small multiples rather than shared axes: the quantities differ by orders of
    magnitude -- a T1 rate near zero beside a shape index near 4 -- and forcing them
    onto one pair of axes would flatten whichever is smaller into the baseline.
    """
    plotstyle.use_style()
    columns = 4
    rows = int(np.ceil(len(reported) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.7 * rows),
                             squeeze=False)

    for index, (key, title) in enumerate(reported):
        ax = axes[index // columns][index % columns]
        series = [state[key] for state in states]
        ax.plot(values, series, "o-", color=plotstyle.SERIES[index % 4], ms=5)
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
    fig = plt.figure(figsize=(3.3 * n, 7.4))
    cells = fig.add_gridspec(2, n, height_ratios=[1.35, 1.0], hspace=0.34, wspace=0.18)

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

    ax = fig.add_subplot(cells[1, : max(1, n // 2)])
    for index, (value, trajectory) in enumerate(results):
        energy_axes(ax, trajectory, colour=plotstyle.SERIES[index % 4],
                    label=f"{label} = {value:g}", relative=True)
    ax.set_title("Approach to steady state", loc="left", pad=8)
    ax.legend(loc="upper right")

    ax = fig.add_subplot(cells[1, max(1, n // 2):])
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


def save_animation(anim, stem: str, fps: int = 20, prefer: str = "mp4") -> Path:
    """Write to ``figures/``, as mp4 when ffmpeg is present and gif otherwise."""
    plotstyle.FIGURES.mkdir(exist_ok=True)
    if prefer == "mp4" and animation.writers.is_available("ffmpeg"):
        path = plotstyle.FIGURES / f"{stem}.mp4"
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
        return path
    if prefer == "mp4":
        print("  ffmpeg not found -- writing a gif instead")
    path = plotstyle.FIGURES / f"{stem}.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    return path
