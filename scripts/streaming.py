"""Dissect the streaming in saved runs: what moves together, over what distance, and how.

Nothing is simulated; this reads ``.npz`` files. For each run, after the transient and
with the whole-tissue drift removed (see :func:`migration_theory.analysis.velocities`):

* the full spatial velocity correlation ``C(r)`` at every lag in
  :data:`~migration_theory.analysis.CORRELATION_LAGS`, not just the ``1/e`` crossing.
  Its shape is the mechanism: a curve that grows with the lag is slow collective
  motion under fast jostling; one that does not is what it is at every timescale.
* the longitudinal and transverse parts ``C_par(r)`` and ``C_perp(r)`` at one lag --
  velocities projected along and across the pair separation. Incompressible flow has
  to close its streams into swirls, which shows as ``C_perp`` dipping *negative* at a
  few spacings: cells across a stream move the opposite way. A tissue with free space
  need not, and ``C_perp`` stays positive.
* the velocity field of one interval as arrows on the cell positions, coloured by
  direction, so a stream is a patch of one colour. The direct picture of what the
  eye picks out of an animation, minus the drift.
* stream clusters: connected groups of neighbours whose velocities lie within 45
  degrees of each other. Their mean size in cells, over the run, is the number the
  animation is suggesting when it shows streams.

Usage::

    python scripts/streaming.py RUN1.npz RUN2.npz ... --label-by rotational_diffusion \\
        --lag 20 --transient 0.3 --out figures/streaming.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotstyle  # noqa: E402
from migration_theory import load_trajectory  # noqa: E402
from migration_theory.analysis import (  # noqa: E402
    CORRELATION_LAGS,
    _after_transient,
    _decay_length,
    drift_speed_ratio,
    neighbour_graph,
    velocities,
    velocity_correlation,
    velocity_correlation_split,
)

ALIGNMENT = np.cos(np.pi / 4)      # neighbours within 45 degrees stream together


def stream_clusters(trajectory, transient, lag, every=5, threshold=0.05):
    """Mean and largest size of groups of touching cells moving within 45 degrees of
    each other, over every ``every``-th interval after the transient."""
    _, velocity, _ = velocities(trajectory, transient, lag)
    start = _after_transient(trajectory, transient)
    sizes, largest = [], 0
    for k in range(0, len(velocity), every):
        snapshot = trajectory.snapshots[start + k]
        speed = np.linalg.norm(velocity[k], axis=1)
        direction = velocity[k] / np.maximum(speed, 1e-12)[:, None]
        aligned = direction @ direction.T > ALIGNMENT
        graph = neighbour_graph(snapshot, threshold) & aligned
        _, labels = connected_components(graph, directed=False)
        counts = np.bincount(labels)
        sizes.extend(counts)
        largest = max(largest, int(counts.max()))
    return float(np.mean(sizes)), largest


def velocity_frame(trajectory, transient, lag):
    """Positions and drift-subtracted velocities at the interval in the middle of the
    measured part of the run, with the snapshot it starts from."""
    _, velocity, positions = velocities(trajectory, transient, lag)
    k = len(velocity) // 2
    start = _after_transient(trajectory, transient)
    return positions[k], velocity[k], trajectory.snapshots[start + k]


def label_for(trajectory, field):
    value = getattr(trajectory.model, field)
    return f"{field} = {value:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--label-by", default="rotational_diffusion", dest="label_by",
                        help="the model field that names each row")
    parser.add_argument("--lag", type=int, default=20,
                        help="lag, in sample intervals, for the split correlation, the "
                             "arrows and the clusters (20 = 500 s at the sweeps' sampling)")
    parser.add_argument("--transient", type=float, default=0.3)
    parser.add_argument("--out", type=Path, required=True, help="the figure to write (.png)")
    args = parser.parse_args()

    plotstyle.use_style()
    rows = len(args.files)
    fig, axes = plt.subplots(rows, 3, figsize=(11.5, 3.4 * rows), squeeze=False,
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
    ramp = plotstyle.FIELD_CMAP(np.linspace(0.35, 1.0, len(CORRELATION_LAGS)))

    print("lengths in cell spacings; 'zero' is where C(r) first turns negative, 'perp zero' where C_perp does\n")
    print(f"{'run':>24} {'drift/speed':>12} " + " ".join(f"{'len lag ' + str(l):>11}" for l in CORRELATION_LAGS)
          + f" {'zero lag ' + str(args.lag):>12} {'min C_perp':>11} {'at r/a':>7} {'perp zero':>10} {'cluster':>8} {'largest':>8}")
    for row, path in enumerate(args.files):
        trajectory = load_trajectory(path)
        model = trajectory.model
        spacing = model.cell_spacing
        label = label_for(trajectory, args.label_by)
        interval = float(np.mean(np.diff(trajectory.times)))
        ax_c, ax_split, ax_map = axes[row]

        # --- C(r) at every lag ---------------------------------------------------
        lengths, zero = [], float("nan")
        available = len(trajectory.snapshots) - _after_transient(trajectory, args.transient)
        r_max = 0.5 * model.box.min_length
        for colour, lag in zip(ramp, CORRELATION_LAGS):
            if lag >= available:
                lengths.append(float("nan"))
                continue
            curve = velocity_correlation(trajectory, args.transient, lag=lag)
            lengths.append(curve["correlation_length"] / spacing)
            if lag == args.lag:
                zero = _decay_length(curve["separation"], curve["correlation"], 0.0, r_max)[0] / spacing
            ax_c.plot(curve["separation"] / spacing, curve["correlation"], "-o", ms=3,
                      color=colour, label=f"lag {lag} ({lag * interval:.0f} {model.time_unit})")
        ax_c.axhline(1 / np.e, color=plotstyle.INK_MUTED, lw=0.8, ls=":")
        ax_c.axhline(0.0, color=plotstyle.INK_MUTED, lw=0.8)
        ax_c.set_xlabel("separation / cell spacing")
        ax_c.set_ylabel("C(r)")
        ax_c.set_title(f"{label}   velocity correlation", loc="left", pad=8)
        ax_c.legend(fontsize=7, frameon=False)

        # --- longitudinal and transverse -----------------------------------------
        split = velocity_correlation_split(trajectory, args.transient, args.lag)
        r, c_par, c_perp = split["separation"], split["parallel"], split["perpendicular"]
        ax_split.plot(r / spacing, c_par, "-o", ms=3, color=plotstyle.SERIES[0],
                      label="along separation  $C_\\parallel$")
        ax_split.plot(r / spacing, c_perp, "-o", ms=3, color=plotstyle.SERIES[1],
                      label="across separation  $C_\\perp$")
        ax_split.axhline(0.0, color=plotstyle.INK_MUTED, lw=0.8)
        ax_split.set_xlabel("separation / cell spacing")
        ax_split.set_title(f"lag {args.lag}: along vs across", loc="left", pad=8)
        ax_split.legend(fontsize=7, frameon=False)

        # --- the velocity field ---------------------------------------------------
        positions, velocity, snapshot = velocity_frame(trajectory, args.transient, args.lag)
        plotstyle.field_axes(ax_map, model.grid)
        speed = np.linalg.norm(velocity, axis=1)
        angle = np.arctan2(velocity[:, 1], velocity[:, 0])
        scale = max(speed.mean(), 1e-12) / (0.8 * spacing)            # mean arrow = 0.8 spacing
        ax_map.quiver(positions[:, 0], positions[:, 1], velocity[:, 0], velocity[:, 1],
                      angle, cmap="hsv", clim=(-np.pi, np.pi), scale=scale, scale_units="xy",
                      angles="xy", width=0.006, pivot="mid")
        ax_map.plot(positions[:, 0], positions[:, 1], ".", color=plotstyle.INK_MUTED, ms=2)
        ax_map.set_title(f"$t$ = {snapshot.time:.0f}, drift removed", loc="left", pad=8)

        # --- numbers --------------------------------------------------------------
        drift = drift_speed_ratio(trajectory, args.transient)
        mean_size, largest = stream_clusters(trajectory, args.transient, args.lag)
        print(f"{label:>24} {drift:12.3f} " + " ".join(f"{x:11.2f}" for x in lengths)
              + f" {zero:12.2f} {split['perpendicular_min']:11.3f}"
              + f" {split['perpendicular_min_separation'] / spacing:7.2f}"
              + f" {split['perpendicular_zero_crossing'] / spacing:10.2f} {mean_size:8.2f} {largest:8d}")

    fig.suptitle("Streaming: how far velocities are shared, and in what pattern",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
