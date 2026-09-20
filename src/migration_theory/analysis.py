"""Observables describing the state of a tissue, measured from a finished run.

Nothing here is a model parameter. These are the quantities you would extract from
imaging -- shape distributions, diffusion coefficients, rearrangement rates,
persistence -- computed from a :class:`~migration_theory.simulate.Trajectory` so that
simulation and experiment can be compared on the same footing.

Everything takes a ``transient`` fraction to discard, because a freshly seeded tissue
is not the tissue you want statistics on: it spends its first few percent relaxing
interfaces and settling shapes, and including that biases every average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulate import Trajectory

__all__ = [
    "Tracks",
    "tracks",
    "shape_statistics",
    "mean_squared_displacement",
    "diffusion_coefficient",
    "persistent_random_walk",
    "neighbour_graph",
    "neighbour_exchange_rate",
    "tissue_state",
]


def _after_transient(trajectory: Trajectory, transient: float) -> int:
    if not 0.0 <= transient < 1.0:
        raise ValueError(f"transient must lie in [0, 1), got {transient}")
    return min(int(transient * len(trajectory.snapshots)), len(trajectory.snapshots) - 2)


# --------------------------------------------------------------------- trajectories


@dataclass(frozen=True)
class Tracks:
    """Cell paths with the periodic boundary undone."""

    times: np.ndarray
    """``(T,)`` sample times."""

    positions: np.ndarray
    """``(T, n_cells, 2)`` cumulative displacement from each cell's starting point."""

    max_step: float
    """Largest single-interval displacement, for checking the unwrapping is sound."""

    @property
    def n_cells(self) -> int:
        return self.positions.shape[1]


def tracks(trajectory: Trajectory) -> Tracks:
    """Unwrap the centre-of-mass paths through the periodic boundary.

    Successive snapshots are joined by the minimum-image step between them, then
    accumulated. That is exact as long as no cell moves more than half a box between
    samples -- which :attr:`Tracks.max_step` lets you verify rather than assume. Sample
    more often if it is ever close.
    """
    centres = np.array([s.centres for s in trajectory.snapshots])
    steps = trajectory.model.box.min_image(np.diff(centres, axis=0))
    positions = np.concatenate(
        [np.zeros((1, *centres.shape[1:])), np.cumsum(steps, axis=0)], axis=0
    )
    return Tracks(
        times=trajectory.times,
        positions=positions,
        max_step=float(np.max(np.hypot(steps[..., 0], steps[..., 1]))) if len(steps) else 0.0,
    )


# --------------------------------------------------------------------- shape


def shape_statistics(trajectory: Trajectory, transient: float = 0.2) -> dict[str, float]:
    """Distribution of the cell shape index ``q = P / sqrt(A)``, pooled over time.

    Reported as a distribution rather than a mean because the spread is the interesting
    part: a jammed tissue has cells that are all alike, a fluid one has a broad tail of
    elongated cells. The mean alone cannot tell those apart.

    Read the absolute values with care at finite interface width -- ``A`` is
    ``INT phi^2``, which undershoots the sharp-interface area and biases ``q`` high (see
    :func:`~migration_theory.diagnostics.shape_indices`). Comparisons between runs at
    the same width are unaffected.
    """
    start = _after_transient(trajectory, transient)
    values = np.concatenate([s.shape_indices for s in trajectory.snapshots[start:]])
    areas = np.concatenate([s.areas for s in trajectory.snapshots[start:]])
    low, median, high = np.percentile(values, [5, 50, 95])
    return {
        "shape_index_mean": float(values.mean()),
        "shape_index_std": float(values.std()),
        "shape_index_median": float(median),
        "shape_index_p5": float(low),
        "shape_index_p95": float(high),
        "area_mean": float(areas.mean()),
        "area_cv": float(areas.std() / areas.mean()),
    }


# --------------------------------------------------------------------- motion


def mean_squared_displacement(
    trajectory: Trajectory, transient: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Time-averaged MSD against lag time.

    Averaged over every pair of samples at a given separation and over all cells, which
    uses the run far more efficiently than measuring from a single origin. Long lags
    have fewer independent origins and so are noisier -- the last quarter or so of the
    curve is usually not worth fitting.

    Returns ``(lags, msd)``.
    """
    path = tracks(trajectory)
    start = _after_transient(trajectory, transient)
    positions = path.positions[start:]
    times = path.times[start:]

    n_samples = len(positions)
    lags = np.arange(1, n_samples)
    values = np.empty(len(lags))
    for index, lag in enumerate(lags):
        difference = positions[lag:] - positions[:-lag]
        values[index] = np.mean((difference**2).sum(axis=-1))
    return times[lags] - times[0], values


def diffusion_coefficient(
    trajectory: Trajectory, transient: float = 0.2, fit_range: tuple[float, float] = (0.4, 0.8)
) -> dict[str, float]:
    """Long-time diffusion coefficient from ``MSD = 4 D t`` in two dimensions.

    Fitted over the middle-to-late part of the curve (``fit_range`` as fractions of the
    longest lag), avoiding both the short-time ballistic regime and the noisy tail.

    Also returns the MSD exponent from a log-log fit over the same window: ~1 means
    diffusive, ~2 ballistic, below 1 subdiffusive -- which is what a caged cell in a
    jammed tissue looks like, and the clearest single indicator that the tissue is a
    solid.
    """
    lags, msd = mean_squared_displacement(trajectory, transient)
    if len(lags) < 4:
        raise ValueError("too few snapshots to fit a diffusion coefficient")

    low, high = (np.array(fit_range) * lags[-1])
    window = (lags >= low) & (lags <= high) & (msd > 0)
    if window.sum() < 3:
        window = msd > 0

    slope = np.polyfit(lags[window], msd[window], 1)[0]
    exponent = np.polyfit(np.log(lags[window]), np.log(msd[window]), 1)[0]
    return {
        "diffusion_coefficient": float(slope / 4.0),
        "msd_exponent": float(exponent),
        "msd_final": float(msd[-1]),
    }


def persistent_random_walk(
    trajectory: Trajectory, transient: float = 0.2
) -> dict[str, float]:
    r"""Fit a persistent random walk to the measured MSD.

    .. math::

        \mathrm{MSD}(t) = 2 v^2 \tau^2 \left[ t/\tau - 1 + e^{-t/\tau} \right]

    which is ballistic (``v^2 t^2``) below the persistence time and diffusive
    (``2 v^2 \tau t``) above it. Fitting recovers an *effective* speed and persistence
    from the motion the cell actually achieved.

    Those are worth comparing against the model's inputs. A cell is given speed ``v0``
    and a polarity persistence ``1/D_r``, but in a crowded tissue it is caged: the
    measured speed comes out below ``v0`` and the measured persistence below ``1/D_r``,
    because the neighbours interrupt the motion long before the polarity forgets its
    direction. The ratio of measured to imposed is a direct read on how confined the
    cells are.
    """
    lags, msd = mean_squared_displacement(trajectory, transient)
    usable = lags > 0
    lags, msd = lags[usable], msd[usable]
    if len(lags) < 4:
        raise ValueError("too few snapshots to fit a persistent random walk")

    def model(tau: float) -> tuple[float, float]:
        """Best speed for a given tau, and the residual, by linear least squares."""
        shape = 2.0 * tau**2 * (lags / tau - 1.0 + np.exp(-lags / tau))
        denominator = float((shape * shape).sum())
        if denominator <= 0:
            return 0.0, np.inf
        v_squared = max(float((shape * msd).sum()) / denominator, 0.0)
        return v_squared, float(((v_squared * shape - msd) ** 2).sum())

    # One nonlinear parameter, so scan it rather than risk a gradient method stalling.
    candidates = np.geomspace(lags[0] / 5.0, lags[-1] * 20.0, 400)
    residuals = [model(tau)[1] for tau in candidates]
    tau = float(candidates[int(np.argmin(residuals))])
    v_squared = model(tau)[0]
    speed = float(np.sqrt(v_squared))

    imposed_speed = trajectory.model.speed
    imposed_time = trajectory.model.persistence_time
    return {
        "measured_speed": speed,
        "measured_persistence_time": tau,
        "measured_persistence_length": speed * tau,
        "speed_ratio": speed / imposed_speed if imposed_speed else float("nan"),
        "persistence_ratio": tau / imposed_time if np.isfinite(imposed_time) else float("nan"),
    }


# --------------------------------------------------------------------- topology


def neighbour_graph(snapshot, threshold: float = 0.05) -> np.ndarray:
    """``(n_cells, n_cells)`` boolean adjacency from the contact overlaps.

    Two cells count as neighbours when their overlap exceeds ``threshold`` times the
    largest overlap present. A relative threshold rather than an absolute one, because
    the scale of the overlap depends on the interface width and the cell size, neither
    of which the caller should have to think about here.
    """
    contacts = snapshot.contacts
    peak = contacts.max()
    if peak <= 0:
        return np.zeros_like(contacts, dtype=bool)
    adjacency = contacts > threshold * peak
    np.fill_diagonal(adjacency, False)
    return adjacency


def neighbour_exchange_rate(
    trajectory: Trajectory, transient: float = 0.2, threshold: float = 0.05
) -> dict[str, float]:
    """Rate at which cells swap neighbours -- the T1 rate.

    Counts adjacency changes between consecutive samples, halved because every exchange
    is seen by both cells, and divided by cell count and elapsed time.

    This is the sharpest distinction between a solid and a fluid tissue. A jammed tissue
    gives essentially zero: cells jostle but keep the same neighbours forever. Any
    non-zero rate means the tissue is flowing.

    It is a *lower bound* when sampling is coarse -- an exchange that happens and
    reverses between two samples is invisible -- so compare rates only between runs
    sampled at the same interval.
    """
    start = _after_transient(trajectory, transient)
    snapshots = trajectory.snapshots[start:]
    if len(snapshots) < 2:
        raise ValueError("too few snapshots to measure neighbour exchange")

    graphs = [neighbour_graph(s, threshold) for s in snapshots]
    changes = sum(int(np.triu(a ^ b, 1).sum()) for a, b in zip(graphs, graphs[1:]))
    span = snapshots[-1].time - snapshots[0].time
    n_cells = trajectory.model.n_cells
    coordination = float(np.mean([g.sum(axis=1).mean() for g in graphs]))
    return {
        "neighbour_changes": float(changes),
        "exchange_rate_per_cell": changes / (n_cells * span) if span > 0 else float("nan"),
        "mean_coordination": coordination,
        "sampling_interval": span / (len(snapshots) - 1),
    }


# --------------------------------------------------------------------- everything


def tissue_state(trajectory: Trajectory, transient: float = 0.2) -> dict[str, float]:
    """Every observable in this module, as one flat dictionary.

    The set of numbers that describes what kind of tissue this run produced, as opposed
    to what was put into it.
    """
    state: dict[str, float] = {}
    state.update(shape_statistics(trajectory, transient))
    state.update(diffusion_coefficient(trajectory, transient))
    state.update(persistent_random_walk(trajectory, transient))
    state.update(neighbour_exchange_rate(trajectory, transient))
    path = tracks(trajectory)
    state["max_unwrap_step"] = path.max_step
    state["confluence"] = trajectory.final.confluence
    state["occupancy"] = trajectory.final.occupancy
    return state
