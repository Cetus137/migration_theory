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

from .grid import magnitude
from .simulate import Trajectory


def _norm(vectors: np.ndarray) -> np.ndarray:
    """``|v|`` over the last axis -- ``np.hypot`` in 2D, as this module always used."""
    return magnitude(np.moveaxis(vectors, -1, 0))

__all__ = [
    "Tracks",
    "tracks",
    "shape_statistics",
    "mean_squared_displacement",
    "diffusion_coefficient",
    "persistent_random_walk",
    "neighbour_graph",
    "neighbour_exchange_rate",
    "neighbour_persistence",
    "velocities",
    "drift_speed_ratio",
    "velocity_correlation",
    "velocity_correlation_split",
    "velocity_correlations",
    "cage_relative_motion",
    "population_state",
    "minority_state",
    "pair_correlation",
    "structure",
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
        max_step=float(np.max(_norm(steps))) if len(steps) else 0.0,
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
    return _msd_of(path.positions[start:], path.times[start:])


def _msd_of(positions: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Time-averaged MSD of an unwrapped ``(T, cells, ndim)`` path, against lag time."""
    n_samples = len(positions)
    lags = np.arange(1, n_samples)
    values = np.empty(len(lags))
    for index, lag in enumerate(lags):
        difference = positions[lag:] - positions[:-lag]
        values[index] = np.mean((difference**2).sum(axis=-1))
    return times[lags] - times[0], values


def _fit_msd(lags: np.ndarray, msd: np.ndarray, ndim: int,
             fit_range: tuple[float, float] = (0.4, 0.8)) -> tuple[float, float]:
    """``(D, exponent)`` from the middle-to-late part of an MSD curve, as
    :func:`diffusion_coefficient` fits them."""
    low, high = (np.array(fit_range) * lags[-1])
    window = (lags >= low) & (lags <= high) & (msd > 0)
    if window.sum() < 3:
        window = msd > 0
    if window.sum() < 2:
        return float("nan"), float("nan")
    slope = np.polyfit(lags[window], msd[window], 1)[0]
    exponent = np.polyfit(np.log(lags[window]), np.log(msd[window]), 1)[0]
    return float(slope / (2.0 * ndim)), float(exponent)


def diffusion_coefficient(
    trajectory: Trajectory, transient: float = 0.2, fit_range: tuple[float, float] = (0.4, 0.8)
) -> dict[str, float]:
    """Long-time diffusion coefficient from ``MSD = 2 d D t`` -- ``4 D t`` in two dimensions.

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
        "diffusion_coefficient": float(slope / (2.0 * trajectory.model.box.ndim)),
        "msd_exponent": float(exponent),
        "msd_final": float(msd[-1]),
    }


def persistent_random_walk(
    trajectory: Trajectory, transient: float = 0.2, fit_fraction: float = 0.5
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

    **How the fit is weighted matters.** The MSD grows with lag, so a plain sum of
    squared residuals is dominated by the longest lags -- which have the fewest
    independent origins and are the noisiest -- while the short lags that carry the
    persistence time count for almost nothing. Measured on four free cells, that fit
    ran the persistence time into the edge of its scan and returned a speed 2.6 times
    the imposed one. So the residual here is *relative*, every lag counting equally,
    and only lags up to ``fit_fraction`` of the longest are used. A fit that still
    lands on the edge of the scan is reported with ``persistence_fit_at_bound = 1``
    rather than silently returned.
    """
    lags, msd = mean_squared_displacement(trajectory, transient)
    usable = (lags > 0) & (msd > 0) & (lags <= fit_fraction * lags[-1])
    lags, msd = lags[usable], msd[usable]
    if len(lags) < 4:
        raise ValueError("too few snapshots to fit a persistent random walk")

    def model(tau: float) -> tuple[float, float]:
        """Best speed for a given tau, and the relative residual, in closed form.

        Minimising ``sum_k (v^2 s_k / m_k - 1)^2`` over ``v^2`` is linear: with
        ``r_k = s_k / m_k`` the optimum is ``v^2 = sum r_k / sum r_k^2``.
        """
        shape = 2.0 * tau**2 * (lags / tau - 1.0 + np.exp(-lags / tau))
        ratio = shape / msd
        denominator = float((ratio * ratio).sum())
        if denominator <= 0:
            return 0.0, np.inf
        v_squared = max(float(ratio.sum()) / denominator, 0.0)
        return v_squared, float(((v_squared * ratio - 1.0) ** 2).sum())

    # One nonlinear parameter, so scan it rather than risk a gradient method stalling.
    candidates = np.geomspace(lags[0] / 5.0, lags[-1] * 20.0, 400)
    residuals = [model(tau)[1] for tau in candidates]
    best = int(np.argmin(residuals))
    tau = float(candidates[best])
    v_squared = model(tau)[0]
    speed = float(np.sqrt(v_squared))

    imposed_speed = trajectory.model.free_speed
    imposed_time = trajectory.model.persistence_time
    return {
        "measured_speed": speed,
        "measured_persistence_time": tau,
        "measured_persistence_length": speed * tau,
        "speed_ratio": speed / imposed_speed if imposed_speed else float("nan"),
        "persistence_ratio": tau / imposed_time if np.isfinite(imposed_time) else float("nan"),
        "persistence_fit_at_bound": float(best in (0, len(candidates) - 1)),
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

    **Two normalised rates come with it.** A rate per second cannot be compared
    between runs whose cells move at different speeds: the energy scale cancels out
    of the dynamics, so a tissue with every coefficient doubled runs the identical
    trajectory twice as fast and shows twice the rate. ``exchanges_per_radius`` is the
    rate times ``R / v_free``, exchanges per cell radius a free cell would have
    crawled -- how much of the intended motion becomes rearrangement -- and it is the
    quantity that nearly collapsed across a decade of tension where the raw rate
    spread eighteenfold. ``exchanges_per_persistence_time`` is the rate times
    ``1 / D_r``, the natural unit along a persistence axis. Both are ``NaN`` for a
    passive tissue, which has no speed and no persistence to normalise by.
    """
    start = _after_transient(trajectory, transient)
    snapshots = trajectory.snapshots[start:]
    if len(snapshots) < 2:
        raise ValueError("too few snapshots to measure neighbour exchange")

    graphs = [neighbour_graph(s, threshold) for s in snapshots]
    changes = sum(int(np.triu(a ^ b, 1).sum()) for a, b in zip(graphs, graphs[1:]))
    span = snapshots[-1].time - snapshots[0].time
    model = trajectory.model
    rate = changes / (model.n_cells * span) if span > 0 else float("nan")
    coordination = float(np.mean([g.sum(axis=1).mean() for g in graphs]))
    speed, persistence = model.free_speed, model.persistence_time
    return {
        "neighbour_changes": float(changes),
        "exchange_rate_per_cell": rate,
        "exchanges_per_radius": rate * model.cell_radius / speed if speed > 0 else float("nan"),
        "exchanges_per_persistence_time": (
            rate * persistence if speed > 0 and np.isfinite(persistence) else float("nan")
        ),
        "mean_coordination": coordination,
        "sampling_interval": span / (len(snapshots) - 1),
    }


def neighbour_persistence(
    trajectory: Trajectory, transient: float = 0.2, threshold: float = 0.05
) -> dict[str, float | np.ndarray]:
    r"""How long a cell keeps its neighbours.

    .. math::

        Q(t) = \frac{\sum_{t_0}\sum_i |N_i(t_0) \cap N_i(t_0 + t)|}
                    {\sum_{t_0}\sum_i |N_i(t_0)|}

    the fraction of the neighbours a cell has at one moment that it still has a time
    ``t`` later, pooled over every starting moment after the transient and every cell.
    ``Q(0) = 1``; in a jammed tissue it stays there, since cells keep their neighbours
    for ever, and in a fluid it decays, on the timescale over which a cell's
    surroundings are renewed. That timescale -- ``neighbour_persistence_time``, where
    ``Q`` first falls to a half, interpolated -- is the cage lifetime of the glass
    literature, and the single number that says how quickly a cell meets new cells.

    It is the right measure where the T1 rate is not. The T1 rate counts every change
    of the contact graph between consecutive samples, so a contact flickering at a gap
    in a sub-confluent tissue, or two cells brushing past, both register. Here a
    neighbour counts as kept or lost over a *span*, so flicker averages out, and a
    brief encounter weighs exactly what it should: little.

    If ``Q`` has not reached a half by the longest lag the run allows, the time is
    that lag and ``censored`` is set -- a lower bound, the signature of a solid.
    ``final`` is ``Q`` at that longest lag, the fraction of original neighbours a cell
    still has at the end, which is informative either way.

    Returns ``lags`` and ``Q`` (the curve), ``neighbour_persistence_time``,
    ``censored`` and ``final``.
    """
    start = _after_transient(trajectory, transient)
    snapshots = trajectory.snapshots[start:]
    if len(snapshots) < 3:
        raise ValueError("too few snapshots to measure neighbour persistence")
    graphs = np.array([neighbour_graph(s, threshold) for s in snapshots])   # (T, n, n)
    times = np.array([s.time for s in snapshots])

    # Use lags up to half the record: beyond that too few starting moments remain.
    max_lag = max(1, (len(snapshots) - 1) // 2)
    lags = np.arange(1, max_lag + 1)
    kept = np.empty(len(lags))
    for k, lag in enumerate(lags):
        had = graphs[:-lag].sum()
        kept[k] = (graphs[:-lag] & graphs[lag:]).sum() / had if had else float("nan")
    curve_t = np.concatenate([[0.0], times[lags] - times[0]])
    curve_q = np.concatenate([[1.0], kept])

    persistence, censored = _decay_length(curve_t, curve_q, 0.5, float(curve_t[-1]))
    return {
        "lags": curve_t,
        "Q": curve_q,
        "neighbour_persistence_time": persistence,
        "censored": censored,
        "final": float(curve_q[-1]),
    }


# --------------------------------------------------------------------- velocities


def velocities(
    trajectory: Trajectory, transient: float = 0.2, lag: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell velocities over ``lag`` samples, with the tissue's drift removed.

    Under force balance the passive forces sum to zero, so the net active force on the
    tissue does not: the whole tissue drifts at the mean of the polarities times the
    free speed, about ``1/sqrt(N)`` of it for random polarities. With 16 cells that is a
    quarter of the free speed, and it is motion of the box's contents as a body rather
    than of any cell relative to the tissue. Every correlation here is therefore
    computed after subtracting the mean velocity over cells at each interval; see
    :func:`drift_speed_ratio` for how large that drift was.

    **The lag is part of the measurement.** A velocity over one sample interval is
    dominated by a cell's jostling inside its cage, fast and only weakly shared with
    its neighbours. Over ten or twenty intervals the jostling averages out and the slow
    collective motion -- the streaming the eye picks out of an animation -- is what
    remains. Correlation lengths therefore grow with the lag, as they do in tracked
    monolayers, and a length is only meaningful with its lag stated.

    Returns ``(times, velocities, positions)``: the midpoint time of each interval
    ``(T-lag,)``, the drift-subtracted velocity of every cell ``(T-lag, n_cells, 2)``,
    and the wrapped position of every cell at the start of the interval, for
    separations.
    """
    if lag < 1:
        raise ValueError(f"lag must be at least 1 sample, got {lag}")
    path = tracks(trajectory)
    start = _after_transient(trajectory, transient)
    positions, times = path.positions[start:], path.times[start:]
    if len(times) <= lag:
        raise ValueError(f"too few snapshots ({len(times)}) for a lag of {lag}")
    interval = (times[lag:] - times[:-lag])[:, None, None]
    velocity = (positions[lag:] - positions[:-lag]) / interval
    velocity = velocity - velocity.mean(axis=1, keepdims=True)
    wrapped = np.array([s.centres for s in trajectory.snapshots[start:-lag]])
    return 0.5 * (times[lag:] + times[:-lag]), velocity, wrapped


def drift_speed_ratio(trajectory: Trajectory, transient: float = 0.2) -> float:
    """How fast the tissue moves as a body, relative to how fast its cells move.

    The mean over intervals of the centre-of-mass speed, divided by the mean cell
    speed, both over one sample interval. About ``1/sqrt(N)`` for independent random
    polarities in a periodic box; it is what :func:`velocities` subtracts, reported so
    that whole-tissue motion is a number in the table rather than an invisible
    correction. In an animation it is the everything-sliding-together component of
    what looks like streaming.
    """
    path = tracks(trajectory)
    start = _after_transient(trajectory, transient)
    positions, times = path.positions[start:], path.times[start:]
    if len(times) < 2:
        return float("nan")
    velocity = np.diff(positions, axis=0) / np.diff(times)[:, None, None]
    drift = _norm(velocity.mean(axis=1)).mean()
    cells = _norm(velocity).mean()
    return float(drift / cells) if cells > 0 else float("nan")


def velocity_correlation(
    trajectory: Trajectory,
    transient: float = 0.2,
    bin_width: float | None = None,
    lag: int = 1,
) -> dict[str, np.ndarray | float | bool]:
    r"""Spatial velocity correlation ``C(r)`` and the length over which it decays.

    .. math::

        C(r) = \frac{\langle \mathbf{v}_i \cdot \mathbf{v}_j \rangle_{|r_{ij}| \approx r}}
                    {\langle |\mathbf{v}|^2 \rangle}

    over all pairs of cells at all sampled intervals, binned by separation, with the
    velocities taken over ``lag`` samples (see :func:`velocities` for why that
    matters). ``C(0)`` is 1 by construction; the correlation length is the separation
    at which ``C`` first falls below ``1/e``, interpolated between bins.

    This is the standard measure of collective motion. The model has no alignment
    term of any kind, so whatever correlation appears is purely mechanical: neighbours
    move together because they push on each other through force balance, and the
    length says how far that transmission carries.

    **Box size limits it.** Separations only go up to half the box, which at 16 cells
    is under two cell spacings. If ``C`` has not fallen to ``1/e`` by then the length
    cannot be measured, and the result reports the half box as a lower bound with
    ``censored`` set. A correlation length worth quoting needs a box several times
    longer than it.

    Returns a dict with ``separation`` and ``correlation`` (the curve, starting at
    ``r = 0``), ``counts`` (pairs per bin, for judging noise), ``correlation_length``
    and ``censored``.
    """
    _, velocity, positions = velocities(trajectory, transient, lag)
    model = trajectory.model
    box = model.box
    n_cells = velocity.shape[1]
    i, j = np.triu_indices(n_cells, 1)

    separation = box.min_image(positions[:, i] - positions[:, j])
    distance = _norm(separation).ravel()
    products = (velocity[:, i] * velocity[:, j]).sum(axis=-1).ravel()
    normalisation = float((velocity**2).sum(axis=-1).mean())

    r_max = 0.5 * box.min_length
    width = 0.5 * model.cell_spacing if bin_width is None else bin_width
    edges = np.linspace(0.0, r_max, max(2, int(np.ceil(r_max / width)) + 1))
    counts, _ = np.histogram(distance, edges)
    sums, _ = np.histogram(distance, edges, weights=products)
    populated = counts > 0

    centres = np.concatenate([[0.0], 0.5 * (edges[1:] + edges[:-1])[populated]])
    if normalisation > 0:
        correlation = np.concatenate([[1.0], sums[populated] / counts[populated] / normalisation])
    else:
        correlation = np.full(len(centres), np.nan)
    counts = np.concatenate([[n_cells], counts[populated]])

    length, censored = _decay_length(centres, correlation, 1.0 / np.e, r_max)
    return {
        "separation": centres,
        "correlation": correlation,
        "counts": counts,
        "correlation_length": length,
        "censored": censored,
    }


def _decay_length(r: np.ndarray, c: np.ndarray, level: float, r_max: float) -> tuple[float, bool]:
    """Where ``c`` first crosses below ``level``, interpolated; ``r_max`` if never."""
    if np.any(np.isnan(c)):
        return float("nan"), True
    below = np.nonzero(c < level)[0]
    if len(below) == 0:
        return float(r_max), True
    k = int(below[0])
    r0, r1, c0, c1 = r[k - 1], r[k], c[k - 1], c[k]
    return float(r0 + (c0 - level) * (r1 - r0) / (c0 - c1)), False


#: Lags, in sample intervals, at which the velocity correlations are reported. At the
#: sweeps' 25 s sampling these are 25, 100, 250, 500 and 1000 s: from cage jostling to
#: the slow collective motion over a whole persistence time, ``1 / D_r = 1000 s`` at
#: the sweeps' ``D_r = 1e-3``.
CORRELATION_LAGS = (1, 4, 10, 20, 40)


#: The lag at which the correlation is split along and across the separation. Long
#: enough for cage jostling to average out; 500 s at the sweeps' sampling.
SPLIT_LAG = 20


def velocity_correlation_split(
    trajectory: Trajectory, transient: float = 0.2, lag: int = SPLIT_LAG,
    bin_width: float | None = None,
) -> dict[str, np.ndarray | float | bool]:
    r"""The velocity correlation along and across the pair separation.

    .. math::

        C_\parallel(r) = \frac{\langle (\mathbf{v}_i\cdot\hat{\mathbf{r}}_{ij})
                                       (\mathbf{v}_j\cdot\hat{\mathbf{r}}_{ij}) \rangle}
                              {\tfrac{1}{2}\langle |\mathbf{v}|^2 \rangle}, \qquad
        C_\perp(r) = \frac{\langle (\mathbf{v}_i\cdot\hat{\mathbf{n}}_{ij})
                                   (\mathbf{v}_j\cdot\hat{\mathbf{n}}_{ij}) \rangle}
                          {\tfrac{1}{2}\langle |\mathbf{v}|^2 \rangle}

    with ``n`` the in-plane normal to the separation, each normalised so that both
    are 1 at ``r = 0``. Two-dimensional.

    The split is what tells a mechanism from a number. In a confluent, area-conserving
    tissue a moving cell must displace the cells ahead of it and draw in those behind,
    so ``C_par`` is positive and long-ranged; the flow has to close, so cells *beside*
    a moving chain move the other way and ``C_perp`` turns negative a spacing or two
    out -- the signature of incompressible, swirling motion. Measured (2026-09-22, 50
    to 100 cells, activity 4): ``C_perp`` crosses zero at 1.4 to 1.6 spacings and
    reaches ``-0.2`` to ``-0.3`` at 2.5 to 3, independently of persistence time, lag
    and box size, while the ``1/e`` correlation length of the full ``C(r)`` sat at the
    first shell and measured nothing.

    Returns ``separation``, ``parallel``, ``perpendicular``, ``counts``, plus
    ``perpendicular_min`` and ``perpendicular_min_separation`` (the depth and place of
    the transverse dip) and ``perpendicular_zero_crossing`` (where ``C_perp`` first
    turns negative; the half box, with ``censored``, if it never does).
    """
    _, velocity, positions = velocities(trajectory, transient, lag)
    model = trajectory.model
    box = model.box
    if box.ndim != 2:
        raise NotImplementedError("the along/across split is two-dimensional")
    n_cells = velocity.shape[1]
    i, j = np.triu_indices(n_cells, 1)

    separation = box.min_image(positions[:, i] - positions[:, j])          # (T, pairs, 2)
    distance = _norm(separation)
    unit = separation / np.maximum(distance, 1e-12)[..., None]
    normal = np.stack([-unit[..., 1], unit[..., 0]], axis=-1)
    along = (velocity[:, i] * unit).sum(-1) * (velocity[:, j] * unit).sum(-1)
    across = (velocity[:, i] * normal).sum(-1) * (velocity[:, j] * normal).sum(-1)
    per_component = 0.5 * float((velocity**2).sum(axis=-1).mean())

    r_max = 0.5 * box.min_length
    width = 0.5 * model.cell_spacing if bin_width is None else bin_width
    edges = np.linspace(0.0, r_max, max(2, int(np.ceil(r_max / width)) + 1))
    counts, _ = np.histogram(distance.ravel(), edges)
    parallel, _ = np.histogram(distance.ravel(), edges, weights=along.ravel())
    perpendicular, _ = np.histogram(distance.ravel(), edges, weights=across.ravel())
    populated = counts > 0
    r = 0.5 * (edges[1:] + edges[:-1])[populated]
    if per_component > 0:
        c_par = parallel[populated] / counts[populated] / per_component
        c_perp = perpendicular[populated] / counts[populated] / per_component
    else:
        c_par = c_perp = np.full(len(r), np.nan)

    if len(r) and not np.any(np.isnan(c_perp)):
        dip = int(np.argmin(c_perp))
        crossing, censored = _decay_length(
            np.concatenate([[0.0], r]), np.concatenate([[1.0], c_perp]), 0.0, r_max)
        minimum, at = float(c_perp[dip]), float(r[dip])
    else:
        minimum = at = crossing = float("nan")
        censored = True
    return {
        "separation": r,
        "parallel": c_par,
        "perpendicular": c_perp,
        "counts": counts[populated],
        "perpendicular_min": minimum,
        "perpendicular_min_separation": at,
        "perpendicular_zero_crossing": crossing,
        "censored": censored,
    }


def _neighbour_correlation(trajectory, transient, threshold, lag) -> float:
    """``C`` restricted to pairs in contact at the start of each interval."""
    _, velocity, _ = velocities(trajectory, transient, lag)
    start = _after_transient(trajectory, transient)
    snapshots = trajectory.snapshots[start:-lag]
    total, pairs = 0.0, 0
    for k, snapshot in enumerate(snapshots):
        i, j = np.nonzero(np.triu(neighbour_graph(snapshot, threshold), 1))
        total += float((velocity[k, i] * velocity[k, j]).sum())
        pairs += len(i)
    normalisation = float((velocity**2).sum(axis=-1).mean())
    return total / pairs / normalisation if pairs and normalisation > 0 else float("nan")


def velocity_correlations(
    trajectory: Trajectory,
    transient: float = 0.2,
    threshold: float = 0.05,
    lags: tuple[int, ...] = CORRELATION_LAGS,
) -> dict[str, float]:
    """The velocity-correlation scalars for :func:`tissue_state`, at several lags.

    For each lag ``L`` in ``lags``: ``velocity_correlation_length_lag{L}`` in microns
    from :func:`velocity_correlation`, and ``neighbour_velocity_correlation_lag{L}``,
    ``C`` restricted to pairs actually in contact by :func:`neighbour_graph`, the
    first-shell value and the one that is robust however small the box is. A lag the
    run is too short for gives ``NaN``.

    Also per lag, ``velocity_zero_crossing_lag{L}``: where ``C(r)`` first turns
    negative, in microns. In a confluent tissue ``C`` falls to the first-shell value
    and then to zero over another spacing or two, so the ``1/e`` length can sit on
    either side of the first shell depending on a few percent in that value, while
    the zero crossing moves smoothly. Measured at 2.2 to 2.5 spacings for 50 to 100
    cells at activity 4, with the half box going from 3.3 to 4.7 spacings.

    At :data:`SPLIT_LAG` (or the longest lag the run allows), the along/across split
    of :func:`velocity_correlation_split`: ``transverse_correlation_min`` and
    ``transverse_min_separation``, the depth and place of the negative dip in
    ``C_perp`` that marks closed, incompressible swirls, ``transverse_zero_crossing``
    where ``C_perp`` first turns negative, and ``transverse_lag`` the lag used.

    The un-suffixed ``velocity_correlation_length`` and
    ``neighbour_velocity_correlation`` are the lag-1 values, as before, with
    ``velocity_correlation_censored`` set to 1 when the length is only a lower bound
    and ``velocity_correlation_bound`` the half box it is then bounded by.
    ``drift_speed_ratio`` is the whole-tissue motion the correlations subtract.
    """
    start = _after_transient(trajectory, transient)
    available = len(trajectory.snapshots) - start
    r_max = 0.5 * trajectory.model.box.min_length
    out: dict[str, float] = {}
    for lag in lags:
        if lag >= available:
            length = neighbour = crossing = float("nan")
            censored = 0.0
        else:
            curve = velocity_correlation(trajectory, transient, lag=lag)
            length = float(curve["correlation_length"])
            censored = float(curve["censored"])
            crossing, _ = _decay_length(curve["separation"], curve["correlation"], 0.0, r_max)
            neighbour = _neighbour_correlation(trajectory, transient, threshold, lag)
        out[f"velocity_correlation_length_lag{lag}"] = length
        out[f"neighbour_velocity_correlation_lag{lag}"] = neighbour
        out[f"velocity_zero_crossing_lag{lag}"] = crossing
        if lag == lags[0]:
            out["velocity_correlation_length"] = length
            out["velocity_correlation_censored"] = censored
            out["neighbour_velocity_correlation"] = neighbour
    out["velocity_correlation_bound"] = r_max

    possible = [lag for lag in (SPLIT_LAG, *lags) if lag < available]
    if possible and trajectory.model.box.ndim == 2:
        split_lag = max(possible) if SPLIT_LAG >= available else SPLIT_LAG
        split = velocity_correlation_split(trajectory, transient, split_lag)
        out["transverse_correlation_min"] = split["perpendicular_min"]
        out["transverse_min_separation"] = split["perpendicular_min_separation"]
        out["transverse_zero_crossing"] = split["perpendicular_zero_crossing"]
        out["transverse_lag"] = float(split_lag)
    else:
        out["transverse_correlation_min"] = float("nan")
        out["transverse_min_separation"] = float("nan")
        out["transverse_zero_crossing"] = float("nan")
        out["transverse_lag"] = float("nan")
    out["drift_speed_ratio"] = drift_speed_ratio(trajectory, transient)
    return out


# --------------------------------------------------------------------- structure


def pair_correlation(
    trajectory: Trajectory, transient: float = 0.2, bin_width: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""The radial distribution function ``g(r)`` of the cell centres.

    The density of pairs at separation ``r`` relative to a uniform arrangement, pooled
    over every snapshot after the transient. ``g = 1`` means no structure; a peak means
    cells prefer that separation. In a confluent tissue the first peak sits at one
    cell spacing whatever the phase, because the area constraint puts it there, and
    the phases differ in what follows: a crystal has sharp peaks at the lattice
    distances -- ``1, sqrt(3), 2, sqrt(7)`` spacings for hexagonal order -- a glass
    a broad second peak, a fluid a second peak that fades with activity. Structure
    and dynamics decouple across a jamming transition, so this locates the transition
    poorly and describes the two sides of it well.

    Separations run to half the box, where the minimum-image count is exact; the
    normalisation is the pair count a uniform arrangement would put in each shell.
    Bins are a tenth of a cell spacing unless ``bin_width`` says otherwise.

    Returns ``(r, g)`` with ``r`` the bin centres in microns.
    """
    start = _after_transient(trajectory, transient)
    box = trajectory.model.box
    n_cells = trajectory.model.n_cells
    width = 0.1 * trajectory.model.cell_spacing if bin_width is None else bin_width
    r_max = 0.5 * box.min_length
    edges = np.arange(0.0, r_max + 0.5 * width, width)
    edges = edges[edges <= r_max + 1e-12]

    i, j = np.triu_indices(n_cells, 1)
    counts = np.zeros(len(edges) - 1)
    frames = 0
    for snapshot in trajectory.snapshots[start:]:
        separation = box.min_image(snapshot.centres[i] - snapshot.centres[j])
        counts += np.histogram(_norm(separation), edges)[0]
        frames += 1

    if box.ndim == 2:
        shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    elif box.ndim == 3:
        shell = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    else:
        raise NotImplementedError(f"pair correlation in {box.ndim}D")
    expected = frames * (n_cells * (n_cells - 1) / 2) * shell / box.volume
    return 0.5 * (edges[1:] + edges[:-1]), counts / expected


def structure(trajectory: Trajectory, transient: float = 0.2, threshold: float = 0.05) -> dict[str, float]:
    r"""Positional and orientational order of the cell arrangement.

    ``hexatic_order`` is the mean over cells and snapshots of
    :math:`|\psi_6| = |\tfrac{1}{n}\sum_j e^{6 i \theta_{j}}|`, the angles being those
    to a cell's contact neighbours: 1 for a perfect hexagonal arrangement, small for a
    disordered one. ``hexagon_fraction`` is the share of cells with exactly six
    contact neighbours. Together they say whether a jammed tissue is a crystal or a
    glass -- which the T1 rate cannot, and which changes what a rearrangement means.

    ``g_first_peak`` and ``g_second_peak`` are the heights of :func:`pair_correlation`
    in the windows ``0.5 - 1.5`` and ``1.5 - 2.5`` cell spacings. The second is the
    number that carries the structural change with activity: it falls as positional
    correlation beyond the first shell is lost. Either is ``NaN`` when the box is too
    small to hold its window -- separations only reach half the box, so the second
    shell needs a box of at least five spacings, which 16 cells at confluence gives
    only partly and 4 cells not at all. All four come from positions and neighbour
    relations alone, so they can be measured on segmented images the same way.

    The hexatic order and hexagon fraction are two-dimensional notions and are ``NaN``
    in 3D; the ``g(r)`` peaks are measured in any dimension.
    """
    start = _after_transient(trajectory, transient)
    box = trajectory.model.box
    spacing = trajectory.model.cell_spacing

    r, g = pair_correlation(trajectory, transient)
    x = r / spacing
    first = g[(x > 0.5) & (x < 1.5)]
    second = g[(x >= 1.5) & (x < 2.5)]

    psi6, hexagons = [], []
    for snapshot in trajectory.snapshots[start:] if box.ndim == 2 else []:
        adjacency = neighbour_graph(snapshot, threshold)
        for k in range(trajectory.model.n_cells):
            neighbours = np.flatnonzero(adjacency[k])
            if len(neighbours) == 0:
                continue
            d = box.min_image(snapshot.centres[neighbours] - snapshot.centres[k])
            psi6.append(abs(np.mean(np.exp(6j * np.arctan2(d[:, 1], d[:, 0])))))
        hexagons.append(float(np.mean(adjacency.sum(axis=1) == 6)))

    return {
        "hexatic_order": float(np.mean(psi6)) if psi6 else float("nan"),
        "hexagon_fraction": float(np.mean(hexagons)) if hexagons else float("nan"),
        "g_first_peak": float(first.max()) if len(first) else float("nan"),
        "g_second_peak": float(second.max()) if len(second) else float("nan"),
    }


# --------------------------------------------------------------------- cage-relative motion


def cage_relative_motion(
    trajectory: Trajectory,
    transient: float = 0.2,
    threshold: float = 0.05,
    lags: tuple[int, ...] = CORRELATION_LAGS,
) -> dict[str, float]:
    r"""Does a cell move *through* its neighbours, or *with* them?

    Over each lag ``L``, every cell's displacement ``d_i`` (whole-tissue drift removed)
    is compared with the mean displacement ``d_cage`` of the cells it was in contact
    with at the start of the interval:

    .. math::

        \rho_L = \frac{\langle |d_i - d_{\mathrm{cage}}|^2 \rangle}{\langle |d_i|^2 \rangle},
        \qquad
        \kappa_L = \frac{\langle |d_{\mathrm{cage}}|^2 \rangle}{\langle |d_i|^2 \rangle}

    ``cage_relative_msd_ratio_lag{L}`` is :math:`\rho_L`: zero when a cell and its
    cage move as one -- a stream -- and about 1 when cells move independently (a
    little above 1, since the cage mean then adds its own noise). ``cage_motion_fraction_lag{L}``
    is :math:`\kappa_L`: how much of the cell's displacement its cage shares. The
    un-suffixed keys are the values at :data:`SPLIT_LAG`, or the longest lag the run
    allows. The same numbers can be taken from tracked cells with segmented contacts,
    which is what makes them comparable with the lymph node.

    Distinct from the velocity correlation, which asks whether *pairs* at a distance
    move alike; this asks, cell by cell, how much of its own motion is its own. A
    cell without contacts at the start of an interval is left out; the ratio is
    ``NaN`` if no cell has any.
    """
    path = tracks(trajectory)
    start = _after_transient(trajectory, transient)
    positions = path.positions[start:]
    snapshots = trajectory.snapshots[start:]
    available = len(positions)
    graphs = [neighbour_graph(s, threshold).astype(float) for s in snapshots]
    out: dict[str, float] = {}
    for lag in lags:
        if lag >= available:
            out[f"cage_relative_msd_ratio_lag{lag}"] = float("nan")
            out[f"cage_motion_fraction_lag{lag}"] = float("nan")
            continue
        displacement = positions[lag:] - positions[:-lag]                    # (T-L, n, ndim)
        displacement = displacement - displacement.mean(axis=1, keepdims=True)
        own, relative, cage = 0.0, 0.0, 0.0
        for k, d in enumerate(displacement):
            adjacency = graphs[k]
            degree = adjacency.sum(axis=1)
            has = degree > 0
            if not np.any(has):
                continue
            cage_mean = (adjacency[has] @ d) / degree[has, None]
            own += float((d[has] ** 2).sum())
            relative += float(((d[has] - cage_mean) ** 2).sum())
            cage += float((cage_mean**2).sum())
        ratio = relative / own if own > 0 else float("nan")
        fraction = cage / own if own > 0 else float("nan")
        out[f"cage_relative_msd_ratio_lag{lag}"] = ratio
        out[f"cage_motion_fraction_lag{lag}"] = fraction
    possible = [lag for lag in (SPLIT_LAG, *lags) if lag < available]
    chosen = (SPLIT_LAG if SPLIT_LAG < available else max(possible)) if possible else None
    out["cage_relative_msd_ratio"] = (
        out[f"cage_relative_msd_ratio_lag{chosen}"] if chosen is not None else float("nan"))
    out["cage_motion_fraction"] = (
        out[f"cage_motion_fraction_lag{chosen}"] if chosen is not None else float("nan"))
    out["cage_lag"] = float(chosen) if chosen is not None else float("nan")
    return out


# --------------------------------------------------------------------- populations


def population_state(
    trajectory: Trajectory, cells: np.ndarray, transient: float = 0.2, threshold: float = 0.05
) -> dict[str, float]:
    """The motion and shape of a subset of the cells, for a tissue with a minority.

    The same measurements :func:`tissue_state` pools over every cell, taken over
    ``cells`` alone: ``speed`` (mean displacement per sample interval, in the lab
    frame), ``diffusion_coefficient`` and ``msd_exponent`` from their own MSD,
    ``exchange_rate_per_cell`` (neighbour changes these cells are party to, per cell
    per time -- so one exchange between a minority and a bulk cell counts once for
    each population), ``coordination`` (mean neighbour count), ``area`` and
    ``shape_index``. A single cell gives one sample of each: fine for following it
    through an animation, noisy for a sweep, which wants several cells or seeds.
    """
    cells = np.atleast_1d(np.asarray(cells, dtype=int))
    if len(cells) == 0:
        raise ValueError("population_state needs at least one cell")
    path = tracks(trajectory)
    start = _after_transient(trajectory, transient)
    positions, times = path.positions[start:, cells], path.times[start:]
    snapshots = trajectory.snapshots[start:]
    out: dict[str, float] = {"n_cells": float(len(cells))}

    steps = np.diff(positions, axis=0)
    intervals = np.diff(times)[:, None]
    out["speed"] = float((_norm(steps) / intervals).mean()) if len(steps) else float("nan")

    if len(positions) >= 4:
        lags, msd = _msd_of(positions, times)
        out["diffusion_coefficient"], out["msd_exponent"] = _fit_msd(lags, msd, positions.shape[-1])
    else:
        out["diffusion_coefficient"] = out["msd_exponent"] = float("nan")

    graphs = [neighbour_graph(s, threshold) for s in snapshots]
    changes = sum(int((a ^ b)[cells].sum()) for a, b in zip(graphs, graphs[1:]))
    span = snapshots[-1].time - snapshots[0].time
    out["exchange_rate_per_cell"] = changes / (len(cells) * span) if span > 0 else float("nan")
    out["coordination"] = float(np.mean([g[cells].sum(axis=1).mean() for g in graphs]))

    out["area"] = float(np.mean([s.areas[cells].mean() for s in snapshots]))
    out["shape_index"] = float(np.mean([s.shape_indices[cells].mean() for s in snapshots]))
    return out


def minority_state(trajectory: Trajectory, transient: float = 0.2) -> dict[str, float]:
    """:func:`population_state` of the minority and of the bulk, keyed ``minority_*``
    and ``bulk_*``, plus ``minority_area_ratio`` (mean minority area over mean bulk
    area, against the ``minority_size**2`` it was asked for) and ``minority_speed_ratio``.
    Empty for a uniform tissue."""
    model = trajectory.model
    if not getattr(model, "has_minority", False):
        return {}
    minority = population_state(trajectory, model.minority_cells, transient)
    bulk = population_state(trajectory, model.bulk_cells, transient)
    out = {f"minority_{k}": v for k, v in minority.items()}
    out.update({f"bulk_{k}": v for k, v in bulk.items()})
    out["minority_area_ratio"] = minority["area"] / bulk["area"] if bulk["area"] > 0 else float("nan")
    out["minority_speed_ratio"] = minority["speed"] / bulk["speed"] if bulk["speed"] > 0 else float("nan")
    return out


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
    persistence = neighbour_persistence(trajectory, transient)
    state["neighbour_persistence_time"] = persistence["neighbour_persistence_time"]
    state["neighbour_persistence_censored"] = float(persistence["censored"])
    state["neighbour_persistence_final"] = persistence["final"]
    state.update(velocity_correlations(trajectory, transient))
    state.update(structure(trajectory, transient))
    state.update(cage_relative_motion(trajectory, transient))
    state.update(minority_state(trajectory, transient))
    path = tracks(trajectory)
    state["max_unwrap_step"] = path.max_step
    state["confluence"] = trajectory.final.confluence
    state["occupancy"] = trajectory.final.occupancy
    return state
