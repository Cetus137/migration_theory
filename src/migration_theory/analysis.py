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
    "velocities",
    "drift_speed_ratio",
    "velocity_correlation",
    "velocity_correlations",
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
    drift = np.hypot(*velocity.mean(axis=1).T).mean()
    cells = np.hypot(velocity[..., 0], velocity[..., 1]).mean()
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
    distance = np.hypot(separation[..., 0], separation[..., 1]).ravel()
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
#: sweeps' 25 s sampling these are 25, 100, 250 and 500 s: from cage jostling to the
#: slow collective motion that lasts a good part of a persistence time.
CORRELATION_LAGS = (1, 4, 10, 20)


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

    The un-suffixed ``velocity_correlation_length`` and
    ``neighbour_velocity_correlation`` are the lag-1 values, as before, with
    ``velocity_correlation_censored`` set to 1 when the length is only a lower bound
    and ``velocity_correlation_bound`` the half box it is then bounded by.
    ``drift_speed_ratio`` is the whole-tissue motion the correlations subtract.
    """
    start = _after_transient(trajectory, transient)
    available = len(trajectory.snapshots) - start
    out: dict[str, float] = {}
    for lag in lags:
        if lag >= available:
            length = neighbour = float("nan")
            censored = 0.0
        else:
            curve = velocity_correlation(trajectory, transient, lag=lag)
            length = float(curve["correlation_length"])
            censored = float(curve["censored"])
            neighbour = _neighbour_correlation(trajectory, transient, threshold, lag)
        out[f"velocity_correlation_length_lag{lag}"] = length
        out[f"neighbour_velocity_correlation_lag{lag}"] = neighbour
        if lag == lags[0]:
            out["velocity_correlation_length"] = length
            out["velocity_correlation_censored"] = censored
            out["neighbour_velocity_correlation"] = neighbour
    out["velocity_correlation_bound"] = 0.5 * trajectory.model.box.min_length
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
        counts += np.histogram(np.hypot(separation[:, 0], separation[:, 1]), edges)[0]
        frames += 1

    shell_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    expected = frames * (n_cells * (n_cells - 1) / 2) * shell_area / box.area
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
    """
    start = _after_transient(trajectory, transient)
    box = trajectory.model.box
    spacing = trajectory.model.cell_spacing

    r, g = pair_correlation(trajectory, transient)
    x = r / spacing
    first = g[(x > 0.5) & (x < 1.5)]
    second = g[(x >= 1.5) & (x < 2.5)]

    psi6, hexagons = [], []
    for snapshot in trajectory.snapshots[start:]:
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
        "hexagon_fraction": float(np.mean(hexagons)),
        "g_first_peak": float(first.max()) if len(first) else float("nan"),
        "g_second_peak": float(second.max()) if len(second) else float("nan"),
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
    state.update(velocity_correlations(trajectory, transient))
    state.update(structure(trajectory, transient))
    path = tracks(trajectory)
    state["max_unwrap_step"] = path.max_step
    state["confluence"] = trajectory.final.confluence
    state["occupancy"] = trajectory.final.occupancy
    return state
