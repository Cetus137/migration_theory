"""Running, storing and measuring: the path from a Model to a number."""

from __future__ import annotations

import numpy as np
import pytest

from migration_theory import (
    Model,
    Snapshot,
    Trajectory,
    diffusion_coefficient,
    load_trajectory,
    mean_squared_displacement,
    neighbour_exchange_rate,
    neighbour_graph,
    persistent_random_walk,
    save_trajectory,
    shape_statistics,
    simulate,
    tissue_state,
    velocities,
    velocity_correlation,
)


# ----------------------------------------------------------------- simulate


def test_duration_and_steps_agree(model):
    by_time = simulate(model, duration=100.0, n_snapshots=3, keep_fields=False)
    by_count = simulate(model, by_time.steps, n_snapshots=3, keep_fields=False)
    assert by_count.duration == pytest.approx(by_time.duration)


def test_exactly_one_of_steps_or_duration(model):
    with pytest.raises(ValueError, match="exactly one"):
        simulate(model)
    with pytest.raises(ValueError, match="exactly one"):
        simulate(model, 10, duration=10.0)


def test_snapshots_span_the_run(model):
    trajectory = simulate(model, duration=100.0, n_snapshots=9, keep_fields=False)
    assert trajectory.snapshots[0].step == 0
    assert trajectory.snapshots[-1].step == trajectory.steps
    assert np.all(np.diff(trajectory.times) > 0)


def test_snapshots_carry_the_per_cell_data(model):
    final = simulate(model, duration=50.0, n_snapshots=3, keep_fields=False).final
    assert final.areas.shape == (model.n_cells,)
    assert final.perimeters.shape == (model.n_cells,)
    assert final.contacts.shape == (model.n_cells, model.n_cells)
    assert final.centres.shape == (model.n_cells, 2)
    assert np.all(np.diag(final.contacts) == 0)
    assert final.shape_indices == pytest.approx(final.perimeters / np.sqrt(final.areas))


def test_keep_fields_controls_the_image_data(model):
    assert simulate(model, duration=20, n_snapshots=3, keep_fields=True).final.field is not None
    assert simulate(model, duration=20, n_snapshots=3, keep_fields=False).final.field is None


def test_warmup_resets_the_clock_and_relaxes_first(model):
    """Warm-up runs the passive relaxation before the clock starts, so the recorded run
    opens from a relaxed state rather than the seeded one.

    Judged by the energy, not the overlap: with tessellated seeding the cold start
    overlaps *little* and relaxing raises the overlap, because in this small model the
    area constraint has to expand the cells into each other to reach their target.
    """
    warmed = simulate(model, duration=50, warmup=400, n_snapshots=3, keep_fields=False)
    cold = simulate(model, duration=50, n_snapshots=3, keep_fields=False)
    assert warmed.snapshots[0].time == pytest.approx(0.0)
    assert warmed.snapshots[0].step == 0
    assert warmed.snapshots[0].energy < 0.1 * cold.snapshots[0].energy


# ----------------------------------------------------------------- storage


def test_save_and_load_round_trip(model, tmp_path):
    original = simulate(model, duration=80.0, n_snapshots=8, keep_fields=False)
    path = save_trajectory(original, tmp_path / "run")
    restored = load_trajectory(path)

    assert restored.model == original.model          # every parameter survives
    assert restored.dt == pytest.approx(original.dt)
    assert restored.steps == original.steps
    assert len(restored.snapshots) == len(original.snapshots)
    assert restored.energies == pytest.approx(original.energies)
    assert restored.final.contacts == pytest.approx(original.final.contacts)
    assert restored.tissue is None                    # fields are not stored


def test_a_dotted_stem_is_not_mistaken_for_an_extension(model, tmp_path):
    """``Path.with_suffix`` would turn "run_v0.1" into "run_v0.npz" -- the name a
    different run would use."""
    trajectory = simulate(model, duration=20.0, n_snapshots=3, keep_fields=False)
    assert save_trajectory(trajectory, tmp_path / "run_v0.1").name == "run_v0.1.npz"
    assert save_trajectory(trajectory, tmp_path / "plain.npz").name == "plain.npz"


def test_analysis_works_on_a_loaded_trajectory(model, tmp_path):
    original = simulate(model, duration=200.0, n_snapshots=20, keep_fields=False)
    restored = load_trajectory(save_trajectory(original, tmp_path / "run"))
    assert tissue_state(restored).keys() == tissue_state(original).keys()


# ----------------------------------------------------------------- analysis


def test_msd_is_quadratic_for_straight_line_motion():
    """No rotational diffusion and cells far apart: pure ballistic motion."""
    free = Model(n_cells=4, cell_radius=5.0, packing=0.25,
                 speed=0.05, rotational_diffusion=0.0)
    trajectory = simulate(free, duration=300, n_snapshots=40, keep_fields=False)
    assert diffusion_coefficient(trajectory)["msd_exponent"] == pytest.approx(2.0, abs=0.15)


def test_persistent_random_walk_recovers_what_went_in():
    """Fit the measured motion of free cells; it must return the imposed parameters.

    Eight cells rather than four, for the statistics at lags near the persistence
    time; a soft area constraint only to loosen the timestep, since free cells do not
    care about it.
    """
    free = Model(n_cells=8, cell_radius=5.0, packing=0.25, area_lambda=600.0,
                 speed=0.05, rotational_diffusion=0.02)
    trajectory = simulate(free, duration=1500, n_snapshots=120, keep_fields=False)
    fit = persistent_random_walk(trajectory)
    assert fit["persistence_fit_at_bound"] == 0.0
    assert fit["measured_speed"] == pytest.approx(free.speed, rel=0.4)
    assert fit["measured_persistence_time"] == pytest.approx(free.persistence_time, rel=0.6)
    assert 0.3 < fit["speed_ratio"] < 2.5


def test_msd_grows_with_lag(model):
    trajectory = simulate(model.replace(speed=0.02), duration=300,
                          n_snapshots=30, keep_fields=False)
    lags, msd = mean_squared_displacement(trajectory)
    assert np.all(lags > 0)
    assert msd[-1] > msd[0]


def test_shape_statistics_report_a_distribution(model):
    trajectory = simulate(model, duration=200, n_snapshots=20, keep_fields=False)
    stats = shape_statistics(trajectory)
    assert stats["shape_index_p5"] <= stats["shape_index_median"] <= stats["shape_index_p95"]
    assert stats["shape_index_std"] >= 0
    assert 3.0 < stats["shape_index_mean"] < 6.0


def test_neighbour_graph_is_symmetric_and_hollow(model):
    final = simulate(model, duration=200, n_snapshots=3, keep_fields=False).final
    adjacency = neighbour_graph(final)
    assert np.array_equal(adjacency, adjacency.T)
    assert not np.any(np.diag(adjacency))


def test_a_settled_passive_tissue_exchanges_no_neighbours(model):
    """Zero rearrangement is the signature of a solid, and the baseline the active
    case has to be compared against."""
    trajectory = simulate(model, duration=600, warmup=400, n_snapshots=30, keep_fields=False)
    assert neighbour_exchange_rate(trajectory)["exchange_rate_per_cell"] == pytest.approx(0.0)


def test_tissue_state_returns_finite_numbers(model):
    trajectory = simulate(model.replace(speed=0.02), duration=400,
                          n_snapshots=40, keep_fields=False)
    state = tissue_state(trajectory)
    assert set(state) >= {"shape_index_mean", "diffusion_coefficient", "msd_exponent",
                          "measured_speed", "exchange_rate_per_cell", "occupancy",
                          "velocity_correlation_length", "neighbour_velocity_correlation"}
    assert all(np.isfinite(v) for v in state.values())


# ----------------------------------------------------------------- velocity correlation


def _walk(model, velocity_of, n_frames=80, interval=10.0, seed=0):
    """A synthetic trajectory whose cells move at ``velocity_of(positions, rng)``.

    Only the centres matter to the velocity analysis; everything else is filled with
    placeholders so a :class:`Snapshot` can be built without simulating.
    """
    rng = np.random.default_rng(seed)
    box = model.box
    n = model.n_cells
    centres = rng.uniform(0.0, 1.0, (n, 2)) * box.lengths
    snapshots = []
    for frame in range(n_frames):
        snapshots.append(Snapshot(
            step=frame, time=frame * interval, energy=0.0, breakdown={},
            area_ratio=1.0, shape_index=3.8, confluence=0.0, occupancy=1.0,
            centres=centres.copy(), areas=np.ones(n), perimeters=np.ones(n),
            contacts=np.zeros((n, n)),
        ))
        centres = box.wrap(centres + interval * velocity_of(centres, rng))
    return Trajectory(model=model, snapshots=snapshots, tissue=None, dt=interval,
                      max_dt=interval, steps=n_frames, wall_seconds=0.0)


def test_velocities_have_no_net_drift():
    """The tissue's own motion is subtracted, whatever it is."""
    model = Model(n_cells=16)
    drifting = _walk(model, lambda x, rng: np.tile(rng.normal(0, 0.1, 2), (len(x), 1)))
    _, velocity, _ = velocities(drifting, transient=0.0)
    assert np.abs(velocity).max() < 1e-9


def test_velocity_correlation_starts_at_one_and_is_short_for_independent_cells():
    model = Model(n_cells=16)
    independent = _walk(model, lambda x, rng: rng.normal(0, 0.1, x.shape))
    curve = velocity_correlation(independent, transient=0.0)
    assert curve["correlation"][0] == 1.0
    assert np.all(np.diff(curve["separation"]) > 0)
    # Independent cells are uncorrelated at every separation, so the drop to 1/e
    # happens before the first populated bin.
    assert curve["correlation_length"] < curve["separation"][1]
    assert not curve["censored"]


def test_velocity_correlation_length_grows_with_a_long_wavelength_mode():
    """Cells moving with a smooth field are correlated over its wavelength."""
    model = Model(n_cells=16)
    L = model.box.lengths

    def wave(x, rng):
        amplitude = rng.normal(0, 0.1, 2)
        phase = 2 * np.pi * x / L
        return amplitude * np.column_stack([np.sin(phase[:, 0]), np.sin(phase[:, 1])])

    collective = _walk(model, wave)
    independent = _walk(model, lambda x, rng: rng.normal(0, 0.1, x.shape))
    assert (velocity_correlation(collective, transient=0.0)["correlation_length"]
            > velocity_correlation(independent, transient=0.0)["correlation_length"])
