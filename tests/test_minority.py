"""A minority population: cells that differ in size, drive, persistence or drag.

Two guarantees are tested. A uniform tissue computes exactly what it always did --
per-cell arrays of equal values reproduce the scalar arithmetic to the last bit, so
``minority_count = 0`` and a minority at ratio 1 are the same trajectory as before.
And a minority that does differ is seeded at its size, moves by its own rule, and is
measured on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from migration_theory import (  # noqa: E402
    ForceBalance,
    Grid,
    Model,
    PeriodicBox,
    Polarity,
    Tissue,
    areas,
    minority_state,
    population_state,
    seed,
    seed_tessellated,
    simulate,
    tissue_state,
)
from naming import encode  # noqa: E402


# ----------------------------------------------------------------- seeding


def test_equal_radii_seed_exactly_as_one_radius(grid):
    centres = np.array([[2.0, 1.5], [5.5, 4.0], [7.0, 1.0]])
    for lay_down in (seed, seed_tessellated):
        one = lay_down(grid, centres, 1.2, 0.3)
        many = lay_down(grid, centres, np.full(3, 1.2), 0.3)
        assert np.array_equal(one.values, many.values)


def test_tessellated_boundary_moves_by_half_the_radius_difference():
    """Two cells 12 um apart, radii 8 and 5: the weighted bisector sits where
    ``d_0 - 8 = d_1 - 5``, at x = 17.5, not at the midpoint 16."""
    grid = Grid(PeriodicBox(40.0, 40.0), (80, 80))                  # dx = 0.5
    centres = np.array([[10.0, 20.0], [22.0, 20.0]])
    fields = seed_tessellated(grid, centres, np.array([8.0, 5.0]), 0.5)
    X, _ = grid.coordinates
    row = 40                                                         # y = 20
    profile = fields[0][row]
    x = X[row]
    # The phi = 0.5 crossing of the large cell, on the side facing the small one.
    facing = (x > 10.0) & (x < 22.0)
    k = np.flatnonzero((profile[facing][:-1] >= 0.5) & (profile[facing][1:] < 0.5))[0]
    xs, ps = x[facing][k:k + 2], profile[facing][k:k + 2]
    crossing = xs[0] + (ps[0] - 0.5) * (xs[1] - xs[0]) / (ps[0] - ps[1])
    assert crossing == pytest.approx(17.5, abs=0.3)
    # And the small cell's crossing is the same place: the two tile without a gap.
    small = fields[1][row]
    k = np.flatnonzero((small[facing][:-1] < 0.5) & (small[facing][1:] >= 0.5))[0]
    xs, ps = x[facing][k:k + 2], small[facing][k:k + 2]
    assert xs[0] + (0.5 - ps[0]) * (xs[1] - xs[0]) / (ps[1] - ps[0]) == pytest.approx(17.5, abs=0.3)
    assert seed_tessellated(grid, centres, 8.0, 0.5)[0][row].tolist() != profile.tolist()


def test_seed_rejects_a_bad_radius_per_cell(grid):
    with pytest.raises(ValueError):
        seed(grid, np.zeros((2, 2)), np.array([1.0, -1.0]), 0.3)
    with pytest.raises(ValueError):
        seed(grid, np.zeros((2, 2)), np.array([1.0, 1.0, 1.0]), 0.3)


# ----------------------------------------------------------------- polarity and forces


def test_equal_per_cell_polarity_parameters_reproduce_the_scalar_bitwise():
    a = Polarity.random(20, speed=0.3, rotational_diffusion=0.02, rng=np.random.default_rng(3))
    b = Polarity.random(20, speed=np.full(20, 0.3), rotational_diffusion=np.full(20, 0.02),
                        rng=np.random.default_rng(3))
    for _ in range(50):
        a.rotate(0.1)
        b.rotate(0.1)
    assert np.array_equal(a.angles, b.angles)
    assert np.array_equal(a.velocities, b.velocities)
    assert isinstance(a.persistence_time, float) and b.persistence_time.shape == (20,)


def test_each_cell_rotates_at_its_own_rate():
    n = 4000
    rates = np.where(np.arange(n) < n // 2, 0.01, 0.1)
    polarity = Polarity.random(n, rotational_diffusion=rates, rng=np.random.default_rng(0))
    start = polarity.directors.copy()
    for _ in range(200):
        polarity.rotate(0.05)
    elapsed = 200 * 0.05
    measured = (polarity.directors * start).sum(axis=1)
    assert measured[: n // 2].mean() == pytest.approx(np.exp(-0.01 * elapsed), rel=0.05)
    assert measured[n // 2:].mean() == pytest.approx(np.exp(-0.1 * elapsed), rel=0.1)
    assert polarity.persistence_time[0] == pytest.approx(100.0)
    assert polarity.persistence_time[-1] == pytest.approx(10.0)


def test_per_cell_force_balance_matches_the_scalar_and_scales_per_cell(model):
    tissue = model.tissue(seed=0)
    mu = model.free_energy().functional_derivative(tissue.fields)
    n = model.n_cells
    scalar = ForceBalance(3.0, 6.0, 10.0).velocities(tissue, mu)
    arrays = ForceBalance(np.full(n, 3.0), np.full(n, 6.0), np.full(n, 10.0)).velocities(tissue, mu)
    assert np.array_equal(scalar, arrays)
    # Doubling one cell's friction halves its velocity and touches no other cell.
    friction = np.full(n, 10.0)
    friction[0] = 20.0
    halved = ForceBalance(3.0, 6.0, friction).velocities(tissue, mu)
    assert halved[0] == pytest.approx(0.5 * scalar[0])
    assert halved[1:] == pytest.approx(scalar[1:])
    with pytest.raises(ValueError):
        ForceBalance(3.0, np.array([6.0, -1.0]), 10.0)


# ----------------------------------------------------------------- the model


def test_a_minority_at_ratio_one_is_the_uniform_tissue_bitwise(model):
    """Every per-cell array is the scalar repeated, so the trajectory is identical."""
    active = model.replace(propulsion="force", active_energy=2.0, rotational_diffusion=0.01)
    uniform = simulate(active, duration=20.0, n_snapshots=3, keep_fields=True)
    odd = simulate(active.replace(minority_count=1), duration=20.0, n_snapshots=3,
                   keep_fields=True)
    assert np.array_equal(uniform.tissue.fields.values, odd.tissue.fields.values)
    assert np.array_equal(uniform.tissue.polarity.angles, odd.tissue.polarity.angles)
    assert uniform.dt == odd.dt


def test_minority_defaults_leave_names_and_boxes_unchanged(model):
    assert not model.has_minority
    assert encode(model, 100.0, 0) == encode(model.replace(minority_size=1.0), 100.0, 0)
    assert "min" not in encode(model, 100.0, 0)
    odd = model.replace(minority_count=1, minority_size=1.5)
    name = encode(odd, 100.0, 0)
    assert "_min1_minR1.5_" in name and "mina" not in name
    assert odd.box_length > model.box_length                       # room for the extra area
    assert odd.total_target_area == pytest.approx(model.target_area * (model.n_cells - 1 + 2.25))
    assert model.total_target_area == model.n_cells * model.target_area


def test_minority_arrays_follow_the_ratios():
    m = Model(n_cells=10, propulsion="force", active_energy=4.0, cell_radius=6.0,
              rotational_diffusion=0.01, minority_count=2, minority_size=1.5,
              minority_activity=0.5, minority_persistence=4.0, minority_friction=2.0)
    assert m.is_minority.tolist() == [True, True] + [False] * 8
    assert m.radii[:2] == pytest.approx(9.0) and m.radii[2:] == pytest.approx(6.0)
    assert m.target_areas[0] == pytest.approx(np.pi * 81.0)
    # Activity ratio 0.5 at 1.5 the radius: E_a = a sigma R gives 0.75 the energy, so
    # the active *force* E_a / R is half the bulk's.
    assert m.active_energies[0] == pytest.approx(0.75 * 4.0)
    assert (m.active_energies[0] / m.radii[0]) == pytest.approx(0.5 * (4.0 / 6.0))
    assert m.rotational_diffusions[0] == pytest.approx(0.0025) and m.rotational_diffusions[-1] == 0.01
    assert m.cell_frictions[0] == pytest.approx(20.0) and m.cell_frictions[-1] == 10.0
    rule = m.propulsion_rule()
    assert isinstance(rule, ForceBalance) and rule.active_energy.shape == (10,)
    assert "minority: 2 cells" in str(m)
    with pytest.raises(ValueError):
        Model(n_cells=4, minority_count=4)
    with pytest.raises(ValueError):
        Model(minority_size=0.0)
    with pytest.raises(ValueError):
        Model(minority_activity=-1.0)
    assert Model(minority_count=1, minority_activity=0.0).active_energies[0] == 0.0


def test_a_large_minority_cell_is_born_large_and_stays_so():
    m = Model(n_cells=9, cell_radius=6.0, grid_spacing=1.0, minority_count=1, minority_size=1.5)
    tissue = m.tissue(seed=0)
    born = areas(tissue.fields)
    assert born[0] > 1.6 * born[1:].mean()                           # seeded at its size
    trajectory = simulate(m, duration=60.0, n_snapshots=3, keep_fields=True)
    settled = areas(trajectory.tissue.fields)
    assert settled[0] / settled[1:].mean() == pytest.approx(2.25, rel=0.2)
    assert trajectory.model.minority_cells.tolist() == [0]


# ----------------------------------------------------------------- measuring it


def test_population_state_measures_the_subset(model):
    active = model.replace(propulsion="force", active_energy=2.0, rotational_diffusion=0.01)
    trajectory = simulate(active, duration=200.0, n_snapshots=12, keep_fields=False)
    everyone = population_state(trajectory, np.arange(model.n_cells), transient=0.2)
    one = population_state(trajectory, [0], transient=0.2)
    assert everyone["n_cells"] == model.n_cells and one["n_cells"] == 1
    for key in ("speed", "diffusion_coefficient", "msd_exponent", "exchange_rate_per_cell",
                "coordination", "area", "shape_index"):
        assert np.isfinite(everyone[key]) and np.isfinite(one[key])
    assert everyone["area"] == pytest.approx(
        np.mean([s.areas.mean() for s in trajectory.snapshots[2:]]))
    assert minority_state(trajectory) == {}                          # uniform: nothing to report
    with pytest.raises(ValueError):
        population_state(trajectory, [])


def test_tissue_state_reports_the_minority_and_the_bulk():
    m = Model(n_cells=9, cell_radius=6.0, propulsion="force", active_energy=2.0,
              rotational_diffusion=0.01, minority_count=1, minority_size=1.5,
              minority_activity=0.0)                              # a passive obstacle
    trajectory = simulate(m, duration=200.0, n_snapshots=12, keep_fields=False)
    state = tissue_state(trajectory)
    assert state["minority_n_cells"] == 1 and state["bulk_n_cells"] == 8
    assert state["minority_area_ratio"] == pytest.approx(2.25, rel=0.25)
    for key in ("minority_speed", "bulk_speed", "minority_diffusion_coefficient",
                "minority_exchange_rate_per_cell", "bulk_exchange_rate_per_cell",
                "minority_shape_index", "minority_speed_ratio"):
        assert np.isfinite(state[key])
    # Still a valid trajectory for the pooled observables.
    assert np.isfinite(state["shape_index_mean"]) and np.isfinite(state["measured_speed"])
