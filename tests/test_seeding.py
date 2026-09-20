"""Cell-centre seeding, phase-field seeding, and the geometric diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from migration_theory import (
    Grid,
    PeriodicBox,
    areas,
    centres_of_mass,
    confluence_error,
    evenly_spaced,
    lloyd_relax,
    overlap_matrix,
    perimeters,
    poisson_disc,
    random_positions,
    seed,
    shape_indices,
    triangular_lattice,
)


def nearest_neighbour_distances(centres, box):
    wrapped = box.wrap(centres)
    distances, _ = cKDTree(wrapped, boxsize=box.lengths).query(wrapped, k=2)
    return distances[:, 1]


# ----------------------------------------------------------------- centres


def test_every_method_stays_inside_the_box(box):
    rng = np.random.default_rng(0)
    for centres in (
        random_positions(12, box, rng),
        poisson_disc(8, box, 0.5, rng=np.random.default_rng(0)),
        evenly_spaced(12, box, rng=np.random.default_rng(0)),
        triangular_lattice(12, 0.8, rng=rng)[0],
    ):
        assert np.all(centres >= 0)
        assert np.all(centres < box.lengths + 1e-12)


def test_poisson_disc_respects_its_minimum(box):
    separation = 0.6
    centres = poisson_disc(10, box, separation, rng=np.random.default_rng(0))
    assert nearest_neighbour_distances(centres, box).min() >= separation - 1e-9


def test_lloyd_makes_a_point_set_more_even(box):
    rng = np.random.default_rng(0)
    before = random_positions(16, box, rng)
    after = lloyd_relax(before, box, n_iterations=25)

    def spread(centres):
        distances = nearest_neighbour_distances(centres, box)
        return distances.std() / distances.mean()

    assert spread(after) < spread(before) / 2


def test_lloyd_is_deterministic(box):
    start = random_positions(12, box, np.random.default_rng(0))
    assert lloyd_relax(start, box, n_iterations=10) == pytest.approx(
        lloyd_relax(start, box, n_iterations=10)
    )


def test_triangular_lattice_tiles_its_own_box():
    centres, box = triangular_lattice(16, spacing=1.0)
    distances = nearest_neighbour_distances(centres, box)
    assert distances == pytest.approx(1.0, rel=1e-9)


def test_poisson_disc_refuses_an_impossible_density(box):
    with pytest.raises(RuntimeError, match="too high"):
        poisson_disc(500, box, 1.0, rng=np.random.default_rng(0), max_attempts=200)


# ----------------------------------------------------------------- fields


@pytest.fixture
def single_cell():
    grid = Grid(PeriodicBox(40.0, 40.0), (80, 80))
    return grid, seed(grid, np.array([[20.0, 20.0]]), 8.0, 1.0)


def test_seeded_profile_matches_the_analytic_tanh(single_cell):
    grid, fields = single_cell
    radius, width = 8.0, 1.0
    distance = grid.distance_to(np.array([20.0, 20.0]))
    expected = 0.5 * (1 - np.tanh((distance - radius) / (np.sqrt(2) * width)))
    assert fields[0] == pytest.approx(expected, abs=1e-12)


def test_measured_area_matches_the_integrated_profile(single_cell):
    """``INT phi^2`` undershoots ``pi R^2`` at finite width -- by exactly this much."""
    grid, fields = single_cell
    radius, width = 8.0, 1.0
    r = np.linspace(0, radius + 12 * width, 20000)
    profile = 0.5 * (1 - np.tanh((r - radius) / (np.sqrt(2) * width)))
    expected = 2 * np.pi * np.trapezoid(profile**2 * r, r)
    assert areas(fields)[0] == pytest.approx(expected, rel=1e-3)
    assert areas(fields)[0] < np.pi * radius**2


def test_perimeter_of_a_disc(single_cell):
    """``INT |grad phi|`` is the level-set length, whatever the profile shape."""
    _, fields = single_cell
    assert perimeters(fields)[0] == pytest.approx(2 * np.pi * 8.0, rel=0.01)


def test_shape_index_of_a_disc_is_near_the_circle_value(single_cell):
    _, fields = single_cell
    assert shape_indices(fields)[0] == pytest.approx(2 * np.sqrt(np.pi), rel=0.08)


def test_centre_of_mass_recovers_a_wrapped_cell():
    grid = Grid(PeriodicBox(40.0, 40.0), (80, 80))
    centres = np.array([[0.5, 20.0], [39.5, 2.0]])
    fields = seed(grid, centres, 6.0, 1.0)
    offset = grid.box.min_image(centres_of_mass(fields) - centres)
    assert np.hypot(offset[:, 0], offset[:, 1]).max() < 0.05


def test_overlap_matrix_is_symmetric_and_zero_when_apart():
    grid = Grid(PeriodicBox(60.0, 60.0), (60, 60))
    apart = seed(grid, np.array([[15.0, 15.0], [45.0, 45.0]]), 5.0, 1.0)
    matrix = overlap_matrix(apart)
    assert matrix == pytest.approx(matrix.T)
    assert matrix[0, 1] == pytest.approx(0.0, abs=1e-10)


def test_confluence_error_is_zero_for_a_full_field():
    grid = Grid(PeriodicBox(10.0, 10.0), (20, 20))
    from migration_theory import PhaseFields

    full = PhaseFields(grid, np.ones((1, *grid.shape)))
    assert confluence_error(full) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [{"radius": 0.0}, {"interface_width": -1.0}])
def test_seed_rejects_unphysical_arguments(bad, grid):
    kwargs = {"radius": 1.0, "interface_width": 0.3, **bad}
    with pytest.raises(ValueError):
        seed(grid, np.array([[1.0, 1.0]]), **kwargs)
