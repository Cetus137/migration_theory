"""Per-cell windows: finding them, reading them across the boundary, measuring and
stepping on them.

Every windowed computation is held against the dense one it replaces, to the tail the
windows drop -- below the threshold in the field, so far below in anything integrated.
"""

from __future__ import annotations

import numpy as np
import pytest

from migration_theory import (
    Adhesion,
    AreaConstraint,
    DoubleWell,
    ExplicitEuler,
    ForceBalance,
    FreeEnergy,
    GradientEnergy,
    Grid,
    ImposedVelocity,
    Model,
    PeriodicBox,
    PhaseFields,
    Repulsion,
    Tissue,
    Windows,
    advection,
    areas,
    centres_of_mass,
    interface_terms,
    passive_forces,
    perimeters,
    run,
    seed,
    simulate,
)

THRESHOLD = 1e-6


def relaxed(model, time=10.0):
    """A tissue that has moved a little from its seeded state."""
    energy = model.free_energy()
    tissue = model.tissue(seed=0)
    stepper = model.stepper(tissue, energy)
    run(tissue, energy, stepper, int(round(time / stepper.dt)), check=False)
    return tissue.fields


@pytest.fixture
def big():
    """A tissue whose cells are small next to the box, so windows are genuinely windows.

    The tanh tail reaches the threshold about 20 um beyond a 6 um cell at this
    interface width, so a window is about 57 points; a packing of 0.3 makes the box
    77 points, which leaves room for the window to be smaller than the grid.
    """
    return Model(n_cells=16, cell_radius=6.0, packing=0.3, grid_spacing=1.0)


# ----------------------------------------------------------------- finding them


def test_windows_cover_every_cell(big):
    fields = big.tissue(seed=0).fields
    windows = fields.find_windows(threshold=THRESHOLD)
    assert windows.shape[0] < fields.grid.ny and windows.shape[1] < fields.grid.nx
    outside = ~windows.mask()
    assert np.all(fields.values[outside] <= THRESHOLD)


def test_windows_still_cover_after_the_tissue_has_moved(big):
    fields = relaxed(big)
    windows = fields.find_windows(threshold=THRESHOLD)
    assert np.all(fields.values[~windows.mask()] <= THRESHOLD)


def test_a_cell_on_the_boundary_gets_a_wrapped_window():
    """The occupied run straddles both edges; the window must follow it round."""
    grid = Grid(PeriodicBox(40.0, 40.0), (40, 40))
    fields = seed(grid, np.array([[0.5, 39.5]]), 4.0, 0.5)   # sharp, so the window is small
    windows = fields.find_windows(threshold=THRESHOLD)
    assert windows.shape[0] < 40 and windows.shape[1] < 40
    rows, cols = windows.indices()
    assert rows.min() == 0 and rows.max() == 39     # wraps in y
    assert cols.min() == 0 and cols.max() == 39     # wraps in x
    assert np.all(fields.values[~windows.mask()] <= THRESHOLD)
    # Extracting through the wrap equals rolling the cell to the origin and slicing.
    origin = windows.origins[0]
    rolled = np.roll(np.roll(fields[0], -origin[0], axis=0), -origin[1], axis=1)
    h, w = windows.shape
    assert windows.extract(fields.values)[0] == pytest.approx(rolled[:h, :w])


def test_extract_is_batched_and_matches_cell_by_cell(big):
    fields = big.tissue(seed=0).fields
    windows = fields.find_windows()
    patches = windows.extract(fields.values)
    assert patches.shape == (fields.n_cells, *windows.shape)
    rows, cols = windows.indices()
    for i in range(fields.n_cells):
        assert patches[i] == pytest.approx(fields[i][np.ix_(rows[i], cols[i])])


def test_a_box_barely_bigger_than_a_cell_gives_the_whole_grid():
    grid = Grid(PeriodicBox(12.0, 12.0), (12, 12))
    fields = seed(grid, np.array([[6.0, 6.0]]), 5.0, 1.0)
    windows = fields.find_windows()
    assert windows.shape == grid.shape
    assert windows.spans_rows and windows.spans_cols


def test_rejects_a_window_that_does_not_fit(grid):
    with pytest.raises(ValueError):
        Windows(grid, (grid.ny + 1, 4), np.zeros((2, 2), dtype=int))


# ----------------------------------------------------------------- measuring on them


def test_windowed_measurements_match_dense(big):
    """To the truncation the threshold implies: the tail beyond the window is below
    1e-6 in the field, so far below in the area and ~1e-5 relative in the perimeter."""
    fields = relaxed(big)
    windows = fields.find_windows(threshold=THRESHOLD)
    assert areas(fields, windows) == pytest.approx(areas(fields), rel=1e-9)
    assert perimeters(fields, windows) == pytest.approx(perimeters(fields), rel=1e-5)
    # Both are exact centroids now, so they agree to the dropped tail.
    offset = fields.grid.box.min_image(centres_of_mass(fields, windows) - centres_of_mass(fields))
    assert np.hypot(offset[:, 0], offset[:, 1]).max() < 1e-6


def test_windowed_centroid_is_exact_for_a_symmetric_cell():
    grid = Grid(PeriodicBox(40.0, 40.0), (80, 80))
    centre = np.array([[0.5, 20.0]])          # straddling the x boundary
    fields = seed(grid, centre, 6.0, 1.0)
    windows = fields.find_windows()
    recovered = centres_of_mass(fields, windows)
    offset = grid.box.min_image(recovered - centre)
    assert np.hypot(*offset.T).max() < 1e-6
    assert centres_of_mass(fields, windows)[0] == pytest.approx(centres_of_mass(fields)[0], abs=1e-6)


def test_degenerate_window_reproduces_dense_exactly():
    """Window == grid: padding wraps, so even the perimeter matches to rounding."""
    grid = Grid(PeriodicBox(12.0, 12.0), (12, 12))
    fields = seed(grid, np.array([[6.0, 6.0]]), 5.0, 1.0)
    windows = fields.find_windows()
    assert perimeters(fields, windows) == pytest.approx(perimeters(fields), rel=1e-12)
    assert areas(fields, windows) == pytest.approx(areas(fields), rel=1e-12)


# ----------------------------------------------------------------- stencils on patches


def _dense_on_windows(windows, dense):
    """A dense per-cell array read back through the windows, for comparison."""
    return windows.extract(dense)


def test_window_stencils_match_the_grid_operators(big):
    """Inside a window the three stencils equal the grid's, up to the border, where the
    zero padding stands in for a field below the threshold -- so up to threshold / dx^2."""
    fields = relaxed(big)
    windows = fields.find_windows(threshold=THRESHOLD)
    patches = windows.extract(fields.values)
    grid = fields.grid
    border = 4.0 * THRESHOLD / grid.dx**2

    assert windows.laplacian(patches) == pytest.approx(
        _dense_on_windows(windows, grid.laplacian(fields.values)), abs=border, rel=1e-10)
    for ours, theirs in zip(windows.gradient(patches), grid.gradient(fields.values)):
        assert ours == pytest.approx(_dense_on_windows(windows, theirs), abs=border, rel=1e-10)
    for ours, theirs in zip(windows.forward_gradient(patches), grid.forward_gradient(fields.values)):
        assert ours == pytest.approx(_dense_on_windows(windows, theirs), abs=border, rel=1e-10)


def test_window_stencils_are_exact_when_the_window_spans_the_grid():
    """Spanning window: padding wraps, so the stencil *is* the periodic one."""
    grid = Grid(PeriodicBox(12.0, 12.0), (12, 12))
    fields = seed(grid, np.array([[6.0, 6.0]]), 5.0, 1.0)
    windows = fields.find_windows()
    assert windows.shape == grid.shape
    patches = windows.extract(fields.values)
    assert windows.laplacian(patches) == pytest.approx(
        _dense_on_windows(windows, grid.laplacian(fields.values)), rel=1e-12, abs=1e-15)
    for ours, theirs in zip(windows.forward_gradient(patches), grid.forward_gradient(fields.values)):
        assert ours == pytest.approx(_dense_on_windows(windows, theirs), rel=1e-12, abs=1e-15)


def test_window_stencils_wrap_across_the_boundary():
    """A window straddling the edge must see the cell as one piece, not two halves."""
    grid = Grid(PeriodicBox(40.0, 40.0), (40, 40))
    fields = seed(grid, np.array([[0.5, 39.5]]), 4.0, 0.5)
    windows = fields.find_windows(threshold=THRESHOLD)
    assert windows.shape[0] < 40 and windows.shape[1] < 40
    patches = windows.extract(fields.values)
    border = 4.0 * THRESHOLD / grid.dx**2
    assert windows.laplacian(patches) == pytest.approx(
        _dense_on_windows(windows, grid.laplacian(fields.values)), abs=border, rel=1e-10)


def test_forward_gradient_is_the_adjoint_of_the_laplacian_on_windows(big):
    """The pairing the gradient energy relies on must survive the move to windows."""
    fields = relaxed(big)
    windows = fields.find_windows(threshold=THRESHOLD)
    rng = np.random.default_rng(0)
    u = rng.standard_normal((fields.n_cells, *windows.shape))
    v = rng.standard_normal((fields.n_cells, *windows.shape))
    # Zero the border ring, so the zero padding is consistent with the fields.
    for a in (u, v):
        a[:, 0, :] = a[:, -1, :] = a[:, :, 0] = a[:, :, -1] = 0.0
    ux, uy = windows.forward_gradient(u)
    vx, vy = windows.forward_gradient(v)
    lhs = (ux * vx + uy * vy).sum()
    rhs = -(u * windows.laplacian(v)).sum()
    assert lhs == pytest.approx(rhs, rel=1e-12)


# ----------------------------------------------------------------- derivatives on windows

TERMS = [
    DoubleWell(alpha=0.7),
    GradientEnergy(K=1.3),
    Repulsion(epsilon=2.5),
    Adhesion(omega=0.3),
    AreaConstraint(target_area=110.0, lambda_=40.0),
]


def assert_patches_match_dense(windows, patches, dense, rel=1e-5):
    """A windowed result equals the dense one where the cell is, and the dense one is
    negligible where it is not -- both relative to the largest value anywhere."""
    assert patches.shape == (dense.shape[0], *windows.shape)
    scale = float(np.abs(dense).max())
    scattered = windows.scatter(patches, np.zeros_like(dense))
    inside = windows.mask()
    assert np.abs(scattered - dense)[inside].max() <= rel * scale
    assert np.abs(dense)[~inside].max() <= rel * scale


@pytest.mark.parametrize("term", TERMS, ids=lambda t: type(t).__name__)
def test_each_term_derivative_matches_dense_on_windows(term, big):
    fields = relaxed(big)
    windows = fields.find_windows(threshold=THRESHOLD)
    assert_patches_match_dense(
        windows, term.functional_derivative(fields, windows), term.functional_derivative(fields)
    )


def test_shared_sum_accumulated_from_windows_matches_the_dense_sum(big):
    """S = sum_j phi_j^2 built by adding patches into a shared grid, against summing
    the dense stack: equal to the dropped tail, on a tissue that has moved and where
    neighbouring windows overlap so the accumulation has to add rather than assign."""
    fields = relaxed(big)
    windows = fields.find_windows(threshold=THRESHOLD)
    assert windows.mask().sum(axis=0).max() > 1     # windows do overlap on the grid
    energy = big.free_energy()
    dense = energy.shared_quantities(fields)["squared_sum"]
    windowed = energy.shared_quantities(fields, windows)["squared_sum"]
    assert windowed.shape == fields.grid.shape
    assert np.abs(windowed - dense).max() < fields.n_cells * THRESHOLD**2


def test_patches_gathered_once_give_the_same_derivative(big):
    """Handing the stepper's single gather to every term must change nothing."""
    fields = relaxed(big)
    windows = fields.find_windows(threshold=THRESHOLD)
    energy = big.free_energy()
    patches = windows.extract(fields.values)
    with_patches = energy.functional_derivative(fields, windows, patches)
    without = energy.functional_derivative(fields, windows)
    assert np.array_equal(with_patches, without)
    shared = energy.shared_quantities(fields, windows, patches)
    assert shared["patches"] is patches
    assert np.array_equal(shared["squared_sum"], energy.shared_quantities(fields, windows)["squared_sum"])


def test_flat_indices_agree_with_rows_and_columns(big):
    fields = big.tissue(seed=0).fields
    windows = fields.find_windows()
    rows, cols = windows.indices()
    expected = (rows[:, :, None] * fields.grid.nx + cols[:, None, :]).reshape(fields.n_cells, -1)
    assert np.array_equal(windows.flat, expected)
    # scatter then extract is the identity on the windows, and add is a plain sum
    patches = np.random.default_rng(0).normal(size=(fields.n_cells, *windows.shape))
    stack = np.zeros_like(fields.values)
    windows.scatter(patches, stack)
    assert np.array_equal(windows.extract(stack), patches)
    windows.add(patches, stack)
    assert np.allclose(windows.extract(stack), 2 * patches)
    assert np.all(stack[~windows.mask()] == 0.0)


def test_summed_derivative_matches_dense_through_the_shared_sums(big):
    """The coupling terms read S and its Laplacian built once by FreeEnergy."""
    fields = relaxed(big)
    windows = fields.find_windows(threshold=THRESHOLD)
    energy = FreeEnergy(*TERMS)
    assert "squared_sum_laplacian" in energy.shared_quantities(fields)
    assert_patches_match_dense(
        windows, energy.functional_derivative(fields, windows), energy.functional_derivative(fields)
    )


def test_adhesion_without_adhesion_gives_patch_shaped_zeros(big):
    fields = big.tissue(seed=0).fields
    windows = fields.find_windows()
    patches = Adhesion(0.0).functional_derivative(fields, windows)
    assert patches.shape == (fields.n_cells, *windows.shape)
    assert np.all(patches == 0.0)


def test_passive_forces_and_advection_match_dense_on_windows(big):
    fields = relaxed(big)
    windows = fields.find_windows(threshold=THRESHOLD)
    energy = big.free_energy()
    mu_dense = energy.functional_derivative(fields)
    mu_patches = energy.functional_derivative(fields, windows)

    dense_forces = passive_forces(fields, mu_dense)
    windowed_forces = passive_forces(fields, mu_patches, windows=windows)
    assert windowed_forces == pytest.approx(dense_forces, rel=1e-5, abs=1e-5 * np.abs(dense_forces).max())

    velocity = np.random.default_rng(0).normal(size=(fields.n_cells, 2))
    assert_patches_match_dense(
        windows, advection(fields, velocity, windows=windows), advection(fields, velocity)
    )


def test_force_balance_velocities_match_dense_on_windows(big):
    model = big.replace(propulsion="force", active_energy=3.0)
    tissue = model.tissue(seed=0)
    energy = model.free_energy()
    run(tissue, energy, model.stepper(tissue, energy), 50, check=False)
    windows = tissue.fields.find_windows(threshold=THRESHOLD)
    rule = model.propulsion_rule()
    assert isinstance(rule, ForceBalance)
    dense = rule.velocities(tissue, energy.functional_derivative(tissue.fields))
    windowed = rule.velocities(tissue, energy.functional_derivative(tissue.fields, windows),
                               windows=windows)
    assert windowed == pytest.approx(dense, rel=1e-5, abs=1e-5 * np.abs(dense).max())


# ----------------------------------------------------------------- the option


# ----------------------------------------------------------------- stepping on them


def test_a_windowed_step_matches_a_dense_step(big):
    """One step both ways from the same tissue: equal to the tail the windows drop."""
    fields = relaxed(big)
    energy = big.free_energy()
    dense, windowed = Tissue(fields.copy()), Tissue(fields.copy())
    dt = 0.4 * big.max_stable_dt(dense, energy)
    ExplicitEuler(dt=dt, friction=big.friction).step(dense, energy)
    ExplicitEuler(dt=dt, friction=big.friction, windowed=True).step(windowed, energy)
    assert np.abs(windowed.fields.values - dense.fields.values).max() < 1e-5
    # The refresh zeroed everything outside the windows, and nothing has written there.
    assert windowed.fields.windows is not None
    assert np.all(windowed.fields.values[~windowed.fields.windows.mask()] == 0.0)


def test_windowed_dynamics_track_dense_while_a_cell_crosses_the_boundary():
    """A propelled cell straddling the box corner, stepped both ways for long enough to
    move several grid points, so its window has to follow it through the wrap."""
    grid = Grid(PeriodicBox(40.0, 40.0), (160, 160))          # dx = 0.25
    well, gradient = interface_terms(0.5, 1.0)
    energy = FreeEnergy(well, gradient, AreaConstraint.from_radius(4.0, lambda_=20.0))
    centre = np.array([[0.5, 39.5]])

    def fresh():
        tissue = Tissue.seeded(grid, centre, 4.0, 0.5, speed=0.3, rotational_diffusion=0.0,
                               rng=np.random.default_rng(0))
        return tissue

    dense, windowed = fresh(), fresh()
    windowed.polarity.angles[:] = dense.polarity.angles       # same direction of travel
    limit = ExplicitEuler(dt=1e-12, friction=1.0).max_stable_dt(dense, energy)
    common = dict(dt=0.4 * limit, friction=1.0, propulsion=ImposedVelocity())
    # dt is ~0.003 here, so 1500 steps at speed 0.3 move the cell ~1.3 um, over five points.
    run(dense, energy, ExplicitEuler(**common), 1500, check=False)
    run(windowed, energy, ExplicitEuler(**common, windowed=True, refresh_every=20), 1500,
        check=False)

    assert windowed.fields.windows.shape[0] < 160          # a real window, not the grid
    assert np.all(windowed.fields.values[~windowed.fields.windows.mask()] == 0.0)
    assert energy.energy(windowed.fields) == pytest.approx(energy.energy(dense.fields), rel=1e-6)
    offset = grid.box.min_image(centres_of_mass(windowed.fields) - centres_of_mass(dense.fields))
    assert np.hypot(*offset.T).max() < 1e-3
    moved = grid.box.min_image(centres_of_mass(dense.fields) - centre)
    assert np.hypot(*moved.T).max() > 4 * grid.dx           # it did travel several points


# ----------------------------------------------------------------- the option


def test_window_option_changes_the_trajectory_only_by_the_dropped_tail(big):
    dense = simulate(big, duration=30, n_snapshots=4, keep_fields=False)
    windowed = simulate(big.replace(window=True), duration=30, n_snapshots=4, keep_fields=False)
    assert windowed.energies == pytest.approx(dense.energies, rel=1e-6)
    assert windowed.final.areas == pytest.approx(dense.final.areas, rel=1e-6)
    assert windowed.final.perimeters == pytest.approx(dense.final.perimeters, rel=1e-5)
    offset = big.box.min_image(windowed.final.centres - dense.final.centres)
    assert np.hypot(offset[:, 0], offset[:, 1]).max() < 1e-3


def test_window_option_with_force_balance_and_warmup(big):
    """The active path and the warm-up stepper both run on windows without error."""
    model = big.replace(window=True, propulsion="force", active_energy=3.0)
    trajectory = simulate(model, duration=20, warmup=10, n_snapshots=3, keep_fields=False)
    assert np.all(np.isfinite(trajectory.energies))
    assert trajectory.tissue.fields.windows is not None


def test_window_flag_appears_in_the_name_only_when_set():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from naming import encode

    assert "win" not in encode(Model(), 100.0, 0)
    assert "_win_" in encode(Model(window=True), 100.0, 0)
