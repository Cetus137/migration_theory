"""The geometric layer in three dimensions: box, grid, fields and windows.

Step one of making the package dimension-free. The 2D suite is untouched and is the
guard that nothing changed there; these are the same identities in 3D, plus the
places where a third axis can go wrong -- axis order, wrapping across three
boundaries at once, and the window index arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from migration_theory import Grid, PeriodicBox, PhaseFields, seed, seed_tessellated

THRESHOLD = 1e-6


@pytest.fixture
def box3():
    return PeriodicBox(4.0, 5.0, 6.0)


@pytest.fixture
def grid3(box3):
    """Deliberately unequal along every axis, so an axis mix-up cannot pass unnoticed."""
    return Grid(box3, (12, 10, 8))       # (nz, ny, nx): dz = 0.5, dy = 0.5, dx = 0.5


# ----------------------------------------------------------------- box


def test_box_in_three_dimensions(box3):
    assert box3.ndim == 3
    assert (box3.Lx, box3.Ly, box3.Lz) == (4.0, 5.0, 6.0)
    assert box3.volume == pytest.approx(120.0) and box3.area == box3.volume
    assert box3.min_length == 4.0
    assert box3.wrap(np.array([[4.5, -0.5, 6.0]]))[0] == pytest.approx([0.5, 4.5, 0.0])
    dr = box3.min_image(np.array([[3.9, -4.9, 5.9]]))
    assert np.all(np.abs(dr) <= 0.5 * box3.lengths + 1e-12)
    assert dr[0] == pytest.approx([-0.1, 0.1, -0.1])


def test_box_of_two_lengths_is_the_old_box():
    box = PeriodicBox(7.0, 3.0)
    assert box.ndim == 2 and box.Lx == 7.0 and box.Ly == 3.0 and box.area == 21.0
    with pytest.raises(AttributeError):
        box.Lz
    assert repr(box) == "PeriodicBox(7, 3)"


def test_for_cells_uses_the_densest_packing_in_each_dimension():
    flat = PeriodicBox.for_cells(100, spacing=2.0)
    assert flat.ndim == 2 and flat.volume / 100 == pytest.approx(np.sqrt(3) / 2 * 4.0)
    solid = PeriodicBox.for_cells(100, spacing=2.0, dimension=3)
    assert solid.ndim == 3 and solid.volume / 100 == pytest.approx(8.0 / np.sqrt(2))
    assert solid.Lx == solid.Ly == solid.Lz


# ----------------------------------------------------------------- grid


def test_grid_axes_are_z_y_x_and_spacings_follow(grid3):
    assert grid3.ndim == 3
    assert (grid3.nz, grid3.ny, grid3.nx) == (12, 10, 8)
    assert (grid3.dx, grid3.dy, grid3.dz) == pytest.approx((0.5, 0.5, 0.5))
    assert grid3.spacings == pytest.approx((0.5, 0.5, 0.5))
    assert grid3.cell_volume == pytest.approx(0.125) and grid3.cell_area == grid3.cell_volume
    assert grid3.n_points == 960 and grid3.spatial_axes == (-3, -2, -1)
    X, Y, Z = grid3.coordinates
    assert X.shape == Y.shape == Z.shape == (12, 10, 8)
    assert X[0, 0, :] == pytest.approx(np.arange(8) * 0.5)      # x varies along the last axis
    assert Y[0, :, 0] == pytest.approx(np.arange(10) * 0.5)
    assert Z[:, 0, 0] == pytest.approx(np.arange(12) * 0.5)


def test_grid_rejects_a_shape_of_the_wrong_dimension(box3):
    with pytest.raises(ValueError):
        Grid(box3, (10, 8))


def test_integrating_a_constant_gives_the_volume(grid3):
    assert grid3.integrate(np.ones(grid3.shape)) == pytest.approx(grid3.box.volume)
    X, _, _ = grid3.coordinates
    assert grid3.integrate(np.sin(2 * np.pi * X / grid3.box.Lx)) == pytest.approx(0.0, abs=1e-12)


def test_laplacian_of_a_known_mode_is_second_order(box3):
    errors = []
    for points in (16, 32, 64):
        fine = Grid(box3, (points, points, points))
        X, Y, Z = fine.coordinates
        k = 2 * np.pi / box3.lengths
        field = np.sin(k[0] * X) * np.cos(k[1] * Y) * np.sin(k[2] * Z)
        exact = -(k**2).sum() * field
        errors.append(np.abs(fine.laplacian(field) - exact).max() / np.abs(exact).max())
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.15)


def test_gradient_components_come_in_x_y_z_order(grid3):
    X, Y, Z = grid3.coordinates
    k = 2 * np.pi / grid3.box.lengths
    field = np.sin(k[0] * X) * np.sin(k[1] * Y) * np.sin(k[2] * Z)
    d_dx, d_dy, d_dz = grid3.gradient(field)
    assert d_dx == pytest.approx(k[0] * np.cos(k[0] * X) * np.sin(k[1] * Y) * np.sin(k[2] * Z), abs=0.2)
    assert d_dy == pytest.approx(k[1] * np.sin(k[0] * X) * np.cos(k[1] * Y) * np.sin(k[2] * Z), abs=0.2)
    assert d_dz == pytest.approx(k[2] * np.sin(k[0] * X) * np.sin(k[1] * Y) * np.cos(k[2] * Z), abs=0.2)


def test_divergence_of_the_gradient_converges_to_the_laplacian(box3):
    """Two central differences compose to the wide 2dx stencil, not the compact one, so
    the two agree only in the limit -- second order, like everything else here."""
    errors = []
    for points in (16, 32, 64):
        fine = Grid(box3, (points, points, points))
        X, Y, Z = fine.coordinates
        k = 2 * np.pi / box3.lengths
        field = np.sin(k[0] * X) * np.cos(k[1] * Y) * np.sin(k[2] * Z)
        exact = -(k**2).sum() * field
        composed = fine.divergence(*fine.gradient(field))
        errors.append(np.abs(composed - exact).max() / np.abs(exact).max())
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.15)


def test_forward_gradient_is_the_adjoint_of_the_laplacian_in_3d(grid3):
    rng = np.random.default_rng(0)
    u, v = rng.standard_normal(grid3.shape), rng.standard_normal(grid3.shape)
    lhs = sum(grid3.integrate(a * b) for a, b in zip(grid3.forward_gradient(u), grid3.forward_gradient(v)))
    assert lhs == pytest.approx(-grid3.integrate(u * grid3.laplacian(v)), rel=1e-12)


def test_centre_of_mass_of_a_blob_at_a_corner_in_3d(box3):
    """A Gaussian resolved by the grid (sigma > 2 dx, so sampling does not alias) and
    narrow next to the box (nothing within half a box of wrapping onto itself)."""
    fine = Grid(box3, (48, 40, 32))                          # dx = dy = dz = 0.125
    for centre in ([0.2, 4.8, 0.1], [3.9, 0.1, 5.9], [2.0, 2.5, 3.0]):
        weights = np.exp(-(fine.distance_to(np.array(centre)) ** 2) / 0.15)
        recovered = fine.centre_of_mass(weights)
        assert np.linalg.norm(fine.box.min_image(recovered - np.array(centre))) < 1e-9


def test_operators_broadcast_over_a_leading_cell_axis(grid3):
    stack = np.random.default_rng(0).standard_normal((3, *grid3.shape))
    assert grid3.laplacian(stack).shape == stack.shape
    assert grid3.integrate(stack).shape == (3,)
    assert grid3.centre_of_mass(stack**2).shape == (3, 3)
    assert all(g.shape == stack.shape for g in grid3.gradient(stack))


# ----------------------------------------------------------------- fields and windows


@pytest.fixture
def corner_cell():
    """One cell straddling all three boundaries of a 64^3 box.

    Radius 3 and interface 1 on a unit grid: resolved enough for its sampled profile
    to be symmetric to a small fraction of a point, and with a window -- about 51
    points at the default threshold -- that is genuinely smaller than the grid.
    """
    grid = Grid(PeriodicBox(64.0, 64.0, 64.0), (64, 64, 64))
    fields = seed(grid, np.array([[0.5, 63.5, 0.3]]), 3.0, 1.0)
    return grid, fields


def test_seeded_cell_has_the_right_volume_and_wraps(corner_cell):
    grid, fields = corner_cell
    assert fields.values.shape == (1, 64, 64, 64)
    # INT phi^2 of a tanh sphere undershoots 4/3 pi R^3 by the tail, as in 2D.
    volume = grid.integrate(fields.values**2)[0]
    assert 0.3 * (4 / 3 * np.pi * 27) < volume < 4 / 3 * np.pi * 27
    assert fields[0][0, 63, 0] > 0.9      # the centre sits across three edges at once
    # The wrap check: a centroid computed without the minimum image would land
    # tens of microns away. Resolution limits the agreement, not the wrapping.
    recovered = grid.centre_of_mass(fields.values**2)[0]
    assert np.linalg.norm(grid.box.min_image(recovered - np.array([0.5, 63.5, 0.3]))) < 0.05


def test_windows_cover_and_wrap_across_three_boundaries(corner_cell):
    grid, fields = corner_cell
    windows = fields.find_windows(threshold=THRESHOLD)
    assert windows.ndim == 3 and windows.origins.shape == (1, 3)
    assert all(n < 64 for n in windows.shape)                 # a real window, not the grid
    assert np.all(fields.values[~windows.mask()] <= THRESHOLD)
    for index in windows.indices():
        assert index.min() == 0 and index.max() == 63          # wraps on every axis
    # Extracting through the wrap equals rolling the cell to the origin and slicing.
    origin = windows.origins[0]
    rolled = fields[0]
    for axis, shift in enumerate(origin):
        rolled = np.roll(rolled, -shift, axis=axis)
    slices = tuple(slice(0, n) for n in windows.shape)
    assert windows.extract(fields.values)[0] == pytest.approx(rolled[slices])


def test_window_transfers_are_consistent_in_3d(corner_cell):
    grid, fields = corner_cell
    windows = fields.find_windows()
    patches = np.random.default_rng(0).normal(size=(1, *windows.shape))
    stack = np.zeros_like(fields.values)
    windows.scatter(patches, stack)
    assert np.array_equal(windows.extract(stack), patches)
    windows.add(patches, stack)
    assert np.allclose(windows.extract(stack), 2 * patches)
    assert np.all(stack[~windows.mask()] == 0.0)
    shared = windows.accumulate(patches, np.zeros(grid.shape))
    assert shared.sum() == pytest.approx(patches.sum())
    X, Y, Z = windows.coordinates()
    assert X.shape == (1, *windows.shape)
    assert np.diff(X[0, 0, 0, :]) == pytest.approx(grid.dx)     # x runs along the last axis
    assert np.diff(Z[0, :, 0, 0]) == pytest.approx(grid.dz)


def test_window_stencils_match_the_grid_operators_in_3d(corner_cell):
    grid, fields = corner_cell
    windows = fields.find_windows(threshold=THRESHOLD)
    patches = windows.extract(fields.values)
    border = 6.0 * THRESHOLD / grid.dx**2
    assert windows.laplacian(patches) == pytest.approx(
        windows.extract(grid.laplacian(fields.values)), abs=border, rel=1e-10)
    for ours, theirs in zip(windows.gradient(patches), grid.gradient(fields.values)):
        assert ours == pytest.approx(windows.extract(theirs), abs=border, rel=1e-10)
    for ours, theirs in zip(windows.forward_gradient(patches), grid.forward_gradient(fields.values)):
        assert ours == pytest.approx(windows.extract(theirs), abs=border, rel=1e-10)


def test_spanning_window_is_exact_in_3d():
    grid = Grid(PeriodicBox(10.0, 10.0, 10.0), (10, 10, 10))
    fields = seed(grid, np.array([[5.0, 5.0, 5.0]]), 4.0, 1.0)
    windows = fields.find_windows()
    assert windows.shape == grid.shape and all(windows.spans(a) for a in range(3))
    patches = windows.extract(fields.values)
    assert windows.laplacian(patches) == pytest.approx(
        windows.extract(grid.laplacian(fields.values)), rel=1e-12, abs=1e-15)


def test_tessellated_seeding_tiles_a_3d_box():
    grid = Grid(PeriodicBox(24.0, 24.0, 24.0), (24, 24, 24))
    centres = np.array([[6.0, 6.0, 6.0], [18.0, 6.0, 12.0], [12.0, 18.0, 18.0], [6.0, 18.0, 6.0]])
    fields = seed_tessellated(grid, centres, 8.0, 1.0)
    assert fields.values.shape == (4, 24, 24, 24)
    assert np.all(fields.values >= 0) and np.all(fields.values <= 1)
    assert 0.5 < fields.occupancy.mean() < 1.5
    assert fields.occupancy.max() < 1.6           # tails overlap at vertices, cells do not
    # Each cell owns its own centre and is nearly absent at the others'.
    for i, centre in enumerate(centres):
        index = tuple(int(round(c)) for c in centre[::-1])       # (z, y, x) into the array
        assert fields[i][index] > 0.9
        assert all(fields[j][index] < 0.1 for j in range(4) if j != i)
