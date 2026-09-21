"""The periodic grid and its operators."""

from __future__ import annotations

import numpy as np
import pytest

from migration_theory import Grid, PeriodicBox


def test_spacing_matches_the_box(grid):
    assert grid.dx == pytest.approx(grid.box.Lx / grid.nx)
    assert grid.dy == pytest.approx(grid.box.Ly / grid.ny)
    assert grid.cell_area == pytest.approx(grid.dx * grid.dy)


def test_integrating_a_constant_gives_the_box_area(grid):
    assert grid.integrate(np.ones(grid.shape)) == pytest.approx(grid.box.area)


def test_integrating_a_full_period_gives_zero(grid):
    X, _ = grid.coordinates
    assert grid.integrate(np.sin(2 * np.pi * X / grid.box.Lx)) == pytest.approx(0.0, abs=1e-12)


def test_laplacian_of_a_known_mode(grid):
    """Second order, so the error must fall as dx^2 -- checked by refining."""
    errors = []
    for points in (32, 64, 128):
        fine = Grid(grid.box, (points, points))
        X, Y = fine.coordinates
        kx, ky = 2 * np.pi / fine.box.Lx, 2 * np.pi / fine.box.Ly
        field = np.sin(kx * X) * np.cos(ky * Y)
        exact = -(kx**2 + ky**2) * field
        errors.append(np.abs(fine.laplacian(field) - exact).max() / np.abs(exact).max())
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.15)  # second order


def test_gradient_of_a_known_mode(grid):
    X, Y = grid.coordinates
    kx, ky = 2 * np.pi / grid.box.Lx, 2 * np.pi / grid.box.Ly
    field = np.sin(kx * X) * np.sin(ky * Y)
    d_dx, d_dy = grid.gradient(field)
    assert d_dx == pytest.approx(kx * np.cos(kx * X) * np.sin(ky * Y), abs=0.05)
    assert d_dy == pytest.approx(ky * np.sin(kx * X) * np.cos(ky * Y), abs=0.05)


def test_forward_gradient_is_the_adjoint_of_the_laplacian(grid):
    """The pairing the gradient energy depends on.

    ``(D+)^T D+ = -laplacian`` exactly, in the discrete sense. If it did not hold, the
    functional derivative of the gradient energy would not be the derivative of the
    discrete energy -- an error of a factor of ~28, measured, when central differences
    are used instead.
    """
    rng = np.random.default_rng(0)
    u = rng.standard_normal(grid.shape)
    v = rng.standard_normal(grid.shape)
    ux, uy = grid.forward_gradient(u)
    vx, vy = grid.forward_gradient(v)
    assert grid.integrate(ux * vx + uy * vy) == pytest.approx(
        -grid.integrate(u * grid.laplacian(v)), rel=1e-12
    )


def test_centre_of_mass_of_a_blob_straddling_the_boundary(grid):
    """The failure a plain weighted mean would show: the centre lands mid-box."""
    for centre in ([0.2, 3.0], [4.0, 0.05], [0.1, 0.1]):
        weights = np.exp(-(grid.distance_to(np.array(centre)) ** 2) / 0.5)
        recovered = grid.centre_of_mass(weights)
        offset = grid.box.min_image(recovered - np.array(centre))
        assert np.hypot(*offset) < 0.05


def test_centre_of_mass_is_the_exact_centroid_for_an_asymmetric_blob(grid):
    """The circular mean alone is biased by the third moment; the refinement must
    remove it. Compare with the plain centroid of the same blob placed mid-box, where
    no wrap is involved and the ordinary weighted mean is the truth."""
    X, Y = grid.coordinates
    Lx, Ly = grid.box.Lx, grid.box.Ly

    def blob(cx, cy):
        dx = grid.box.min_image(np.stack([X - cx, Y - cy], axis=-1))
        # Skewed: a Gaussian core with a one-sided tail along +x. Narrow enough that
        # nothing reaches half the box, so the plain mean of the mid-box copy is exact.
        return np.exp(-(dx[..., 0] ** 2 + dx[..., 1] ** 2) / 0.3) * (1.0 + 0.8 * np.tanh(dx[..., 0]))

    mid = blob(Lx / 2, Ly / 2)
    truth = np.array([np.sum(mid * X), np.sum(mid * Y)]) / np.sum(mid)      # no wrap: plain mean
    expected_shift = truth - np.array([Lx / 2, Ly / 2])                       # the skew's offset

    for cx, cy in ((0.1, Ly / 2), (Lx - 0.2, 0.3), (Lx / 2, Ly - 0.1)):
        recovered = grid.centre_of_mass(blob(cx, cy))
        offset = grid.box.min_image(recovered - (np.array([cx, cy]) + expected_shift))
        assert np.hypot(*offset) < 1e-9


def test_centre_of_mass_of_an_empty_field_is_nan(grid):
    assert np.all(np.isnan(grid.centre_of_mass(np.zeros(grid.shape))))


def test_distance_uses_the_shortest_way_round(grid):
    distance = grid.distance_to(np.array([0.0, 0.0]))
    assert distance.max() <= np.hypot(grid.box.Lx / 2, grid.box.Ly / 2) + 1e-12
    assert distance[0, 0] == pytest.approx(0.0)


def test_operators_broadcast_over_a_leading_axis(grid):
    """Fields are stacked by cell, so every operator must act on the trailing axes."""
    stack = np.random.default_rng(0).standard_normal((3, *grid.shape))
    assert grid.laplacian(stack).shape == stack.shape
    assert grid.integrate(stack).shape == (3,)
    assert grid.centre_of_mass(stack**2).shape == (3, 2)


def test_rejects_a_grid_too_small_to_difference():
    with pytest.raises(ValueError):
        Grid(PeriodicBox(1.0, 1.0), (2, 2))
