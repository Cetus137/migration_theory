"""Every free-energy term against its own definition and its own derivative.

The pattern for each: check the energy equals the formula written out literally, then
check the functional derivative against a finite difference of that energy. Together
those catch a wrong coefficient, a dropped factor, a flipped sign, and an inconsistent
discretisation -- which is most of what can go wrong in a term.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from conftest import finite_difference_derivative, sample_significant
from migration_theory import (
    Adhesion,
    AreaConstraint,
    DoubleWell,
    FreeEnergy,
    GradientEnergy,
    Repulsion,
    interface_terms,
    interface_width,
    surface_tension,
)

TERMS = [
    DoubleWell(alpha=0.7),
    GradientEnergy(K=1.3),
    Repulsion(epsilon=2.5),
    Adhesion(omega=0.3),
    AreaConstraint(target_area=110.0, lambda_=40.0),
]


@pytest.mark.parametrize("term", TERMS, ids=lambda t: type(t).__name__)
def test_derivative_matches_a_finite_difference(term, fields):
    energy = FreeEnergy(term)
    analytic = energy.functional_derivative(fields)
    rng = np.random.default_rng(0)
    for index in sample_significant(analytic, 8, rng):
        numeric = finite_difference_derivative(energy, fields, index)
        assert numeric == pytest.approx(analytic[index], rel=2e-4)


@pytest.mark.parametrize("term", TERMS, ids=lambda t: type(t).__name__)
def test_density_integrates_to_the_energy(term, fields):
    if not hasattr(term, "density"):
        pytest.skip("AreaConstraint has no local density")
    assert fields.grid.integrate(term.density(fields)) == pytest.approx(term.energy(fields))


def test_double_well_minima_and_barrier():
    well = DoubleWell(alpha=3.0)
    assert well.potential(0.0) == pytest.approx(0.0)
    assert well.potential(1.0) == pytest.approx(0.0)
    assert well.potential(0.5) == pytest.approx(well.barrier)
    assert well.gradient(np.array([0.0, 0.5, 1.0])) == pytest.approx(0.0, abs=1e-14)


def test_gradient_energy_is_zero_for_a_uniform_field(fields):
    flat = fields.copy()
    flat.values[:] = 0.37
    assert GradientEnergy(K=2.0).energy(flat) == pytest.approx(0.0, abs=1e-20)


def test_repulsion_equals_the_sum_over_pairs(fields):
    """The linear-time identity against the literal double sum it replaces."""
    term = Repulsion(epsilon=2.5)
    literal = sum(
        fields.grid.integrate(fields[i] ** 2 * fields[j] ** 2)
        for i, j in combinations(range(fields.n_cells), 2)
    )
    assert term.energy(fields) == pytest.approx(2.5 * literal, rel=1e-12)


def test_repulsion_density_equals_the_sum_over_pairs_pointwise(fields):
    """Not just the integral: the identity must hold at every grid point."""
    term = Repulsion(epsilon=2.5)
    literal = np.zeros(fields.grid.shape)
    for i, j in combinations(range(fields.n_cells), 2):
        literal += fields[i] ** 2 * fields[j] ** 2
    assert term.density(fields) == pytest.approx(2.5 * literal, rel=1e-12, abs=1e-15)


def test_repulsion_derivative_equals_the_literal_loop(fields):
    """``2 eps phi_k sum_{j != k} phi_j^2``, written out as the loop it used to be."""
    term = Repulsion(epsilon=2.5)
    literal = np.zeros_like(fields.values)
    for k in range(fields.n_cells):
        others = np.zeros(fields.grid.shape)
        for j in range(fields.n_cells):
            if j != k:
                others += fields[j] ** 2
        literal[k] = 2.0 * 2.5 * fields[k] * others
    assert term.functional_derivative(fields) == pytest.approx(literal, rel=1e-12, abs=1e-15)


def test_adhesion_is_exactly_zero_without_adhesion(fields):
    """The omega = 0 shortcut must give what the full computation gives: nothing."""
    term = Adhesion(omega=0.0)
    assert term.energy(fields) == 0.0
    assert np.all(term.density(fields) == 0.0)
    assert np.all(term.functional_derivative(fields) == 0.0)


def test_adhesion_equals_the_sum_over_pairs(fields):
    term = Adhesion(omega=0.3)
    d_dx, d_dy = fields.grid.forward_gradient(fields.values**2)
    literal = sum(
        fields.grid.integrate(d_dx[i] * d_dx[j] + d_dy[i] * d_dy[j])
        for i, j in combinations(range(fields.n_cells), 2)
    )
    assert term.energy(fields) == pytest.approx(0.3 * literal, rel=1e-12)


def test_adhesion_lowers_the_energy_of_a_contact(fields):
    """Contact must be favourable, or the sign is wrong."""
    assert Adhesion(omega=0.3).energy(fields) < 0


@pytest.mark.parametrize("term_type", [Repulsion, Adhesion])
def test_pair_terms_vanish_for_isolated_cells(grid, term_type):
    from migration_theory import seed

    apart = seed(grid, np.array([[1.0, 1.0], [5.0, 4.0]]), 0.6, 0.25)
    term = term_type(1.0) if term_type is Adhesion else term_type(1.0)
    assert term.energy(apart) == pytest.approx(0.0, abs=1e-12)


def test_area_constraint_is_zero_at_its_target(fields):
    from migration_theory import areas

    exact = AreaConstraint(target_area=float(areas(fields).mean()), lambda_=10.0)
    assert exact.energy(fields) == pytest.approx(0.0, abs=1e-3)


def test_area_constraint_drives_the_right_way(fields):
    """Too small must grow, too large must shrink."""
    from migration_theory import areas

    measured = float(areas(fields).mean())
    for factor, expected_sign in ((0.5, +1.0), (2.0, -1.0)):
        term = AreaConstraint(target_area=measured / factor, lambda_=10.0)
        # dphi/dt = -dF/dphi; take its sign where the cell actually is
        rate = -term.functional_derivative(fields)
        inside = fields.values > 0.9
        assert np.sign(rate[inside].mean()) == expected_sign


@pytest.mark.parametrize("width,tension", [(0.5, 1.0), (2.0, 0.25), (1.3, 3.7)])
def test_interface_terms_round_trip(width, tension):
    well, gradient = interface_terms(width, tension)
    assert interface_width(well, gradient) == pytest.approx(width)
    assert surface_tension(well, gradient) == pytest.approx(tension)


def test_surface_tension_of_a_flat_slab():
    """The physics check: slab energy must equal sigma times the interface length."""
    from migration_theory import Grid, PeriodicBox, PhaseFields

    width, tension = 0.1, 1.0
    well, gradient = interface_terms(width, tension)
    grid = Grid(PeriodicBox(4.0, 4.0), (320, 320))
    X, _ = grid.coordinates
    slab = 0.5 * (1 - np.tanh((np.abs(X - 2.0) - 1.0) / (np.sqrt(2) * width)))
    energy = FreeEnergy(well, gradient).energy(PhaseFields(grid, slab[None]))
    assert energy == pytest.approx(2 * tension * grid.box.Ly, rel=1e-3)


def test_equipartition_across_the_interface():
    """Well and gradient must contribute equally -- the first integral of the profile."""
    from migration_theory import Grid, PeriodicBox, PhaseFields

    width = 0.1
    well, gradient = interface_terms(width, 1.0)
    grid = Grid(PeriodicBox(4.0, 4.0), (320, 320))
    X, _ = grid.coordinates
    slab = 0.5 * (1 - np.tanh((np.abs(X - 2.0) - 1.0) / (np.sqrt(2) * width)))
    parts = FreeEnergy(well, gradient).breakdown(PhaseFields(grid, slab[None]))
    assert parts["DoubleWell"] == pytest.approx(parts["GradientEnergy"], rel=2e-3)


def test_terms_are_linear_in_their_coefficient(fields):
    assert Repulsion(2.0).energy(fields) == pytest.approx(2 * Repulsion(1.0).energy(fields))
    assert Adhesion(2.0).energy(fields) == pytest.approx(2 * Adhesion(1.0).energy(fields))


def test_free_energy_sums_its_terms(fields):
    terms = [DoubleWell(0.7), Repulsion(2.5)]
    total = FreeEnergy(*terms)
    assert total.energy(fields) == pytest.approx(sum(t.energy(fields) for t in terms))
    assert total.functional_derivative(fields) == pytest.approx(
        sum(t.functional_derivative(fields) for t in terms)
    )


@pytest.mark.parametrize(
    "construct",
    [
        lambda: DoubleWell(alpha=0.0),
        lambda: GradientEnergy(K=-1.0),
        lambda: Repulsion(epsilon=-1.0),
        lambda: Adhesion(omega=-1.0),
        lambda: AreaConstraint(target_area=0.0),
    ],
)
def test_rejects_unphysical_coefficients(construct):
    with pytest.raises(ValueError):
        construct()
