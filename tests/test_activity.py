"""Polarity, propulsion, and the mechanical forces between cells."""

from __future__ import annotations

import numpy as np
import pytest

from migration_theory import (
    ForceBalance,
    FreeEnergy,
    Grid,
    ImposedVelocity,
    Model,
    PeriodicBox,
    Polarity,
    Repulsion,
    advection,
    passive_forces,
    run,
    seed,
    simulate,
    tracks,
)


def test_polarity_is_a_wiener_process():
    """``<dtheta^2> = 2 D_r t``, which is what defines Eq. (3)."""
    diffusion, step = 0.01, 0.05
    polarity = Polarity.random(4000, rotational_diffusion=diffusion,
                               rng=np.random.default_rng(0))
    start = polarity.angles.copy()
    for _ in range(400):
        polarity.rotate(step)
    elapsed = 400 * step
    assert np.mean((polarity.angles - start) ** 2) == pytest.approx(
        2 * diffusion * elapsed, rel=0.08
    )


def test_direction_autocorrelation_decays_as_expected():
    """``<n(0).n(t)> = exp(-D_r t)`` follows from the same process."""
    diffusion, step = 0.01, 0.05
    polarity = Polarity.random(4000, rotational_diffusion=diffusion,
                               rng=np.random.default_rng(1))
    start = polarity.directors.copy()
    for _ in range(400):
        polarity.rotate(step)
    elapsed = 400 * step
    measured = (polarity.directors * start).sum(axis=1).mean()
    assert measured == pytest.approx(np.exp(-diffusion * elapsed), rel=0.08)


def test_polarity_is_frozen_without_rotational_diffusion():
    polarity = Polarity.random(10, rotational_diffusion=0.0, rng=np.random.default_rng(0))
    before = polarity.angles.copy()
    polarity.rotate(1.0)
    assert polarity.angles == pytest.approx(before)


def test_advection_translates_a_field_rigidly(grid):
    """With no free energy, a cell must move at exactly its velocity."""
    fields = seed(grid, np.array([[4.0, 3.0]]), 1.0, 0.3)
    velocity = np.array([[0.4, -0.2]])
    step, steps = 0.02, 40
    start = grid.centre_of_mass(fields.values**2)[0]
    for _ in range(steps):
        fields.values += step * advection(fields, velocity)
    moved = grid.box.min_image(grid.centre_of_mass(fields.values**2)[0] - start)
    assert moved == pytest.approx(velocity[0] * step * steps, rel=0.05)


def test_passive_forces_sum_to_zero(model):
    """Newton's third law. Follows from the free energy being translation invariant.

    Exact only in the continuum. On the grid the residual is discretisation error:
    small for a smooth tissue and falling with refinement, which is what is checked.
    White noise on the fields breaks it outright -- measured, the jittered fixture
    gives a residual of 0.6 -- so this uses a relaxed tissue instead.
    """

    def residual(m):
        energy = m.free_energy()
        tissue = m.tissue(seed=0)
        stepper = m.stepper(tissue, energy)
        run(tissue, energy, stepper, int(round(10.0 / stepper.dt)), check=False)
        forces = passive_forces(tissue.fields, energy.functional_derivative(tissue.fields))
        return np.abs(forces.sum(axis=0)).max() / np.abs(forces).max()

    coarse, fine = residual(model), residual(model.refine(2))
    assert coarse < 1e-3
    assert fine < coarse / 2


def test_passive_forces_push_overlapping_cells_apart():
    """The sign check. Attraction here would invert the whole mechanics."""
    grid = Grid(PeriodicBox(40.0, 40.0), (40, 40))
    fields = seed(grid, np.array([[18.0, 20.0], [22.0, 20.0]]), 6.0, 1.0)
    forces = passive_forces(fields, FreeEnergy(Repulsion(10.0)).functional_derivative(fields))
    assert forces[0, 0] < 0 < forces[1, 0]
    assert forces.sum(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_a_cached_gradient_changes_nothing(fields, free_energy):
    """The stepper hands one gradient to both consumers; that must be a pure saving."""
    mu = free_energy.functional_derivative(fields)
    gradient = fields.gradient()
    velocity = np.random.default_rng(0).normal(size=(fields.n_cells, 2))
    assert passive_forces(fields, mu, gradient) == pytest.approx(passive_forces(fields, mu))
    assert advection(fields, velocity, gradient) == pytest.approx(advection(fields, velocity))


def test_force_balance_free_speed():
    rule = ForceBalance(active_energy=6.0, cell_radius=3.0, cell_friction=4.0)
    assert rule.free_speed == pytest.approx(6.0 / (3.0 * 4.0))


def test_model_selects_the_propulsion_rule():
    assert Model().propulsion_rule() is None                      # passive
    assert isinstance(Model(speed=0.1).propulsion_rule(), ImposedVelocity)
    forced = Model(propulsion="force", active_energy=1.0).propulsion_rule()
    assert isinstance(forced, ForceBalance)


def test_force_balance_resists_crowding_where_imposed_velocity_does_not(model):
    """The reason force balance exists.

    At matched free speed the imposed mode drives cells through each other while force
    balance lets them stall, so the crowded one must end up with more overlap.
    """
    from migration_theory import overlap_matrix

    def overlap(trajectory):
        contacts = overlap_matrix(trajectory.tissue.fields)
        np.fill_diagonal(contacts, 0.0)
        return contacts.sum() / 2

    speed = 0.08
    friction, radius = model.friction, model.cell_radius
    imposed = simulate(model.replace(speed=speed), duration=300, warmup=200,
                       n_snapshots=4, keep_fields=False)
    balanced = simulate(
        model.replace(propulsion="force", active_energy=speed * radius * friction),
        duration=300, warmup=200, n_snapshots=4, keep_fields=False,
    )
    assert overlap(balanced) < overlap(imposed)


def test_velocities_have_the_right_shape(model):
    tissue = model.replace(propulsion="force", active_energy=1.0).tissue(seed=0)
    energy = model.free_energy()
    rule = model.replace(propulsion="force", active_energy=1.0).propulsion_rule()
    velocities = rule.velocities(tissue, energy.functional_derivative(tissue.fields))
    assert velocities.shape == (model.n_cells, 2)


def test_tracks_unwrap_through_the_boundary(model):
    """A cell crossing the edge must not register a box-sized jump."""
    moving = model.replace(speed=0.05, rotational_diffusion=0.0)
    trajectory = simulate(moving, duration=400, n_snapshots=60, keep_fields=False)
    path = tracks(trajectory)
    assert path.max_step < moving.box.min_length / 2
    assert path.positions.shape == (len(trajectory.snapshots), model.n_cells, 2)
