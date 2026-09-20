"""Time stepping: the identities a gradient flow has to satisfy, and the guards."""

from __future__ import annotations

import numpy as np
import pytest

from migration_theory import (
    Diverged,
    ExplicitEuler,
    FreeEnergy,
    GradientEnergy,
    Model,
    UnstableTimestep,
    interface_terms,
    run,
    simulate,
)


def test_passive_energy_decreases_monotonically(model):
    """Gradient flow can only go downhill. The single most informative check there is."""
    trajectory = simulate(model, duration=60, n_snapshots=25, keep_fields=False)
    assert np.all(np.diff(trajectory.energies) < 0)


def test_energy_falls_at_the_rate_the_equation_says(model):
    r"""``dF/dt = -(1/gamma) INT (dF/dphi)^2``.

    Ties the stepper to the free energy quantitatively rather than just checking the
    sign -- it would catch a wrong friction, a wrong sign, or a missing factor.
    """
    tissue = model.tissue(seed=0)
    energy = model.free_energy()
    stepper = model.stepper(tissue, energy)

    derivative = energy.functional_derivative(tissue.fields)
    predicted = -tissue.grid.integrate((derivative**2).sum(axis=0)) / model.friction

    before = energy.energy(tissue.fields)
    stepper.step(tissue, energy)
    measured = (energy.energy(tissue.fields) - before) / stepper.dt
    assert measured == pytest.approx(predicted, rel=1e-3)


def test_each_term_reports_a_stability_limit(model):
    tissue = model.tissue(seed=0)
    energy = model.free_energy()
    limits = ExplicitEuler(dt=1e-12, friction=model.friction).stability_limits(tissue, energy)
    assert set(limits) >= {"DoubleWell", "GradientEnergy", "Repulsion", "AreaConstraint"}
    assert all(value > 0 for value in limits.values())


def test_gradient_limit_matches_the_diffusive_formula(grid):
    """``gamma dx^2 / (4K)`` on a square grid."""
    from migration_theory import PhaseFields

    square = type(grid)(grid.box, (32, 32))
    fields = PhaseFields(square, np.zeros((2, *square.shape)))
    term, friction = GradientEnergy(K=1.7), 3.0
    expected = 2.0 * friction / (term.K * (4 / square.dx**2 + 4 / square.dy**2))
    assert term.stability_limit(fields, friction) == pytest.approx(expected)


def test_too_large_a_timestep_is_refused_by_name(model):
    tissue = model.tissue(seed=0)
    energy = model.free_energy()
    limit = model.max_stable_dt(tissue, energy)
    with pytest.raises(UnstableTimestep, match="stability limit"):
        ExplicitEuler(dt=10 * limit, friction=model.friction).check(tissue, energy)


def test_an_unbounded_free_energy_raises_rather_than_returning_nan(model):
    """Adhesion above its ceiling. The run must fail loudly, not return NaN."""
    runaway = model.replace(adhesion=4.0 * model.K)
    with pytest.raises(Diverged, match="finite"):
        simulate(runaway, duration=200, n_snapshots=8, keep_fields=False)


def test_friction_only_rescales_time_for_a_passive_run(model):
    """A passive steady state cannot depend on gamma -- it is a time unit there."""
    slow = model.replace(friction=4 * model.friction)
    fast = simulate(model, 400, n_snapshots=3, keep_fields=False)
    slower = simulate(slow, 400, n_snapshots=3, keep_fields=False)
    assert slower.dt == pytest.approx(4 * fast.dt)
    assert slower.tissue.fields.values == pytest.approx(fast.tissue.fields.values, abs=1e-6)


def test_curvature_flow_shrinks_a_droplet_at_the_predicted_rate():
    """``dA/dt = -2 pi K / gamma`` in the sharp-interface limit."""
    from migration_theory import Grid, PeriodicBox, Tissue, areas

    width, friction = 0.1, 1.0
    well, gradient = interface_terms(width, 1.0)
    grid = Grid.from_spacing(PeriodicBox(4.0, 4.0), width / 3)
    tissue = Tissue.seeded(grid, np.array([[2.0, 2.0]]), 1.0, width)
    energy = FreeEnergy(well, gradient)
    limit = ExplicitEuler(dt=1e-12, friction=friction).max_stable_dt(tissue, energy)
    stepper = ExplicitEuler(dt=0.4 * limit, friction=friction)

    times, measured = [], []
    for _ in range(8):
        run(tissue, energy, stepper, 150, check=False)
        times.append(tissue.time)
        measured.append(areas(tissue.fields)[0])
    slope = np.polyfit(times, measured, 1)[0]
    assert slope == pytest.approx(-2 * np.pi * gradient.K / friction, rel=0.12)


def test_area_constraint_residual_falls_as_lambda_rises(model):
    from migration_theory import areas

    residuals = []
    for stiffness in (1.0, 100.0):
        tuned = model.replace(area_lambda=stiffness)
        trajectory = simulate(tuned, duration=400, n_snapshots=3, keep_fields=False)
        measured = areas(trajectory.tissue.fields).mean()
        residuals.append(abs(1 - measured / tuned.target_area))
    assert residuals[1] < residuals[0] / 5
