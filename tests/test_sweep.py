"""The sweep script's mapping from swept quantities to models.

The script lives in ``scripts/`` rather than the package, so it is imported from there.
What is checked is the part that decides what each array task simulates: how the
pseudo-quantities ``tension`` and ``activity`` set the model's fields, and that the
point numbering is the one the job scripts document.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sweep  # noqa: E402
from migration_theory import Model  # noqa: E402


def test_tension_sets_alpha_and_K_at_fixed_width():
    base = Model()
    for tension in (0.1, 0.2357, 0.5, 1.0):
        model = sweep.apply_point(base, ["tension"], (tension,))
        assert model.surface_tension == pytest.approx(tension)
        assert model.interface_width == pytest.approx(base.interface_width)
        assert (model.alpha, model.K) != (base.alpha, base.K) or tension == pytest.approx(base.surface_tension, rel=1e-3)


def test_activity_sets_active_energy_from_the_swept_tension():
    """Whatever order the fields come in, activity reads the tension of *this* point."""
    base = Model(propulsion="force")
    for fields in (["activity", "tension"], ["tension", "activity"]):
        values = {"activity": 3.0, "tension": 0.5}
        model = sweep.apply_point(base, fields, tuple(values[f] for f in fields))
        assert model.surface_tension == pytest.approx(0.5)
        assert model.active_energy == pytest.approx(3.0 * 0.5 * model.cell_radius)
        assert model.activity == pytest.approx(3.0)


def test_a_fixed_activity_follows_the_swept_tension():
    """Sweeping tension against friction at one dimensionless activity: E_a is set from
    each point's own tension, so a = E_a/(sigma R) is the same everywhere, while the
    remodelling rate K/gamma changes with both axes."""
    base = Model(propulsion="force", active_energy=99.0)
    for tension, friction in ((0.1, 5.0), (0.5, 20.0)):
        model = sweep.apply_point(base, ["tension", "friction"], (tension, friction), activity=3.0)
        assert model.surface_tension == pytest.approx(tension)
        assert model.friction == friction
        assert model.activity == pytest.approx(3.0)
        assert model.active_energy == pytest.approx(3.0 * tension * model.cell_radius)
    # Not swept and not asked for: the absolute active energy stands.
    assert sweep.apply_point(base, ["tension"], (0.5,)).active_energy == 99.0
    # Swept activity or free speed wins over the fixed one.
    swept = sweep.apply_point(base, ["activity", "tension"], (5.0, 0.5), activity=3.0)
    assert swept.activity == pytest.approx(5.0)
    fast = sweep.apply_point(base, ["free_speed"], (0.1,), activity=3.0)
    assert fast.free_speed == pytest.approx(0.1)


def test_free_speed_holds_the_speed_and_scales_the_force_with_friction():
    """At one free speed on three frictions the speed is equal and E_a rises with xi."""
    base = Model(propulsion="force")
    models = [sweep.apply_point(base, ["free_speed", "cell_friction"], (0.1, xi)) for xi in (3.0, 10.0, 30.0)]
    for model, xi in zip(models, (3.0, 10.0, 30.0)):
        assert model.effective_cell_friction == xi
        assert model.free_speed == pytest.approx(0.1)
        assert model.active_energy == pytest.approx(0.1 * model.cell_radius * xi)
    forces = [m.active_energy / m.cell_radius for m in models]
    assert forces[1] == pytest.approx(10 / 3 * forces[0]) and forces[2] == pytest.approx(3 * forces[1])


def test_activity_and_free_speed_cannot_both_be_swept():
    with pytest.raises(SystemExit):
        sweep.apply_point(Model(), ["activity", "free_speed"], (1.0, 0.1))


def test_plain_fields_still_go_straight_in():
    model = sweep.apply_point(Model(), ["active_energy", "adhesion"], (12.0, 0.3))
    assert model.active_energy == 12.0 and model.adhesion == 0.3


def test_grid_peclet_is_independent_of_tension_at_fixed_activity():
    """The reason the activity axis is safe across a decade of tension: the free speed
    scales with sigma and so does K, so v dx gamma / K does not move."""
    base = Model(propulsion="force", epsilon=40.0)
    peclets = []
    for tension in (0.1, 1.0):
        model = sweep.apply_point(base, ["activity", "tension"], (8.0, tension))
        peclets.append(model.free_speed * model.grid_spacing * model.friction / model.K)
    assert peclets[0] == pytest.approx(peclets[1])
    assert peclets[0] < 2.0


def test_sweepable_lists_the_pseudo_quantities_but_not_flags_or_strings():
    assert "tension" in sweep.SWEEPABLE and "activity" in sweep.SWEEPABLE
    assert "window" not in sweep.SWEEPABLE and "seeding" not in sweep.SWEEPABLE
    assert "active_energy" in sweep.SWEEPABLE


def test_point_numbering_matches_the_job_scripts():
    """Seed fastest, then the second field, then the first."""

    class Args:
        over = ["activity", "tension"]
        values = [[0.0, 1.0, 2.0], [0.1, 0.5]]
        seeds = [0, 1]

    fields, axes, points = sweep.sweep_points(Args)
    assert fields == ["activity", "tension"]
    assert len(points) == 3 * 2 * 2
    n2, ns = 2, 2
    for k, (combination, seed) in enumerate(points):
        assert combination == (axes[0][k // (n2 * ns)], axes[1][(k // ns) % n2])
        assert seed == Args.seeds[k % ns]
