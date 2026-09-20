"""The Model: derived quantities, the resolution/physics separation, and validation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from migration_theory import Model, simulate


def test_interface_width_and_tension_from_alpha_and_K():
    model = Model(alpha=0.5, K=2.0)
    assert model.interface_width == pytest.approx(np.sqrt(2.0 / 0.5))
    assert model.surface_tension == pytest.approx(np.sqrt(2 * 2.0 * 0.5) / 6)


def test_target_area_is_pi_r_squared():
    model = Model(cell_radius=7.0)
    assert model.target_area == pytest.approx(np.pi * 49.0)


def test_grid_spacing_is_exactly_what_was_asked_for():
    """The whole point of separating dx from the physics."""
    for spacing in (0.25, 0.5, 1.0, 2.0):
        grid = Model(grid_spacing=spacing).grid
        assert grid.dx == pytest.approx(spacing)
        assert grid.dy == pytest.approx(spacing)


def test_refine_changes_only_the_grid():
    """Every physical coefficient must be untouched, or a convergence check is
    meaningless -- it would be comparing two different models."""
    base = Model()
    fine = base.refine(2)
    assert fine.grid_spacing == pytest.approx(base.grid_spacing / 2)
    assert fine.box_points == pytest.approx(2 * base.box_points, rel=0.02)
    for field in dataclasses.fields(Model):
        if field.name != "grid_spacing":
            assert getattr(fine, field.name) == getattr(base, field.name)
    assert fine.interface_width == pytest.approx(base.interface_width)
    assert fine.surface_tension == pytest.approx(base.surface_tension)


def test_refining_leaves_the_physics_alone():
    """The measurement that makes the claim above worth anything."""
    base = Model(n_cells=4, cell_radius=6.0)
    coarse = simulate(base, duration=300, n_snapshots=3, keep_fields=False)
    fine = simulate(base.refine(2), duration=300, n_snapshots=3, keep_fields=False)
    assert fine.final.area_ratio == pytest.approx(coarse.final.area_ratio, rel=0.02)
    assert fine.final.shape_index == pytest.approx(coarse.final.shape_index, rel=0.03)


def test_packing_follows_the_radius():
    dense = Model(cell_radius=12.0, packing=1.0)
    sparse = Model(cell_radius=12.0, packing=0.5)
    assert sparse.box.area == pytest.approx(2 * dense.box.area, rel=0.02)
    assert dense.realised_packing == pytest.approx(1.0, rel=0.02)


def test_free_speed_differs_by_mode():
    assert Model(speed=0.3).free_speed == pytest.approx(0.3)
    forced = Model(propulsion="force", active_energy=6.0, cell_radius=3.0, cell_friction=4.0)
    assert forced.free_speed == pytest.approx(0.5)


def test_activity_is_the_dimensionless_group():
    model = Model(propulsion="force", active_energy=3.0)
    assert model.activity == pytest.approx(
        3.0 / (model.surface_tension * model.cell_radius)
    )


def test_cell_friction_falls_back_to_friction():
    assert Model(friction=7.0).effective_cell_friction == pytest.approx(7.0)
    assert Model(friction=7.0, cell_friction=2.0).effective_cell_friction == pytest.approx(2.0)


def test_cell_cell_tension_and_its_ceiling():
    model = Model(K=2.0, adhesion=1.0)
    assert model.adhesion_ceiling == pytest.approx(model.K)
    assert model.cell_cell_tension == pytest.approx(
        model.surface_tension * (2 - model.adhesion / model.K)
    )


def test_replace_leaves_the_original_alone():
    base = Model(K=2.0)
    assert base.replace(K=5.0).K == 5.0
    assert base.K == 2.0


def test_concerns_flag_the_traps():
    assert any("no clear inside" in note for note in Model(K=200.0).concerns())
    assert any("grid points wide" in note for note in Model(grid_spacing=4.0).concerns())
    assert any("ceiling" in note for note in Model(adhesion=2.0).concerns())
    assert any("not touch" in note for note in Model(packing=0.2).concerns())
    assert Model().concerns() == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_cells": 2},
        {"cell_radius": 0.0},
        {"packing": -1.0},
        {"grid_spacing": 0.0},
        {"safety": 1.5},
        {"timestep": -1.0},
        {"adhesion": -1.0},
        {"cell_friction": 0.0},
        {"propulsion": "teleport"},
    ],
)
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        Model(**kwargs)


def test_summary_covers_every_reported_quantity():
    summary = Model().summary()
    assert set(summary) >= {
        "interface_width", "surface_tension", "cell_radius", "target_area",
        "packing", "grid_spacing", "points_per_interface", "box_points",
        "persistence_time", "shape_relaxation_time",
    }
    assert all(np.isfinite(v) or np.isinf(v) for v in summary.values())
