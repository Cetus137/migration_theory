"""Shared fixtures.

Everything here is deliberately tiny. The point of the suite is to catch a coefficient
flipping sign or an operator losing its adjoint, and that shows up on a 20x20 grid just
as clearly as on 200x200 -- at a thousandth of the cost. Nothing here should take more
than a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from migration_theory import (  # noqa: E402
    FreeEnergy,
    Grid,
    Model,
    PeriodicBox,
    PhaseFields,
    seed,
)


@pytest.fixture
def box() -> PeriodicBox:
    return PeriodicBox(7.0, 5.0)


@pytest.fixture
def grid() -> Grid:
    """Deliberately non-square, so an x/y mix-up cannot pass unnoticed."""
    return Grid(PeriodicBox(8.0, 6.0), (24, 32))


@pytest.fixture
def model() -> Model:
    """Four cells on a ~21x21 grid. Small enough to simulate inside a test."""
    return Model(n_cells=4, cell_radius=6.0, grid_spacing=1.0)


@pytest.fixture
def fields(model: Model) -> PhaseFields:
    """A relaxed-ish configuration: seeded, then jittered so nothing is symmetric.

    Perturbing matters -- a symmetric configuration can make a wrong sign or a dropped
    cross-term integrate to zero and look correct.
    """
    tissue = model.tissue(seed=0)
    rng = np.random.default_rng(0)
    tissue.fields.values += 0.03 * rng.standard_normal(tissue.fields.values.shape)
    return tissue.fields


@pytest.fixture
def free_energy(model: Model) -> FreeEnergy:
    return model.free_energy()


def finite_difference_derivative(free_energy, fields, index, step=1e-5):
    """``dF/dphi`` at one grid point, by central difference of the discrete energy.

    Dividing by the cell volume converts the derivative with respect to the stored value
    into the functional derivative, which is what the terms return. ``index`` is
    ``(cell, *grid_index)`` in any dimension.
    """
    index = tuple(index)
    up, down = fields.copy(), fields.copy()
    up.values[index] += step
    down.values[index] -= step
    return (free_energy.energy(up) - free_energy.energy(down)) / (
        2.0 * step * fields.grid.cell_volume
    )


def sample_significant(array, count, rng, fraction=0.3):
    """Indices where ``array`` is large, so relative errors are meaningful.

    Sampling uniformly would mostly hit points where the derivative is ~0, and a
    relative error there is dominated by roundoff rather than by correctness.
    """
    candidates = np.argwhere(np.abs(array) > fraction * np.abs(array).max())
    chosen = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    return [tuple(candidates[i]) for i in chosen]
