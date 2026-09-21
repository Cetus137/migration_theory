"""Per-cell windows: finding them, reading them across the boundary, measuring on them.

Stage one of windowed storage. Nothing here may change a result: every windowed
measurement is held against the dense one it replaces.
"""

from __future__ import annotations

import numpy as np
import pytest

from migration_theory import (
    Grid,
    Model,
    PeriodicBox,
    PhaseFields,
    Windows,
    areas,
    centres_of_mass,
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
    windows = fields.windows(threshold=THRESHOLD)
    assert windows.shape[0] < fields.grid.ny and windows.shape[1] < fields.grid.nx
    outside = ~windows.mask()
    assert np.all(fields.values[outside] <= THRESHOLD)


def test_windows_still_cover_after_the_tissue_has_moved(big):
    fields = relaxed(big)
    windows = fields.windows(threshold=THRESHOLD)
    assert np.all(fields.values[~windows.mask()] <= THRESHOLD)


def test_a_cell_on_the_boundary_gets_a_wrapped_window():
    """The occupied run straddles both edges; the window must follow it round."""
    grid = Grid(PeriodicBox(40.0, 40.0), (40, 40))
    fields = seed(grid, np.array([[0.5, 39.5]]), 4.0, 0.5)   # sharp, so the window is small
    windows = fields.windows(threshold=THRESHOLD)
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
    windows = fields.windows()
    patches = windows.extract(fields.values)
    assert patches.shape == (fields.n_cells, *windows.shape)
    rows, cols = windows.indices()
    for i in range(fields.n_cells):
        assert patches[i] == pytest.approx(fields[i][np.ix_(rows[i], cols[i])])


def test_a_box_barely_bigger_than_a_cell_gives_the_whole_grid():
    grid = Grid(PeriodicBox(12.0, 12.0), (12, 12))
    fields = seed(grid, np.array([[6.0, 6.0]]), 5.0, 1.0)
    windows = fields.windows()
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
    windows = fields.windows(threshold=THRESHOLD)
    assert areas(fields, windows) == pytest.approx(areas(fields), rel=1e-9)
    assert perimeters(fields, windows) == pytest.approx(perimeters(fields), rel=1e-5)
    # The dense centroid is a circular mean, biased by a few hundredths of a micron
    # for an asymmetric cell; the windowed one is exact. Agree to that, not rounding.
    offset = fields.grid.box.min_image(centres_of_mass(fields, windows) - centres_of_mass(fields))
    assert np.hypot(offset[:, 0], offset[:, 1]).max() < 0.2 * fields.grid.dx


def test_windowed_centroid_is_exact_for_a_symmetric_cell():
    grid = Grid(PeriodicBox(40.0, 40.0), (80, 80))
    centre = np.array([[0.5, 20.0]])          # straddling the x boundary
    fields = seed(grid, centre, 6.0, 1.0)
    windows = fields.windows()
    recovered = centres_of_mass(fields, windows)
    offset = grid.box.min_image(recovered - centre)
    assert np.hypot(*offset.T).max() < 1e-6
    assert centres_of_mass(fields, windows)[0] == pytest.approx(centres_of_mass(fields)[0], abs=1e-6)


def test_degenerate_window_reproduces_dense_exactly():
    """Window == grid: padding wraps, so even the perimeter matches to rounding."""
    grid = Grid(PeriodicBox(12.0, 12.0), (12, 12))
    fields = seed(grid, np.array([[6.0, 6.0]]), 5.0, 1.0)
    windows = fields.windows()
    assert perimeters(fields, windows) == pytest.approx(perimeters(fields), rel=1e-12)
    assert areas(fields, windows) == pytest.approx(areas(fields), rel=1e-12)


# ----------------------------------------------------------------- the option


def test_window_option_changes_no_trajectory(big):
    """Only the diagnostics differ; the dynamics are untouched."""
    dense = simulate(big, duration=30, n_snapshots=4, keep_fields=False)
    windowed = simulate(big.replace(window=True), duration=30, n_snapshots=4, keep_fields=False)
    assert windowed.energies == pytest.approx(dense.energies, rel=1e-12)
    assert windowed.final.areas == pytest.approx(dense.final.areas, rel=1e-9)
    assert windowed.final.perimeters == pytest.approx(dense.final.perimeters, rel=1e-5)
    offset = big.box.min_image(windowed.final.centres - dense.final.centres)
    assert np.hypot(offset[:, 0], offset[:, 1]).max() < 0.2 * big.grid_spacing


def test_window_flag_appears_in_the_name_only_when_set():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from naming import encode

    assert "win" not in encode(Model(), 100.0, 0)
    assert "_win_" in encode(Model(window=True), 100.0, 0)
