import numpy as np
import pytest

from migration_theory import PeriodicBox

BOX = PeriodicBox(7.0, 3.0)


def test_wrap_lands_strictly_inside_the_box():
    rng = np.random.default_rng(0)
    scattered = rng.uniform(-50.0, 50.0, (2000, 2))
    wrapped = BOX.wrap(scattered)
    assert np.all(wrapped >= 0.0)
    assert np.all(wrapped < BOX.lengths)


def test_wrap_of_a_tiny_negative_does_not_land_on_the_upper_edge():
    # np.mod(-1e-18, L) returns exactly L, which is outside [0, L).
    wrapped = BOX.wrap(np.array([[-1e-18, -1e-20]]))
    assert np.all(wrapped < BOX.lengths)


def test_wrapping_is_invisible_to_displacements():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.0, 1.0, (500, 2)) * BOX.lengths
    b = rng.uniform(0.0, 1.0, (500, 2)) * BOX.lengths
    shifted = a + rng.integers(-3, 4, (500, 2)) * BOX.lengths
    assert np.allclose(BOX.displacement(a, b), BOX.displacement(shifted, b))


def test_min_image_never_exceeds_half_the_box():
    rng = np.random.default_rng(2)
    dr = BOX.min_image(rng.uniform(-40.0, 40.0, (2000, 2)))
    assert np.all(np.abs(dr) <= 0.5 * BOX.lengths + 1e-12)


def test_for_cells_gives_the_requested_mean_spacing():
    box = PeriodicBox.for_cells(400, spacing=2.0)
    assert box.area / 400 == pytest.approx(np.sqrt(3) / 2 * 2.0**2)


def test_rejects_non_positive_lengths():
    with pytest.raises(ValueError):
        PeriodicBox(1.0, 0.0)
