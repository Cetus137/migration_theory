r"""Measurements taken from the phase fields.

The two quantities the particle picture needed a Delaunay tessellation for come
directly out of the fields here, as integrals rather than as polygon geometry:

.. math::

    A_i = \int \phi_i^2 \,\mathrm{d}x, \qquad
    B_{ij} = \int \phi_i^2 \phi_j^2 \,\mathrm{d}x

``A_i`` is cell ``i``'s area and ``B_{ij}`` its contact with cell ``j`` -- the overlap
region *is* the interface between them, so contact is graded rather than binary and no
topology has to be inferred. The ``phi**2`` weighting is the usual choice: it suppresses
the interface tails relative to a plain ``phi``, so the measured area tracks the sharp
interface limit closely.
"""

from __future__ import annotations

import numpy as np

from .fields import PhaseFields, Windows

__all__ = [
    "areas",
    "perimeters",
    "shape_indices",
    "overlap_matrix",
    "contact_lengths",
    "centres_of_mass",
    "confluence_error",
]

# Each per-cell measurement takes an optional ``windows``. Without it the measurement
# runs over the whole grid for every cell, the dense reference. With it, only each
# cell's own patch is touched -- the same number to rounding, at a cost that scales
# with the cell rather than the box. The windows are computed by
# ``PhaseFields.windows`` and are the first stage of storing cells that way outright.


def areas(
    fields: PhaseFields, windows: Windows | None = None, patches: np.ndarray | None = None
) -> np.ndarray:
    """``(n_cells,)`` area of each cell, ``\\int phi_i^2``.

    ``patches`` is the windows' contents if the caller has gathered them already.
    """
    if windows is None:
        return fields.grid.integrate(fields.values**2)
    if patches is None:
        patches = windows.extract(fields.values)
    return (patches**2).sum(axis=(-2, -1)) * fields.grid.cell_area


def perimeters(fields: PhaseFields, windows: Windows | None = None) -> np.ndarray:
    r"""``(n_cells,)`` perimeter of each cell, ``\int |grad phi_i| d^2r``.

    Exact in the sharp-interface limit for any profile that runs monotonically from 1
    to 0: integrating ``|grad phi|`` along the normal gives 1 regardless of how the
    transition is shaped, so what survives is the length of the level set. No contour
    has to be extracted and no shape assumed.
    """
    grid = fields.grid
    if windows is None:
        d_dx, d_dy = grid.gradient(fields.values)
        return grid.integrate(np.hypot(d_dx, d_dy))
    d_dx, d_dy = windows.gradient(windows.extract(fields.values))
    return np.hypot(d_dx, d_dy).sum(axis=(-2, -1)) * grid.cell_area


def shape_indices(fields: PhaseFields) -> np.ndarray:
    r"""``(n_cells,)`` dimensionless shape index ``P / sqrt(A)``.

    The standard order parameter of tissue mechanics: 3.545 for a circle, 3.72 for a
    regular hexagon, and rising as cells elongate, with ~3.81 the usual rigidity
    transition threshold.

    Read these with care at finite interface width. ``A`` here is ``\int phi^2``, which
    undershoots the sharp-interface area by a factor depending on the width-to-radius
    ratio, so the index comes out systematically high -- about 3.9 for a circle at
    ``w/R = 0.25``. Compare values against each other, not against the literature
    thresholds, unless the interface is sharp.
    """
    return perimeters(fields) / np.sqrt(areas(fields))


def overlap_matrix(fields: PhaseFields) -> np.ndarray:
    """``(n_cells, n_cells)`` symmetric matrix of ``\\int phi_i^2 phi_j^2``.

    The diagonal is ``\\int phi_i^4``, which is *not* the area -- it is the cell's
    self-overlap -- so read contacts off the off-diagonal entries.

    Formed as a single matrix product over flattened fields rather than a loop over
    pairs, which hands the O(N^2 * grid) work to BLAS and keeps this usable every step.
    """
    squared = (fields.values**2).reshape(fields.n_cells, -1)
    return (squared @ squared.T) * fields.grid.cell_area


def contact_lengths(fields: PhaseFields, interface_width: float) -> np.ndarray:
    """``(n_cells, n_cells)`` overlap areas rescaled to lengths, with a zero diagonal.

    The overlap of two cells is a strip along their shared boundary, of area roughly
    ``length * interface_width``, so dividing by the width recovers something directly
    comparable to the shared Voronoi edge length of the particle model. The prefactor
    depends on the free energy you settle on, so treat this as proportional to contact
    length rather than calibrated to it.
    """
    if interface_width <= 0:
        raise ValueError(f"interface_width must be positive, got {interface_width}")
    lengths = overlap_matrix(fields) / interface_width
    np.fill_diagonal(lengths, 0.0)
    return lengths


def centres_of_mass(fields: PhaseFields, windows: Windows | None = None) -> np.ndarray:
    """``(n_cells, 2)`` centroid of each cell, exact across the periodic boundary.

    Dense: :meth:`Grid.centre_of_mass`, a circular-mean guess refined to the exact
    centroid with minimum-image displacements. Windowed: a window never wraps
    internally, so it is the plain centroid over unwrapped coordinates, wrapped once
    at the end. The two agree to the tail the window drops.
    """
    if windows is None:
        return fields.grid.centre_of_mass(fields.values**2)
    weights = windows.extract(fields.values) ** 2
    X, Y = windows.coordinates()
    total = weights.sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = np.column_stack([
            (weights * X).sum(axis=(-2, -1)) / total,
            (weights * Y).sum(axis=(-2, -1)) / total,
        ])
    return np.where((total == 0)[:, None], np.nan, fields.grid.box.wrap(centre))


def confluence_error(fields: PhaseFields) -> float:
    """RMS deviation of ``sum_i phi_i`` from 1.

    Zero in a perfectly confluent tissue with no gaps and no piled-up cells. Useful as a
    single number to watch while a configuration relaxes.
    """
    return float(np.sqrt(np.mean((fields.occupancy - 1.0) ** 2)))
