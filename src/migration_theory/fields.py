"""Storage for the per-cell phase fields.

One field ``phi_i`` per cell, each defined over the whole grid. ``phi_i`` is near 1
inside cell ``i``, near 0 outside, and passes through a smooth interface of width
``interface_width`` in between; cell shape is whatever the dynamics make it, never
imposed.

Storage is a dense ``(n_cells, ny, nx)`` array. That is the simple, obviously correct
choice and it vectorises perfectly, but it costs ``n_cells * ny * nx`` floats even
though each field is zero almost everywhere -- roughly 34 MB for 64 cells on a 256x256
grid, and it grows linearly in cell count. The eventual answer for large N is a moving
window per cell. Everything outside this module goes through :class:`PhaseFields`
rather than touching ``.values`` directly, so that change stays local when you want it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid

__all__ = ["PhaseFields", "seed", "seed_tessellated"]


@dataclass
class PhaseFields:
    """A stack of per-cell phase fields on a shared grid."""

    grid: Grid
    values: np.ndarray
    """``(n_cells, ny, nx)``. Prefer the methods below; this is the dense backing store."""

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 3 or self.values.shape[1:] != self.grid.shape:
            raise ValueError(
                f"fields must have shape (n_cells, {self.grid.ny}, {self.grid.nx}), "
                f"got {self.values.shape}"
            )

    @classmethod
    def empty(cls, grid: Grid, n_cells: int) -> PhaseFields:
        return cls(grid, grid.zeros(n_cells))

    @property
    def n_cells(self) -> int:
        return len(self.values)

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, index: int) -> np.ndarray:
        """The field of a single cell, ``(ny, nx)``."""
        return self.values[index]

    def __iter__(self):
        return iter(self.values)

    @property
    def occupancy(self) -> np.ndarray:
        """``sum_i phi_i``, ``(ny, nx)``. Close to 1 everywhere in a confluent tissue."""
        return self.values.sum(axis=0)

    def gradient(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell gradients, each ``(n_cells, ny, nx)``."""
        return self.grid.gradient(self.values)

    def laplacian(self) -> np.ndarray:
        return self.grid.laplacian(self.values)

    def copy(self) -> PhaseFields:
        return PhaseFields(self.grid, self.values.copy())


def seed(
    grid: Grid,
    centres: np.ndarray,
    radius: float,
    interface_width: float,
) -> PhaseFields:
    r"""Circular cells of the given ``radius``, one per row of ``centres``.

    Each field is laid down as the equilibrium interface profile of a symmetric double
    well,

    .. math:: \phi(r) = \tfrac{1}{2}\left[1 - \tanh\frac{r - R}{\sqrt{2}\,\lambda}\right]

    so the fields start close to a stationary state of the free energy and the first
    steps relax cell *arrangement* rather than burning time sharpening interfaces that
    were seeded with the wrong profile.

    ``interface_width`` needs several grid points across it -- roughly
    ``interface_width >= 3 * dx`` -- or the interface will be under-resolved and pinned
    to the grid. A warning-free check of that is left to you once the free energy fixes
    what the width should be.
    """
    centres = np.atleast_2d(np.asarray(centres, dtype=float))
    if centres.ndim != 2 or centres.shape[1] != 2:
        raise ValueError(f"centres must have shape (n_cells, 2), got {centres.shape}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if interface_width <= 0:
        raise ValueError(f"interface_width must be positive, got {interface_width}")

    fields = grid.zeros(len(centres))
    for i, centre in enumerate(centres):
        distance = grid.distance_to(centre)
        fields[i] = 0.5 * (1.0 - np.tanh((distance - radius) / (np.sqrt(2.0) * interface_width)))
    return PhaseFields(grid, fields)


def seed_tessellated(
    grid: Grid,
    centres: np.ndarray,
    radius: float,
    interface_width: float,
) -> PhaseFields:
    r"""Cells shaped like their Voronoi regions, clipped to ``radius``.

    Circles cannot tile the plane, so seeding a confluent tissue with them guarantees
    heavy overlap -- and the mechanical forces that produces are far larger than
    anything the relaxed tissue ever sees. Under force balance those forces set the
    velocity, so the run opens with a spike that is pure seeding artefact. This starts
    the tissue where it was going to end up instead.

    For each grid point, let :math:`d_i` be the distance to cell ``i`` and :math:`d_j`
    the distance to the nearest *other* cell. The perpendicular bisector between them
    lies where the two are equal, so

    .. math:: s_i = \min\!\left( \frac{d_j - d_i}{2},\; R - d_i \right)

    is the signed distance into cell ``i`` -- from its Voronoi boundary, or from a
    circle of radius ``R``, whichever is nearer. The profile is then the same ``tanh``
    of that distance as :func:`seed` uses.

    Taking the smaller of the two keeps this right at any density: near confluence the
    Voronoi term binds and cells tile, while below it the circle binds and cells are
    round and separate, as they should be.
    """
    centres = np.atleast_2d(np.asarray(centres, dtype=float))
    if centres.ndim != 2 or centres.shape[1] != 2:
        raise ValueError(f"centres must have shape (n_cells, 2), got {centres.shape}")
    if len(centres) < 2:
        return seed(grid, centres, radius, interface_width)
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if interface_width <= 0:
        raise ValueError(f"interface_width must be positive, got {interface_width}")

    distances = np.stack([grid.distance_to(centre) for centre in centres])
    closest, runner_up = np.partition(distances, 1, axis=0)[:2]

    # For whichever cell is nearest at a point, the competitor is the runner-up; for
    # every other cell the competitor is the nearest one.
    competitor = np.where(distances <= closest + 1e-12, runner_up, closest)
    signed = np.minimum(0.5 * (competitor - distances), radius - distances)
    return PhaseFields(grid, 0.5 * (1.0 + np.tanh(signed / (np.sqrt(2.0) * interface_width))))
