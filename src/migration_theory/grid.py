"""The discretised, periodic domain the phase fields live on.

Every field is an array whose last two axes are ``(y, x)``, so a leading axis can carry
the cell index and every operator below vectorises over all cells at once. Derivatives
use ``np.roll``, which makes periodicity automatic rather than something to remember.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .box import PeriodicBox

__all__ = ["Grid"]


@dataclass(frozen=True)
class Grid:
    """A uniform periodic grid over a :class:`PeriodicBox`.

    Node-centred: sample points sit at ``i * dx`` for ``i`` in ``[0, nx)``. On a periodic
    domain that makes a plain sum times the cell area an exact trapezoidal integral.
    """

    box: PeriodicBox
    shape: tuple[int, int]
    """``(ny, nx)`` -- number of grid points along y and x, matching the array axes."""

    def __post_init__(self) -> None:
        ny, nx = self.shape
        if ny < 4 or nx < 4:
            raise ValueError(f"grid must be at least 4x4, got {self.shape}")

    @classmethod
    def from_spacing(cls, box: PeriodicBox, spacing: float) -> Grid:
        """Grid with points at most ``spacing`` apart, rounded up to fit the box exactly.

        The interface width has to be resolved by several points, so in practice
        ``spacing`` should be a fraction of it -- see :func:`~migration_theory.fields.seed`.
        """
        if spacing <= 0:
            raise ValueError(f"spacing must be positive, got {spacing}")
        nx = max(4, int(np.ceil(box.Lx / spacing)))
        ny = max(4, int(np.ceil(box.Ly / spacing)))
        return cls(box, (ny, nx))

    @property
    def ny(self) -> int:
        return self.shape[0]

    @property
    def nx(self) -> int:
        return self.shape[1]

    @property
    def dx(self) -> float:
        return self.box.Lx / self.nx

    @property
    def dy(self) -> float:
        return self.box.Ly / self.ny

    @property
    def cell_area(self) -> float:
        """Area each grid point represents; the weight in every integral."""
        return self.dx * self.dy

    @property
    def n_points(self) -> int:
        return self.ny * self.nx

    @property
    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """``(X, Y)``, each ``(ny, nx)``, giving the position of every grid point."""
        x = np.arange(self.nx) * self.dx
        y = np.arange(self.ny) * self.dy
        return np.meshgrid(x, y, indexing="xy")

    def zeros(self, n_fields: int | None = None) -> np.ndarray:
        """An empty field, or a stack of ``n_fields`` of them."""
        return np.zeros(self.shape if n_fields is None else (n_fields, *self.shape))

    def distance_to(self, centre: np.ndarray) -> np.ndarray:
        """``(ny, nx)`` distance from every grid point to ``centre``, across the boundary."""
        X, Y = self.coordinates
        offset = self.box.min_image(
            np.stack([X - centre[0], Y - centre[1]], axis=-1)
        )
        return np.hypot(offset[..., 0], offset[..., 1])

    def integrate(self, field: np.ndarray) -> np.ndarray:
        """Integrate over the domain, reducing the two trailing axes."""
        return np.sum(field, axis=(-2, -1)) * self.cell_area

    def gradient(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(d/dx, d/dy)`` by second-order central differences."""
        d_dx = (np.roll(field, -1, axis=-1) - np.roll(field, 1, axis=-1)) / (2.0 * self.dx)
        d_dy = (np.roll(field, -1, axis=-2) - np.roll(field, 1, axis=-2)) / (2.0 * self.dy)
        return d_dx, d_dy

    def forward_gradient(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(d/dx, d/dy)`` by first differences to the next point.

        First order, so worse than :meth:`gradient` for measuring a slope -- but this is
        the pair to use inside a gradient *energy*. The discrete operator that is the
        exact adjoint of forward differencing is the standard five-point
        :meth:`laplacian`, so an energy built from this and a functional derivative
        built from that agree to machine precision rather than merely to discretisation
        error.

        Central differences would instead pair with a wide ``2dx`` stencil that
        decouples even and odd grid points, leaving checkerboard modes free of any
        energy cost -- which is how a phase field grows grid-scale noise.
        """
        d_dx = (np.roll(field, -1, axis=-1) - field) / self.dx
        d_dy = (np.roll(field, -1, axis=-2) - field) / self.dy
        return d_dx, d_dy

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """Second-order five-point Laplacian."""
        d2_dx2 = (
            np.roll(field, -1, axis=-1) + np.roll(field, 1, axis=-1) - 2.0 * field
        ) / self.dx**2
        d2_dy2 = (
            np.roll(field, -1, axis=-2) + np.roll(field, 1, axis=-2) - 2.0 * field
        ) / self.dy**2
        return d2_dx2 + d2_dy2

    def divergence(self, field_x: np.ndarray, field_y: np.ndarray) -> np.ndarray:
        """Divergence of a vector field given by its two components."""
        d_dx = (np.roll(field_x, -1, axis=-1) - np.roll(field_x, 1, axis=-1)) / (2.0 * self.dx)
        d_dy = (np.roll(field_y, -1, axis=-2) - np.roll(field_y, 1, axis=-2)) / (2.0 * self.dy)
        return d_dx + d_dy

    def centre_of_mass(self, field: np.ndarray) -> np.ndarray:
        """Weighted centroid of each field, exact across the periodic boundary.

        A plain weighted mean is wrong for a blob straddling an edge -- it would place
        the centre in the middle of the box. So: first a rough position from the
        circular mean, averaging the angle each coordinate maps to around its periodic
        circle; then the exact centroid as that position plus the weighted mean of
        every point's *minimum-image* displacement from it. For a blob smaller than
        half the box those displacements contain no wrap, so the second step is an
        ordinary centroid and the result is exact.

        The circular mean alone is not the centroid: its error is set by the blob's
        third moment and, measured, reached 0.36 um for elongated cells in a fluid
        tissue. The refinement removes it.
        """
        weights = np.asarray(field, dtype=float)
        total = np.sum(weights, axis=(-2, -1))
        X, Y = self.coordinates

        guess = []
        for coordinate, length in ((X, self.box.Lx), (Y, self.box.Ly)):
            angle = 2.0 * np.pi * coordinate / length
            mean_cos = np.sum(weights * np.cos(angle), axis=(-2, -1))
            mean_sin = np.sum(weights * np.sin(angle), axis=(-2, -1))
            guess.append(length * np.arctan2(mean_sin, mean_cos) / (2.0 * np.pi))

        with np.errstate(invalid="ignore", divide="ignore"):
            centre = []
            for coordinate, length, near in ((X, self.box.Lx, guess[0]), (Y, self.box.Ly, guess[1])):
                offset = coordinate - near[..., None, None]
                offset -= length * np.round(offset / length)          # minimum image
                centre.append(near + np.sum(weights * offset, axis=(-2, -1)) / total)
        centre = self.box.wrap(np.stack(centre, axis=-1))
        # A field that is identically zero has no centroid; report NaN rather than the
        # arbitrary answer arctan2(0, 0) would give.
        return np.where((total == 0)[..., None], np.nan, centre)
