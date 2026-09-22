"""The discretised, periodic domain the phase fields live on, in any dimension.

Every field is an array whose trailing ``ndim`` axes are the grid, in ``(z, y, x)``
order -- so a 2D field is ``(ny, nx)`` and a 3D one ``(nz, ny, nx)`` -- and a leading
axis can carry the cell index so every operator below vectorises over all cells at
once. Vector quantities (gradients, positions, velocities) are given by components in
``(x, y, z)`` order, the reverse of the axes. Derivatives use ``np.roll``, which makes
periodicity automatic rather than something to remember.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .box import PeriodicBox

__all__ = ["Grid", "sum_of_squares", "magnitude"]


def sum_of_squares(components) -> np.ndarray:
    """``sum_k c_k**2`` over the components of a vector field.

    Accumulated left to right, so for two components it is exactly the ``a**2 + b**2``
    the 2D code always computed, to the last bit.
    """
    components = list(components)
    total = components[0] ** 2
    for component in components[1:]:
        total = total + component**2
    return total


def magnitude(components) -> np.ndarray:
    """``|c|`` over the components of a vector field.

    ``np.hypot`` for two components -- what the 2D code always used, and rounding
    differs between it and a square root of a sum -- and the square root of
    :func:`sum_of_squares` otherwise.
    """
    components = list(components)
    if len(components) == 2:
        return np.hypot(components[0], components[1])
    return np.sqrt(sum_of_squares(components))


@dataclass(frozen=True)
class Grid:
    """A uniform periodic grid over a :class:`PeriodicBox`.

    Node-centred: sample points sit at ``i * dx`` for ``i`` in ``[0, nx)``. On a periodic
    domain that makes a plain sum times the cell volume an exact trapezoidal integral.
    """

    box: PeriodicBox
    shape: tuple[int, ...]
    """Number of grid points along each array axis, ``(ny, nx)`` or ``(nz, ny, nx)``."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))
        if len(self.shape) != self.box.ndim:
            raise ValueError(
                f"grid shape {self.shape} has {len(self.shape)} axes for a "
                f"{self.box.ndim}D box"
            )
        if any(n < 4 for n in self.shape):
            raise ValueError(f"grid must be at least 4 points along every axis, got {self.shape}")

    @classmethod
    def from_spacing(cls, box: PeriodicBox, spacing: float) -> Grid:
        """Grid with points at most ``spacing`` apart, rounded up to fit the box exactly.

        The interface width has to be resolved by several points, so in practice
        ``spacing`` should be a fraction of it -- see :func:`~migration_theory.fields.seed`.
        """
        if spacing <= 0:
            raise ValueError(f"spacing must be positive, got {spacing}")
        # Array axes run (z, y, x): the reverse of the box's (x, y, z) lengths.
        shape = tuple(max(4, int(np.ceil(length / spacing))) for length in box.lengths[::-1])
        return cls(box, shape)

    # ------------------------------------------------------------------ geometry

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def nx(self) -> int:
        return self.shape[-1]

    @property
    def ny(self) -> int:
        return self.shape[-2]

    @property
    def nz(self) -> int:
        if self.ndim < 3:
            raise AttributeError(f"a {self.ndim}D grid has no nz")
        return self.shape[-3]

    @property
    def spacings(self) -> tuple[float, ...]:
        """Grid spacing along each *array* axis, ``(dy, dx)`` or ``(dz, dy, dx)``."""
        return tuple(length / n for length, n in zip(self.box.lengths[::-1], self.shape))

    def spacing(self, component: int) -> float:
        """Grid spacing along spatial component ``k``: 0 for x, 1 for y, 2 for z."""
        return self.box.lengths[component] / self.shape[-1 - component]

    @property
    def dx(self) -> float:
        return self.spacing(0)

    @property
    def dy(self) -> float:
        return self.spacing(1)

    @property
    def dz(self) -> float:
        if self.ndim < 3:
            raise AttributeError(f"a {self.ndim}D grid has no dz")
        return self.spacing(2)

    @property
    def cell_volume(self) -> float:
        """Measure each grid point represents; the weight in every integral."""
        return float(np.prod(self.spacings))

    @property
    def cell_area(self) -> float:
        """:attr:`cell_volume` under its 2D name."""
        return self.cell_volume

    @property
    def n_points(self) -> int:
        return int(np.prod(self.shape))

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        """The trailing axes of a field array that are the grid: ``(-2, -1)`` in 2D."""
        return tuple(range(-self.ndim, 0))

    @property
    def coordinates(self) -> tuple[np.ndarray, ...]:
        """``(X, Y[, Z])``, each of the grid's shape, giving the position of every point."""
        axes = [np.arange(n) * h for n, h in zip(self.shape, self.spacings)]
        meshes = np.meshgrid(*axes, indexing="ij")           # in array (z, y, x) order
        return tuple(meshes[::-1])                            # returned in (x, y, z) order

    def zeros(self, n_fields: int | None = None) -> np.ndarray:
        """An empty field, or a stack of ``n_fields`` of them."""
        return np.zeros(self.shape if n_fields is None else (n_fields, *self.shape))

    def distance_to(self, centre: np.ndarray) -> np.ndarray:
        """Distance from every grid point to ``centre`` (in ``(x, y, z)``), across the boundary."""
        centre = np.asarray(centre, dtype=float)
        offset = self.box.min_image(
            np.stack([X - c for X, c in zip(self.coordinates, centre)], axis=-1)
        )
        return np.sqrt((offset**2).sum(axis=-1))

    # ------------------------------------------------------------------ calculus

    def integrate(self, field: np.ndarray) -> np.ndarray:
        """Integrate over the domain, reducing the grid axes."""
        return np.sum(field, axis=self.spatial_axes) * self.cell_volume

    def _shift(self, field: np.ndarray, component: int, steps: int) -> np.ndarray:
        """``field`` shifted ``steps`` points along spatial component ``k``, periodically."""
        return np.roll(field, -steps, axis=-1 - component)

    def gradient(self, field: np.ndarray) -> tuple[np.ndarray, ...]:
        """``(d/dx, d/dy[, d/dz])`` by second-order central differences."""
        return tuple(
            (self._shift(field, k, 1) - self._shift(field, k, -1)) / (2.0 * self.spacing(k))
            for k in range(self.ndim)
        )

    def forward_gradient(self, field: np.ndarray) -> tuple[np.ndarray, ...]:
        """``(d/dx, d/dy[, d/dz])`` by first differences to the next point.

        First order, so worse than :meth:`gradient` for measuring a slope -- but this is
        the pair to use inside a gradient *energy*. The discrete operator that is the
        exact adjoint of forward differencing is the standard ``2 ndim + 1``-point
        :meth:`laplacian`, so an energy built from this and a functional derivative
        built from that agree to machine precision rather than merely to discretisation
        error.

        Central differences would instead pair with a wide ``2dx`` stencil that
        decouples even and odd grid points, leaving checkerboard modes free of any
        energy cost -- which is how a phase field grows grid-scale noise.
        """
        return tuple(
            (self._shift(field, k, 1) - field) / self.spacing(k) for k in range(self.ndim)
        )

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """Second-order ``2 ndim + 1``-point Laplacian: five points in 2D, seven in 3D."""
        total = np.zeros_like(field, dtype=float)
        for k in range(self.ndim):
            total += (
                self._shift(field, k, 1) + self._shift(field, k, -1) - 2.0 * field
            ) / self.spacing(k) ** 2
        return total

    def divergence(self, *components: np.ndarray) -> np.ndarray:
        """Divergence of a vector field given by its ``(x, y[, z])`` components."""
        if len(components) != self.ndim:
            raise ValueError(f"expected {self.ndim} components, got {len(components)}")
        total = np.zeros_like(components[0], dtype=float)
        for k, component in enumerate(components):
            total += (
                self._shift(component, k, 1) - self._shift(component, k, -1)
            ) / (2.0 * self.spacing(k))
        return total

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
        axes = self.spatial_axes
        total = np.sum(weights, axis=axes)

        guess = []
        for coordinate, length in zip(self.coordinates, self.box.lengths):
            angle = 2.0 * np.pi * coordinate / length
            mean_cos = np.sum(weights * np.cos(angle), axis=axes)
            mean_sin = np.sum(weights * np.sin(angle), axis=axes)
            guess.append(length * np.arctan2(mean_sin, mean_cos) / (2.0 * np.pi))

        expand = (...,) + (None,) * self.ndim
        with np.errstate(invalid="ignore", divide="ignore"):
            centre = []
            for coordinate, length, near in zip(self.coordinates, self.box.lengths, guess):
                offset = coordinate - near[expand]
                offset -= length * np.round(offset / length)          # minimum image
                centre.append(near + np.sum(weights * offset, axis=axes) / total)
        centre = self.box.wrap(np.stack(centre, axis=-1))
        # A field that is identically zero has no centroid; report NaN rather than the
        # arbitrary answer arctan2(0, 0) would give.
        return np.where((total == 0)[..., None], np.nan, centre)
