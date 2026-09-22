"""Periodic simulation box: wrapping and minimum-image displacements, in any dimension."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PeriodicBox"]

#: Volume per cell of the densest sphere packing, per unit spacing^d: the triangular
#: lattice in 2D, face-centred cubic in 3D. The natural reference density for a
#: repulsion whose range is the spacing.
_DENSEST_PACKING = {2: np.sqrt(3.0) / 2.0, 3: 1.0 / np.sqrt(2.0)}


@dataclass(frozen=True, init=False)
class PeriodicBox:
    """A periodic box spanning ``[0, L_k)`` along each axis.

    ``PeriodicBox(Lx, Ly)`` is a 2D box, ``PeriodicBox(Lx, Ly, Lz)`` a 3D one, and the
    lengths are always given in ``(x, y, z)`` order -- the reverse of the array axes a
    field on the box uses, which run ``(z, y, x)``. Positions and displacements are
    arrays whose last axis is that same ``(x, y, z)`` order.
    """

    _lengths: tuple[float, ...]

    def __init__(self, *lengths: float) -> None:
        if len(lengths) == 1 and np.ndim(lengths[0]) == 1:
            lengths = tuple(lengths[0])
        values = tuple(float(length) for length in lengths)
        if not values or not all(length > 0 for length in values):
            raise ValueError(f"box lengths must be positive, got {values}")
        object.__setattr__(self, "_lengths", values)

    def __repr__(self) -> str:
        return f"PeriodicBox({', '.join(f'{length:g}' for length in self._lengths)})"

    @property
    def ndim(self) -> int:
        return len(self._lengths)

    @property
    def lengths(self) -> np.ndarray:
        """``(ndim,)`` box lengths in ``(x, y, z)`` order."""
        return np.array(self._lengths, dtype=float)

    @property
    def Lx(self) -> float:
        return self._lengths[0]

    @property
    def Ly(self) -> float:
        if self.ndim < 2:
            raise AttributeError("a 1D box has no Ly")
        return self._lengths[1]

    @property
    def Lz(self) -> float:
        if self.ndim < 3:
            raise AttributeError(f"a {self.ndim}D box has no Lz")
        return self._lengths[2]

    @property
    def volume(self) -> float:
        """The box's measure: area in 2D, volume in 3D."""
        return float(np.prod(self._lengths))

    @property
    def area(self) -> float:
        """:attr:`volume` under its 2D name, for the code that grew up in 2D."""
        return self.volume

    @property
    def min_length(self) -> float:
        return float(min(self._lengths))

    @classmethod
    def for_cells(
        cls, n_cells: int, spacing: float, aspect: float = 1.0, dimension: int = 2
    ) -> PeriodicBox:
        """Box holding ``n_cells`` at a mean centre-to-centre distance of ``spacing``.

        Sized from the volume per cell of the densest packing of spheres of diameter
        ``spacing``: the triangular lattice in 2D, face-centred cubic in 3D. That
        lattice is the natural reference state for a repulsion whose range is
        ``spacing``. ``aspect`` is ``Lx / Ly`` and applies in 2D; a 3D box is a cube.
        """
        if dimension not in _DENSEST_PACKING:
            raise ValueError(f"dimension must be 2 or 3, got {dimension}")
        volume = n_cells * _DENSEST_PACKING[dimension] * spacing**dimension
        if dimension == 2:
            Ly = float(np.sqrt(volume / aspect))
            return cls(aspect * Ly, Ly)
        side = float(volume ** (1.0 / 3.0))
        return cls(side, side, side)

    def wrap(self, positions: np.ndarray) -> np.ndarray:
        """Fold positions back into the box."""
        lengths = self.lengths
        wrapped = np.mod(np.asarray(positions, dtype=float), lengths)
        # np.mod returns exactly L for tiny negative inputs, which is outside [0, L).
        return np.where(wrapped >= lengths, 0.0, wrapped)

    def min_image(self, dr: np.ndarray) -> np.ndarray:
        """Reduce displacement vectors to the nearest periodic image."""
        lengths = self.lengths
        dr = np.asarray(dr, dtype=float)
        return dr - lengths * np.round(dr / lengths)

    def displacement(self, r_i: np.ndarray, r_j: np.ndarray) -> np.ndarray:
        """Minimum-image vector pointing from ``r_j`` to ``r_i``."""
        return self.min_image(np.asarray(r_i, dtype=float) - np.asarray(r_j, dtype=float))
