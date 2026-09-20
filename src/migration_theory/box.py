"""Periodic simulation box: wrapping and minimum-image displacements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PeriodicBox"]


@dataclass(frozen=True)
class PeriodicBox:
    """A rectangular, doubly-periodic 2D box spanning ``[0, Lx) x [0, Ly)``."""

    Lx: float
    Ly: float

    def __post_init__(self) -> None:
        if not (self.Lx > 0 and self.Ly > 0):
            raise ValueError(f"box lengths must be positive, got ({self.Lx}, {self.Ly})")

    @property
    def lengths(self) -> np.ndarray:
        return np.array([self.Lx, self.Ly], dtype=float)

    @property
    def area(self) -> float:
        return float(self.Lx * self.Ly)

    @property
    def min_length(self) -> float:
        return float(min(self.Lx, self.Ly))

    @classmethod
    def for_cells(cls, n_cells: int, spacing: float, aspect: float = 1.0) -> PeriodicBox:
        """Box holding ``n_cells`` at a mean centre-to-centre distance of ``spacing``.

        Sized from the area per cell of a triangular lattice, ``sqrt(3)/2 * spacing**2``.
        That lattice is the densest packing of discs of diameter ``spacing``, so it is
        the natural reference state for a repulsion whose range is ``spacing``.

        ``aspect`` is ``Lx / Ly``.
        """
        area = n_cells * np.sqrt(3.0) / 2.0 * spacing**2
        Ly = float(np.sqrt(area / aspect))
        return cls(aspect * Ly, Ly)

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
