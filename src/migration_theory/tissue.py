"""The evolving state: the fields, the polarities, and how far time has run."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .activity import Polarity
from .box import PeriodicBox
from .diagnostics import areas, centres_of_mass, confluence_error
from .fields import PhaseFields, seed
from .grid import Grid

__all__ = ["Tissue"]


@dataclass
class Tissue:
    """Everything that changes as a simulation runs.

    Deliberately thin. Geometry is derived on demand from the fields rather than
    stored, so it can never fall out of step with them, and the free energy is *not*
    held here -- it is a fixed description of the physics, passed to the stepper, not
    part of the state.
    """

    fields: PhaseFields
    polarity: Polarity = None  # type: ignore[assignment]
    time: float = 0.0

    def __post_init__(self) -> None:
        if self.polarity is None:
            self.polarity = Polarity.still(self.fields.n_cells, self.fields.grid.ndim)
        if self.polarity.n_cells != self.fields.n_cells:
            raise ValueError(
                f"{self.polarity.n_cells} polarities for {self.fields.n_cells} cells"
            )
        if self.polarity.dimension != self.fields.grid.ndim:
            raise ValueError(
                f"{self.polarity.dimension}D polarity for a {self.fields.grid.ndim}D tissue"
            )

    @classmethod
    def seeded(
        cls,
        grid: Grid,
        centres: np.ndarray,
        radius: float,
        interface_width: float,
        speed: float = 0.0,
        rotational_diffusion: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> Tissue:
        """Round cells at ``centres``, with random initial polarities."""
        fields = seed(grid, centres, radius, interface_width)
        polarity = Polarity.random(fields.n_cells, speed, rotational_diffusion, rng, grid.ndim)
        return cls(fields, polarity)

    @property
    def n_cells(self) -> int:
        return self.fields.n_cells

    @property
    def grid(self) -> Grid:
        return self.fields.grid

    @property
    def box(self) -> PeriodicBox:
        return self.fields.grid.box

    def areas(self) -> np.ndarray:
        return areas(self.fields)

    def centres_of_mass(self) -> np.ndarray:
        return centres_of_mass(self.fields)

    def confluence_error(self) -> float:
        return confluence_error(self.fields)

    def copy(self) -> Tissue:
        """An independent snapshot, for checkpointing or for comparing two runs."""
        return Tissue(self.fields.copy(), self.polarity.copy(), self.time)
