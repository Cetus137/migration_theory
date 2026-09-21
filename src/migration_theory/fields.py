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

__all__ = ["PhaseFields", "Windows", "seed", "seed_tessellated"]


@dataclass(frozen=True)
class Windows:
    """Where each cell is: one small patch of the grid per cell, all the same shape.

    A cell of radius ``R`` with interface ``w`` is non-zero within about ``R + 3w`` of
    its centre, a few percent of a large box, yet the dense ``(n_cells, ny, nx)``
    layout stores and processes every cell over the whole grid. A window is the
    patch that actually holds the cell: a common ``shape`` -- one size for all cells,
    so a stack of them is a single array -- and per-cell ``origins``, the grid row and
    column where each window starts.

    Periodicity lives in one place, :meth:`indices`: a window that runs off the edge
    of the box has its indices taken modulo the grid size, so a cell straddling the
    boundary reads and writes as an ordinary small array with no wrap inside it.
    """

    grid: Grid
    shape: tuple[int, int]
    """``(h, w)``: rows and columns of every window."""

    origins: np.ndarray
    """``(n_cells, 2)`` integers: the ``(row, col)`` of each window's first point."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "origins", np.asarray(self.origins, dtype=int))
        h, w = self.shape
        if not (1 <= h <= self.grid.ny and 1 <= w <= self.grid.nx):
            raise ValueError(f"window {self.shape} does not fit the grid {self.grid.shape}")
        if self.origins.ndim != 2 or self.origins.shape[1] != 2:
            raise ValueError(f"origins must have shape (n_cells, 2), got {self.origins.shape}")

    @property
    def n_cells(self) -> int:
        return len(self.origins)

    @property
    def spans_rows(self) -> bool:
        """Whether the window covers the whole y axis, so it wraps onto itself."""
        return self.shape[0] == self.grid.ny

    @property
    def spans_cols(self) -> bool:
        return self.shape[1] == self.grid.nx

    def indices(self) -> tuple[np.ndarray, np.ndarray]:
        """``(rows, cols)``, each ``(n_cells, h or w)``, wrapped into the grid."""
        h, w = self.shape
        rows = (self.origins[:, 0, None] + np.arange(h)) % self.grid.ny
        cols = (self.origins[:, 1, None] + np.arange(w)) % self.grid.nx
        return rows, cols

    def extract(self, values: np.ndarray) -> np.ndarray:
        """Every cell's window from a dense ``(n_cells, ny, nx)`` stack: ``(n_cells, h, w)``."""
        rows, cols = self.indices()
        cells = np.arange(self.n_cells)[:, None, None]
        return values[cells, rows[:, :, None], cols[:, None, :]]

    def mask(self) -> np.ndarray:
        """``(n_cells, ny, nx)`` booleans, ``True`` inside each cell's window."""
        rows, cols = self.indices()
        inside = np.zeros((self.n_cells, self.grid.ny, self.grid.nx), dtype=bool)
        cells = np.arange(self.n_cells)[:, None, None]
        inside[cells, rows[:, :, None], cols[:, None, :]] = True
        return inside

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """``(X, Y)``, each ``(n_cells, h, w)``: positions of the window points.

        *Unwrapped*: a window straddling the boundary gets coordinates that run past
        the box edge rather than jumping back to zero, so any average over a window
        is continuous. Wrap the result with the box afterwards.
        """
        h, w = self.shape
        x = (self.origins[:, 1, None] + np.arange(w)) * self.grid.dx
        y = (self.origins[:, 0, None] + np.arange(h)) * self.grid.dy
        target = (self.n_cells, h, w)
        return np.broadcast_to(x[:, None, :], target), np.broadcast_to(y[:, :, None], target)


def _circular_spans(occupied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and extent of the occupied run along a periodic axis, per cell.

    ``occupied`` is ``(n_cells, L)`` booleans. On a periodic axis the occupied indices
    may straddle the edge -- ``[0, 1, L-1]`` is a run of three, not of ``L`` -- so the
    run is taken to begin just after the *largest* empty gap around the circle.
    """
    starts = np.zeros(len(occupied), dtype=int)
    extents = np.zeros(len(occupied), dtype=int)
    length = occupied.shape[1]
    for k, row in enumerate(occupied):
        where = np.flatnonzero(row)
        if len(where) == 0:
            continue
        if len(where) == length:
            extents[k] = length
            continue
        gaps = np.diff(np.concatenate([where, [where[0] + length]]))
        largest = int(np.argmax(gaps))
        starts[k] = where[(largest + 1) % len(where)]
        extents[k] = length - gaps[largest] + 1
    return starts, extents


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

    def windows(self, margin: int = 3, threshold: float = 1e-6) -> Windows:
        """The patch of the grid each cell occupies, with ``margin`` points to spare.

        Found from the fields themselves: the rows and columns where a cell exceeds
        ``threshold``, allowing for a cell that straddles the periodic boundary. All
        windows share one shape, the largest extent over cells plus the margin on
        each side, capped at the grid -- so on a box barely bigger than a cell the
        window *is* the grid and everything degenerates to the dense computation.
        Each cell's occupied run is centred in its window.

        The margin is what lets a stencil on a window treat the outside as zero: the
        field there is below ``threshold`` by construction. Cells move, so the windows
        go stale; recompute them every so often, which costs one pass over the stack.

        **On the threshold.** The interface is a ``tanh``, whose tail falls by a factor
        ``exp(-sqrt(2) dx / w)`` per grid point -- a half per point at ``w = 2 dx`` --
        so where the window ends is a choice, not a fact: at ``1e-6`` it reaches about
        ``10 w`` beyond the cell radius, at ``1e-8`` about ``13 w``. The default keeps
        the window near a tenth of a 75-cell box while what it drops is below ``1e-6``
        in the field and ``1e-5`` relative in any perimeter.
        """
        if margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        present = self.values > threshold
        row_starts, row_extents = _circular_spans(present.any(axis=2))
        col_starts, col_extents = _circular_spans(present.any(axis=1))
        h = min(self.grid.ny, int(row_extents.max()) + 2 * margin)
        w = min(self.grid.nx, int(col_extents.max()) + 2 * margin)
        origins = np.column_stack([
            (row_starts - (h - row_extents) // 2) % self.grid.ny,
            (col_starts - (w - col_extents) // 2) % self.grid.nx,
        ])
        return Windows(self.grid, (h, w), origins)

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
