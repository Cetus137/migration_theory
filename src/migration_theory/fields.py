"""Storage for the per-cell phase fields.

One field ``phi_i`` per cell, each defined over the whole grid. ``phi_i`` is near 1
inside cell ``i``, near 0 outside, and passes through a smooth interface of width
``interface_width`` in between; cell shape is whatever the dynamics make it, never
imposed.

Storage is a dense ``(n_cells, *grid.shape)`` array -- ``(n_cells, ny, nx)`` in 2D,
``(n_cells, nz, ny, nx)`` in 3D. That is the simple, obviously correct choice and it
vectorises perfectly, but it costs ``n_cells * n_points`` floats even though each field
is zero almost everywhere. :class:`Windows` is the answer to the *time* that costs: each
cell is computed on its own small patch of the grid, and the dense array is only the
backing store. Everything outside this module goes through :class:`PhaseFields` rather
than touching ``.values`` directly, so a change of backing store stays local.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .grid import Grid

__all__ = ["PhaseFields", "Windows", "seed", "seed_tessellated"]


@dataclass(frozen=True)
class Windows:
    """Where each cell is: one small patch of the grid per cell, all the same shape.

    A cell of radius ``R`` with interface ``w`` is non-zero within about ``R + 3w`` of
    its centre, a few percent of a large box, yet the dense layout stores and
    processes every cell over the whole grid. A window is the patch that actually
    holds the cell: a common ``shape`` -- one size for all cells, so a stack of them is
    a single array -- and per-cell ``origins``, the grid index along each axis where
    each window starts. Both are in *array* order, ``(y, x)`` or ``(z, y, x)``, like
    the grid's shape.

    Periodicity lives in one place, :meth:`indices`: a window that runs off the edge
    of the box has its indices taken modulo the grid size, so a cell straddling the
    boundary reads and writes as an ordinary small array with no wrap inside it.
    """

    grid: Grid
    shape: tuple[int, ...]
    """Points along each array axis of every window."""

    origins: np.ndarray
    """``(n_cells, ndim)`` integers: each window's first grid index along each axis."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))
        object.__setattr__(self, "origins", np.asarray(self.origins, dtype=int))
        if len(self.shape) != self.grid.ndim:
            raise ValueError(f"window shape {self.shape} for a {self.grid.ndim}D grid")
        if not all(1 <= n <= N for n, N in zip(self.shape, self.grid.shape)):
            raise ValueError(f"window {self.shape} does not fit the grid {self.grid.shape}")
        if self.origins.ndim != 2 or self.origins.shape[1] != self.grid.ndim:
            raise ValueError(
                f"origins must have shape (n_cells, {self.grid.ndim}), got {self.origins.shape}"
            )

    @property
    def n_cells(self) -> int:
        return len(self.origins)

    @property
    def ndim(self) -> int:
        return self.grid.ndim

    @property
    def n_points(self) -> int:
        """Points in one window."""
        return int(np.prod(self.shape))

    def spans(self, axis: int) -> bool:
        """Whether the window covers the whole of array axis ``axis``, so it wraps onto itself."""
        return self.shape[axis] == self.grid.shape[axis]

    @property
    def spans_rows(self) -> bool:
        """2D: whether the window covers the whole y axis."""
        return self.spans(self.ndim - 2)

    @property
    def spans_cols(self) -> bool:
        """2D: whether the window covers the whole x axis."""
        return self.spans(self.ndim - 1)

    def indices(self) -> tuple[np.ndarray, ...]:
        """One ``(n_cells, shape[axis])`` array of wrapped grid indices per array axis."""
        return tuple(
            (self.origins[:, axis, None] + np.arange(n)) % N
            for axis, (n, N) in enumerate(zip(self.shape, self.grid.shape))
        )

    # Every transfer between windows and grid goes through one set of flattened grid
    # indices, ``(n_cells, n_points)``, computed once per set of windows. With them a
    # gather is a contiguous ``take_along_axis`` and an accumulation a ``bincount``,
    # each a few nanoseconds per element. The three-array fancy indexing they replace
    # cost about ten times that, and -- measured, stage 3 at 50 cells -- a per-cell
    # loop of fancy-indexed adds was slower than the dense sum it was meant to beat.

    @cached_property
    def flat(self) -> np.ndarray:
        """``(n_cells, n_points)`` index of every window point into the flattened grid."""
        flat = np.zeros((self.n_cells,) + (1,) * self.ndim, dtype=int)
        for axis, index in enumerate(self.indices()):
            expand = [None] * self.ndim
            expand[axis] = slice(None)
            flat = flat * self.grid.shape[axis] + index[(slice(None), *expand)]
        return flat.reshape(self.n_cells, -1)

    def _stack(self, values: np.ndarray) -> np.ndarray:
        return values.reshape(self.n_cells, -1)

    def _patches(self, flat_values: np.ndarray) -> np.ndarray:
        return flat_values.reshape(self.n_cells, *self.shape)

    def extract(self, values: np.ndarray) -> np.ndarray:
        """Every cell's window from the dense stack: ``(n_cells, *shape)``."""
        return self._patches(np.take_along_axis(self._stack(values), self.flat, axis=1))

    def extract_field(self, field: np.ndarray) -> np.ndarray:
        """Every cell's window of one shared grid-sized field: ``(n_cells, *shape)``.

        For the quantities that couple cells -- the sum of squared fields the
        repulsion needs, its Laplacian for adhesion -- which live on the grid once and
        are read by every cell where it sits.
        """
        return self._patches(np.take(field.ravel(), self.flat))

    def scatter(self, patches: np.ndarray, into: np.ndarray) -> np.ndarray:
        """Write patches into a dense stack at the windows, in place.

        Different cells' windows overlap on the grid but live in different slices of
        the stack, so the assignment never collides with itself.
        """
        np.put_along_axis(self._stack(into), self.flat, self._stack(patches), axis=1)
        return into

    def add(self, patches: np.ndarray, into: np.ndarray) -> np.ndarray:
        """Add patches into a dense stack at the windows, in place.

        The update of a windowed step: the rate is known only on the windows and the
        field outside them is left alone. Unique indices per cell, so gather, add and
        put back is exact.
        """
        stack = self._stack(into)
        current = np.take_along_axis(stack, self.flat, axis=1)
        np.put_along_axis(stack, self.flat, current + self._stack(patches), axis=1)
        return into

    def accumulate(self, patches: np.ndarray, into: np.ndarray) -> np.ndarray:
        """Sum patches into one shared grid-sized field, in place.

        The way a grid-sized quantity that couples cells -- the sum of squared fields
        the repulsion reads -- is built without touching the dense stack: each cell
        contributes its patch where its window sits. Different cells' windows overlap
        on the shared grid, so this is an accumulation, which ``bincount`` does in one
        contiguous pass over the patch values with the flattened indices as bins.
        """
        into += np.bincount(
            self.flat.ravel(), weights=patches.ravel(), minlength=self.grid.n_points
        ).reshape(self.grid.shape)
        return into

    def mask(self) -> np.ndarray:
        """``(n_cells, *grid.shape)`` booleans, ``True`` inside each cell's window."""
        inside = np.zeros((self.n_cells, self.grid.n_points), dtype=bool)
        np.put_along_axis(inside, self.flat, True, axis=1)
        return inside.reshape(self.n_cells, *self.grid.shape)

    def coordinates(self) -> tuple[np.ndarray, ...]:
        """``(X, Y[, Z])``, each ``(n_cells, *shape)``: positions of the window points.

        *Unwrapped*: a window straddling the boundary gets coordinates that run past
        the box edge rather than jumping back to zero, so any average over a window
        is continuous. Wrap the result with the box afterwards.
        """
        target = (self.n_cells, *self.shape)
        spacings = self.grid.spacings
        components = []
        for component in range(self.ndim):
            axis = self.ndim - 1 - component                     # x is the last array axis
            line = (self.origins[:, axis, None] + np.arange(self.shape[axis])) * spacings[axis]
            expand = [None] * self.ndim
            expand[axis] = slice(None)
            components.append(np.broadcast_to(line[(slice(None), *expand)], target))
        return tuple(components)

    # ------------------------------------------------------------- stencils on patches
    #
    # The same operators the grid provides, on a stack of windows ``(n_cells, *shape)``.
    # A window has no periodic wrap inside it, so a stencil needs a border: zero, since
    # the field beyond a window is below the window threshold by construction -- or
    # wrapped, on an axis the window spans entirely, where it is the periodic grid
    # itself and must give the dense result exactly. Every windowed derivative of the
    # free energy is built from these three.

    def pad(self, patches: np.ndarray) -> np.ndarray:
        """Patches with a one-point border on every grid axis."""
        padded = patches
        for axis in range(self.ndim):
            width = [(0, 0)] * (self.ndim + 1)
            width[1 + axis] = (1, 1)
            padded = np.pad(padded, width, mode="wrap" if self.spans(axis) else "constant")
        return padded

    def _shifted(self, padded: np.ndarray, axis: int, offset: int) -> np.ndarray:
        """The interior of ``padded``, displaced ``offset`` points along array ``axis``."""
        index = [slice(None)] + [slice(1, -1)] * self.ndim
        index[1 + axis] = slice(1 + offset, padded.shape[1 + axis] - 1 + offset)
        return padded[tuple(index)]

    def gradient(self, patches: np.ndarray) -> tuple[np.ndarray, ...]:
        """``(d/dx, d/dy[, d/dz])`` by second-order central differences, as :meth:`Grid.gradient`."""
        padded = self.pad(patches)
        spacings = self.grid.spacings
        return tuple(
            (self._shifted(padded, axis, 1) - self._shifted(padded, axis, -1)) / (2.0 * spacings[axis])
            for axis in reversed(range(self.ndim))
        )

    def forward_gradient(self, patches: np.ndarray) -> tuple[np.ndarray, ...]:
        """First differences to the next point, as :meth:`Grid.forward_gradient` --
        the pair that is the exact adjoint of :meth:`laplacian`."""
        padded = self.pad(patches)
        centre = self._shifted(padded, 0, 0)
        spacings = self.grid.spacings
        return tuple(
            (self._shifted(padded, axis, 1) - centre) / spacings[axis]
            for axis in reversed(range(self.ndim))
        )

    def laplacian(self, patches: np.ndarray) -> np.ndarray:
        """``2 ndim + 1``-point Laplacian, as :meth:`Grid.laplacian`."""
        padded = self.pad(patches)
        centre = self._shifted(padded, 0, 0)
        total = np.zeros_like(centre)
        for axis, h in enumerate(self.grid.spacings):
            total += (
                self._shifted(padded, axis, 1) + self._shifted(padded, axis, -1) - 2.0 * centre
            ) / h**2
        return total


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
    """``(n_cells, *grid.shape)``. Prefer the methods below; this is the dense backing store."""

    windows: Windows | None = None
    """Where each cell currently is, when the dynamics run on windows; ``None`` otherwise.

    Part of the state because it goes stale as cells move: :meth:`refresh_windows`
    recomputes it, and the stepper calls that every so many steps.
    """

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != self.grid.ndim + 1 or self.values.shape[1:] != self.grid.shape:
            raise ValueError(
                f"fields must have shape (n_cells, {', '.join(map(str, self.grid.shape))}), "
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
        """The field of a single cell, of the grid's shape."""
        return self.values[index]

    def __iter__(self):
        return iter(self.values)

    @property
    def occupancy(self) -> np.ndarray:
        """``sum_i phi_i``, of the grid's shape. Close to 1 everywhere in a confluent tissue."""
        return self.values.sum(axis=0)

    def gradient(self) -> tuple[np.ndarray, ...]:
        """Per-cell gradients, one ``(n_cells, *grid.shape)`` array per component."""
        return self.grid.gradient(self.values)

    def laplacian(self) -> np.ndarray:
        return self.grid.laplacian(self.values)

    def find_windows(self, margin: int = 3, threshold: float = 1e-6) -> Windows:
        """The patch of the grid each cell occupies, with ``margin`` points to spare.

        Found from the fields themselves: along each axis, the indices where a cell
        exceeds ``threshold``, allowing for a cell that straddles the periodic
        boundary. All windows share one shape, the largest extent over cells plus the
        margin on each side, capped at the grid -- so on a box barely bigger than a
        cell the window *is* the grid and everything degenerates to the dense
        computation. Each cell's occupied run is centred in its window.

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
        ndim = self.grid.ndim
        shape, origins = [], []
        for axis in range(ndim):
            others = tuple(1 + a for a in range(ndim) if a != axis)
            starts, extents = _circular_spans(present.any(axis=others))
            n = min(self.grid.shape[axis], int(extents.max()) + 2 * margin)
            shape.append(n)
            origins.append((starts - (n - extents) // 2) % self.grid.shape[axis])
        return Windows(self.grid, tuple(shape), np.column_stack(origins))

    def refresh_windows(self, margin: int = 3, threshold: float = 1e-6) -> Windows:
        """Recompute :attr:`windows` from the fields and zero everything outside them.

        The zeroing is what keeps a windowed run honest. Between refreshes the field
        outside a window is never updated, so as a cell moves on it would leave behind
        a frozen tail -- below the threshold, but litter all the same, and litter that
        the next refresh would count as part of the cell. Clearing it makes "outside
        the window" mean exactly zero, and costs one pass over the stack.
        """
        self.windows = self.find_windows(margin, threshold)
        self.values[~self.windows.mask()] = 0.0
        return self.windows

    def copy(self) -> PhaseFields:
        return PhaseFields(self.grid, self.values.copy(), self.windows)


def _radii(radius, n_cells: int) -> np.ndarray:
    """One positive radius per cell, from a single value or one per cell."""
    radii = np.broadcast_to(np.asarray(radius, dtype=float), (n_cells,))
    if np.any(radii <= 0):
        raise ValueError(f"radius must be positive, got {radius}")
    return radii


def seed(
    grid: Grid,
    centres: np.ndarray,
    radius: float | np.ndarray,
    interface_width: float,
) -> PhaseFields:
    r"""Round cells of the given ``radius``, one per row of ``centres``.

    Each field is laid down as the equilibrium interface profile of a symmetric double
    well,

    .. math:: \phi(r) = \tfrac{1}{2}\left[1 - \tanh\frac{r - R}{\sqrt{2}\,\lambda}\right]

    so the fields start close to a stationary state of the free energy and the first
    steps relax cell *arrangement* rather than burning time sharpening interfaces that
    were seeded with the wrong profile. ``centres`` has one ``(x, y[, z])`` row per cell.
    ``radius`` is one value for every cell, or one per cell.

    ``interface_width`` needs several grid points across it -- roughly
    ``interface_width >= 3 * dx`` -- or the interface will be under-resolved and pinned
    to the grid. A warning-free check of that is left to you once the free energy fixes
    what the width should be.
    """
    centres = np.atleast_2d(np.asarray(centres, dtype=float))
    if centres.ndim != 2 or centres.shape[1] != grid.ndim:
        raise ValueError(f"centres must have shape (n_cells, {grid.ndim}), got {centres.shape}")
    radii = _radii(radius, len(centres))
    if interface_width <= 0:
        raise ValueError(f"interface_width must be positive, got {interface_width}")

    fields = grid.zeros(len(centres))
    for i, centre in enumerate(centres):
        distance = grid.distance_to(centre)
        fields[i] = 0.5 * (1.0 - np.tanh((distance - radii[i]) / (np.sqrt(2.0) * interface_width)))
    return PhaseFields(grid, fields)


def seed_tessellated(
    grid: Grid,
    centres: np.ndarray,
    radius: float,
    interface_width: float,
) -> PhaseFields:
    r"""Cells shaped like their Voronoi regions, clipped to ``radius``.

    Round cells cannot tile space, so seeding a confluent tissue with them guarantees
    heavy overlap -- and the mechanical forces that produces are far larger than
    anything the relaxed tissue ever sees. Under force balance those forces set the
    velocity, so the run opens with a spike that is pure seeding artefact. This starts
    the tissue where it was going to end up instead.

    For each grid point, let :math:`d_i` be the distance to cell ``i`` and :math:`d_j`
    the distance to the nearest *other* cell. The perpendicular bisector between them
    lies where the two are equal, so

    .. math:: s_i = \min\!\left( \frac{d_j - d_i}{2},\; R - d_i \right)

    is the signed distance into cell ``i`` -- from its Voronoi boundary, or from a
    sphere of radius ``R``, whichever is nearer. The profile is then the same ``tanh``
    of that distance as :func:`seed` uses.

    Taking the smaller of the two keeps this right at any density: near confluence the
    Voronoi term binds and cells tile, while below it the sphere binds and cells are
    round and separate, as they should be.

    With a radius per cell the bisector moves: it sits where :math:`d_j - R_j = d_i -
    R_i`, the additively weighted Voronoi boundary, so a larger cell is born larger
    and its neighbours correspondingly smaller, with the boundary displaced by half
    the difference in radii. With one radius for all this reduces to the plain
    bisector, computed exactly as before.
    """
    centres = np.atleast_2d(np.asarray(centres, dtype=float))
    if centres.ndim != 2 or centres.shape[1] != grid.ndim:
        raise ValueError(f"centres must have shape (n_cells, {grid.ndim}), got {centres.shape}")
    if len(centres) < 2:
        return seed(grid, centres, radius, interface_width)
    radii = _radii(radius, len(centres))
    if interface_width <= 0:
        raise ValueError(f"interface_width must be positive, got {interface_width}")

    distances = np.stack([grid.distance_to(centre) for centre in centres])
    expand = (slice(None),) + (None,) * grid.ndim
    if np.all(radii == radii[0]):
        # One radius: the plain bisector, kept on its own path so that the arithmetic
        # -- and every trajectory seeded this way -- is unchanged to the last bit.
        shifted = distances
        own = radii[0] - distances
    else:
        shifted = distances - radii[expand]
        own = radii[expand] - distances
    closest, runner_up = np.partition(shifted, 1, axis=0)[:2]

    # For whichever cell is nearest at a point, the competitor is the runner-up; for
    # every other cell the competitor is the nearest one.
    competitor = np.where(shifted <= closest + 1e-12, runner_up, closest)
    signed = np.minimum(0.5 * (competitor - shifted), own)
    return PhaseFields(grid, 0.5 * (1.0 + np.tanh(signed / (np.sqrt(2.0) * interface_width))))
