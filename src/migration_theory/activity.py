"""Self-propulsion: the polarity each cell crawls along, and how it enters the fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import PhaseFields, Windows

__all__ = ["Polarity", "advection", "passive_forces", "ImposedVelocity", "ForceBalance"]


class Polarity:
    r"""A propulsion direction per cell, rotating diffusively.

    .. math::

        \mathbf{v}_i = v_0 \mathbf{n}_i, \qquad
        \mathrm{d}\theta_i = \sqrt{2 D_r \,\mathrm{d}t}\; \eta_i \quad (2D)

    The minimal active cell: it crawls at a fixed speed along a direction that is
    persistent and then forgotten. No alignment with neighbours and no coupling back
    from the field -- both are extensions that belong on top of a working passive
    model.

    In 2D the direction is an angle, diffusing as above, and ``angles`` is the state:
    ``Polarity(angles, ...)`` as it always was. In 3D the direction is a unit vector
    diffusing on the sphere, and ``directors`` is the state: ``Polarity(directors=...)``
    or :meth:`random` with ``dimension=3``. In either case the direction
    autocorrelation decays as ``exp(-(d - 1) D_r t)``, so the persistence time is
    ``1 / ((d - 1) D_r)`` -- ``1 / D_r`` in 2D, as before, and half that in 3D.

    ``speed = 0`` (the default) makes the tissue passive without any other change, so
    the passive limit is always one parameter away.

    ``speed`` and ``rotational_diffusion`` are one number for every cell, or one per
    cell -- a population of cells that differ in how fast they crawl or how long they
    hold a direction. A single number is kept as a plain float, so a uniform tissue
    computes exactly what it always did.
    """

    def __init__(
        self,
        angles: np.ndarray | None = None,
        speed: float | np.ndarray = 0.0,
        rotational_diffusion: float | np.ndarray = 1.0,
        rng: np.random.Generator | None = None,
        *,
        directors: np.ndarray | None = None,
    ) -> None:
        if (angles is None) == (directors is None):
            raise ValueError("give either angles (2D) or directors (any dimension)")
        if angles is not None:
            self.angles = np.asarray(angles, dtype=float).ravel()
            self._directors = None
        else:
            directors = np.asarray(directors, dtype=float)
            if directors.ndim != 2 or directors.shape[1] < 2:
                raise ValueError(f"directors must have shape (n_cells, ndim), got {directors.shape}")
            self.angles = None
            self._directors = directors / np.linalg.norm(directors, axis=1, keepdims=True)
        self.speed = _per_cell(speed, self.n_cells, "speed")
        self.rotational_diffusion = _per_cell(rotational_diffusion, self.n_cells,
                                              "rotational_diffusion")
        self.rng = np.random.default_rng() if rng is None else rng

    def __repr__(self) -> str:
        return (f"Polarity({self.n_cells} cells in {self.dimension}D, "
                f"speed={_show(self.speed)}, rotational_diffusion={_show(self.rotational_diffusion)})")

    @classmethod
    def random(
        cls,
        n_cells: int,
        speed: float = 0.0,
        rotational_diffusion: float = 1.0,
        rng: np.random.Generator | None = None,
        dimension: int = 2,
    ) -> Polarity:
        """Uniformly random initial directions."""
        rng = np.random.default_rng() if rng is None else rng
        if dimension == 2:
            return cls(rng.uniform(0.0, 2.0 * np.pi, n_cells), speed, rotational_diffusion, rng)
        return cls(None, speed, rotational_diffusion, rng,
                   directors=rng.standard_normal((n_cells, dimension)))

    @classmethod
    def still(cls, n_cells: int, dimension: int = 2) -> Polarity:
        """A passive tissue's polarity: some direction, never used."""
        if dimension == 2:
            return cls(np.zeros(n_cells))
        directors = np.zeros((n_cells, dimension))
        directors[:, 0] = 1.0
        return cls(None, directors=directors)

    @property
    def n_cells(self) -> int:
        return len(self.angles) if self.angles is not None else len(self._directors)

    @property
    def dimension(self) -> int:
        return 2 if self.angles is not None else self._directors.shape[1]

    @property
    def directors(self) -> np.ndarray:
        """``(n_cells, ndim)`` unit vectors along each cell's polarity."""
        if self.angles is not None:
            return np.column_stack([np.cos(self.angles), np.sin(self.angles)])
        return self._directors

    @property
    def velocities(self) -> np.ndarray:
        """``(n_cells, ndim)`` propulsion velocity of each cell."""
        return _column(self.speed) * self.directors

    @property
    def persistence_time(self) -> float | np.ndarray:
        """``1 / ((d - 1) D_r)``: how long a cell remembers its direction. Per cell if
        ``rotational_diffusion`` is."""
        rate = (self.dimension - 1) * np.asarray(self.rotational_diffusion, dtype=float)
        with np.errstate(divide="ignore"):
            time = np.where(rate == 0, np.inf, 1.0 / np.where(rate == 0, 1.0, rate))
        return float(time) if time.ndim == 0 else time

    @property
    def persistence_length(self) -> float | np.ndarray:
        """How far a cell travels before its direction decorrelates."""
        return self.speed * self.persistence_time

    def rotate(self, dt: float) -> None:
        """Advance the directions by one timestep of rotational diffusion, in place.

        2D: the angle takes a Gaussian step of variance ``2 D_r dt``. Higher: the unit
        vector takes a Gaussian step of the same variance per transverse direction,
        projected onto the tangent plane and renormalised -- diffusion on the sphere,
        whose autocorrelation decays as ``exp(-(d - 1) D_r t)``.
        """
        if np.all(np.asarray(self.rotational_diffusion) == 0):
            return
        scale = np.sqrt(2.0 * self.rotational_diffusion * dt)
        if self.angles is not None:
            self.angles = self.angles + scale * self.rng.standard_normal(self.n_cells)
            return
        p = self._directors
        kick = _column(scale) * self.rng.standard_normal(p.shape)
        kick -= (kick * p).sum(axis=1, keepdims=True) * p          # tangent to the sphere
        p = p + kick
        self._directors = p / np.linalg.norm(p, axis=1, keepdims=True)

    def copy(self) -> Polarity:
        if self.angles is not None:
            return Polarity(self.angles.copy(), self.speed, self.rotational_diffusion, self.rng)
        return Polarity(None, self.speed, self.rotational_diffusion, self.rng,
                        directors=self._directors.copy())


def _per_cell(value, n_cells: int, name: str) -> float | np.ndarray:
    """A non-negative parameter: a plain float, or an ``(n_cells,)`` array."""
    if np.ndim(value) == 0:
        value = float(value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        return value
    values = np.asarray(value, dtype=float).ravel()
    if len(values) != n_cells:
        raise ValueError(f"{name} must have one value per cell ({n_cells}), got {len(values)}")
    if np.any(values < 0):
        raise ValueError(f"{name} must be non-negative, got {values}")
    return values


def _column(value):
    """A per-cell array as an ``(n_cells, 1)`` column, so it broadcasts over vectors;
    a float unchanged."""
    return value[:, None] if np.ndim(value) else value


def _show(value) -> str:
    if np.ndim(value) == 0:
        return f"{value:g}"
    return f"[{np.min(value):g} .. {np.max(value):g}]"


Gradient = tuple[np.ndarray, ...]
"""The components of a gradient, in ``(x, y[, z])`` order."""


def field_gradient(
    fields: PhaseFields, windows: Windows | None = None, patches: np.ndarray | None = None
) -> Gradient:
    """The central gradient of every cell's field: dense, or on windows.

    ``patches`` is the windows' contents if the caller has gathered them already.
    """
    if windows is None:
        return fields.gradient()
    if patches is None:
        patches = windows.extract(fields.values)
    return windows.gradient(patches)


def advection(
    fields: PhaseFields,
    velocities: np.ndarray,
    gradient: Gradient | None = None,
    windows: Windows | None = None,
) -> np.ndarray:
    r"""``-v_i . grad(phi_i)``, the contribution of self-propulsion to ``d(phi_i)/dt``.

    Translating a field rigidly at velocity ``v`` means ``d(phi)/dt = -v . grad(phi)``,
    so this term is what makes a cell move rather than merely change shape. It is
    non-variational -- it does not come from any free energy -- which is exactly what
    makes the tissue active.

    ``gradient`` is the central-difference gradient of the fields, if the caller
    already has it: under force balance the same gradient enters
    :func:`passive_forces`, and the stepper computes it once for both. With
    ``windows`` everything is on each cell's patch and the result is ``(n_cells, h, w)``;
    otherwise ``(n_cells, ny, nx)``.
    """
    ndim = fields.grid.ndim
    velocities = np.atleast_2d(np.asarray(velocities, dtype=float))
    if velocities.shape != (fields.n_cells, ndim):
        raise ValueError(
            f"velocities must have shape ({fields.n_cells}, {ndim}), got {velocities.shape}"
        )
    components = field_gradient(fields, windows) if gradient is None else gradient
    per_cell = (slice(None), None) + (None,) * (ndim - 1)          # (n_cells, 1, ..., 1)
    total = velocities[:, 0][per_cell] * components[0]
    for k in range(1, ndim):
        total = total + velocities[:, k][per_cell] * components[k]
    return -total


def passive_forces(
    fields: PhaseFields,
    mu: np.ndarray,
    gradient: Gradient | None = None,
    windows: Windows | None = None,
) -> np.ndarray:
    r"""``(n_cells, ndim)`` mechanical force on each cell from the free energy.

    .. math:: \mathbf{F}_i = \int \mu_i \nabla\phi_i \, \mathrm{d}x,
              \qquad \mu_i = \frac{\delta F}{\delta \phi_i}

    Rigidly translating cell ``i`` by ``u`` changes the field by ``-u.grad(phi_i)``, so
    the energy changes by ``-u . INT mu_i grad(phi_i)`` and the force is that integral.

    ``mu`` is passed in rather than recomputed because the stepper already has it --
    the same array drives both the relaxation and the motion. ``gradient`` likewise,
    when the caller has it already. With ``windows``, ``mu`` and ``gradient`` are
    patches and the integral is a window sum.

    Because the free energy cannot change when every cell is translated together, these
    forces sum to zero: a cell pushed by a neighbour pushes back just as hard. That is
    what makes a blocked cell stall instead of ploughing on, and it is worth checking
    numerically rather than assuming -- see the tests.
    """
    components = field_gradient(fields, windows) if gradient is None else gradient
    if windows is None:
        return np.column_stack([fields.grid.integrate(mu * c) for c in components])
    axes = tuple(range(-windows.ndim, 0))
    volume = fields.grid.cell_volume
    return np.column_stack([(mu * c).sum(axis=axes) * volume for c in components])


@dataclass(frozen=True)
class ImposedVelocity:
    """Cells translate at ``v0`` along their polarity, whatever the tissue does.

    The simple choice, and wrong once activity is strong: an imposed velocity is not a
    force, so nothing can resist it. Driven hard enough, cells pass *through* their
    neighbours rather than stalling. Kept for comparison against :class:`ForceBalance`.
    """

    def velocities(self, tissue, mu: np.ndarray, gradient: Gradient | None = None,
                   windows: Windows | None = None) -> np.ndarray:
        return tissue.polarity.velocities


@dataclass(frozen=True)
class ForceBalance:
    r"""Velocity from overdamped force balance, so neighbours can push back.

    .. math::

        \mathbf{F}^{a}_i = \frac{E_a}{R}\,\mathbf{p}_i,
        \qquad
        \xi \mathbf{v}_i = \mathbf{F}^{a}_i + \mathbf{F}^{\mathrm{passive}}_i

    ``E_a`` is the mechanical work a cell can do over about one cell radius, which
    makes it directly comparable with the passive free energy -- unlike ``v0``, which
    has no relation to it. Dividing by ``R`` turns that energy into a force.

    The velocity now depends on the fields through the passive force, and that feedback
    is the whole point: a cell driven into an immovable neighbour develops an opposing
    force, its velocity falls to zero, and the neighbour feels an equal and opposite
    push. With an imposed velocity none of that can happen.

    A cell with nothing in its way moves at ``E_a / (R xi)``.

    Each of the three parameters is one number for every cell or an array with one
    per cell, for a population of cells of different sizes, drives or drags.
    """

    active_energy: float | np.ndarray
    cell_radius: float | np.ndarray
    cell_friction: float | np.ndarray

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.cell_radius) <= 0):
            raise ValueError(f"cell_radius must be positive, got {self.cell_radius}")
        if np.any(np.asarray(self.cell_friction) <= 0):
            raise ValueError(f"cell_friction must be positive, got {self.cell_friction}")

    @property
    def free_speed(self) -> float | np.ndarray:
        """``E_a / (R xi)``: the speed of a cell with nothing in its way."""
        return self.active_energy / (self.cell_radius * self.cell_friction)

    def velocities(self, tissue, mu: np.ndarray, gradient: Gradient | None = None,
                   windows: Windows | None = None) -> np.ndarray:
        active = _column(self.active_energy / self.cell_radius) * tissue.polarity.directors
        passive = passive_forces(tissue.fields, mu, gradient, windows)
        return (active + passive) / _column(self.cell_friction)
