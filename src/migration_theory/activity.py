"""Self-propulsion: the polarity each cell crawls along, and how it enters the fields."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fields import PhaseFields

__all__ = ["Polarity", "advection", "passive_forces", "ImposedVelocity", "ForceBalance"]


@dataclass
class Polarity:
    r"""A propulsion direction per cell, rotating diffusively.

    .. math::

        \mathbf{v}_i = v_0 \mathbf{n}_i, \qquad
        \mathbf{n}_i = (\cos\theta_i, \sin\theta_i), \qquad
        \mathrm{d}\theta_i = \sqrt{2 D_r \,\mathrm{d}t}\; \eta_i

    The minimal active cell: it crawls at a fixed speed along a direction that is
    persistent over ``1 / D_r`` and forgotten after that. No alignment with neighbours
    and no coupling back from the field -- both are extensions that belong on top of a
    working passive model.

    ``speed = 0`` (the default) makes the tissue passive without any other change, so
    the passive limit is always one parameter away.
    """

    angles: np.ndarray
    speed: float = 0.0
    rotational_diffusion: float = 1.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        self.angles = np.asarray(self.angles, dtype=float).ravel()
        if self.speed < 0:
            raise ValueError(f"speed must be non-negative, got {self.speed}")
        if self.rotational_diffusion < 0:
            raise ValueError(
                f"rotational_diffusion must be non-negative, got {self.rotational_diffusion}"
            )

    @classmethod
    def random(
        cls,
        n_cells: int,
        speed: float = 0.0,
        rotational_diffusion: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> Polarity:
        """Uniformly random initial directions."""
        rng = np.random.default_rng() if rng is None else rng
        return cls(rng.uniform(0.0, 2.0 * np.pi, n_cells), speed, rotational_diffusion, rng)

    @property
    def n_cells(self) -> int:
        return len(self.angles)

    @property
    def directors(self) -> np.ndarray:
        """``(n_cells, 2)`` unit vectors along each cell's polarity."""
        return np.column_stack([np.cos(self.angles), np.sin(self.angles)])

    @property
    def velocities(self) -> np.ndarray:
        """``(n_cells, 2)`` propulsion velocity of each cell."""
        return self.speed * self.directors

    @property
    def persistence_time(self) -> float:
        """``1 / D_r``: how long a cell remembers its direction."""
        return np.inf if self.rotational_diffusion == 0 else 1.0 / self.rotational_diffusion

    @property
    def persistence_length(self) -> float:
        """How far a cell travels before its direction decorrelates."""
        return self.speed * self.persistence_time

    def rotate(self, dt: float) -> None:
        """Advance the directions by one timestep of rotational diffusion, in place."""
        if self.rotational_diffusion:
            scale = np.sqrt(2.0 * self.rotational_diffusion * dt)
            self.angles = self.angles + scale * self.rng.standard_normal(self.n_cells)

    def copy(self) -> Polarity:
        return Polarity(self.angles.copy(), self.speed, self.rotational_diffusion, self.rng)


Gradient = tuple[np.ndarray, np.ndarray]


def advection(
    fields: PhaseFields, velocities: np.ndarray, gradient: Gradient | None = None
) -> np.ndarray:
    r"""``-v_i . grad(phi_i)``, the contribution of self-propulsion to ``d(phi_i)/dt``.

    Translating a field rigidly at velocity ``v`` means ``d(phi)/dt = -v . grad(phi)``,
    so this term is what makes a cell move rather than merely change shape. It is
    non-variational -- it does not come from any free energy -- which is exactly what
    makes the tissue active.

    ``gradient`` is the central-difference ``fields.gradient()``, if the caller already
    has it: under force balance the same gradient enters :func:`passive_forces`, and
    the stepper computes it once for both.

    Returns ``(n_cells, ny, nx)``.
    """
    velocities = np.atleast_2d(np.asarray(velocities, dtype=float))
    if velocities.shape != (fields.n_cells, 2):
        raise ValueError(
            f"velocities must have shape ({fields.n_cells}, 2), got {velocities.shape}"
        )
    d_dx, d_dy = fields.gradient() if gradient is None else gradient
    return -(velocities[:, 0, None, None] * d_dx + velocities[:, 1, None, None] * d_dy)


def passive_forces(
    fields: PhaseFields, mu: np.ndarray, gradient: Gradient | None = None
) -> np.ndarray:
    r"""``(n_cells, 2)`` mechanical force on each cell from the free energy.

    .. math:: \mathbf{F}_i = \int \mu_i \nabla\phi_i \, \mathrm{d}x,
              \qquad \mu_i = \frac{\delta F}{\delta \phi_i}

    Rigidly translating cell ``i`` by ``u`` changes the field by ``-u.grad(phi_i)``, so
    the energy changes by ``-u . INT mu_i grad(phi_i)`` and the force is that integral.

    ``mu`` is passed in rather than recomputed because the stepper already has it --
    the same array drives both the relaxation and the motion. ``gradient`` likewise,
    when the caller has ``fields.gradient()`` already.

    Because the free energy cannot change when every cell is translated together, these
    forces sum to zero: a cell pushed by a neighbour pushes back just as hard. That is
    what makes a blocked cell stall instead of ploughing on, and it is worth checking
    numerically rather than assuming -- see the tests.
    """
    d_dx, d_dy = fields.gradient() if gradient is None else gradient
    return np.column_stack(
        [fields.grid.integrate(mu * d_dx), fields.grid.integrate(mu * d_dy)]
    )


@dataclass(frozen=True)
class ImposedVelocity:
    """Cells translate at ``v0`` along their polarity, whatever the tissue does.

    The simple choice, and wrong once activity is strong: an imposed velocity is not a
    force, so nothing can resist it. Driven hard enough, cells pass *through* their
    neighbours rather than stalling. Kept for comparison against :class:`ForceBalance`.
    """

    def velocities(self, tissue, mu: np.ndarray, gradient: Gradient | None = None) -> np.ndarray:
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
    """

    active_energy: float
    cell_radius: float
    cell_friction: float

    def __post_init__(self) -> None:
        if self.cell_radius <= 0:
            raise ValueError(f"cell_radius must be positive, got {self.cell_radius}")
        if self.cell_friction <= 0:
            raise ValueError(f"cell_friction must be positive, got {self.cell_friction}")

    @property
    def free_speed(self) -> float:
        """``E_a / (R xi)``: the speed of a cell with nothing in its way."""
        return self.active_energy / (self.cell_radius * self.cell_friction)

    def velocities(self, tissue, mu: np.ndarray, gradient: Gradient | None = None) -> np.ndarray:
        active = (self.active_energy / self.cell_radius) * tissue.polarity.directors
        return (active + passive_forces(tissue.fields, mu, gradient)) / self.cell_friction
