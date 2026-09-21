r"""Time stepping.

.. math::

    \frac{\partial \phi_i}{\partial t}
        + \mathbf{v}_i \cdot \nabla \phi_i
        = -\frac{1}{\gamma} \frac{\delta F}{\delta \phi_i}

The right-hand side is gradient flow against a friction ``gamma``: on its own it can
only lower the free energy. The advection on the left comes from no free energy at all
-- that asymmetry is precisely what makes the tissue active rather than merely relaxing.

``gamma`` is taken to be a constant, so the interface width is a physical parameter of
the model rather than a resolution knob: changing it changes how fast shapes relax.
``POINTS_PER_INTERFACE`` controls resolution; the width itself does not.

Stepping is explicit Euler. The timestep it tolerates is not a matter of taste: each
free-energy term reports its own limit and :class:`ExplicitEuler` takes the smallest,
so a badly chosen ``dt`` is caught before a run rather than discovered in the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .activity import advection, field_gradient
from .free_energy import FreeEnergy
from .tissue import Tissue

__all__ = ["ExplicitEuler", "UnstableTimestep", "Diverged", "run"]


class UnstableTimestep(RuntimeError):
    """Raised when the requested ``dt`` exceeds what the physics will tolerate."""


class Diverged(RuntimeError):
    """Raised when the fields stop being finite.

    A free energy that is unbounded below -- too much adhesion, most likely -- sends
    the fields to infinity and then to NaN. Without this the run completes and returns
    a trajectory full of NaN, which is worse than an error because it looks like a
    result.
    """


@dataclass
class ExplicitEuler:
    """Forward Euler on the fields, with the polarity rotated afterwards."""

    dt: float
    friction: float = 1.0
    """The friction ``gamma``. Larger means slower relaxation."""

    safety: float = 0.9
    """Fraction of the stability limit ``dt`` is allowed to reach."""

    propulsion: object | None = None
    """How cell velocities arise -- an :class:`~migration_theory.activity.ImposedVelocity`
    or :class:`~migration_theory.activity.ForceBalance`. ``None`` means passive."""

    windowed: bool = False
    """Compute each step on the cells' windows rather than over the whole grid.

    The derivative of every term, the velocities and the advection are evaluated on
    each cell's own patch and the update touches only the windows; the field beyond
    them is zero. What remains grid-sized is the sum of squared fields the coupling
    terms read, a few passes over the stack. The trajectory agrees with the dense one
    to the tail the windows drop, below ``1e-6`` in the field.
    """

    refresh_every: int = 20
    """Steps between recomputing the windows, when ``windowed``.

    Cells move, so a window has to follow its cell. A refresh costs two passes over
    the stack and must come before a cell has moved further than the window margin
    of three points: at the fastest speed in these sweeps, about ``0.3 um/s``, and a
    step of ``0.06 s``, that is one point in about fifty steps, so twenty is safe with
    room to spare.
    """

    steps_taken: int = field(default=0, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.friction <= 0:
            raise ValueError(f"friction must be positive, got {self.friction}")
        if not 0 < self.safety <= 1:
            raise ValueError(f"safety must lie in (0, 1], got {self.safety}")
        if self.refresh_every < 1:
            raise ValueError(f"refresh_every must be at least 1, got {self.refresh_every}")

    def stability_limits(self, tissue: Tissue, free_energy: FreeEnergy) -> dict[str, float]:
        """Each term's largest tolerable timestep, for seeing which one binds."""
        return {
            type(term).__name__: float(term.stability_limit(tissue.fields, self.friction))
            for term in free_energy
            if hasattr(term, "stability_limit")
        }

    def max_stable_dt(self, tissue: Tissue, free_energy: FreeEnergy) -> float:
        limits = self.stability_limits(tissue, free_energy)
        return min(limits.values(), default=np.inf)

    def grid_peclet(self, tissue: Tissue, free_energy: FreeEnergy) -> float:
        """``v dx gamma / K``: advection strength relative to diffusion, per grid cell.

        Central differences with forward Euler are unconditionally unstable for pure
        advection. Here they survive only because the gradient term supplies diffusion,
        and only while this stays below about 2 -- past that, advection outruns what
        the diffusion can damp and grid-scale oscillations grow behind the moving
        interface. Zero when the tissue is passive.
        """
        speed = self.peak_speed(tissue, free_energy)
        if speed == 0:
            return 0.0
        diffusivity = sum(
            getattr(term, "K", 0.0)
            for term in free_energy
            if type(term).__name__ == "GradientEnergy"
        ) / self.friction
        if diffusivity <= 0:
            return np.inf
        return speed * tissue.grid.dx / diffusivity

    def peak_speed(self, tissue: Tissue, free_energy: FreeEnergy) -> float:
        """Largest cell speed right now.

        Measured rather than read off a parameter, because under force balance the
        velocity depends on the configuration -- a blocked cell is slower than ``v0``
        and a cell being squeezed out of a gap can be faster.
        """
        if self.propulsion is None:
            return 0.0
        mu = free_energy.functional_derivative(tissue.fields)
        velocities = self.propulsion.velocities(tissue, mu)
        return float(np.max(np.hypot(velocities[:, 0], velocities[:, 1])))

    def check(self, tissue: Tissue, free_energy: FreeEnergy) -> None:
        """Raise if ``dt`` is too large, naming the term responsible."""
        limits = self.stability_limits(tissue, free_energy)
        allowed = self.safety * min(limits.values(), default=np.inf)
        if self.dt > allowed:
            binding = min(limits, key=limits.get)  # type: ignore[arg-type]
            raise UnstableTimestep(
                f"dt = {self.dt:.3e} exceeds {self.safety:g} x the stability limit "
                f"{min(limits.values()):.3e}, set by {binding}. "
                f"Limits: { {k: f'{v:.2e}' for k, v in limits.items()} }"
            )
        peclet = self.grid_peclet(tissue, free_energy)
        if peclet > 2.0:
            raise UnstableTimestep(
                f"grid Peclet number {peclet:.2f} exceeds 2: self-propulsion outruns the "
                "diffusion that keeps central-difference advection stable. Reduce the "
                "speed, refine the grid, or switch the advection term to upwinding."
            )

    def step(self, tissue: Tissue, free_energy: FreeEnergy) -> None:
        """Advance by ``dt``, in place.

        No stability check here -- it costs a pass over the fields and the answer
        barely changes step to step, so :func:`run` does it once up front.
        """
        fields = tissue.fields
        windows = patches = None
        if self.windowed:
            if fields.windows is None or self.steps_taken % self.refresh_every == 0:
                fields.refresh_windows()
            windows = fields.windows
            # Gathered once and handed to everything below. Measured, gathering the
            # patches afresh in every term was most of the windowed step's cost.
            patches = windows.extract(fields.values)

        mu = free_energy.functional_derivative(fields, windows, patches)
        rate = -mu / self.friction
        if self.propulsion is not None:
            # mu and the gradient are handed on rather than recomputed: under force
            # balance the velocity is built from the same functional derivative that
            # drives the relaxation, and the passive force and the advection both
            # need the same central gradient of the fields.
            gradient = field_gradient(fields, windows, patches)
            velocities = self.propulsion.velocities(tissue, mu, gradient, windows)
            rate = rate + advection(fields, velocities, gradient, windows)

        if windows is None:
            fields.values += self.dt * rate
        else:
            windows.add(self.dt * rate, fields.values)
        self.steps_taken += 1
        tissue.polarity.rotate(self.dt)
        tissue.time += self.dt


def run(
    tissue: Tissue,
    free_energy: FreeEnergy,
    stepper: ExplicitEuler,
    n_steps: int,
    sample_every: int = 0,
    observer: Callable[[Tissue], dict[str, Any]] | None = None,
    check: bool = True,
) -> list[dict[str, Any]]:
    """Step ``n_steps`` times, sampling as it goes.

    Each sample records the time, the total free energy and its per-term breakdown,
    plus whatever ``observer`` returns -- so what gets measured stays out of the
    integrator. The final state is always sampled, so the last row is the end of the
    run rather than the last multiple of ``sample_every``.
    """
    if check:
        stepper.check(tissue, free_energy)

    history: list[dict[str, Any]] = []

    def record(index: int) -> None:
        row: dict[str, Any] = {
            "step": index,
            "time": tissue.time,
            "energy": free_energy.energy(tissue.fields),
        }
        row.update(free_energy.breakdown(tissue.fields))
        if observer is not None:
            row.update(observer(tissue))
        history.append(row)

    for index in range(n_steps):
        if sample_every and index % sample_every == 0:
            record(index)
        stepper.step(tissue, free_energy)
    if sample_every:
        record(n_steps)
    return history
