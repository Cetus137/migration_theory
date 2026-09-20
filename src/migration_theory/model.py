"""The model: the physical parameters, and everything derived from them.

One place where the parameters live, so that a script cannot quietly disagree with
another about what the model is. Everything a run needs -- the box, the grid, the free
energy, the initial tissue, the timestep -- is derived here rather than assembled at
each call site.

**Units.** Lengths are in microns. The grid spacing is a separate parameter, so
physics and resolution never share a knob: ``R = 12`` is a cell of radius 12 um
whatever ``grid_spacing`` happens to be, and halving ``grid_spacing`` refines the
grid without touching a single physical coefficient.

Times are in the unit ``gamma``, ``v_0`` and ``D_r`` are quoted in, named by
``time_unit``. That name is a label: nothing computes differently if you call it
seconds rather than minutes, but every reported timescale is then in a unit you can
reason about. Energies are in whatever ``alpha`` is quoted in -- the energy scale
cancels out of the dynamics, so only ratios matter.

A paper quoting lengths in grid spacings maps onto this directly by reading its
``dx`` as 1 um: its ``R = 12`` and interface width 2 become 12 um and 2 um here. Note ``xi`` here
means the *cell friction*, following Eq. (4) of the second paper; the interface width
is written ``w``.

Dimensions, in that system:

===========  ==========  ====================================================
``alpha``    ``E/L^2``   double-well depth
``K``        ``E``       gradient energy coefficient
``epsilon``  ``E/L^2``   overlap repulsion
``lambda``   ``E``       area constraint
``gamma``    ``ET/L^2``  friction
``R``        ``L``       cell radius
``D_r``      ``1/T``     rotational diffusion
``v_0``      ``L/T``     self-propulsion speed
===========  ==========  ====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .box import PeriodicBox
from .dynamics import ExplicitEuler
from .free_energy import (
    Adhesion,
    AreaConstraint,
    DoubleWell,
    FreeEnergy,
    GradientEnergy,
    Repulsion,
)
from .grid import Grid
from .activity import ForceBalance, ImposedVelocity, Polarity
# aliased: `seed` is also the name of tissue()'s random-seed argument
from .fields import seed as seed_circles
from .fields import seed_tessellated
from .initialise import evenly_spaced
from .tissue import Tissue

__all__ = ["Model"]

_HEX = np.sqrt(3.0) / 2.0  # area per cell of a triangular lattice, per unit spacing^2


@dataclass(frozen=True)
class Model:
    r"""A complete specification of one tissue simulation.

    .. math::

        F = \sum_i \int \left[ \alpha \phi_i^2 (\phi_i - 1)^2
              + \tfrac{K}{2} |\nabla \phi_i|^2 \right] \mathrm{d}x
          + \epsilon \sum_{i<j} \int \phi_i^2 \phi_j^2 \, \mathrm{d}x
          + \lambda \sum_i \left(1 - \frac{A_i}{A_0}\right)^2,
        \qquad A_0 = \pi R^2

    evolved by

    .. math::

        \frac{\partial \phi_i}{\partial t} + \mathbf{v}_i \cdot \nabla \phi_i
            = -\frac{1}{\gamma} \frac{\delta F}{\delta \phi_i}

    Frozen, so a sweep is :meth:`replace` and no run can mutate the model it came from.
    """

    # --- free energy ---
    alpha: float = 0.5
    """Double-well depth, ``E/L^2``."""

    K: float = 2.0
    """Gradient energy coefficient, ``E``. With ``alpha`` it fixes width and tension."""

    epsilon: float = 0.1
    """Overlap repulsion, ``E/L^2``."""

    adhesion: float = 0.0
    """``omega``: cell-cell adhesion strength, ``E``.

    Refunds part of the interfacial cost at a contact, giving an effective cell-cell
    tension ``sigma (2 - omega/K)``. Must stay below ``K``: past that the out-of-phase
    mode of the gradient Hessian goes unstable and the energy is unbounded below.
    """

    area_lambda: float = 6000.0
    """Area constraint strength, the ``lambda`` of the model, ``E``."""

    # --- geometry ---
    cell_radius: float = 12.0
    """``R``: the target cell radius in microns. Gives ``A_0 = pi R^2``."""

    packing: float = 1.0
    """Total target cell area as a fraction of the box. 1 is confluent.

    Sets the box, since ``R`` now fixes the cell size rather than the density.
    """

    n_cells: int = 16

    seeding: str = "tessellated"
    """How the initial fields are laid down: ``"tessellated"`` or ``"circles"``.

    Circles cannot tile the plane, so seeding a confluent tissue with them starts it
    heavily overlapped and produces mechanical forces several times larger than the
    relaxed tissue ever sees. Under force balance those forces set the velocity, so the
    run opens with a spike that is an artefact of the seeding. Tessellated cells are
    shaped like their Voronoi regions instead, clipped to ``cell_radius``, which is
    both closer to the answer and correct at any density.
    """

    # --- dynamics ---
    friction: float = 10.0
    """The friction ``gamma``, ``ET/L^2``. For passive runs this is only a time unit."""

    propulsion: str = "velocity"
    """How cells move: ``"velocity"`` or ``"force"``.

    ``"velocity"`` imposes ``v = v0 p`` directly. Simple, and wrong once activity is
    strong -- an imposed velocity is not a force, so nothing can resist it and cells
    drive straight through their neighbours.

    ``"force"`` balances an active force ``E_a/R`` against the mechanical force from
    the free energy, so a blocked cell stalls and its neighbour gets pushed. Uses
    ``active_energy`` and ``cell_friction`` instead of ``speed``.
    """

    speed: float = 0.0
    """Self-propulsion ``v0``, ``L/T``. Used by ``propulsion="velocity"`` only."""

    active_energy: float = 0.0
    """``E_a``, ``E``: the mechanical work a cell can do over about one cell radius.

    Used by ``propulsion="force"``. Unlike ``speed`` it is directly comparable with the
    passive free energy -- ``E_a/(sigma R)`` is the dimensionless activity.
    """

    cell_friction: float | None = None
    """``xi``, ``ET/L^2``: drag on a translating cell. ``None`` reuses ``friction``.

    Distinct from ``friction``, which resists the *field* changing shape. Same
    dimensions in 2D, different physics: a cell can be hard to drag but easy to deform.
    """

    rotational_diffusion: float = 1e-4
    """``D_r``, ``1/T``. The persistence time is ``1/D_r``."""

    time_unit: str = "s"
    """Name of the time unit ``gamma``, ``v0`` and ``D_r`` are quoted in.

    A label only -- it changes nothing, but it makes the reported timescales legible.
    """

    # --- numerics ---
    grid_spacing: float = 1.0
    """``dx`` in microns. Pure resolution: it appears in no physical quantity.

    Smaller resolves the interface better and costs ``dx^-2`` in memory. See
    :meth:`refine`.
    """

    timestep: float | None = None
    """``dt``. ``None`` derives it from the stability limit, which is the usual choice.

    Set it to override -- useful for matching a published run, or for deliberately
    stepping close to the limit. It is still checked: a value above what the physics
    tolerates raises :class:`~migration_theory.dynamics.UnstableTimestep` naming the
    term responsible, rather than quietly producing nonsense.
    """

    safety: float = 0.4
    """Fraction of the stability limit to take when ``timestep`` is not given."""

    def __post_init__(self) -> None:
        if self.n_cells < 3:
            raise ValueError(f"need at least 3 cells, got {self.n_cells}")
        if self.cell_radius <= 0:
            raise ValueError(f"cell_radius must be positive, got {self.cell_radius}")
        if self.packing <= 0:
            raise ValueError(f"packing must be positive, got {self.packing}")
        if self.grid_spacing <= 0:
            raise ValueError(f"grid_spacing must be positive, got {self.grid_spacing}")
        if not 0 < self.safety <= 1:
            raise ValueError(f"safety must lie in (0, 1], got {self.safety}")
        if self.seeding not in ("tessellated", "circles"):
            raise ValueError(
                f'seeding must be "tessellated" or "circles", got {self.seeding!r}'
            )
        if self.propulsion not in ("velocity", "force"):
            raise ValueError(
                f'propulsion must be "velocity" or "force", got {self.propulsion!r}'
            )
        if self.cell_friction is not None and self.cell_friction <= 0:
            raise ValueError(f"cell_friction must be positive, got {self.cell_friction}")
        if self.adhesion < 0:
            raise ValueError(f"adhesion must be non-negative, got {self.adhesion}")
        if self.timestep is not None and self.timestep <= 0:
            raise ValueError(f"timestep must be positive, got {self.timestep}")

    # ------------------------------------------------------------------ derived

    @property
    def interface_width(self) -> float:
        """``w = sqrt(K/alpha)``, in microns.

        Not an input: ``alpha`` and ``K`` decide it between them. Independent of the
        grid -- see :attr:`points_per_interface` for how well it is resolved.
        """
        return float(np.sqrt(self.K / self.alpha))

    @property
    def surface_tension(self) -> float:
        """``sqrt(2 K alpha)/6``."""
        return float(np.sqrt(2.0 * self.K * self.alpha) / 6.0)

    @property
    def cell_cell_tension(self) -> float:
        """``sigma (2 - omega/K)``: roughly what a shared interface costs.

        Where two cells abut, *both* have an interface there -- one field runs 1 to 0
        while the other runs 0 to 1 -- so a cell-cell boundary costs ``2 sigma`` where a
        free surface against the medium costs ``sigma``. Adhesion refunds part of one.

        Read it as indicative, for two reasons. The expression reaches zero only at
        ``omega = 2K``, but the stability ceiling is :attr:`adhesion_ceiling` at
        ``omega = K``, so in the usable range this only ever falls from ``2 sigma`` to
        about ``sigma`` -- it is not a measure of how close the tissue is to unjamming.
        And it was derived for an adhesion term in ``grad(phi).grad(phi)``, whereas
        :class:`~migration_theory.free_energy.Adhesion` uses ``grad(phi^2).grad(phi^2)``,
        whose ``4 phi_i phi_j`` weight varies across the interface rather than being
        constant. The measured observables -- shape index, T1 rate -- are the reliable
        guide to fluidity.
        """
        return self.surface_tension * (2.0 - self.adhesion / self.K)

    @property
    def adhesion_ceiling(self) -> float:
        """``K``: an optimistic bound on the adhesion the energy stays bounded below at.

        The binding limit is the field-level one, not the point where the
        sharp-interface tension ``sigma_cc`` vanishes at ``2K``: the gradient Hessian
        loses positive-definiteness on its out-of-phase subspace once adhesion matches
        the gradient term, after which neighbouring fields vary against each other for
        free and interface proliferates.

        For the ``grad(phi^2)`` form the coupling carries a weight ``4 phi_i phi_j``, so
        the true ceiling is ``K`` divided by that weight -- about 1 at a clean interface
        but larger wherever cells overlap, which lowers the ceiling. This property
        returns the clean-interface value; use
        :meth:`~migration_theory.free_energy.Adhesion.coupling_strength` on an actual
        configuration for the realised one.
        """
        return self.K

    @property
    def target_area(self) -> float:
        """``A_0 = pi R^2``, the area each cell is held towards."""
        return float(np.pi * self.cell_radius**2)

    @property
    def box_length(self) -> float:
        """Side of the box in microns, before rounding to whole grid points."""
        return float(np.sqrt(self.n_cells * self.target_area / self.packing))

    @property
    def box_points(self) -> int:
        """Side of the box in grid points.

        Rounded to a whole number so ``dx`` is exactly ``grid_spacing`` rather than
        nearly it. The cost is that the realised packing differs from the requested one
        by a fraction of a percent.
        """
        return max(8, int(round(self.box_length / self.grid_spacing)))

    @property
    def box(self) -> PeriodicBox:
        side = self.box_points * self.grid_spacing
        return PeriodicBox(side, side)

    @property
    def grid(self) -> Grid:
        """Square grid whose spacing is exactly :attr:`grid_spacing`."""
        return Grid(self.box, (self.box_points, self.box_points))

    @property
    def points_per_radius(self) -> float:
        """``R/dx``: how many grid points resolve a cell radius."""
        return self.cell_radius / self.grid_spacing

    @property
    def points_per_interface(self) -> float:
        """``w/dx``: how many grid points resolve the interface."""
        return self.interface_width / self.grid_spacing

    @property
    def realised_packing(self) -> float:
        """Packing after rounding the box to whole grid points."""
        return self.n_cells * self.target_area / self.box.area

    @property
    def cell_spacing(self) -> float:
        """Mean centre-to-centre distance, in microns."""
        return float(np.sqrt(self.box.area / self.n_cells / _HEX))

    @property
    def width_to_radius(self) -> float:
        """How much of a cell is boundary. Above ~0.5 there is no clear inside left."""
        return self.interface_width / self.cell_radius

    @property
    def effective_cell_friction(self) -> float:
        """``xi``, falling back to ``friction`` when not set separately."""
        return self.friction if self.cell_friction is None else self.cell_friction

    @property
    def free_speed(self) -> float:
        """Speed of a cell with nothing in its way.

        ``v0`` under ``propulsion="velocity"``; ``E_a/(R xi)`` under ``"force"``. The
        quantity to compare against ``sigma/gamma`` when judging whether activity can
        overcome cohesion.
        """
        if self.propulsion == "velocity":
            return self.speed
        return self.active_energy / (self.cell_radius * self.effective_cell_friction)

    @property
    def activity(self) -> float:
        """``E_a/(sigma R)``: active work per cell radius against cohesion.

        The dimensionless activity. Around 1 is where propulsion becomes comparable
        with what holds the tissue together.
        """
        return self.active_energy / (self.surface_tension * self.cell_radius)

    @property
    def persistence_time(self) -> float:
        """``1/D_r``: how long a cell holds its direction."""
        if self.rotational_diffusion == 0:
            return float("inf")
        return 1.0 / self.rotational_diffusion

    @property
    def persistence_length(self) -> float:
        """How far a free cell travels before forgetting its direction."""
        return self.free_speed * self.persistence_time

    @property
    def shape_relaxation_time(self) -> float:
        """``gamma R^2 / K``: how long a deformed cell takes to relax its shape.

        The slowest mechanical timescale, and the one a run has to be long compared to
        before anything it shows is a steady state rather than a transient.
        """
        return self.friction * self.cell_radius**2 / self.K

    @property
    def interface_relaxation_time(self) -> float:
        """``gamma w^2 / K``: how long the interface profile takes to equilibrate.

        The fastest timescale, and the one that sets the timestep. ``dt`` must be a
        small fraction of it.
        """
        return self.friction * self.interface_width**2 / self.K

    @property
    def traversal_time(self) -> float:
        """How long a free cell takes to crawl its own radius."""
        if self.free_speed == 0:
            return float("inf")
        return self.cell_radius / self.free_speed

    # ------------------------------------------------------------------ builders

    def free_energy(self) -> FreeEnergy:
        return FreeEnergy(
            DoubleWell(self.alpha),
            GradientEnergy(self.K),
            Repulsion(self.epsilon),
            Adhesion(self.adhesion),
            AreaConstraint(self.target_area, lambda_=self.area_lambda),
        )

    def tissue(self, seed: int = 0) -> Tissue:
        """A fresh tissue: evenly spaced centres, seeded at the target radius."""
        box, grid = self.box, self.grid
        centres = evenly_spaced(self.n_cells, box, rng=np.random.default_rng(seed))
        lay_down = seed_tessellated if self.seeding == "tessellated" else seed_circles
        fields = lay_down(grid, centres, self.cell_radius, self.interface_width)
        return Tissue(
            fields,
            Polarity.random(self.n_cells, self.speed, self.rotational_diffusion,
                            np.random.default_rng(seed + 1)),
        )

    def propulsion_rule(self):
        """The velocity rule implied by :attr:`propulsion`. ``None`` if passive."""
        if self.propulsion == "velocity":
            return ImposedVelocity() if self.speed else None
        if self.active_energy == 0:
            return None
        return ForceBalance(
            active_energy=self.active_energy,
            cell_radius=self.cell_radius,
            cell_friction=self.effective_cell_friction,
        )

    def max_stable_dt(self, tissue: Tissue, free_energy: FreeEnergy) -> float:
        """The largest timestep the physics tolerates, over all free-energy terms."""
        probe = ExplicitEuler(dt=1e-12, friction=self.friction)
        return probe.max_stable_dt(tissue, free_energy)

    def stepper(self, tissue: Tissue, free_energy: FreeEnergy) -> ExplicitEuler:
        """The timestepper for this model.

        With ``timestep`` unset, ``dt`` is the stability limit times ``safety`` --
        computed from the free energy rather than guessed, so changing a coefficient
        retunes it automatically instead of silently going unstable. With ``timestep``
        set, that value is used and the safety margin is the caller's business, so the
        check only refuses a step that is genuinely unstable.
        """
        if self.timestep is not None:
            return ExplicitEuler(dt=self.timestep, friction=self.friction, safety=1.0,
                                 propulsion=self.propulsion_rule())
        return ExplicitEuler(
            dt=self.safety * self.max_stable_dt(tissue, free_energy),
            friction=self.friction,
            propulsion=self.propulsion_rule(),
        )

    def replace(self, **changes) -> Model:
        """A copy with some parameters changed. The natural unit of a sweep."""
        return replace(self, **changes)

    def refine(self, factor: float) -> Model:
        """The same physics on a grid ``factor`` times finer.

        Just ``grid_spacing / factor``. Because lengths are physical and ``dx`` is not
        one of them, refining touches no coefficient at all -- which is what makes a
        convergence check meaningful: refine, rerun, and any change in the result is
        discretisation error rather than a different model.

        The grid grows as ``factor^2``, and the gradient term's stability limit
        ``gamma dx^2 / (4K)`` shrinks as ``factor^2`` too, so once that term binds the
        total cost rises as ``factor^4``.
        """
        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}")
        return self.replace(grid_spacing=self.grid_spacing / factor)

    # ------------------------------------------------------------------ reporting

    def summary(self) -> dict[str, float]:
        """Every derived number, for printing next to a result."""
        return {
            "interface_width": self.interface_width,
            "surface_tension": self.surface_tension,
            "cell_cell_tension": self.cell_cell_tension,
            "cell_radius": self.cell_radius,
            "width_to_radius": self.width_to_radius,
            "target_area": self.target_area,
            "packing": self.realised_packing,
            "cell_spacing": self.cell_spacing,
            "grid_spacing": self.grid_spacing,
            "points_per_radius": self.points_per_radius,
            "points_per_interface": self.points_per_interface,
            "box_length": self.box_length,
            "box_points": self.box_points,
            "persistence_length": self.persistence_length,
            "persistence_time": self.persistence_time,
            "shape_relaxation_time": self.shape_relaxation_time,
            "interface_relaxation_time": self.interface_relaxation_time,
            "traversal_time": self.traversal_time,
        }

    def concerns(self) -> list[str]:
        """Things about this parameter choice worth knowing before running it.

        Returned rather than warned, so a caller decides whether to print, raise or
        ignore -- none of these is wrong, they are just easy to walk into.
        """
        notes = []
        if self.adhesion >= self.adhesion_ceiling:
            notes.append(
                f"adhesion {self.adhesion:g} is at or above K = {self.adhesion_ceiling:g}; "
                "the gradient Hessian is no longer positive definite, so the energy is "
                "unbounded below and the run will diverge rather than relax."
            )
        elif self.adhesion > 0.7 * self.adhesion_ceiling:
            notes.append(
                f"adhesion {self.adhesion:g} is {self.adhesion / self.adhesion_ceiling:.0%} "
                f"of the clean-interface ceiling K = {self.adhesion_ceiling:g}. The "
                "realised ceiling is lower wherever cells overlap -- check "
                "Adhesion.coupling_strength on the configuration."
            )
        if self.points_per_interface < 1.5:
            notes.append(
                f"interface is {self.points_per_interface:.2f} grid points wide; below "
                "about 1.5 it is barely resolved and will pin to the grid. Reduce "
                "grid_spacing, or raise K / lower alpha to widen it physically."
            )
        if self.width_to_radius > 0.5:
            notes.append(
                f"interface is {self.width_to_radius:.2f} of the cell radius; cells will "
                "have no clear inside. Lower K, raise alpha, or raise R."
            )
        if self.box_points > 400:
            notes.append(
                f"box is {self.box_points}^2 points; cost scales as the square, so this "
                "will be slow."
            )
        if self.realised_packing > 1.4:
            notes.append(
                f"packing {self.realised_packing:.2f} is well above confluence; cells "
                "start heavily compressed and the first steps are stiff."
            )
        if self.realised_packing < 0.5:
            notes.append(
                f"packing {self.realised_packing:.2f}; cells will not touch, so repulsion "
                "and contact play no part."
            )
        return notes

    def __str__(self) -> str:
        s = self.summary()
        u = self.time_unit
        activity = (
            f"  {self.propulsion}: "
            + (f"v0 {self.speed:g}" if self.propulsion == "velocity"
               else f"E_a {self.active_energy:g}  xi {self.effective_cell_friction:g}"
                    f"  activity E_a/(sigma R) {self.activity:.3f}")
            + f"  -> free speed {self.free_speed:g} um/{u}"
            + f"  Dr {self.rotational_diffusion:g} /{u}\n"
            f"  persistence {s['persistence_time']:.0f} {u} / "
            f"{s['persistence_length']:.1f} um = {s['persistence_length'] / self.cell_radius:.1f} R"
            f"   traversal (R/v0) {s['traversal_time']:.0f} {u}\n"
            if self.free_speed
            else ""
        )
        times = (
            f"  timescales: interface {s['interface_relaxation_time']:.3g} {u}, "
            f"shape {s['shape_relaxation_time']:.3g} {u}"
            + (f", persistence {s['persistence_time']:.3g} {u}" if self.free_speed else "")
            + "\n"
        )
        return (
            f"Model(alpha={self.alpha:g}, K={self.K:g}, epsilon={self.epsilon:g}, "
            f"omega={self.adhesion:g}, lambda={self.area_lambda:g}, "
            f"gamma={self.friction:g}, "
            f"R={self.cell_radius:g} um, {self.n_cells} cells)\n"
            f"  width {s['interface_width']:.3f} um  width/R {s['width_to_radius']:.3f}  "
            f"sigma {s['surface_tension']:.4f}  sigma_cc {s['cell_cell_tension']:.4f}"
            f"  A0 {s['target_area']:.1f} um^2\n"
            f"  box {s['box_length']:.1f} um = {int(s['box_points'])}^2 points at "
            f"dx {s['grid_spacing']:g} um\n"
            f"  resolution: {s['points_per_radius']:.1f} points per radius, "
            f"{s['points_per_interface']:.1f} across the interface\n"
            f"  spacing {s['cell_spacing']:.2f} um  packing {s['packing']:.3f}\n"
            + times + activity
        )
