r"""The tissue free energy, assembled from independent terms.

Each term knows two things: its total energy, and its functional derivative
``delta F / delta phi_i``. The dynamics only ever ask for the derivative, so adding a
new piece of physics means adding a term here and nothing else.

Most terms are *local* -- their energy is the integral of a density over the grid --
and those also provide ``density()`` so the energy can be mapped. Some are not:
:class:`AreaConstraint` depends on each cell's area, a number obtained by integrating
over the whole domain, so there is no density to map and it provides ``energy()``
alone.

:class:`FreeEnergy` sums terms. Because the derivative of a sum is the sum of the
derivatives, terms compose with no bookkeeping -- they stay independent, and each can
be checked on its own against the finite difference of its own energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .diagnostics import areas
from .fields import PhaseFields, Windows
from .grid import magnitude, sum_of_squares


def ball_volume(radius: float, dimension: int = 2) -> float:
    """``pi R^2`` in 2D, ``4/3 pi R^3`` in 3D: the volume a round cell of radius ``R`` wants."""
    if dimension == 2:
        return np.pi * radius**2
    if dimension == 3:
        return 4.0 / 3.0 * np.pi * radius**3
    raise NotImplementedError(f"ball volume in {dimension}D")


def _stiffness(grid) -> float:
    """``sum_k 4 / h_k**2``: the most negative eigenvalue of the Laplacian, in magnitude."""
    return sum(4.0 / h**2 for h in grid.spacings)

# Every term's ``functional_derivative`` takes an optional ``windows``. Without it the
# derivative is the dense ``(n_cells, ny, nx)`` stack, the reference. With it, the
# derivative is computed on each cell's own patch and returned as ``(n_cells, h, w)``,
# the same numbers where the cell is and nothing spent where it is not. The two terms
# that couple cells need one grid-sized quantity each, the sum of squared fields and
# its Laplacian; :class:`FreeEnergy` builds those once per step and passes them down
# as ``shared``, so no term computes them twice. ``shared`` also carries the patches
# themselves, gathered from the dense stack once per step: measured, gathering them
# afresh in every term was most of the windowed step's cost.


def _patches(fields: PhaseFields, windows: Windows, shared: dict | None) -> np.ndarray:
    """The patch stack: the one already gathered this step if there is one."""
    if shared is not None and "patches" in shared:
        return shared["patches"]
    return windows.extract(fields.values)

__all__ = [
    "FreeEnergyTerm",
    "FreeEnergy",
    "DoubleWell",
    "GradientEnergy",
    "Repulsion",
    "Adhesion",
    "AreaConstraint",
    "ball_volume",
    "interface_terms",
    "interface_width",
    "surface_tension",
]


@runtime_checkable
class FreeEnergyTerm(Protocol):
    """One additive contribution to the free energy."""

    def energy(self, fields: PhaseFields) -> float:
        """The term's total contribution, a single number."""

    def functional_derivative(
        self, fields: PhaseFields, windows: Windows | None = None, shared: dict | None = None
    ) -> np.ndarray:
        """``delta F / delta phi_i`` for every cell: ``(n_cells, ny, nx)``, or
        ``(n_cells, h, w)`` patches when ``windows`` is given."""

    # Terms may additionally provide:
    #     density(fields) -> (ny, nx)
    #         energy per unit area, summed over cells, such that
    #         grid.integrate(density(fields)) == energy(fields). Local terms only.
    #     stability_limit(fields, mobility) -> float
    #         the largest explicit-Euler timestep this term alone tolerates. The
    #         stepper takes the smallest over all terms, so a new term brings its own
    #         constraint and nothing has to be told about it.


@dataclass(frozen=True)
class DoubleWell:
    r"""A bulk double well, :math:`\alpha \sum_i \int \phi_i^2 (\phi_i - 1)^2 \, dx`.

    .. math::

        G(\phi) = \alpha\,\phi^2 (\phi - 1)^2,
        \qquad
        \frac{\mathrm{d}G}{\mathrm{d}\phi} = 2\alpha\,\phi\,(\phi - 1)(2\phi - 1)

    Degenerate minima at :math:`\phi = 0` and :math:`\phi = 1` -- outside and inside a
    cell -- separated by a barrier of height :math:`\alpha/16` at :math:`\phi = 1/2`.
    This is the term that gives a cell an inside and an outside at all: it penalises
    every intermediate value, pushing the field towards one phase or the other
    everywhere, and the barrier is what stops the two mixing freely.

    Note what it does *not* do. The term is purely local -- the derivative at a point
    depends only on the field at that point -- so it carries **no length scale**. Under
    gradient dynamics every grid point independently rolls into whichever minimum it is
    nearer, and an interface sharpens until it is one grid cell wide, where it pins and
    the answer starts depending on ``dx``. A gradient term is what penalises that
    collapse and holds the interface at a finite width.

    Once a gradient term :math:`\tfrac{K}{2}|\nabla\phi|^2` is added, the two together
    fix the interface profile and width analytically: minimising them gives exactly the
    ``tanh`` that :func:`~migration_theory.fields.seed` lays down, with

    .. math:: w = \sqrt{K/\alpha}, \qquad \sigma = \sqrt{2 K \alpha}\,/\,6

    for the interface width ``w`` and surface tension. That is the point at which the
    width stops being a seeding choice and becomes a consequence of the free energy.
    """

    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

    @property
    def minima(self) -> tuple[float, float]:
        """The two field values the well drives towards: outside and inside."""
        return (0.0, 1.0)

    @property
    def barrier(self) -> float:
        """Energy density at the top of the barrier, ``G(1/2)``."""
        return self.alpha / 16.0

    def potential(self, phi: np.ndarray) -> np.ndarray:
        """``G(phi)``, for plotting the well itself."""
        phi = np.asarray(phi, dtype=float)
        return self.alpha * phi**2 * (phi - 1.0) ** 2

    def gradient(self, phi: np.ndarray) -> np.ndarray:
        """``dG/dphi``, in factored form so its three roots are exact."""
        phi = np.asarray(phi, dtype=float)
        return 2.0 * self.alpha * phi * (phi - 1.0) * (2.0 * phi - 1.0)

    def density(self, fields: PhaseFields) -> np.ndarray:
        return self.potential(fields.values).sum(axis=0)

    def energy(self, fields: PhaseFields) -> float:
        return float(fields.grid.integrate(self.density(fields)))

    def functional_derivative(
        self, fields: PhaseFields, windows: Windows | None = None, shared: dict | None = None
    ) -> np.ndarray:
        """Just ``dG/dphi``: the well contains no derivatives of the field, so its
        variational derivative is the ordinary one -- pointwise, so on a window it is
        the same function of the patch."""
        if windows is None:
            return self.gradient(fields.values)
        return self.gradient(_patches(fields, windows, shared))

    def stability_limit(self, fields: PhaseFields, friction: float) -> float:
        """``gamma / alpha``.

        Linearising ``gamma dphi/dt = -G'(phi)`` gives a decay rate ``G''/gamma``, and
        forward Euler needs ``dt`` times that below 2. On ``[0, 1]`` the curvature peaks
        at ``|G''| = 2 alpha``, reached at both minima.
        """
        return friction / self.alpha


@dataclass(frozen=True)
class GradientEnergy:
    r"""The cost of varying the field, :math:`\tfrac{K}{2} \sum_i \int |\nabla\phi_i|^2 dx`.

    .. math::

        F = \frac{K}{2} \sum_i \int |\nabla\phi_i|^2 \, \mathrm{d}x,
        \qquad
        \frac{\delta F}{\delta \phi_i} = -K \nabla^2 \phi_i

    The partner to :class:`DoubleWell`, and the term that supplies the length scale the
    well lacks. The well wants every point at 0 or 1 and would happily make the
    transition infinitely abrupt; this term charges for abruptness. The compromise
    between them is an interface of finite width, and it is that competition -- not
    either term alone -- that makes a phase field.

    Minimising the pair in one dimension gives exactly the profile
    :func:`~migration_theory.fields.seed` lays down,
    :math:`\phi = \tfrac{1}{2}[1 - \tanh(x / \sqrt{2}w)]`, with

    .. math:: w = \sqrt{K/\alpha}, \qquad \sigma = \sqrt{2 K \alpha} \,/\, 6

    See :func:`interface_terms` to go the other way and choose the width and
    ``sigma`` directly.

    **On the discretisation.** The energy uses forward differences while the derivative
    uses the five-point Laplacian. That pairing is deliberate: those two operators are
    exact discrete adjoints, so the functional derivative really is the derivative of
    the discrete energy, to machine precision. Using central differences in the energy
    would leave checkerboard modes costing nothing and let grid-scale noise grow.
    """

    K: float = 1.0
    """The gradient energy coefficient."""

    def __post_init__(self) -> None:
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")

    def density(self, fields: PhaseFields) -> np.ndarray:
        components = fields.grid.forward_gradient(fields.values)
        return 0.5 * self.K * sum_of_squares(components).sum(axis=0)

    def energy(self, fields: PhaseFields) -> float:
        return float(fields.grid.integrate(self.density(fields)))

    def functional_derivative(
        self, fields: PhaseFields, windows: Windows | None = None, shared: dict | None = None
    ) -> np.ndarray:
        if windows is None:
            return -self.K * fields.grid.laplacian(fields.values)
        return -self.K * windows.laplacian(_patches(fields, windows, shared))

    def stability_limit(self, fields: PhaseFields, friction: float) -> float:
        """``gamma dx**2 / (2 ndim K)`` on a cubic grid -- usually the binding constraint.

        This term makes the evolution a diffusion equation with diffusivity ``K/gamma``.
        The compact Laplacian's most negative eigenvalue is ``-sum_k 4/h_k^2``, and
        forward Euler needs ``dt`` times that magnitude below 2.
        """
        rate = self.K * _stiffness(fields.grid) / friction
        return 2.0 / rate


@dataclass(frozen=True)
class Repulsion:
    r"""Overlap repulsion: ``epsilon * sum over pairs i<j of INT phi_i^2 phi_j^2 dx``.

    ``phi_i^2 phi_j^2`` is non-zero only where two cells are both present, so this
    charges ``epsilon`` per unit of shared area and keeps cells from interpenetrating.

    The energy depends on every field, so each cell has its own derivative. The pairs
    involving cell ``k`` sum to ``phi_k^2 * sum_{j != k} phi_j^2``, giving

        dF/dphi_k = 2 * epsilon * phi_k * sum_{j != k} phi_j^2

    A single pair ``(i, j)`` thus enters both equations of motion -- as
    ``2 eps phi_i phi_j^2`` and as ``2 eps phi_j phi_i^2`` -- which is what makes the
    two cells separate rather than one simply fleeing the other.

    Each unordered pair is counted once. Summing over ordered pairs would just double
    ``epsilon``.

    **Computed in linear time, not as the double sum.** Written literally, the sum over
    pairs is ``N(N-1)/2`` grid operations -- 120 at 16 cells, 4950 at 100 -- and it
    would dominate everything else once the tissue is large. It is not necessary. At
    each grid point write ``a_i = phi_i^2``; then squaring the plain sum gives every
    product twice plus the squares once,

        (a_1 + a_2 + ... + a_N)^2 = sum_i a_i^2 + 2 sum_{i<j} a_i a_j

    so the pair sum is recovered from two sums that each cost one pass over the cells:

        sum_{i<j} a_i a_j = [ (sum_i a_i)^2 - sum_i a_i^2 ] / 2

    With three cells and ``a = (1, 2, 3)``: the pairs give ``1*2 + 1*3 + 2*3 = 11``, and
    ``(6^2 - 14) / 2 = 11``. The derivative for cell ``k`` needs ``sum_{j != k} phi_j^2``,
    which is the same total with cell ``k``'s own term taken back out,
    ``S - phi_k^2`` with ``S = sum_j phi_j^2``, again computed once for all cells.

    Both identities are exact algebra, not approximations; the tests hold this
    version against the literal pair loop to machine precision.
    """

    epsilon: float = 1.0

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")

    def density(self, fields: PhaseFields) -> np.ndarray:
        squared = fields.values**2                        # a_i = phi_i^2, per cell
        total = squared.sum(axis=0)                       # sum_i a_i, one pass over cells
        return 0.5 * self.epsilon * (total**2 - (squared**2).sum(axis=0))

    def energy(self, fields: PhaseFields) -> float:
        return float(fields.grid.integrate(self.density(fields)))

    def functional_derivative(
        self, fields: PhaseFields, windows: Windows | None = None, shared: dict | None = None
    ) -> np.ndarray:
        if windows is None:
            squared = fields.values**2
            others = squared.sum(axis=0) - squared        # sum_{j != k} phi_j^2 for every k
            return 2.0 * self.epsilon * fields.values * others
        # On windows the same identity: cell k needs S = sum_j phi_j^2 only where k is.
        patch = _patches(fields, windows, shared)
        if shared is not None and "squared_sum" in shared:
            total = shared["squared_sum"]
        else:
            total = windows.accumulate(patch**2, np.zeros(fields.grid.shape))
        return 2.0 * self.epsilon * patch * (windows.extract_field(total) - patch**2)

    def stability_limit(self, fields: PhaseFields, friction: float) -> float:
        """``gamma / (eps max_k sum_{j!=k} phi_j^2)``.

        Holding the other cells fixed, cell ``k`` sees a local stiffness
        ``2 eps sum_{j!=k} phi_j^2``. Depends on the configuration, so it is measured
        rather than assumed -- it tightens where cells pile up.
        """
        if self.epsilon == 0:
            return np.inf
        squared = fields.values**2
        crowding = float((squared.sum(axis=0) - squared).max())
        return np.inf if crowding <= 0 else friction / (self.epsilon * crowding)


@dataclass(frozen=True)
class Adhesion:
    r"""Cell-cell adhesion: ``omega * sum over pairs i<j of INT grad(phi_i^2).grad(phi_j^2)``.

    Where two cells meet, one field falls as the other rises, so their gradients are
    antiparallel and the integrand is negative -- contact *lowers* the energy, for
    ``omega > 0``. Away from a contact at least one gradient vanishes and the term is
    zero.

    Note it is the gradient of ``phi^2``, not of ``phi``. Since
    ``grad(phi_i^2).grad(phi_j^2) = 4 phi_i phi_j (grad phi_i . grad phi_j)``, that
    carries an extra ``4 phi_i phi_j`` weight which confines the term to where *both*
    cells are genuinely present. Adhesion on ``grad phi`` alone would also fire in
    regions where one cell has only a faint tail, which is not where a junction is.

    Writing ``psi_i = phi_i^2``, the derivative follows from the ``psi`` form by the
    chain rule:

        dF/dphi_k = -2 omega phi_k sum_{j != k} laplacian(phi_j^2)

    evaluated as the total Laplacian minus cell ``k``'s own, so the term is O(N) in the
    cell count rather than O(N^2).

    **Effective cell-cell tension.** Two cells in contact each pay their own interface
    and adhesion refunds part of it, giving roughly ``sigma (2 - omega/K)`` -- but only
    roughly here, because the ``4 phi_i phi_j`` weight varies across the interface
    rather than being constant.

    **Stability ceiling.** Collecting the gradient terms, the off-diagonal Hessian block
    is ``4 omega phi_k phi_j`` times the same operator the gradient term contributes
    ``K`` to, so the out-of-phase mode loses stability near

        omega ~ K / max(4 phi_i phi_j)

    At a clean interface both fields are about a half there, so the weight is about 1
    and the ceiling is near ``K``. Where cells overlap more the weight exceeds 1 and the
    ceiling drops, so it is configuration dependent -- :meth:`coupling_strength`
    measures it. Past the ceiling, neighbouring fields vary against each other for free,
    interface proliferates and the energy runs away.

    **On identifiability.** In a confluent, single-cell-type tissue with no free
    surface, the mechanics depend on ``omega`` and ``K`` largely through the combined
    tension, so the two are close to unidentifiable from mechanics alone. Breaking that
    needs a free surface, a second cell type, or an independent measurement. Adhesion
    and cortical tension remain physically distinct ingredients regardless.

    **Discretisation.** The energy uses forward differences on ``phi^2`` and the
    derivative the five-point Laplacian on ``phi^2``, which are exact discrete adjoints
    -- the same pairing, and for the same reason, as :class:`GradientEnergy`.
    """

    omega: float = 0.0

    def __post_init__(self) -> None:
        if self.omega < 0:
            raise ValueError(f"omega must be non-negative, got {self.omega}")

    # With omega = 0 the term is identically zero, so the gradients and Laplacians
    # below would be computed only to be multiplied by nothing. Skipping them saves a
    # noticeable slice of every step in the common no-adhesion case.

    def density(self, fields: PhaseFields) -> np.ndarray:
        if self.omega == 0:
            return np.zeros(fields.grid.shape)
        components = fields.grid.forward_gradient(fields.values**2)
        total = sum_of_squares(c.sum(axis=0) for c in components)
        own = sum_of_squares(components).sum(axis=0)
        return 0.5 * self.omega * (total - own)

    def energy(self, fields: PhaseFields) -> float:
        if self.omega == 0:
            return 0.0
        return float(fields.grid.integrate(self.density(fields)))

    def functional_derivative(
        self, fields: PhaseFields, windows: Windows | None = None, shared: dict | None = None
    ) -> np.ndarray:
        if windows is None:
            if self.omega == 0:
                return np.zeros_like(fields.values)
            laplacian = fields.grid.laplacian(fields.values**2)
            return -2.0 * self.omega * fields.values * (laplacian.sum(axis=0) - laplacian)
        if self.omega == 0:
            return np.zeros((fields.n_cells, *windows.shape))
        # sum_j lap(phi_j^2) = lap(sum_j phi_j^2) by linearity: one Laplacian of the
        # shared sum, read through the windows, minus each cell's own on its patch.
        patch = _patches(fields, windows, shared)
        if shared is not None and "squared_sum_laplacian" in shared:
            total = shared["squared_sum_laplacian"]
        else:
            squared_sum = windows.accumulate(patch**2, np.zeros(fields.grid.shape))
            total = fields.grid.laplacian(squared_sum)
        own = windows.laplacian(patch**2)
        return -2.0 * self.omega * patch * (windows.extract_field(total) - own)

    def coupling_strength(self, fields: PhaseFields) -> float:
        """``max(4 phi_i phi_j)`` over the grid and over pairs.

        The weight the ``phi^2`` form puts on the coupling, relative to the plain
        ``grad phi . grad phi`` form. About 1 at a clean interface where both fields sit
        near a half; larger wherever cells overlap. It scales both the stability limit
        and the ceiling on ``omega``.
        """
        if fields.n_cells < 2:
            return 0.0
        largest_two = np.partition(fields.values, -2, axis=0)[-2:]
        return float(np.max(4.0 * largest_two[0] * largest_two[1]))

    def contact_multiplicity(self, fields: PhaseFields) -> float:
        """Effective number of cells sharing a point, weighted by gradient.

        The participation ratio ``(sum_i |grad psi_i|)^2 / sum_i |grad psi_i|^2``,
        maximised over the grid. Two cells meeting cleanly gives 2, a three-cell vertex
        gives 3. It sets how strongly the off-diagonal coupling adds up.
        """
        size = magnitude(fields.grid.forward_gradient(fields.values**2))
        total = size.sum(axis=0) ** 2
        squares = (size**2).sum(axis=0)
        return float(np.max(total / np.maximum(squares, 1e-30)))

    def stability_limit(self, fields: PhaseFields, friction: float) -> float:
        """``2 gamma / ((z-1) w omega (4/dx^2 + 4/dy^2))``.

        ``z`` is the contact multiplicity and ``w`` the coupling strength, both measured
        from the configuration rather than assumed, because this term's strength depends
        on how much the cells actually overlap.

        Two caveats, as for the plain gradient form. The limit is for this term alone,
        while in practice it adds to :class:`GradientEnergy` in the same operator, so
        the combined limit is tighter than the smaller of the two -- the safety factor
        absorbs the difference. And the real test is empirical: a passive run whose
        energy fails to decrease has too large a step, whatever this returns.
        """
        if self.omega == 0:
            return np.inf
        multiplicity = max(self.contact_multiplicity(fields) - 1.0, 1e-6)
        weight = max(self.coupling_strength(fields), 1e-6)
        rate = multiplicity * weight * self.omega * _stiffness(fields.grid)
        return 2.0 * friction / rate


@dataclass(frozen=True)
class AreaConstraint:
    r"""Holds each cell near a target area.

    .. math::

        F = \lambda \sum_i \left(1 - \frac{A_i}{A_0}\right)^2,
        \qquad
        A_i = \int \phi_i^2 \, \mathrm{d}^2 r,
        \qquad
        A_0 = \pi R^2

    **Why it is needed.** :class:`DoubleWell` and :class:`GradientEnergy` together
    produce motion by mean curvature: a closed interface shrinks at a rate set by its
    curvature, so an isolated cell contracts and eventually vanishes. Nothing in those
    two terms cares how much area a cell has. This term supplies that, and is what
    makes a cell a persistent object rather than a transient blob.

    **It is a soft constraint.** Area is held near ``A_0``, not at it: the equilibrium
    trades a small area error against whatever the other terms want, and the residual
    error falls as ``lambda`` rises. Exact conservation needs a Lagrange multiplier or
    conserved dynamics instead.

    **It is not local.** ``A_i`` is an integral over the whole domain, so

    .. math::

        \frac{\delta F}{\delta \phi_k}
            = -\frac{4\lambda}{A_0}\left(1 - \frac{A_k}{A_0}\right)\phi_k

    couples every point of cell ``k`` through one number. The sign works out as it
    should: a cell that is too small has :math:`A_k < A_0`, hence a negative
    derivative, and under :math:`\partial_t\phi = -M\,\delta F/\delta\phi` it grows.

    ``target_area`` may be a single value or one per cell, which is the hook for
    growth and division later.
    """

    target_area: float | np.ndarray = 1.0
    lambda_: float = 1.0
    """The constraint coefficient ``lambda``. Trailing underscore only because
    ``lambda`` is a Python keyword."""

    @classmethod
    def from_radius(cls, radius: float, lambda_: float = 1.0, dimension: int = 2) -> AreaConstraint:
        """Target the volume of a ball of the given radius: ``pi R^2`` in 2D, ``4/3 pi R^3`` in 3D."""
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        return cls(float(ball_volume(radius, dimension)), lambda_)

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.target_area, dtype=float) <= 0):
            raise ValueError(f"target_area must be positive, got {self.target_area}")
        if self.lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative, got {self.lambda_}")

    def mismatch(
        self, fields: PhaseFields, windows: Windows | None = None, patches: np.ndarray | None = None
    ) -> np.ndarray:
        """``(n_cells,)`` fractional area error ``1 - A_i/A_0``. Positive = too small.

        Worth watching directly: it says how well the soft constraint is holding, which
        no single energy number tells you.
        """
        target = np.asarray(self.target_area, dtype=float)
        return 1.0 - areas(fields, windows, patches) / target

    def energy(self, fields: PhaseFields) -> float:
        return float(self.lambda_ * np.sum(self.mismatch(fields) ** 2))

    def functional_derivative(
        self, fields: PhaseFields, windows: Windows | None = None, shared: dict | None = None
    ) -> np.ndarray:
        target = np.asarray(self.target_area, dtype=float)
        per_cell = (-1,) + (1,) * fields.grid.ndim        # broadcast one number per cell
        if windows is None:
            factor = -4.0 * self.lambda_ * self.mismatch(fields) / target
            return np.reshape(factor, per_cell) * fields.values
        patch = _patches(fields, windows, shared)
        factor = -4.0 * self.lambda_ * self.mismatch(fields, windows, patch) / target
        return np.reshape(factor, per_cell) * patch

    def stability_limit(self, fields: PhaseFields, friction: float) -> float:
        """``gamma A_0 / (4 lambda)``.

        The second variation has a local part of size ``4 lambda |1 - A/A_0| / A_0`` and
        a rank-one non-local part ``8 lambda A / A_0^2`` from the area's own dependence
        on the field. Near the target the second dominates at ``8 lambda / A_0``, and
        forward Euler needs ``dt`` times that below 2.
        """
        if self.lambda_ == 0:
            return np.inf
        smallest = float(np.min(np.asarray(self.target_area, dtype=float)))
        return friction * smallest / (4.0 * self.lambda_)


def interface_terms(
    interface_width: float, surface_tension: float
) -> tuple[DoubleWell, GradientEnergy]:
    r"""The well and gradient pair having a given interface width and surface tension.

    Inverts :math:`w = \sqrt{K/\alpha}` and :math:`\sigma = \sqrt{2K\alpha}/6`:

    .. math:: \alpha = 3\sqrt{2}\,\sigma / w, \qquad K = 3\sqrt{2}\,\sigma w

    Prefer this to setting ``alpha`` and ``K`` by hand. Those two are a basis,
    not a description -- neither one alone corresponds to anything you can observe,
    whereas the width and the tension are both measurable and control separate things:
    the width is a resolution cost, the tension is physics.
    """
    if interface_width <= 0:
        raise ValueError(f"interface_width must be positive, got {interface_width}")
    if surface_tension <= 0:
        raise ValueError(f"surface_tension must be positive, got {surface_tension}")
    scale = 3.0 * np.sqrt(2.0) * surface_tension
    return DoubleWell(scale / interface_width), GradientEnergy(scale * interface_width)


def interface_width(well: DoubleWell, gradient: GradientEnergy) -> float:
    """``sqrt(K/alpha)`` -- the width implied by a well and gradient pair."""
    return float(np.sqrt(gradient.K / well.alpha))


def surface_tension(well: DoubleWell, gradient: GradientEnergy) -> float:
    """``sqrt(2 K alpha)/6`` -- the tension implied by a well and gradient pair."""
    return float(np.sqrt(2.0 * gradient.K * well.alpha) / 6.0)


@dataclass(frozen=True)
class FreeEnergy:
    """A sum of :class:`FreeEnergyTerm` s, which is itself a term."""

    terms: tuple[FreeEnergyTerm, ...]

    def __init__(self, *terms: FreeEnergyTerm) -> None:
        object.__setattr__(self, "terms", tuple(terms))

    def __iter__(self):
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def energy(self, fields: PhaseFields) -> float:
        """The total free energy, a single number."""
        return float(sum(term.energy(fields) for term in self.terms))

    def functional_derivative(
        self,
        fields: PhaseFields,
        windows: Windows | None = None,
        patches: np.ndarray | None = None,
    ) -> np.ndarray:
        """The summed derivative: dense, or on windows when ``windows`` is given.

        On windows the quantities every term shares are built here once and handed
        down as ``shared``: the patches themselves, gathered from the dense stack
        (or taken from ``patches`` if the caller already has them), the sum of squared
        fields accumulated from those patches, and its Laplacian if any term has
        adhesion. The last is the only grid-sized work left in a windowed step.
        """
        if windows is None:
            total = np.zeros_like(fields.values)
            for term in self.terms:
                total += term.functional_derivative(fields)
            return total
        shared = self.shared_quantities(fields, windows, patches)
        total = np.zeros((fields.n_cells, *windows.shape))
        for term in self.terms:
            total += term.functional_derivative(fields, windows, shared)
        return total

    def shared_quantities(
        self,
        fields: PhaseFields,
        windows: Windows | None = None,
        patches: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """What every term reads on a windowed step, built once.

        ``patches`` is the ``(n_cells, h, w)`` stack of the cells' windows -- gathered
        here if the caller has not already -- and ``squared_sum`` is ``sum_j phi_j^2``
        accumulated from it onto the grid, so no term touches the dense stack. The
        Laplacian adhesion needs is of that one shared field and stays grid-sized,
        which is cheap. Dense: only the sum, from the stack.
        """
        if windows is None:
            shared = {"squared_sum": (fields.values**2).sum(axis=0)}
        else:
            if patches is None:
                patches = windows.extract(fields.values)
            shared = {
                "patches": patches,
                "squared_sum": windows.accumulate(patches**2, np.zeros(fields.grid.shape)),
            }
        if any(isinstance(term, Adhesion) and term.omega > 0 for term in self.terms):
            shared["squared_sum_laplacian"] = fields.grid.laplacian(shared["squared_sum"])
        return shared

    def density(self, fields: PhaseFields) -> np.ndarray:
        """Energy density of the local terms only.

        Non-local terms are skipped rather than smeared out: :class:`AreaConstraint`
        has no density to show, and inventing a uniform one would put energy in places
        it does not come from. Use :meth:`breakdown` for the full accounting.
        """
        total = np.zeros(fields.grid.shape)
        for term in self.terms:
            if hasattr(term, "density"):
                total += term.density(fields)
        return total

    def breakdown(self, fields: PhaseFields) -> dict[str, float]:
        """Each term's contribution separately, for seeing which one dominates."""
        return {type(term).__name__: term.energy(fields) for term in self.terms}

    def interface_properties(self) -> dict[str, float] | None:
        """The width and surface tension implied by ``alpha`` and ``K``.

        Neither is an input when the terms are built from raw coefficients, but both
        are still real: the width sets how finely the grid must resolve the interface,
        and the tension is what the model actually does. Reported here so that tuning
        ``K`` directly does not mean losing sight of them. ``None`` if the free energy
        has no well and gradient pair.
        """
        well = next((t for t in self.terms if isinstance(t, DoubleWell)), None)
        gradient = next((t for t in self.terms if isinstance(t, GradientEnergy)), None)
        if well is None or gradient is None:
            return None
        return {
            "interface_width": interface_width(well, gradient),
            "surface_tension": surface_tension(well, gradient),
        }

    def points_per_interface(self, grid) -> float:
        """How many grid points span the interface. Infinite if there is no interface."""
        properties = self.interface_properties()
        if properties is None:
            return np.inf
        return properties["interface_width"] / max(grid.spacings)
