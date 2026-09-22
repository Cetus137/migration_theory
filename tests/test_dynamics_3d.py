"""The physics in three dimensions: free energy, forces, propulsion, stepping, windows.

Step two of making the package dimension-free. Every identity the 2D suite checks is
checked again here on a 3D tissue built by hand -- ``Model`` is still two-dimensional
at this stage -- plus the places a third axis changes the answer: the sphere's surface
and volume, the Laplacian's stiffness, polarity diffusing on a sphere rather than a
circle, and the shape index's exponent.

Everything is small. A 24^3 grid with four cells is 55 thousand points per cell,
about the size of one 2D window, so nothing here should take more than a few seconds.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from conftest import finite_difference_derivative, sample_significant
from migration_theory import (
    Adhesion,
    AreaConstraint,
    DoubleWell,
    ExplicitEuler,
    ForceBalance,
    FreeEnergy,
    GradientEnergy,
    Grid,
    ImposedVelocity,
    PeriodicBox,
    PhaseFields,
    Polarity,
    Repulsion,
    Tissue,
    advection,
    areas,
    ball_volume,
    centres_of_mass,
    interface_terms,
    passive_forces,
    perimeters,
    run,
    seed,
    shape_index,
    shape_indices,
)

THRESHOLD = 1e-6
RADIUS, WIDTH = 5.0, 1.0


@pytest.fixture
def grid3():
    """A cube on a unit grid, unequal in points per axis so an axis mix-up cannot pass."""
    return Grid(PeriodicBox(24.0, 26.0, 22.0), (22, 26, 24))       # (nz, ny, nx)


@pytest.fixture
def tissue3(grid3):
    """Four spheres in the cube, jittered so nothing is symmetric.

    Perturbing matters: a symmetric configuration can make a wrong sign or a dropped
    cross-term integrate to zero and look correct.
    """
    # Centres 8 to 13 um apart at R = 5: every cell is in real contact with two or
    # three others, so the pair terms and the forces between cells are O(1), not noise.
    centres = np.array([[6.0, 7.0, 5.0], [13.0, 6.0, 10.0], [10.0, 15.0, 14.0], [5.0, 15.0, 6.0]])
    tissue = Tissue.seeded(grid3, centres, RADIUS, WIDTH, rng=np.random.default_rng(0))
    rng = np.random.default_rng(0)
    tissue.fields.values += 0.03 * rng.standard_normal(tissue.fields.values.shape)
    return tissue


@pytest.fixture
def fields3(tissue3):
    return tissue3.fields


@pytest.fixture
def roomy3():
    """Four clean spheres in a box big enough that their windows are smaller than it.

    The tail of the seeded profile reaches the window threshold about ``9.8 w`` beyond
    the radius, so at ``R = 4, w = 0.75`` a window is about 29 points on a unit grid;
    the box is 38 to 42 points on a side.
    """
    grid = Grid(PeriodicBox(40.0, 42.0, 38.0), (38, 42, 40))
    centres = np.array([[10.0, 10.0, 10.0], [19.0, 12.0, 14.0], [13.0, 21.0, 19.0], [24.0, 24.0, 8.0]])
    return Tissue.seeded(grid, centres, 4.0, 0.75, rng=np.random.default_rng(0))


def roomy_energy():
    well, gradient = interface_terms(0.75, 1.0)
    return FreeEnergy(well, gradient, Repulsion(4.0), Adhesion(0.2),
                      AreaConstraint.from_radius(4.0, 10.0, dimension=3))


def energy_for(**overrides):
    """The model free energy at the fixture's radius and width, in 3D."""
    well, gradient = interface_terms(WIDTH, 1.0)
    settings = dict(epsilon=4.0, omega=0.2, lambda_=10.0)
    settings.update(overrides)
    return FreeEnergy(
        well,
        gradient,
        Repulsion(settings["epsilon"]),
        Adhesion(settings["omega"]),
        AreaConstraint.from_radius(RADIUS, settings["lambda_"], dimension=3),
    )


TERMS = [
    DoubleWell(alpha=0.7),
    GradientEnergy(K=1.3),
    Repulsion(epsilon=2.5),
    Adhesion(omega=0.3),
    AreaConstraint(target_area=400.0, lambda_=40.0),
]


# ----------------------------------------------------------------- free energy


@pytest.mark.parametrize("term", TERMS, ids=lambda t: type(t).__name__)
def test_derivative_matches_a_finite_difference_in_3d(term, fields3):
    energy = FreeEnergy(term)
    analytic = energy.functional_derivative(fields3)
    assert analytic.shape == fields3.values.shape
    rng = np.random.default_rng(0)
    for index in sample_significant(analytic, 6, rng):
        numeric = finite_difference_derivative(energy, fields3, index)
        assert numeric == pytest.approx(analytic[index], rel=2e-4)


@pytest.mark.parametrize("term", TERMS, ids=lambda t: type(t).__name__)
def test_density_integrates_to_the_energy_in_3d(term, fields3):
    if not hasattr(term, "density"):
        pytest.skip("AreaConstraint has no local density")
    assert fields3.grid.integrate(term.density(fields3)) == pytest.approx(term.energy(fields3))


def test_repulsion_and_adhesion_equal_their_pair_sums_in_3d(fields3):
    grid, squared = fields3.grid, fields3.values**2
    literal = sum(grid.integrate(squared[i] * squared[j])
                  for i, j in combinations(range(fields3.n_cells), 2))
    assert Repulsion(2.5).energy(fields3) == pytest.approx(2.5 * literal)

    gradients = grid.forward_gradient(squared)              # (x, y, z), each (n, nz, ny, nx)
    literal = sum(
        grid.integrate(sum(g[i] * g[j] for g in gradients))
        for i, j in combinations(range(fields3.n_cells), 2)
    )
    assert Adhesion(0.3).energy(fields3) == pytest.approx(0.3 * literal)


def test_gradient_limit_matches_the_diffusive_formula_in_3d(grid3):
    """``2 gamma / (K sum_k 4/h_k^2)``: the compact Laplacian's stiffness in 3D."""
    fields = PhaseFields(grid3, np.zeros((2, *grid3.shape)))
    term, friction = GradientEnergy(K=1.7), 3.0
    expected = 2.0 * friction / (term.K * sum(4.0 / h**2 for h in grid3.spacings))
    assert term.stability_limit(fields, friction) == pytest.approx(expected)
    # Half the 2D limit at the same spacing: a third axis adds a third of the stiffness.
    flat = PhaseFields(Grid(PeriodicBox(24.0, 24.0), (24, 24)), np.zeros((2, 24, 24)))
    assert term.stability_limit(fields, friction) == pytest.approx(
        2.0 / 3.0 * term.stability_limit(flat, friction))


def test_ball_volume_and_the_radius_constructor():
    assert ball_volume(2.0) == pytest.approx(np.pi * 4.0)
    assert ball_volume(2.0, 3) == pytest.approx(4.0 / 3.0 * np.pi * 8.0)
    assert AreaConstraint.from_radius(2.0).target_area == pytest.approx(np.pi * 4.0)
    assert AreaConstraint.from_radius(2.0, dimension=3).target_area == pytest.approx(ball_volume(2.0, 3))


# ----------------------------------------------------------------- measurements


def test_surface_and_volume_of_a_sphere():
    """``INT |grad phi|`` is the surface for any monotone profile; ``INT phi^2`` the
    volume less the interface tail, as in 2D."""
    grid = Grid(PeriodicBox(20.0, 20.0, 20.0), (40, 40, 40))       # dx = 0.5
    fields = seed(grid, np.array([[10.0, 10.0, 10.0]]), 6.0, 1.0)
    assert perimeters(fields)[0] == pytest.approx(4.0 * np.pi * 36.0, rel=0.05)
    # The phi^2 weighting pulls the effective radius in by about half the profile's
    # decay length, sqrt(2) w: measured, 0.77 of the sharp volume at w/R = 1/6.
    volume = areas(fields)[0]
    assert 0.7 * ball_volume(6.0, 3) < volume < ball_volume(6.0, 3)
    # The shape index is S / V^(2/3): 4.836 for a sharp sphere, biased high by the tail.
    q = shape_indices(fields)[0]
    assert 4.836 < q < 6.5
    assert q == pytest.approx(perimeters(fields)[0] / volume ** (2.0 / 3.0))


def test_shape_index_helper_keeps_the_2d_formula():
    P, A = np.array([10.0, 12.0]), np.array([7.0, 9.0])
    assert np.array_equal(shape_index(P, A), P / np.sqrt(A))
    assert shape_index(P, A, 3) == pytest.approx(P / A ** (2.0 / 3.0))


def test_windowed_measurements_match_dense_in_3d(roomy3):
    fields = roomy3.fields
    windows = fields.find_windows(threshold=THRESHOLD)
    assert windows.ndim == 3
    assert all(n < g for n, g in zip(windows.shape, fields.grid.shape))      # real windows
    assert areas(fields, windows) == pytest.approx(areas(fields), rel=1e-6)
    assert perimeters(fields, windows) == pytest.approx(perimeters(fields), rel=1e-3)
    offset = fields.grid.box.min_image(centres_of_mass(fields, windows) - centres_of_mass(fields))
    assert np.linalg.norm(offset, axis=1).max() < 1e-4


# ----------------------------------------------------------------- forces and motion


def test_passive_forces_push_overlapping_spheres_apart():
    grid = Grid(PeriodicBox(30.0, 30.0, 30.0), (30, 30, 30))
    fields = seed(grid, np.array([[13.0, 15.0, 15.0], [17.0, 15.0, 15.0]]), 5.0, 1.0)
    forces = passive_forces(fields, FreeEnergy(Repulsion(10.0)).functional_derivative(fields))
    assert forces.shape == (2, 3)
    assert forces[0, 0] < 0 < forces[1, 0]                       # apart, along x
    assert forces[:, 1:] == pytest.approx(0.0, abs=1e-9)         # nothing across
    assert forces.sum(axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_passive_forces_sum_to_zero_for_an_asymmetric_pair_in_3d():
    """Newton's third law, from translation invariance, on two unequal spheres pushed
    well into each other along a generic direction, so the pair force is O(1) and
    nothing cancels by symmetry. Exact in the continuum; on the grid the residual is
    discretisation error -- 0.2% at two points per interface width, measured -- and
    what is checked is that it is small and falls with refinement. Every term is
    present."""
    centres = np.array([[7.0, 8.0, 8.0], [11.5, 9.5, 9.0]])            # 4.85 apart, radii 4 and 3
    well, gradient = interface_terms(1.0, 1.0)
    energy = FreeEnergy(well, gradient, Repulsion(4.0), Adhesion(0.2),
                        AreaConstraint(ball_volume(np.array([4.0, 3.0]), 3), 10.0))

    def forces_at(points):
        grid = Grid(PeriodicBox(16.0, 16.0, 16.0), (points, points, points))
        big, small = seed(grid, centres[:1], 4.0, 1.0), seed(grid, centres[1:], 3.0, 1.0)
        fields = PhaseFields(grid, np.concatenate([big.values, small.values]))
        return passive_forces(fields, energy.functional_derivative(fields))

    def residual(forces):
        return np.abs(forces.sum(axis=0)).max() / np.abs(forces).max()

    coarse, fine = forces_at(32), forces_at(64)                  # dx = 0.5, then 0.25
    assert coarse.shape == (2, 3)
    assert np.abs(coarse).max() > 0.5                            # a real push
    assert coarse[0] @ (centres[1] - centres[0]) < 0             # and apart, not together
    assert residual(coarse) < 1e-2
    assert residual(fine) < residual(coarse) / 2                 # discretisation, converging


def test_advection_translates_a_field_rigidly_in_3d(grid3):
    """Central differences conserve the first moment of ``phi`` exactly, so the centroid
    of ``phi`` (not of ``phi^2``, which dispersion distorts) must move at exactly ``v``."""
    fields = seed(grid3, np.array([[12.0, 13.0, 11.0]]), 3.0, 1.0)
    velocity = np.array([[0.4, -0.2, 0.3]])
    step, steps = 0.05, 40
    start = grid3.centre_of_mass(fields.values)[0]
    for _ in range(steps):
        fields.values += step * advection(fields, velocity)
    moved = grid3.box.min_image(grid3.centre_of_mass(fields.values)[0] - start)
    assert moved == pytest.approx(velocity[0] * step * steps, rel=1e-3)


def test_advection_rejects_a_velocity_of_the_wrong_dimension(fields3):
    with pytest.raises(ValueError, match="shape"):
        advection(fields3, np.zeros((fields3.n_cells, 2)))


def test_force_balance_velocities_have_three_components(tissue3):
    energy = energy_for()
    rule = ForceBalance(active_energy=1.0, cell_radius=RADIUS, cell_friction=10.0)
    velocities = rule.velocities(tissue3, energy.functional_derivative(tissue3.fields))
    assert velocities.shape == (tissue3.n_cells, 3)
    assert np.all(np.isfinite(velocities))


# ----------------------------------------------------------------- polarity


def test_polarity_in_3d_diffuses_on_the_unit_sphere():
    """``<n(0).n(t)> = exp(-2 D_r t)`` on the sphere -- twice the 2D rate, because two
    transverse directions carry the noise -- and ``|n| = 1`` throughout."""
    diffusion, step = 0.01, 0.05
    polarity = Polarity.random(4000, rotational_diffusion=diffusion,
                               rng=np.random.default_rng(0), dimension=3)
    assert polarity.dimension == 3 and polarity.angles is None
    start = polarity.directors.copy()
    for _ in range(400):
        polarity.rotate(step)
    elapsed = 400 * step
    assert np.linalg.norm(polarity.directors, axis=1) == pytest.approx(1.0)
    measured = (polarity.directors * start).sum(axis=1).mean()
    assert measured == pytest.approx(np.exp(-2 * diffusion * elapsed), rel=0.08)
    assert polarity.persistence_time == pytest.approx(1.0 / (2 * diffusion))
    # Isotropic: no component is preferred after many turns.
    assert np.abs(polarity.directors.mean(axis=0)).max() < 0.05


def test_polarity_in_2d_is_unchanged():
    polarity = Polarity.random(10, rng=np.random.default_rng(0))
    assert polarity.dimension == 2 and polarity.directors.shape == (10, 2)
    assert polarity.persistence_time == pytest.approx(1.0)
    with pytest.raises(ValueError):
        Polarity(np.zeros(3), directors=np.ones((3, 3)))
    with pytest.raises(ValueError):
        Polarity()


def test_polarity_dimension_must_match_the_tissue(fields3):
    with pytest.raises(ValueError, match="3D tissue"):
        Tissue(fields3, Polarity(np.zeros(fields3.n_cells)))
    assert Tissue(fields3).polarity.dimension == 3            # the passive default follows the grid
    assert Tissue(fields3).polarity.velocities.shape == (fields3.n_cells, 3)


# ----------------------------------------------------------------- stepping


def test_passive_energy_decreases_in_3d(tissue3):
    energy = energy_for()
    limit = ExplicitEuler(dt=1e-12, friction=1.0).max_stable_dt(tissue3, energy)
    stepper = ExplicitEuler(dt=0.4 * limit, friction=1.0)
    history = run(tissue3, energy, stepper, 40, sample_every=10)
    energies = [row["energy"] for row in history]
    assert all(later < earlier for earlier, later in zip(energies, energies[1:]))
    assert tissue3.time == pytest.approx(40 * stepper.dt)


def test_a_windowed_step_matches_a_dense_step_in_3d(roomy3):
    """One force-balance step both ways from the same tissue: equal to the dropped tail."""
    energy = roomy_energy()
    dense, windowed = roomy3.copy(), roomy3.copy()
    dt = 0.4 * ExplicitEuler(dt=1e-12, friction=1.0).max_stable_dt(dense, energy)
    rule = ForceBalance(active_energy=2.0, cell_radius=4.0, cell_friction=10.0)
    ExplicitEuler(dt=dt, friction=1.0, propulsion=rule).step(dense, energy)
    ExplicitEuler(dt=dt, friction=1.0, propulsion=rule, windowed=True).step(windowed, energy)
    assert np.abs(windowed.fields.values - dense.fields.values).max() < 1e-5
    windows = windowed.fields.windows
    assert windows is not None and windows.ndim == 3
    assert all(n < g for n, g in zip(windows.shape, roomy3.grid.shape))
    assert np.all(windowed.fields.values[~windows.mask()] == 0.0)


def test_a_propelled_sphere_crosses_the_corner_windowed_and_dense():
    """One cell straddling all three boundaries, driven at an imposed velocity, stepped
    both ways for long enough to move several points, so its window has to follow."""
    grid = Grid(PeriodicBox(32.0, 32.0, 32.0), (64, 64, 64))           # dx = 0.5
    well, gradient = interface_terms(0.75, 1.0)
    # The constraint has to hold the sphere against its own curvature pressure,
    # 2 sigma / R: its derivative is 4 lambda (1 - V/V0) / V0 per unit phi, so with
    # V0 = 113 a lambda of 20 -- enough for a disc in 2D -- lets a 3D sphere evaporate
    # within one time unit (measured: the field sum fell from 1184 to 2). 500 holds it.
    energy = FreeEnergy(well, gradient, AreaConstraint.from_radius(3.0, 500.0, dimension=3))
    target = ball_volume(3.0, 3)
    centre = np.array([[0.5, 31.5, 0.3]])

    def fresh():
        # R = 3, w = 0.75: the tail reaches the threshold ~7.4 um out, so the window is
        # about 48 points of the 64 -- a real window, with room to follow the cell.
        tissue = Tissue.seeded(grid, centre, 3.0, 0.75, speed=0.3, rotational_diffusion=0.0,
                               rng=np.random.default_rng(0))
        return tissue

    dense, windowed = fresh(), fresh()
    assert np.array_equal(dense.polarity.directors, windowed.polarity.directors)
    limit = ExplicitEuler(dt=1e-12, friction=1.0).max_stable_dt(dense, energy)
    common = dict(dt=0.4 * limit, friction=1.0, propulsion=ImposedVelocity())
    steps = int(np.ceil(3 * grid.dx / (0.3 * common["dt"])))       # travel about three points
    run(dense, energy, ExplicitEuler(**common), steps, check=False)
    run(windowed, energy, ExplicitEuler(**common, windowed=True, refresh_every=20), steps,
        check=False)

    assert 0.6 * target < areas(dense.fields)[0] < 1.2 * target      # still a sphere
    assert all(n < 64 for n in windowed.fields.windows.shape)        # a real window
    assert np.all(windowed.fields.values[~windowed.fields.windows.mask()] == 0.0)
    assert energy.energy(windowed.fields) == pytest.approx(energy.energy(dense.fields), rel=1e-6)
    assert np.abs(windowed.fields.values - dense.fields.values).max() < 1e-5
    offset = grid.box.min_image(centres_of_mass(windowed.fields) - centres_of_mass(dense.fields))
    assert np.linalg.norm(offset, axis=1).max() < 1e-3
    moved = grid.box.min_image(centres_of_mass(dense.fields) - centre)
    assert np.linalg.norm(moved, axis=1).max() > 2.5 * grid.dx
