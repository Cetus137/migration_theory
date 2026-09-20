"""Visual test: disordered cell centres seeded as phase fields on a periodic grid.

Nothing here evolves -- this checks that the domain, the seeding and the measurements
agree with each other before any dynamics are built on them. Three things are on trial:

1. **Periodicity.** Cells near an edge must wrap, appearing on both sides of the box.
2. **Periodic centre of mass.** A cell straddling a boundary must report its centroid
   back where it was seeded, not in the middle of the box -- the failure mode a plain
   weighted mean would give.
3. **The seeded profile.** Plotting phi against distance-from-centre for every grid
   point should collapse onto the analytic tanh. Scatter in that collapse means the
   periodic distance is wrong; a shifted curve means the profile is.

**Every parameter here is a ratio that means something on its own.** Lengths are
measured in mean cell spacings, so that unit never appears as a tunable number -- there
is no free overall scale to get wrong, and each knob below changes exactly one physical
or numerical property:

==================== ==========================================================
``--packing``        cell area / box area. 1.0 is confluence.
``--sharpness``      interface width / cell radius. How blurry a cell edge is.
``--cells``          how many cells.
``POINTS_PER_INTERFACE`` grid points across the interface. Numerical only.
==================== ==========================================================

Usage::

    python scripts/seed_random_cells.py                     # Lloyd-relaxed, the default
    python scripts/seed_random_cells.py --method poisson
    python scripts/seed_random_cells.py --method lattice --cells 36
    python scripts/seed_random_cells.py --packing 1.0       # confluent
    python scripts/seed_random_cells.py --packing 0.6       # sparse, with gaps

Each run writes its own figure, named by method and packing.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotstyle
from migration_theory import (
    Grid,
    PeriodicBox,
    areas,
    centres_of_mass,
    confluence_error,
    evenly_spaced,
    poisson_disc,
    random_positions,
    seed,
    triangular_lattice,
)

# Lengths are in units of the mean cell spacing, which is therefore exactly 1 and is
# never a parameter. Only ratios below.
PACKING_FRACTION = 0.907  # 0.907 = hexagonal close packing; 1.0 = confluent
INTERFACE_SHARPNESS = 0.16  # interface width as a fraction of the cell radius
POINTS_PER_INTERFACE = 3  # resolution: grid points across the interface
MIN_SEPARATION = 0.6  # Poisson disc: closest approach allowed, in spacings
LATTICE_JITTER = 0.15  # lattice: displacement standard deviation, in spacings
LLOYD_ITERATIONS = 25


@dataclass(frozen=True)
class Setup:
    """One configuration to seed and measure.

    Holds only the dimensionless inputs. Every length is derived from them, so the
    cell radius, interface width and grid spacing can never drift out of step with
    each other -- which is what went wrong when the interface width was pinned to an
    external unit instead of to the cell.
    """

    n_cells: int = 25
    packing_fraction: float = PACKING_FRACTION
    interface_sharpness: float = INTERFACE_SHARPNESS
    points_per_interface: int = POINTS_PER_INTERFACE

    def __post_init__(self) -> None:
        if self.packing_fraction <= 0:
            raise ValueError(f"packing_fraction must be positive, got {self.packing_fraction}")
        if self.interface_sharpness <= 0:
            raise ValueError(
                f"interface_sharpness must be positive, got {self.interface_sharpness}"
            )

    @property
    def radius(self) -> float:
        """Cell radius, in spacings.

        The box allots each cell ``sqrt(3)/2`` of area, so a disc of radius ``R``
        covers a fraction ``(2 pi / sqrt(3)) R**2``. This inverts that.
        """
        return float(np.sqrt(self.packing_fraction * np.sqrt(3.0) / (2.0 * np.pi)))

    @property
    def interface_width(self) -> float:
        """Interface width, in spacings. Tied to the cell, so cells stay equally sharp
        as the packing changes rather than getting blurrier as they shrink."""
        return self.interface_sharpness * self.radius

    @property
    def grid_spacing(self) -> float:
        return self.interface_width / self.points_per_interface

    def profile(self, r: np.ndarray) -> np.ndarray:
        """The radial profile :func:`seed` lays down."""
        return 0.5 * (
            1.0 - np.tanh((r - self.radius) / (np.sqrt(2.0) * self.interface_width))
        )

    def expected_area(self) -> float:
        """``2 pi INT phi(r)^2 r dr`` for an isolated seeded cell.

        The right reference for the measured area. Area is defined as ``INT phi^2``,
        and ``phi^2`` is not a sharp step at ``R`` -- at ``r = R`` it is 0.25, not 0.5 --
        so the measured value sits well below ``pi R^2`` with nothing wrong.
        Integrating the seeded profile itself says how far below, with no fitted
        constant.
        """
        r = np.linspace(0.0, self.radius + 12.0 * self.interface_width, 20_000)
        return float(2.0 * np.pi * np.trapezoid(self.profile(r) ** 2 * r, r))


# Each seeding returns its centres *and* its box: triangular_lattice picks a box its
# lattice tiles exactly, which is not quite the one the others use, and it rounds the
# cell count to fit. Letting each method own that keeps the difference honest rather
# than forcing them all through a box that only suits three of them. The spacing passed
# is 1.0 because that is the unit, not a choice.

def _even(n_cells, rng):
    box = PeriodicBox.for_cells(n_cells, 1.0)
    return evenly_spaced(n_cells, box, n_iterations=LLOYD_ITERATIONS, rng=rng), box


def _poisson(n_cells, rng):
    box = PeriodicBox.for_cells(n_cells, 1.0)
    return poisson_disc(n_cells, box, MIN_SEPARATION, rng=rng), box


def _lattice(n_cells, rng):
    return triangular_lattice(n_cells, 1.0, jitter=LATTICE_JITTER, rng=rng)


def _random(n_cells, rng):
    box = PeriodicBox.for_cells(n_cells, 1.0)
    return random_positions(n_cells, box, rng), box


METHODS = {
    "even": (_even, "Lloyd-relaxed (evenly_spaced)"),
    "poisson": (_poisson, f"Poisson disc (min sep {MIN_SEPARATION:g})"),
    "lattice": (_lattice, f"Triangular lattice (jitter {LATTICE_JITTER:g})"),
    "random": (_random, "Uniform random"),
}


def build(setup: Setup, method: str, random_seed: int):
    """Seed the configuration under test with the chosen method."""
    rng = np.random.default_rng(random_seed)
    centres, box = METHODS[method][0](setup.n_cells, rng)
    grid = Grid.from_spacing(box, setup.grid_spacing)
    fields = seed(grid, centres, setup.radius, setup.interface_width)
    return box, grid, centres, fields, rng


def spacing_stats(centres, box):
    """Nearest-neighbour distances, the cheapest measure of how even a packing is."""
    wrapped = box.wrap(centres)
    distances, _ = cKDTree(wrapped, boxsize=box.lengths).query(wrapped, k=2)
    nearest = distances[:, 1]
    return nearest.mean(), nearest.std() / nearest.mean(), nearest.min()


def report(setup: Setup, method, box, grid, centres, fields) -> None:
    """Print the numbers the picture should be consistent with."""
    measured = areas(fields)
    recovered = centres_of_mass(fields)
    drift = np.hypot(*box.min_image(recovered - centres).T)
    occupancy = fields.occupancy
    mean_nn, cv_nn, min_nn = spacing_stats(centres, box)

    print(f"method         {METHODS[method][1]}")
    print(f"cells          {fields.n_cells}")
    print("               -- lengths below are in mean cell spacings --")
    print(f"packing        {setup.packing_fraction:.3f}   ->  radius {setup.radius:.4f}")
    print(
        f"sharpness      {setup.interface_sharpness:.3f}   ->  interface "
        f"{setup.interface_width:.4f}"
    )
    print(f"box            {box.Lx:.3f} x {box.Ly:.3f}")
    print(
        f"grid           {grid.ny} x {grid.nx}   dx = {grid.dx:.4f}"
        f"   ({setup.interface_width / grid.dx:.1f} points across the interface)"
    )
    print(f"memory         {fields.values.nbytes / 1e6:.1f} MB for {fields.n_cells} fields")
    print()
    print(f"spacing        nearest-neighbour mean {mean_nn:.3f}  min {min_nn:.3f}")
    print(f"               CV {cv_nn:.3f}   (lower = more even)")
    print()
    print(f"area measured  {measured.mean():.4f} +/- {measured.std():.4f}")
    print(f"  expected     {setup.expected_area():.4f}   (2 pi INT phi^2 r dr, isolated cell)")
    print(f"  pi R^2       {np.pi * setup.radius**2:.4f}   (sharp-interface limit, for scale)")
    print()
    print(f"centroid drift max {drift.max() / grid.dx:.2f} grid spacings")
    print(f"occupancy      min {occupancy.min():.3f}  max {occupancy.max():.3f}")
    print(f"confluence err {confluence_error(fields):.3f}   (0 = gapless, no pile-up)")


def figure(setup: Setup, method, box, grid, centres, fields, rng):
    """Three panels: the tissue, one wrapped cell, and the profile collapse."""
    plotstyle.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), width_ratios=[1.0, 1.0, 1.15])

    X, Y = grid.coordinates
    occupancy = fields.occupancy

    # Panel 1 -- the whole tissue. Outlines are a single neutral ink, not one colour
    # per cell: with this many cells identity cannot be carried by hue, and need not be.
    ax = plotstyle.field_axes(axes[0], grid, "Occupancy  $\\sum_i \\phi_i$")
    image = plotstyle.show_field(ax, occupancy, grid, vmax=max(1.0, occupancy.max()))
    for field in fields:
        ax.contour(X, Y, field, levels=[0.5], colors=[plotstyle.INK], linewidths=0.7)
    ax.plot(centres[:, 0], centres[:, 1], ".", color=plotstyle.INK, ms=3)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)

    # Panel 2 -- the cell closest to an edge, which must visibly wrap.
    edge_distance = np.minimum(centres, box.lengths - centres).min(axis=1)
    target = int(np.argmin(edge_distance))
    recovered = centres_of_mass(fields)[target]

    ax = plotstyle.field_axes(
        axes[1], grid, f"Cell {target} alone  ($\\phi_{{{target}}}$, nearest the edge)"
    )
    plotstyle.show_field(ax, fields[target], grid)
    ax.plot(*centres[target], "+", color=plotstyle.INK, ms=9, mew=1.6, label="seeded centre")
    ax.plot(
        *recovered, "o", mfc="none", mec=plotstyle.SERIES[1], ms=10, mew=1.8,
        label="recovered centroid",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.04), ncol=2)

    # Panel 3 -- every grid point plotted against its distance from the cell centre.
    # Correct periodic distance collapses the whole 2D field onto the 1D profile.
    ax = axes[2]
    distance = grid.distance_to(centres[target]).ravel()
    values = fields[target].ravel()
    sample = rng.choice(distance.size, size=min(6000, distance.size), replace=False)
    ax.plot(
        distance[sample], values[sample], ".", ms=2.5, alpha=0.25,
        color=plotstyle.SERIES[0], label="grid points", rasterized=True,
    )
    r = np.linspace(0.0, 4.0 * setup.radius, 400)
    ax.plot(r, setup.profile(r), color=plotstyle.SERIES[1], label="analytic $\\tanh$ profile")
    ax.set_xlim(0.0, 4.0 * setup.radius)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("distance from centre  (spacings)")
    ax.set_ylabel("$\\phi$")
    ax.set_title("Profile collapse", loc="left", pad=8)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    fig.suptitle(
        f"{METHODS[method][1]}  --  {fields.n_cells} cells, "
        f"packing {setup.packing_fraction:g}, sharpness {setup.interface_sharpness:g}",
        x=0.01, ha="left", fontsize=11, fontweight="semibold", color=plotstyle.INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--method", choices=sorted(METHODS), default="even",
        help="seeding method (default: even, i.e. Lloyd-relaxed)",
    )
    parser.add_argument("--cells", type=int, default=25, help="target number of cells")
    parser.add_argument(
        "--packing", type=float, default=PACKING_FRACTION,
        help=f"cell area / box area (default {PACKING_FRACTION}; 1.0 = confluent)",
    )
    parser.add_argument(
        "--sharpness", type=float, default=INTERFACE_SHARPNESS,
        help=f"interface width / cell radius (default {INTERFACE_SHARPNESS})",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup = Setup(
        n_cells=args.cells,
        packing_fraction=args.packing,
        interface_sharpness=args.sharpness,
    )
    box, grid, centres, fields, rng = build(setup, args.method, args.seed)
    report(setup, args.method, box, grid, centres, fields)
    path = plotstyle.save(
        figure(setup, args.method, box, grid, centres, fields, rng),
        f"seed_{args.method}_p{args.packing:g}",
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
