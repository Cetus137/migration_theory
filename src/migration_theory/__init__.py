"""A 2D phase-field simulation framework for interacting, active cells in a tissue.

Each cell is a smooth field ``phi_i(x)`` on a shared periodic grid, near 1 inside the
cell and 0 outside. Cell shape is an output rather than an imposed polygon, and the
quantities a particle model needs a tessellation for come out as integrals:
``A_i = INT phi_i^2`` is a cell's area and ``B_ij = INT phi_i^2 phi_j^2`` its contact
with a neighbour.

This is scaffolding only. The free energy and the timestepper are not implemented yet,
so nothing here evolves a configuration; what exists is the domain, the field storage,
the measurements and the self-propulsion term.
"""

from .activity import ForceBalance, ImposedVelocity, Polarity, advection, passive_forces
from .analysis import (
    Tracks,
    diffusion_coefficient,
    mean_squared_displacement,
    neighbour_exchange_rate,
    neighbour_graph,
    persistent_random_walk,
    shape_statistics,
    tissue_state,
    tracks,
    velocities,
    velocity_correlation,
    velocity_correlations,
)
from .box import PeriodicBox
from .dynamics import Diverged, ExplicitEuler, UnstableTimestep, run
from .diagnostics import (
    areas,
    centres_of_mass,
    confluence_error,
    contact_lengths,
    overlap_matrix,
    perimeters,
    shape_indices,
)
from .fields import PhaseFields, Windows, seed, seed_tessellated
from .free_energy import (
    Adhesion,
    AreaConstraint,
    DoubleWell,
    FreeEnergy,
    FreeEnergyTerm,
    GradientEnergy,
    Repulsion,
    interface_terms,
    interface_width,
    surface_tension,
)
from .grid import Grid
from .model import Model
from .simulate import (
    Snapshot,
    Trajectory,
    load_trajectory,
    save_trajectory,
    simulate,
)
from .tissue import Tissue
from .initialise import (
    evenly_spaced,
    lloyd_relax,
    poisson_disc,
    random_positions,
    triangular_lattice,
)

__version__ = "0.1.0"

__all__ = [
    "Adhesion",
    "AreaConstraint",
    "DoubleWell",
    "FreeEnergy",
    "FreeEnergyTerm",
    "GradientEnergy",
    "Grid",
    "Model",
    "Diverged",
    "ForceBalance",
    "ImposedVelocity",
    "ExplicitEuler",
    "PeriodicBox",
    "PhaseFields",
    "Repulsion",
    "Snapshot",
    "Tissue",
    "Tracks",
    "Trajectory",
    "UnstableTimestep",
    "Windows",
    "Polarity",
    "advection",
    "areas",
    "centres_of_mass",
    "confluence_error",
    "diffusion_coefficient",
    "mean_squared_displacement",
    "neighbour_exchange_rate",
    "neighbour_graph",
    "persistent_random_walk",
    "shape_statistics",
    "tissue_state",
    "tracks",
    "contact_lengths",
    "evenly_spaced",
    "interface_terms",
    "interface_width",
    "surface_tension",
    "lloyd_relax",
    "load_trajectory",
    "overlap_matrix",
    "passive_forces",
    "perimeters",
    "shape_indices",
    "simulate",
    "poisson_disc",
    "random_positions",
    "run",
    "save_trajectory",
    "seed",
    "seed_tessellated",
    "triangular_lattice",
    "velocities",
    "velocity_correlation",
    "velocity_correlations",
]
