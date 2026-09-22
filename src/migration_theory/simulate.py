"""Running a :class:`~migration_theory.model.Model` and recording what happened.

No plotting here and no plotting dependency. A :class:`Trajectory` is the handoff
between simulating and looking at the result: it holds the scalars over time, a
sequence of snapshots light enough to keep a few hundred of, and the final tissue.

Snapshots store ``sum_i phi_i^2`` rather than the individual fields. That single array
is near 1 inside any cell and dips towards 0.5 where two meet, so it shows the cell
boundaries on its own -- one array per frame rather than ``n_cells``, which is the
difference between a few hundred frames fitting in memory and not.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .diagnostics import (
    areas,
    centres_of_mass,
    confluence_error,
    overlap_matrix,
    perimeters,
    shape_index,
)
from .dynamics import Diverged, ExplicitEuler, run
from .model import Model
from .tissue import Tissue

__all__ = ["Snapshot", "Trajectory", "simulate", "save_trajectory", "load_trajectory"]


@dataclass(frozen=True)
class Snapshot:
    """One moment of a run: the scalars, and enough field to draw it."""

    step: int
    time: float
    energy: float
    breakdown: dict[str, float]
    area_ratio: float
    shape_index: float
    confluence: float
    occupancy: float
    centres: np.ndarray
    """``(n_cells, 2)`` centre of mass of each cell."""

    areas: np.ndarray
    """``(n_cells,)`` area of each cell."""

    perimeters: np.ndarray
    """``(n_cells,)`` perimeter of each cell."""

    contacts: np.ndarray
    """``(n_cells, n_cells)`` overlap matrix, zero on the diagonal.

    Kept per snapshot because neighbour *changes* over time are what a rearrangement
    rate is made of, and they cannot be recovered from the final state.
    """

    field: np.ndarray | None = None
    """``sum_i phi_i^2``, or ``None`` if fields were not kept."""

    @property
    def ndim(self) -> int:
        """The tissue's dimension, read from the centres."""
        return self.centres.shape[1]

    @property
    def shape_indices(self) -> np.ndarray:
        """``(n_cells,)`` dimensionless shape index ``P / sqrt(A)`` -- ``S / V^(2/3)`` in 3D."""
        return shape_index(self.perimeters, self.areas, self.ndim)


@dataclass
class Trajectory:
    """Everything a run produced."""

    model: Model
    snapshots: list[Snapshot]
    tissue: Tissue | None
    """Final state. ``None`` on a trajectory loaded from disk -- the fields are not saved."""
    dt: float
    max_dt: float
    steps: int
    wall_seconds: float
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def times(self) -> np.ndarray:
        return np.array([s.time for s in self.snapshots])

    @property
    def energies(self) -> np.ndarray:
        return np.array([s.energy for s in self.snapshots])

    @property
    def final(self) -> Snapshot:
        return self.snapshots[-1]

    @property
    def duration(self) -> float:
        """Physical time covered."""
        return self.dt * self.steps

    @property
    def drift(self) -> float:
        """``dF/dt`` over the last interval: how far from settled the run ended."""
        if len(self.snapshots) < 2:
            return float("nan")
        last, previous = self.snapshots[-1], self.snapshots[-2]
        span = last.time - previous.time
        return (last.energy - previous.energy) / span if span else float("nan")

    def settled(self, tolerance: float = 1e-3) -> bool:
        return abs(self.drift) < tolerance

    def summary_line(self) -> str:
        final = self.final
        state = "settled" if self.settled() else "NOT SETTLED"
        return (
            f"dt {self.dt:.3g} ({self.dt / self.max_dt:.0%} of limit)  "
            f"{self.steps} steps  t {self.duration:.0f}  "
            f"dF/dt {self.drift:+.2e} ({state})  "
            f"A/A0 {final.area_ratio:.3f}  q {final.shape_index:.2f}  "
            f"occ {final.occupancy:.3f}  {self.wall_seconds:.0f}s"
        )


def simulate(
    model: Model,
    steps: int | None = None,
    *,
    duration: float | None = None,
    seed: int = 0,
    warmup: float = 0.0,
    n_snapshots: int = 60,
    keep_fields: bool = True,
    tissue: Tissue | None = None,
    progress: Callable[[int, int, Snapshot], None] | None = None,
) -> Trajectory:
    """Run ``model`` and return the trajectory.

    Give exactly one of ``steps`` or ``duration``. Prefer ``duration``: it is a physical
    time, so it means the same thing when a coefficient changes, whereas a step count
    silently covers less ground whenever the stability limit tightens.

    Snapshots are taken at ``n_snapshots`` evenly spaced points, always including the
    start and the end. ``keep_fields=False`` drops the image data, which is what a
    parameter sweep wants -- it only needs the scalars and the final state.

    ``warmup`` runs that much time passively first, then resets the clock and starts
    recording. Worth doing for any active run: the seeded circles overlap heavily -- they
    must, since circles cannot tile -- so the mechanical forces at ``t = 0`` are far
    larger than anything the relaxed tissue ever sees. Under force balance those forces
    set the velocity, so without a warm-up the run opens with a velocity spike that is
    an artefact of the initial condition, not of the activity.

    Pass ``tissue`` to continue from an existing state instead of seeding a new one.
    """
    if (steps is None) == (duration is None):
        raise ValueError("give exactly one of steps or duration")

    tissue = model.tissue(seed) if tissue is None else tissue
    free_energy = model.free_energy()

    if warmup > 0:
        passive = ExplicitEuler(
            dt=model.stepper(tissue, free_energy).dt, friction=model.friction, safety=1.0,
            windowed=model.window,
        )
        passive.check(tissue, free_energy)
        run(tissue, free_energy, passive, max(1, round(warmup / passive.dt)), check=False)
        tissue.time = 0.0
    stepper = model.stepper(tissue, free_energy)
    max_dt = model.max_stable_dt(tissue, free_energy)
    stepper.check(tissue, free_energy)

    if duration is not None:
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        steps = max(1, int(round(duration / stepper.dt)))
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")

    target_area = model.target_area
    snapshots: list[Snapshot] = []

    def capture(step: int) -> Snapshot:
        fields = tissue.fields
        # With model.window the per-cell measurements run on each cell's own patch of
        # the grid; the windows are refreshed at each snapshot since cells move.
        windows = fields.refresh_windows() if model.window else None
        cell_areas = areas(fields, windows)
        cell_perimeters = perimeters(fields, windows)
        contacts = overlap_matrix(fields)
        np.fill_diagonal(contacts, 0.0)
        snapshot = Snapshot(
            step=step,
            time=tissue.time,
            energy=free_energy.energy(fields),
            breakdown=free_energy.breakdown(fields),
            area_ratio=float(cell_areas.mean() / target_area),
            shape_index=float(shape_index(cell_perimeters, cell_areas, fields.grid.ndim).mean()),
            confluence=confluence_error(fields),
            occupancy=float(fields.occupancy.max()),
            centres=centres_of_mass(fields, windows),
            areas=cell_areas,
            perimeters=cell_perimeters,
            contacts=contacts,
            field=(fields.values**2).sum(axis=0) if keep_fields else None,
        )
        if not np.isfinite(snapshot.energy) or not np.isfinite(fields.values).all():
            raise Diverged(
                f"fields stopped being finite by step {step} (t = {tissue.time:.4g}). "
                "Two usual causes. With adhesion: the free energy is unbounded below -- "
                "check that adhesion is below its ceiling, K divided by max(4 phi_i phi_j), "
                "which is lower than K wherever cells overlap. With activity: the grid "
                "Peclet number v dx gamma / K passed its check at the start but the "
                "cells sped up -- under force balance squeezed cells move faster than "
                "the free speed -- and central-difference advection went unstable. "
                "Measured (2026-09-22): at activity 3 this happens on a 1 um grid once "
                "gamma / K exceeds about 2.5 s/um^2. Refine the grid or lower gamma."
            )
        snapshots.append(snapshot)
        return snapshot

    # Even spacing, with any remainder folded into the last chunk so the final state is
    # always captured rather than landing between snapshots.
    n_snapshots = max(2, n_snapshots)
    chunk = max(1, steps // (n_snapshots - 1))
    boundaries = list(range(0, steps, chunk))[: n_snapshots - 1]

    started = time.perf_counter()
    capture(0)
    for index, start in enumerate(boundaries):
        length = (steps - start) if index == len(boundaries) - 1 else chunk
        run(tissue, free_energy, stepper, length, check=False)
        snapshot = capture(start + length)
        if progress is not None:
            progress(index + 1, len(boundaries), snapshot)

    return Trajectory(
        model=model,
        snapshots=snapshots,
        tissue=tissue,
        dt=stepper.dt,
        max_dt=max_dt,
        steps=steps,
        wall_seconds=time.perf_counter() - started,
        # The run arguments that are not Model fields but change the result. Saved
        # with the trajectory so a file on disk is fully reproducible from itself.
        extras={"warmup": float(warmup), "seed": int(seed)},
    )


# --------------------------------------------------------------------------- storage

_SCALARS = ("step", "time", "energy", "area_ratio", "shape_index", "confluence", "occupancy")


def save_trajectory(trajectory: Trajectory, path: str | Path, fields: bool = False) -> Path:
    """Write a trajectory to a compressed ``.npz``.

    Saves the model parameters and every per-snapshot measurement, and by default not
    the fields: the display arrays are large and the final tissue larger, while
    everything :mod:`~migration_theory.analysis` needs is the scalars, centres, areas,
    perimeters and contacts. A few hundred snapshots of a few dozen cells is well under
    a megabyte.

    ``fields=True`` also stores each snapshot's display field ``sum_i phi_i^2`` (single
    precision, so about 30 MB for 200 frames of a 190^2 grid), which is what lets a run
    be re-rendered -- in another frame, at another resolution -- without simulating it
    again. The final tissue is still not saved.

    Written because rendering a run and keeping only the video throws away every number
    in it -- and re-simulating to recover them costs far more than storing them did.
    """
    # Not with_suffix: a stem like "run_K1_v0.1" has ".1" as its suffix as far as
    # pathlib is concerned, so with_suffix would silently rename it to "run_K1_v0.npz"
    # -- the name a v0 = 0 run would use.
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = trajectory.snapshots
    arrays = {name: np.array([getattr(s, name) for s in snapshots]) for name in _SCALARS}
    arrays.update(
        centres=np.array([s.centres for s in snapshots]),
        areas=np.array([s.areas for s in snapshots]),
        perimeters=np.array([s.perimeters for s in snapshots]),
        contacts=np.array([s.contacts for s in snapshots]),
    )
    terms = sorted(snapshots[0].breakdown)
    arrays["breakdown"] = np.array([[s.breakdown[t] for t in terms] for s in snapshots])
    if fields and all(s.field is not None for s in snapshots):
        arrays["field"] = np.array([s.field for s in snapshots], dtype=np.float32)
    meta = {
        "model": {f.name: getattr(trajectory.model, f.name)
                  for f in dataclasses.fields(trajectory.model)},
        "terms": terms,
        "dt": trajectory.dt,
        "max_dt": trajectory.max_dt,
        "steps": trajectory.steps,
        "wall_seconds": trajectory.wall_seconds,
        "extras": trajectory.extras,
    }
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)
    return path


def load_trajectory(path: str | Path) -> Trajectory:
    """Read back a trajectory saved by :func:`save_trajectory`.

    ``tissue`` comes back as ``None`` -- the final fields are not saved -- so the result
    supports everything in :mod:`~migration_theory.analysis` but not the figures that
    draw the final configuration. The per-snapshot display fields come back if the file
    was written with ``fields=True``, and a video can then be rendered from it.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        terms = meta["terms"]
        arrays = {key: data[key] for key in data.files if key != "meta"}

    snapshots = [
        Snapshot(
            step=int(arrays["step"][i]),
            time=float(arrays["time"][i]),
            energy=float(arrays["energy"][i]),
            breakdown=dict(zip(terms, arrays["breakdown"][i])),
            area_ratio=float(arrays["area_ratio"][i]),
            shape_index=float(arrays["shape_index"][i]),
            confluence=float(arrays["confluence"][i]),
            occupancy=float(arrays["occupancy"][i]),
            centres=arrays["centres"][i],
            areas=arrays["areas"][i],
            perimeters=arrays["perimeters"][i],
            contacts=arrays["contacts"][i],
            field=arrays["field"][i].astype(float) if "field" in arrays else None,
        )
        for i in range(len(arrays["time"]))
    ]
    return Trajectory(
        model=Model(**meta["model"]),
        snapshots=snapshots,
        tissue=None,
        dt=meta["dt"],
        max_dt=meta["max_dt"],
        steps=meta["steps"],
        wall_seconds=meta["wall_seconds"],
        extras=meta.get("extras", {}),  # absent from files written before it was saved
    )
