"""Sweep one or two parameters and measure what they do to the tissue.

Runs the same model at several values of one field -- or on a grid of two -- at one or
more seeds, saves every trajectory, and tabulates the observables. Everything else is
held fixed by construction: each run is ``base.replace(...)``, so nothing can drift
between points.

Usage::

    # the activity sweep: does the tissue unjam?
    python scripts/sweep.py --over active-energy --values 1 2 3 6 12 \\
        --propulsion force --epsilon 10 --rotational-diffusion 1e-3 \\
        --duration 10000 --warmup 25 --figure

    # activity against adhesion: a two-parameter grid, three replicates each
    python scripts/sweep.py --over active-energy --values 0 6 12 18 24 \\
        --over adhesion --values 0 0.2 0.4 0.6 --seeds 0 1 2 \\
        --propulsion force --epsilon 40 --duration 5000 --warmup 25 --figure

    # a passive sweep: how the steady state depends on density
    python scripts/sweep.py --over packing --values 0.7 0.9 1.1 1.3 \\
        --duration 3000 --states

The points of a sweep are independent, so they can run in parallel -- one SLURM array
task each -- and be assembled afterwards from the saved files::

    python scripts/sweep.py <same arguments> --task-index $SLURM_ARRAY_TASK_ID
    python scripts/sweep.py <same arguments> --aggregate --figure

Points are numbered with the seed varying fastest, then the second swept field, then
the first: with ``n2`` second-field values and ``n_s`` seeds, point ``k`` is first-field
index ``k // (n2 * n_s)``, second-field index ``(k // n_s) % n2``, seed ``k % n_s``.
``--aggregate`` simulates nothing: it loads every point's ``.npz`` from ``--outdir`` and
lists any that are missing, so a partial sweep still tabulates. See
``SLURM/sweep_point.sl``.

``--figure`` plots the observables against the first swept field, one line per value
of the second, with the spread over seeds when there is more than one. ``--states``
draws the final configuration of each run of a one-parameter sweep, which needs the
fields and so only works when simulating in-process. Neither is needed for the
numbers, which are printed and saved regardless.

The three numbers that distinguish a solid tissue from a fluid one:

===========================  ================================================
``msd_exponent``             ~1 diffusive, <1 caged, ~2 ballistic
``exchange_rate_per_cell``   zero means cells never swap neighbours
``occupancy``                a validity check -- above ~1.4 means cells are
                             interpenetrating and the result is numerical
===========================  ================================================
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from naming import argument_type, encode
from progress import every_tenth
from migration_theory import Model, load_trajectory, save_trajectory, simulate, tissue_state

#: What to print, and what to plot. Keys are `tissue_state` entries.
REPORTED = (
    ("msd_exponent", "MSD exponent"),
    ("diffusion_coefficient", "D"),
    ("exchange_rate_per_cell", "T1 rate per cell"),
    ("shape_index_mean", "shape index q"),
    ("shape_index_std", "q spread"),
    ("measured_speed", "measured speed"),
    ("velocity_correlation_length", "v corr length (um)"),
    ("neighbour_velocity_correlation", "neighbour v corr"),
    ("occupancy", "max occupancy"),
    ("confluence", "confluence error"),
)

_FIELDS = {f.name: f for f in dataclasses.fields(Model)}

#: The fields a sweep can take. The string-valued ones -- seeding, propulsion,
#: time_unit -- are choices rather than quantities, so they are held fixed instead.
SWEEPABLE = [name for name, f in _FIELDS.items() if not isinstance(f.default, str)]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--over", action="append", required=True, metavar="FIELD",
                        choices=[name.replace("_", "-") for name in SWEEPABLE],
                        help="which model parameter to sweep; give it twice, each "
                             "followed by its own --values, for a two-parameter grid")
    parser.add_argument("--values", action="append", type=float, nargs="+", required=True,
                        metavar="V", help="the values for the preceding --over")

    model = parser.add_argument_group("held fixed")
    for field in _FIELDS.values():
        model.add_argument(f"--{field.name.replace('_', '-')}", dest=field.name,
                           type=argument_type(field.default), default=field.default)
    run = parser.add_argument_group("run")
    run.add_argument("--duration", type=float, default=5000.0)
    run.add_argument("--warmup", type=float, default=25.0,
                     help="passive relaxation before recording; the seeding transient "
                          "is over by about 25")
    run.add_argument("--snapshots", type=int, default=200)
    run.add_argument("--seeds", type=int, nargs="+", default=[0],
                     help="random seeds; more than one gives a spread per point")
    run.add_argument("--transient", type=float, default=0.2)
    parallel = parser.add_argument_group("parallel")
    parallel.add_argument("--task-index", type=int, default=None, dest="task_index",
                          help="run only this point and exit; for one SLURM array task")
    parallel.add_argument("--aggregate", action="store_true",
                          help="simulate nothing; tabulate the points already saved in "
                               "--outdir")
    output = parser.add_argument_group("output")
    output.add_argument("--figure", action="store_true",
                        help="plot the observables against the first swept parameter")
    output.add_argument("--states", action="store_true",
                        help="draw the final configuration of each run (one-parameter "
                             "sweeps, in-process only)")
    output.add_argument("--outdir", default="figures")
    output.add_argument("--tag", default=None, help="prefix for the figure filenames")
    args = parser.parse_args()
    if len(args.over) != len(args.values):
        parser.error("give one --values list per --over, in the same order")
    if len(args.over) > 2:
        parser.error("at most two parameters can be swept")
    if len(set(args.over)) != len(args.over):
        parser.error("the swept parameters must differ")
    if args.task_index is not None and args.aggregate:
        parser.error("--task-index and --aggregate are exclusive")
    return args


def sweep_points(args):
    """The swept fields, each one's values cast to its type, and the points in order.

    ``--values`` arrive as floats whatever the field is. An integer field such as
    ``n_cells`` feeds array shapes, which refuse a float, so each value is cast to the
    type the field's default has -- and refused if that would silently truncate it.

    A point is ``(combination, seed)`` with ``combination`` one value per swept field.
    """
    fields = [name.replace("-", "_") for name in args.over]
    axes = []
    for field, raw in zip(fields, args.values):
        cast = argument_type(_FIELDS[field].default)
        axis = []
        for value in raw:
            if cast is int and value != int(value):
                raise SystemExit(f"{field} takes whole numbers, got {value:g}")
            axis.append(cast(value))
        axes.append(axis)
    points = [(combination, seed)
              for combination in itertools.product(*axes) for seed in args.seeds]
    return fields, axes, points


def describe(fields, combination, seed=None) -> str:
    text = "  ".join(f"{field}={value:g}" for field, value in zip(fields, combination))
    return text if seed is None else f"{text}  seed {seed}"


def run_point(base, fields, combination, seed, args, keep_fields):
    """Simulate one point, save it, and return the trajectory and its observables."""
    model = base.replace(**dict(zip(fields, combination)))
    label = describe(fields, combination, seed)
    for note in model.concerns():
        print(f"  NOTE ({label}): {note}")
    print(f"  running {label}"
          + (f"; warming up for {args.warmup:g} {model.time_unit} first, "
             "progress starts after that" if args.warmup else ""), flush=True)
    trajectory = simulate(
        model, duration=args.duration, warmup=args.warmup, seed=seed,
        n_snapshots=args.snapshots, keep_fields=keep_fields,
        progress=every_tenth(label),
    )
    path = save_trajectory(
        trajectory, Path(args.outdir) / encode(model, args.duration, seed, args.warmup)
    )
    state = tissue_state(trajectory, transient=args.transient)
    print(f"  {label}  {trajectory.summary_line()}")
    print(f"    wrote {path}")
    return trajectory, state


def load_point(base, fields, combination, seed, args):
    """The observables of a point saved earlier, or ``None`` if its file is absent.

    A file that cannot be read -- most likely one an array task is writing at this
    very moment -- also counts as missing, with a note, rather than stopping the
    whole aggregation: aggregating while the array is still running is a normal
    thing to want.
    """
    model = base.replace(**dict(zip(fields, combination)))
    path = Path(args.outdir) / (encode(model, args.duration, seed, args.warmup) + ".npz")
    if not path.exists():
        return None
    try:
        return tissue_state(load_trajectory(path), transient=args.transient)
    except Exception as error:  # noqa: BLE001 -- any unreadable file is "not there yet"
        print(f"  could not read {path.name}: {type(error).__name__}: {error}")
        return None


def tabulate(fields, combinations, results, with_spread):
    """Print the table; return each point's mean and spread over seeds, per observable."""
    means, spreads = {}, {}
    header = "".join(f"{field:>16}" for field in fields) + f"{'seeds':>7}"
    print("\n" + header + "".join(f"{label:>22}" for _, label in REPORTED))
    for combination in combinations:
        states = [state for c, _, state in results if c == combination]
        # Every observable is averaged, not only the printed ones: the figure reads
        # some of the others, such as the bound on the correlation length.
        keys = states[0].keys()
        mean = {key: float(np.nanmean([s[key] for s in states])) for key in keys}
        spread = {key: float(np.nanstd([s[key] for s in states])) for key in keys}
        means[combination], spreads[combination] = mean, spread
        cells = (
            f"{mean[key]:.4g} +/- {spread[key]:.2g}" if with_spread else f"{mean[key]:.5g}"
            for key, _ in REPORTED
        )
        print("".join(f"{value:>16g}" for value in combination) + f"{len(states):>7}"
              + "".join(f"{cell:>22}" for cell in cells))
    return means, spreads


def main() -> None:
    args = parse_args()
    base = Model(**{name: getattr(args, name) for name in _FIELDS})
    fields, axes, points = sweep_points(args)

    print(base)
    print("\nsweeping " + "  x  ".join(f"{field} over {axis}" for field, axis in zip(fields, axes))
          + f"  at seeds {args.seeds}: {len(points)} points")
    print(f"  duration {args.duration:g}, warmup {args.warmup:g}, outdir {args.outdir}\n")

    # ---- one array task: simulate a single point and stop ----
    if args.task_index is not None:
        if not 0 <= args.task_index < len(points):
            raise SystemExit(f"task index {args.task_index} is outside 0..{len(points) - 1}")
        combination, seed = points[args.task_index]
        run_point(base, fields, combination, seed, args, keep_fields=False)
        return

    results = []        # (combination, seed, state)
    trajectories = {}   # combination -> the first seed's trajectory, for --states
    if args.aggregate:
        missing = []
        for index, (combination, seed) in enumerate(points):
            state = load_point(base, fields, combination, seed, args)
            if state is None:
                missing.append((index, combination, seed))
            else:
                results.append((combination, seed, state))
        if missing:
            print(f"  missing {len(missing)} of {len(points)} points:")
            for index, combination, seed in missing:
                print(f"    task {index:<4d} {describe(fields, combination, seed)}")
            # As one list, so the missing points can be resubmitted directly with
            #   sbatch --array=<list> sweep_point.sl
            print("  to rerun them:  sbatch --array="
                  + ",".join(str(index) for index, _, _ in missing) + " sweep_point.sl")
        if not results:
            raise SystemExit("nothing to tabulate")
        if args.states:
            print("  NOTE: --states needs the fields, which are not saved; skipping it")
    else:
        for combination, seed in points:
            trajectory, state = run_point(base, fields, combination, seed, args,
                                          keep_fields=args.states)
            results.append((combination, seed, state))
            trajectories.setdefault(combination, trajectory)

    # ---- the table ----
    present = [c for c in itertools.product(*axes) if any(r[0] == c for r in results)]
    with_spread = len(args.seeds) > 1
    means, spreads = tabulate(fields, present, results, with_spread)

    suspect = [describe(fields, c) for c in present if means[c]["occupancy"] > 1.4]
    if suspect:
        print("\n  WARNING: occupancy above 1.4 at " + "; ".join(suspect) + ". Cells are "
              "interpenetrating there, so the apparent fluidity is numerical, not "
              "physical. Raise epsilon, lower the activity, or refine the grid.")

    # ---- the figures ----
    if args.figure or (args.states and trajectories):
        import matplotlib

        matplotlib.use("Agg")
        import plotstyle
        import render

        tag = args.tag or "sweep_" + "_".join(fields)
        if args.figure:
            # One line per value of the second field, against the first.
            second = axes[1] if len(fields) == 2 else [None]
            series = []
            for other in second:
                xs = [c[0] for c in present if len(c) == 1 or c[1] == other]
                cs = [c for c in present if len(c) == 1 or c[1] == other]
                if not xs:
                    continue
                series.append({
                    "name": None if other is None else f"{fields[1]} = {other:g}",
                    "values": xs,
                    "states": [means[c] for c in cs],
                    "spreads": [spreads[c] for c in cs] if with_spread else None,
                })
            figure = render.observables_figure(series, fields[0], REPORTED)
            print(f"\nwrote {plotstyle.save(figure, tag, args.outdir)}")
        if args.states and trajectories:
            if len(fields) == 1:
                figure = render.sweep_figure([(c[0], trajectories[c]) for c in present],
                                             fields[0])
                print(f"wrote {plotstyle.save(figure, tag + '_states', args.outdir)}")
            else:
                print("  NOTE: --states draws one-parameter sweeps only; skipping it")


if __name__ == "__main__":
    main()
