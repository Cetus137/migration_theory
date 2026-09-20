"""Sweep one parameter and measure what it does to the tissue.

Runs the same model at several values of one field, saves every trajectory, and
tabulates the observables. Everything else is held fixed by construction -- each run is
``base.replace(field=value)`` -- so nothing can drift between points.

Usage::

    # the activity sweep: does the tissue unjam?
    python scripts/sweep.py --over active-energy --values 1 2 3 6 12 \\
        --propulsion force --epsilon 10 --rotational-diffusion 1e-3 \\
        --duration 10000 --warmup 25 --figure

    # adhesion, at fixed activity
    python scripts/sweep.py --over adhesion --values 0 0.3 0.6 0.9 \\
        --propulsion force --active-energy 3 --duration 5000 --warmup 25

    # a passive sweep: how the steady state depends on density
    python scripts/sweep.py --over packing --values 0.7 0.9 1.1 1.3 \\
        --duration 3000 --states

``--figure`` plots the observables against the swept value; ``--states`` draws the final
configuration of each run. Neither is needed for the numbers, which are printed and
saved regardless.

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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from naming import argument_type, encode
from migration_theory import Model, save_trajectory, simulate, tissue_state

#: What to print, and what to plot. Keys are `tissue_state` entries.
REPORTED = (
    ("msd_exponent", "MSD exponent"),
    ("diffusion_coefficient", "D"),
    ("exchange_rate_per_cell", "T1 rate per cell"),
    ("shape_index_mean", "shape index q"),
    ("shape_index_std", "q spread"),
    ("measured_speed", "measured speed"),
    ("occupancy", "max occupancy"),
    ("confluence", "confluence error"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    names = [f.name.replace("_", "-") for f in dataclasses.fields(Model)]
    parser.add_argument("--over", required=True, choices=names,
                        help="which model parameter to sweep")
    parser.add_argument("--values", type=float, nargs="+", required=True)

    model = parser.add_argument_group("held fixed")
    for field in dataclasses.fields(Model):
        model.add_argument(f"--{field.name.replace('_', '-')}", dest=field.name,
                           type=argument_type(field.default), default=field.default)
    run = parser.add_argument_group("run")
    run.add_argument("--duration", type=float, default=5000.0)
    run.add_argument("--warmup", type=float, default=25.0,
                     help="passive relaxation before recording; the seeding transient "
                          "is over by about 25")
    run.add_argument("--snapshots", type=int, default=200)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--transient", type=float, default=0.2)
    output = parser.add_argument_group("output")
    output.add_argument("--figure", action="store_true",
                        help="plot the observables against the swept value")
    output.add_argument("--states", action="store_true",
                        help="draw the final configuration of each run")
    output.add_argument("--outdir", default="figures")
    output.add_argument("--tag", default=None, help="prefix for the figure filenames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    field = args.over.replace("-", "_")
    base = Model(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Model)})
    keep_fields = args.states

    print(base)
    print(f"\nsweeping {field} over {args.values}")
    print(f"  duration {args.duration:g}, warmup {args.warmup:g}, "
          f"{args.duration / base.persistence_time:.1f} persistence times\n")

    results, states = [], []
    for value in args.values:
        model = base.replace(**{field: value})
        for note in model.concerns():
            print(f"  NOTE ({field}={value:g}): {note}")
        trajectory = simulate(
            model, duration=args.duration, warmup=args.warmup, seed=args.seed,
            n_snapshots=args.snapshots, keep_fields=keep_fields,
        )
        name = encode(model, args.duration, args.seed)
        save_trajectory(trajectory, Path(args.outdir) / name)
        state = tissue_state(trajectory, transient=args.transient)
        results.append((value, trajectory, state))
        states.append(state)
        print(f"  {field}={value:<8g} {trajectory.summary_line()}")

    # ---- the table ----
    print(f"\n{field:>14}" + "".join(f"{label:>20}" for _, label in REPORTED))
    for value, _, state in results:
        print(f"{value:>14g}" + "".join(f"{state[key]:>20.5g}" for key, _ in REPORTED))

    suspect = [v for v, _, s in results if s["occupancy"] > 1.4]
    if suspect:
        print(f"\n  WARNING: occupancy above 1.4 at {field} = {suspect}. Cells are "
              "interpenetrating there, so the apparent fluidity is numerical, not "
              "physical. Raise epsilon, lower the activity, or refine the grid.")

    if args.figure or args.states:
        import matplotlib

        matplotlib.use("Agg")
        import plotstyle
        import render

        tag = args.tag or f"sweep_{field}"
        if args.figure:
            figure = render.observables_figure(
                [v for v, _, _ in results], states, field, REPORTED
            )
            print(f"\nwrote {plotstyle.save(figure, tag)}")
        if args.states:
            figure = render.sweep_figure([(v, t) for v, t, _ in results], field)
            print(f"wrote {plotstyle.save(figure, tag + '_states')}")


if __name__ == "__main__":
    main()
