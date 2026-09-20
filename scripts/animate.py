"""Run one simulation, save it, and render it.

The command line is generated from the fields of :class:`~migration_theory.model.Model`,
so it cannot drift out of step with the model -- a parameter added there appears here
without anyone having to remember.

Usage::

    python scripts/animate.py                                   # defaults, with video
    python scripts/animate.py --speed 0.03 --duration 10000
    python scripts/animate.py --no-video --analyse              # numbers only, much faster
    python scripts/animate.py --K 1 --adhesion 0.5 --energy

Every run writes ``figures/<name>.npz`` holding the parameters and every per-snapshot
measurement, readable later with
:func:`~migration_theory.simulate.load_trajectory` -- so the analysis can be redone, or
redone differently, without paying for the simulation again. The name encodes every
parameter, so a directory listing is self-describing.

Optional outputs: the video unless ``--no-video``, the free-energy figure with
``--energy``, and the tissue observables printed with ``--analyse``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from naming import argument_type, encode
from migration_theory import Model, save_trajectory, simulate, tissue_state

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model = parser.add_argument_group("model parameters")
    for field in dataclasses.fields(Model):
        model.add_argument(
            f"--{field.name.replace('_', '-')}",
            dest=field.name,
            type=argument_type(field.default),
            default=field.default,
        )
    run = parser.add_argument_group("run")
    run.add_argument("--duration", type=float, default=2000.0, help="physical time")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--warmup", type=float, default=0.0,
                     help="passive relaxation time before recording starts")
    run.add_argument("--transient", type=float, default=0.2,
                     help="fraction of the run discarded before measuring")
    output = parser.add_argument_group("output")
    output.add_argument("--no-video", action="store_true", dest="no_video",
                        help="skip rendering; much faster if you only want numbers")
    output.add_argument("--analyse", action="store_true", help="print tissue observables")
    output.add_argument("--energy", action="store_true", help="write the free-energy figure")
    output.add_argument("--frames", type=int, default=200)
    output.add_argument("--fps", type=int, default=20)
    output.add_argument("--format", choices=("gif", "mp4"), default="mp4")
    output.add_argument("--vmin", type=float, default=None, help="colour-scale floor")
    output.add_argument("--vmax", type=float, default=None, help="colour-scale ceiling")
    output.add_argument("--name", default=None, help="override the generated filename")
    output.add_argument("--outdir", default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = Model(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Model)})

    print(model)
    for note in model.concerns():
        print(f"  NOTE: {note}")

    # Activity statistics mean nothing over less than a few persistence times, and that
    # is easy to miss by orders of magnitude -- say so before spending the compute.
    if model.free_speed:
        turns = args.duration / model.persistence_time
        print(f"  run covers {turns:.2f} persistence times"
              + ("" if turns >= 3 else "   <- too short for migration statistics"))

    name = args.name or encode(model, args.duration, args.seed)
    print(f"\nrunning -> {name}")

    def progress(done, total, snapshot):
        if done % max(1, total // 10) == 0 or done == total:
            print(f"  {done}/{total}  t={snapshot.time:.1f}", flush=True)

    trajectory = simulate(
        model, duration=args.duration, seed=args.seed, warmup=args.warmup,
        n_snapshots=args.frames + 1, keep_fields=not args.no_video, progress=progress,
    )
    print(f"  {trajectory.summary_line()}")
    print(f"  wrote {save_trajectory(trajectory, Path(args.outdir) / name)}")

    if not args.no_video or args.energy:
        import matplotlib

        matplotlib.use("Agg")
        import plotstyle
        import render

        if not args.no_video:
            animation = render.make_animation(trajectory, fps=args.fps,
                                              vmin=args.vmin, vmax=args.vmax)
            print(f"  wrote {render.save_animation(animation, name, fps=args.fps, prefer=args.format)}")
        if args.energy:
            print(f"  wrote {plotstyle.save(render.energy_figure(trajectory), name + '_energy')}")

    if args.analyse:
        print("\ntissue state:")
        for key, value in tissue_state(trajectory, transient=args.transient).items():
            print(f"  {key:30} {value:12.6g}")


if __name__ == "__main__":
    main()
