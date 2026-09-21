"""The structure of saved runs: g(r) curves on one figure, and the order parameters.

Takes any number of ``.npz`` files, labels each by a model field (the active energy by
default), draws their pair correlation functions together with the hexagonal lattice
shells marked, and prints the hexatic order, hexagon fraction and g(r) peak heights.

Usage::

    python scripts/structure.py figures/run/*Ea0_*seed0.npz figures/run/*Ea18.9474_*seed0.npz \\
        --out figures/run/gofr.png

    python scripts/structure.py FILES --label-by adhesion --out figures/run/gofr_adhesion.png

Nothing is simulated; everything comes from the stored centres and contacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import plotstyle  # noqa: E402
import render  # noqa: E402
from migration_theory import load_trajectory, pair_correlation, structure  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--label-by", default="active_energy", dest="label_by",
                        help="model field that names each curve")
    parser.add_argument("--transient", type=float, default=0.3)
    parser.add_argument("--out", type=Path, required=True, help="the figure to write (.png)")
    args = parser.parse_args()

    curves, rows = [], []
    for path in args.files:
        trajectory = load_trajectory(path)
        value = getattr(trajectory.model, args.label_by)
        r, g = pair_correlation(trajectory, args.transient)
        curves.append((f"{args.label_by.replace('_', ' ')} = {value:g}",
                       r / trajectory.model.cell_spacing, g))
        rows.append((value, structure(trajectory, args.transient)))

    curves.sort(key=lambda c: c[0])
    rows.sort(key=lambda row: row[0])

    print(f"{args.label_by:>16}{'hexatic |psi6|':>16}{'hexagons':>10}{'g 1st peak':>12}{'g 2nd peak':>12}")
    for value, s in rows:
        print(f"{value:>16g}{s['hexatic_order']:>16.3f}{s['hexagon_fraction']:>10.2f}"
              f"{s['g_first_peak']:>12.2f}{s['g_second_peak']:>12.2f}")

    figure = render.pair_correlation_figure(curves)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
