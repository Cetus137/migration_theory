"""Compare runs that should agree: the same points simulated two ways.

Pairs every ``.npz`` in one directory with the file of the same name in another --
ignoring a token such as ``_win`` that marks the variant -- and reports how far each
stored quantity and each derived observable differs. Written to check the windowed
computation against the dense one, but it compares any two directories of runs.

Usage::

    python scripts/compare_runs.py figures/activity_eps40 figures/activity_eps40_win --token win

What "agree" means depends on the quantity, and the report says so: with the
dynamics unchanged the energies and contacts must match to rounding, and the areas,
perimeters and centres to the tail the window drops (below 1e-6 in the field, ~1e-5
relative in a perimeter). With the dynamics on windows too, the trajectories part
company at that level, and in a fluid tissue a long run then differs in *when* each
rearrangement happens -- compare the observables against their spread over seeds
rather than file by file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from migration_theory import load_trajectory, tissue_state  # noqa: E402


def strip_token(name: str, token: str) -> str:
    return name.replace(f"_{token}_", "_")


def pair_files(reference: Path, variant: Path, token: str) -> list[tuple[Path, Path]]:
    by_name = {strip_token(p.name, token): p for p in variant.glob("*.npz")}
    pairs = []
    for path in sorted(reference.glob("*.npz")):
        other = by_name.get(strip_token(path.name, token))
        if other is not None:
            pairs.append((path, other))
    return pairs


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """Largest |a - b| relative to the largest |a|; 0 when both are all zero."""
    scale = float(np.max(np.abs(a)))
    return float(np.max(np.abs(a - b))) / scale if scale > 0 else 0.0


def compare(reference, variant, transient: float) -> dict[str, float]:
    box = reference.model.box
    snaps_a, snaps_b = reference.snapshots, variant.snapshots
    if len(snaps_a) != len(snaps_b):
        raise ValueError(f"{len(snaps_a)} snapshots against {len(snaps_b)}")
    stack = lambda snaps, name: np.array([getattr(s, name) for s in snaps])  # noqa: E731

    out = {
        "energy (rel)": relative(stack(snaps_a, "energy"), stack(snaps_b, "energy")),
        "contacts (rel)": relative(stack(snaps_a, "contacts"), stack(snaps_b, "contacts")),
        "areas (rel)": relative(stack(snaps_a, "areas"), stack(snaps_b, "areas")),
        "perimeters (rel)": relative(stack(snaps_a, "perimeters"), stack(snaps_b, "perimeters")),
    }
    offsets = box.min_image(stack(snaps_a, "centres") - stack(snaps_b, "centres"))
    out["centres (um, max)"] = float(np.max(np.hypot(offsets[..., 0], offsets[..., 1])))

    state_a = tissue_state(reference, transient)
    state_b = tissue_state(variant, transient)
    for key in ("shape_index_mean", "exchange_rate_per_cell", "diffusion_coefficient",
                "measured_speed", "velocity_correlation_length",
                "neighbour_velocity_correlation"):
        a, b = state_a[key], state_b[key]
        out[f"{key} (rel)"] = abs(a - b) / abs(a) if a else abs(b)
    return out


OBSERVABLES = (
    "exchange_rate_per_cell",
    "shape_index_mean",
    "shape_index_std",
    "diffusion_coefficient",
    "measured_speed",
    "neighbour_velocity_correlation",
    "velocity_correlation_length",
    "occupancy",
)


def short(name: str) -> str:
    """The swept part of a filename, for a table row."""
    tokens = [t for t in name.replace(".npz", "").split("_") if t.startswith(("Ea", "om", "seed"))]
    return " ".join(tokens)


def strip_seed(name: str) -> str:
    return "_".join(t for t in name.replace(".npz", "").split("_") if not t.startswith("seed"))


def spreads_over_seeds(directory: Path, token: str, transient: float) -> dict[str, dict[str, float]]:
    """Per point, the standard deviation of each observable over the seeds found there."""
    groups: dict[str, list[dict[str, float]]] = {}
    for path in sorted(directory.glob("*.npz")):
        key = strip_seed(strip_token(path.name, token))
        groups.setdefault(key, []).append(tissue_state(load_trajectory(path), transient))
    return {
        key: {name: float(np.std([s[name] for s in states])) for name in OBSERVABLES}
        for key, states in groups.items() if len(states) > 1
    }


def decoherence_time(reference, variant, tolerance: float = 1e-8) -> float:
    """When the two energy traces first part company beyond ``tolerance`` relative.

    Two runs that differ by a dropped tail agree for a while and then, in a fluid
    tissue, diverge once the perturbation has decided a rearrangement one way rather
    than the other. Reported so a difference in the observables can be read against
    how long the runs were actually the same trajectory.
    """
    a = np.array([s.energy for s in reference.snapshots])
    b = np.array([s.energy for s in variant.snapshots])
    apart = np.nonzero(np.abs(a - b) > tolerance * np.abs(a))[0]
    return float(reference.snapshots[apart[0]].time) if len(apart) else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path, help="directory of the runs to compare against")
    parser.add_argument("variant", type=Path, help="directory of the runs to check")
    parser.add_argument("--token", default="win",
                        help="filename token that marks the variant and is ignored in pairing")
    parser.add_argument("--transient", type=float, default=0.2)
    parser.add_argument("--mode", choices=("exact", "statistics"), default="exact",
                        help="exact: the trajectories must be the same, to rounding in the "
                             "energy. statistics: they may differ, and the observables are "
                             "compared instead, against their spread over seeds if --spread-from "
                             "gives a directory with several seeds per point")
    parser.add_argument("--spread-from", type=Path, default=None, dest="spread_from",
                        help="a directory of runs with several seeds per point, for the "
                             "seed-to-seed spread each observable is judged against")
    args = parser.parse_args()

    pairs = pair_files(args.reference, args.variant, args.token)
    if not pairs:
        raise SystemExit(f"no matching pairs between {args.reference} and {args.variant}")
    print(f"{len(pairs)} pairs, mode {args.mode}\n")

    if args.mode == "exact":
        worst: dict[str, float] = {}
        for ref_path, var_path in pairs:
            result = compare(load_trajectory(ref_path), load_trajectory(var_path), args.transient)
            for key, value in result.items():
                worst[key] = max(worst.get(key, 0.0), value)
            flag = ("  <- energies differ: the dynamics are not the same"
                    if result["energy (rel)"] > 1e-12 else "")
            print(f"{ref_path.name[:60]:60}  energy {result['energy (rel)']:.1e}  "
                  f"areas {result['areas (rel)']:.1e}  perim {result['perimeters (rel)']:.1e}  "
                  f"centres {result['centres (um, max)']:.2e} um{flag}")
        print("\nworst over all pairs:")
        for key, value in worst.items():
            print(f"  {key:40} {value:.3e}")
        print("\nexpected with the dynamics unchanged: energy and contacts ~1e-15, areas <1e-9, "
              "perimeters <1e-5, centres <1e-6 um, observables <1e-5.")
        if worst["energy (rel)"] > 1e-12:
            raise SystemExit("energies differ between the two sets: the dynamics changed")
        return

    # ---- statistics: the trajectories are allowed to differ ----
    spreads = (spreads_over_seeds(args.spread_from, args.token, args.transient)
               if args.spread_from else {})
    if args.spread_from and not spreads:
        print(f"  (no point in {args.spread_from} has more than one seed; spreads unavailable)\n")

    header = f"{'point':>18}{'decohere t':>12}{'speedup':>9}" + "".join(f"{n[:14]:>16}" for n in OBSERVABLES)
    print(header)
    ratios: dict[str, list[float]] = {name: [] for name in OBSERVABLES}
    speedups = []
    for ref_path, var_path in pairs:
        reference, variant = load_trajectory(ref_path), load_trajectory(var_path)
        a, b = tissue_state(reference, args.transient), tissue_state(variant, args.transient)
        spread = spreads.get(strip_seed(strip_token(ref_path.name, args.token)))
        cells = []
        for name in OBSERVABLES:
            diff = b[name] - a[name]
            if diff == 0:
                # Identical, whatever the spread -- including a spread of zero, where
                # every seed gave the same value, as a T1 rate of exactly 0 does.
                ratios[name].append(0.0)
                cells.append("+0.00 sd" if spread else "+0.0e+00")
            elif spread and spread[name] > 0:
                z = diff / spread[name]
                ratios[name].append(abs(z))
                cells.append(f"{z:+.2f} sd")
            else:
                rel = diff / a[name] if a[name] else float("nan")
                ratios[name].append(abs(rel))
                cells.append(f"{rel:+.1e}")
        speedup = reference.wall_seconds / variant.wall_seconds if variant.wall_seconds else float("nan")
        speedups.append(speedup)
        print(f"{short(ref_path.name):>18}{decoherence_time(reference, variant):>12.0f}"
              f"{speedup:>9.2f}" + "".join(f"{c:>16}" for c in cells))

    unit = "in seed standard deviations" if spreads else "relative (no spreads given)"
    print(f"\nmedian |difference| over points, {unit}:")
    for name in OBSERVABLES:
        values = np.array(ratios[name], dtype=float)
        print(f"  {name:34} median {np.nanmedian(values):.2f}   max {np.nanmax(values):.2f}")
    print(f"\nwall-time speedup, windowed over dense: median {np.nanmedian(speedups):.2f}x, "
          f"min {np.nanmin(speedups):.2f}x, max {np.nanmax(speedups):.2f}x")
    if spreads:
        print("\nread: |difference| well under 1 sd everywhere means the two computations sample "
              "the same tissue; a systematic offset of several sd in one observable would not.")


if __name__ == "__main__":
    main()
