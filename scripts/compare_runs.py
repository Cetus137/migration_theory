"""Compare runs that should agree: the same points simulated two ways.

Pairs every ``.npz`` in one directory with the file of the same name in another --
ignoring a token such as ``_win`` that marks the variant -- and reports how far each
stored quantity and each derived observable differs. Written to check the windowed
computation against the dense one, but it compares any two directories of runs.

Usage::

    python scripts/compare_runs.py figures/activity_eps40 figures/activity_eps40_win --token win

What "agree" means depends on the quantity, and the report says so: with the
dynamics unchanged the energies and contacts must match to rounding, the areas and
perimeters to the tail the window drops (below 1e-6 in the field, ~1e-5 relative in
a perimeter), and the centres to the small bias of the dense circular mean -- a few
hundredths of a micron for an asymmetric cell.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path, help="directory of the runs to compare against")
    parser.add_argument("variant", type=Path, help="directory of the runs to check")
    parser.add_argument("--token", default="win",
                        help="filename token that marks the variant and is ignored in pairing")
    parser.add_argument("--transient", type=float, default=0.2)
    args = parser.parse_args()

    pairs = pair_files(args.reference, args.variant, args.token)
    if not pairs:
        raise SystemExit(f"no matching pairs between {args.reference} and {args.variant}")
    print(f"{len(pairs)} pairs\n")

    worst: dict[str, float] = {}
    for ref_path, var_path in pairs:
        result = compare(load_trajectory(ref_path), load_trajectory(var_path), args.transient)
        for key, value in result.items():
            worst[key] = max(worst.get(key, 0.0), value)
        flag = "  <- energies differ: the dynamics are not the same" if result["energy (rel)"] > 1e-12 else ""
        print(f"{ref_path.name[:60]:60}  energy {result['energy (rel)']:.1e}  "
              f"areas {result['areas (rel)']:.1e}  perim {result['perimeters (rel)']:.1e}  "
              f"centres {result['centres (um, max)']:.2e} um{flag}")

    print("\nworst over all pairs:")
    for key, value in worst.items():
        print(f"  {key:40} {value:.3e}")

    print("\nexpected with the dynamics unchanged: energy and contacts ~1e-15, areas <1e-9, "
          "perimeters <1e-5, centres <0.1 um, observables <1e-3 except those built on "
          "centres, which inherit the centre bias.")
    if worst["energy (rel)"] > 1e-12:
        raise SystemExit("energies differ between the two sets: the dynamics changed")


if __name__ == "__main__":
    main()
