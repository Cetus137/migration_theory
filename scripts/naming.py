"""Turning a Model into a filename.

Shared so that a run made by one script is named the same as one made by another, and
so a parameter added to ``Model`` reaches every output path at once.
"""

from __future__ import annotations

import dataclasses

__all__ = ["ABBREVIATIONS", "NOT_IN_NAME", "encode", "argument_type", "add_model_argument"]

#: Parameters deliberately left out of the filename. ``time_unit`` is a label -- it
#: changes nothing about a run, so including it would only make names longer.
NOT_IN_NAME = {"time_unit"}

#: Short tokens for the filename. Any field without one falls back to its own name, so
#: a parameter added to Model appears in the name without anyone remembering to add it.
ABBREVIATIONS = {
    "alpha": "a",
    "K": "K",
    "epsilon": "eps",
    "adhesion": "om",
    "area_lambda": "lam",
    "cell_radius": "R",
    "packing": "pack",
    "n_cells": "N",
    "seeding": "",
    "friction": "gam",
    "propulsion": "",
    "speed": "v",
    "active_energy": "Ea",
    "cell_friction": "xi",
    "rotational_diffusion": "Dr",
    "grid_spacing": "dx",
    "timestep": "dt",
    "safety": "saf",
    "window": "win",
}


def argument_type(default):
    """Infer a CLI type from a field's default -- annotations are strings at runtime."""
    if isinstance(default, int) and not isinstance(default, bool):
        return int
    if isinstance(default, str):
        return str
    return float  # floats, and the None-valued timestep


def add_model_argument(group, field) -> None:
    """One command-line option for one ``Model`` field.

    A boolean field becomes a plain flag, ``--window`` rather than ``--window 1``,
    that flips the default; everything else takes a value of the field's type.
    """
    flag = f"--{field.name.replace('_', '-')}"
    if isinstance(field.default, bool):
        group.add_argument(flag, dest=field.name, default=field.default,
                           action="store_false" if field.default else "store_true")
    else:
        group.add_argument(flag, dest=field.name, default=field.default,
                           type=argument_type(field.default))


def encode(model, duration: float, seed: int, warmup: float = 0.0) -> str:
    """A filename carrying every parameter, so a run is identifiable from it alone.

    Derived from the model's own fields rather than a hand-written list, for the same
    reason the command line is: a hand-written list drifts, and a run that collides
    with another because a parameter was forgotten is worse than a long name.

    The ``.npz`` stores every parameter too and is authoritative; this is for the human
    reading a directory listing. Fields left as ``None`` -- an unset ``timestep`` or
    ``cell_friction`` -- are omitted, since ``None`` means "derived" and printing a
    value would be a lie. A boolean field appears as its bare token only when set, so
    names of runs made before the flag existed are unchanged.

    The run arguments that change the result -- duration, warm-up and seed -- go on the
    end. Warm-up is included even when zero: two runs differing only in warm-up are
    different runs, and must not overwrite each other.
    """
    parts = []
    for field in dataclasses.fields(model):
        if field.name in NOT_IN_NAME:
            continue
        value = getattr(model, field.name)
        if value is None:
            continue
        token = ABBREVIATIONS.get(field.name, field.name)
        if isinstance(value, bool):
            if value:
                parts.append(token)
            continue
        parts.append(f"{token}{value}" if isinstance(value, str) else f"{token}{value:g}")
    return "_".join([*parts, f"dur{duration:g}", f"wu{warmup:g}", f"seed{seed}"])
