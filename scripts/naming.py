"""Turning a Model into a filename.

Shared so that a run made by one script is named the same as one made by another, and
so a parameter added to ``Model`` reaches every output path at once.
"""

from __future__ import annotations

import dataclasses

__all__ = ["ABBREVIATIONS", "NOT_IN_NAME", "encode", "argument_type"]

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
}


def argument_type(default):
    """Infer a CLI type from a field's default -- annotations are strings at runtime."""
    if isinstance(default, int) and not isinstance(default, bool):
        return int
    if isinstance(default, str):
        return str
    return float  # floats, and the None-valued timestep


def encode(model, duration: float, seed: int) -> str:
    """A filename carrying every parameter, so a run is identifiable from it alone.

    Derived from the model's own fields rather than a hand-written list, for the same
    reason the command line is: a hand-written list drifts, and a run that collides
    with another because a parameter was forgotten is worse than a long name.

    The ``.npz`` stores every parameter too and is authoritative; this is for the human
    reading a directory listing. Fields left as ``None`` -- an unset ``timestep`` or
    ``cell_friction`` -- are omitted, since ``None`` means "derived" and printing a
    value would be a lie.
    """
    parts = []
    for field in dataclasses.fields(model):
        if field.name in NOT_IN_NAME:
            continue
        value = getattr(model, field.name)
        if value is None:
            continue
        token = ABBREVIATIONS.get(field.name, field.name)
        parts.append(f"{token}{value}" if isinstance(value, str) else f"{token}{value:g}")
    return "_".join([*parts, f"dur{duration:g}", f"seed{seed}"])
