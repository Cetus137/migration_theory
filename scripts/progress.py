"""Reporting how far a simulation has got, for the scripts.

:func:`~migration_theory.simulate.simulate` calls back after every snapshot; this turns
that into a line every tenth of the run with the physical time reached, the wall time
spent, and an estimate of what is left. The library itself never prints, so the
choice of what to say and how often is made here, once, for every script.
"""

from __future__ import annotations

import time

__all__ = ["every_tenth"]


def every_tenth(label: str = ""):
    """A progress callback that prints at 10 %, 20 %, ... and at the end.

    The estimate of time remaining assumes the remaining steps cost what the
    completed ones did, which holds well here: the step count is fixed and each step
    costs the same whatever the tissue is doing.

    The first line appears only after any warm-up and the first tenth of the run, so
    a quiet log at the start is not a sign of trouble.
    """
    started = time.perf_counter()
    prefix = f"  {label}  " if label else "  "

    def report(done: int, total: int, snapshot) -> None:
        stride = max(1, total // 10)
        if done % stride and done != total:
            return
        elapsed = time.perf_counter() - started
        fraction = done / total
        remaining = elapsed * (1.0 - fraction) / fraction if fraction else float("nan")
        print(
            f"{prefix}{fraction:4.0%}  t = {snapshot.time:.1f}  "
            f"elapsed {_clock(elapsed)}  remaining ~{_clock(remaining)}",
            flush=True,
        )

    return report


def _clock(seconds: float) -> str:
    """``h:mm:ss``, so a log can be read at a glance."""
    seconds = int(round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"
