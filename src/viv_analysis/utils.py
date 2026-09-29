# src/utils.py
"""
Shared utilities used across preprocess, training, and inference.

The most important thing here is `format_ur_label` — having ONE
canonical way to convert a float Ur value into a case-name string,
imported everywhere, so case-name drift can't happen silently.
"""

from __future__ import annotations
from pathlib import Path
import re

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TIME_GAP_FACTOR = 20.0  # matches reference_quality.py's discontinuity_jump_factor


def segment_by_time_gaps(t, gap_factor: float = DEFAULT_TIME_GAP_FACTOR) -> list[tuple[int, int]]:
    """Split a time array into contiguous [start, end) index segments,
    cutting wherever a step-to-step gap exceeds gap_factor times the
    array's own median timestep.

    Used to stop sliding-window construction (training sequences, teacher-
    forcing rollouts) from silently spanning a discontinuous CFD restart --
    a window built from consecutive ROWS across such a gap would present
    real-time-separated states to the GRU as if they were one continuous,
    smoothly-evolving trajectory (see Ur=6.9491's ~77.5s report-file gap).

    Returns [(0, len(t))] (a single segment covering everything) when there
    are no gaps -- the common case, and the exact input a caller with no
    gap-handling would have used, so callers that pass a gap-free time
    array see identical behavior to before this function existed.
    """
    t = np.asarray(t, dtype=float)
    n = len(t)
    if n < 2:
        return [(0, n)]
    dt = np.diff(t)
    dt_pos = dt[dt > 0]
    if len(dt_pos) == 0:
        return [(0, n)]
    med_dt = float(np.median(dt_pos))
    if med_dt <= 0:
        return [(0, n)]
    gap_after = np.where(dt > gap_factor * med_dt)[0]  # gap between i and i+1
    if len(gap_after) == 0:
        return [(0, n)]
    bounds = [0] + [int(i) + 1 for i in gap_after] + [n]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def format_ur_label(value: float) -> str:
    """
    Float → canonical 'Ur{value}' label, trailing zeros stripped.

    Examples:
        format_ur_label(5.0)  → 'Ur5'
        format_ur_label(5.25) → 'Ur5.25'
        format_ur_label(4.6)  → 'Ur4.6'
    """
    txt = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return f"Ur{txt}"


def parse_ur_label(label: str) -> float:
    """
    Inverse of format_ur_label. Robust to small variations.

    Examples:
        parse_ur_label('Ur5')    → 5.0
        parse_ur_label('Ur5.25') → 5.25
        parse_ur_label('Ur_4.6') → 4.6
    """
    m = re.search(r"[Uu][Rr][_\-]?([0-9]+(?:\.[0-9]+)?)", str(label))
    if not m:
        raise ValueError(f"Could not parse Ur label from '{label}'")
    return float(m.group(1))


def present_model_label(dataset: str, default: str) -> str:
    """
    Plot-legend label for the model trace. For the completed Re=200
    cylinder dataset this must read "Present model: (Re=200, m*=10,
    zeta=0.01)"; every other dataset keeps its existing label unchanged.
    """
    from viv_analysis.config import CYLINDER200_ALIASES
    if dataset.strip().lower() in CYLINDER200_ALIASES:
        return r"Present model: ($Re=200,\ m^*=10,\ \zeta=0.01$)"
    return default