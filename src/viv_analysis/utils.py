from __future__ import annotations
from pathlib import Path
import re

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TIME_GAP_FACTOR = 20.0


def segment_by_time_gaps(t, gap_factor: float = DEFAULT_TIME_GAP_FACTOR) -> list[tuple[int, int]]:
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
    gap_after = np.where(dt > gap_factor * med_dt)[0]
    if len(gap_after) == 0:
        return [(0, n)]
    bounds = [0] + [int(i) + 1 for i in gap_after] + [n]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def format_ur_label(value: float) -> str:
    txt = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return f"Ur{txt}"


def parse_ur_label(label: str) -> float:
    m = re.search(r"[Uu][Rr][_\-]?([0-9]+(?:\.[0-9]+)?)", str(label))
    if not m:
        raise ValueError(f"Could not parse Ur label from '{label}'")
    return float(m.group(1))


def present_model_label(dataset: str, default: str) -> str:
    from viv_analysis.config import CYLINDER200_ALIASES
    if dataset.strip().lower() in CYLINDER200_ALIASES:
        return r"Present model: ($Re=200,\ m^*=10,\ \zeta=0.01$)"
    return default
