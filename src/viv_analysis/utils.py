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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"


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