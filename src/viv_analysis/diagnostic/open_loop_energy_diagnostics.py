#!/usr/bin/env python3
"""
Open-loop (teacher-forced) energy-balance diagnostic.

Purely diagnostic -- no training, no architecture/loss changes. Feeds the
model EXACT CFD [disp,vel](+Ur) history at every step (teacher forcing,
not closed loop) and asks: does the predicted C_L, combined with the TRUE
CFD velocity, transfer the right amount of energy into the structure?

    W_f^(j) = int_{t_j}^{t_{j+1}} F_L(t) hdot(t) dt      (fluid work in)
    W_d^(j) = int_{t_j}^{t_{j+1}} c hdot(t)^2 dt          (structural loss)

For a sustained response, W_f - W_d ~ 0 per cycle/block. A model can have
R^2=0.998 on raw C_L and still get this badly wrong, because net work is
the small residual of much larger positive/negative half-cycle
contributions -- pointwise accuracy does not bound work-integral accuracy.

Computed per FIXED-DURATION time block (not just the whole record), for
both CFD C_L and predicted C_L, so a "collapses without the regulator"
case's block-by-block trend is visible, not just a single aggregate number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from viv_analysis.utils import parse_ur_label


def single_bin_dft_complex(t: np.ndarray, x: np.ndarray, f: float) -> complex:
    """Complex single-bin DFT coefficient of x at frequency f (Hz), tolerant
    of a non-uniform grid. Magnitude matches
    closed_loop_metrics.single_bin_dft_amplitude; this also keeps phase."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n == 0:
        return complex(np.nan, np.nan)
    kernel = np.exp(-2j * np.pi * f * np.asarray(t, dtype=float))
    return (2.0 / n) * np.sum(x * kernel)


def phase_difference_rad(t: np.ndarray, x1: np.ndarray, x2: np.ndarray, f: float) -> float:
    """Phase of x1 relative to x2 at frequency f, wrapped to [-pi, pi].
    phase=0 means x1 and x2 rise and fall together (e.g. C_L in phase with
    hdot -> F_L*hdot is single-signed -> maximal net work of that sign)."""
    c1 = single_bin_dft_complex(t, x1, f)
    c2 = single_bin_dft_complex(t, x2, f)
    if not (np.isfinite(c1.real) and np.isfinite(c2.real)):
        return float("nan")
    diff = np.angle(c1) - np.angle(c2)
    return float(np.angle(np.exp(1j * diff)))  # wrap to [-pi, pi]
