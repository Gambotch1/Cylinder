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
from viv_analysis.reference_quality import _dominant_freq


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


def teacher_forced_rollout_with_state(
    model, case_df: pd.DataFrame, input_cols: list[str], seq_len: int,
    release_t: float, x_scaler, y_scaler, case_name: str, device: str,
    use_ur_context: bool, ur_stats: tuple[float, float] | None,
    nd_inputs: bool, D: float, fn: float,
    window_steps: int | None = None, handoff_offset_steps: int = 2000,
) -> dict:
    """Teacher-forced rollout that ALSO records, per step: the hidden-state
    norm (||h_n||, last-layer GRU state feeding the prediction head) and
    the standardized (training-sigma-unit) input values -- neither of
    which train_gru.teacher_forcing_rollout exposes. Duplicates that
    function's windowing logic rather than modifying it (diagnostic-only,
    zero risk to the training/eval path already in use elsewhere).

    Applies the SAME nd_inputs coordinate transform + x_scaler/y_scaler
    standardization the model was trained under, computed here directly
    (rather than via apply_nd_transform/apply_scalers_to_df on the whole
    df) so this function only needs the raw physical case_df as input.
    """
    from viv_analysis.utils import parse_ur_label as _parse

    model.eval()
    ordered = case_df.sort_values("time").reset_index(drop=True)
    times_full = ordered["time"].to_numpy(dtype=np.float64)
    h_full = ordered["disp"].to_numpy(dtype=np.float64)
    hdot_full = ordered["vel"].to_numpy(dtype=np.float64)
    cl_true_full = ordered["cl"].to_numpy(dtype=np.float64)

    U = _parse(str(case_name)) * fn * D
    divisor = {"disp": D, "vel": U}

    phys = ordered[input_cols].to_numpy(dtype=np.float64).copy()
    if nd_inputs:
        for k, col in enumerate(input_cols):
            phys[:, k] = phys[:, k] / divisor[col]
    x_mean = x_scaler.mean_.astype(np.float64)
    x_scale = x_scaler.scale_.astype(np.float64)
    signal_z = (phys - x_mean) / x_scale  # standardized, training-sigma units

    signal = signal_z.astype(np.float32)
    if use_ur_context:
        ur_mean, ur_std = ur_stats
        ur_std = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
        ur_scaled = (_parse(str(case_name)) - float(ur_mean)) / ur_std
        ur_col = np.full((signal.shape[0], 1), ur_scaled, dtype=np.float32)
        signal = np.hstack([signal, ur_col])

    release_idx = int(np.searchsorted(times_full, release_t))
    # Aligned to the SAME t_handoff the closed-loop pipeline hands off at
    # (coupled_inference.py::warmup_history): handoff_idx = release_idx +
    # seq_len + handoff_offset_steps -- NOT release_idx + handoff_offset_steps.
    # An earlier version of this function was missing the "+ seq_len" term,
    # which put its window start 5s (seq_len*dt at bridge's seq_len=2500,
    # dt~0.002s) EARLIER than the real closed-loop handoff point -- verified
    # directly by comparing t[0] against a real coupled_*.npz's t[0], which
    # disagreed by exactly 5.000s before this fix.
    handoff_idx = release_idx + seq_len + handoff_offset_steps
    start = max(seq_len, handoff_idx)
    end = len(ordered) if window_steps is None else min(start + window_steps, len(ordered))

    preds, hidden_norms, hidden_vecs = [], [], []
    with torch.no_grad():
        for i in range(start, end):
            w = np.array(signal[i - seq_len: i], copy=True)
            x = torch.from_numpy(w).unsqueeze(0).to(device)
            p, hn = model(x)
            preds.append(p.item())
            hidden_norms.append(float(torch.linalg.norm(hn[-1]).item()))
            hidden_vecs.append(hn[-1].detach().cpu().numpy().ravel().copy())

    cl_pred_s = np.array(preds, dtype=np.float32)
    cl_pred = y_scaler.inverse_transform(cl_pred_s.reshape(-1, 1)).ravel().astype(np.float64)

    sl = slice(start, end)
    return dict(
        t=times_full[sl], h=h_full[sl], h_dot=hdot_full[sl],
        cl_cfd=cl_true_full[sl], cl_pred=cl_pred,
        z_disp=signal_z[sl, input_cols.index("disp")] if "disp" in input_cols else None,
        z_vel=signal_z[sl, input_cols.index("vel")] if "vel" in input_cols else None,
        hidden_norm=np.array(hidden_norms, dtype=np.float64),
        hidden_state=np.stack(hidden_vecs, axis=0),  # (n_steps, hidden_size)
        release_idx=release_idx, start_idx=start,
    )


def blockwise_energy_and_phase(
    result: dict, rho: float, U: float, B: float, c: float,
    block_duration_s: float = 20.0,
) -> pd.DataFrame:
    """Per fixed-duration block: W_f (CFD-C_L-based and predicted-C_L-
    based), W_d, and the C_L-vs-hdot phase for both, at the block's own
    dominant h_dot frequency."""
    t, h_dot = result["t"], result["h_dot"]
    cl_cfd, cl_pred = result["cl_cfd"], result["cl_pred"]
    q = 0.5 * rho * U ** 2 * B

    t0 = t[0]
    n_blocks = max(1, int(np.ceil((t[-1] - t0) / block_duration_s)))
    rows = []
    for j in range(n_blocks):
        lo, hi = t0 + j * block_duration_s, t0 + (j + 1) * block_duration_s
        mask = (t >= lo) & (t < hi)
        if mask.sum() < 4:
            continue
        t_b, hdot_b = t[mask], h_dot[mask]
        cl_cfd_b, cl_pred_b = cl_cfd[mask], cl_pred[mask]

        F_cfd = q * cl_cfd_b
        F_pred = q * cl_pred_b
        W_f_cfd = float(np.trapezoid(F_cfd * hdot_b, t_b))
        W_f_pred = float(np.trapezoid(F_pred * hdot_b, t_b))
        W_d = float(np.trapezoid(c * hdot_b ** 2, t_b))

        f_dom = _dominant_freq(t_b, hdot_b)
        phase_cfd = phase_difference_rad(t_b, cl_cfd_b, hdot_b, f_dom) if np.isfinite(f_dom) and f_dom > 0 else float("nan")
        phase_pred = phase_difference_rad(t_b, cl_pred_b, hdot_b, f_dom) if np.isfinite(f_dom) and f_dom > 0 else float("nan")

        rows.append(dict(
            block=j, t_start=lo, t_end=hi, f_dom=f_dom,
            W_f_cfd=W_f_cfd, W_f_pred=W_f_pred, W_d=W_d,
            balance_cfd=W_f_cfd - W_d, balance_pred=W_f_pred - W_d,
            phase_cl_hdot_cfd_rad=phase_cfd, phase_cl_hdot_pred_rad=phase_pred,
            hidden_norm_mean=float(np.mean(result["hidden_norm"][mask])) if result["hidden_norm"] is not None else float("nan"),
        ))
    return pd.DataFrame(rows)


def detect_disagreement_onset(t: np.ndarray, cl_pred: np.ndarray, cl_cfd: np.ndarray,
                              threshold_std_mult: float = 2.0,
                              sustain_window_s: float = 5.0) -> float | None:
    """Time (relative to t[0], i.e. relative to handoff+seq_len) at which
    |cl_pred-cl_cfd| first exceeds threshold_std_mult * (the whole
    window's residual std) and STAYS above it for sustain_window_s
    continuously (single-sample exceedances from noise don't count)."""
    resid = np.abs(cl_pred - cl_cfd)
    thresh = threshold_std_mult * float(np.std(cl_pred - cl_cfd))
    if thresh <= 0:
        return None
    bad = resid > thresh
    if not np.any(bad):
        return None
    dt = float(np.median(np.diff(t)))
    sustain_n = max(1, int(sustain_window_s / dt))
    # first index where `bad` stays True for sustain_n consecutive samples
    run = 0
    for i, b in enumerate(bad):
        run = run + 1 if b else 0
        if run >= sustain_n:
            onset_idx = i - sustain_n + 1
            return float(t[onset_idx] - t[0])
    return None
