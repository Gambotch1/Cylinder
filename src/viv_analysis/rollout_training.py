#!/usr/bin/env python3
"""
Closed-loop simulation connecting a PyTorch GRU model with a Newmark-beta
physics solver to fine-tune predictions over time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from viv_analysis.coupled_inference import Newmark_beta, warmup_history
from viv_analysis.utils import parse_ur_label


@dataclass
class ScalerConstants:
    """Stores normalization parameters (mean and scaling factor)."""
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float

    @classmethod
    def from_sklearn(cls, x_scaler, y_scaler) -> "ScalerConstants":
        return cls(
            x_mean=x_scaler.mean_.astype(np.float64),
            x_scale=x_scaler.scale_.astype(np.float64),
            y_mean=float(y_scaler.mean_[0]),
            y_scale=float(y_scaler.scale_[0]),
        )


def build_next_row_torch(
    h_i: torch.Tensor, hdot_i: torch.Tensor, hddot_i: torch.Tensor,
    input_cols: list[str], nd_inputs: bool, D: float, U: float,
    x_mean: np.ndarray, x_scale: np.ndarray,
    use_ur_context: bool, ur_scaled: float,
) -> torch.Tensor:
    """Normalizes current displacement, velocity, and acceleration into a GRU input row."""
    state_by_name = {"disp": h_i, "vel": hdot_i, "acc": hddot_i}
    kin = torch.stack([state_by_name[c] for c in input_cols], dim=-1)  # (B, n_kin)

    if nd_inputs:
        divisor_by_name = {"disp": D, "vel": U, "acc": (U * U) / D}
        divisors = torch.tensor([divisor_by_name[c] for c in input_cols],
                                dtype=kin.dtype, device=kin.device)
        kin = kin / divisors

    # Standardise: (X - mean) / scale
    x_mean_t = torch.as_tensor(x_mean, dtype=kin.dtype, device=kin.device)
    x_scale_t = torch.as_tensor(x_scale, dtype=kin.dtype, device=kin.device)
    kin_scaled = (kin - x_mean_t) / x_scale_t

    # Append Ur_context (Ur) if provided
    if use_ur_context:
        ur_col = torch.full((kin_scaled.shape[0], 1), float(ur_scaled),
                            dtype=kin.dtype, device=kin.device)
        return torch.cat([kin_scaled, ur_col], dim=-1)
    return kin_scaled


def rollout_chunk(
    model, window: torch.Tensor,
    h_state: torch.Tensor, hdot_state: torch.Tensor, hddot_state: torch.Tensor,
    n_steps: int, dt: float, m: float, c: float, k: float, q: float, U: float, D: float,
    input_cols: list[str], nd_inputs: bool, sc: ScalerConstants,
    use_ur_context: bool, ur_scaled: float,
) -> dict:
    """Run n_steps of the closed GRU-Newmark loop while tracking gradients
    """
    win = window
    h_i, hdot_i, hddot_i = h_state, hdot_state, hddot_state
    cl_scaled_steps, cl_phys_steps, h_steps, hdot_steps = [], [], [], []

    for _ in range(n_steps):
        # 1.Predict aerodynamic force coefficient (scaled) using GRU
        pred_scaled, _ = model(win)                      
        cl_phys = pred_scaled * sc.y_scale + sc.y_mean     
        F = q * cl_phys

        # 2.Phyics step: Calculate net structural response (h, hdot, hddot) using Newmark-beta
        h_next, hdot_next, hddot_next = Newmark_beta(
            F=F, h=h_i, h_dot=hdot_i, h_ddot=hddot_i, dt=dt, m=m, c=c, k=k,
        )

        # 3.Slide window: Drop oldest time step, append new physical state
        new_row = build_next_row_torch(
            h_i, hdot_i, hddot_i, input_cols, nd_inputs, D, U,
            sc.x_mean, sc.x_scale, use_ur_context, ur_scaled,
        )
        win = torch.cat([win[:, 1:, :], new_row.unsqueeze(1)], dim=1)

        # 4.Track ouptuts 
        cl_scaled_steps.append(pred_scaled)
        cl_phys_steps.append(cl_phys)
        h_steps.append(h_i)
        hdot_steps.append(hdot_i)

        # 5.Adanve state
        h_i, hdot_i, hddot_i = h_next, hdot_next, hddot_next

    return dict(
        cl_scaled=torch.stack(cl_scaled_steps, dim=1),   # (B, n_steps)
        cl_phys=torch.stack(cl_phys_steps, dim=1),
        h=torch.stack(h_steps, dim=1),
        hdot=torch.stack(hdot_steps, dim=1),
        window=win, h_state=h_i, hdot_state=hdot_i, hddot_state=hddot_i,
    )


def sample_batch_starts(
    case_df: pd.DataFrame, release_t: float, seq_len: int, max_future_steps: int,
    batch_size: int, rng: np.random.Generator,
) -> list[int]:
    """Selects random starting points across usable time-series data."""
    ordered_times = case_df["time"].to_numpy(dtype=np.float64)
    release_idx = int(np.searchsorted(ordered_times, release_t))
    lo = release_idx + seq_len
    hi = len(ordered_times) - max_future_steps - 1
    edges = np.linspace(lo, hi, batch_size + 1)
    starts = [int(rng.integers(int(edges[i]), int(edges[i + 1]) + 1))
              for i in range(batch_size)]
    return starts


def build_batch_from_case(
    case_df: pd.DataFrame, case_name: str, starts: list[int], seq_len: int,
    input_cols: list[str], x_scaler, nd_inputs: bool, D: float, fn: float,
    use_ur_context: bool, ur_stats: tuple[float, float],
    max_future_steps: int, device: str,
) -> dict:
    """Build a batched initial (window, h_state, hdot_state, hddot_state)
    plus the CFD reference (h, hdot, cl) for up to max_future_steps beyond
    each start index, for `starts` (all from the SAME case, so Ur/U/q are
    shared batch-wide scalars, not per-sample tensors -- the significant
    simplification this training design relies on; cross-case variety comes
    from cycling through cases across iterations, not within one batch).
    Reuses warmup_history (coupled_inference.py) per start index -- exact
    same construction the real closed-loop pipeline uses at its handoff
    point, just invoked at an arbitrary chosen index instead of a fixed
    release+handoff_offset.
    """
    ur_value = parse_ur_label(str(case_name))
    U = ur_value * fn * D
    ordered = case_df.sort_values("time").reset_index(drop=True)
    times = ordered["time"].to_numpy(dtype=np.float64)

    histories, h0s, hdot0s, hddot0s = [], [], [], []
    cfd_h, cfd_hdot, cfd_cl = [], [], []
    for start_idx in starts:
        release_t_eq = times[start_idx - seq_len]  # forces release_idx == start_idx-seq_len
        hist, init_state, _, handoff_idx = warmup_history(
            ordered, release_t_eq, seq_len, input_cols, x_scaler,
            nd_inputs=nd_inputs, D=D, U=U, use_ur_context=use_ur_context,
            ur_value=ur_value, ur_stats=ur_stats, handoff_offset_steps=0,
        )
        assert handoff_idx == start_idx, (handoff_idx, start_idx)
        histories.append(hist)
        h0s.append(init_state["h"]); hdot0s.append(init_state["h_dot"])
        hddot0s.append(init_state["h_ddot"])
        sl = slice(start_idx, start_idx + max_future_steps)
        cfd_h.append(ordered["disp"].to_numpy(dtype=np.float64)[sl])
        cfd_hdot.append(ordered["vel"].to_numpy(dtype=np.float64)[sl])
        cfd_cl.append(ordered["cl"].to_numpy(dtype=np.float64)[sl])

    dt = float(np.median(np.diff(times)))
    ur_mean, ur_std = ur_stats
    ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
    ur_scaled = (ur_value - float(ur_mean)) / ur_std_safe

    return dict(
        window=torch.tensor(np.stack(histories), dtype=torch.float32, device=device),
        h_state=torch.tensor(h0s, dtype=torch.float32, device=device),
        hdot_state=torch.tensor(hdot0s, dtype=torch.float32, device=device),
        hddot_state=torch.tensor(hddot0s, dtype=torch.float32, device=device),
        cfd_h=torch.tensor(np.stack(cfd_h), dtype=torch.float32, device=device),
        cfd_hdot=torch.tensor(np.stack(cfd_hdot), dtype=torch.float32, device=device),
        cfd_cl=torch.tensor(np.stack(cfd_cl), dtype=torch.float32, device=device),
        dt=dt, U=U, ur_scaled=ur_scaled, ur_value=ur_value,
    )


def build_tf_batch_from_case(
    case_df: pd.DataFrame, case_name: str, start_idx: int, n_steps: int, seq_len: int,
    input_cols: list[str], x_scaler, nd_inputs: bool, D: float, fn: float,
    use_ur_context: bool, ur_stats: tuple[float, float], device: str,
) -> dict:
    """Teacher-forced batch for the REVISED Model-A objective's L_TF branch:
    n_steps INDEPENDENT (window, target) pairs built from TRUE CFD kinematic
    histories -- no structural integrator, no self-generated state, no
    recurrence across steps (each window is scaled and fed to the model on
    its own, exactly as VIVSequenceDataset/apply_scalers_to_df would during
    ordinary pointwise training). Once the rollout branch's generated state
    departs from the CFD state, C_L^CFD(t) is no longer the correct label
    for the generated input -- this function's whole purpose is to give the
    open-loop label its own, uncontaminated forward pass.

    Uses the SAME start_idx as the rollout branch's chunk for that case/step
    (not a fresh random draw), so both branches look at the same stretch of
    the trajectory at each training update.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    ur_value = parse_ur_label(str(case_name))
    U = ur_value * fn * D
    ordered = case_df.sort_values("time").reset_index(drop=True)

    kin_phys = ordered[input_cols].to_numpy(dtype=np.float64).copy()
    if nd_inputs:
        divisor = {"disp": D, "vel": U}
        for k, col in enumerate(input_cols):
            kin_phys[:, k] = kin_phys[:, k] / divisor[col]
    x_mean = x_scaler.mean_.astype(np.float64)
    x_scale = x_scaler.scale_.astype(np.float64)
    kin_scaled = (kin_phys - x_mean) / x_scale

    if use_ur_context:
        ur_mean, ur_std = ur_stats
        ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
        ur_scaled = (ur_value - float(ur_mean)) / ur_std_safe
        ur_col = np.full((kin_scaled.shape[0], 1), ur_scaled, dtype=np.float64)
        signal = np.hstack([kin_scaled, ur_col])
    else:
        signal = kin_scaled

    all_windows = sliding_window_view(signal, window_shape=seq_len, axis=0)
    all_windows = np.transpose(all_windows, (0, 2, 1))  # (n_windows, seq_len, n_features)
    # all_windows[i] covers signal[i : i+seq_len] and is used to predict target[i+seq_len]
    tf_windows = all_windows[start_idx - seq_len: start_idx - seq_len + n_steps]
    if tf_windows.shape[0] != n_steps:
        raise ValueError(f"TF batch out of bounds: got {tf_windows.shape[0]} windows, "
                        f"expected {n_steps} (start_idx={start_idx}, case len={len(ordered)}).")

    cl_true = ordered["cl"].to_numpy(dtype=np.float64)[start_idx: start_idx + n_steps]

    return dict(
        x=torch.tensor(np.ascontiguousarray(tf_windows), dtype=torch.float32, device=device),
        cl_cfd=torch.tensor(cl_true, dtype=torch.float32, device=device),
    )


# ── Losses ───────────────────────────────────────────────────────────────────

def loss_cl(cl_scaled_pred: torch.Tensor, cl_cfd_phys: torch.Tensor,
           y_mean: float, y_scale: float) -> torch.Tensor:
    """MSE Loss for predicted force against ground truth CFD force."""
    cl_cfd_scaled = (cl_cfd_phys - y_mean) / y_scale
    
    return torch.mean((cl_scaled_pred - cl_cfd_scaled) ** 2)


def loss_roll(h_pred: torch.Tensor, hdot_pred: torch.Tensor,
             h_cfd: torch.Tensor, hdot_cfd: torch.Tensor,
             D: float, U: float, x_mean: np.ndarray, x_scale: np.ndarray,
             disp_idx: int, vel_idx: int) -> torch.Tensor:
    """MSE Loss for predicted physical motion vs ground truth CFD trajectory."""
    h_pred_std = (h_pred / D - x_mean[disp_idx]) / x_scale[disp_idx]
    h_cfd_std = (h_cfd / D - x_mean[disp_idx]) / x_scale[disp_idx]
    hdot_pred_std = (hdot_pred / U - x_mean[vel_idx]) / x_scale[vel_idx]
    hdot_cfd_std = (hdot_cfd / U - x_mean[vel_idx]) / x_scale[vel_idx]

    return torch.mean((h_pred_std - h_cfd_std) ** 2) + torch.mean((hdot_pred_std - hdot_cfd_std) ** 2)
