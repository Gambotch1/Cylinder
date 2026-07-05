import argparse
import json
from math import gamma
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from viv_analysis.models.gru import VIV_GRU
import pickle
import matplotlib.pyplot as plt

from viv_analysis.preprocess import compute_kinematics, merge_dataframes, downsample
from viv_analysis.utils import PROJECT_ROOT, format_ur_label
from viv_analysis.self_excitation import build_seed_history, analyze_run, measure_cfd_amplitude, measure_mu_from_cfd
from viv_analysis.config import config


def Newmark_beta( F, h, h_dot, h_ddot, dt, m, c, k, beta=0.25, gamma=0.5):
    """
    Newmark-beta
    """
    # coefficients
    a1 = m / (beta * dt**2) + gamma * c / (beta * dt)
    a2 = m / (beta * dt) + (gamma / beta - 1.0) * c
    a3 = (0.5 / beta - 1.0) * m + dt * (gamma / (2.0 * beta) - 1.0) * c
    kbar = k + a1
    
    # Next time step
    pbar = F + a1 * h + a2 * h_dot + a3 * h_ddot
    h_new = pbar / kbar
    
    h_dot_new = (gamma / (beta * dt)) * (h_new - h) + (1.0 - gamma / beta) * h_dot \
                + dt * (1.0 - gamma / (2.0 * beta)) * h_ddot
    h_ddot_new = (1.0 / (beta * dt**2)) * (h_new - h) - (1.0 / (beta * dt)) * h_dot \
                 - (0.5 / beta - 1.0) * h_ddot

    return h_new, h_dot_new, h_ddot_new

def warmup_history(
    cfd_case_df,
    release_t,
    seq_len,
    input_cols,
    x_scaler,
    use_ur_context,
    ur_value,
    ur_stats,
    handoff_offset_steps: int = 0,
):
    ordered = cfd_case_df.sort_values("time").reset_index(drop=True)
    times = ordered["time"].to_numpy(dtype=np.float32)

    release_idx = int(np.searchsorted(times, release_t))

    # Standard handoff: release + seq_len.
    # Optional offset lets you test deeper warm-starts.
    handoff_idx = release_idx + seq_len + int(handoff_offset_steps)

    if handoff_idx >= len(ordered):
        raise ValueError(
            f"CFD trajectory too short: handoff_idx={handoff_idx}, "
            f"trajectory length={len(ordered)}."
        )

    win_start = handoff_idx - seq_len
    win_end = handoff_idx

    if win_start < 0:
        raise ValueError(
            f"Invalid warmup window: win_start={win_start}, seq_len={seq_len}."
        )

    win = ordered.iloc[win_start:win_end]
    if len(win) != seq_len:
        raise ValueError(
            f"Warmup window length mismatch: got {len(win)}, expected {seq_len}."
        )

    kinematics = win[input_cols].to_numpy(dtype=np.float32)
    kinematics_scaled = x_scaler.transform(kinematics)

    if use_ur_context:
        ur_mean, ur_std = ur_stats
        ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
        ur_scaled = (float(ur_value) - float(ur_mean)) / ur_std_safe
        ur_column = np.full((seq_len, 1), ur_scaled, dtype=np.float32)
        history = np.hstack([kinematics_scaled, ur_column])
    else:
        history = kinematics_scaled

    initial_state = {
        "h": float(ordered["disp"].iloc[handoff_idx]),
        "h_dot": float(ordered["vel"].iloc[handoff_idx]),
        "h_ddot": float(ordered["acc"].iloc[handoff_idx]),
    }

    handoff_time = float(times[handoff_idx])

    return history.astype(np.float32), initial_state, handoff_time, handoff_idx

def diagnostic_teacher_forcing_vs_coupled(
    model,
    x_scaler,
    y_scaler,
    case_df,
    initial_history,
    initial_state,
    handoff_idx,
    seq_len,
    input_cols,
    m,
    c,
    k,
    rho,
    U,
    D,
    B,
    dt,
    use_ur_context,
    ur_value,
    ur_stats,
    device,
    n_steps=500,
    tf_batch_size=1024,
):
    ordered = case_df.sort_values("time").reset_index(drop=True)

    if handoff_idx + n_steps >= len(ordered):
        n_steps = len(ordered) - handoff_idx - 1

    ur_mean, ur_std = ur_stats
    ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
    ur_scaled = (float(ur_value) - float(ur_mean)) / ur_std_safe

    # ------------------------------------------------------------
    # 1. Teacher forcing: true CFD kinematics → GRU → CL
    # ------------------------------------------------------------
    # Scale the whole trajectory once (StandardScaler is row-wise, so this is
    # identical to scaling each window separately) instead of re-slicing the
    # DataFrame and re-fitting/transforming per step — needed now that n_steps
    # can span the full post-release record (1e5+ steps) rather than a short window.
    kinematics_full_scaled = x_scaler.transform(
        ordered[input_cols].to_numpy(dtype=np.float32)
    )
    if use_ur_context:
        ur_col_full = np.full((len(ordered), 1), ur_scaled, dtype=np.float32)
        kinematics_full_scaled = np.hstack([kinematics_full_scaled, ur_col_full])

    # Every TF window is driven by true CFD kinematics, never by the model's
    # own output, so the n_steps windows are independent of each other and can
    # be batched through the GRU instead of run one at a time (h0 still resets
    # to zero per window — same convention VIVSequenceDataset trained on).
    # sliding_window_view is a zero-copy strided view; only the per-batch
    # slice below gets materialized, so this stays cheap even for 1e5+ steps.
    all_windows = sliding_window_view(kinematics_full_scaled, window_shape=seq_len, axis=0)
    all_windows = np.transpose(all_windows, (0, 2, 1))  # (n_windows, seq_len, n_features)

    start_k = handoff_idx - seq_len
    if start_k < 0 or start_k + n_steps > all_windows.shape[0]:
        raise ValueError(
            f"Teacher-forcing window range out of bounds: start_k={start_k}, "
            f"n_steps={n_steps}, n_windows={all_windows.shape[0]}"
        )
    tf_windows = all_windows[start_k : start_k + n_steps]

    tf_cl = np.empty(n_steps, dtype=np.float32)
    model.eval()

    with torch.no_grad():
        for start in range(0, n_steps, tf_batch_size):
            end = min(start + tf_batch_size, n_steps)
            batch = np.ascontiguousarray(tf_windows[start:end])

            x = torch.from_numpy(batch).to(device)
            cl_s, _ = model(x)  # (B,)

            cl_s = cl_s.detach().cpu().numpy().reshape(-1, 1)
            tf_cl[start:end] = y_scaler.inverse_transform(cl_s).reshape(-1)

    # ------------------------------------------------------------
    # 2. Coupled: self-generated kinematics → GRU → CL
    # ------------------------------------------------------------
    coupled = run_coupled_viv(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        seq_len=seq_len,
        input_cols=input_cols,
        initial_history=initial_history,
        initial_state=initial_state,
        m=m,
        c=c,
        k=k,
        rho=rho,
        U=U,
        D=D,
        B=B,
        dt=dt,
        n_steps=n_steps,
        use_ur_context=use_ur_context,
        ur_value=ur_value,
        ur_stats=ur_stats,
        device=device,
    )

    coupled_cl = coupled["CL"]
    coupled_h = coupled["displacement"]

    # ------------------------------------------------------------
    # 3. CFD truth
    # ------------------------------------------------------------
    cfd_cl = ordered["cl"].to_numpy(dtype=np.float32)[handoff_idx : handoff_idx + n_steps]
    cfd_h = ordered["disp"].to_numpy(dtype=np.float32)[handoff_idx : handoff_idx + n_steps]
    cfd_v = ordered["vel"].to_numpy(dtype=np.float32)[handoff_idx : handoff_idx + n_steps]
    times = ordered["time"].to_numpy(dtype=np.float32)[handoff_idx : handoff_idx + n_steps]

    def rmse(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    horizons = [10, 50, 100, 250, 500]
    horizon_rows = []

    for H in horizons:
        if H <= n_steps:
            horizon_rows.append({
                "horizon_steps": H,
                "horizon_seconds": H * dt,
                "rmse_tf_vs_cfd_cl": rmse(tf_cl[:H], cfd_cl[:H]),
                "rmse_coupled_vs_tf_cl": rmse(coupled_cl[:H], tf_cl[:H]),
                "rmse_coupled_vs_cfd_cl": rmse(coupled_cl[:H], cfd_cl[:H]),
                "max_abs_coupled_minus_tf_cl": float(np.max(np.abs(coupled_cl[:H] - tf_cl[:H]))),
                "max_abs_h_error_over_D": float(np.max(np.abs(coupled_h[:H] - cfd_h[:H])) / D),
            })

    print("\nTeacher-forcing vs coupled diagnostic")
    print("--------------------------------------")
    print(f"n_steps = {n_steps}")
    print(f"RMSE TF CL vs CFD CL       = {rmse(tf_cl, cfd_cl):.6f}")
    print(f"RMSE coupled CL vs TF CL   = {rmse(coupled_cl, tf_cl):.6f}")
    print(f"RMSE coupled CL vs CFD CL  = {rmse(coupled_cl, cfd_cl):.6f}")
    print(f"Max |coupled CL - TF CL|   = {float(np.max(np.abs(coupled_cl - tf_cl))):.6f}")
    print(f"Max |coupled h - CFD h|/D  = {float(np.max(np.abs(coupled_h - cfd_h)) / D):.6f}")

    print("\nBy horizon:")
    for row in horizon_rows:
        print(
            f"  {row['horizon_steps']:>4} steps "
            f"({row['horizon_seconds']:.3f}s): "
            f"TF-CFD CL RMSE={row['rmse_tf_vs_cfd_cl']:.5f}, "
            f"CPL-TF CL RMSE={row['rmse_coupled_vs_tf_cl']:.5f}, "
            f"max |h_err|/D={row['max_abs_h_error_over_D']:.5f}"
        )

    return {
        "time": times,
        "cfd_cl": cfd_cl,
        "tf_cl": tf_cl,
        "coupled_cl": coupled_cl,
        "cfd_h": cfd_h,
        "cfd_v": cfd_v,
        "coupled_h": coupled_h,
        "horizon_rows": horizon_rows,
        "coupled_result": coupled,
    }

def diagnostic_true_force_newmark_replay(
    case_df,
    handoff_idx,
    n_steps,
    m,
    c,
    k,
    rho,
    U,
    D,
    B=None,
    dt=None,
    force_timing: str = "current",
):
    """
    Replay the structure using true CFD CL/force instead of GRU-predicted CL.

    If this fails, the issue is Newmark/sign/force-scaling/timing,
    not the GRU.
    """
    ordered = case_df.sort_values("time").reset_index(drop=True)

    if handoff_idx + n_steps + 1 >= len(ordered):
        n_steps = len(ordered) - handoff_idx - 2

    cfd_h = ordered["disp"].to_numpy(dtype=np.float64)
    cfd_v = ordered["vel"].to_numpy(dtype=np.float64)
    cfd_a = ordered["acc"].to_numpy(dtype=np.float64)
    cfd_cl = ordered["cl"].to_numpy(dtype=np.float64)
    times = ordered["time"].to_numpy(dtype=np.float64)

    B = D if B is None else B
    if dt is None:
        raise ValueError("dt must be provided for diagnostic_true_force_newmark_replay")

    h = np.zeros(n_steps + 1, dtype=np.float64)
    v = np.zeros(n_steps + 1, dtype=np.float64)
    a = np.zeros(n_steps + 1, dtype=np.float64)

    h[0] = cfd_h[handoff_idx]
    v[0] = cfd_v[handoff_idx]
    a[0] = cfd_a[handoff_idx]

    qD = 0.5 * rho * U**2 * B

    for j in range(n_steps):
        if force_timing == "current":
            cl_used = cfd_cl[handoff_idx + j]
        elif force_timing == "next":
            cl_used = cfd_cl[handoff_idx + j + 1]
        else:
            raise ValueError("force_timing must be 'current' or 'next'")

        F = qD * cl_used

        h[j + 1], v[j + 1], a[j + 1] = Newmark_beta(
            F=F,
            h=h[j],
            h_dot=v[j],
            h_ddot=a[j],
            dt=dt,
            m=m,
            c=c,
            k=k,
        )

        if not np.isfinite([h[j+1], v[j+1], a[j+1]]).all():
            raise FloatingPointError(f"Non-finite Newmark replay at step={j}")

    # Compare h[1:] with CFD at handoff_idx+1 ... handoff_idx+n_steps
    cfd_h_cmp = cfd_h[handoff_idx + 1 : handoff_idx + n_steps + 1]
    cfd_v_cmp = cfd_v[handoff_idx + 1 : handoff_idx + n_steps + 1]
    cfd_a_cmp = cfd_a[handoff_idx + 1 : handoff_idx + n_steps + 1]
    t_cmp = times[handoff_idx + 1 : handoff_idx + n_steps + 1]

    def rmse(x, y):
        return float(np.sqrt(np.mean((np.asarray(x) - np.asarray(y)) ** 2)))

    result = {
        "time": t_cmp,
        "h_replay": h[1:],
        "v_replay": v[1:],
        "a_replay": a[1:],
        "h_cfd": cfd_h_cmp,
        "v_cfd": cfd_v_cmp,
        "a_cfd": cfd_a_cmp,
        "rmse_h": rmse(h[1:], cfd_h_cmp),
        "rmse_v": rmse(v[1:], cfd_v_cmp),
        "rmse_a": rmse(a[1:], cfd_a_cmp),
        "max_abs_h_over_D": float(np.max(np.abs(h[1:] - cfd_h_cmp)) / D),
        "force_timing": force_timing,
    }

    print(f"\nTrue-force Newmark replay [{force_timing}]")
    print("-----------------------------------------")
    print(f"RMSE h       = {result['rmse_h']:.6e} m")
    print(f"RMSE v       = {result['rmse_v']:.6e} m/s")
    print(f"RMSE a       = {result['rmse_a']:.6e} m/s²")
    print(f"Max |h err|/D= {result['max_abs_h_over_D']:.6f}")

    return result


def run_coupled_viv(
    model:        VIV_GRU,
    x_scaler,
    y_scaler,
    seq_len:      int,
    input_cols:   list[str],
    initial_history: np.ndarray,
    initial_state: dict,
    # Structural parameters
    m:            float,   # mass per unit span [kg/m]
    c:            float,   # damping coefficient [N·s/m]
    k:            float,   # stiffness [N/m]
    # Flow parameters
    rho:          float,   # fluid density [kg/m³]
    U:            float,   # freestream velocity [m/s]
    D:            float,   # reference depth [m]
    n_steps:      int,     # total timesteps to simulate
    B:          float = None,   # reference span [m]
    dt:           float = None,   # timestep [s]
    use_ur_context: bool = False,
    ur_value:     float = 0.0,
    ur_stats:     tuple   = (0.0, 1.0),
    device:       str = "cpu",
    e_forcing:    np.ndarray = None,  # additive residual forcing on CL, shape (n_steps,)
    gru_off:      bool = False,       # if True, zero GRU lift (forcing-only null control)
    # Optional closure / forcing parameters
    forcing_mode: str = "v1_additive",
    fn: Optional[float] = None,
    a_ref: Optional[float] = None,
    mu: float = 0.0,
) -> dict:
    """
    Fully coupled GRU-structural VIV simulation.
    
    No CFD required. The GRU predicts CL from kinematics,
    which drives the structural equation of motion,
    which produces new kinematics for the next GRU prediction.
    
    This is the 2-way FSI loop described in the thesis.
    """
    model.eval()
    B = D if B is None else B
    if dt is None:
        raise ValueError("dt must be provided for run_coupled_viv")

    # State vectors
    h      = np.zeros(n_steps + 1, dtype=np.float32)  # displacement
    h_dot  = np.zeros(n_steps + 1, dtype=np.float32)  # velocity
    h_ddot = np.zeros(n_steps + 1, dtype=np.float32) # acceleration
    CL     = np.zeros(n_steps + 1, dtype=np.float32)  # lift coefficient


    h[0]      = initial_state["h"]
    h_dot[0]  = initial_state["h_dot"]
    h_ddot[0] = initial_state["h_ddot"]
    history   = initial_history.copy()  

    expected_features = len(input_cols) + (1 if use_ur_context else 0)
    if history.ndim != 2 or history.shape[1] != expected_features:
        raise ValueError(
            f"History feature mismatch: got {history.shape}, "
            f"expected (*, {expected_features})."
        )

    max_abs_z_seen = 0.0
    n_ood_warnings = 0

    if e_forcing is None or len(e_forcing) == 0:
        _e = np.zeros(n_steps, dtype=np.float32)
    else:
        _e = np.asarray(e_forcing, dtype=np.float32)
        if len(_e) < n_steps:
            raise ValueError(
                f"e_forcing length {len(_e)} < n_steps {n_steps}; pad or tile before passing."
            )
        _e = _e[:n_steps]

    # Use CFD warm-start history, then update it with coupled predictions.
    ur_mean, ur_std = ur_stats
    ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
    ur_scaled = (ur_value - float(ur_mean)) / ur_std_safe

    # Precompute scaling math once: StandardScaler.transform/inverse_transform
    # do input validation on every call, which is real overhead at n_steps~1e5.
    # (x - mean) / scale and (z * scale + mean) are exactly what sklearn does
    # internally, just without the per-call validation cost.
    x_mean = x_scaler.mean_.astype(np.float32)
    x_scale = x_scaler.scale_.astype(np.float32)
    y_mean = float(y_scaler.mean_[0])
    y_scale = float(y_scaler.scale_[0])

    # ── Closure / forcing-mode setup ───────────────────────────────────
    # Validate required inputs for non-default forcing modes and precompute
    # oscillator frequency used by v2/v3 laws.
    if forcing_mode != "v1_additive":
        if fn is None or a_ref is None or a_ref <= 0:
            raise ValueError(
                f"forcing_mode={forcing_mode} requires fn and a_ref>0 "
                f"(got fn={fn}, a_ref={a_ref})")
    omega_n = 2.0 * np.pi * float(fn) if fn is not None else None

    # Component logging arrays (for decomposition/diagnostics)
    CL_det_arr = np.zeros(n_steps, dtype=np.float32)
    CL_vdp_arr = np.zeros(n_steps, dtype=np.float32)

    with torch.inference_mode():
        for i in range(n_steps):

            x = torch.from_numpy(history).unsqueeze(0).to(device)
            cl_scaled, _ = model(x)
            cl_det = 0.0 if gru_off else float(cl_scaled.item()) * y_scale + y_mean

            # Forcing law selection:
            # - v1_additive: additive residual forcing (legacy)
            # - v2_multiplicative: amplitude-gained noise: (a_env/a_ref)*_e[i]
            # - v3_coherent: VdP-like coherent closure term
            cl_vdp = 0.0
            if forcing_mode == "v3_coherent":
                a_env = np.sqrt(h[i] ** 2 + (h_dot[i] / omega_n) ** 2)
                cl_vdp = mu * (1.0 - (a_env / a_ref) ** 2) * (h_dot[i] / (omega_n * a_ref))
                cl = cl_det + cl_vdp + _e[i]
            elif forcing_mode == "v2_multiplicative":
                a_env = np.sqrt(h[i] ** 2 + (h_dot[i] / omega_n) ** 2)
                cl = cl_det + (a_env / a_ref) * _e[i]
            else:  # v1_additive
                cl = cl_det + _e[i]

            CL_det_arr[i] = cl_det
            CL_vdp_arr[i] = cl_vdp
            CL[i] = cl

            # ── Step 2: compute aerodynamic force ─────────────────────
            F_aero = 0.5 * rho * U**2 * B * cl

            h[i+1], h_dot[i+1], h_ddot[i+1] = Newmark_beta(
                F=F_aero,
                h=h[i],
                h_dot=h_dot[i],
                h_ddot=h_ddot[i],
                dt=dt,
                m=m,
                c=c,
                k=k,
            )
            # ── Step 4: update kinematic history window ────────────────────
            # Push h[i] (the state that *drove* this step) so that at the next
            # iteration the window ends at i, matching the training convention:
            #   predict CL[i+1] from kinematics [..., h[i]].
            new_kinematics_raw = np.array([h[i], h_dot[i], h_ddot[i]], dtype=np.float32)
            new_kinematics_scaled = (new_kinematics_raw - x_mean) / x_scale

            if use_ur_context:
                new_row = np.append(new_kinematics_scaled, ur_scaled).astype(np.float32)
            else:
                new_row = new_kinematics_scaled.astype(np.float32)

            if not np.isfinite([h[i+1], h_dot[i+1], h_ddot[i+1], cl]).all():
                raise FloatingPointError(
                    f"Non-finite state at step={i}: "
                    f"h={h[i+1]}, v={h_dot[i+1]}, a={h_ddot[i+1]}, CL={cl}"
                )

            zmax = float(max(abs(v) for v in new_kinematics_scaled))
            max_abs_z_seen = max(max_abs_z_seen, zmax)

            if zmax > 6.0 and n_ood_warnings < 10:
                n_ood_warnings += 1
                print(
                    f"WARNING OOD step={i}: max|z|={zmax:.2f}, "
                    f"h={h[i]:.4e}, v={h_dot[i]:.4e}, "
                    f"a={h_ddot[i]:.4e}, CL={cl:.4e}"
                )

            history = np.roll(history, -1, axis=0)
            history[-1] = new_row


    t = np.arange(n_steps + 1) * dt

    return {
        "time": t[:n_steps],
        "displacement": h[:n_steps],
        "velocity": h_dot[:n_steps],
        "acceleration": h_ddot[:n_steps],
        "CL": CL[:n_steps],
        "CL_det": CL_det_arr,
        "CL_vdp": CL_vdp_arr,
        "e_forcing": _e[:n_steps].copy(),
        "max_abs_scaled_kinematics": float(max_abs_z_seen),
        "n_ood_warnings": int(n_ood_warnings),
    }




def main(
    Ur: float = 6.0, # reduced velocity to simulate
    cfd_dataset: str = "cylinder1000", # dataset name for loading CFD data
    model_dataset: str = "Cylinder1000", # dataset name for loading model artifacts
    total_time: float = 700.0, # total simulation time in seconds
    checkpoint: str = "gru_best.pt",
    model_subdir: Optional[str] = None,
    handoff_offset_steps: int = 2000,
    residual_npz: Optional[str] = None,
    noise_mode: str = "none",
    noise_scale: float = 1.0,
    noise_seed: int = 0,
    gru_off: bool = False,
    ad_seed: Optional[float] = None,
    cfd_target_AD: float = 0.23,
    forcing_mode: str = "v1_additive",
    mu: Optional[float] = None,
    make_tf_residual: bool = False,
    run_replay_diag: bool = False,
    ):

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    if model_subdir is not None:
        artifact_dir = PROJECT_ROOT / "results" / model_subdir
    else:
        artifact_dir = PROJECT_ROOT / "results" / f"gru_{model_dataset}"
        artifact_dir_base = PROJECT_ROOT / "results" / f"gru_{cfd_dataset}"
    

    # ── Load model + scalers + ur_stats ───────────────────────────────────
    with open(artifact_dir / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(artifact_dir / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    if model_subdir is not None:
        with open(artifact_dir / "ur_stats.pkl", "rb") as f:
            ur_info = pickle.load(f)
    else:
        with open(artifact_dir_base / "ur_stats.pkl", "rb") as f:
            ur_info = pickle.load(f)



    # ── Ur statistics from training cases ─────────────────────────────────
    use_ur_context = bool(ur_info["use_ur_context"])
    ur_mean   = float(ur_info["mean"])
    ur_std    = float(ur_info["std"]) + 1e-8
    print(f"Loaded scalers and UR stats: use_ur_context={use_ur_context}"  
          f"mean={ur_mean:.4f}, std={ur_std:.4f}")
    print(f"y_scaler: mean={float(y_scaler.mean_[0]):.6f}  "
          f"scale={float(y_scaler.scale_[0]):.6f}")


    # ── Physical parameters — must match UDF exactly ───────────────────────
    ds = cfd_dataset.strip().lower()
    params_Re1000 = None; bsp = None
    if ds == "bridge":
        from viv_analysis.config import bridge_structural_params
        bsp = bridge_structural_params()
        m, c, k = bsp["m"], bsp["c"], bsp["k"]
        rho = config["bridge_rho"]
        fn  = config["bridge_fn_hz"]
        D   = config["bridge_D_ref"]
        B   = config["bridge_B_ref"]
        t_star_release = config["bridge_t_star_release"]
        dt  = None
    else:
        rho = 1.0
        D   = config['cylinder1000_D_ref']
        B   = D
        fn  = 0.2
        M_star, zeta = 2.0, 0.007
        m = M_star * rho * (np.pi * D**2 / 4.0)
        k = m * (2*np.pi*fn)**2
        c = 2.0 * m * (2*np.pi*fn) * zeta
        t_star_release = 80.0
        dt = 0.005
        params_Re1000 = {"m": m, "c": c, "k": k,
                         "cylinder_mass": m, "c_struct": c, "k_struct": k}

    U = Ur * fn * D
    t_release = t_star_release * D / U
    print(f"[{ds}] Ur={Ur}  U={U:.4f} m/s  fn={fn}  D(Ur,rel,h/D)={D}  B(force)={B}")
    print(f"  m={m:.6e}  c={c:.6e}  k={k:.6e}  rho={rho}  t_release={t_release:.4f}s")


    # ── Build model (read hidden_size from saved metrics) ─────────────────
    input_cols = ["disp", "vel", "acc"]
    input_size = len(input_cols) + (1 if use_ur_context else 0)
    hidden_size, num_layers = 64, 2
    if model_subdir is not None:
        metrics_path = artifact_dir / "metrics_gru.json"
    else:
        metrics_path = artifact_dir_base / "metrics_gru.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            saved_metrics = json.load(f)
        hidden_size = saved_metrics["gru_config"].get("hidden_size", hidden_size)
        num_layers  = saved_metrics["gru_config"].get("num_layers", num_layers)
        seq_len = saved_metrics["gru_config"].get("seq_len", 1000)


    # ── Load CFD trajectory at this Ur ────────────────────────────────────
    print(f"\nLoading CFD trajectory at Ur={Ur} for warm-start...")
    if ds == "bridge":
        from viv_analysis.preprocess import load_bridge_df_cached

        raw_df = load_bridge_df_cached(
            fn_hz=fn,
            d_ref=D,
            bridge_structural_params=bsp,
        )
    else:
        raw_df = merge_dataframes(dataset=cfd_dataset)
        if raw_df.empty:
            raise RuntimeError("Could not load CFD data for warmup.")
        raw_df = compute_kinematics(
            raw_df,
            dataset=cfd_dataset,
            structural_params=params_Re1000,
        )
    if raw_df.empty:
        raise RuntimeError("Could not load CFD data for warmup.")

    case_label = format_ur_label(Ur)
    case_df = raw_df[raw_df["case"] == case_label].copy()
    if case_df.empty:
        available_cases = sorted(raw_df["case"].unique().tolist())
        raise ValueError(f"No CFD data found for case '{case_label}'."
                        f"Available cases: {available_cases}")
    print(f"  CFD trajectory has {len(case_df)} steps")

    if dt is None:
        tt = np.sort(np.unique(case_df["time"].to_numpy(dtype=np.float64)))
        dt = float(np.median(np.diff(tt)))
        print(f"  [bridge] Newmark dt = {dt:.6f}s  ({(1/fn)/dt:.0f} steps/cycle)")
 
    
    initial_history, initial_state, t_handoff, handoff_idx = warmup_history(
        cfd_case_df=case_df,
        release_t=t_release,
        seq_len=seq_len,
        input_cols=input_cols,
        x_scaler=x_scaler,
        use_ur_context=use_ur_context,
        ur_value=Ur,
        ur_stats=(ur_mean, ur_std),
        handoff_offset_steps=handoff_offset_steps,

        
    )
    print(f"  Warm-start window: CFD steps "
          f"{handoff_idx}]  (post-release)")
    print(f"  Handoff at t={t_handoff:.4f}s   "
          f"({seq_len*dt:.2f}s after release)")
    print(f"  Initial state at handoff: "
          f"h={initial_state['h']:.6f}  h_dot={initial_state['h_dot']:.6f}  "
          f"h_ddot={initial_state['h_ddot']:.6f}")
    

        # Sanity check: warm-start kinematics should NOT all be zero
    raw_kin = case_df.iloc[handoff_idx - seq_len : handoff_idx][input_cols].to_numpy()
    print(f"  Warm-start raw stats:")
    print(f"    disp range: [{raw_kin[:,0].min():.5f}, {raw_kin[:,0].max():.5f}]")
    print(f"    vel  range: [{raw_kin[:,1].min():.5f}, {raw_kin[:,1].max():.5f}]")
    print(f"    acc  range: [{raw_kin[:,2].min():.5f}, {raw_kin[:,2].max():.5f}]")





    model = VIV_GRU(input_size=input_size, hidden_size=hidden_size, 
                    num_layers=num_layers, dropout=0.1).to(device)
    
    # Load checkpoint: handle both training checkpoints (with 'model' key) and raw state_dicts
    ckpt = torch.load(artifact_dir / checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    print(f"Model loaded: input_size={input_size}  hidden_size={hidden_size}")

    # ── Attractor test: seed with artificial A/D ───────────────────────
    if ad_seed is not None:
        print(f"\n[ATTRACTOR TEST] Seeding loop with artificial A/D = {ad_seed}")
        initial_history, initial_state = build_seed_history(
            ad_seed=ad_seed, seq_len=seq_len, dt=dt, D=D, fn=fn,
            x_scaler=x_scaler, use_ur_context=use_ur_context,
            ur_value=Ur, ur_stats=(ur_mean, ur_std),
        )
        t_handoff = 0.0
        n_steps = int(total_time / dt)

        result = run_coupled_viv(
            model=model, x_scaler=x_scaler, y_scaler=y_scaler,
            initial_history=initial_history, initial_state=initial_state,
            seq_len=seq_len, input_cols=input_cols,
            m=m, c=c, k=k, rho=rho, U=U, D=D, B=B, dt=dt, n_steps=n_steps,
            use_ur_context=use_ur_context, ur_value=Ur, ur_stats=(ur_mean, ur_std),
            device=device, e_forcing=None, gru_off=False,
        )

        h = result["displacement"]
        diag = analyze_run(h, D=D, dt=dt, fn=fn, ad_cfd_ref=cfd_target_AD)
        print(f"[ATTRACTOR] seed A/D={ad_seed}  final A/D={diag['ad_final']:.4f}  "
              f"target={cfd_target_AD}  f_dom={diag['f_dominant']:.3f}  "
              f"converged={diag['converged']}")

        # save envelope for the two-sided figure
        np.savez(PROJECT_ROOT / "results" / f"attractor_Ur{Ur}_seed{ad_seed}.npz",
                 t=result["time"], h=h, cl=result["CL"], D=D, Ur=float(Ur),
                 ad_seed=ad_seed, ad_final=diag["ad_final"],
                 env_t=diag["env_t"], env_ad=diag["env_ad"],
                 cfd_target_AD=cfd_target_AD)
        return

    # Guards 
    expected_features = len(input_cols) + (1 if use_ur_context else 0)

    if initial_history.shape != (seq_len, expected_features):
        raise ValueError(
            f"Initial history shape mismatch: got {initial_history.shape}, "
            f"expected ({seq_len}, {expected_features})."
        )

    print(
        f"Model/input check: input_size={input_size}, "
        f"history_shape={initial_history.shape}"
    )

    # Run TF-vs-coupled over the full post-release CFD record (not a short
    # window) so a Welch PSD on the dumped arrays has the resolution to
    # separate the lock-in peak from the broadband floor.
    n_steps_tf = len(case_df) - handoff_idx - 1
    print(f"\nTF-vs-coupled diagnostic over full post-release record: "
          f"{n_steps_tf} steps ({n_steps_tf * dt:.1f}s)")

    diag = None
    if make_tf_residual:
        diag = diagnostic_teacher_forcing_vs_coupled(
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            case_df=case_df,
            initial_history=initial_history,
            initial_state=initial_state,
            handoff_idx=handoff_idx,
            seq_len=seq_len,
            input_cols=input_cols,
            m=m,
            c=c,
            k=k,
            rho=rho,
            U=U,
            D=D,
            B=B,
            dt=dt,
            use_ur_context=use_ur_context,
            ur_value=Ur,
            ur_stats=(ur_mean, ur_std),
            device=device,
            n_steps=n_steps_tf,
            tf_batch_size=1024,
        )

        tf_residual_path = PROJECT_ROOT / "results" / f"tf_residual_Ur{Ur}.npz"
        np.savez(
            tf_residual_path,
            cl_true=diag["cfd_cl"],
            cl_tf=diag["tf_cl"],
            vel=diag["cfd_v"],
            dt=dt,
            fn=fn,
        )
        print(f"Saved TF residual arrays to {tf_residual_path}")

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

        axes[0].plot(diag["time"], diag["cfd_cl"], color="black", lw=0.8, label="CFD CL")
        axes[0].plot(diag["time"], diag["tf_cl"], color="tab:blue", lw=0.8, label="GRU teacher forcing")
        axes[0].plot(diag["time"], diag["coupled_cl"], color="tab:orange", lw=0.8, label="GRU coupled")
        axes[0].set_ylabel("$C_L$")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(diag["time"], diag["cfd_h"] / D, color="black", lw=0.8, label="CFD h/D")
        axes[1].plot(diag["time"], diag["coupled_h"] / D, color="tab:green", lw=0.8, label="Coupled h/D")
        axes[1].set_ylabel("$h/D$")
        axes[1].set_xlabel("Time [s]")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        diag_png = PROJECT_ROOT / "results" / f"diagnostic_tf_vs_coupled_Ur_{Ur}_{checkpoint}_handoff_{handoff_offset_steps}_{model_subdir}.png"
        fig.savefig(diag_png, dpi=150)
        plt.close(fig)
        print(f"Saved diagnostic plot to {diag_png}_{checkpoint}")
    else:
        print("[info] --make_tf_residual not set; skipping TF diagnostic (diag plot and residual generation)")

    if run_replay_diag:
        replay_current = diagnostic_true_force_newmark_replay(
            case_df=case_df,
            handoff_idx=handoff_idx,
            n_steps=5000,
            m=m,
            c=c,
            k=k,
            rho=rho,
            U=U,
            D=D,
            B=B,
            dt=dt,
            force_timing="current",
        )

        replay_next = diagnostic_true_force_newmark_replay(
            case_df=case_df,
            handoff_idx=handoff_idx,
            n_steps=5000,
            m=m,
            c=c,
            k=k,
            rho=rho,
            U=U,
            D=D,
            B=B,
            dt=dt,
            force_timing="next",
        )

        best_replay = (
            replay_current
            if replay_current["max_abs_h_over_D"] <= replay_next["max_abs_h_over_D"]
            else replay_next
        )

        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        ax.plot(best_replay["time"], best_replay["h_cfd"] / D, color="black", lw=0.8, label="CFD h/D")
        ax.plot(best_replay["time"], best_replay["h_replay"] / D, color="tab:purple", lw=0.8, label=f"Newmark replay ({best_replay['force_timing']})")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("$h/D$")
        ax.legend()
        ax.grid(True, alpha=0.3)

        replay_png = PROJECT_ROOT / "results" / f"diagnostic_newmark_replay_Ur_{Ur}_{checkpoint}_handoff_{handoff_offset_steps}_{model_subdir}.png"
        fig.savefig(replay_png, dpi=150)
        plt.close(fig)

        print(f"Saved Newmark replay diagnostic to {replay_png}_{checkpoint}")
    else:
        print("[info] --run_replay_diag not set; skipping Newmark replay diagnostic")

    # ── Run coupled inference ─────────────────────────────────────────────
    n_steps = int((total_time - t_handoff) / dt)
    if n_steps <= 0:
        raise ValueError(
            f"total_time={total_time}s is before handoff t={t_handoff:.2f}s "
            f"(total_time is absolute sim end-time, not duration); need > {t_handoff:.1f}"
        )
    print(f"\nRunning coupled inference for {total_time}s ({n_steps} steps)...")
    print(f"{n_steps} steps) ...")

    # ── Precompute stochastic forcing vector ──────────────────────────────
    e_forcing = np.zeros(n_steps, dtype=np.float32)
    if noise_mode != "none" and residual_npz is not None:
        from viv_analysis.residual_forcing import make_forcing
        d_npz = np.load(residual_npz)
        if "Ur" in d_npz and abs(float(d_npz["Ur"]) - Ur) > 1e-6:
            raise ValueError(f"residual_npz is Ur={float(d_npz['Ur'])} but run is Ur={Ur}")
        resid = np.asarray(d_npz["cl_true"], float) - np.asarray(d_npz["cl_tf"], float)
        skip = int(5.0 / float(d_npz["dt"]))          # match the diagnostic's --skip_s 5
        resid = resid[skip:]
        e_forcing = make_forcing(resid, n_steps, mode=noise_mode, scale=noise_scale, seed=noise_seed)
        print(f"[stochastic] mode={noise_mode} scale={noise_scale} "
              f"resid_std={resid.std():.4f} e_std={e_forcing.std():.4f}")
    # Closure / coherent forcing parameters: measure CFD amplitude and (optionally)
    # derive the negative-damping mu from the CFD growth transient for v3_coherent.
    qD = 0.5 * rho * U**2 * B
    amp = measure_cfd_amplitude(case_df, handoff_idx, D)
    a_ref = amp["a_ref_peak"]                       # PEAK physical amplitude
    print(f"[closure] CFD RMS A/D={amp['rms_AD']:.4f}  a_ref(peak)={a_ref:.4f} m")

    mu_used = mu
    if forcing_mode == "v3_coherent" and mu is None:
        mm = measure_mu_from_cfd(case_df, t_release, m, c, D, dt, fn, a_ref, qD)
        mu_used = mm["mu"]
        print(f"[closure] measured growth lambda={mm['lam']:.4f}/s (R2={mm['r2']:.3f}, "
              f"n={mm['n']}, t=[{mm.get('t0',0):.0f},{mm.get('t1',0):.0f}]s) "
              f"-> mu={mu_used:.4f}")
        if mm["r2"] < 0.9:
            print(f"  [warn] growth fit R2={mm['r2']:.2f} < 0.9 -- transient not cleanly "
                  f"exponential; mu is unreliable, inspect the envelope before trusting v3.")

    result = run_coupled_viv(
        model        = model,
        x_scaler     = x_scaler,
        y_scaler     = y_scaler,
        initial_history = initial_history,
        initial_state = initial_state,
        seq_len      = seq_len,
        input_cols   = input_cols,
        m            = m,
        c            = c,
        k            = k,
        rho          = rho,
        U            = U,
        D            = D,
        B            = B,
        dt           = dt,
        n_steps      = n_steps,
        device       = device,
        use_ur_context = use_ur_context,
        ur_value     = Ur,
        ur_stats     = (ur_mean, ur_std),
        e_forcing    = e_forcing,
        gru_off      = gru_off,
        forcing_mode = forcing_mode,
        fn = fn,
        a_ref = a_ref,
        mu = (mu_used or 0.0),
    )

    # ── Plot ───────────────────────────────────────────────────────────────
    t    = result["time"] + t_handoff
    h    = result["displacement"]
    CL   = result["CL"]
    FL   = 0.5 * rho * U**2 * B * CL

    # ── Save coupled trajectory for harness ───────────────────────────────
    h_cfd_tail = case_df.sort_values("time")["disp"].to_numpy(dtype=np.float32)
    h_cfd_tail = h_cfd_tail[handoff_idx : handoff_idx + n_steps]
    # canonical subdir name (use provided subdir or fallback to model_dataset)
    _sub = model_subdir or f"gru_{model_dataset}"
    # short tags to ensure filenames are unique per experimental factors
    _gru_tag = "_gruoff" if gru_off else ""
    _noise_scale_tag = f"s{noise_scale:.6g}"
    _exp_tag = f"{_sub}_forc-{forcing_mode}_noise-{noise_mode}{_gru_tag}_{_noise_scale_tag}_seed{noise_seed}"
    npz_out = PROJECT_ROOT / "results" / f"coupled_{cfd_dataset}_Ur{Ur}_{_exp_tag}_mu{mu_used:.4g}.npz"
    np.savez(npz_out, t=t, h=h, cl=CL, h_cfd=h_cfd_tail, D=D, Ur=float(Ur),
             model_subdir=_sub, forcing_mode=forcing_mode, noise_mode=noise_mode,
             gru_off=bool(gru_off), noise_scale=float(noise_scale), noise_seed=int(noise_seed),
             cl_det=result.get("CL_det"), cl_vdp=result.get("CL_vdp"), e=result.get("e_forcing"))
    print(f"Saved coupled trajectory -> {npz_out}")

    # ── Plot: GRU result alongside CFD ground truth for comparison ────────
    CFD_t = case_df["time"].values
    CFD_h = case_df["disp"].values
    CFD_cl = case_df["cl"].values


    fig, axes = plt.subplots(4, 1, figsize=(13, 12), constrained_layout=True)

    axes[0].plot(CFD_t, CFD_h / D, lw=0.8, color="black", alpha=0.7, label="CFD")
    axes[0].plot(t, h / D, lw=1, color="tab:blue", alpha=0.9, label="GRU coupled")
    axes[0].axvline(t_handoff, color="green", ls="--", lw=1, alpha=0.7,
                    label=f"handoff t={t_handoff:.1f}s")
    axes[0].axhline(0, color="gray", lw=0.5, ls=":")
    axes[0].set_ylabel(r"$h/D$")
    axes[0].set_title(f"Coupled GRU-Structural VIV  —  $U_r={Ur}_{_exp_tag}$ (post-release warm-start)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
 
    axes[1].plot(CFD_t, CFD_cl, lw=0.8, color="black", alpha=0.7, label="CFD")
    axes[1].plot(t, CL, lw=1, color="tab:orange", alpha=0.9, label="GRU coupled")
    axes[1].axvline(t_handoff, color="green", ls="--", lw=1, alpha=0.7)
    axes[1].axhline(0, color="gray", lw=0.5, ls=":")
    axes[1].set_ylabel("$C_L$")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
 
    axes[2].plot(t, FL, lw=1, color="tab:red")
    axes[2].axhline(0, color="gray", lw=0.5, ls=":")
    axes[2].set_ylabel("Lift force [N/m]")
    axes[2].grid(True, alpha=0.3)
 
    axes[3].plot(t, result["acceleration"], lw=1, color="tab:green")
    axes[3].set_ylabel("Acc [m/s²]")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)


    out_png = PROJECT_ROOT / "results" / f"coupled_viv_Ur_{Ur}_{checkpoint}_offset_{handoff_offset_steps}_{_exp_tag}_mu{mu_used:.4g}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved coupled VIV plot to {out_png}")

    # ── Steady-state amplitude ─────────────────────────────────────────────
    ss_start = int(0.7 * len(h))
    amp      = (h[ss_start:].max() - h[ss_start:].min()) / (2 * D)
    print(f"\nSteady-state A/D = {amp:.4f}")
    print(f"Displacement range: {h.min():.4f} to {h.max():.4f} m")
    print(f"CL range: {CL.min():.4f} to {CL.max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ur",type=float, default=6.0)
    parser.add_argument("--total_time", type=float, default=500.0)
    parser.add_argument("--cfd_dataset", type=str, default="cylinder1000",
                    help="Dataset key for loading CFD data")
    parser.add_argument("--model_dataset", type=str, default="cylinder_re_1000",
                    help="Dataset key for loading model artifacts")
    parser.add_argument("--checkpoint", type=str, default="gru_best.pt",
                    help="Checkpoint filename within model artifact_dir")
    parser.add_argument("--model_subdir", type=str, default=None,
                    help="Override: full subdir name like 'gru_rollout_cylinder_re_1000'")
    parser.add_argument("--handoff_offset", type=int, default=2000,
                    help="CFD steps past (release+seq_len) for handoff. "
                         "Default 2000 = existing sweep; vary for noise floor.")
    parser.add_argument("--residual_npz", default=None,
                    help="npz with cl_true, cl_tf (the TF-residual you measured)")
    parser.add_argument("--noise_mode", default="none",
                    choices=["surrogate", "replay", "white", "none"])
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--gru_off", action="store_true",
                    help="zero the GRU lift; forcing only (resonance null control)")
    parser.add_argument("--ad_seed", type=float, default=None, 
                    help="Run self-excitation attractor test. Specify initial A/D (e.g., 0.1 or 0.3)")
    parser.add_argument("--cfd_target_AD", type=float, default=0.23, 
                    help="The expected true limit-cycle A/D for convergence checking")
    parser.add_argument("--forcing_mode", default="v1_additive",
                    choices=["v1_additive", "v2_multiplicative", "v3_coherent"])
    parser.add_argument("--mu", type=float, default=None,
                    help="v3 negative-damping strength. If omitted, MEASURED from CFD growth.")
    parser.add_argument("--make_tf_residual", action="store_true",
                    help="Regenerate TF residual and diagnostic plot. Run once per Ur. Required for new Ur with --noise_mode surrogate.")
    parser.add_argument("--run_replay_diag", action="store_true",
                    help="Run Newmark replay diagnostic (oracle for force/timing validation). Safe to run on demand.")
    args = parser.parse_args()
    main(
        Ur=args.Ur,
        total_time=args.total_time,
        cfd_dataset=args.cfd_dataset,
        model_dataset=args.model_dataset,
        checkpoint=args.checkpoint,
        model_subdir=args.model_subdir,
        handoff_offset_steps=args.handoff_offset,
        residual_npz=args.residual_npz,
        noise_mode=args.noise_mode,
        noise_scale=args.noise_scale,
        noise_seed=args.noise_seed,
        gru_off=args.gru_off,
        ad_seed=args.ad_seed,
        cfd_target_AD=args.cfd_target_AD,
        forcing_mode=args.forcing_mode,
        mu=args.mu,
        make_tf_residual=args.make_tf_residual,
        run_replay_diag=args.run_replay_diag,
    )
    
