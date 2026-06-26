# src/train_gru.py

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
import os

import matplotlib

from viv_analysis.utils import parse_ur_label
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from viv_analysis.config import bridge_structural_params, config, prepare_gru_config, structural_params
from viv_analysis.preprocess import merge_dataframes, compute_kinematics, downsample
from viv_analysis.models.gru import VIV_GRU, VIVSequenceDataset, apply_scalers_to_df, fit_scalers
from viv_analysis.evaluate import evaluate
from viv_analysis.utils import PROJECT_ROOT
from viv_analysis.coupled_inference import Newmark_beta

ROOT_DIR = PROJECT_ROOT


# ── Dataset-specific split definitions ────────────────────────────────────────

def _cylinder_split(cases: list[str], release_t: float) -> tuple:
    train = {"Ur3.0","Ur5.6","Ur5.0","Ur6.0","Ur6.5","Ur9.0","Ur4.6","Ur5.4", "Ur2.0","Ur3.6","Ur2.5","Ur3.8"}
    val   = {"Ur4.0","Ur4.4","Ur5.2","Ur4.8","Ur8.0","Ur3.2","Ur12.0","Ur10.0"}
    test  = {"Ur4.2","Ur5.8","Ur7.0","Ur3.4","Ur11.0"}
    rt    = {v: release_t for v in cases}
    return train, val, test, rt


def _cylinder1000_split(cases: list[str], release_t: float) -> tuple:
    test  = { "Ur11", "Ur4.75", "Ur5.5",  "Ur7"}
    val   = { "Ur4.25", "Ur5.25", "Ur6.5",  "Ur9"}
    eliminate = { "Ur2", "Ur2.5","Ur3"}
    train = set(cases) - test - val  - eliminate
    print("train (computed):", sorted(set(cases) - test - val - eliminate))

    missing = (test | val) - set(cases)
    if missing:
        print(f"WARNING: split references cases not in data: {missing}")

    rt = { c: 400.0 / parse_ur_label(c) for c in cases }

    return train, val, test, rt


def _bridge_split(cases: list[str], fn_hz: float,
                  d_ref: float, t_star_release: float) -> tuple:
    """
    Split bridge cases (60/20/20) with release time from the UDF law
    t_release = t* * D / U,  U = Ur * fn_hz * D.

    Cases MUST arrive as canonical Ur labels (merge_dataframes called with
    convert_bridge_to_ur=True). parse_ur_label raises on a raw-speed label —
    that is the loud guard against the speed-vs-Ur double-convert that would
    otherwise put every release time ~3.5x off, silently.
    """
    # hard guard: labels must be Ur, not raw speed
    for c in cases:
        try:
            parse_ur_label(c)
        except ValueError as e:
            raise ValueError(
                f"_bridge_split got non-Ur label '{c}'. Cases must be Ur-labelled "
                f"(merge_dataframes(dataset='bridge', ..., convert_bridge_to_ur=True)). "
                f"A raw-speed label here double-converts U and corrupts release times."
            ) from e

    ordered = sorted(cases, key=parse_ur_label)
    n       = len(ordered)
    n_test  = max(1, round(0.20 * n))
    n_val   = max(1, round(0.20 * n))

    # Spread test cases across the Ur range so all regimes are covered
    test_idx  = list(range(0, n, max(1, n // n_test)))[:n_test]
    remaining = [c for i, c in enumerate(ordered) if i not in test_idx]
    val_idx   = list(range(0, len(remaining),
                           max(1, len(remaining) // n_val)))[:n_val]

    test  = {ordered[i] for i in test_idx}
    val   = {remaining[i] for i in val_idx}
    train = set(cases) - test - val

    # Release time per case from the UDF law (D = d_ref = 7.42 for the bridge).
    rt = {}
    for case in cases:
        ur = parse_ur_label(case)
        U  = ur * fn_hz * d_ref               # recovers the wind speed
        if not (1.0 < U < 100.0):
            raise ValueError(
                f"Recovered U={U:.2f} m/s for '{case}' is outside the plausible "
                f"bridge sweep — likely a unit/double-convert error.")
        rt[case] = float(t_star_release * d_ref / U)
        print(f"  case={case}  Ur={ur:.4f}  U={U:.4f} m/s  t_release={rt[case]:.4f}s")
    return train, val, test, rt


def split_cases(cases: list[str], dataset: str,
                cfg: dict) -> tuple[set, set, set, dict]:
    ds = dataset.strip().lower()
    if ds == "cylinder":
        return _cylinder_split(cases, release_t=cfg["cylinder_t_release"])
    if ds in {"cylinder1000", "cylinder_re_1000", "re1000", "re1000_disp","re1000_vel","re1000_acc", "re1000_"}:
        return _cylinder1000_split(cases, release_t=cfg["cylinder1000_t_release"])
    if ds == "bridge":
        return _bridge_split(
            cases,
            fn_hz=cfg["bridge_fn_hz"],
            d_ref=cfg["bridge_D_ref"],
            t_star_release=cfg["bridge_t_star_release"],
        )
    raise ValueError(f"Unknown dataset: {dataset}")


# ── Training utilities ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, input_noise_std=0.0):
    model.train()
    total = 0.0
    for x_b, y_b, _ in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        if input_noise_std > 0.0:
            x_b = x_b.clone()
            x_b[..., :3] = x_b[..., :3] + torch.randn_like(x_b[..., :3]) * input_noise_std
        optimizer.zero_grad()
        pred, _ = model(x_b)
        loss = criterion(pred, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def train_rollout_epoch(
    model, loader, optimizer, criterion, device,
    rollout_k: int, dt: float, m: float, c: float, k_phys: float,
    idx_h: int, idx_hdot: int, idx_hddot: int, expected_features: int,
    x_mean, x_scale, y_mean, y_scale,
    rho: float, D: float, fn: float,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    x_mean_t  = torch.tensor(x_mean,  dtype=torch.float32, device=device)
    x_scale_t = torch.tensor(x_scale, dtype=torch.float32, device=device)
    total = 0.0

    for x_b, y_b, case_b in loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)
        optimizer.zero_grad()

        if rollout_k == 1:
            pred, _ = model(x_b)
            loss = criterion(pred, y_b)
        else:
            loss = _rollout_loss_k(
                model, x_b, y_b, case_b,
                rollout_k, dt, m, c, k_phys,
                idx_h, idx_hdot, idx_hddot,
                x_mean_t, x_scale_t,
                float(y_mean), float(y_scale),
                rho, D, fn, criterion, device,
            )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += loss.item()

    return total / max(len(loader), 1)


def eval_rollout_epoch(
    model, loader, criterion, device,
    rollout_k: int, dt: float, m: float, c: float, k_phys: float,
    idx_h: int, idx_hdot: int, idx_hddot: int, expected_features: int,
    x_mean, x_scale, y_mean, y_scale,
    rho: float, D: float, fn: float,
) -> float:
    model.eval()
    x_mean_t  = torch.tensor(x_mean,  dtype=torch.float32, device=device)
    x_scale_t = torch.tensor(x_scale, dtype=torch.float32, device=device)
    total = 0.0

    with torch.no_grad():
        for x_b, y_b, case_b in loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            if rollout_k == 1:
                pred, _ = model(x_b)
                loss = criterion(pred, y_b)
            else:
                loss = _rollout_loss_k(
                    model, x_b, y_b, case_b,
                    rollout_k, dt, m, c, k_phys,
                    idx_h, idx_hdot, idx_hddot,
                    x_mean_t, x_scale_t,
                    float(y_mean), float(y_scale),
                    rho, D, fn, criterion, device,
                )
            total += loss.item()

    return total / max(len(loader), 1)


def _rollout_loss_k(
    model, x_b, y_b, case_b,
    rollout_k, dt, m, c, k_phys,
    idx_h, idx_hdot, idx_hddot,
    x_mean_t, x_scale_t,
    y_mean, y_scale,
    rho, D, fn, criterion, device,
):
    """
    Multi-step rollout loss for rollout_k > 1.

    Starts from the ground-truth window, predicts CL, propagates kinematics
    via Newmark-beta (inside torch.no_grad), slides the window, and repeats
    rollout_k times. Returns the mean MSE across all k steps.

    The ODE propagation is detached from the computation graph — gradients
    flow only through each step's GRU call, not through the physics.
    """
    last = x_b[:, -1, :]
    h_c   = last[:, idx_h]     * x_scale_t[idx_h]     + x_mean_t[idx_h]
    hd_c  = last[:, idx_hdot]  * x_scale_t[idx_hdot]  + x_mean_t[idx_hdot]
    hdd_c = last[:, idx_hddot] * x_scale_t[idx_hddot] + x_mean_t[idx_hddot]

    U_vals = torch.tensor(
        [parse_ur_label(cn) * fn * D for cn in case_b],
        dtype=torch.float32, device=device,
    )

    window     = x_b.clone()
    total_loss = None

    for step in range(rollout_k):
        pred, _ = model(window)
        step_loss = criterion(pred, y_b[:, step])
        total_loss = step_loss if total_loss is None else total_loss + step_loss

        if step < rollout_k - 1:
            with torch.no_grad():
                cl_phys = pred.detach() * y_scale + y_mean
                F = 0.5 * rho * U_vals**2 * D * cl_phys

                h_c, hd_c, hdd_c = Newmark_beta(
                    F, h_c, hd_c, hdd_c, dt, m, c, k_phys,
                )

                h_s   = (h_c   - x_mean_t[idx_h])    / x_scale_t[idx_h]
                hd_s  = (hd_c  - x_mean_t[idx_hdot]) / x_scale_t[idx_hdot]
                hdd_s = (hdd_c - x_mean_t[idx_hddot]) / x_scale_t[idx_hddot]

                new_row               = window[:, -1, :].clone()
                new_row[:, idx_h]     = h_s
                new_row[:, idx_hdot]  = hd_s
                new_row[:, idx_hddot] = hdd_s
                window = torch.cat([window[:, 1:, :], new_row.unsqueeze(1)], dim=1)

    return total_loss / rollout_k


def run_validation(model, loader, criterion, device):
    model.eval()
    total, preds, trues, cases = 0.0, [], [], []
    with torch.no_grad():
        for x_b, y_b, case_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            pred, _ = model(x_b)
            total += criterion(pred, y_b).item()
            preds.append(pred.cpu().numpy())
            trues.append(y_b.cpu().numpy())
            cases.extend(case_b)
    return (
        total / len(loader),
        np.concatenate(preds),
        np.concatenate(trues),
        np.array(cases, dtype=object),
    )


def per_case_normalize(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """
    Normalise target column per case.
    Returns normalised df and dict of {case: (mean, std)} for inverse transform.
    """
    out_df = df.copy()
    stats: dict[str, tuple[float, float]] = {}
    for case, idx in out_df.groupby("case").groups.items():
        vals = out_df.loc[idx, target_col].to_numpy(dtype=np.float32)
        mu = float(vals.mean())
        sigma = float(vals.std()) + 1e-8
        out_df.loc[idx, target_col] = (vals - mu) / sigma
        stats[str(case)] = (mu, sigma)
    return out_df, stats


def inverse_per_case_scale(
    values_scaled: np.ndarray,
    case_names: np.ndarray,
    stats: dict[str, tuple[float, float]],
) -> np.ndarray:
    out = np.empty_like(values_scaled, dtype=np.float32)
    for i, (val_s, case_name) in enumerate(zip(values_scaled, case_names)):
        mu, sigma = stats[str(case_name)]
        out[i] = float(val_s) * sigma + mu
    return out


def teacher_forcing_rollout(
        model, case_df, input_cols, seq_len, release_t,
        y_scaler, case_name, device, 
        use_ur_context=False, ur_stats=None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:


    model.eval()
    ordered   = case_df.sort_values("time").reset_index(drop=True)
    signal    = ordered[input_cols].to_numpy(dtype=np.float32)

    if use_ur_context:
        ur_mean, ur_std = ur_stats if ur_stats is not None else (0.0, 1.0)
        ur_std = float(ur_std) if abs(float(ur_std)) > 0 else 1.0
        ur_val = parse_ur_label(str(case_name))
        ur_scaled = (ur_val - float(ur_mean)) / ur_std
        ur_col = np.full((signal.shape[0], 1), ur_scaled, dtype=np.float32)
        signal = np.hstack([signal, ur_col])

    cl_true_s = ordered["cl"].to_numpy(dtype=np.float32)
    times     = ordered["time"].to_numpy(dtype=np.float32)

    release_idx = int(np.searchsorted(times, release_t))
    start       = max(seq_len, release_idx + seq_len)

    preds = []
    with torch.no_grad():
        for i in range(start, len(ordered)):
            w = signal[i - seq_len : i]
            w = np.array(w, copy=True)
            x = torch.from_numpy(w).unsqueeze(0).to(device)
            p, _ = model(x)
            preds.append(p.item())

    cl_pred_s   = np.array(preds, dtype=np.float32)
    cl_true_s_s = cl_true_s[start:]
    times_s     = times[start:]

    cl_pred = y_scaler.inverse_transform(cl_pred_s.reshape(-1, 1)).ravel()
    cl_true = y_scaler.inverse_transform(cl_true_s_s.reshape(-1, 1)).ravel()
    return cl_pred, cl_true, times_s


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_tf_result(cl_pred, cl_true, times, case_name, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    ax = axes[0]
    ax.plot(times, cl_true, lw=0.8, color="black",    label="CFD (ground truth)")
    ax.plot(times, cl_pred, lw=0.8, color="tab:blue",
            alpha=0.85, label="GRU (teacher forcing)")
    ax.set_ylabel("$C_L$", fontsize=13)
    ax.set_title(f"{case_name} — Teacher Forcing Rollout  "
                 f"$R^2={r2_score(cl_true, cl_pred):.4f}$")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(times, cl_pred - cl_true, lw=0.6, color="tab:red", alpha=0.8)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel(r"Residual ($\hat{C}_L - C_L$)", fontsize=13)
    ax2.set_xlabel("Time [s]", fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.savefig(output_dir / f"ar_{case_name}.png", dpi=200)
    plt.close(fig)


def amplitude_comparison(model, all_df_s, release_time,
                          input_cols, seq_len, y_scaler,
                          device, output_dir,
                          train_cases, val_cases, test_cases,
                          use_ur_context=False, ur_stats=None):
    results = []
    for case_name, case_df in all_df_s.groupby("case"):
        ur        = parse_ur_label(str(case_name))
        release_t = release_time[case_name]
        cl_pred, cl_true, _ = teacher_forcing_rollout(
            model,
            case_df,
            input_cols,
            seq_len,
            release_t,
            y_scaler,
            case_name,
            device,
            use_ur_context=use_ur_context,
            ur_stats=ur_stats,
        )

        ss       = int(0.7 * len(cl_true))
        amp_cfd  = (cl_true[ss:].max() - cl_true[ss:].min()) / 2
        amp_gru  = (cl_pred[ss:].max() - cl_pred[ss:].min()) / 2
        amp_abs_err = abs(amp_gru - amp_cfd)
        amp_rel_err_pct = 100.0 * amp_abs_err / (abs(amp_cfd) + 1e-12)

        results.append({
            "Ur": ur,
            "CL_amp_CFD": amp_cfd,
            "CL_amp_GRU": amp_gru,
            "CL_amp_abs_error": amp_abs_err,
            "CL_amp_rel_error_pct": amp_rel_err_pct,
            "ar_r2": r2_score(cl_true, cl_pred),
            "split": ("test" if case_name in test_cases
                    else "val" if case_name in val_cases
                    else "train"),
        })
        print(
            f"  {case_name}: CFD={amp_cfd:.4f}  GRU={amp_gru:.4f}  "
            f"amp_err={amp_rel_err_pct:.2f}%  "
            f"R²={results[-1]['ar_r2']:.4f}  [{results[-1]['split']}]"
        )

    df_res = pd.DataFrame(results).sort_values("Ur")
    df_res.to_csv(output_dir / "amplitude_comparison.csv", index=False)
    return df_res


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Select dataset: python src/train_gru.py [cylinder|cylinder1000|bridge]
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cylinder"
    print(f"Dataset: {dataset}")

    OUTPUT_DIR = ROOT_DIR / "results" / f"gru_{dataset}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build dataset-specific GRU config ─────────────────────────────────
    cfg    = prepare_gru_config(dataset, config)
    ds = dataset.strip().lower()
    if ds in {"cylinder1000", "cylinder_re_1000", "re1000", "re1000_disp","re1000_vel","re1000_acc","re1000_Rollout"}:
        cfg["input_cols"] = ["disp","vel","acc"]
        cfg["use_ur_context"] = True
    elif ds == "bridge":
        cfg["input_cols"] = ["disp","vel","acc"]
        cfg["use_ur_context"] = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  seq_len={cfg['seq_len']}  "
          f"stride_train={cfg['stride_train']}")
    print(f"input feature:s: {cfg['input_cols']}  |  use_ur_context={cfg['use_ur_context']}")

    # ── Load data ──────────────────────────────────────────────────────────
    if dataset == "bridge":
        raw_df = merge_dataframes(
            dataset="bridge",
            fn_hz=cfg["bridge_fn_hz"],
            d_ref=cfg["bridge_D_ref"],
        )
        # Downsample bridge to reduce memory footprint
        raw_df = downsample(raw_df, cfg["bridge_downsample"])
    else:
        raw_df = merge_dataframes(dataset=dataset)
        ds = dataset.strip().lower()

    if raw_df.empty:
        print("No data found. Check data directories."); return

    params_Re1000 = None
    if dataset in {"cylinder1000", "cylinder_re_1000", "re1000", "re1000_disp","re1000_vel","re1000_acc"}:
        params_Re1000 = structural_params()

    params_bridge = None
    if dataset in {"bridge"}:
        params_bridge = bridge_structural_params()
    
    # debug
    print("raw_df type:", type(raw_df))
    print("raw_df head:", getattr(raw_df, "head", lambda: None)())

    if dataset in {"bridge"}:
        raw_df = compute_kinematics(raw_df, dataset=dataset, bridge_structural_params=params_bridge)
    else:
        raw_df    = compute_kinematics(raw_df, dataset=dataset, structural_params=params_Re1000)

    # ── Quarantine truncated / non-converged bridge cases ──────────────────
    if dataset == "bridge":
        sizes = raw_df.groupby("case").size().sort_values()
        print("\nPer-case length (post-downsample):")
        print(sizes.to_string())
        # Conservative default: drop cases under HALF the median length.
        # READ the print above on the first run; tune these two if a borderline
        # case should be kept/dropped, then rerun.
        BRIDGE_MIN_FRAC  = 0.05
        BRIDGE_ELIMINATE = set()      # explicit labels to force-drop, e.g. {"Ur5.48"}
        med  = float(sizes.median())
        drop = set(sizes[sizes < BRIDGE_MIN_FRAC * med].index) | \
               (BRIDGE_ELIMINATE & set(sizes.index))
        if drop:
            print(f"\nQuarantining {len(drop)} case(s) "
                  f"(< {BRIDGE_MIN_FRAC:.0%} of median {med:.0f} rows, or explicit): "
                  f"{sorted(drop)}")
            raw_df = raw_df[~raw_df['case'].isin(drop)].copy()
        else:
            print("\nNo cases quarantined.")

    all_cases = sorted(str(c) for c in raw_df["case"].drop_duplicates())
    print(f"Cases loaded: {all_cases}")

    train_cases, val_cases, test_cases, release_time = split_cases(
        all_cases, dataset, cfg)

    print(f"Train: {sorted(train_cases)}")
    print(f"Val:   {sorted(val_cases)}")
    print(f"Test:  {sorted(test_cases)}")

    train_df = raw_df[raw_df["case"].isin(train_cases)].copy()
    val_df   = raw_df[raw_df["case"].isin(val_cases)].copy()
    test_df  = raw_df[raw_df["case"].isin(test_cases)].copy()

    # ── Scale ──────────────────────────────────────────────────────────────
    # Global scaling
    x_scaler, y_scaler = fit_scalers(train_df, cfg["input_cols"], cfg["target_col"])

    print(f"x_scaler: mean={x_scaler.mean_} scale={x_scaler.scale_}")
    print(f"y_scaler ({cfg['target_col']}): "
          f"mean={float(y_scaler.mean_[0]):.6f} scale={float(y_scaler.scale_[0]):.6f}")
    
    noise_std = float(os.getenv("VIV_INPUT_NOISE", "0.0"))
    suffix = f"_noise{noise_std}" if noise_std > 0 else ""
    OUTPUT_DIR = ROOT_DIR / "results" / f"gru_{dataset}{suffix}"
    print(f"input_noise_std = {noise_std}")


    train_df_s = apply_scalers_to_df(train_df, x_scaler, y_scaler, cfg["input_cols"], cfg["target_col"])
    val_df_s   = apply_scalers_to_df(val_df,   x_scaler, y_scaler, cfg["input_cols"], cfg["target_col"])
    test_df_s  = apply_scalers_to_df(test_df,  x_scaler, y_scaler, cfg["input_cols"], cfg["target_col"])


    train_ur = np.array([parse_ur_label(c) for c in sorted(train_cases)], dtype=np.float32)

    ur_mean = float(train_ur.mean())
    ur_std  = float(train_ur.std()) + 1e-6
    use_ur_context = bool(cfg.get("use_ur_context", False))
    if use_ur_context:
        print(f"Using Ur context feature: mean={ur_mean:.4f}, std={ur_std:.4f}")


    # ── Datasets & loaders ─────────────────────────────────────────────────
    seq_len = int(cfg["seq_len"])
    target_col = str(cfg["target_col"])
    input_cols = list(cfg["input_cols"])
    stride_train = int(cfg["stride_train"])

    common = dict(seq_len=seq_len, target_col=target_col, input_cols=input_cols,
                  release_time=release_time, use_ur_context=use_ur_context,
                  ur_mean=ur_mean, ur_std=ur_std)
    
    train_ds = VIVSequenceDataset(train_df_s, stride=stride_train, **common)
    val_ds   = VIVSequenceDataset(val_df_s, stride=1, **common)
    test_ds  = VIVSequenceDataset(test_df_s, stride=1, **common)

    print(f"Dataset sizes — train: {len(train_ds)} val: {len(val_ds)} test: {len(test_ds)}")


    

    batch_size = int(cfg["batch_size"])
    default_workers = max(0, (os.cpu_count() or 4) - 2)
    num_workers = int(os.getenv("VIV_NUM_WORKERS", str(min(20, default_workers))))
    pin_memory = True
    train_loader = DataLoader( train_ds, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,batch_size=batch_size,
                                shuffle=False,num_workers=num_workers,pin_memory=pin_memory)


    # ── Model ──────────────────────────────────────────────────────────────
    input_size = len(cfg["input_cols"]) + (1 if use_ur_context else 0)
    model = VIV_GRU(
        input_size  = input_size,
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
    ).to(device)
    print(f"GRU parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_loss  = float("inf"); best_state     = None
    patience_count = 0
    train_losses, val_losses = [], []

    print(f"\nTraining for up to {cfg['n_epochs']} epochs...")
    for epoch in range(1, cfg["n_epochs"] + 1):
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device,
                     input_noise_std=noise_std)
        vl, vp, vt, _ = run_validation(model, val_loader, criterion, device)
        scheduler.step(vl)
        train_losses.append(tl); val_losses.append(vl)

        ss_res = np.sum((vt - vp) ** 2)
        ss_tot = np.sum((vt - vt.mean()) ** 2)
        val_r2 = 1 - ss_res / (ss_tot + 1e-10)

        print(f"Epoch {epoch:3d}/{cfg['n_epochs']}  "
              f"train={tl:.5f}  val={vl:.5f}  "
              f"R²={val_r2:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if vl < best_val_loss:
            best_val_loss = vl
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a best model state.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(best_state, OUTPUT_DIR / "gru_best.pt")
    print(f"Best model saved (val_loss={best_val_loss:.5f})")

    # Save artifacts
    with open(OUTPUT_DIR / "x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open(OUTPUT_DIR / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)
    with open(OUTPUT_DIR / "ur_stats.pkl", "wb") as f:
        pickle.dump({"mean": ur_mean, "std": ur_std, 
                     "use_ur_context": use_ur_context}, f)



    # ── Final evaluation ───────────────────────────────────────────────────
    _, tp, tt, tcases = run_validation(model, test_loader, criterion, device)
    _, vp, vt, vcases = run_validation(model, val_loader, criterion, device)

    test_pred = y_scaler.inverse_transform(tp.reshape(-1, 1)).ravel()
    test_true = y_scaler.inverse_transform(tt.reshape(-1, 1)).ravel()
    val_pred  = y_scaler.inverse_transform(vp.reshape(-1, 1)).ravel()
    val_true  = y_scaler.inverse_transform(vt.reshape(-1, 1)).ravel()

    val_metrics  = evaluate(val_true,  val_pred)
    test_metrics = evaluate(test_true, test_pred)
    print(f"\nValidation: {json.dumps(val_metrics, indent=2)}")
    print(f"Test:       {json.dumps(test_metrics, indent=2)}")

    # ── Teacher forcing rollout ─────────────────────────────────────────────
    print("\nTeacher forcing rollout on test cases:")
    tf_results = {}
    for case_name in sorted(test_cases):
        case_df_s = test_df_s[test_df_s["case"] == case_name].copy()
        if case_df_s.empty:
            continue
        cl_pred, cl_true, times_ar = teacher_forcing_rollout(
            model,
            case_df_s,
            cfg["input_cols"],
            seq_len,
            release_time[case_name],
            y_scaler,
            case_name,
            device,
            use_ur_context=use_ur_context,
            ur_stats=(ur_mean, ur_std),
        )

        if len(cl_true) < 20:
            print(f"  {case_name}: too short, skipping"); continue
        tf_m = evaluate(cl_true, cl_pred)
        tf_results[case_name] = tf_m
        print(f"  {case_name}: R²={tf_m['r2']:.4f}  RMSE={tf_m['rmse']:.4f}")
        plot_tf_result(cl_pred, cl_true, times_ar, case_name, OUTPUT_DIR)

    # ── Amplitude response curve ───────────────────────────────────────────
    all_df_s = pd.concat([train_df_s, val_df_s, test_df_s], ignore_index=True)
    amp_df   = amplitude_comparison(
        model, all_df_s, release_time,
        cfg["input_cols"], seq_len, y_scaler, device, OUTPUT_DIR,
        train_cases, val_cases, test_cases,
        use_ur_context=use_ur_context,
        ur_stats=(ur_mean, ur_std),
    )

    # ── Learning curve ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(train_losses, color="black",    label="train")
    ax.plot(val_losses,   color="tab:blue", label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (scaled)")
    ax.set_title(f"GRU learning curve — {dataset}")
    ax.legend()
    fig.savefig(OUTPUT_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)

    # ── Save metrics ───────────────────────────────────────────────────────
    metrics = {
        "dataset":      dataset,
        "gru_config":   {k: v for k, v in cfg.items()
                         if not callable(v)},
        "case_split":   {"train": sorted(train_cases),
                         "val":   sorted(val_cases),
                         "test":  sorted(test_cases)},
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "tf_results":   tf_results,
    }
    with open(OUTPUT_DIR / "metrics_gru.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
