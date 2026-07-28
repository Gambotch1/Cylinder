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

ROOT_DIR = PROJECT_ROOT

import argparse
import random


# ── Helper functions ───────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Seed workers in DataLoader when num_workers > 0."""
    seed = int(torch.initial_seed() % 2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def apply_nd_transform(df: pd.DataFrame, nd_inputs: bool, D: float, fn: float,
                       input_cols: list[str]) -> pd.DataFrame:
    """
    Apply nondimensional transform to physical kinematics.
    
    For each case, compute case-dependent U and apply:
        disp_model = disp / D
        vel_model = vel / U
        acc_model = acc / (U^2 / D)
    
    Returns a transformed copy; physical (h, h_dot, h_ddot) and metadata remain.
    """
    if not nd_inputs:
        return df.copy()
    
    if D <= 0 or fn <= 0:
        raise ValueError(f"apply_nd_transform requires D>0, fn>0 (got D={D}, fn={fn})")
    
    df_out = df.copy()
    
    for case_name, case_df_idx in df_out.groupby("case", sort=False).groups.items():
        ur = parse_ur_label(str(case_name))
        U_case = float(ur * fn * D)
        
        if not (U_case > 0 and np.isfinite(U_case)):
            raise ValueError(
                f"apply_nd_transform: invalid U_case={U_case} for case={case_name}, "
                f"Ur={ur}, fn={fn}, D={D}")
        
        # Divisors for each input column
        divisors = np.array([D, U_case, U_case**2 / D], dtype=np.float32)
        
        for col_idx, col_name in enumerate(input_cols):
            if col_name in ["disp", "vel", "acc"]:
                df_out.loc[case_df_idx, col_name] = (
                    df_out.loc[case_df_idx, col_name].astype(np.float32) / divisors[col_idx]
                )
    
    return df_out


def enforce_holdout(train_cases: set, val_cases: set, test_cases: set,
                    all_cases: set, holdout_ur: float | None) -> tuple:
    """
    Enforce holdout Ur removal from train/val and addition to test.
    """
    if holdout_ur is None:
        return train_cases, val_cases, test_cases
    
    holdout_label = format_ur_label(holdout_ur)
    
    if holdout_label not in all_cases:
        raise ValueError(f"Holdout Ur={holdout_ur} (label={holdout_label}) not found in data.")
    
    train_cases = train_cases - {holdout_label}
    val_cases = val_cases - {holdout_label}
    test_cases = test_cases | {holdout_label}
    
    # Check for overlaps
    overlap_tv = train_cases & val_cases
    overlap_tt = train_cases & test_cases
    overlap_vt = val_cases & test_cases
    
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"Split overlap detected: train∩val={overlap_tv}, "
            f"train∩test={overlap_tt}, val∩test={overlap_vt}")
    
    return train_cases, val_cases, test_cases


def format_ur_label(ur: float) -> str:
    """Format a Ur value as a canonical label."""
    from viv_analysis.utils import format_ur_label as original_format
    return original_format(ur)


def resolve_use_ur_context(dataset_default: bool, use_ur_context_flag: bool,
                           no_ur_context_flag: bool) -> bool:
    """
    Resolve the effective use_ur_context setting.

    --use_ur_context / --no_ur_context are mutually exclusive CLI switches.
    If neither was passed, retain the dataset's own default (backward
    compatibility); otherwise the explicit CLI choice wins.
    """
    if use_ur_context_flag:
        return True
    if no_ur_context_flag:
        return False
    return bool(dataset_default)


def resolve_dataset(dataset_pos: str | None, dataset_cli: str | None) -> str:
    """
    Resolve the effective --cfd_dataset value from the legacy positional
    argument and/or the new --cfd_dataset flag.

    Uses "is not None" (never `or`) so an explicitly-passed empty string is
    preserved and rejected by the caller's dataset validation, instead of
    being silently treated as "not given" and defaulted to the Re=200
    cylinder dataset (e.g. a batch script passing --cfd_dataset "$UNSET_VAR").
    """
    if dataset_pos is not None and dataset_cli is not None:
        if dataset_pos != dataset_cli:
            raise ValueError(
                f"Dataset mismatch: positional='{dataset_pos}' vs --cfd_dataset='{dataset_cli}'"
            )
    if dataset_cli is not None:
        return dataset_cli
    if dataset_pos is not None:
        return dataset_pos
    return "cylinder"


def check_artifact_collision(output_dir: Path, overwrite: bool) -> None:
    """Raise FileExistsError if completed artifacts already exist and
    --overwrite was not passed, so one arm can never silently clobber another."""
    if overwrite:
        return
    existing_artifacts = [
        output_dir / "gru_best.pt",
        output_dir / "metrics_gru.json",
    ]
    if any(f.exists() for f in existing_artifacts):
        raise FileExistsError(
            f"Artifacts already exist in {output_dir}. "
            f"Use --overwrite to replace them."
        )


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

def setup_argparse() -> argparse.ArgumentParser:
    """Create argument parser for train_gru."""
    parser = argparse.ArgumentParser(
        description="Train GRU model for VIV lift prediction (cylinder1000 and bridge)."
    )
    
    # Legacy support: optional positional dataset argument
    parser.add_argument(
        "dataset_pos", nargs="?", default=None,
        help="(Legacy) Dataset: cylinder, cylinder1000, or bridge"
    )
    
    # Primary options
    parser.add_argument(
        "--cfd_dataset", type=str, default=None,
        help="Dataset: cylinder, cylinder1000, or bridge"
    )
    parser.add_argument(
        "--nd_inputs", action="store_true",
        help="Use nondimensional inputs [h/D, hdot/U, hddot/(U²/D)]"
    )
    
    # Context option group (mutually exclusive)
    context_group = parser.add_mutually_exclusive_group()
    context_group.add_argument(
        "--use_ur_context", action="store_true",
        help="Include Ur as a context feature (overrides dataset default)"
    )
    context_group.add_argument(
        "--no_ur_context", action="store_true",
        help="Disable Ur context (overrides dataset default)"
    )
    
    # Training parameters
    parser.add_argument(
        "--holdout_ur", type=float, default=None,
        help="Hold out a specific Ur from training (e.g., 5.5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: dataset config)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (default: dataset config)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=None,
        help="Sequence length (default: dataset config)"
    )
    parser.add_argument(
        "--noise_std", type=float, default=0.05,
        help="Input noise std after standardization (default: 0.05)"
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="Random seed for reproducibility (default: 123)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader workers (default: 0)"
    )
    
    # Output/mode options
    parser.add_argument(
        "--exp_subdir", type=str, default=None,
        help="Artifact subdirectory in results/; if omitted use default gru_{dataset}"
    )
    parser.add_argument(
        "--preflight_only", action="store_true",
        help="Stop after preflight (data load, split, scaler fit); do not train"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow overwriting existing artifacts"
    )
    
    return parser


def main() -> None:
    parser = setup_argparse()
    args = parser.parse_args()
    
    # ── Resolve dataset ────────────────────────────────────────────────────
    # Handle legacy positional + new --cfd_dataset
    dataset = resolve_dataset(args.dataset_pos, args.cfd_dataset)
    print(f"Dataset: {dataset}")
    
    # Pre-validate datasetusing _resolve_data_dirs (which now raises on unknown)
    try:
        from viv_analysis.preprocess import _resolve_data_dirs
        _resolve_data_dirs(dataset)
    except ValueError as e:
        raise ValueError(f"Invalid dataset '{dataset}': {e}")
    
    # ── Prepare config ─────────────────────────────────────────────────────
    cfg = prepare_gru_config(dataset, config).copy()
    
    # Handle --nd_inputs and --use_ur_context if specified
    nd_inputs = bool(args.nd_inputs)
    use_ur_context = resolve_use_ur_context(
        cfg.get("use_ur_context", False), args.use_ur_context, args.no_ur_context)

    cfg["use_ur_context"] = use_ur_context
    cfg["nd_inputs"] = nd_inputs
    
    # Override config with CLI if provided
    if args.epochs is not None:
        cfg["n_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.seq_len is not None:
        cfg["seq_len"] = args.seq_len
    
    # ── Seeding ────────────────────────────────────────────────────────────
    seed_everything(args.seed)
    
    # ── Determine artifact directory ───────────────────────────────────────
    # coord_suffix and ctx_suffix must BOTH be encoded here -- a default name
    # keyed on nd_inputs alone would collide between the context/no-context
    # arms of the same coordinate mode whenever --exp_subdir is omitted.
    if args.exp_subdir:
        output_dir = ROOT_DIR / "results" / args.exp_subdir
    else:
        coord_suffix = "_nd" if nd_inputs else ""
        ctx_suffix = "_ctx" if use_ur_context else "_noctx"
        output_dir = ROOT_DIR / "results" / f"gru_{dataset}{coord_suffix}{ctx_suffix}"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for collision
    check_artifact_collision(output_dir, args.overwrite)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  seq_len={cfg['seq_len']}  "
          f"coord_mode={'nondimensional' if nd_inputs else 'dimensional'}  "
          f"use_ur_context={use_ur_context}")
    
    # ── Load raw CFD data ──────────────────────────────────────────────────
    from viv_analysis.preprocess import _resolve_data_dirs, downsample
    
    if dataset == "bridge":
        raw_df = merge_dataframes(
            dataset="bridge",
            fn_hz=cfg["bridge_fn_hz"],
            d_ref=cfg["bridge_D_ref"],
        )
        raw_df = downsample(raw_df, cfg["bridge_downsample"])
    else:
        raw_df = merge_dataframes(dataset=dataset)
    
    if raw_df.empty:
        raise RuntimeError("No data found. Check data directories.")
    
    # ── Compute kinematics in physical units ───────────────────────────────
    params_Re1000 = None
    params_bridge = None
    
    if dataset in {"cylinder1000", "cylinder_re_1000", "re1000",
                   "re1000_disp", "re1000_vel", "re1000_acc"}:
        params_Re1000 = structural_params()
        raw_df = compute_kinematics(raw_df, dataset=dataset,
                                    structural_params=params_Re1000)
    elif dataset == "bridge":
        params_bridge = bridge_structural_params()
        raw_df = compute_kinematics(raw_df, dataset=dataset,
                                    bridge_structural_params=params_bridge)
    else:
        raw_df = compute_kinematics(raw_df, dataset=dataset)
    
    # Quarantine short bridge cases
    if dataset == "bridge":
        sizes = raw_df.groupby("case").size().sort_values()
        BRIDGE_MIN_FRAC = 0.05
        med = float(sizes.median())
        drop = set(sizes[sizes < BRIDGE_MIN_FRAC * med].index)
        if drop:
            print(f"Quarantining {len(drop)} bridge case(s): {sorted(drop)}")
            raw_df = raw_df[~raw_df['case'].isin(drop)].copy()
    
    all_cases_unsplit = sorted(str(c) for c in raw_df["case"].drop_duplicates())
    print(f"Cases loaded (before split): {all_cases_unsplit}")
    
    # ── Split cases ────────────────────────────────────────────────────────
    train_cases, val_cases, test_cases, release_time = split_cases(
        all_cases_unsplit, dataset, cfg)
    
    # ── Enforce holdout ────────────────────────────────────────────────────
    holdout_label = format_ur_label(args.holdout_ur) if args.holdout_ur is not None else None
    print(f"Holdout: ur={args.holdout_ur}  label={holdout_label}")
    if args.holdout_ur is not None:
        train_cases, val_cases, test_cases = enforce_holdout(
            train_cases, val_cases, test_cases, set(all_cases_unsplit),
            args.holdout_ur
        )

    print(f"Train: {sorted(train_cases)}")
    print(f"Val:   {sorted(val_cases)}")
    print(f"Test:  {sorted(test_cases)}")
    
    # Verify no overlap
    if (train_cases & val_cases) or (train_cases & test_cases) or (val_cases & test_cases):
        raise ValueError("Split overlap detected after holdout enforcement.")
    
    # ── Apply ND transform (if requested) and split ──────────────────────
    if nd_inputs:
        D = config["cylinder1000_D_ref"]
        fn = config["cylinder1000_fn"]
        input_cols = ["disp", "vel", "acc"]
        
        print(f"\nApplying nondimensional transform (D={D}, fn={fn}):")
        raw_df = apply_nd_transform(raw_df, nd_inputs=True, D=D, fn=fn,
                                    input_cols=input_cols)
        
        # Print receipts for verification
        for case_name in ["Ur5", "Ur7"]:
            if case_name in raw_df["case"].values:
                ur = parse_ur_label(case_name)
                U_case = ur * fn * D
                divisors = [D, U_case, U_case**2 / D]
                divisors_str = "[" + ", ".join(f"{v:.3f}" for v in divisors) + "]"
                print(f"  {case_name}: U={U_case:.3f}, divisors={divisors_str}")
    
    train_df = raw_df[raw_df["case"].isin(train_cases)].copy()
    val_df = raw_df[raw_df["case"].isin(val_cases)].copy()
    test_df = raw_df[raw_df["case"].isin(test_cases)].copy()
    
    # ── Verify holdout is absent from train/val ────────────────────────────
    if args.holdout_ur is not None:
        holdout_label = format_ur_label(args.holdout_ur)
        if holdout_label in train_df["case"].values:
            raise ValueError(f"Holdout {holdout_label} still in train_df!")
        if holdout_label in val_df["case"].values:
            raise ValueError(f"Holdout {holdout_label} still in val_df!")
        if holdout_label not in test_df["case"].values:
            raise ValueError(f"Holdout {holdout_label} not in test_df!")
    
    # ── Fit scalers on training data only ──────────────────────────────────
    input_cols = list(cfg.get("input_cols", ["disp", "vel", "acc"]))
    target_col = str(cfg.get("target_col", "cl"))
    seq_len = int(cfg["seq_len"])
    batch_size = int(cfg["batch_size"])
    
    x_scaler, y_scaler = fit_scalers(train_df, input_cols, target_col)
    
    print(f"\nx_scaler: mean={x_scaler.mean_} scale={x_scaler.scale_}")
    print(f"y_scaler ({target_col}): "
          f"mean={float(y_scaler.mean_[0]):.6f} scale={float(y_scaler.scale_[0]):.6f}")
    
    # ── Ur statistics (from training cases only) ───────────────────────────
    train_ur = np.array([parse_ur_label(c) for c in sorted(train_cases)],
                        dtype=np.float32)
    ur_mean = float(train_ur.mean())
    ur_std = float(train_ur.std()) + 1e-6
    
    if use_ur_context:
        print(f"Using Ur context feature: mean={ur_mean:.4f}, std={ur_std:.4f} "
              f"(fitted from {len(train_cases)} training cases)")
    
    # ── Save run_config.json (preflight receipt) ───────────────────────────
    coord_mode = "nondimensional" if nd_inputs else "dimensional"
    if nd_inputs:
        transform_formula = "disp/D,  vel/U,  acc*D/U²  (case-dependent U)"
        D = config["cylinder1000_D_ref"]
        fn = config["cylinder1000_fn"]
    else:
        transform_formula = "identity (physical units)" 
        D = None
        fn = None
    
    run_config = {
        "cfd_dataset": dataset,
        "coordinate_mode": coord_mode,
        "nd_inputs": nd_inputs,
        "transform_formula": transform_formula,
        "D": float(D) if D is not None else None,
        "fn": float(fn) if fn is not None else None,
        "use_ur_context": use_ur_context,
        "holdout_ur": args.holdout_ur,
        "holdout_label": holdout_label,
        "input_cols": input_cols,
        "target_col": target_col,
        "epochs": cfg["n_epochs"],
        "batch_size": batch_size,
        "seq_len": seq_len,
        "noise_std": float(args.noise_std),
        "noise_coordinate_space": "standardized model inputs",
        "seed": args.seed,
        "num_workers": args.num_workers,
        "output_directory": str(output_dir),
        "train_cases": sorted(train_cases),
        "val_cases": sorted(val_cases),
        "test_cases": sorted(test_cases),
        "scaler_fit_cases": sorted(train_cases),
        "ur_stats_fit_cases": sorted(train_cases),
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"\nRun config saved to {output_dir / 'run_config.json'}")

    # ── Audit printing (always, so preflight mode sees the full receipt) ───
    print(f"\nScaler-fit cases:   {sorted(train_cases)}")
    print(f"Ur-stats-fit cases: {sorted(train_cases)}")
    print(f"Noise std: {args.noise_std}  "
          f"(applied to standardized model inputs, after scaler transform)")

    # ── Preflight only mode ────────────────────────────────────────────────
    if args.preflight_only:
        print("\n[PREFLIGHT MODE] Stopping before training.")
        print(f"CFD dataset: {dataset}")
        print(f"Coordinate mode: {coord_mode}")
        print(f"Use Ur context: {use_ur_context}")
        print(f"Holdout: ur={args.holdout_ur}  label={holdout_label}")
        print(f"Output directory: {output_dir}")
        print(f"Ready for training: {len(train_df)} train rows, "
              f"{len(val_df)} val rows, {len(test_df)} test rows")
        return
    
    # ── Apply scalers ──────────────────────────────────────────────────────
    train_df_s = apply_scalers_to_df(train_df, x_scaler, y_scaler,
                                      input_cols, target_col)
    val_df_s = apply_scalers_to_df(val_df, x_scaler, y_scaler,
                                    input_cols, target_col)
    test_df_s = apply_scalers_to_df(test_df, x_scaler, y_scaler,
                                     input_cols, target_col)
    
    # ── Create datasets and loaders ────────────────────────────────────────
    stride_train = int(cfg["stride_train"])
    common_ds_args = dict(
        seq_len=seq_len, target_col=target_col, input_cols=input_cols,
        release_time=release_time, use_ur_context=use_ur_context,
        ur_mean=ur_mean, ur_std=ur_std
    )
    
    train_ds = VIVSequenceDataset(train_df_s, stride=stride_train, **common_ds_args)
    val_ds = VIVSequenceDataset(val_df_s, stride=1, **common_ds_args)
    test_ds = VIVSequenceDataset(test_df_s, stride=1, **common_ds_args)
    
    print(f"\nDataset sizes — train: {len(train_ds)} val: {len(val_ds)} test: {len(test_ds)}")
    
    # Create generator for reproducible DataLoader shuffling
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    # ── Build model ────────────────────────────────────────────────────────
    input_size = len(input_cols) + (1 if use_ur_context else 0)
    model = VIV_GRU(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    
    print(f"GRU parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Random seed: {args.seed} (controlled reproducibility)")
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    # ── Training loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state = None
    patience_count = 0
    train_losses, val_losses = [], []
    
    n_epochs = cfg["n_epochs"]
    print(f"\nTraining for up to {n_epochs} epochs "
          f"(noise_std={args.noise_std} in standardized space)...")
    
    for epoch in range(1, n_epochs + 1):
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device,
                             input_noise_std=args.noise_std)
        vl, vp, vt, _ = run_validation(model, val_loader, criterion, device)
        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)
        
        ss_res = np.sum((vt - vp) ** 2)
        ss_tot = np.sum((vt - vt.mean()) ** 2)
        val_r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        print(f"Epoch {epoch:3d}/{n_epochs}  "
              f"train={tl:.5f}  val={vl:.5f}  "
              f"R²={val_r2:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")
        
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break
    
    if best_state is None:
        raise RuntimeError("Training failed to produce a best model state.")
    
    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / "gru_best.pt")
    print(f"\nBest model saved (val_loss={best_val_loss:.5f})")
    
    # ── Save scalers and metadata ──────────────────────────────────────────
    with open(output_dir / "x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open(output_dir / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)
    
    ur_stats_dict = {
        "mean": ur_mean,
        "std": ur_std,
        "use_ur_context": use_ur_context,
        "nd_inputs": nd_inputs,
        "coordinate_mode": coord_mode,
        "cfd_dataset": dataset,
        "holdout_ur": args.holdout_ur,
        "exp_subdir": args.exp_subdir or f"gru_{dataset}",
    }
    with open(output_dir / "ur_stats.pkl", "wb") as f:
        pickle.dump(ur_stats_dict, f)
    
    # ── Final evaluation ───────────────────────────────────────────────────
    _, tp, tt, tcases = run_validation(model, test_loader, criterion, device)
    _, vp, vt, vcases = run_validation(model, val_loader, criterion, device)
    
    test_pred = y_scaler.inverse_transform(tp.reshape(-1, 1)).ravel()
    test_true = y_scaler.inverse_transform(tt.reshape(-1, 1)).ravel()
    val_pred = y_scaler.inverse_transform(vp.reshape(-1, 1)).ravel()
    val_true = y_scaler.inverse_transform(vt.reshape(-1, 1)).ravel()
    
    val_metrics = evaluate(val_true, val_pred)
    test_metrics = evaluate(test_true, test_pred)
    
    print(f"\nValidation: {json.dumps(val_metrics, indent=2)}")
    print(f"Test:       {json.dumps(test_metrics, indent=2)}")
    
    # ── Teacher forcing rollout ────────────────────────────────────────────
    print("\nTeacher forcing rollout on test cases:")
    tf_results = {}
    for case_name in sorted(test_cases):
        case_df_s = test_df_s[test_df_s["case"] == case_name].copy()
        if case_df_s.empty:
            continue
        
        cl_pred, cl_true, times_ar = teacher_forcing_rollout(
            model, case_df_s, input_cols, seq_len,
            release_time[case_name], y_scaler, case_name, device,
            use_ur_context=use_ur_context,
            ur_stats=(ur_mean, ur_std),
        )
        
        if len(cl_true) < 20:
            print(f"  {case_name}: too short, skipping")
            continue
        
        tf_m = evaluate(cl_true, cl_pred)
        tf_results[case_name] = tf_m
        print(f"  {case_name}: R²={tf_m['r2']:.4f}  RMSE={tf_m['rmse']:.4f}")
        plot_tf_result(cl_pred, cl_true, times_ar, case_name, output_dir)
    
    # ── Amplitude response curve ───────────────────────────────────────────
    all_df_s = pd.concat([train_df_s, val_df_s, test_df_s], ignore_index=True)
    amplitude_comparison(
        model, all_df_s, release_time, input_cols, seq_len, y_scaler,
        device, output_dir, train_cases, val_cases, test_cases,
        use_ur_context=use_ur_context, ur_stats=(ur_mean, ur_std),
    )
    
    # ── Learning curve ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(train_losses, color="black", label="train")
    ax.plot(val_losses, color="tab:blue", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled)")
    ax.set_title(f"GRU learning curve — {dataset} ({coord_mode})")
    ax.legend()
    fig.savefig(output_dir / "learning_curve.png", dpi=150)
    plt.close(fig)
    
    # ── Save metrics ───────────────────────────────────────────────────────
    metrics = {
        "dataset": dataset,
        "coordinate_mode": coord_mode,
        "nd_inputs": nd_inputs,
        "use_ur_context": use_ur_context,
        "holdout_ur": args.holdout_ur,
        "holdout_label": holdout_label,
        "gru_config": {k: v for k, v in cfg.items() if not callable(v)},
        "case_split": {
            "train": sorted(train_cases),
            "val": sorted(val_cases),
            "test": sorted(test_cases),
        },
        "scaler_fit_cases": sorted(train_cases),
        "ur_stats_fit_cases": sorted(train_cases),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "tf_results": tf_results,
    }
    with open(output_dir / "metrics_gru.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
