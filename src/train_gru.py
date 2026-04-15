# src/train_gru.py

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.metrics import r2_score
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import config
from preprocess import merge_dataframes, compute_kinematics, downsample
from models.gru import VIV_GRU, VIVSequenceDataset, fit_scalers, apply_scalers_to_df
from evaluate import evaluate

ROOT_DIR   = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "results" / "gru_model"


# ── Configuration ──────────────────────────────────────────────────────────────

GRU_CONFIG = {
    "seq_len":     900,       # full 2-cycle lookback
    "input_cols":  ["disp", "vel", "acc"],
    "target_col":  "cl",
    "hidden_size": 64,
    "num_layers":  2,
    "dropout":     0.1,
    "lr":          1e-3,
    "weight_decay": 1e-5,
    "n_epochs":    100,
    "batch_size":  256,
    "patience":    15,        # early stopping patience
    "stride_train": 3,        # use every 3rd window for training
    "stride_val":   1,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
}

BRIDGE_GRU_CONFIG = {
    **GRU_CONFIG,
    "seq_len": config["bridge_seq_len"],
    "downsample": config["bridge_downsample"],
    "stride_train": config["bridge_stride_train"],
    "dataset": "bridge",
    "fn_hz": config["bridge_fn_hz"],
    "d_ref": config["bridge_d_ref"],
    "t_star_release": config["bridge_t_star_release"],
}

DATASET = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "cylinder"
ACTIVE_CONFIG = BRIDGE_GRU_CONFIG if DATASET == "bridge" else GRU_CONFIG


# ── Case split ─────────────────────────────────────────────────────────────────

def case_to_float(case_name: str) -> float:
    name = str(case_name)
    if name.startswith("Ur"):
        return float(name[2:])
    return float(name)


def make_dynamic_split(cases: list[str]) -> tuple[set[str], set[str], set[str]]:
    ordered = sorted(set(cases), key=case_to_float)
    n = len(ordered)
    if n < 3:
        raise ValueError("Need at least 3 cases to create train/val/test splits.")

    n_test = max(1, int(round(0.2 * n)))
    n_val = max(1, int(round(0.2 * n)))
    max_holdout = n - 1
    if n_test + n_val > max_holdout:
        overflow = n_test + n_val - max_holdout
        n_val = max(1, n_val - overflow)

    test_cases = set(ordered[-n_test:])
    val_pool = [c for c in ordered if c not in test_cases]
    val_cases = set(val_pool[-n_val:])
    train_cases = set([c for c in ordered if c not in test_cases and c not in val_cases])
    return train_cases, val_cases, test_cases


def split_cases(cases, dataset="cylinder", fn_hz=0.32, d_ref=7.42, t_star_release=20.0):
    
    fixed_train = {"Ur3.0", "Ur5.6", "Ur5.0", "Ur6.0", "Ur6.5", "Ur9.0", "Ur4.6", "Ur5.4"}
    fixed_val = {"Ur4.0", "Ur4.4", "Ur5.2", "Ur4.8", "Ur8.0"}
    fixed_test = {"Ur4.2", "Ur5.8", "Ur7.0"}
    fixed_all = fixed_train | fixed_val | fixed_test

    case_set = set(cases)
    if fixed_all.issubset(case_set):
        train_cases, val_cases, test_cases = fixed_train, fixed_val, fixed_test
        print("Using fixed cylinder split.")
    else:
        train_cases, val_cases, test_cases = make_dynamic_split(cases)
        print("Using dynamic split from available cases.")

    if dataset == "cylinder":
        release_time = {v: 60.0 for v in cases}
    else:
        # Bridge: t_release = T* * D / U, where U = Ur * fn * D
        release_time = {}
        for case in cases:
            ur = case_to_float(case)
            U  = ur * fn_hz * d_ref
            release_time[case] = float(t_star_release * d_ref / U)

    missing = (train_cases | val_cases | test_cases) - case_set
    if missing:
        print(f"WARNING: cases not found in data: {missing}")

    return train_cases, val_cases, test_cases, release_time


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      VIV_GRU,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     str,
    forcing_ratio: float = 1.0,   # 1.0 = full teacher forcing, 0.0 = autoregressive
) -> float:
    """
    Train for one epoch.
    
    forcing_ratio controls teacher forcing. During early training, always
    use true inputs (forcing_ratio=1.0). Gradually reduce to encourage
    the model to be robust during autoregressive rollout.
    
    For the kinematic inputs (disp, vel, acc), we always use true values
    since they come from the structural solver, not the ML model.
    Teacher forcing only applies if CL is also an input feature.
    """
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)   # (batch, seq_len, n_features)
        y_batch = y_batch.to(device)   # (batch,)

        optimizer.zero_grad()
        pred, _ = model(x_batch)
        loss     = criterion(pred, y_batch)
        loss.backward()

        # Gradient clipping — important for RNNs to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(
    model:     VIV_GRU,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_true   = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred, _ = model(x_batch)
            loss    = criterion(pred, y_batch)
            total_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_true.append(y_batch.cpu().numpy())

    return (
        total_loss / len(loader),
        np.concatenate(all_preds),
        np.concatenate(all_true),
    )


def autoregressive_rollout_gru(
    model:        VIV_GRU,
    case_df:      pd.DataFrame,  # already scaled
    input_cols:   list[str],
    seq_len:      int,
    release_t:    float,         # in original (unscaled) time units
    y_scaler,
    device:       str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the GRU autoregressively on a single case.
    
    Since our inputs are purely kinematic (disp, vel, acc from the structural
    solver), we always feed true kinematics — no teacher forcing needed here.
    The GRU hidden state is carried forward across timesteps, which is the
    key advantage over the ELM: memory is maintained explicitly.
    """
    import pandas as pd

    model.eval()
    ordered = case_df.sort_values("time").reset_index(drop=True)
    signal  = ordered[input_cols].to_numpy(dtype=np.float32)
    cl_true_s = ordered["cl"].to_numpy(dtype=np.float32)
    times   = ordered["time"].to_numpy(dtype=np.float32)

    release_idx = int(np.searchsorted(times, release_t))
    start       = max(seq_len, release_idx + seq_len)

    cl_pred_s = []
    h = None  # carry hidden state forward

    with torch.no_grad():
        for i in range(start, len(ordered)):
            window = signal[i - seq_len : i]              # (seq_len, n_feat)
            x = torch.from_numpy(window).unsqueeze(0).to(device)  # (1, seq_len, n_feat)
            pred, h = model(x, h)
            # Detach h to prevent backprop through entire history
            h = h.detach()
            cl_pred_s.append(pred.item())

    cl_pred_s = np.array(cl_pred_s, dtype=np.float32)
    cl_true_s_slice = cl_true_s[start:]
    times_slice = times[start:]

    # Inverse transform to original CL scale
    cl_pred = y_scaler.inverse_transform(
        cl_pred_s.reshape(-1, 1)).ravel()
    cl_true = y_scaler.inverse_transform(
        cl_true_s_slice.reshape(-1, 1)).ravel()

    return cl_pred, cl_true, times_slice

def plot_ar_result(cl_pred, cl_true, times, case_name, output_dir):
    """
    Two-panel plot:
    Top: time series comparison (CFD vs GRU)
    Bottom: residual time series (GRU - CFD)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

    # Time series
    ax = axes[0]
    ax.plot(times, cl_true, lw=0.8, color="black", label="CFD (ground truth)")
    ax.plot(times, cl_pred, lw=0.8, color="tab:blue", 
            alpha=0.85, label="GRU (autoregressive)")
    ax.set_ylabel("$C_L$", fontsize=13)
    ax.set_title(f"{case_name} — autoregressive rollout  "
                 f"$R^2={r2_score(cl_true, cl_pred):.4f}$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residual
    ax2 = axes[1]
    residual = cl_pred - cl_true
    ax2.plot(times, residual, lw=0.6, color="tab:red", alpha=0.8)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel(r"Residual ($\hat{C}_L - C_L$)", fontsize=13)
    ax2.set_xlabel("Time [s]", fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.savefig(output_dir / f"ar_{case_name}.png", dpi=200)
    plt.close(fig)

def amplitude_comparison(model, all_df_scaled, release_time, 
                          input_cols, seq_len, y_scaler, device, output_dir):
    """
    For each case, run AR rollout and compute steady-state CL amplitude.
    Plot GRU amplitude vs CFD amplitude vs Ur.
    This is the VIV response curve — the primary validation for your thesis.
    """
    from sklearn.metrics import r2_score as sk_r2
    results = []

    for case_name, case_df in all_df_scaled.groupby("case"):
        ur = case_to_float(case_name)
        release_t = release_time[case_name]
        
        cl_pred, cl_true, times = autoregressive_rollout_gru(
            model, case_df, input_cols, seq_len,
            release_t, y_scaler, device)

        # Steady-state: last 30% of signal
        ss = int(0.7 * len(cl_true))
        amp_cfd = (cl_true[ss:].max() - cl_true[ss:].min()) / 2
        amp_gru = (cl_pred[ss:].max() - cl_pred[ss:].min()) / 2

        results.append({
            "Ur": ur,
            "CL_amp_CFD": amp_cfd,
            "CL_amp_GRU": amp_gru,
            "ar_r2": sk_r2(cl_true, cl_pred),
        })
        print(f"  {case_name}: Ur={ur}  "
              f"CL_amp_CFD={amp_cfd:.4f}  CL_amp_GRU={amp_gru:.4f}")

    df_res = pd.DataFrame(results).sort_values("Ur")

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(df_res["Ur"], df_res["CL_amp_CFD"], "o-", 
            color="black", label="CFD", markersize=6)
    ax.plot(df_res["Ur"], df_res["CL_amp_GRU"], "s--",
            color="tab:blue", label="GRU", markersize=6)
    ax.set_xlabel("Reduced velocity $U_r$", fontsize=13)
    ax.set_ylabel("CL amplitude", fontsize=13)
    ax.set_title("VIV aerodynamic response curve — CFD vs GRU surrogate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "CL_amplitude_response_curve.png", dpi=200)
    plt.close(fig)
    df_res.to_csv(output_dir / "amplitude_comparison.csv", index=False)
    return df_res


def plot_amplitude_response(df_res, output_dir, train_cases, val_cases, test_cases):
    """
    Publication-quality VIV response curve.
    Shows CFD ground truth vs GRU surrogate with train/val/test annotation.
    """
    train_cases_f = {case_to_float(c) for c in train_cases}
    val_cases_f = {case_to_float(c) for c in val_cases}
    test_cases_f = {case_to_float(c) for c in test_cases}

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    # ── Panel 1: amplitude curves ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(df_res["Ur"], df_res["CL_amp_CFD"], "o-",
            color="black", lw=1.5, markersize=5, label="CFD (ground truth)", zorder=3)
    ax.plot(df_res["Ur"], df_res["CL_amp_GRU"], "s--",
            color="tab:blue", lw=1.5, markersize=5, label="GRU surrogate", zorder=3)

    # Annotate split membership
    for _, row in df_res.iterrows():
        ur = float(row["Ur"])
        if ur in test_cases_f:
            ax.axvline(row["Ur"], color="tab:orange", lw=0.5, 
                       alpha=0.4, zorder=1)
        elif ur in val_cases_f:
            ax.axvline(row["Ur"], color="tab:green", lw=0.5,
                       alpha=0.4, zorder=1)

    # Dummy lines for legend
    ax.axvline(-99, color="tab:green",  lw=2, alpha=0.6, label="validation cases")
    ax.axvline(-99, color="tab:orange", lw=2, alpha=0.6, label="test cases")

    ax.set_xlim(df_res["Ur"].min() - 0.2, df_res["Ur"].max() + 0.2)
    ax.set_ylabel(r"$C_L$ amplitude", fontsize=13)
    ax.set_title("VIV aerodynamic response curve — cylinder baseline", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: relative error ────────────────────────────────────────────
    ax2 = axes[1]
    rel_error = 100 * (df_res["CL_amp_GRU"] - df_res["CL_amp_CFD"]) \
                / (df_res["CL_amp_CFD"] + 1e-8)
    colors = ["tab:orange" if float(r["Ur"]) in test_cases_f
              else "tab:green" if float(r["Ur"]) in val_cases_f
              else "black"
              for _, r in df_res.iterrows()]
    ax2.bar(df_res["Ur"], rel_error, width=0.15, color=colors, alpha=0.8)
    ax2.axhline(0,  color="black", lw=0.8)
    ax2.axhline(+10, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax2.axhline(-10, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax2.set_xlabel(r"Reduced velocity $U_r = U / (f_n D)$", fontsize=13)
    ax2.set_ylabel("Relative error [%]", fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.savefig(output_dir / "VIV_response_curve_comparison.png", dpi=200)
    plt.close(fig)
    print(f"Saved: VIV_response_curve_comparison.png")

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import pandas as pd
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = ACTIVE_CONFIG["device"]
    print(f"Device: {device}")
    print(f"Dataset: {DATASET}")

    # ── Load and prepare data ──────────────────────────────────────────────
    raw_df = merge_dataframes(
        dataset=DATASET,
        fn_hz=ACTIVE_CONFIG.get("fn_hz"),
        d_ref=ACTIVE_CONFIG.get("d_ref"),
    )
    if raw_df.empty:
        print("No data found."); return

    if DATASET == "bridge":
        raw_df = downsample(raw_df, ACTIVE_CONFIG["downsample"])

    raw_df = compute_kinematics(raw_df)
    all_cases = sorted(str(c) for c in raw_df["case"].drop_duplicates())
    train_cases, val_cases, test_cases, release_time = split_cases(
        all_cases,
        dataset=DATASET,
        fn_hz=ACTIVE_CONFIG.get("fn_hz", 0.32),
        d_ref=ACTIVE_CONFIG.get("d_ref", 7.42),
        t_star_release=ACTIVE_CONFIG.get("t_star_release", 20.0),
    )

    train_df = raw_df[raw_df["case"].isin(train_cases)].copy()
    val_df   = raw_df[raw_df["case"].isin(val_cases)].copy()
    test_df  = raw_df[raw_df["case"].isin(test_cases)].copy()

    # Scale: fit on training data only, apply to all splits
    x_scaler, y_scaler = fit_scalers(
        train_df,
        ACTIVE_CONFIG["input_cols"],
        ACTIVE_CONFIG["target_col"],
    )

    train_df_s = apply_scalers_to_df(
        train_df, x_scaler, y_scaler,
        ACTIVE_CONFIG["input_cols"], ACTIVE_CONFIG["target_col"])
    val_df_s   = apply_scalers_to_df(
        val_df,   x_scaler, y_scaler,
        ACTIVE_CONFIG["input_cols"], ACTIVE_CONFIG["target_col"])
    test_df_s  = apply_scalers_to_df(
        test_df,  x_scaler, y_scaler,
        ACTIVE_CONFIG["input_cols"], ACTIVE_CONFIG["target_col"])

    # ── Build datasets and loaders ─────────────────────────────────────────
    seq_len = ACTIVE_CONFIG["seq_len"]

    train_ds = VIVSequenceDataset(
        train_df_s, seq_len,
        ACTIVE_CONFIG["target_col"], ACTIVE_CONFIG["input_cols"],
        release_time, stride=ACTIVE_CONFIG["stride_train"])
    val_ds   = VIVSequenceDataset(
        val_df_s, seq_len,
        ACTIVE_CONFIG["target_col"], ACTIVE_CONFIG["input_cols"],
        release_time, stride=ACTIVE_CONFIG["stride_val"])
    test_ds  = VIVSequenceDataset(
        test_df_s, seq_len,
        ACTIVE_CONFIG["target_col"], ACTIVE_CONFIG["input_cols"],
        release_time, stride=ACTIVE_CONFIG["stride_val"])

    print(f"Dataset sizes — train: {len(train_ds)}  "
          f"val: {len(val_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=ACTIVE_CONFIG["batch_size"],
        shuffle=True, num_workers=2, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(
        val_ds,   batch_size=ACTIVE_CONFIG["batch_size"],
        shuffle=False, num_workers=2)
    test_loader  = DataLoader(
        test_ds,  batch_size=ACTIVE_CONFIG["batch_size"],
        shuffle=False, num_workers=2)

    # ── Build model ────────────────────────────────────────────────────────
    model = VIV_GRU(
        input_size  = len(ACTIVE_CONFIG["input_cols"]),
        hidden_size = ACTIVE_CONFIG["hidden_size"],
        num_layers  = ACTIVE_CONFIG["num_layers"],
        dropout     = ACTIVE_CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GRU parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ACTIVE_CONFIG["lr"],
        weight_decay=ACTIVE_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    # ── Training loop with early stopping ─────────────────────────────────
    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    train_losses   = []
    val_losses     = []

    print(f"\nTraining GRU for up to {ACTIVE_CONFIG['n_epochs']} epochs...")
    for epoch in range(1, ACTIVE_CONFIG["n_epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_pred_s, val_true_s = validate(
            model, val_loader, criterion, device)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # R² in scaled space for monitoring
        ss_res = np.sum((val_true_s - val_pred_s) ** 2)
        ss_tot = np.sum((val_true_s - val_true_s.mean()) ** 2)
        val_r2 = 1 - ss_res / (ss_tot + 1e-10)

        print(
            f"Epoch {epoch:3d}/{ACTIVE_CONFIG['n_epochs']}  "
            f"train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}  "
            f"val_R²={val_r2:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= ACTIVE_CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {ACTIVE_CONFIG['patience']} epochs)")
                break

    # ── Restore best model ─────────────────────────────────────────────────
    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")
    model.load_state_dict(best_state)
    torch.save(best_state, OUTPUT_DIR / "gru_best.pt")
    print(f"Best model saved (val_loss={best_val_loss:.5f})")

    # ── Final evaluation ───────────────────────────────────────────────────
    _, test_pred_s, test_true_s = validate(model, test_loader, criterion, device)

    # Inverse transform
    test_pred = y_scaler.inverse_transform(
        test_pred_s.reshape(-1, 1)).ravel()
    test_true = y_scaler.inverse_transform(
        test_true_s.reshape(-1, 1)).ravel()
    val_pred  = y_scaler.inverse_transform(
        val_pred_s.reshape(-1, 1)).ravel()
    val_true  = y_scaler.inverse_transform(
        val_true_s.reshape(-1, 1)).ravel()

    val_metrics  = evaluate(val_true,  val_pred)
    test_metrics = evaluate(test_true, test_pred)

    print(f"\nValidation: {json.dumps(val_metrics, indent=2)}")
    print(f"Test:       {json.dumps(test_metrics, indent=2)}")

    all_df_s = pd.concat([train_df_s, val_df_s, test_df_s], ignore_index=True)

    # ── Autoregressive rollout on test cases ───────────────────────────────
    print("\nAutoregressive rollout on test cases:")
    ar_results = {}
    for case_name in sorted(test_cases):
        case_df_s = test_df_s[test_df_s["case"] == case_name].copy()
        if case_df_s.empty:
            continue
        release_t = release_time[case_name]
        cl_pred, cl_true, times_ar = autoregressive_rollout_gru(
            model, case_df_s,
            GRU_CONFIG["input_cols"], seq_len,
            release_t, y_scaler, device,
        )
        if len(cl_true) < 20:
            print(f"  WARNING: {case_name} rollout too short ({len(cl_true)} points); skipping metrics/plot")
            continue

        ar_m = evaluate(cl_true, cl_pred)
        ar_results[case_name] = ar_m
        print(f"  {case_name}: R²={ar_m['r2']:.4f}  RMSE={ar_m['rmse']:.4f}")

        plot_ar_result(
            cl_pred=cl_pred,
            cl_true=cl_true,
            times=times_ar,
            case_name=case_name,
            output_dir=OUTPUT_DIR,
        )

    # ── Learning curve ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(train_losses, label="train loss", color="black")
    ax.plot(val_losses,   label="val loss",   color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled)")
    ax.set_title("GRU learning curve")
    ax.legend()
    fig.savefig(OUTPUT_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)

    # ── Plot AR results ─────────────────────────────────────────────────────



    amp_df = amplitude_comparison(
        model=model,
        all_df_scaled=all_df_s,
        release_time=release_time,
        input_cols=GRU_CONFIG["input_cols"],
        seq_len=seq_len,
        y_scaler=y_scaler,
        device=device,
        output_dir=OUTPUT_DIR,
    )

    plot_amplitude_response(amp_df, OUTPUT_DIR, train_cases, val_cases, test_cases) 


    # ── Save metrics ───────────────────────────────────────────────────────
    metrics = {
        "gru_config":    ACTIVE_CONFIG,
        "case_split":    {
            "train": sorted(train_cases),
            "val":   sorted(val_cases),
            "test":  sorted(test_cases),
        },
        "val_metrics":   val_metrics,
        "test_metrics":  test_metrics,
        "ar_results":    ar_results,
    }
    with open(OUTPUT_DIR / "metrics_gru.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")



if __name__ == "__main__":
    main()