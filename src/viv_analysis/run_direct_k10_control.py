# run_direct_k10_control.py
#
# TRAINING-DURATION CONTROL for the rollout-curriculum headline result.
#
# Question it answers:
#   curriculum-k10 (0.145 mean A/D err) vs baseline (0.227) confounds two things:
#     (i)  staged warm-up   k1 -> k2 -> k5 moved the weights before k=10, and
#     (ii) the k=10 phase simply ran 15 epochs (vs 5 each for k2/k5).
#   This control removes (i): train 15 epochs at k=10 DIRECTLY from the raw
#   trim-[2,2.5,3] baseline, no staging. Then evaluate with the SAME closed-loop
#   sweep harness.
#     - direct-k10  ~=  curriculum-k10   => staging contributes nothing; the
#                                           claim shrinks to "train at k=10, 15 ep".
#     - curriculum-k10 clearly better    => staging is real; a small (<=3-4)
#                                           schedule sweep becomes warranted.
#
# What is held identical to curriculum-k10 (so the ONLY variable is staging):
#   * warm-start checkpoint, scalers, ur_stats, split, physics, dataset kwargs
#   * rollout mechanics: we import the SAME run_phase the curriculum used
#   * learning rate: PINNED to 1.25e-4 = config['lr']/8, the value the curriculum
#     actually reached at k=10 (1e-3 ->/2@k2 ->/2@k5 ->/2@k10). Do NOT rely on the
#     curriculum loop's `if k>1: lr*=0.5`: for a single phase it fires ONCE and
#     lands at 5e-4 (4x too high), silently re-confounding LR with staging.
#
# Residual asymmetry (acknowledged, not fixable for a true control):
#   Adam starts COLD here (curriculum-k10 inherited moments from k1/k2/k5).
#   This slightly disadvantages the control, so a tie => "staging adds nothing"
#   is the robust reading; a control-loss is partly cold-Adam, not only staging.
#
# Usage:
#   python -m viv_analysis.run_direct_k10_control cylinder_re_1000
#   (then point evaluate_all.py at OUTPUT_DIR / "gru_rollout_k10.pt")

import copy
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

from viv_analysis.config import config, prepare_gru_config
from viv_analysis.models.gru import VIV_GRU, apply_scalers_to_df
from viv_analysis.preprocess import merge_dataframes, compute_kinematics
from viv_analysis.train_gru import split_cases
from viv_analysis.run_curriculum import run_phase          # identical phase mechanics
from viv_analysis.utils import PROJECT_ROOT


# ── Control knobs ──────────────────────────────────────────────────────────────
K = 10
EPOCHS = 15
LR_K10 = config["lr"] / 8.0          # = 1.25e-4, the LR curriculum used at k=10
SMOKE_TEST = True                    # cheap shape/loss check before the 15-ep run


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cylinder_re_1000"

    OUTPUT_DIR = PROJECT_ROOT / "results" / f"gru_rollout_{dataset}_directk10"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = prepare_gru_config(dataset, config)

    input_cols = ["disp", "vel", "acc"]
    cfg["input_cols"] = input_cols
    cfg["use_ur_context"] = True
    use_ur_context = True

    artifact_dir = PROJECT_ROOT / "results" / f"gru_{dataset}"

    # ── scalers + ur_stats from the BASELINE dir (never recompute) ─────────────
    with open(artifact_dir / "ur_stats.pkl", "rb") as f:
        ur_stats = pickle.load(f)
    with open(artifact_dir / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(artifact_dir / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    print(f"ur_stats: mean={ur_stats['mean']:.4f} std={ur_stats['std']:.4f}")

    # ── data + kinematics (force-derived; self-consistent) ─────────────────────
    raw_df = merge_dataframes(dataset=dataset)

    D = config["cylinder1000_D_ref"]
    fn = 0.2
    omega_n = 2 * np.pi * fn
    M_star, zeta = 2.0, 0.007
    cylinder_mass = M_star * 1.0 * (np.pi * (D / 2) ** 2)
    damper = cylinder_mass * 2 * omega_n * zeta
    stiffness = cylinder_mass * omega_n ** 2

    structural_params = {
        "m": cylinder_mass, "c": damper, "k": stiffness,
        "cylinder_mass": cylinder_mass, "c_struct": damper, "k_struct": stiffness,
    }
    raw_df = compute_kinematics(raw_df, dataset=dataset,
                                structural_params=structural_params)

    all_cases = sorted(str(c) for c in raw_df["case"].drop_duplicates())
    train_cases, val_cases, _, release_time = split_cases(all_cases, dataset, cfg)

    train_df = raw_df[raw_df["case"].isin(train_cases)].copy()
    val_df = raw_df[raw_df["case"].isin(val_cases)].copy()
    print(f"train cases: {sorted(train_cases)}")
    print(f"val   cases: {sorted(val_cases)}")
    assert set(train_df["case"].unique()) <= train_cases, \
        "train_df contains val/test cases — check split logic"

    train_df_s = apply_scalers_to_df(
        train_df, x_scaler, y_scaler, input_cols, cfg["target_col"])
    val_df_s = apply_scalers_to_df(
        val_df, x_scaler, y_scaler, input_cols, cfg["target_col"])

    # ── kwarg dicts (identical to run_curriculum) ──────────────────────────────
    dataset_kwargs = dict(
        seq_len=int(cfg["seq_len"]),
        target_col=str(cfg["target_col"]),
        input_cols=input_cols,
        release_time=release_time,
        use_ur_context=use_ur_context,
        ur_mean=ur_stats["mean"],
        ur_std=ur_stats["std"],
    )
    physics_kwargs = dict(
        dt=0.005, m=cylinder_mass, c=damper, k_phys=stiffness,
        rho=1.0, D=D, fn=fn,
        idx_h=0, idx_hdot=1, idx_hddot=2,
        expected_features=len(input_cols) + 1,
        x_mean=x_scaler.mean_, x_scale=x_scaler.scale_,
        y_mean=float(np.asarray(y_scaler.mean_)[0]),
        y_scale=float(np.asarray(y_scaler.scale_)[0]),
        criterion=nn.MSELoss(), device=device,
        batch_size=int(cfg["batch_size"]),
        stride_train=int(cfg["stride_train"]),
    )

    # ── model: warm-start from the RAW baseline (this is the control) ──────────
    model = VIV_GRU(
        input_size=len(input_cols) + 1,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    baseline_ckpt = artifact_dir / "gru_best.pt"
    if not baseline_ckpt.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {baseline_ckpt}. "
            "The control MUST warm-start from the trim-[2,2.5,3] baseline.")
    model.load_state_dict(torch.load(baseline_ckpt, map_location=device))
    print(f"Warm-started from {baseline_ckpt} (raw baseline, no staging)")

    # ── optimizer: LR PINNED to the curriculum's k=10 value (NOT halved) ───────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR_K10, weight_decay=cfg["weight_decay"])
    print(f"LR pinned to {LR_K10:.3e}  (= config lr {config['lr']:.0e} / 8, "
          f"the value curriculum-k10 used). Adam starts cold by construction.")

    # ── cheap pre-flight smoke test before the expensive run ───────────────────
    if SMOKE_TEST:
        from viv_analysis.models.gru import VIVSequenceDataset
        from viv_analysis.train_gru import _rollout_loss_k
        ds = VIVSequenceDataset(train_df_s, stride=int(cfg["stride_train"]),
                                rollout_k=K, **dataset_kwargs)
        x, y, cn = ds[min(100, len(ds) - 1)]
        print("\n[smoke] one sample:")
        print(f"  x.shape={tuple(x.shape)}  (expect [{cfg['seq_len']}, "
              f"{len(input_cols) + 1}])")
        print(f"  y.shape={tuple(y.shape)}  (expect [{K}])  case={cn}")
        xb = x.unsqueeze(0).to(device)
        yb = y.unsqueeze(0).to(device)
        with torch.no_grad():
            loss = _rollout_loss_k(
                model, xb, yb, [cn], K,
                physics_kwargs["dt"], cylinder_mass, damper, stiffness,
                0, 1, 2,
                torch.tensor(x_scaler.mean_, dtype=torch.float32, device=device),
                torch.tensor(x_scaler.scale_, dtype=torch.float32, device=device),
                physics_kwargs["y_mean"], physics_kwargs["y_scale"],
                1.0, D, fn, nn.MSELoss(), device,
            )
        print(f"  k={K} rollout loss (sum over steps) = {float(loss):.6f}  "
              f"[finite={np.isfinite(float(loss))}]")
        assert tuple(x.shape) == (int(cfg["seq_len"]), len(input_cols) + 1)
        assert tuple(y.shape) == (K,)
        assert np.isfinite(float(loss)), "non-finite rollout loss — stop, debug"
        print("[smoke] passed.\n")

    # ── single phase: k=10, 15 epochs, from raw baseline ───────────────────────
    ckpt_path = OUTPUT_DIR / f"gru_rollout_k{K}.pt"
    run_phase(
        model, optimizer,
        train_df_s, val_df_s,
        dataset_kwargs, physics_kwargs,
        rollout_k=K, epochs=EPOCHS,
        checkpoint_path=ckpt_path,
    )

    # ── save artifacts (scalers copied so eval loads from the right dir) ───────
    torch.save(model.state_dict(), OUTPUT_DIR / "gru_best.pt")
    with open(OUTPUT_DIR / "x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open(OUTPUT_DIR / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)
    print(f"\nDirect-k10 control saved to {OUTPUT_DIR}")
    print("Next: add ('gru_rollout_cylinder_re_1000_directk10', "
          "'gru_rollout_k10.pt', 'direct_k10') to evaluate_all.py CONFIGS and "
          "run the sweep. ur_stats still load from the baseline dir.")


if __name__ == "__main__":
    main()