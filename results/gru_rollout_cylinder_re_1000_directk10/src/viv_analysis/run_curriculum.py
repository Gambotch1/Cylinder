# run_curriculum.py
import copy
import argparse
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from viv_analysis.config import config, prepare_gru_config
from viv_analysis.models.gru import VIV_GRU, VIVSequenceDataset, apply_scalers_to_df, fit_scalers
from viv_analysis.preprocess import merge_dataframes, compute_kinematics
from viv_analysis.train_gru import (
    train_rollout_epoch, eval_rollout_epoch, split_cases,
)
from viv_analysis.utils import PROJECT_ROOT







def save_ckpt(path, model, optimizer, phase, epoch, rollout_k):
    torch.save(
        {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "phase":     phase,
            "epoch":     epoch,
            "rollout_k": rollout_k,
        },
        path,
    )


def run_phase(
    model, optimizer,
    train_df_s, val_df_s,
    dataset_kwargs, physics_kwargs,
    rollout_k, epochs, checkpoint_path,
):
    """One curriculum phase at a fixed rollout horizon."""
    batch_size   = physics_kwargs["batch_size"]
    stride_train = physics_kwargs["stride_train"]

    # Issue 3 fix: only dataset keys go to VIVSequenceDataset
    train_ds = VIVSequenceDataset(
        train_df_s, stride=stride_train, rollout_k=rollout_k, **dataset_kwargs)
    val_ds = VIVSequenceDataset(
        val_df_s, stride=1, rollout_k=rollout_k, **dataset_kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    epoch_kwargs = dict(
        criterion=physics_kwargs["criterion"],
        device=physics_kwargs["device"],
        rollout_k=rollout_k,
        dt=physics_kwargs["dt"],
        m=physics_kwargs["m"],
        c=physics_kwargs["c"],
        k_phys=physics_kwargs["k_phys"],
        idx_h=physics_kwargs["idx_h"],
        idx_hdot=physics_kwargs["idx_hdot"],
        idx_hddot=physics_kwargs["idx_hddot"],
        expected_features=physics_kwargs["expected_features"],
        x_mean=physics_kwargs["x_mean"],
        x_scale=physics_kwargs["x_scale"],
        y_mean=physics_kwargs["y_mean"],
        y_scale=physics_kwargs["y_scale"],
        rho=physics_kwargs["rho"],
        D=physics_kwargs["D"],
        fn=physics_kwargs["fn"],
    )


    # Issue 1 fix: track validation, save best state per phase
    best_val   = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_rollout_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            max_grad_norm=1.0, **epoch_kwargs,
        )
        val_loss = eval_rollout_epoch(
            model=model, loader=val_loader, **epoch_kwargs,
        )
        print(f"  k={rollout_k}  epoch={epoch}/{epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())

    # Restore best from this phase before moving to next k
    model.load_state_dict(best_state)
    save_ckpt(checkpoint_path, model, optimizer,
              phase=f"k={rollout_k}", epoch=epochs, rollout_k=rollout_k)
    print(f"  Phase k={rollout_k} complete — best_val={best_val:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="cylinder1000",
                        help="Dataset key (case-insensitive), e.g. cylinder1000")
    parser.add_argument("--mode", choices=["curriculum", "direct"], default="curriculum",
                        help="Train the full curriculum or a direct fixed-k run")
    parser.add_argument("--rollout_k", type=int, default=10,
                        help="Rollout horizon for direct mode")
    args = parser.parse_args()

    dataset = args.dataset.strip().lower()
    OUTPUT_DIR = PROJECT_ROOT / "results" / (
        f"gru_rollout_{dataset}" if args.mode == "curriculum"
        else f"gru_direct_k{args.rollout_k}_{dataset}"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg    = prepare_gru_config(dataset, config)

    input_cols = ["disp", "vel", "acc"]
    cfg["input_cols"]     = input_cols
    cfg["use_ur_context"] = True
    use_ur_context = bool(cfg.get("use_ur_context", False))

    artifact_dir = PROJECT_ROOT / "results" / f"gru_{dataset}"

    # ── Load model + scalers + ur_stats ───────────────────────────────────
    with open(artifact_dir / "ur_stats.pkl", "rb") as f:
        ur_stats = pickle.load(f)
    with open(artifact_dir / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(artifact_dir / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    # ── Load and preprocess ────────────────────────────────────────────────
    raw_df = merge_dataframes(dataset=dataset)

    D       = config["cylinder1000_D_ref"]
    fn      = 0.2
    omega_n = 2 * np.pi * fn
    M_star, zeta = 2.0, 0.007
    cylinder_mass = M_star * 1.0 * (np.pi * (D / 2) ** 2)
    damper        = cylinder_mass * 2 * omega_n * zeta
    stiffness     = cylinder_mass * omega_n ** 2

    structural_params = {
        "m": cylinder_mass, "c": damper, "k": stiffness,
        "cylinder_mass": cylinder_mass, "c_struct": damper, "k_struct": stiffness,
    }
    raw_df = compute_kinematics(raw_df, dataset=dataset,
                                structural_params=structural_params)

    all_cases = sorted(str(c) for c in raw_df["case"].drop_duplicates())
    train_cases, val_cases, _, release_time = split_cases(
        all_cases, dataset, cfg)

    train_df = raw_df[raw_df["case"].isin(train_cases)].copy()
    val_df   = raw_df[raw_df["case"].isin(val_cases)].copy()

    # Issue 4 check: confirm split is applied correctly
    print(f"train cases: {sorted(train_cases)}")
    print(f"val   cases: {sorted(val_cases)}")
    assert set(train_df["case"].unique()) <= train_cases, \
        "train_df contains val/test cases — check split logic"

    train_df_s = apply_scalers_to_df(
        train_df, x_scaler, y_scaler, input_cols, cfg["target_col"])
    val_df_s   = apply_scalers_to_df(
        val_df,   x_scaler, y_scaler, input_cols, cfg["target_col"])

    # ── Separate kwarg dicts (Issue 3 fix) ────────────────────────────────
    dataset_kwargs = dict(
        seq_len     = int(cfg["seq_len"]),
        target_col  = str(cfg["target_col"]),
        input_cols  = input_cols,
        release_time= release_time,
        use_ur_context = use_ur_context,
        ur_mean=  ur_stats["mean"],
        ur_std= ur_stats["std"],
    )

    physics_kwargs = dict(
        dt=0.005,
        m=cylinder_mass,
        c=damper,
        k_phys=stiffness,
        rho=1.0,
        D=D,
        fn=fn,
        idx_h=0, idx_hdot=1, idx_hddot=2,
        expected_features=len(input_cols) + (1 if use_ur_context else 0),
        x_mean=x_scaler.mean_,
        x_scale=x_scaler.scale_,
        y_mean=float(np.asarray(y_scaler.mean_)[0]),
        y_scale=float(np.asarray(y_scaler.scale_)[0]),
        criterion=nn.MSELoss(),
        device=device,
        batch_size=int(cfg["batch_size"]),
        stride_train=int(cfg["stride_train"]),
    )

    

    # ── Model — optionally warm-start from baseline checkpoint ────────────
    model = VIV_GRU(
        input_size  = len(input_cols) + (1 if use_ur_context else 0),
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
    ).to(device)

    baseline_ckpt = PROJECT_ROOT / "results" / f"gru_{dataset}" / "gru_best.pt"
    if baseline_ckpt.exists():
        model.load_state_dict(torch.load(baseline_ckpt, map_location=device))
        print(f"Warm-started from {baseline_ckpt}")
    else:
        print("No baseline checkpoint found — training from scratch.")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    if args.mode == "curriculum":
        # ── Curriculum (structural fix: actually call run_phase) ──────────
        curriculum = [
            (1,  10),   # warm-up: identical to one-step baseline
            (2,   5),
            (5,   5),
            (10, 15),
        ]

        for k, n_epochs in curriculum:
            # Issue 2 fix: halve LR at each phase transition (keep Adam moments)
            if k > 1:
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] * 0.5

            ckpt_path = OUTPUT_DIR / f"gru_rollout_k{k}.pt"
            run_phase(
                model, optimizer,
                train_df_s, val_df_s,
                dataset_kwargs, physics_kwargs,
                rollout_k=k, epochs=n_epochs,
                checkpoint_path=ckpt_path,
            )
    else:
        # Direct training-duration control: no curriculum, fixed rollout horizon.
        ckpt_path = OUTPUT_DIR / f"gru_rollout_k{args.rollout_k}.pt"
        run_phase(
            model, optimizer,
            train_df_s, val_df_s,
            dataset_kwargs, physics_kwargs,
            rollout_k=args.rollout_k, epochs=int(cfg["n_epochs"]),
            checkpoint_path=ckpt_path,
        )

    sample_k = 1 if args.mode == "curriculum" else int(args.rollout_k)
    sample_ds = VIVSequenceDataset(
        train_df_s,
        stride=int(cfg["stride_train"]),
        rollout_k=sample_k,
        **dataset_kwargs,
    )
    sample_idx = min(100, len(sample_ds) - 1)
    x, y, case_name = sample_ds[sample_idx]
    print(f"Case: {case_name}")
    print(f"x.shape: {x.shape} (should be [seq_len, input_size])")
    print(f"y.shape: {y.shape} (should be [{sample_k}])")
    print(f"x last row (kinematics at window-end time): {x[-1]}")
    print(f"y[0] (CL target): {y if y.dim() == 0 else y[0]}")

    # ── Save final artifacts ───────────────────────────────────────────────
    torch.save(model.state_dict(), OUTPUT_DIR / "gru_best.pt")
    with open(OUTPUT_DIR / "x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open(OUTPUT_DIR / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)
    print(f"\nFinal model saved to {OUTPUT_DIR}/gru_best.pt")


if __name__ == "__main__":
    main()
