#!/usr/bin/env python3
"""
Rollout-aware fine-tuning for the bridge no-acceleration GRU checkpoint
(Model A: C_L + trajectory rollout loss; Model B: + in-rollout work loss).

Context (see results/open_loop_energy_diagnostic/ and
results/closed_loop_diagnostic_Ur6.7385/ for the full diagnostic record):
the frozen no-acc checkpoint (results/gru_bridge_nd_context_noacc) is
pointwise-accurate under teacher forcing but its own closed-loop rollout
desynchronizes from the CFD lock-in frequency within 1-2s and the surrogate
aerodynamic force flips from exciting to damping within ~6.5-7s -- a failure
mode invisible to any loss computed only on CFD-prescribed histories. An
oracle-force Newmark replay confirmed the structural integrator itself is
not at fault (it reproduces the Fluent trajectory almost exactly given the
true force), so the fix targets the GRU's own predictions under
self-generated (closed-loop) history, not the structural solver.

Two loss variants, both starting from the SAME frozen checkpoint (weights,
x_scaler, y_scaler, ur_stats, and the checkpoint's own architecture/input
columns are all used AS-IS -- see --init_checkpoint; nothing about the base
model, the structural solver, or the scaler convention is changed here):

    Model A:  L = L_CL + lambda_roll * L_roll
    Model B:  L = L_CL + lambda_roll * L_roll + lambda_W * L_W,roll

  L_CL      pointwise MSE (scaled C_L space) between the rollout's own
            predicted C_L and CFD's C_L at matching absolute time.
  L_roll    trajectory-tracking MSE (nondimensional h/D, hdot/U) between the
            rollout's self-generated (h,hdot) and CFD's (h,hdot).
  L_W,roll  (Model B only) blockwise net-aerodynamic-work mismatch between
            the rollout's OWN self-generated (C_L,hdot) and CFD's, evaluated
            INSIDE the rollout (not on CFD-prescribed histories) -- see
            rollout_training.loss_W_roll for the exact formula (normalized
            by that block's gross exchanged energy).

Case list for sampling training/validation windows comes from the CURRENT,
approved 27-case manifest (train_gru.py::_bridge_split on the live cache /
results/bridge_dataset_manifest.json), NOT from the frozen checkpoint's own
(stale, pre-19.5-m/s-exclusion, 28-case) run_config.json -- see the dataset-
split audit in this diagnostic campaign for why that distinction matters.

Rollout curriculum (dt~0.002s, fn=0.32Hz, T_n~3.125s): the per-example
rollout LENGTH grows over the course of fine-tuning, from ~1-2s (capture the
early hidden-state departure) up to ~9-10s (~3 structural cycles, past the
~6.5-7s excitation-reversal point) -- see --stage_steps. Within a stage, if
the target length exceeds --tbptt_chunk, the rollout is split into
sequential truncated-BPTT chunks: each chunk gets its own forward + backward
+ optimizer step, after which the carried (window, h, hdot, hddot) state is
DETACHED (.detach()) before the next chunk begins -- gradients never flow
across a chunk boundary, only within one chunk. This is the only place
gradients are cut off; everything else (across steps within one chunk, and
through the Newmark integration) is fully differentiable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import torch

from viv_analysis.config import bridge_structural_params, config
from viv_analysis.models.gru import VIV_GRU
from viv_analysis.preprocess import load_bridge_df_cached
from viv_analysis.rollout_training import (
    ScalerConstants, build_batch_from_case, loss_cl, loss_roll, loss_W_roll,
    rollout_chunk, sample_batch_starts,
)
from viv_analysis.train_gru import _bridge_split
from viv_analysis.utils import PROJECT_ROOT, parse_ur_label

DEFAULT_STAGE_STEPS = [750, 1560, 3250, 4750]  # ~1.5s, ~1 cycle(3.125s), ~6.5s, ~9.5s at dt~0.002s
DEFAULT_ITERS_PER_STAGE = [40, 40, 20, 10]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state() -> dict:
    def _run(args):
        try:
            return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True).strip()
        except Exception as e:
            return f"<unavailable: {e}>"
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": _run(["git", "status", "--porcelain"]) != "",
    }


def load_frozen_checkpoint(checkpoint_dir: Path, device: str):
    with open(checkpoint_dir / "run_config.json") as f:
        rc = json.load(f)
    with open(checkpoint_dir / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(checkpoint_dir / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    with open(checkpoint_dir / "ur_stats.pkl", "rb") as f:
        ur_stats_pkl = pickle.load(f)

    input_cols = rc["input_cols"]
    use_ur_context = bool(rc["use_ur_context"])
    nd_inputs = bool(rc["nd_inputs"])
    input_size = len(input_cols) + (1 if use_ur_context else 0)

    model = VIV_GRU(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    ckpt_path = checkpoint_dir / "gru_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    return dict(
        model=model, x_scaler=x_scaler, y_scaler=y_scaler,
        ur_stats=(ur_stats_pkl["mean"], ur_stats_pkl["std"]),
        input_cols=input_cols, use_ur_context=use_ur_context, nd_inputs=nd_inputs,
        run_config=rc, checkpoint_sha256=_sha256(ckpt_path), checkpoint_path=str(ckpt_path),
    )


def load_canonical_bridge_data():
    """Live 27-case cache + the canonical 17/5/5 split -- see the dataset-
    split audit note in this module's docstring for why this must NOT come
    from a checkpoint's own run_config.json."""
    D = config["bridge_D_ref"]; fn = config["bridge_fn_hz"]
    params_bridge = bridge_structural_params()
    # load_bridge_df_cached runs the IDENTICAL merge_dataframes -> downsample
    # -> compute_kinematics(acc_source="force_residual" default) pipeline
    # train_gru.py's own bridge branch uses, but from the parquet cache
    # (data/cache/bridge_ds20_trim100_v2.parquet, already built with the
    # 19.5 m/s exclusion applied) instead of re-parsing raw .out files
    # (~1h cold-parse cost) -- see preprocess.py::load_bridge_df_cached.
    raw_df = load_bridge_df_cached(fn_hz=fn, d_ref=D, bridge_structural_params=params_bridge)

    # Mirrors train_gru.py's "quarantine short bridge cases" step exactly
    # (same threshold), so all_cases_unsplit below matches what a real
    # training run would see even if some case happens to be anomalously
    # short in the current cache.
    sizes = raw_df.groupby("case").size().sort_values()
    BRIDGE_MIN_FRAC = 0.05
    med = float(sizes.median())
    drop = set(sizes[sizes < BRIDGE_MIN_FRAC * med].index)
    if drop:
        print(f"Quarantining {len(drop)} bridge case(s): {sorted(drop)}")
        raw_df = raw_df[~raw_df["case"].isin(drop)].copy()

    all_cases = sorted((str(c) for c in raw_df["case"].drop_duplicates()), key=parse_ur_label)
    train_cases, val_cases, test_cases, release_time = _bridge_split(
        all_cases, fn_hz=fn, d_ref=D, t_star_release=config["bridge_t_star_release"],
    )
    assert "Ur8.2126" not in train_cases | val_cases | test_cases
    assert len(train_cases) == 17 and len(val_cases) == 5 and len(test_cases) == 5, (
        f"Expected the approved 17/5/5 split, got "
        f"{len(train_cases)}/{len(val_cases)}/{len(test_cases)} -- refusing to train "
        f"on an unexpected dataset composition."
    )
    return raw_df, sorted(train_cases), sorted(val_cases), sorted(test_cases), release_time


def run_stage_on_batch(
    model, batch: dict, stage_steps: int, tbptt_chunk: int,
    sc: ScalerConstants, q: float, D: float, m: float, c: float, k: float,
    input_cols: list[str], nd_inputs: bool, use_ur_context: bool,
    lambda_roll: float, lambda_W: float, block_steps_W: int,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    """Run one training example's stage-length rollout as a sequence of
    TBPTT chunks (each own forward+backward+step if optimizer is given;
    optimizer=None runs in no-grad validation mode instead). Returns mean
    per-chunk loss components for logging."""
    window = batch["window"]
    h_state, hdot_state, hddot_state = batch["h_state"], batch["hdot_state"], batch["hddot_state"]
    U, dt, ur_scaled = batch["U"], batch["dt"], batch["ur_scaled"]

    n_chunks = int(np.ceil(stage_steps / tbptt_chunk))
    logs = {"cl": [], "roll": [], "W": [], "total": []}

    for chunk_i in range(n_chunks):
        lo = chunk_i * tbptt_chunk
        hi = min(lo + tbptt_chunk, stage_steps)
        n_steps = hi - lo
        if n_steps <= 0:
            break

        ctx = torch.enable_grad() if optimizer is not None else torch.no_grad()
        with ctx:
            out = rollout_chunk(
                model=model, window=window, h_state=h_state, hdot_state=hdot_state,
                hddot_state=hddot_state, n_steps=n_steps, dt=dt, m=m, c=c, k=k, q=q,
                U=U, D=D, input_cols=input_cols, nd_inputs=nd_inputs, sc=sc,
                use_ur_context=use_ur_context, ur_scaled=ur_scaled,
            )
            cfd_h_sl = batch["cfd_h"][:, lo:hi]
            cfd_hdot_sl = batch["cfd_hdot"][:, lo:hi]
            cfd_cl_sl = batch["cfd_cl"][:, lo:hi]

            l_cl = loss_cl(out["cl_scaled"], cfd_cl_sl, sc.y_mean, sc.y_scale)
            l_roll = loss_roll(out["h"], out["hdot"], cfd_h_sl, cfd_hdot_sl, D=D, U=U,
                               x_mean=sc.x_mean, x_scale=sc.x_scale,
                               disp_idx=input_cols.index("disp"), vel_idx=input_cols.index("vel"))
            total = l_cl + lambda_roll * l_roll
            l_W = torch.tensor(0.0)
            if lambda_W > 0.0:
                block_steps = min(block_steps_W, n_steps)
                l_W = loss_W_roll(out["cl_phys"], out["hdot"], cfd_cl_sl, cfd_hdot_sl,
                                  q=q, c=c, dt=dt, block_steps=block_steps)
                total = total + lambda_W * l_W

        if optimizer is not None:
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        logs["cl"].append(float(l_cl.detach()))
        logs["roll"].append(float(l_roll.detach()))
        logs["W"].append(float(l_W.detach()) if torch.is_tensor(l_W) else float(l_W))
        logs["total"].append(float(total.detach()))

        # TBPTT: sever the graph at the chunk boundary, keep the numerics.
        window = out["window"].detach()
        h_state, hdot_state, hddot_state = (
            out["h_state"].detach(), out["hdot_state"].detach(), out["hddot_state"].detach())

    return {k: float(np.mean(v)) if v else float("nan") for k, v in logs.items()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_variant", choices=["A", "B"], required=True,
                  help="A: L_CL + lambda_roll*L_roll.  B: + lambda_W*L_W,roll.")
    p.add_argument("--init_checkpoint", default=str(PROJECT_ROOT / "results" / "gru_bridge_nd_context_noacc"),
                  help="Frozen checkpoint directory to fine-tune from (weights/scalers used AS-IS).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lambda_roll", type=float, default=1.0)
    p.add_argument("--lambda_W", type=float, default=0.1,
                  help="Only used when --model_variant B (Model A implicitly uses 0).")
    p.add_argument("--stage_steps", default=",".join(str(s) for s in DEFAULT_STAGE_STEPS),
                  help="Comma-separated rollout length (steps) per curriculum stage.")
    p.add_argument("--iters_per_stage", default=",".join(str(s) for s in DEFAULT_ITERS_PER_STAGE),
                  help="Comma-separated number of training examples (gradient-updated "
                       "rollouts) per curriculum stage.")
    p.add_argument("--tbptt_chunk", type=int, default=500,
                  help="Max rollout steps backpropagated through in one go; longer "
                       "stages are split into sequential detached chunks.")
    p.add_argument("--block_steps_W", type=int, default=1560,
                  help="loss_W_roll block size in steps (~one structural cycle, T_n=1/fn~3.125s).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_every", type=int, default=10,
                  help="Run validation every N training examples within a stage.")
    p.add_argument("--val_batch_size", type=int, default=1,
                  help="Val cases evaluated one at a time (deterministic start point).")
    p.add_argument("--smoke_test", action="store_true",
                  help="Override stage_steps/iters_per_stage with a tiny (~10-step, "
                       "1-iteration) config for a fast end-to-end sanity check. "
                       "Does not produce a usable checkpoint.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    lambda_W = args.lambda_W if args.model_variant == "B" else 0.0

    ckpt = load_frozen_checkpoint(Path(args.init_checkpoint), device)
    model, x_scaler, y_scaler = ckpt["model"], ckpt["x_scaler"], ckpt["y_scaler"]
    input_cols, use_ur_context, nd_inputs = ckpt["input_cols"], ckpt["use_ur_context"], ckpt["nd_inputs"]
    ur_stats = ckpt["ur_stats"]
    sc = ScalerConstants.from_sklearn(x_scaler, y_scaler)

    D = config["bridge_D_ref"]; fn = config["bridge_fn_hz"]; B_ref = config["bridge_B_ref"]
    rho = config["bridge_rho"]; seq_len = config["bridge_seq_len"]
    sp = bridge_structural_params(); m, c, k = sp["m"], sp["c"], sp["k"]

    print(f"Loaded init checkpoint: {ckpt['checkpoint_path']}  sha256={ckpt['checkpoint_sha256']}")
    print(f"  input_cols={input_cols}  use_ur_context={use_ur_context}  nd_inputs={nd_inputs}")

    raw_df, train_cases, val_cases, test_cases, release_time = load_canonical_bridge_data()
    print(f"Canonical split: train({len(train_cases)})={train_cases}")
    print(f"                 val({len(val_cases)})={val_cases}")

    stage_steps = [int(x) for x in args.stage_steps.split(",")]
    iters_per_stage = [int(x) for x in args.iters_per_stage.split(",")]
    if args.smoke_test:
        stage_steps = [20]
        iters_per_stage = [2]
        args.batch_size = 2
        args.tbptt_chunk = 10
        args.val_every = 1
        print("[SMOKE TEST] overriding curriculum to stage_steps=[20] iters=[2] "
              "batch_size=2 tbptt_chunk=10 -- NOT a usable checkpoint.")
    assert len(stage_steps) == len(iters_per_stage)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model_variant": args.model_variant,
        "init_checkpoint": ckpt["checkpoint_path"],
        "init_checkpoint_sha256": ckpt["checkpoint_sha256"],
        "lambda_roll": args.lambda_roll, "lambda_W": lambda_W,
        "stage_steps": stage_steps, "iters_per_stage": iters_per_stage,
        "tbptt_chunk": args.tbptt_chunk, "block_steps_W": args.block_steps_W,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "train_cases": train_cases, "val_cases": val_cases, "test_cases": test_cases,
        "dataset_manifest": "results/bridge_dataset_manifest.json (17/5/5 canonical split)",
        "input_cols": input_cols, "use_ur_context": use_ur_context, "nd_inputs": nd_inputs,
        "git": _git_state(), "smoke_test": args.smoke_test,
        "loss_formulas": {
            "L_CL": "MSE in scaled C_L space, rollout prediction vs CFD at matching time",
            "L_roll": "MSE in (h/D, hdot/U) between rollout trajectory and CFD",
            "L_W_roll": ("blockwise (W_net,pred - W_net,cfd)^2 / (S_W,b+eps)^2, "
                        "S_W,b = block gross exchanged energy of the rollout's own "
                        "F*hdot; see rollout_training.loss_W_roll"),
        },
        "tbptt_detach_points": "window/h_state/hdot_state/hddot_state detached at every "
                              "tbptt_chunk-step boundary within a stage; never detached "
                              "within a chunk; never carried across different training examples.",
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Run config saved to {output_dir/'run_config.json'}")

    q = 0.5 * rho * B_ref  # multiplied by U^2 per-case below (U varies by Ur)

    best_val = float("inf")
    val_history = []

    for stage_i, (n_steps, n_iters) in enumerate(zip(stage_steps, iters_per_stage)):
        print(f"\n=== Stage {stage_i+1}/{len(stage_steps)}: {n_steps} steps "
              f"(~{n_steps*0.002:.2f}s), {n_iters} training examples ===")
        for it in range(n_iters):
            case_name = train_cases[rng.integers(0, len(train_cases))]
            case_df = raw_df[raw_df["case"] == case_name].sort_values("time").reset_index(drop=True)
            U_case = parse_ur_label(case_name) * fn * D
            starts = sample_batch_starts(
                case_df, release_time[case_name], seq_len, n_steps, args.batch_size, rng)
            batch = build_batch_from_case(
                case_df, case_name, starts, seq_len, input_cols, x_scaler, nd_inputs, D, fn,
                use_ur_context, ur_stats, n_steps, device,
            )
            model.train()
            logs = run_stage_on_batch(
                model, batch, n_steps, args.tbptt_chunk, sc, q * U_case ** 2, D, m, c, k,
                input_cols, nd_inputs, use_ur_context, args.lambda_roll, lambda_W,
                args.block_steps_W, optimizer,
            )
            print(f"  stage={stage_i+1} it={it+1}/{n_iters} case={case_name:10s} "
                  f"total={logs['total']:.5f} cl={logs['cl']:.5f} roll={logs['roll']:.5f} "
                  f"W={logs['W']:.5f}")

            if (it + 1) % args.val_every == 0 or it == n_iters - 1:
                model.eval()
                val_logs = []
                for vc in val_cases:
                    vdf = raw_df[raw_df["case"] == vc].sort_values("time").reset_index(drop=True)
                    U_v = parse_ur_label(vc) * fn * D
                    try:
                        v_starts = sample_batch_starts(vdf, release_time[vc], seq_len, n_steps,
                                                       args.val_batch_size, np.random.default_rng(0))
                    except ValueError as e:
                        print(f"    [val skip] {vc}: {e}")
                        continue
                    v_batch = build_batch_from_case(
                        vdf, vc, v_starts, seq_len, input_cols, x_scaler, nd_inputs, D, fn,
                        use_ur_context, ur_stats, n_steps, device,
                    )
                    vl = run_stage_on_batch(
                        model, v_batch, n_steps, args.tbptt_chunk, sc, q * U_v ** 2, D, m, c, k,
                        input_cols, nd_inputs, use_ur_context, args.lambda_roll, lambda_W,
                        args.block_steps_W, optimizer=None,
                    )
                    val_logs.append(vl["total"])
                val_mean = float(np.mean(val_logs)) if val_logs else float("nan")
                val_history.append(dict(stage=stage_i + 1, it=it + 1, val_total=val_mean))
                print(f"    [val] mean total loss over {len(val_logs)} cases = {val_mean:.5f}")

                if val_mean < best_val and not args.smoke_test:
                    best_val = val_mean
                    torch.save(model.state_dict(), output_dir / "gru_best.pt")
                    print(f"    [val] new best ({best_val:.5f}), checkpoint saved")

    if args.smoke_test:
        print("\n[SMOKE TEST] complete -- no checkpoint saved (by design).")
        return

    if best_val == float("inf"):
        torch.save(model.state_dict(), output_dir / "gru_best.pt")
        print("\nNo validation improvement recorded; saved final model state as gru_best.pt.")

    with open(output_dir / "x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open(output_dir / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)
    with open(output_dir / "ur_stats.pkl", "wb") as f:
        pickle.dump({"mean": ur_stats[0], "std": ur_stats[1], "use_ur_context": use_ur_context,
                    "nd_inputs": nd_inputs, "coordinate_mode": "nondimensional" if nd_inputs else "dimensional",
                    "cfd_dataset": "bridge", "exp_subdir": output_dir.name}, f)
    with open(output_dir / "val_history.json", "w") as f:
        json.dump(val_history, f, indent=2)
    print(f"\nDone. best_val={best_val:.5f}. Artifacts in {output_dir}")


if __name__ == "__main__":
    main()
