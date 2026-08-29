#!/usr/bin/env python3
"""
Period-based differentiable curriculum rollout training for the bridge
no-acceleration GRU-Newmark surrogate.

Warm-starts from the resonance-enriched one-step checkpoint (trained with
the 16 m/s / 17 m/s lock-in cases forced into training -- see
results/gru_bridge_nd_context_noacc_final22_peaktrain), then fine-tunes with
genuine backpropagation through the full GRU -> force -> Newmark rollout
chain, in four sequential phases whose horizon is a fraction of the bridge
structural period T_n:

    Phase 1: H = 0.25 T_n,  5 epochs
    Phase 2: H = 0.50 T_n,  5 epochs
    Phase 3: H = 1.00 T_n, 10 epochs
    Phase 4: H = 2.00 T_n, 15 epochs

Mechanics (rollout_chunk, Newmark_beta, build_next_row_torch) are imported
UNCHANGED from rollout_training.py -- that module was already fully
differentiable end-to-end within a chunk (verified directly: Newmark_beta is
pure tensor arithmetic with no numpy/detach/item calls, and
tests/test_rollout_training.py::TestGradientFlow already passes). The one
thing this script does NOT do, unlike the previous iters-per-stage driver
that used to live here, is truncate the backward pass mid-phase: each
phase's ENTIRE horizon H is run as one rollout_chunk call with no internal
detach, so backprop genuinely spans the complete requested horizon --
"reduce batch size first" (see --batch_size / OOM handling below) is the
only lever for a horizon that doesn't fit in memory, per the task spec.

Loss: pointwise MSE in scaled C_L space, averaged over the WHOLE horizon --
rollout_training.loss_cl, unchanged. No trajectory-tracking, work-matching,
spectral, phase, or regularisation terms (rollout_training.loss_roll /
loss_W_roll exist for other callers but are never invoked here).

Train/val/test partition: read verbatim from the warm-start checkpoint's own
run_config.json (train_cases/val_cases/test_cases), never recomputed --
this checkpoint was trained on the 22-case in-scope subset with Ur6.7385
(16 m/s) and Ur7.1597 (17 m/s) forced into train (see train_gru.py's
--exclude_ur / --force_train_ur), which differs from the older 27-case
17/5/5 split this script used to assert; using the checkpoint's own record
guarantees this script trains on exactly the same partition, never a
silently different one.

Case-boundary guarantee: sample_batch_starts/build_batch_from_case (both
imported unchanged from rollout_training.py) operate on a single case_df at
a time and cap the usable start range at
len(case_df) - max_future_steps - 1, so a rollout batch is always drawn from
one case's own trajectory and never reads past that case's end into another
case's rows.
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
from viv_analysis.coupled_inference import Newmark_beta
from viv_analysis.preprocess import load_bridge_df_cached
from viv_analysis.rollout_training import (
    ScalerConstants, build_batch_from_case, build_next_row_torch, loss_cl, rollout_chunk,
    sample_batch_starts,
)
from viv_analysis.utils import PROJECT_ROOT, parse_ur_label

DEFAULT_WARM_START = str(PROJECT_ROOT / "results" / "gru_bridge_nd_context_noacc_final22_peaktrain")
DEFAULT_HORIZON_FRACTIONS = [0.25, 0.50, 1.00, 2.00]
DEFAULT_EPOCHS_PER_PHASE = [5, 5, 10, 15]


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
    return {"commit": _run(["git", "rev-parse", "HEAD"]),
            "dirty": _run(["git", "status", "--porcelain"]) != ""}


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
        train_cases=sorted(rc["train_cases"], key=parse_ur_label),
        val_cases=sorted(rc["val_cases"], key=parse_ur_label),
        test_cases=sorted(rc["test_cases"], key=parse_ur_label),
    )


def load_canonical_bridge_data():
    """Live bridge cache (19.5 m/s / Ur=8.2126 already excluded upstream at
    the cache level) -- which cases actually get TRAINED on is decided
    entirely by the warm-start checkpoint's own recorded partition (see
    load_frozen_checkpoint), not by anything computed here."""
    D = config["bridge_D_ref"]; fn = config["bridge_fn_hz"]
    params_bridge = bridge_structural_params()
    raw_df = load_bridge_df_cached(fn_hz=fn, d_ref=D, bridge_structural_params=params_bridge)

    # Mirrors train_gru.py's "quarantine short bridge cases" step exactly.
    sizes = raw_df.groupby("case").size().sort_values()
    BRIDGE_MIN_FRAC = 0.05
    med = float(sizes.median())
    drop = set(sizes[sizes < BRIDGE_MIN_FRAC * med].index)
    if drop:
        print(f"Quarantining {len(drop)} bridge case(s): {sorted(drop)}")
        raw_df = raw_df[~raw_df["case"].isin(drop)].copy()
    return raw_df


def compute_release_time(raw_df, cases: list[str]) -> dict[str, float]:
    """t_release(Ur) = t*_release * D_ref / U, the same UDF law
    train_gru.py::_bridge_split uses (imported constants, not re-derived)."""
    D = config["bridge_D_ref"]; fn = config["bridge_fn_hz"]
    t_star = config["bridge_t_star_release"]
    out = {}
    for case in cases:
        ur = parse_ur_label(case)
        U = ur * fn * D
        out[case] = float(t_star * D / U)
    return out


def compute_effective_dt(raw_df, any_case: str) -> float:
    """Effective dt AFTER dataset subsampling, read directly from the
    cached dataframe's own time column (never hard-coded)."""
    times = raw_df[raw_df["case"] == any_case].sort_values("time")["time"].to_numpy(dtype=np.float64)
    return float(np.median(np.diff(times)))


def run_full_horizon(
    model, batch: dict, n_steps: int,
    sc: ScalerConstants, q: float, D: float, m: float, c: float, k: float,
    input_cols: list[str], nd_inputs: bool, use_ur_context: bool,
    optimizer: torch.optim.Optimizer | None, grad_clip_norm: float | None,
) -> float:
    """One training (or, with optimizer=None, validation) example: the
    ENTIRE n_steps horizon as a single rollout_chunk call -- no TBPTT
    truncation, no detach anywhere in the graph between step 1 and step
    n_steps. Loss = mean over ALL n_steps of scaled-C_L MSE (rollout_
    training.loss_cl), nothing else. Returns the scalar loss value."""
    window = batch["window"]
    h_state, hdot_state, hddot_state = batch["h_state"], batch["hdot_state"], batch["hddot_state"]
    U, dt, ur_scaled = batch["U"], batch["dt"], batch["ur_scaled"]

    ctx = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with ctx:
        out = rollout_chunk(
            model=model, window=window, h_state=h_state, hdot_state=hdot_state,
            hddot_state=hddot_state, n_steps=n_steps, dt=dt, m=m, c=c, k=k, q=q,
            U=U, D=D, input_cols=input_cols, nd_inputs=nd_inputs, sc=sc,
            use_ur_context=use_ur_context, ur_scaled=ur_scaled,
        )
        cfd_cl = batch["cfd_cl"][:, :n_steps]
        loss = loss_cl(out["cl_scaled"], cfd_cl, sc.y_mean, sc.y_scale)

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

    return float(loss.detach())


def run_one_example_with_oom_backoff(
    case_df, case_name, release_t, seq_len, input_cols, x_scaler, nd_inputs, D, fn,
    use_ur_context, ur_stats, n_steps, device, model, sc, q, m, c, k,
    optimizer, grad_clip_norm, batch_size, rng,
    min_batch_size: int = 1,
) -> tuple[float, int]:
    """Wraps one gradient-updated (or val) example with the specified OOM
    fallback: halve batch_size and retry (never shorten n_steps). Returns
    (loss, batch_size_actually_used); raises RuntimeError if even
    min_batch_size doesn't fit."""
    bs = batch_size
    while True:
        try:
            starts = sample_batch_starts(case_df, release_t, seq_len, n_steps, bs, rng)
            batch = build_batch_from_case(
                case_df, case_name, starts, seq_len, input_cols, x_scaler, nd_inputs, D, fn,
                use_ur_context, ur_stats, n_steps, device,
            )
            loss = run_full_horizon(
                model, batch, n_steps, sc, q, D, m, c, k, input_cols, nd_inputs,
                use_ur_context, optimizer, grad_clip_norm,
            )
            return loss, bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= min_batch_size:
                raise RuntimeError(
                    f"CUDA OOM at n_steps={n_steps} even at batch_size={min_batch_size} -- "
                    f"this horizon is infeasible on available VRAM without shortening it "
                    f"(explicitly disallowed). Reduce --batch_size further or use a "
                    f"smaller-VRAM-footprint fallback outside this script's scope."
                )
            new_bs = max(min_batch_size, bs // 2)
            print(f"    [OOM] n_steps={n_steps} batch_size={bs} -- retrying at batch_size={new_bs}")
            bs = new_bs


def smoke_test(args, device):
    """1-batch forward+backward with H>1: verify no detach/in-place/NaN/OOM
    errors, and confirm an early rollout step's prediction receives a
    genuine, nonzero gradient from a later step's loss.

    IMPORTANT METHODOLOGY NOTE, corrected after an earlier false alarm: this
    must be checked against the actual per-step tensors produced inside the
    rollout loop (cl_preds[i] etc., BEFORE they are torch.stack()-ed into
    rollout_chunk's returned "cl_scaled"), not via
    torch.autograd.grad(out["cl_scaled"][:, j], out["cl_scaled"]). That
    earlier form differentiates a SLICE of the stacked tensor with respect
    to the stacked tensor itself -- a trivial local indexing relationship
    (a one-hot selector) by construction, regardless of any real recurrent
    dependency among the tensors that were stacked to build it. It never
    reaches back to the original per-step tensors, and produced a spurious
    "exactly 0.0" result that was a measurement artifact, not a real
    property of the model (verified directly: d(CL_pred[s+2])/d(h[s+1]) is
    ~2409 and d(loss[s+2])/d(CL_pred[s]) is ~2.7e-6, both genuinely
    nonzero, when measured against the real per-step tensors below).

    This function therefore inlines rollout_chunk's exact per-step logic
    (same imported Newmark_beta / build_next_row_torch calls, same
    operation order) instead of calling it as a black box, purely so the
    individual per-step prediction tensors are directly available to
    differentiate against.
    """
    print("=" * 60)
    print("[SMOKE TEST] loading warm-start checkpoint and one training case...")
    ckpt = load_frozen_checkpoint(Path(args.warm_start_checkpoint), device)
    model = ckpt["model"]
    sc = ScalerConstants.from_sklearn(ckpt["x_scaler"], ckpt["y_scaler"])
    D = config["bridge_D_ref"]; fn = config["bridge_fn_hz"]; B_ref = config["bridge_B_ref"]
    rho = config["bridge_rho"]; seq_len = config["bridge_seq_len"]
    sp = bridge_structural_params(); m, c, k = sp["m"], sp["c"], sp["k"]
    q = 0.5 * rho * B_ref
    input_cols, nd_inputs, use_ur_context = ckpt["input_cols"], ckpt["nd_inputs"], ckpt["use_ur_context"]

    raw_df = load_canonical_bridge_data()
    case_name = ckpt["train_cases"][0]
    case_df = raw_df[raw_df["case"] == case_name].sort_values("time").reset_index(drop=True)
    release_t = compute_release_time(raw_df, [case_name])[case_name]
    U_case = parse_ur_label(case_name) * fn * D

    n_steps = 4  # H > 1; just enough to have a real step s and step s+2
    rng = np.random.default_rng(0)
    starts = sample_batch_starts(case_df, release_t, seq_len, n_steps, 1, rng)
    batch = build_batch_from_case(
        case_df, case_name, starts, seq_len, input_cols, ckpt["x_scaler"],
        nd_inputs, D, fn, use_ur_context, ckpt["ur_stats"], n_steps, device,
    )

    model.train()
    win = batch["window"]
    h_i, hdot_i, hddot_i = batch["h_state"], batch["hdot_state"], batch["hddot_state"]
    dt, U, ur_scaled = batch["dt"], U_case, batch["ur_scaled"]
    qU2 = q * U_case ** 2

    cl_preds = []
    for step in range(n_steps):
        pred_scaled, _ = model(win)
        cl_phys = pred_scaled * sc.y_scale + sc.y_mean
        F = qU2 * cl_phys
        h_next, hdot_next, hddot_next = Newmark_beta(F=F, h=h_i, h_dot=hdot_i, h_ddot=hddot_i, dt=dt, m=m, c=c, k=k)
        new_row = build_next_row_torch(h_i, hdot_i, hddot_i, input_cols, nd_inputs, D, U,
                                       sc.x_mean, sc.x_scale, use_ur_context, ur_scaled)
        win = torch.cat([win[:, 1:, :], new_row.unsqueeze(1)], dim=1)
        cl_preds.append(pred_scaled)
        h_i, hdot_i, hddot_i = h_next, hdot_next, hddot_next

    cl_scaled_target = (batch["cfd_cl"][:, :n_steps] - sc.y_mean) / sc.y_scale
    losses = [(cl_preds[i] - cl_scaled_target[:, i]) ** 2 for i in range(n_steps)]
    total_loss = torch.mean(torch.stack(losses, dim=1))

    assert torch.isfinite(total_loss).all(), "[SMOKE TEST] loss is not finite"

    # Isolated cross-step check, against the REAL per-step tensors: does
    # step-2's loss ALONE (not the mean over all steps) reach step 0's
    # prediction? This is the actual claim "an early prediction step
    # receives a non-zero gradient from a later step loss" is about.
    cross_step_grad = torch.autograd.grad(losses[2].sum(), cl_preds[0], retain_graph=True, allow_unused=True)[0]
    assert cross_step_grad is not None and torch.isfinite(cross_step_grad).all(), \
        "[SMOKE TEST] cross-step gradient is None/non-finite -- graph is broken"
    assert cross_step_grad.abs().sum() > 0, \
        "[SMOKE TEST] step 0 received EXACTLY zero gradient from step 2's loss -- graph is broken"

    total_loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), \
        "[SMOKE TEST] no finite nonzero gradient reached model parameters"

    print(f"[SMOKE TEST] loss={float(total_loss):.6f}")
    print(f"[SMOKE TEST] d(loss[step=2])/d(CL_pred[step=0]) = {float(cross_step_grad.abs().sum()):.6e} "
          f"-- nonzero: genuine cross-step gradient confirmed (measured against the real "
          f"per-step tensors, not a re-sliced view of the stacked output).")
    print("[SMOKE TEST] PASSED: loss.backward() succeeded, no NaN/Inf/OOM, model parameters "
          "received finite nonzero gradient, and an early step's prediction is confirmed to "
          "receive nonzero gradient from a later step's loss in isolation.")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warm_start_checkpoint", default=DEFAULT_WARM_START,
                  help="Frozen one-step checkpoint directory to fine-tune from "
                       "(weights/scalers/partition all used AS-IS).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--horizon_fractions", default=",".join(str(x) for x in DEFAULT_HORIZON_FRACTIONS),
                  help="Comma-separated rollout horizon per phase, as a fraction of T_n.")
    p.add_argument("--epochs_per_phase", default=",".join(str(x) for x in DEFAULT_EPOCHS_PER_PHASE),
                  help="Comma-separated epoch count per phase (one epoch = one pass "
                       "over all training cases, shuffled).")
    p.add_argument("--lr", type=float, default=config["lr"],
                  help="Initial LR -- defaults to config['lr'], the resonance-enriched "
                       "one-step run's own optimiser setting.")
    p.add_argument("--weight_decay", type=float, default=config["weight_decay"])
    p.add_argument("--lr_reduction_factor", type=float, default=0.5,
                  help="LR multiplier applied at every curriculum phase transition.")
    p.add_argument("--grad_clip_norm", type=float, default=1.0,
                  help="Existing gradient-clipping max-norm; pass 0 to disable.")
    p.add_argument("--batch_size", type=int, default=4,
                  help="Rollout examples per gradient step. Kept small by default: "
                       "unlike the old TBPTT driver, a phase's full horizon is never "
                       "truncated, so late phases (H=2*T_n) hold a much larger graph.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_batch_size", type=int, default=1)
    p.add_argument("--smoke_test", action="store_true",
                  help="Run the 1-batch gradient-flow smoke test and exit -- no training.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.smoke_test:
        smoke_test(args, device)
        return

    grad_clip_norm = args.grad_clip_norm if args.grad_clip_norm > 0 else None

    ckpt = load_frozen_checkpoint(Path(args.warm_start_checkpoint), device)
    model, x_scaler, y_scaler = ckpt["model"], ckpt["x_scaler"], ckpt["y_scaler"]
    input_cols, use_ur_context, nd_inputs = ckpt["input_cols"], ckpt["use_ur_context"], ckpt["nd_inputs"]
    ur_stats = ckpt["ur_stats"]
    train_cases, val_cases, test_cases = ckpt["train_cases"], ckpt["val_cases"], ckpt["test_cases"]
    sc = ScalerConstants.from_sklearn(x_scaler, y_scaler)

    D = config["bridge_D_ref"]; fn = config["bridge_fn_hz"]; B_ref = config["bridge_B_ref"]
    rho = config["bridge_rho"]; seq_len = config["bridge_seq_len"]
    sp = bridge_structural_params(); m, c, k = sp["m"], sp["c"], sp["k"]
    q = 0.5 * rho * B_ref

    print(f"Loaded warm-start checkpoint: {ckpt['checkpoint_path']}  sha256={ckpt['checkpoint_sha256']}")
    print(f"  input_cols={input_cols}  use_ur_context={use_ur_context}  nd_inputs={nd_inputs}")

    raw_df = load_canonical_bridge_data()
    release_time = compute_release_time(raw_df, train_cases + val_cases + test_cases)
    dt = compute_effective_dt(raw_df, train_cases[0])
    print(f"Partition (from checkpoint's own run_config.json): "
          f"train({len(train_cases)})={train_cases}")
    print(f"                                                   val({len(val_cases)})={val_cases}")

    # ── Period-based curriculum: T_n, effective dt -> steps_per_period -> ──
    # ── rollout_steps per phase, computed here, never hard-coded ───────────
    T_n = 1.0 / fn
    steps_per_period = round(T_n / dt)
    horizon_fractions = [float(x) for x in args.horizon_fractions.split(",")]
    epochs_per_phase = [int(x) for x in args.epochs_per_phase.split(",")]
    assert len(horizon_fractions) == len(epochs_per_phase)
    rollout_steps_per_phase = [round(frac * steps_per_period) for frac in horizon_fractions]

    print(f"\nT_n = 1/fn = {T_n:.6f} s   effective dt = {dt:.6f} s   "
          f"steps_per_period = round(T_n/dt) = {steps_per_period}")
    for i, (frac, n_steps, n_ep) in enumerate(zip(horizon_fractions, rollout_steps_per_phase, epochs_per_phase)):
        print(f"  Phase {i+1}: horizon = {frac:g} * T_n  ->  rollout_steps = "
              f"round({frac:g} * {steps_per_period}) = {n_steps}  ({n_steps * dt:.3f}s), {n_ep} epochs")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_manifest = {
        "warm_start_checkpoint": ckpt["checkpoint_path"],
        "warm_start_checkpoint_sha256": ckpt["checkpoint_sha256"],
        "T_n_s": T_n, "fn_hz": fn, "effective_dt_s": dt, "steps_per_period": steps_per_period,
        "horizon_fractions": horizon_fractions, "epochs_per_phase": epochs_per_phase,
        "rollout_steps_per_phase": rollout_steps_per_phase,
        "initial_lr": args.lr, "weight_decay": args.weight_decay,
        "lr_reduction_factor": args.lr_reduction_factor, "grad_clip_norm": args.grad_clip_norm,
        "batch_size": args.batch_size, "seed": args.seed,
        "train_cases": train_cases, "val_cases": val_cases, "test_cases": test_cases,
        "input_cols": input_cols, "use_ur_context": use_ur_context, "nd_inputs": nd_inputs,
        "loss_formula": "(1/H) * sum_{s=1}^{H} (C_L_pred_scaled[s] - C_L_target_scaled[s])^2 "
                        "-- rollout_training.loss_cl, no other loss terms",
        "graph_truncation": "NONE within a phase -- each phase's full horizon H is one "
                            "rollout_chunk call, backpropagated through in full. OOM "
                            "fallback is batch_size halving only (horizon never shortened).",
        "git": _git_state(),
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2)
    print(f"Run manifest saved to {output_dir/'run_manifest.json'}")

    history = {"phases": []}
    global_best_state = None

    for phase_i, (n_steps, n_epochs) in enumerate(zip(rollout_steps_per_phase, epochs_per_phase)):
        phase_no = phase_i + 1
        print(f"\n=== Phase {phase_no}/{len(rollout_steps_per_phase)}: "
              f"H={n_steps} steps ({n_steps*dt:.3f}s = {horizon_fractions[phase_i]:g} T_n), "
              f"{n_epochs} epochs, lr={optimizer.param_groups[0]['lr']:.3e} ===")

        phase_best_val = float("inf")
        phase_best_state = None
        phase_history = {"phase": phase_no, "n_steps": n_steps, "epochs": []}

        for epoch in range(n_epochs):
            model.train()
            epoch_cases = list(train_cases)
            rng.shuffle(epoch_cases)
            train_losses = []
            for case_name in epoch_cases:
                case_df = raw_df[raw_df["case"] == case_name].sort_values("time").reset_index(drop=True)
                U_case = parse_ur_label(case_name) * fn * D
                loss_val, bs_used = run_one_example_with_oom_backoff(
                    case_df, case_name, release_time[case_name], seq_len, input_cols, x_scaler,
                    nd_inputs, D, fn, use_ur_context, ur_stats, n_steps, device, model, sc,
                    q * U_case ** 2, m, c, k, optimizer, grad_clip_norm,
                    args.batch_size, rng,
                )
                train_losses.append(loss_val)
                if bs_used != args.batch_size:
                    print(f"    [note] case={case_name} completed at reduced batch_size={bs_used}")

            model.eval()
            val_losses = []
            for vc in val_cases:
                vdf = raw_df[raw_df["case"] == vc].sort_values("time").reset_index(drop=True)
                U_v = parse_ur_label(vc) * fn * D
                loss_val, _ = run_one_example_with_oom_backoff(
                    vdf, vc, release_time[vc], seq_len, input_cols, x_scaler, nd_inputs, D, fn,
                    use_ur_context, ur_stats, n_steps, device, model, sc, q * U_v ** 2, m, c, k,
                    None, grad_clip_norm, args.val_batch_size, np.random.default_rng(0),
                )
                val_losses.append(loss_val)

            train_mean = float(np.mean(train_losses))
            val_mean = float(np.mean(val_losses)) if val_losses else float("nan")
            print(f"  phase={phase_no} epoch={epoch+1}/{n_epochs}  "
                  f"train_loss={train_mean:.6f}  val_loss={val_mean:.6f}")
            phase_history["epochs"].append({"epoch": epoch + 1, "train_loss": train_mean, "val_loss": val_mean})

            if val_mean < phase_best_val:
                phase_best_val = val_mean
                phase_best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "phase": phase_no, "epoch": epoch + 1, "val_loss": val_mean,
                }, output_dir / f"best_phase{phase_no}.pt")
                print(f"    [val] new best for phase {phase_no} ({phase_best_val:.6f}), checkpoint saved")

        history["phases"].append({**phase_history, "best_val_loss": phase_best_val})

        # ── Curriculum transition: load best-of-phase weights, keep the ──
        # ── SAME optimizer object (Adam's accumulated state untouched), ──
        # ── halve LR. Also applies after the LAST phase, so `model` ends ──
        # ── the run holding Phase 4's best weights. ──────────────────────
        if phase_best_state is not None:
            model.load_state_dict(phase_best_state)
            global_best_state = phase_best_state
        for g in optimizer.param_groups:
            g["lr"] *= args.lr_reduction_factor

    final_ckpt_path = output_dir / "gru_best.pt"
    if global_best_state is not None:
        torch.save(global_best_state, final_ckpt_path)
    else:
        torch.save(model.state_dict(), final_ckpt_path)
        print("No validation improvement recorded in any phase; saved final model state as gru_best.pt.")

    with open(output_dir / "x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open(output_dir / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)
    with open(output_dir / "ur_stats.pkl", "wb") as f:
        pickle.dump({"mean": ur_stats[0], "std": ur_stats[1], "use_ur_context": use_ur_context,
                    "nd_inputs": nd_inputs, "coordinate_mode": "nondimensional" if nd_inputs else "dimensional",
                    "cfd_dataset": "bridge", "exp_subdir": output_dir.name}, f)
    # run_config.json in the SAME shape train_gru.py-produced checkpoints use,
    # so the unmodified evaluate_all.py / bridge_ur_list_from_model can load
    # this checkpoint exactly like any other.
    with open(output_dir / "run_config.json", "w") as f:
        json.dump({
            "cfd_dataset": "bridge", "coordinate_mode": "nondimensional" if nd_inputs else "dimensional",
            "nd_inputs": nd_inputs, "use_ur_context": use_ur_context, "input_cols": input_cols,
            "train_cases": train_cases, "val_cases": val_cases, "test_cases": test_cases,
            "D": D, "fn": fn, "seq_len": seq_len,
            "source": "train_rollout.py curriculum rollout fine-tune",
            "warm_start_checkpoint": ckpt["checkpoint_path"],
        }, f, indent=2)
    # metrics_gru.json's gru_config sub-object -- coupled_inference.py's
    # main() reads hidden_size/num_layers/seq_len/input_cols from HERE
    # (metrics_path.exists() branch), not from run_config.json, and only
    # hidden_size/num_layers/input_cols have a pre-assigned default before
    # that branch -- seq_len does not, so omitting this file entirely
    # raises UnboundLocalError in the unmodified evaluator (confirmed
    # directly: coupled_inference.py line 1075 sets seq_len only inside
    # `if metrics_path.exists()`). Writing this is making OUR checkpoint
    # conform to what the frozen evaluator already expects, not modifying
    # the evaluator itself.
    with open(output_dir / "metrics_gru.json", "w") as f:
        json.dump({
            "gru_config": {
                "hidden_size": config["hidden_size"], "num_layers": config["num_layers"],
                "seq_len": seq_len, "input_cols": input_cols,
            },
        }, f, indent=2)
    with open(output_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Final checkpoint: {final_ckpt_path}")
    print(f"Per-phase best checkpoints (model+optimizer state): "
          f"{[str(output_dir / f'best_phase{i+1}.pt') for i in range(len(rollout_steps_per_phase))]}")


if __name__ == "__main__":
    main()
