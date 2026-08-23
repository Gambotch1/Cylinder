"""
End-to-end smoke test for one dataset. Temporary validation only -- every
artifact goes under studies/gru_architecture_history_sensitivity/smoke/,
never results/, and is invisible to collect_results.py, select_configuration.py,
the frozen selection manifest, and (therefore) any thesis table.

Config: H=32, L=1, seed=123, epochs=1, one canonical validation case for
open-loop evaluation, and one short (20s) closed-loop trajectory for the
SAME case. Not a Stage 1/2 run -- no other seeds, no full validation sweep,
no full-duration protocol.

Runs train_sensitivity.py's real main() in-process (--smoke, --epochs 1),
then _common.compute_open_loop_metrics and _common.run_coupled_sweep
directly (the same underlying functions evaluate_open_loop.py /
evaluate_closed_loop.py call, just without their production-scope hard
assertions on the FULL canonical case list -- a smoke test is explicitly
scoped to one case and must not be blocked by those checks).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from _common import (
    REPO_ROOT, STUDY_ROOT, compute_open_loop_metrics, load_json,
    release_time_for, run_coupled_sweep, run_output_dir, write_json,
)

SMOKE_HIDDEN_SIZE = 32
SMOKE_NUM_LAYERS = 1
SMOKE_SEED = 123
SMOKE_HANDOFF_OFFSET_STEPS = 2000  # matches evaluate_closed_loop.py's default
SMOKE_CLOSED_LOOP_MARGIN_S = 20.0  # actual closed-loop duration AFTER handoff
SEQ_LEN_BASELINE = {"cylinder200": 1000, "bridge": 2500}
DT_S = {"cylinder200": 0.005, "bridge": 0.002}


def _safe_smoke_total_time(dataset: str, case: str, seq_len: int) -> float:
    """coupled_inference.py's --total_time is the ABSOLUTE simulation
    end-time (from the case's own t=0), not a duration measured from
    handoff -- passing a short duration that lands before handoff raises
    ValueError. This computes total_time = release_time + warmup(seq_len) +
    handoff_offset + a short margin, so the smoke run is genuinely short
    (only SMOKE_CLOSED_LOOP_MARGIN_S of real closed-loop simulation) while
    staying valid for whichever case is chosen."""
    from viv_analysis.config import config as viv_config, prepare_gru_config

    cfg = prepare_gru_config(dataset, viv_config).copy()
    rt = release_time_for(dataset, case, cfg)
    dt = DT_S[dataset]
    handoff_t = rt + seq_len * dt + SMOKE_HANDOFF_OFFSET_STEPS * dt
    return handoff_t + SMOKE_CLOSED_LOOP_MARGIN_S


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["cylinder200", "bridge"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    dataset = args.dataset
    seq_len = SEQ_LEN_BASELINE[dataset]
    checks = {}

    run_dir = run_output_dir(dataset, stage=1, hidden_size=SMOKE_HIDDEN_SIZE,
                              num_layers=SMOKE_NUM_LAYERS, seq_len=seq_len,
                              seed=SMOKE_SEED, smoke=True)
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)

    # This study's OWN results/ tree (studies/.../results/, read by
    # collect_results.py) -- --smoke guarantees run_output_dir() resolves
    # under smoke/ instead, so a non-recursive top-level snapshot is enough
    # to confirm nothing new landed here. (The separate, much larger
    # top-level repo results/ directory holding production checkpoints is
    # a different concern, already protected structurally: --exp_subdir's
    # '..' escape means train_gru.py's own mkdir call never resolves inside
    # it in the first place -- see _common.exp_subdir_arg.)
    results_dir_before = set(p.name for p in (STUDY_ROOT / "results").iterdir()) \
        if (STUDY_ROOT / "results").exists() else set()

    # ── 1. Train (H=32, L=1, seed=123, epochs=1, --smoke) ───────────────────
    train_argv = [
        "train_sensitivity.py",
        "--dataset", dataset, "--hidden_size", str(SMOKE_HIDDEN_SIZE),
        "--num_layers", str(SMOKE_NUM_LAYERS), "--seq_len", str(seq_len),
        "--seed", str(SMOKE_SEED), "--stage", "1", "--epochs", "1", "--smoke",
    ]
    if args.overwrite:
        train_argv.append("--overwrite")
    old_argv = sys.argv
    sys.argv = train_argv
    try:
        import train_sensitivity
        train_sensitivity.main()
    finally:
        sys.argv = old_argv

    # ── Checks: run_config / receipt-level ───────────────────────────────────
    run_config = load_json(run_dir / "run_config.json")
    receipt = load_json(run_dir / "study_receipt.json")

    checks["test_evaluation_not_called"] = (
        run_config["skip_test_eval"] is True
        and receipt["skip_test_eval"] is True
        and receipt["test_partition_evaluated"] is False
    )
    metrics = load_json(run_dir / "metrics_gru.json")
    checks["no_test_metrics_or_plots"] = (
        metrics["test_metrics"] is None and metrics["tf_results"] == {}
        and not list(run_dir.glob("ar_*.png"))
    )
    checks["input_is_exactly_h_star_hdot_star_Ur"] = (
        run_config["input_cols"] == ["disp", "vel"]
        and run_config["use_ur_context"] is True
        and run_config["nd_inputs"] is True
        and run_config["coordinate_mode"] == "nondimensional"
    )
    checks["nd_transform_applied"] = "disp/D" in run_config["transform_formula"]
    checks["checkpoint_produced"] = (run_dir / "gru_best.pt").exists()

    import pickle
    with open(run_dir / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    checks["training_only_scalers_loaded"] = (
        (run_dir / "x_scaler.pkl").exists() and (run_dir / "y_scaler.pkl").exists()
        and set(run_config["scaler_fit_cases"]) == set(run_config["train_cases"])
        and hasattr(x_scaler, "mean_")
    )

    # ── 2. Open-loop validation on exactly ONE canonical validation case ────
    single_val_case = sorted(run_config["val_cases"])[0]
    ol_result = compute_open_loop_metrics(run_dir, [single_val_case], case_kind="val")
    write_json(run_dir / "smoke_open_loop.json", ol_result)
    checks["open_loop_validation_completed"] = (
        ol_result["val_cases"] == [single_val_case]
        and single_val_case in ol_result["per_case_val_metrics"]
    )

    # ── 3. Short closed-loop trajectory for the SAME case ───────────────────
    # total_time is coupled_inference.py's ABSOLUTE sim end-time (from the
    # case's own t=0), not a duration measured from handoff -- computed
    # per-case so this is genuinely short (SMOKE_CLOSED_LOOP_MARGIN_S of
    # real closed-loop time) without landing before handoff and erroring.
    smoke_total_time = _safe_smoke_total_time(dataset, single_val_case, seq_len)
    cl_out_dir = run_dir / "smoke_closed_loop"
    sweep_df, timings, label = run_coupled_sweep(
        run_dir, [single_val_case], cl_out_dir,
        handoff_offset=SMOKE_HANDOFF_OFFSET_STEPS,
        total_time_override=smoke_total_time,
    )
    sweep_df.to_csv(cl_out_dir / "smoke_sweep_results.csv", index=False)
    npz_files = list(cl_out_dir.glob("coupled_*.npz"))
    checks["closed_loop_reused_production_newmark_path"] = len(npz_files) == 1
    if npz_files:
        import numpy as np
        d = np.load(npz_files[0])
        checks["closed_loop_reused_production_newmark_path"] = (
            "h_cfd" in d and "cl_cfd" in d and "h" in d and "cl" in d
        )

    # ── 4. Uniqueness / non-overwrite checks ─────────────────────────────────
    checks["receipt_and_paths_unique"] = (
        str(run_dir).startswith(str(STUDY_ROOT / "smoke"))
        and not str(run_dir).startswith(str(STUDY_ROOT / "results"))
    )
    results_dir_after = set(p.name for p in (STUDY_ROOT / "results").iterdir()) \
        if (STUDY_ROOT / "results").exists() else set()
    checks["no_existing_result_overwritten"] = results_dir_before == results_dir_after

    all_passed = all(checks.values())
    report = {
        "dataset": dataset, "run_dir": str(run_dir),
        "single_val_case_used": single_val_case,
        "checks": checks, "all_passed": all_passed,
    }
    write_json(STUDY_ROOT / "smoke" / f"smoke_report_{dataset}.json", report)

    print("=" * 60)
    print(f"SMOKE TEST REPORT -- {dataset}")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"ALL PASSED: {all_passed}")
    print("=" * 60)
    if not all_passed:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
