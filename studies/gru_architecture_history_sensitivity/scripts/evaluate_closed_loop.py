"""
Full-duration closed-loop VALIDATION-ONLY evaluation for one trained
sensitivity-study run. This is the sweep that feeds Stage 1/2 selection
(collect_results.py / select_configuration.py).

Hard-asserts the swept case list equals the run's own canonical validation
list EXACTLY (count matches the Stage-0-audited canonical count, and the
set matches the audit's own frozen validation-case record) -- never
train_cases, never train UNION val, never test_cases. Training-case
diagnostics (including the bridge Ur=6.7385 mechanistic control) live in
the separate, explicitly-labelled evaluate_closed_loop_train_diagnostic.py,
which writes to a different output directory that collect_results.py never
reads.

Reuses coupled_inference.py's own CLI (via _common.run_coupled_sweep) --
see that function's docstring for why direct run_coupled_viv() calls would
require duplicating production setup code.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    CANONICAL_VAL_CASE_COUNT, FULL_DURATION_S, STUDY_ROOT,
    load_json, run_coupled_sweep, write_json,
)
from viv_analysis.reference_quality import build_status_aware_report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--handoff_offset", type=int, default=2000)
    p.add_argument("--forcing_mode", default="v1_additive")
    p.add_argument("--window_frac", type=float, default=0.5)
    p.add_argument("--pass_amp_rel_error_threshold", type=float, default=0.20)
    p.add_argument("--timeout_s", type=int, default=1800)
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_config = load_json(run_dir / "run_config.json")
    receipt = load_json(run_dir / "study_receipt.json")

    dataset = run_config["cfd_dataset"]
    val_cases = run_config["val_cases"]
    canonical_n = CANONICAL_VAL_CASE_COUNT[dataset]

    assert len(val_cases) == canonical_n, (
        f"{run_dir}: run_config['val_cases'] has {len(val_cases)} case(s), "
        f"expected exactly {canonical_n} for {dataset}.")
    audit = load_json(STUDY_ROOT / "manifests" / "stage0_audit.json")
    canonical_val_cases = set(audit[dataset]["val_cases"])
    assert set(val_cases) == canonical_val_cases, (
        f"{run_dir}: run_config['val_cases']={sorted(val_cases)} != the "
        f"Stage 0 audit's canonical validation set {sorted(canonical_val_cases)}.")
    assert not (set(val_cases) & set(run_config["train_cases"])), \
        "val/train overlap -- refusing to run the selection sweep"
    assert not (set(val_cases) & set(run_config["test_cases"])), \
        "val/test overlap -- refusing to run the selection sweep"

    out_dir = run_dir / "closed_loop_eval"
    print(f"Dataset={dataset}  total_time={FULL_DURATION_S[dataset]}s  "
          f"sweep={len(val_cases)} case(s) -- EXACTLY the canonical "
          f"validation partition, no train/test cases.")

    sweep_df, timings, label = run_coupled_sweep(
        run_dir, val_cases, out_dir,
        handoff_offset=args.handoff_offset, forcing_mode=args.forcing_mode,
        window_frac=args.window_frac,
        pass_amp_rel_error_threshold=args.pass_amp_rel_error_threshold,
        timeout_s=args.timeout_s,
    )
    sweep_df.to_csv(out_dir / "sweep_results.csv", index=False)

    summary = {
        "run_dir": str(run_dir), "dataset": dataset,
        "total_time_s": FULL_DURATION_S[dataset],
        "n_sweep_cases": len(val_cases),
        "sweep_cases_are_canonical_validation_partition": True,
        "trainable_parameter_count": receipt["trainable_parameter_count"],
        "mean_inference_time_s": float(np.nanmean(list(timings.values()))) if timings else None,
    }

    # Frequency error is always computable (FFT-based dominant frequency
    # doesn't require amplitude convergence), so it's aggregated the same
    # way for both datasets, ahead of the dataset-specific amplitude
    # scoring below. This is the 4th, lowest-priority tier of the
    # closed-loop ranking hierarchy: stable count > median amp error >
    # worst/p90 amp error > frequency error.
    freq_errs = sweep_df[f"{label}_f_osc_rel_error"].dropna()
    summary["validation_median_abs_f_osc_rel_error"] = (
        float(freq_errs.abs().median()) if len(freq_errs) else None)

    if dataset == "cylinder200":
        errs = sweep_df[f"{label}_A_star_rel_error"].dropna()
        summary.update(
            validation_stable_count=int(sweep_df[f"{label}_stability_label"]
                                         .isin(["stationary_lco", "settled_lco"]).sum()),
            validation_n=len(sweep_df),
            validation_median_abs_rel_amp_error=float(errs.abs().median()) if len(errs) else None,
            validation_worst_abs_rel_amp_error=float(errs.abs().max()) if len(errs) else None,
            validation_p90_abs_rel_amp_error=float(errs.abs().quantile(0.90)) if len(errs) else None,
        )
    else:
        report = build_status_aware_report(str(out_dir), label)
        report.to_csv(out_dir / "status_aware_report.csv", index=False)

        is_settled = report.get("scoring_method", pd.Series(dtype=object)) == "lco_gate"
        settled = report[is_settled]
        non_lco = report[~is_settled & ~report.get("unscored", pd.Series(dtype=bool)).fillna(False)]
        settled_errs = settled["A_star_rel_error"].dropna() if "A_star_rel_error" in settled else pd.Series(dtype=float)

        summary.update(
            status_aware_report_csv=str(out_dir / "status_aware_report.csv"),
            validation_n=len(report),
            validation_unscored=int(report.get("unscored", pd.Series(dtype=bool)).sum())
                if "unscored" in report else None,
            # "Stable" is only a meaningful label for settled_lco reference
            # cases (the amplitude-gate framework non_lco cases don't
            # support -- see reference_quality.build_status_aware_report).
            # Do not conflate this with cylinder's validation_stable_count,
            # which is a fraction of ALL validation cases; this is a
            # fraction of the settled_lco SUBSET only.
            validation_n_settled_lco=int(is_settled.sum()),
            validation_stable_count_among_settled_lco=(
                int(settled["pass_"].sum()) if "pass_" in settled else None),
            validation_median_abs_rel_amp_error_among_settled_lco=(
                float(settled_errs.abs().median()) if len(settled_errs) else None),
            validation_worst_abs_rel_amp_error_among_settled_lco=(
                float(settled_errs.abs().max()) if len(settled_errs) else None),
            validation_p90_abs_rel_amp_error_among_settled_lco=(
                float(settled_errs.abs().quantile(0.90)) if len(settled_errs) else None),
            validation_n_non_lco=int(len(non_lco)),
            validation_mean_rms_ratio_among_non_lco=(
                float(non_lco["mean_rms_ratio"].dropna().mean())
                if "mean_rms_ratio" in non_lco and len(non_lco) else None),
        )

    write_json(out_dir / "closed_loop_summary.json", summary)
    print(f"\nWrote {out_dir / 'sweep_results.csv'} and "
          f"{out_dir / 'closed_loop_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
