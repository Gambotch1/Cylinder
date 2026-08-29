"""
Predeclared selection rule (validation-only). Never reads test-partition
results -- none exist to read, since every training run in this study is
produced with --skip_test_eval. Test evaluation happens, for the first
time, only in scripts/unlock_test_evaluation.py, gated behind the frozen
manifest this script writes.

Selects at CONFIGURATION level, never a single run/seed:
  Stage 1 selects (dataset, hidden_size, num_layers).
  Stage 2 selects (dataset, hidden_size, num_layers, sequence_samples,
  physical_duration_s).
The frozen manifest lists all 3 seed-specific run_dirs (123/456/789)
belonging to the selected configuration -- selecting a favorable individual
seed/checkpoint is never possible through this script.

Hierarchy (frozen before the validation arrays were run, not tunable here
without approval):
  1. Treat (hidden_size, num_layers[, seq_len]) as the configuration and
     the 3 seeds (123/456/789) as repetitions, never as 18 (or 36)
     independent candidates.
  2. Reject configurations with missing runs (any of the 3 required seeds
     absent), NaNs, or numerical failure -- see REQUIRED_SEEDS in _common
     .py and collect_results.py's all_valid.
  3. Give closed-loop performance priority, in this exact order:
       a. number of stable validation responses (maximize);
       b. median absolute amplitude error (minimize);
       c. worst-case amplitude error (minimize);
       d. frequency error (minimize).
     See _closed_loop_rank_cols for the exact columns and the bridge-
     specific "stable among settled_lco" reading.
  4. Use open-loop R2/NRMSE as a required fidelity check (the
     passes_retention_threshold gate below), not the main ranking
     criterion. Short/intermediate rollout metrics are noted as
     unavailable (this study's closed-loop protocol is full-duration-only
     by design) rather than silently omitted.
  5. Aggregate each configuration across all 3 seeds using median and IQR
     (done in collect_results.py's aggregate(), not here).
  6. Tie-break on effectively-equivalent configurations: smaller, then
     faster. Never choose one fortunate seed or the lowest single
     validation error -- there is no per-seed selection path in this
     script at all, only per-configuration medians.

Produces a Pareto table (every metric kept separate) and a hierarchical
ranking. Does NOT collapse metrics into a weighted scalar.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import REQUIRED_SEEDS, STUDY_ROOT, git_commit, write_json

RETENTION_THRESHOLD = -0.005
BASELINE_H, BASELINE_L = 64, 2


def _closed_loop_rank_cols(dataset: str, agg: pd.DataFrame) -> list[tuple[str, bool]]:
    """(column, ascending) pairs in predeclared priority order, frozen
    BEFORE the validation arrays were run:
      1. number of stable validation responses (maximize)
      2. median absolute amplitude error (minimize)
      3. worst-case amplitude error (minimize)
      4. frequency error (minimize)
    "Worst-case or 90th-percentile" was specified as one tier, not two:
    worst-case is used here since these validation sets are tiny (4
    cylinder / <=5 settled-bridge cases), where a 90th percentile is not a
    materially different statistic from the max -- p90 is still computed
    and kept in the Pareto table for inspection, just not as a separate
    ordinal tier.

    Bridge has no single "stable" label spanning ALL validation cases --
    the amplitude-gate framework in reference_quality.build_status_aware_
    report only scores settled_lco cases. "Number of stable validation
    responses" is therefore read, for bridge, as the stable count among
    the settled_lco subset (validation_stable_count_among_settled_lco),
    not a fraction of all validation cases.

    In practice, for THIS study's fixed 5-case bridge validation
    partition, EVERY case classifies as statistically_stationary_les --
    zero are settled_lco for any of the 18 Stage 1 configs (confirmed
    directly: validation_n_settled_lco=0 across the board after fixing
    the closed-loop npz misplacement bug and rescoring). The 3
    settled-lco columns above are therefore uniformly NaN for bridge
    right now, contributing no discrimination at all (sort_values with
    na_position="last" just ties every row and falls through) --
    silently, not as an error, which would otherwise let bridge selection
    degrade to "smallest model wins" with zero real closed-loop signal
    used. _bridge_non_lco_rms_ratio_deviation_median (added to `agg` by
    build_pareto_table, from validation_mean_rms_ratio_among_non_lco --
    surrogate/CFD blockwise RMS ratio, ideally exactly 1.0) is inserted as
    a 4th tier ahead of frequency error specifically so bridge selection
    still uses real closed-loop evidence under this partition. Left AFTER
    the settled-lco columns (not replacing them) so a future validation
    set that does contain settled_lco cases still gets first priority
    from the originally-specified amplitude tiers.
    """
    if dataset == "cylinder200":
        pairs = [
            ("closed_loop_validation_stable_count_median", False),
            ("closed_loop_validation_median_abs_rel_amp_error_median", True),
            ("closed_loop_validation_worst_abs_rel_amp_error_median", True),
            ("closed_loop_validation_median_abs_f_osc_rel_error_median", True),
        ]
    else:
        pairs = [
            ("closed_loop_validation_stable_count_among_settled_lco_median", False),
            ("closed_loop_validation_median_abs_rel_amp_error_among_settled_lco_median", True),
            ("closed_loop_validation_worst_abs_rel_amp_error_among_settled_lco_median", True),
            ("_bridge_non_lco_rms_ratio_deviation_median", True),
            ("closed_loop_validation_median_abs_f_osc_rel_error_median", True),
        ]
    return [(c, asc) for c, asc in pairs if c in agg.columns]


def build_pareto_table(dataset: str, agg: pd.DataFrame) -> pd.DataFrame:
    baseline = agg[(agg["hidden_size"] == BASELINE_H) & (agg["num_layers"] == BASELINE_L)]
    if baseline.empty:
        raise SystemExit(f"No fresh H={BASELINE_H},L={BASELINE_L} baseline found in "
                          f"the aggregated results -- cannot compute retention deltas.")
    baseline_r2 = float(baseline["open_loop_val_r2_median"].iloc[0])
    baseline_r2_iqr = float(baseline["open_loop_val_r2_iqr"].iloc[0] or 0.0)

    df = agg.copy()
    df = df[df["all_valid"]]
    df["delta_r2_val_vs_baseline"] = df["open_loop_val_r2_median"] - baseline_r2
    df["passes_retention_threshold"] = df["delta_r2_val_vs_baseline"] >= RETENTION_THRESHOLD

    # See _closed_loop_rank_cols's bridge docstring: derived once here (not
    # inside collect_results.py's generic per-column aggregation, since it's
    # specifically a distance-from-1.0, not a plain median/IQR of a raw
    # metric) so bridge selection still has a real closed-loop tier when
    # the settled-lco columns are uniformly empty for this partition.
    rms_col = "closed_loop_validation_mean_rms_ratio_among_non_lco_median"
    if dataset == "bridge" and rms_col in df.columns:
        df["_bridge_non_lco_rms_ratio_deviation_median"] = (df[rms_col] - 1.0).abs()

    review_flags = []
    if df["passes_retention_threshold"].sum() == 0:
        review_flags.append("NO candidate satisfies delta_R2_val >= -0.005 -- "
                             "the threshold may be too strict for this grid; flagging "
                             "for review rather than silently rejecting everything.")
    if df["passes_retention_threshold"].all() and len(df) > 1:
        review_flags.append("EVERY candidate satisfies delta_R2_val >= -0.005 -- "
                             "the threshold is not discriminating between architectures "
                             "here; open-loop retention alone will not drive the ranking.")
    if baseline_r2_iqr > abs(RETENTION_THRESHOLD):
        review_flags.append(
            f"Baseline (H{BASELINE_H},L{BASELINE_L}) open-loop val R2 IQR across seeds "
            f"({baseline_r2_iqr:.4f}) exceeds the retention threshold magnitude "
            f"({abs(RETENTION_THRESHOLD)}) -- seed noise in the baseline itself is "
            f"comparable to the gate. Treat rejections/acceptances near the boundary "
            f"as unreliable.")

    rank_pairs = _closed_loop_rank_cols(dataset, df)
    if not rank_pairs:
        review_flags.append("No closed-loop validation columns found -- ranking falls "
                             "back to open-loop retention only. Run "
                             "evaluate_closed_loop.py + collect_results.py first.")

    df["short_intermediate_rollout_evidence"] = (
        "not computed -- this study's closed-loop protocol is full-duration only "
        "(500s cylinder200 / 300s bridge, matching bridge's actual max CFD "
        "reference duration), per the task's explicit requirement that "
        "short-horizon improvement alone is not sufficient evidence."
    )

    # Tie-break on "effectively equivalent" configurations: prefer smaller,
    # then faster. Both are minimized.
    tie_break_pairs = [
        (c, True) for c in
        ["trainable_parameter_count_median", "closed_loop_mean_inference_time_s_median"]
        if c in df.columns
    ]

    # passes_retention_threshold is the open-loop fidelity GATE (point 4:
    # required check, not the main ranking criterion) -- sorted first only
    # so a configuration that fails it never outranks one that passes,
    # never used as a tiebreaker among passing configurations. Every tier
    # after it is the closed-loop hierarchy from _closed_loop_rank_cols,
    # in order, then the smaller/faster tie-break last.
    sort_pairs = [("passes_retention_threshold", False)] + rank_pairs + tie_break_pairs
    sort_pairs = [(c, asc) for c, asc in sort_pairs if c in df.columns]
    sort_cols = [c for c, _ in sort_pairs]
    ascending = [asc for _, asc in sort_pairs]
    df = df.sort_values(sort_cols, ascending=ascending, na_position="last")
    df["hierarchical_rank"] = range(1, len(df) + 1)
    df.attrs["review_flags"] = review_flags
    df.attrs["baseline_r2"] = baseline_r2
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["cylinder200", "bridge"])
    p.add_argument("--stage", type=int, required=True, choices=[1, 2])
    p.add_argument("--freeze", action="store_true",
                   help="Write manifests/selection_manifest_{dataset}_stage{stage}"
                        ".json with frozen=true, naming the selected "
                        "CONFIGURATION's 3 seed-specific run_dirs. Filename is "
                        "PER-DATASET (not shared between cylinder200 and "
                        "bridge) -- both datasets need their own frozen "
                        "selection simultaneously for Stage 2 job generation, "
                        "and a shared filename would let freezing one dataset "
                        "silently overwrite the other's. This is the ONLY way "
                        "test evaluation can later be unlocked (via "
                        "unlock_test_evaluation.py) -- pass this only after "
                        "you have reviewed the Pareto table and approved a pick.")
    args = p.parse_args()

    agg_csv = STUDY_ROOT / "reports" / f"aggregated_{args.dataset}_stage{args.stage}.csv"
    if not agg_csv.exists():
        raise SystemExit(f"{agg_csv} not found -- run collect_results.py first.")
    agg = pd.read_csv(agg_csv)
    if "history_label" in agg.columns:
        agg["history_label"] = agg["history_label"].astype(object).where(agg["history_label"].notna(), None)

    pareto = build_pareto_table(args.dataset, agg)
    reports_dir = STUDY_ROOT / "reports"
    pareto_csv = reports_dir / f"pareto_{args.dataset}_stage{args.stage}.csv"
    pareto.to_csv(pareto_csv, index=False)

    print(f"Pareto/hierarchical ranking written to {pareto_csv}")
    for flag in pareto.attrs.get("review_flags", []):
        print(f"REVIEW FLAG: {flag}")

    top = pareto.iloc[0] if len(pareto) else None
    if top is None:
        print("No candidates to select from.")
        return

    print(f"\nTop-ranked (hierarchical_rank=1): H={int(top['hidden_size'])} "
          f"L={int(top['num_layers'])} seq_len={int(top['seq_len'])} "
          f"history_label={top.get('history_label')}")

    if args.freeze:
        seeds = top["seeds"] if isinstance(top["seeds"], list) else eval(str(top["seeds"]))
        assert sorted(seeds) == sorted(REQUIRED_SEEDS), (
            f"Selected configuration has seeds {sorted(seeds)}, expected exactly "
            f"{sorted(REQUIRED_SEEDS)} -- refusing to freeze a manifest for an "
            f"incomplete configuration (this would let a single favorable "
            f"seed stand in for the configuration).")

        run_dirs = []
        base = STUDY_ROOT / "results" / args.dataset / f"stage{args.stage}"
        for seed in seeds:
            tag = f"H{int(top['hidden_size'])}_L{int(top['num_layers'])}_seq{int(top['seq_len'])}_seed{seed}"
            if top.get("history_label"):
                tag = f"{top['history_label']}_{tag}"
            run_dirs.append(str(base / tag))

        # Stage 1 selects (dataset, hidden_size, num_layers); Stage 2 selects
        # (dataset, hidden_size, num_layers, sequence_samples,
        # physical_duration_s) -- seq_len/duration are NOT part of the Stage 1
        # selection tuple (every Stage 1 point shares the same baseline
        # seq_len by construction), so they are deliberately omitted there.
        selected_configuration = {
            "hidden_size": int(top["hidden_size"]),
            "num_layers": int(top["num_layers"]),
        }
        if args.stage == 2:
            selected_configuration["sequence_samples"] = int(top["seq_len"])
            selected_configuration["physical_duration_s"] = float(top["sequence_duration_s_median"])
            selected_configuration["history_label"] = top.get("history_label")

        # selection_complete is distinct from frozen: frozen means THIS
        # stage's ranking was reviewed and approved; selection_complete is
        # only ever true for stage 2, since that is the final stage --
        # a frozen Stage 1 manifest unblocks Stage 2 job generation, but
        # does not mean the architecture/history-length choice is final
        # (Stage 2 could still change the picture). Downstream consumers
        # outside this study (e.g. the time-varying-Ur continuation study)
        # gate on selection_complete==True specifically, not on frozen
        # alone, so they never run against a configuration that could
        # still be superseded.
        manifest_path = STUDY_ROOT / "manifests" / f"selection_manifest_{args.dataset}_stage{args.stage}.json"
        manifest = {
            "frozen": True,
            "selection_complete": bool(args.stage == 2),
            "dataset": args.dataset,
            "stage": args.stage,
            "git_commit": git_commit(),
            "selected_configuration": selected_configuration,
            "hierarchical_rank": 1,
            "selection_basis": "validation partition only (open-loop retention gate + "
                                "full-duration closed-loop ranking); see pareto csv",
            "pareto_csv": str(pareto_csv),
            "selected_run_dirs": run_dirs,
            "seeds": sorted(seeds),
            "review_flags": pareto.attrs.get("review_flags", []),
        }
        write_json(manifest_path, manifest)
        print(f"\nFROZEN manifest written to {manifest_path}")
        print(f"Selected CONFIGURATION (not a single run): {selected_configuration}")
        print(f"All 3 seed run_dirs: {run_dirs}")
        print("Test-partition evaluation for this configuration can now be run via:\n"
              f"  python unlock_test_evaluation.py --selection-manifest {manifest_path}")
    else:
        print("\n(not frozen -- pass --freeze once you have reviewed and approved "
              "this ranking; Stage 2 job generation requires a frozen Stage 1 manifest.)")


if __name__ == "__main__":
    main()
