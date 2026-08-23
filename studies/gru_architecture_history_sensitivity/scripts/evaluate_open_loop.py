"""
Open-loop (teacher-forced) VALIDATION-ONLY evaluation for one trained
sensitivity-study run.

Hard-asserts the evaluated case list equals the run's own canonical
validation list EXACTLY -- not a subset, not train UNION val, and its count
matches the Stage-0-audited canonical count for the dataset (4 for
cylinder200, 5 for bridge). Never touches the test partition: run_config.json
records skip_test_eval=True (train_sensitivity.py always passes
--skip_test_eval) and metrics_gru.json's test_metrics/tf_results are already
None/{} at the source -- there is nothing test-related for this script to
avoid, by construction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    CANONICAL_VAL_CASE_COUNT, STUDY_ROOT, compute_open_loop_metrics, load_json, write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True,
                   help="Study results dir containing gru_best.pt/run_config.json "
                        "(e.g. studies/.../results/cylinder200/stage1/H64_L2_seq1000_seed123)")
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_config = load_json(run_dir / "run_config.json")

    dataset = run_config["cfd_dataset"]
    val_cases = run_config["val_cases"]
    canonical_n = CANONICAL_VAL_CASE_COUNT[dataset]

    assert len(val_cases) == canonical_n, (
        f"{run_dir}: run_config['val_cases'] has {len(val_cases)} case(s), "
        f"expected exactly {canonical_n} for {dataset} -- refusing to "
        f"evaluate against a non-canonical validation partition.")
    assert not (set(val_cases) & set(run_config["train_cases"])), \
        "val/train overlap in run_config -- refusing to evaluate"
    assert not (set(val_cases) & set(run_config["test_cases"])), \
        "val/test overlap in run_config -- refusing to evaluate"

    # Cross-check against the Stage 0 audit's own frozen record of the
    # canonical validation set (recomputed independently, from real cached
    # CFD data, in scripts/audit_pipeline.py) -- catches a future change to
    # production split logic that happens to preserve the case COUNT but
    # silently changes WHICH cases are in it.
    audit = load_json(STUDY_ROOT / "manifests" / "stage0_audit.json")
    canonical_val_cases = set(audit[dataset]["val_cases"])
    assert set(val_cases) == canonical_val_cases, (
        f"{run_dir}: run_config['val_cases']={sorted(val_cases)} does not "
        f"match the Stage 0 audit's canonical validation set "
        f"{sorted(canonical_val_cases)} for {dataset}.")

    result = compute_open_loop_metrics(run_dir, val_cases, case_kind="val")

    write_json(run_dir / "open_loop_val_metrics.json", result)
    print(f"Wrote {run_dir / 'open_loop_val_metrics.json'}")
    agg = result["aggregate_val_metrics"]
    print(f"  aggregate: R2={agg['r2']:.4f}  RMSE={agg['rmse']:.4f}  "
          f"NRMSE={agg['nrmse']:.4f}  MAE={agg['mae']:.4f}")
    print(f"  median per-case val R2={result['median_val_r2']}")
    print(f"  evaluated exactly {len(val_cases)} canonical validation case(s): "
          f"{sorted(val_cases)}")
    return result


if __name__ == "__main__":
    main()
