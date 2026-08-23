"""
Checks tied directly to the corrections in this revision:
- evaluate_open_loop.py / evaluate_closed_loop.py refuse to evaluate
  anything other than the exact canonical validation partition.
- evaluate_closed_loop_train_diagnostic.py refuses val/test cases.
- select_configuration.py's frozen manifest only ever names a complete
  3-seed configuration, with the correct Stage 1 vs Stage 2 schema.
"""
import json

import pytest

import evaluate_closed_loop_train_diagnostic as diag


def _write_run_config(run_dir, dataset="cylinder200", train=None, val=None, test=None,
                       nd_inputs=True):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "cfd_dataset": dataset,
        "nd_inputs": nd_inputs,
        "D": 0.2, "fn": 0.2,
        "input_cols": ["disp", "vel"],
        "use_ur_context": True,
        "seq_len": 1000,
        "train_cases": train or ["Ur2", "Ur3"],
        "val_cases": val or ["Ur4.25", "Ur6.25", "Ur9", "Ur10"],
        "test_cases": test or ["Ur3.5", "Ur5.5", "Ur7", "Ur11"],
        "skip_test_eval": True,
    }))


def _write_stage0_audit(manifests_dir, cylinder_val, cylinder_test=None):
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "stage0_audit.json").write_text(json.dumps({
        "cylinder200": {"val_cases": cylinder_val,
                        "test_cases": cylinder_test or ["Ur3.5", "Ur5.5", "Ur7", "Ur11"]},
        "bridge": {"val_cases": ["Ur4.8433", "Ur5.6856", "Ur6.4227", "Ur7.1597", "Ur8.002"],
                   "test_cases": ["Ur4.6327", "Ur5.5804", "Ur6.3174", "Ur6.9491", "Ur7.7914"]},
    }))


def test_evaluate_open_loop_rejects_non_canonical_validation_set(tmp_path, monkeypatch):
    import evaluate_open_loop
    monkeypatch.setattr(evaluate_open_loop, "STUDY_ROOT", tmp_path)
    _write_stage0_audit(tmp_path / "manifests", cylinder_val=["Ur4.25", "Ur6.25", "Ur9", "Ur10"])

    run_dir = tmp_path / "results" / "cylinder200" / "stage1" / "H64_L2_seq1000_seed123"
    # A wrong val set: same COUNT (4) as canonical, but different cases --
    # must be caught even though the naive count check alone would pass.
    _write_run_config(run_dir, val=["Ur2", "Ur3", "Ur4", "Ur5"])

    monkeypatch.setattr(evaluate_open_loop, "parse_args",
                         lambda: type("A", (), {"run_dir": str(run_dir)})())
    with pytest.raises(AssertionError):
        evaluate_open_loop.main()


def test_evaluate_open_loop_accepts_canonical_validation_set(tmp_path, monkeypatch):
    import evaluate_open_loop
    canonical = ["Ur4.25", "Ur6.25", "Ur9", "Ur10"]
    monkeypatch.setattr(evaluate_open_loop, "STUDY_ROOT", tmp_path)
    _write_stage0_audit(tmp_path / "manifests", cylinder_val=canonical)

    run_dir = tmp_path / "results" / "cylinder200" / "stage1" / "H64_L2_seq1000_seed123"
    _write_run_config(run_dir, val=canonical)
    monkeypatch.setattr(evaluate_open_loop, "parse_args",
                         lambda: type("A", (), {"run_dir": str(run_dir)})())

    # Assertions themselves must pass (we stop before the real (heavy) CFD
    # load/model call by monkeypatching compute_open_loop_metrics).
    monkeypatch.setattr(evaluate_open_loop, "compute_open_loop_metrics",
                         lambda run_dir, cases, case_kind: {
                             f"{case_kind}_cases": sorted(cases),
                             f"aggregate_{case_kind}_metrics": {"r2": 1.0, "rmse": 0.0, "mae": 0.0, "nrmse": 0.0},
                             f"per_case_{case_kind}_metrics": {},
                             f"median_{case_kind}_r2": None, f"median_{case_kind}_nrmse": None,
                         })
    result = evaluate_open_loop.main()
    assert sorted(result["val_cases"]) == sorted(canonical)


def test_train_diagnostic_rejects_val_case_leak(tmp_path, monkeypatch):
    run_dir = tmp_path / "results" / "cylinder200" / "stage1" / "H64_L2_seq1000_seed123"
    # Deliberately corrupt: a "train" case list that actually contains a
    # val case -- the diagnostic script must catch this via its own
    # explicit overlap assertion, not rely on run_config being trustworthy.
    _write_run_config(run_dir, train=["Ur2", "Ur4.25"], val=["Ur4.25", "Ur6.25", "Ur9", "Ur10"])
    monkeypatch.setattr(diag, "parse_args",
                         lambda: type("A", (), {"run_dir": str(run_dir), "ur": None})())
    with pytest.raises(AssertionError):
        diag.main()


def test_train_diagnostic_ur_override_must_be_a_training_case(tmp_path, monkeypatch):
    run_dir = tmp_path / "results" / "bridge" / "stage1" / "H64_L2_seq2500_seed123"
    _write_run_config(run_dir, dataset="bridge", train=["Ur6.7385"],
                       val=["Ur4.8433"], test=["Ur4.6327"])
    monkeypatch.setattr(diag, "parse_args",
                         lambda: type("A", (), {"run_dir": str(run_dir), "ur": 4.8433})())
    with pytest.raises(SystemExit):
        diag.main()  # 4.8433 is a VAL case here, not in train_cases


def test_selection_manifest_stage1_schema_omits_seq_len(tmp_path, monkeypatch):
    """Stage 1 selects (dataset, hidden_size, num_layers) only -- seq_len/
    duration must not appear in the frozen selected_configuration."""
    import select_configuration as sc
    monkeypatch.setattr(sc, "STUDY_ROOT", tmp_path)

    agg = [{
        "dataset": "cylinder200", "stage": 1, "hidden_size": 64, "num_layers": 2,
        "seq_len": 1000, "history_label": None, "n_seeds": 3, "seeds": [123, 456, 789],
        "all_valid": True, "open_loop_val_r2_median": 0.995, "open_loop_val_r2_iqr": 0.001,
        "closed_loop_validation_median_abs_rel_amp_error_median": 0.03,
        "closed_loop_validation_worst_abs_rel_amp_error_median": 0.1,
        "trainable_parameter_count_median": 40321.0,
        "closed_loop_mean_inference_time_s_median": 5.0,
        "sequence_duration_s_median": 5.0,
    }]
    import pandas as pd
    (tmp_path / "reports").mkdir(parents=True)
    pd.DataFrame(agg).to_csv(tmp_path / "reports" / "aggregated_cylinder200_stage1.csv", index=False)

    import sys
    old_argv = sys.argv
    sys.argv = ["select_configuration.py", "--dataset", "cylinder200", "--stage", "1", "--freeze"]
    try:
        sc.main()
    finally:
        sys.argv = old_argv

    manifest = json.loads((tmp_path / "manifests" / "selection_manifest_cylinder200_stage1.json").read_text())
    assert set(manifest["selected_configuration"].keys()) == {"hidden_size", "num_layers"}
    assert len(manifest["selected_run_dirs"]) == 3
    assert manifest["seeds"] == [123, 456, 789]
    # selection_complete is reserved for stage 2 -- a frozen Stage 1
    # manifest unblocks Stage 2 job generation but is not itself the
    # final architecture/history-length choice (downstream consumers like
    # the time-varying-Ur study must not treat it as such).
    assert manifest["frozen"] is True
    assert manifest["selection_complete"] is False


def test_selection_complete_true_only_for_stage2(tmp_path, monkeypatch):
    import select_configuration as sc
    monkeypatch.setattr(sc, "STUDY_ROOT", tmp_path)

    agg = [{
        "dataset": "cylinder200", "stage": 2, "hidden_size": 64, "num_layers": 2,
        "seq_len": 2000, "history_label": "2Tn", "n_seeds": 3, "seeds": [123, 456, 789],
        "all_valid": True, "open_loop_val_r2_median": 0.995, "open_loop_val_r2_iqr": 0.001,
        "closed_loop_validation_median_abs_rel_amp_error_median": 0.03,
        "closed_loop_validation_worst_abs_rel_amp_error_median": 0.1,
        "trainable_parameter_count_median": 40321.0,
        "closed_loop_mean_inference_time_s_median": 5.0,
        "sequence_duration_s_median": 10.0,
    }]
    import pandas as pd
    (tmp_path / "reports").mkdir(parents=True)
    pd.DataFrame(agg).to_csv(tmp_path / "reports" / "aggregated_cylinder200_stage2.csv", index=False)

    import sys
    old_argv = sys.argv
    sys.argv = ["select_configuration.py", "--dataset", "cylinder200", "--stage", "2", "--freeze"]
    try:
        sc.main()
    finally:
        sys.argv = old_argv

    manifest = json.loads((tmp_path / "manifests" / "selection_manifest_cylinder200_stage2.json").read_text())
    assert manifest["frozen"] is True
    assert manifest["selection_complete"] is True
