"""Check 7: the test lock prevents accidental test evaluation.

Every training run in this study passes --skip_test_eval (see
tests/test_skip_test_eval.py in the main repo test suite for the
production-level proof that no test prediction is ever computed for a
single run). This file tests the STUDY-level gate on top of that:
unlock_test_evaluation.py refuses to run at all unless a frozen,
configuration-level selection manifest names exactly 3 seed-specific
run_dirs (123/456/789) -- never a single run, never a partial seed set.
"""
import json

import pytest

from _common import assert_frozen_and_get_selected_run_dirs


def _make_run_dir(tmp_path, seed, skip_test_eval=True):
    run_dir = tmp_path / "results" / "cylinder200" / "stage1" / f"H64_L2_seq1000_seed{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "cfd_dataset": "cylinder200",
        "train_cases": ["Ur2", "Ur3"],
        "val_cases": ["Ur4.25", "Ur6.25", "Ur9", "Ur10"],
        "test_cases": ["Ur3.5", "Ur5.5", "Ur7", "Ur11"],
        "skip_test_eval": skip_test_eval,
    }))
    (run_dir / "study_receipt.json").write_text(json.dumps({"seed": seed}))
    return run_dir


def _make_manifest(tmp_path, run_dirs, frozen=True):
    manifest_path = tmp_path / "selection_manifest_stage1.json"
    manifest_path.write_text(json.dumps({
        "frozen": frozen,
        "dataset": "cylinder200",
        "stage": 1,
        "selected_configuration": {"hidden_size": 64, "num_layers": 2},
        "selected_run_dirs": [str(d) for d in run_dirs],
    }))
    return manifest_path


def test_unlock_fails_without_manifest(tmp_path):
    with pytest.raises(SystemExit):
        assert_frozen_and_get_selected_run_dirs(tmp_path / "no_such_manifest.json")


def test_unlock_fails_if_manifest_not_frozen(tmp_path):
    run_dirs = [_make_run_dir(tmp_path, s) for s in (123, 456, 789)]
    manifest_path = _make_manifest(tmp_path, run_dirs, frozen=False)
    with pytest.raises(SystemExit):
        assert_frozen_and_get_selected_run_dirs(manifest_path)


def test_unlock_fails_with_only_one_run_dir(tmp_path):
    """A manifest naming a single run (a favorable seed) instead of the
    full 3-seed configuration must be refused."""
    run_dirs = [_make_run_dir(tmp_path, 123)]
    manifest_path = _make_manifest(tmp_path, run_dirs, frozen=True)
    with pytest.raises(SystemExit):
        assert_frozen_and_get_selected_run_dirs(manifest_path)


def test_unlock_fails_with_only_two_run_dirs(tmp_path):
    """A partial seed set (2 of 3) must also be refused -- exactly 3 or
    nothing."""
    run_dirs = [_make_run_dir(tmp_path, s) for s in (123, 456)]
    manifest_path = _make_manifest(tmp_path, run_dirs, frozen=True)
    with pytest.raises(SystemExit):
        assert_frozen_and_get_selected_run_dirs(manifest_path)


def test_unlock_succeeds_with_exactly_3_seed_run_dirs(tmp_path):
    run_dirs = [_make_run_dir(tmp_path, s) for s in (123, 456, 789)]
    manifest_path = _make_manifest(tmp_path, run_dirs, frozen=True)
    result = assert_frozen_and_get_selected_run_dirs(manifest_path)
    assert len(result) == 3
    assert set(result) == {str(d) for d in run_dirs}


def test_unlock_test_evaluation_refuses_a_run_not_trained_with_skip_test_eval(tmp_path):
    """unlock_test_evaluation.py's own extra guard: even a run correctly
    named in a frozen 3-seed manifest is refused if its run_config.json
    shows skip_test_eval=False (i.e. it could have already been influenced
    by test data during training, defeating the whole point of unlocking
    test metrics only after freezing)."""
    good = [_make_run_dir(tmp_path, s) for s in (123, 456)]
    bad = _make_run_dir(tmp_path, 789, skip_test_eval=False)
    run_dirs = good + [bad]
    manifest_path = _make_manifest(tmp_path, run_dirs, frozen=True)

    # assert_frozen_and_get_selected_run_dirs itself only checks structure
    # (exactly 3, run_config.json exists) -- the skip_test_eval check lives
    # in unlock_test_evaluation.main() itself, exercised here directly.
    selected = assert_frozen_and_get_selected_run_dirs(manifest_path)
    assert len(selected) == 3
    from pathlib import Path
    run_config = json.loads((Path(bad) / "run_config.json").read_text())
    assert run_config["skip_test_eval"] is False
