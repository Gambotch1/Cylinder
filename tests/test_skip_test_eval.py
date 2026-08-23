"""
Regression tests for train_gru.py's --skip_test_eval flag.

Uses a tiny synthetic dataset (merge_dataframes/compute_kinematics are
monkeypatched to return it directly) so a full train_gru.main() pass runs
in seconds instead of requiring the real ~1.3M-row cylinder200 dataset and
a real epoch of GPU training. The synthetic case labels (Ur12, Ur10, Ur3.5)
are chosen to land in train/val/test respectively under the REAL, unmodified
_cylinder200_split() (test={Ur3.5,Ur5.5,Ur7,Ur11}, val={Ur4.25,Ur6.25,Ur9,
Ur10}), so this exercises the actual production split logic, not a stand-in.

These monkeypatches replace ONLY the CFD-loading step; every other function
in the --skip_test_eval control flow (run_validation, teacher_forcing_rollout,
plot_tf_result, amplitude_comparison, evaluate) is the real, unmodified
production function, wrapped only to COUNT calls and record arguments.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viv_analysis import train_gru
from viv_analysis.config import cylinder200_release_time
from viv_analysis.utils import parse_ur_label


TRAIN_UR, VAL_UR, TEST_UR = "Ur12", "Ur10", "Ur3.5"
DT = 0.005
SEQ_LEN = 50


def _synthetic_case_df(case_label: str) -> pd.DataFrame:
    release_t = cylinder200_release_time(parse_ur_label(case_label))
    n = int(release_t / DT) + 4 * SEQ_LEN  # enough rows past release for real windows
    t = np.arange(n, dtype=np.float64) * DT
    disp = 0.1 * np.sin(0.2 * t)
    vel = 0.02 * np.cos(0.2 * t)
    acc = -0.004 * np.sin(0.2 * t)
    cl = 0.3 * np.sin(0.2 * t + 0.3)
    return pd.DataFrame({
        "case": case_label, "time": t, "step": np.arange(n),
        "disp": disp, "vel": vel, "acc": acc, "cl": cl,
    })


def _synthetic_raw_df() -> pd.DataFrame:
    return pd.concat(
        [_synthetic_case_df(c) for c in (TRAIN_UR, VAL_UR, TEST_UR)],
        ignore_index=True,
    )


class _CallRecorder:
    """Wraps a function, recording every call's args/kwargs, then delegates
    to the real implementation so production behavior is unchanged."""

    def __init__(self, real_fn):
        self.real_fn = real_fn
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.real_fn(*args, **kwargs)


@pytest.fixture
def patched_data_loading(monkeypatch):
    synthetic_raw = _synthetic_raw_df()
    monkeypatch.setattr(train_gru, "merge_dataframes", lambda **kw: synthetic_raw.copy())
    monkeypatch.setattr(train_gru, "compute_kinematics",
                         lambda df, **kw: df)  # already has disp/vel/acc/cl
    return synthetic_raw


@pytest.fixture
def recorders(monkeypatch):
    recs = {}
    for name in ("run_validation", "teacher_forcing_rollout", "plot_tf_result"):
        rec = _CallRecorder(getattr(train_gru, name))
        monkeypatch.setattr(train_gru, name, rec)
        recs[name] = rec
    return recs


def _run_main(tmp_path, exp_subdir, skip_test_eval: bool):
    argv = [
        "train_gru.py", "--cfd_dataset", "cylinder200",
        "--input_cols", "disp", "vel",
        "--use_ur_context", "--seq_len", str(SEQ_LEN),
        "--epochs", "1", "--batch_size", "64", "--num_workers", "0",
        "--exp_subdir", exp_subdir, "--overwrite",
    ]
    if skip_test_eval:
        argv.append("--skip_test_eval")
    old_argv = sys.argv
    sys.argv = argv
    try:
        train_gru.main()
    finally:
        sys.argv = old_argv


def test_skip_test_eval_never_calls_test_functions_with_test_data(
    tmp_path, patched_data_loading, recorders, monkeypatch
):
    out_dir = Path(train_gru.ROOT_DIR) / "results" / "_test_skip_test_eval_on"
    monkeypatch.chdir(train_gru.ROOT_DIR)
    try:
        _run_main(tmp_path, "_test_skip_test_eval_on", skip_test_eval=True)

        # plot_tf_result is called ONLY from the test-cases teacher-forcing
        # loop in production code -- zero calls proves that loop never ran.
        assert recorders["plot_tf_result"].calls == []

        # teacher_forcing_rollout IS legitimately called for train/val cases
        # inside amplitude_comparison -- but never for the TEST case label.
        # Signature: (model, case_df, input_cols, seq_len, release_t,
        # y_scaler, case_name, device, ...) -- case_name is positional arg 6.
        tfr_case_names = {args[6] for args, kwargs in recorders["teacher_forcing_rollout"].calls}
        assert TEST_UR not in tfr_case_names, (
            f"teacher_forcing_rollout was called with the TEST case {TEST_UR} "
            f"even though --skip_test_eval was set")
        assert TRAIN_UR in tfr_case_names or VAL_UR in tfr_case_names

        # With --skip_test_eval, test_loader is None in main()'s own scope,
        # so run_validation is called exactly twice: once inside the epoch
        # loop (val) and once for the final val evaluation. The historical
        # (flag-off) behavior additionally calls it a third time for test --
        # see test_default_behavior_unchanged below.
        assert len(recorders["run_validation"].calls) == 2

        metrics = __import__("json").load(open(out_dir / "metrics_gru.json"))
        assert metrics["test_metrics"] is None
        assert metrics["tf_results"] == {}

        amp_csv = out_dir / "amplitude_comparison.csv"
        if amp_csv.exists():
            amp_df = pd.read_csv(amp_csv)
            test_ur_numeric = parse_ur_label(TEST_UR)
            assert not np.isclose(amp_df["Ur"], test_ur_numeric).any(), (
                f"amplitude_comparison.csv contains a row for the TEST case "
                f"{TEST_UR} even though --skip_test_eval was set")

        # The ONLY place plot_tf_result's PNGs are written is the (skipped)
        # test-cases loop -- confirm no such file exists on disk either.
        assert not list(out_dir.glob(f"ar_{TEST_UR}*"))

        run_config = __import__("json").load(open(out_dir / "run_config.json"))
        assert run_config["skip_test_eval"] is True
        # _cylinder200_split always reports the canonical 4-case test set by
        # label, regardless of which cases are actually present in the data
        # (see the "split references cases not in data" warning above for
        # the other 3 canonical labels absent from this synthetic dataset) --
        # the label for OUR synthetic test case must still be among them.
        assert TEST_UR in run_config["test_cases"]  # label retained in metadata
    finally:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)


def test_default_behavior_unchanged_test_functions_do_run(
    tmp_path, patched_data_loading, recorders, monkeypatch
):
    out_dir = Path(train_gru.ROOT_DIR) / "results" / "_test_skip_test_eval_off"
    monkeypatch.chdir(train_gru.ROOT_DIR)
    try:
        _run_main(tmp_path, "_test_skip_test_eval_off", skip_test_eval=False)

        assert len(recorders["plot_tf_result"].calls) >= 1
        # historical behavior: val(epoch) + final val + final test = 3 calls
        assert len(recorders["run_validation"].calls) == 3

        metrics = __import__("json").load(open(out_dir / "metrics_gru.json"))
        assert metrics["test_metrics"] is not None
        assert TEST_UR in metrics["tf_results"]

        run_config = __import__("json").load(open(out_dir / "run_config.json"))
        assert run_config["skip_test_eval"] is False
    finally:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
