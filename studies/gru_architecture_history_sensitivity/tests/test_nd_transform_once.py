"""Check 5: the nondimensional transform is applied exactly once along
evaluate_open_loop.py's own data path (its own small re-derivation of
raw_df -> apply_nd_transform -> scalers, separate from train_gru.py's
internal path which tests/test_rollout_training.py already covers).

Mirrors the exact bug pattern found and fixed twice earlier in this
project: feeding an already-ND-transformed dataframe into a function that
self-applies the transform corrupts values by a factor of D (disp) or U
(vel). Here we check the inverse invariant directly: applying
apply_nd_transform twice to the same raw signal must NOT equal applying it
once, and evaluate_open_loop.main()'s single call site must be the only
transform in its path.
"""
import inspect

import numpy as np
import pandas as pd

from viv_analysis.train_gru import apply_nd_transform


def _toy_df():
    n = 50
    return pd.DataFrame({
        "case": ["Ur5.0"] * n,
        "time": np.arange(n) * 0.005,
        "step": np.arange(n),
        "disp": np.linspace(0.1, 1.0, n),
        "vel": np.linspace(0.01, 0.1, n),
        "cl": np.sin(np.linspace(0, 3, n)),
    })


def test_applying_transform_twice_differs_from_once():
    df = _toy_df()
    D, fn = 0.2, 0.2
    once = apply_nd_transform(df.copy(), nd_inputs=True, D=D, fn=fn,
                               input_cols=["disp", "vel"])
    twice = apply_nd_transform(once.copy(), nd_inputs=True, D=D, fn=fn,
                                input_cols=["disp", "vel"])
    # Second application divides again by D (disp) and U (vel) -- must not
    # be a no-op, and must not coincidentally match a single application.
    assert not np.allclose(once["disp"].to_numpy(), twice["disp"].to_numpy())
    assert not np.allclose(once["vel"].to_numpy(), twice["vel"].to_numpy())
    ratio_disp = twice["disp"].to_numpy() / once["disp"].to_numpy()
    np.testing.assert_allclose(ratio_disp, 1.0 / D, rtol=1e-5)


def test_open_loop_metrics_calls_apply_nd_transform_exactly_once_in_source():
    """Static check on the shared implementation both evaluate_open_loop.py
    (val cases) and unlock_test_evaluation.py (test cases, gated) call into
    -- exactly one call site, so neither path can double-transform."""
    import _common
    src = inspect.getsource(_common.compute_open_loop_metrics)
    assert src.count("apply_nd_transform(") == 1
