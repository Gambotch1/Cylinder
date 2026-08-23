"""Check 10: batched open-loop inference matches the validated reference
implementation. teacher_forcing_rollout (used by evaluate_open_loop.py)
calls model(x) fresh (h0=None) on each trailing window with no carried
recurrent state -- by construction this must equal a batched forward pass
over the same windows (what VIVSequenceDataset + a plain model(x) call
would produce). This test builds a tiny random model and confirms the two
paths agree at every step, using a synthetic single-case dataframe (no
CFD data, no I/O)."""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from viv_analysis.models.gru import VIV_GRU
from viv_analysis.train_gru import teacher_forcing_rollout


def test_teacher_forcing_rollout_matches_batched_windowed_forward():
    torch.manual_seed(0)
    seq_len = 10
    n = 40
    model = VIV_GRU(input_size=2, hidden_size=8, num_layers=1, dropout=0.0)
    model.eval()

    signal = np.random.randn(n, 2).astype(np.float32)
    cl = np.random.randn(n).astype(np.float32)
    times = np.arange(n, dtype=np.float32) * 0.005

    df = pd.DataFrame({"case": ["Ur5.0"] * n, "time": times,
                        "disp": signal[:, 0], "vel": signal[:, 1], "cl": cl})

    y_scaler = StandardScaler().fit(cl.reshape(-1, 1))  # no-op-ish inverse for the check

    cl_pred, _, times_out = teacher_forcing_rollout(
        model, df, ["disp", "vel"], seq_len, release_t=-1e9,
        y_scaler=y_scaler, case_name="Ur5.0", device="cpu",
        use_ur_context=False, ur_stats=None,
    )

    # Reference: manual batched forward pass over every trailing window,
    # independent of teacher_forcing_rollout's own loop.
    starts = range(seq_len, n)
    windows = np.stack([signal[i - seq_len:i] for i in starts])
    with torch.no_grad():
        batched_pred_s, _ = model(torch.from_numpy(windows))
    batched_pred = y_scaler.inverse_transform(
        batched_pred_s.numpy().reshape(-1, 1)).ravel()

    assert len(cl_pred) == len(batched_pred) == len(starts)
    np.testing.assert_allclose(cl_pred, batched_pred, rtol=1e-5, atol=1e-6)
