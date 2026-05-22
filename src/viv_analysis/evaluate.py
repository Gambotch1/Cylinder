# src/evaluate.py

import numpy as np
from typing import Callable
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def Teacher_Forcing_rollout(
    predict_fn:    Callable[[np.ndarray], float],
    x_scaler,
    y_scaler,
    case_df,
    input_cols:    list[str],
    lookback:      int,
    release_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Model-agnostic teacher-forcing rollout.

    predict_fn : callable(x_scaled_2d) → float in scaled space
    All kinematic inputs (disp, vel, acc) use true CFD values.
    If 'cl' is in input_cols it uses the model's own previous prediction.
    """
    ordered = case_df.sort_values("time").reset_index(drop=True)
    times   = ordered["time"].to_numpy()
    disp    = ordered["disp"].to_numpy()
    vel     = ordered["vel"].to_numpy()
    acc     = ordered["acc"].to_numpy()
    cl_true = ordered["cl"].to_numpy()

    release_idx = int(np.searchsorted(times, release_time_s))
    start       = max(lookback, release_idx + lookback)

    cl_pred_history = cl_true[:start].tolist()

    col_arrays = {
        col: ordered[col].to_numpy()
        for col in input_cols
        if col != "cl"
    }

    for i in range(start, len(ordered)):
        cols = []
        for col in input_cols:
            if col == "cl":
                cols.append(np.array(cl_pred_history[i - lookback : i]))
            else:
                cols.append(col_arrays[col][i - lookback : i])
        window = np.column_stack(cols)           # (lookback, n_features)
        x      = x_scaler.transform(window.reshape(1, -1))
        cl_s   = predict_fn(x)
        cl     = float(y_scaler.inverse_transform([[cl_s]])[0][0])
        cl_pred_history.append(cl)

    return (np.array(cl_pred_history[start:]),
            cl_true[start:],
            times[start:])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2":   float(r2_score(y_true, y_pred)),
    }