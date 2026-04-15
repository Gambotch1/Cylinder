import numpy as np
from typing import Callable
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def autoregressive_rollout(predict_fn: Callable[[np.ndarray], float], x_scaler, y_scaler, case_df,
                            input_cols, lookback, release_time_s):
    """
    Run autoregressive rollout by feeding predicted CL back into the
    displacement window via a model-specific prediction callable.
    """
    ordered = case_df.sort_values("time").reset_index(drop=True)
    times   = ordered["time"].to_numpy()
    disp    = ordered["disp"].to_numpy()
    vel     = ordered["vel"].to_numpy()
    acc     = ordered["acc"].to_numpy()
    cl_true = ordered["cl"].to_numpy()
    
    release_idx = int(np.searchsorted(times, release_time_s))
    start = max(lookback, release_idx + lookback)
    
    cl_pred_history = cl_true[:start].tolist()  # seed with true values
    
    for i in range(start, len(ordered)):
        # Build input window from kinematics (always true) 
        # and CL history (predicted after release)
        if "cl" in input_cols:
            window_cl  = np.array(cl_pred_history[i-lookback:i])
            window_kin = np.column_stack([
                disp[i-lookback:i],
                vel[i-lookback:i],
                acc[i-lookback:i],
            ])
            window = np.hstack([window_kin, window_cl.reshape(-1,1)])
        else:
            window = np.column_stack([
                disp[i-lookback:i],
                vel[i-lookback:i],
                acc[i-lookback:i],
            ])
        
        x = x_scaler.transform(window.reshape(1, -1))
        cl_s = predict_fn(x)
        cl   = float(y_scaler.inverse_transform([[cl_s]])[0][0])
        cl_pred_history.append(cl)
    
    return np.array(cl_pred_history[start:]), cl_true[start:], times[start:]

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }