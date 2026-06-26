import numpy as np
from typing import List, Tuple, Dict
import warnings
import pandas as pd

class ExtremeLearningMachine:
    """Extreme Learning Machine with SVD-based ridge regression."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.W = rng.normal(scale=1.0 / np.sqrt(input_size), size=(input_size, hidden_size))
        self.b = rng.uniform(-1, 1, size=(hidden_size,))
        self.beta = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.W + self.b)

    def fit(self, x: np.ndarray, y: np.ndarray, lambda_reg: float = 1e-6):
        """Train using SVD-based ridge regression."""
        H = self.forward(x)
        U, S, Vh = np.linalg.svd(H, full_matrices=False)
        
        # Warn if highly ill-conditioned
        if (S[0] / (S[-1] + 1e-12)) > 1e10:
            warnings.warn("Hidden layer is ill-conditioned. Consider increasing lambda_reg.")
            
        reg = S / (S**2 + lambda_reg**2)
        self.beta = (Vh.T * reg) @ (U.T @ y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.beta is None:
            raise RuntimeError("Model is not trained!")
        return self.forward(x) @ self.beta


# ================= DATA PREPARATION =================

def compute_lookback(ur_values, dt, D=1.0, U=1.0, n_periods=2):
    """
    Lookback must cover n_periods oscillation cycles at the largest Ur.
    T = Ur * D / U  (oscillation period = reduced velocity in model units)
    """
    max_period_steps = max(int(np.ceil(ur * D / U / dt)) for ur in ur_values)
    return n_periods * max_period_steps

def build_lookback_dataset(
    df: pd.DataFrame,
    lookback: int,
    target_col: str = "cl",
    input_cols: list[str] | None = None,
    release_time: dict[str, float] | None = None,
    release_window: bool = True,
    include_meta: bool = True,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    rows: list[dict[str, object]] | None = [] if include_meta else None

    if input_cols is None:
        input_cols = ["disp"]

    for case_name, case_df in df.groupby("case", sort=True):
        ordered = case_df.sort_values(by=["time", "step"]).reset_index(drop=True)
        signal = ordered[input_cols].to_numpy(dtype=np.float32)
        target = ordered[target_col].to_numpy(dtype=np.float32)
        times = ordered["time"].to_numpy(dtype=np.float32)
        steps = ordered["step"].to_numpy(dtype=np.int64)

        release_t = -np.inf
        if release_time is not None:
            release_t = float(release_time.get(str(case_name), -np.inf))
            # print(f"Case {case_name}: release time = {release_t}")

        release_idx = int(np.searchsorted(times, release_t, side="left"))

        if release_window:
            start_i = max(lookback, release_idx + lookback)
        else:
            start_i = max(lookback, release_idx)

        if len(ordered) <= lookback:
            continue

        for i in range(start_i, len(ordered), stride):
            window = signal[i - lookback : i, :]
            features.append(window.reshape(-1))
            targets.append(target[i])
            if include_meta and rows is not None:
                rows.append(
                    {
                        "case": str(case_name),
                        "time": float(times[i]),
                        "step": int(steps[i]),
                        "time_release": float(times[i] - release_t),
                        "y_true": float(target[i]),
                    }
                )

    if not features:
        empty_meta = (
            pd.DataFrame(columns=["case", "time", "step", "time_release", "y_true"])
            if include_meta
            else pd.DataFrame()
        )
        return np.empty((0, lookback * len(input_cols))), np.empty((0,)), empty_meta

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    meta = pd.DataFrame(rows) if include_meta and rows is not None else pd.DataFrame()
    return X, y, meta
# ================= ENSEMBLE METHODS =================

def train_ensemble_elm(X: np.ndarray, y: np.ndarray, hidden_size: int, 
                       lambda_reg: float, n_models: int = 10, seed: int = 3) -> List[ExtremeLearningMachine]:
    """Train an ensemble of ELMs using Bootstrap Aggregating (Bagging)."""
    models = []
    n_samples = len(X)
    
    for i in range(n_models):
        rng = np.random.RandomState(seed + i)
        elm = ExtremeLearningMachine(X.shape[1], hidden_size, y.shape[0], seed=seed + i)
        
        # Bagging
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        elm.fit(X[idx], y[idx], lambda_reg)
        models.append(elm)
        
    return models

def ensemble_predict(models: List[ExtremeLearningMachine], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (mean_predictions, std_predictions) across the ensemble."""
    preds = np.array([m.predict(X) for m in models])
    return np.mean(preds, axis=0), np.std(preds, axis=0)


# ================= DIAGNOSTICS (Streamlined) =================

def error_budget_analysis(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> Dict:
    """Decompose prediction error into bias, variance, and unexplained components."""
    bias_sq = np.mean(y_pred - y_true, axis=0)**2
    model_var = np.mean(y_std**2, axis=0)
    mse = np.mean((y_true - y_pred)**2, axis=0)
    
    return {
        'mse': mse,
        'bias_squared': bias_sq,
        'model_variance': model_var,
        'residual_variance': np.var(y_true - y_pred, axis=0),
        'pct_breakdown': {
            'bias': np.nan_to_num((bias_sq / mse) * 100),
            'variance': np.nan_to_num((model_var / mse) * 100),
            'residual': np.nan_to_num((np.var(y_true - y_pred, axis=0) / mse) * 100)
        }
    }