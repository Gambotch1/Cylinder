import numpy as np
from typing import List, Tuple, Dict
import warnings

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

def estimate_min_lookback(runs: List[dict], B: float, U: float, dt: float, n_periods: int = 2) -> int:
    """Estimate minimum lookback window to capture wake memory effects."""
    periods = [int(np.ceil(n_periods * (vr * B / U) / dt)) for run in runs for vr in [run['vr']]]
    return max(periods)

def create_lookback_features(runs: List[dict], lookback: int, B: float, U: float) -> Tuple[np.ndarray, np.ndarray]:
    """Transform time-series data into feature matrix with lookback window (Scanlan's convention)."""
    all_X, all_y = [], []

    for run in runs:
        pos, vel, acc = run["h_or_p"], run["hdot_or_pdot"], run["hddot_or_pddot"]
        
        if run["motion_type"] == "heave":
            features = np.column_stack([pos / B, vel / U, (acc * B) / U**2])
        else:
            features = np.column_stack([pos, (vel * B) / U, (acc * B**2) / U**2])
            
        targets = np.column_stack([run["CL"], run["CM"]])

        for i in range(lookback, len(features)):
            all_X.append(features[i - lookback : i, :].flatten())
            all_y.append(targets[i, :])

    return np.array(all_X), np.array(all_y)


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