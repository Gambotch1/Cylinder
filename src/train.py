from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from models.elm import ExtremeLearningMachine, ensemble_predict, train_ensemble_elm
from preprocess import merge_dataframes
from scipy.signal import savgol_filter


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "results" / "elm_model"

def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates velocity and acceleration from displacement and time.
    Calculations are strictly grouped by 'case' to avoid differentiating 
    across the boundaries of different simulations.
    """
    print("Computing velocity and acceleration...")
    df = df.sort_values(by=["case", "time", "step"]).reset_index(drop=True)
    
    velocities = []
    accelerations = []
    
    for case_name, case_df in df.groupby("case", sort=False):
        t = case_df["time"].to_numpy()
        d = case_df["disp"].to_numpy()
        d_smooth = savgol_filter(d, window_length=11, polyorder=3)
        v = np.gradient(d_smooth, t)
        a = np.gradient(v, t)
        
        velocities.extend(v)
        accelerations.extend(a)
        
    df["vel"] = velocities
    df["acc"] = accelerations
    
    return df


def build_lookback_dataset(
    df: pd.DataFrame,
    lookback: int,
    target_col: str = "cl",
    input_cols: list[str] | None = None,
    release_time: dict[str, float] | None = None,
    release_window: bool = True,

) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    rows: list[dict[str, object]] = []

    if input_cols is None:
        input_cols = ["disp"]

    for case_name, case_df in df.groupby("case", sort=True):
        ordered = case_df.sort_values(by=["time", "step"]).reset_index(drop=True)
        signal = ordered[input_cols].to_numpy(dtype=np.float64)
        target = ordered[target_col].to_numpy(dtype=np.float64)
        times = ordered["time"].to_numpy(dtype=np.float64)
        steps = ordered["step"].to_numpy(dtype=np.int64)

        release_t = -np.inf
        if release_time is not None:
            release_t = float(release_time.get(str(case_name), -np.inf))
            print(f"Case {case_name}: release time = {release_t}")

        release_idx = int(np.searchsorted(times, release_t, side="left"))

        if release_window:
            start_i = max(lookback, release_idx + lookback)
        else:
            start_i = max(lookback, release_idx)

        if len(ordered) <= lookback:
            continue

        for i in range(start_i, len(ordered)):
            window = signal[i - lookback : i, :]
            features.append(window.reshape(-1))
            targets.append(target[i])
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
        return np.empty((0, lookback * len(input_cols))), np.empty((0,)), pd.DataFrame(columns=["case", "time", "step", "y_true"])

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    meta = pd.DataFrame(rows)
    return X, y, meta


def split_cases(cases: list[str]) -> tuple[set[str], set[str], set[str], dict[str, float]]:
    train_cases = {"Ur3.0", "Ur5.6", "Ur5.0", "Ur6.0", "Ur6.5", "Ur9.0", "Ur4.6", "Ur5.4"}
    val_cases = {"Ur4.0", "Ur4.4", "Ur5.2", "Ur4.8", "Ur8.0"}
    test_cases = {"Ur4.2", "Ur5.8", "Ur7.0"}
    release_time = {v: 60.0 for v in cases}

    return train_cases, val_cases, test_cases, release_time


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_one_case(pred_df: pd.DataFrame, case_name: str, output_path: Path) -> None:
    case_df = pred_df.loc[pred_df["case"] == case_name].sort_values(by=["time", "step"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(case_df["time"], case_df["y_true"], label="True cl", linewidth=1.2, color="black")
    ax.plot(case_df["time"], case_df["y_pred"], label="Predicted cl", linewidth=1.0, color="tab:blue", alpha=0.9)
    ax.set_xlabel("Time")
    ax.set_ylabel("cl")
    ax.set_title(f"ELM ensemble prediction on test case: {case_name}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    lookback = 30
    target_col = "cl"
    input_cols = ["disp", "vel", "acc"]
    n_models = 1
    n_hidden_nodes = 300
    alpha_reg = 1.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = merge_dataframes()
    if raw_df.empty:
        print("Merged dataframe is empty. Run preprocess first.")
        return

    raw_df = compute_kinematics(raw_df)

    # Debugging: Print case distribution before splitting
    # print("Cases in raw_df:", sorted(raw_df["case"].drop_duplicates()))
    # for case in sorted(raw_df["case"].drop_duplicates()):
    #     case_rows = len(raw_df[raw_df["case"] == case])
    #     print(f"  Case {case}: {case_rows} rows")

    all_cases = sorted([str(case) for case in raw_df["case"].drop_duplicates()])
    # all_cases = sorted(raw_df["case"].drop_duplicates()) 
    train_cases, val_cases, test_cases, release_time = split_cases(all_cases)
    print(release_time)

    train_df = raw_df.loc[raw_df["case"].isin(train_cases)]
    val_df = raw_df.loc[raw_df["case"].isin(val_cases)]
    test_df = raw_df.loc[raw_df["case"].isin(test_cases)]

    X_train, y_train, meta_train = build_lookback_dataset(train_df, lookback, target_col, input_cols, release_time, release_window=True)
    # print(X_train.shape, y_train.shape)
    X_val, y_val, meta_val = build_lookback_dataset(val_df, lookback, target_col, input_cols, release_time, release_window=True)
    # print(X_val.shape, y_val.shape)
    X_test, y_test, meta_test = build_lookback_dataset(test_df, lookback, target_col, input_cols, release_time, release_window=True)
    # print(X_test.shape, y_test.shape)

    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        print("Insufficient data after split/lookback construction.")
        return

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train_s = x_scaler.transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()

    print(f"Training ensemble of {n_models} ELM models...")
    ensemble_models: list[ExtremeLearningMachine] = train_ensemble_elm(
        X_train_s,
        y_train_s,
        hidden_size=n_hidden_nodes,
        n_models=n_models,
        lambda_reg=alpha_reg,
        seed=3,
    )

    y_val_pred_s, y_val_std_s = ensemble_predict(ensemble_models, X_val_s)
    y_test_pred_s, y_test_std_s = ensemble_predict(ensemble_models, X_test_s)

    y_val_pred = y_scaler.inverse_transform(y_val_pred_s.reshape(-1, 1)).ravel()
    y_test_pred = y_scaler.inverse_transform(y_test_pred_s.reshape(-1, 1)).ravel()

    y_scale = float(np.asarray(y_scaler.scale_).reshape(-1)[0])
    y_val_std = y_val_std_s * y_scale
    y_test_std = y_test_std_s * y_scale

    val_metrics = evaluate(y_val, y_val_pred)
    test_metrics = evaluate(y_test, y_test_pred)

    metrics = {
        "config": {
            "lookback": lookback,
            "input_cols": input_cols,
            "target_col": target_col,
            "model": f"EnsembleELM(n_models={n_models}, n_hidden={n_hidden_nodes}, alpha={alpha_reg})",
            "elm_params": {
                "n_models": n_models,
                "n_hidden": n_hidden_nodes,
                "alpha": alpha_reg,
                "random_state_base": 3,
            },
            "split_rule": "case index mod 5: train=(0,1,2), val=3, test=4",
        },
        "case_split": {
            "train_cases": sorted(train_cases),
            "val_cases": sorted(val_cases),
            "test_cases": sorted(test_cases),
        },
        "sizes": {
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "test_samples": int(len(y_test)),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    metrics_path = OUTPUT_DIR / "metrics_cl_elm.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    val_pred_df = meta_val.copy()
    val_pred_df["y_pred"] = y_val_pred
    val_pred_df["y_std"] = y_val_std
    val_pred_path = OUTPUT_DIR / "val_predictions_cl_elm.csv"
    val_pred_df.to_csv(val_pred_path, index=False)

    test_pred_df = meta_test.copy()
    test_pred_df["y_pred"] = y_test_pred
    test_pred_df["y_std"] = y_test_std
    test_pred_path = OUTPUT_DIR / "test_predictions_cl_elm.csv"
    test_pred_df.to_csv(test_pred_path, index=False)

    example_case = sorted(test_cases)[0]
    plot_path = OUTPUT_DIR / f"test_case_{example_case.replace('.', 'p')}_cl.png"
    plot_one_case(test_pred_df, example_case, plot_path)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved validation predictions: {val_pred_path}")
    print(f"Saved test predictions: {test_pred_path}")
    print(f"Saved test-case plot: {plot_path}")
    print("Validation metrics:")
    print(json.dumps(val_metrics, indent=2))
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
