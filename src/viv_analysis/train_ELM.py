from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from viv_analysis.models.elm import (
    ExtremeLearningMachine,
    ensemble_predict,
    train_ensemble_elm,
    build_lookback_dataset,
    compute_lookback,
)
from viv_analysis.plotting.plot import plot_one_case
from viv_analysis.config import config, cylinder200_structural_params, cylinder200_release_time
from viv_analysis.preprocess import merge_dataframes, compute_kinematics
from viv_analysis.evaluate import evaluate, Teacher_Forcing_rollout
from viv_analysis.utils import PROJECT_ROOT, parse_ur_label

ROOT_DIR   = PROJECT_ROOT
OUTPUT_DIR = ROOT_DIR / "results" / "elm_model_cylinder200"

# -------------------------------

def pick_cases(cases: list[str], ur_values: list[float]) -> set[str]:
    by_ur = {round(parse_ur_label(c), 6): c for c in cases}
    selected = set()

    for ur in ur_values:
        key = round(float(ur), 6)
        if key not in by_ur:
            raise ValueError(
                f"Requested Ur{ur} not found.\n"
                f"Available cases: {sorted(cases)}"
            )
        selected.add(by_ur[key])

    return selected


def split_cases(cases: list[str]) -> tuple[set[str], set[str], set[str], dict[str, float]]:
    train_cases = pick_cases(
        cases,
        [
            2.0, 3.0, 4.0, 4.25, 4.75,
            5.0, 5.5, 5.75,
            6.25, 6.5,
            8.0, 9.0, 10.0, 12.0,
        ],
    )

    val_cases = pick_cases(cases, [2.5, 3.5, 6.0, 11.0])
    test_cases = pick_cases(cases, [4.5, 5.25, 7.0])

    release_time = {
        c: cylinder200_release_time(parse_ur_label(c))
        for c in cases
    }

    return train_cases, val_cases, test_cases, release_time

# -------------------------------

def make_objective(X_train_s, y_train_s, X_val_s, y_val_s, n_seeds):
    def objective(trial):
        h   = trial.suggest_int(
            "hl", config["hl_range"][0], config["hl_range"][1],
            step=config["hl_step"])
        lam = trial.suggest_float(
            "lam", config["lam_range"][0], config["lam_range"][1], log=True)

        mse_list = []
        for seed_offset in range(n_seeds):
            elm = ExtremeLearningMachine(
                input_size=X_train_s.shape[1],
                hidden_size=h,
                output_size=1,          # fixed: always 1 output
                seed=config["seed"] + seed_offset,
            )
            elm.fit(X_train_s, y_train_s, lam)
            y_pred_s = elm.predict(X_val_s)
            mse      = mean_squared_error(y_val_s, y_pred_s)
            mse_list.append(mse)
            trial.report(float(mse), step=seed_offset)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(mse_list))
    return objective

# -------------------------------

def reduce_lookback_features(X: np.ndarray, n_cols: int = 3, step: int = 4) -> np.ndarray:
    """Downsamples the lookback history to save RAM."""
    N = X.shape[0]
    lookback = X.shape[1] // n_cols
    # Reshape back to (Samples, Time, Features)
    X_reshaped = X.reshape(N, lookback, n_cols)
    # Take every 4th time step
    X_reduced = X_reshaped[:, ::step, :]
    # Flatten back to 2D for the ELM
    return X_reduced.reshape(N, -1)

def cl_amplitude_metrics(cl_true: np.ndarray, cl_pred: np.ndarray, ss_frac: float = 0.3) -> dict:
    """Amplitude comparison on the last ss_frac of the rollout window."""
    if len(cl_true) < 20:
        return {
            "CL_amp_true": float("nan"),
            "CL_amp_pred": float("nan"),
            "CL_amp_abs_error": float("nan"),
            "CL_amp_rel_error_pct": float("nan"),
        }

    start = int((1.0 - ss_frac) * len(cl_true))
    true_ss = cl_true[start:]
    pred_ss = cl_pred[start:]

    amp_true = float((true_ss.max() - true_ss.min()) / 2.0)
    amp_pred = float((pred_ss.max() - pred_ss.min()) / 2.0)
    abs_err = abs(amp_pred - amp_true)
    rel_err = 100.0 * abs_err / (abs(amp_true) + 1e-12)

    return {
        "CL_amp_true": amp_true,
        "CL_amp_pred": amp_pred,
        "CL_amp_abs_error": float(abs_err),
        "CL_amp_rel_error_pct": float(rel_err),
    }


def plot_rollout_case(times, cl_true, cl_pred, case_name: str, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

    axes[0].plot(times, cl_true, lw=0.8, color="black", label="CFD true CL")
    axes[0].plot(times, cl_pred, lw=0.8, color="tab:blue", label="ELM predicted CL")
    axes[0].set_title(f"{case_name} — ELM teacher-forcing rollout")
    axes[0].set_ylabel("$C_L$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, cl_pred - cl_true, lw=0.7, color="tab:red")
    axes[1].axhline(0.0, color="black", lw=0.5)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Residual")
    axes[1].grid(True, alpha=0.3)

    safe_case = case_name.replace(".", "p")
    fig.savefig(output_dir / f"test_{safe_case}_cl_ar.png", dpi=200)
    plt.close(fig)

def prepare_data_bundle() -> dict:
    raw_df = merge_dataframes(dataset="cylinder200")
    if raw_df.empty:
        raise ValueError("Merged dataframe is empty.")

    fn = config["cylinder200_fn"]
    d_ref = config["cylinder200_D_ref"]

    raw_df["ur"] = raw_df["case"].apply(parse_ur_label).astype("float32")

    cylinder200_params = cylinder200_structural_params()

    raw_df = compute_kinematics(
        raw_df,
        dataset="cylinder200",
        structural_params=cylinder200_params,
    )

    # keep ur after merges/fallbacks
    if "ur" not in raw_df.columns:
        raw_df["ur"] = raw_df["case"].apply(parse_ur_label).astype("float32")

    all_cases = sorted(str(c) for c in raw_df["case"].drop_duplicates())
    train_cases, val_cases, test_cases, release_time = split_cases(all_cases)

    dt_eff = (
        raw_df.sort_values(["case", "time"])
              .groupby("case")["time"]
              .diff()
              .dropna()
              .median()
    )

    lookback_seconds = 2.0 / fn
    lookback = int(round(lookback_seconds / dt_eff))

    print(f"Effective dt={dt_eff:.6f}, lookback={lookback}, window={lookback * dt_eff:.2f}s")


    config["lookback_steps"] = lookback

    train_df = raw_df.loc[raw_df["case"].isin(train_cases)]
    val_df   = raw_df.loc[raw_df["case"].isin(val_cases)]
    test_df  = raw_df.loc[raw_df["case"].isin(test_cases)]

    build_kwargs = dict(
        lookback=lookback,
        target_col=config["target_col"],
        input_cols=config["input_cols"],
        release_time=release_time,
        release_window=True,
    )

    X_train, y_train, _ = build_lookback_dataset(
        train_df, **build_kwargs, include_meta=False, stride=15)   # was 5
    X_val,  y_val,  meta_val  = build_lookback_dataset(
        val_df,  **build_kwargs, stride=10) # was 1
    X_test, y_test, meta_test = build_lookback_dataset(
        test_df, **build_kwargs, stride=10) # was 1

    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        raise ValueError("Insufficient data after split/lookback construction.")

    # Shrink the number of COLUMNS (Features)
    n_features = len(config["input_cols"])
    # X_train = reduce_lookback_features(X_train, n_cols=n_features, step=4)
    # X_val   = reduce_lookback_features(X_val,   n_cols=n_features, step=4)
    # X_test  = reduce_lookback_features(X_test,  n_cols=n_features, step=4)
    
    print(f"Final X_train shape for ELM: {X_train.shape} (RAM safe!)")

    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        raise ValueError("Insufficient data after split/lookback construction.")

    return {
        "raw_df":       raw_df,
        "train_cases":  train_cases,
        "val_cases":    val_cases,
        "test_cases":   test_cases,
        "release_time": release_time,
        "lookback":     lookback,
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,   "meta_val":  meta_val,
        "X_test":  X_test,  "y_test":  y_test,  "meta_test": meta_test,
    }

# -------------------------------

def scale_bundle(bundle: dict) -> dict:
    x_scaler = StandardScaler().fit(bundle["X_train"])
    y_scaler = StandardScaler().fit(bundle["y_train"].reshape(-1, 1))

    X_train_s = x_scaler.transform(bundle["X_train"])
    X_val_s   = x_scaler.transform(bundle["X_val"])
    X_test_s  = x_scaler.transform(bundle["X_test"])
    y_train_s = y_scaler.transform(bundle["y_train"].reshape(-1,1)).ravel()
    y_val_s   = y_scaler.transform(bundle["y_val"].reshape(-1,1)).ravel()

    # Free originals immediately — they are no longer needed
    del bundle["X_train"], bundle["X_val"], bundle["X_test"]

    return {
        **bundle,
        "x_scaler":  x_scaler,
        "y_scaler":  y_scaler,
        "X_train_s": X_train_s,
        "X_val_s":   X_val_s,
        "X_test_s":  X_test_s,
        "y_train_s": y_train_s,
        "y_val_s":   y_val_s,
    }

# -------------------------------

def resolve_optuna_jobs() -> int:
    env_jobs = os.getenv("OPTUNA_N_JOBS")
    requested = int(env_jobs) if env_jobs else int(config.get("n_jobs", 1))
    return 1 if requested == -1 else max(1, requested)

# -------------------------------

class DownsampleScaler:
    """Wraps the StandardScaler to downsample features during rollout."""
    def __init__(self, base_scaler, n_cols=3, step=4):
        self.base_scaler = base_scaler
        self.n_cols = n_cols
        self.step = step
        
    def transform(self, X):
        # X comes in as shape (1, 6000)
        N = X.shape[0]
        lookback = X.shape[1] // self.n_cols
        # Reshape, downsample, and flatten back to (1, 1500)
        X_reshaped = X.reshape(N, lookback, self.n_cols)
        X_reduced = X_reshaped[:, ::self.step, :]
        X_flattened = X_reduced.reshape(N, -1)
        # Pass the 1500 features to the real scaler
        return self.base_scaler.transform(X_flattened)

# -------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        bundle = scale_bundle(prepare_data_bundle())
    except ValueError as err:
        print(err)
        return

    # Unpack everything needed in main scope — no more undefined variables
    all_cases   = bundle["raw_df"]["case"].drop_duplicates().tolist()
    X_train_s   = bundle["X_train_s"]
    y_train_s   = bundle["y_train_s"]
    X_val_s     = bundle["X_val_s"]
    y_val_s     = bundle["y_val_s"]
    X_test_s    = bundle["X_test_s"]
    y_val       = bundle["y_val"]
    y_test      = bundle["y_test"]
    x_scaler    = bundle["x_scaler"]
    y_scaler    = bundle["y_scaler"]
    meta_val    = bundle["meta_val"]
    meta_test   = bundle["meta_test"]
    train_cases = bundle["train_cases"]
    val_cases   = bundle["val_cases"]
    test_cases  = bundle["test_cases"]
    release_time = bundle["release_time"]
    lookback    = bundle["lookback"]
    raw_df      = bundle["raw_df"]

    # ── Hyperparameter search ──────────────────────────────────────────────
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///optuna_dash.db",
        study_name=f"elm_viv_v2_lb{lookback}",  
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=config["seed"],
            n_startup_trials=10,
            n_ei_candidates=24,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20,
            interval_steps=5,
        ),
    )

    study.optimize(
        make_objective(X_train_s, y_train_s, X_val_s, y_val_s, n_seeds=3),
        n_trials=config["n_trials"],
        n_jobs=resolve_optuna_jobs(),
        show_progress_bar=True,
    )

    h_best   = study.best_params["hl"]
    lam_best = study.best_params["lam"]
    print(f"Best params: hidden={h_best}  lambda={lam_best:.2e}")

    # ── Diagnostics ────────────────────────────────────────────────────────
    print("Available cases:", all_cases)
    print("Train:", sorted(train_cases))
    print("Val:", sorted(val_cases))
    print("Test:", sorted(test_cases))
    print("Columns:", raw_df.columns.tolist())
    print(raw_df[["case", "time", "disp", "vel", "acc", "ur", "cl"]].head())

    # ── Train final ensemble ───────────────────────────────────────────────
    print(f"Training ensemble of {config['n_models']} ELMs...")
    ensemble = train_ensemble_elm(
        X_train_s, y_train_s,
        hidden_size=h_best,
        n_models=config["n_models"],
        lambda_reg=lam_best,
        seed=config["seed"],
    )

    # ── One-step-ahead evaluation ──────────────────────────────────────────
    y_val_pred_s,  y_val_std_s  = ensemble_predict(ensemble, X_val_s)
    y_test_pred_s, y_test_std_s = ensemble_predict(ensemble, X_test_s)

    y_val_pred  = y_scaler.inverse_transform(y_val_pred_s.reshape(-1,1)).ravel()
    y_test_pred = y_scaler.inverse_transform(y_test_pred_s.reshape(-1,1)).ravel()
    y_scale     = float(y_scaler.scale_[0])
    y_val_std   = y_val_std_s  * y_scale
    y_test_std  = y_test_std_s * y_scale

    val_metrics  = evaluate(y_val,  y_val_pred)
    test_metrics = evaluate(y_test, y_test_pred)

    # ── Teacher-forcing rollout on each test case ─────────────
    tf_results = {}
    amp_results = {}
    tf_pred_frames = []

    for case_name in sorted(test_cases):
        case_df = raw_df[raw_df["case"] == case_name].copy()
        if case_df.empty:
            print(f"WARNING: no data for test case {case_name}")
            continue

        release_t = release_time.get(case_name, 60.0)
        elm_predict_fn = lambda x: float(ensemble_predict(ensemble, x)[0][0])

        cl_pred, cl_true, times_tf = Teacher_Forcing_rollout(
            predict_fn=elm_predict_fn,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            case_df=case_df,
            input_cols=config["input_cols"],
            lookback=lookback,
            release_time_s=release_t,
        )

        tf_metrics = evaluate(cl_true, cl_pred)
        amp_metrics = cl_amplitude_metrics(cl_true, cl_pred)

        tf_results[case_name] = {
            "tf_metrics": tf_metrics,
            "tf_r2": tf_metrics["r2"],
        }
        amp_results[case_name] = amp_metrics

        print(
            f"  TF rollout {case_name}: "
            f"R²={tf_metrics['r2']:.4f}  RMSE={tf_metrics['rmse']:.4f}  "
            f"CL_amp_true={amp_metrics['CL_amp_true']:.4f}  "
            f"CL_amp_pred={amp_metrics['CL_amp_pred']:.4f}  "
            f"amp_err={amp_metrics['CL_amp_rel_error_pct']:.2f}%"
        )

        plot_rollout_case(times_tf, cl_true, cl_pred, case_name, OUTPUT_DIR)

        tf_pred_frames.append(pd.DataFrame({
            "case": case_name,
            "time": times_tf,
            "cl_true": cl_true,
            "cl_pred": cl_pred,
            "residual": cl_pred - cl_true,
        }))

    if tf_pred_frames:
        pd.concat(tf_pred_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "test_tf_predictions_cl_elm.csv",
            index=False,
        )

    # ── Save metrics ───────────────────────────────────────────────────────
    metrics = {
        "config": {
            "lookback":    lookback,
            "input_cols":  config["input_cols"],
            "target_col":  config["target_col"],
            "model":       f"EnsembleELM(n={config['n_models']}, "
                           f"h={h_best}, lam={lam_best:.2e})",
            "split_rule":  "manual case sets",
        },
        "case_split": {
            "train_cases": sorted(train_cases),
            "val_cases":   sorted(val_cases),
            "test_cases":  sorted(test_cases),
        },
        "sizes": {
            "train_samples": int(len(y_train_s)),
            "val_samples":   int(len(y_val)),
            "test_samples":  int(len(y_test)),
        },
        "val_metrics":        val_metrics,
        "test_metrics":       test_metrics,
        "Teacher_Forcing":    tf_results,
        "amplitude":          amp_results,
    }

    metrics_path = OUTPUT_DIR / "metrics_cl_elm.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ── Save predictions ───────────────────────────────────────────────────
    val_pred_df = meta_val.copy()
    val_pred_df["y_pred"] = y_val_pred
    val_pred_df["y_std"]  = y_val_std
    val_pred_df.to_csv(OUTPUT_DIR / "val_predictions_cl_elm.csv", index=False)

    test_pred_df = meta_test.copy()
    test_pred_df["y_pred"] = y_test_pred
    test_pred_df["y_std"]  = y_test_std
    test_pred_df.to_csv(OUTPUT_DIR / "test_predictions_cl_elm.csv", index=False)

    # ── Plot first test case ───────────────────────────────────────────────
    example_case = sorted(test_cases)[0]
    plot_path    = OUTPUT_DIR / f"test_{example_case.replace('.','p')}_cl.png"
    plot_one_case(test_pred_df, example_case, plot_path)

    print(f"\nSaved: {metrics_path}")
    print("Validation metrics:"); print(json.dumps(val_metrics, indent=2))
    print("Test metrics:");       print(json.dumps(test_metrics, indent=2))

# -------------------------------

if __name__ == "__main__":
    main()