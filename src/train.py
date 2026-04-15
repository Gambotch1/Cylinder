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

from models.elm import (
    ExtremeLearningMachine,
    ensemble_predict,
    train_ensemble_elm,
    build_lookback_dataset,
    compute_lookback,
)
from plot import plot_one_case
from config import config
from preprocess import merge_dataframes, compute_kinematics
from evaluate import evaluate, autoregressive_rollout

ROOT_DIR   = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "results" / "elm_model"


def split_cases(
    cases: list[str],
) -> tuple[set[str], set[str], set[str], dict[str, float]]:
    train_cases  = {"Ur3.0","Ur5.6","Ur5.0","Ur6.0","Ur6.5",
                    "Ur9.0","Ur4.6","Ur5.4"}
    val_cases    = {"Ur4.0","Ur4.4","Ur5.2","Ur4.8","Ur8.0"}
    test_cases   = {"Ur4.2","Ur5.8","Ur7.0"}

    all_defined  = train_cases | val_cases | test_cases
    missing      = all_defined - set(cases)
    extra        = set(cases) - all_defined
    if missing:
        print(f"WARNING: defined cases not in data: {missing}")
    if extra:
        print(f"WARNING: data cases not assigned to any split: {extra}")

    release_time = {v: 60.0 for v in cases}
    return train_cases, val_cases, test_cases, release_time


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


def prepare_data_bundle() -> dict:
    raw_df = merge_dataframes()
    if raw_df.empty:
        raise ValueError("Merged dataframe is empty.")

    raw_df     = compute_kinematics(raw_df)
    all_cases  = sorted(str(c) for c in raw_df["case"].drop_duplicates())
    train_cases, val_cases, test_cases, release_time = split_cases(all_cases)

    # Compute physically motivated lookback
    ur_values = [float(c[2:]) for c in all_cases if c.startswith("Ur")]
    lookback  = compute_lookback(ur_values, dt=0.02, D=1.0, U=1.0)
    lookback = int(min(lookback, 200))

    config["lookback_steps"] = lookback
    print(f"Lookback set to {lookback} steps "
          f"({lookback * 0.02:.1f} s, covers 2 cycles at max Ur)")

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
        train_df, **build_kwargs, include_meta=False, stride=5)   # type: ignore
    X_val,  y_val,  meta_val  = build_lookback_dataset(
        val_df,  **build_kwargs, stride=1) # type: ignore
    X_test, y_test, meta_test = build_lookback_dataset(
        test_df, **build_kwargs, stride=1) # type: ignore

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


def resolve_optuna_jobs() -> int:
    env_jobs = os.getenv("OPTUNA_N_JOBS")
    requested = int(env_jobs) if env_jobs else int(config.get("n_jobs", 1))
    return 1 if requested == -1 else max(1, requested)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        bundle = scale_bundle(prepare_data_bundle())
    except ValueError as err:
        print(err)
        return

    # Unpack everything needed in main scope — no more undefined variables
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
        study_name=f"elm_viv_v1_lb{lookback}",  # versioned name
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

    # ── Autoregressive rollout on each test case ───────────────────────────
    ar_results = {}
    for case_name in sorted(test_cases):
        case_df = raw_df[raw_df["case"] == case_name].copy()
        if case_df.empty:
            print(f"WARNING: no data for test case {case_name}")
            continue

        release_t = release_time.get(case_name, 60.0)
        elm_predict_fn = lambda x: float(ensemble_predict(ensemble, x)[0][0])
        cl_pred, cl_true, times_ar = autoregressive_rollout(
            predict_fn=elm_predict_fn,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            case_df=case_df,
            input_cols=config["input_cols"],
            lookback=lookback,
            release_time_s=release_t,
        )
        ar_metrics = evaluate(cl_true, cl_pred)
        ar_results[case_name] = {
            "ar_metrics":  ar_metrics,
            "ar_r2":       ar_metrics["r2"],
        }
        print(f"  AR rollout {case_name}: R²={ar_metrics['r2']:.4f}  "
              f"RMSE={ar_metrics['rmse']:.4f}")

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
        "autoregressive":     ar_results,
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


if __name__ == "__main__":
    main()