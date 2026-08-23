import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import numpy as np

from viv_analysis.utils import PROJECT_ROOT
from viv_analysis.preprocess import merge_dataframes, compute_kinematics
from viv_analysis.config import config, CYLINDER200_ALIASES


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = PROJECT_ROOT
RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots_validation"




def estimate_oscillation_frequency(time: np.ndarray, disp: np.ndarray, tail_fraction: float = 0.4) -> float:
    n = len(time)
    start = int((1.0 - tail_fraction) * n)
    t = np.asarray(time[start:], dtype=float)
    x = np.asarray(disp[start:], dtype=float)

    if len(t) < 4:
        return float("nan")

    x = x - np.mean(x)
    dt = float(np.median(np.diff(t)))
    freqs = np.fft.rfftfreq(len(x), d=dt)
    spectrum = np.abs(np.fft.rfft(x))

    if len(freqs) <= 1:
        return float("nan")

    peak_idx = np.argmax(spectrum[1:]) + 1
    return float(freqs[peak_idx])

def summarize_case(case_df: pd.DataFrame) -> dict[str, object]:
    ordered = case_df.sort_values(by=["time", "step"]).reset_index(drop=True)
    time_diff = ordered["time"].diff().dropna()

    return {
        "case": ordered["case"].iat[0],
        "rows": int(len(ordered)),
        "start_time": float(ordered["time"].iat[0]),
        "end_time": float(ordered["time"].iat[-1]),
        "mean_dt": float(time_diff.mean()) if not time_diff.empty else 0.0,
        "time_monotone": bool(ordered["time"].is_monotonic_increasing),
        "has_duplicates": bool(ordered.duplicated(subset=["step", "time"]).any()),
        "disp_min": float(ordered["disp"].min()),
        "disp_max": float(ordered["disp"].max()),
        "disp_std": float(ordered["disp"].std(ddof=0)),
        "cl_min": float(ordered["cl"].min()),
        "cl_max": float(ordered["cl"].max()),
        "cl_std": float(ordered["cl"].std(ddof=0)),
        "cd_min": float(ordered["cd"].min()),
        "cd_max": float(ordered["cd"].max()),
        "cd_std": float(ordered["cd"].std(ddof=0)),
        "vel_min": float(ordered["vel"].min()),
        "vel_max": float(ordered["vel"].max()),
        "vel_std": float(ordered["vel"].std(ddof=0)),
        "acc_min": float(ordered["acc"].min()),
        "acc_max": float(ordered["acc"].max()),
        "acc_std": float(ordered["acc"].std(ddof=0)),
    }


def plot_case(case_df: pd.DataFrame, output_dir: Path) -> Path:
    ordered = case_df.sort_values(by=["time", "step"]).reset_index(drop=True)
    case_name = str(ordered["case"].iat[0])

    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    series_info = [
        ("disp", "Displacement", "black"),
        ("cl", "Lift Coefficient", "tab:blue"),
        ("cd", "Drag Coefficient", "tab:orange"),
        ("vel", "Velocity", "tab:green"),
        ("acc", "Acceleration", "tab:red"),
    ]

    for axis, (column, ylabel, color) in zip(axes, series_info):
        axis.plot(ordered["time"], ordered[column], color=color, linewidth=1.0)
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Case {case_name} inspection")

    output_path = output_dir / f"{case_name.replace('.', 'p')}_inspection.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main(dataset: str = "Cylinder1000") -> None:
    # New re200 outputs go to dataset-suffixed dirs so the legacy
    # cylinder1000 plots_validation/metrics artifacts are never overwritten.
    is_re200 = dataset.strip().lower() in CYLINDER200_ALIASES
    metrics_dir = (RESULTS_DIR / "metrics_cylinder200") if is_re200 else METRICS_DIR
    plots_dir   = (RESULTS_DIR / "plots_validation_cylinder200") if is_re200 else PLOTS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = compute_kinematics(merge_dataframes(dataset=dataset), dataset=dataset)
    if df.empty:
        print("Merged dataframe is empty. Nothing to inspect.")
        return

    fn_hz = config["cylinder200_fn"] if is_re200 else 0.2
    ur_values = []
    freq_ratios = []
    summaries: list[dict[str, object]] = []
    for case_label, case_df in df.groupby("case", sort=True):
        summaries.append(summarize_case(case_df))
        plot_path = plot_case(case_df, plots_dir)
        print(f"Saved plot: {plot_path}")

        ordered = case_df.sort_values(by=["time", "step"]).reset_index(drop=True)
        f_osc = estimate_oscillation_frequency(ordered["time"].to_numpy(), ordered["disp"].to_numpy(), tail_fraction=0.4)
        if not np.isfinite(f_osc):
            print(f"Warning: could not estimate f_osc for case {case_label}; skipping in frequency curve.")
            continue
        ur_values.append(float(str(case_label).replace("Ur", "").replace("p", ".")))
        freq_ratios.append(f_osc / fn_hz)

    idx = np.argsort(ur_values)
    ur_values = np.array(ur_values)[idx]
    freq_ratios = np.array(freq_ratios)[idx]

    # Sort by Ur numerically before plotting
    if len(ur_values) == 0:
        print("No valid frequency estimates; skipping frequency response plot.")
    else:
        ur_values = np.array(ur_values)
        freq_ratios = np.array(freq_ratios)
        order = np.argsort(ur_values)
        ur_values = ur_values[order]
        freq_ratios = freq_ratios[order]

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(ur_values, freq_ratios, marker="o", lw=1.5)
    ax.set_xlabel(r"Reduced velocity $U_r$")
    ax.set_ylabel(r"$f_{\mathrm{osc}} / f_n$")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.savefig(plots_dir / "frequency_response.png", dpi=180)
    plt.close(fig)

    summary_df = pd.DataFrame(summaries).sort_values(by="case").reset_index(drop=True)
    summary_path = metrics_dir / "data_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    pd.set_option("display.max_columns", None)
    print(f"Saved summary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--dataset", default="Cylinder1000",
                    help="Dataset to inspect, e.g. Cylinder1000 (default, legacy) "
                         "or cylinder200 (completed Re=200 dataset).")
    args = parser.parse_args()
    main(dataset=args.dataset)