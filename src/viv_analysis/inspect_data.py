from pathlib import Path

import matplotlib
import pandas as pd

from viv_analysis.utils import PROJECT_ROOT
from viv_analysis.preprocess import merge_dataframes


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = PROJECT_ROOT
RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots_validation"


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
    }


def plot_case(case_df: pd.DataFrame, output_dir: Path) -> Path:
    ordered = case_df.sort_values(by=["time", "step"]).reset_index(drop=True)
    case_name = str(ordered["case"].iat[0])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    series_info = [
        ("disp", "Displacement", "black"),
        ("cl", "Lift Coefficient", "tab:blue"),
        ("cd", "Drag Coefficient", "tab:orange"),
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


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = merge_dataframes()
    if df.empty:
        print("Merged dataframe is empty. Nothing to inspect.")
        return

    summaries: list[dict[str, object]] = []
    for _, case_df in df.groupby("case", sort=True):
        summaries.append(summarize_case(case_df))
        plot_path = plot_case(case_df, PLOTS_DIR)
        print(f"Saved plot: {plot_path}")

    summary_df = pd.DataFrame(summaries).sort_values(by="case").reset_index(drop=True)
    summary_path = METRICS_DIR / "data_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    pd.set_option("display.max_columns", None)
    print(f"Saved summary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()