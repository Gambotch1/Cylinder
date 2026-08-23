from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viv_analysis.utils import PROJECT_ROOT, present_model_label


ROOT_DIR    = PROJECT_ROOT
OUT_DIR     = ROOT_DIR / "results" / "plots_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main(
    model_subdir: str = "gru_Cylinder1000",
    dataset: str = "cylinder1000",
    title: str = "Cylinder1000 — lock-in amplitude response (closed-loop GRU vs CFD)",
    ur_min: float = 5.0,
    ur_max: float = 7.0,
    out_name: str = "Cylinder1000_lockin_amplitude_response.png",
) -> None:
    """
    Plot closed-loop GRU vs CFD steady-state amplitude in the lock-in
    regime, from a results/<model_subdir>/coupled_amplitude_sweep.csv
    produced by run_lockin_sweep.py. Defaults reproduce the original
    cylinder1000 plot byte-for-byte; pass model_subdir/dataset/title/
    ur_min/ur_max/out_name for a different dataset (e.g. cylinder200) so
    the legacy cylinder1000 figure at OUT_DIR/out_name is never overwritten.
    """
    results_dir = ROOT_DIR / "results" / model_subdir

    # ── Load coupled-inference sweep results ───────────────────────────────────
    sweep_path = results_dir / "coupled_amplitude_sweep.csv"
    if not sweep_path.exists():
        raise FileNotFoundError(
            f"Coupled-inference sweep not found: {sweep_path}\n"
            "Run  python src/run_lockin_sweep.py  first."
        )

    df = pd.read_csv(sweep_path)
    df = df[(df["Ur"] >= ur_min) & (df["Ur"] <= ur_max)].copy()
    df = df.sort_values("Ur").reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            f"No data found in [{ur_min}, {ur_max}] in coupled_amplitude_sweep.csv"
        )

    # ── Load case split labels from metrics ────────────────────────────────────
    metrics_path = results_dir / "metrics_gru.json"
    split_map: dict[str, str] = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        split = metrics.get("case_split", {})
        for role in ("train", "val", "test"):
            for c in split.get(role, []):
                split_map[c] = role

    def _split(ur: float) -> str:
        txt = f"{float(ur):.4f}".rstrip("0").rstrip(".")
        return split_map.get(f"Ur{txt}", "unknown")

    df["split"] = df["Ur"].apply(_split)

    # ── Thesis-style formatting ────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "-",
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)

    # CFD ground-truth line
    ax.plot(
        df["Ur"], df["ad_cfd"],
        color="#1f2937", lw=2.2, marker="o", ms=5.5,
        label="CFD (ground truth)", zorder=2,
    )
    # GRU closed-loop line
    ax.plot(
        df["Ur"], df["ad_gru"],
        color="#d97706", lw=2.2, marker="s", ms=5.0,
        label=present_model_label(dataset, "GRU closed-loop"), zorder=2,
    )

    # error fill
    ax.fill_between(df["Ur"], df["ad_cfd"], df["ad_gru"],
                    alpha=0.10, color="#d97706")

    # per-point split markers on the CFD curve
    split_styles = {
        "train": {"marker": "o", "facecolor": "#2563eb", "edgecolor": "white", "size": 80},
        "val":   {"marker": "D", "facecolor": "#10b981", "edgecolor": "white", "size": 92},
        "test":  {"marker": "^", "facecolor": "#ef4444", "edgecolor": "white", "size": 92},
    }
    label_offsets = {
        "train": (0.03,  0.022),
        "val":   (0.03, -0.035),
        "test":  (0.03,  0.022),
    }

    for split_name, style in split_styles.items():
        sub = df[df["split"] == split_name]
        if sub.empty:
            continue
        ax.scatter(
            sub["Ur"], sub["ad_cfd"],
            s=style["size"], marker=style["marker"],
            facecolor=style["facecolor"], edgecolor=style["edgecolor"],
            linewidth=1.2, zorder=4,
            label=f"{split_name.capitalize()} case",
        )
        for _, row in sub.iterrows():
            dx, dy = label_offsets[split_name]
            ax.annotate(
                f"Ur{row['Ur']:g}",
                (row["Ur"], row["ad_cfd"]),
                xytext=(row["Ur"] + dx, row["ad_cfd"] + dy),
                textcoords="data",
                fontsize=9, color="#111827", ha="left", va="bottom",
            )

    ax.set_xlim(ur_min - 0.05, ur_max + 0.05)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Reduced velocity $U_r$", fontsize=13)
    ax.set_ylabel(r"Steady-state amplitude $A/D$", fontsize=13)
    ax.set_title(title, fontsize=15, pad=10)

    ax.text(
        0.02, 0.98,
        "Lock-in regime\nTrain: blue circles\nVal: green diamonds\nTest: red triangles",
        transform=ax.transAxes, fontsize=10,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#d1d5db", alpha=0.9),
    )
    ax.legend(ncols=2, frameon=False, loc="lower left")

    out_png = OUT_DIR / out_name
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--model_subdir", default="gru_Cylinder1000")
    parser.add_argument("--dataset", default="cylinder1000")
    parser.add_argument("--title",
                    default="Cylinder1000 — lock-in amplitude response (closed-loop GRU vs CFD)")
    parser.add_argument("--ur_min", type=float, default=5.0)
    parser.add_argument("--ur_max", type=float, default=7.0)
    parser.add_argument("--out_name", default="Cylinder1000_lockin_amplitude_response.png")
    args = parser.parse_args()
    main(
        model_subdir=args.model_subdir, dataset=args.dataset, title=args.title,
        ur_min=args.ur_min, ur_max=args.ur_max, out_name=args.out_name,
    )
