from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viv_analysis.utils import PROJECT_ROOT


ROOT_DIR    = PROJECT_ROOT
RESULTS_DIR = ROOT_DIR / "results" / "gru_Cylinder1000"
OUT_DIR     = ROOT_DIR / "results" / "plots_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ── Load coupled-inference sweep results ───────────────────────────────────
    sweep_path = RESULTS_DIR / "coupled_amplitude_sweep.csv"
    if not sweep_path.exists():
        raise FileNotFoundError(
            f"Coupled-inference sweep not found: {sweep_path}\n"
            "Run  python src/run_lockin_sweep.py  first."
        )

    df = pd.read_csv(sweep_path)
    df = df[(df["Ur"] >= 5.0) & (df["Ur"] <= 7.0)].copy()
    df = df.sort_values("Ur").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No lock-in regime data found in coupled_amplitude_sweep.csv")

    # ── Load case split labels from metrics ────────────────────────────────────
    metrics_path = RESULTS_DIR / "metrics_gru.json"
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
        label="GRU closed-loop", zorder=2,
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

    ax.set_xlim(4.95, 7.05)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Reduced velocity $U_r$", fontsize=13)
    ax.set_ylabel(r"Steady-state amplitude $A/D$", fontsize=13)
    ax.set_title(
        "Cylinder1000 — lock-in amplitude response (closed-loop GRU vs CFD)",
        fontsize=15, pad=10,
    )

    ax.text(
        0.02, 0.98,
        "Lock-in regime\nTrain: blue circles\nVal: green diamonds\nTest: red triangles",
        transform=ax.transAxes, fontsize=10,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#d1d5db", alpha=0.9),
    )
    ax.legend(ncols=2, frameon=False, loc="lower left")

    out_png = OUT_DIR / "Cylinder1000_lockin_amplitude_response.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_png}")


if __name__ == "__main__":
    main()
