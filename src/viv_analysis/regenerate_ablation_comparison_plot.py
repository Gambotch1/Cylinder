#!/usr/bin/env python3
"""
Merge two ALREADY-SAVED closed-loop coupled_eval sweeps (each written by
evaluate_all.py / regenerate_cylinder_plots.py) into ONE amplitude-response
figure: CFD reference plus two named GRU-input-ablation series, via
thesis_plots.plot_amplitude_response_thesis's existing literature_columns
mechanism (built for exactly this -- an explicit, named extra series, never
auto-discovered). No model inference, no coupled simulation is re-run.

Refuses to merge if the two sweeps' CFD columns disagree at any shared Ur
(would silently mean the two runs weren't actually swept against the same
reference) or if their Ur grids don't match exactly.

Usage:
    python -m src.viv_analysis.regenerate_ablation_comparison_plot \
        --primary_coupled_eval_dir gru_cylinder200_dim_nocontext_noacc_coupled_eval \
        --primary_model_column gru_cylinder200_dim_nocontext_noacc \
        --primary_label "Dimensional inputs (no context)" \
        --secondary_coupled_eval_dir gru_cylinder200_nd_nocontext_noacc_coupled_eval \
        --secondary_model_column gru_cylinder200_nd_nocontext_noacc \
        --secondary_label "Nondimensional inputs (no context)" \
        --out_name amplitude_response_dim_vs_nd_nocontext \
        --output_dir gru_cylinder200_nocontext_noacc_comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from viv_analysis.plot_style import apply_thesis_style
apply_thesis_style()

from viv_analysis.thesis_plots import plot_amplitude_response_thesis
from viv_analysis.utils import PROJECT_ROOT


def _load_sweep(coupled_eval_dir: str, model_column: str) -> pd.DataFrame:
    csv_path = PROJECT_ROOT / "results" / coupled_eval_dir / "sweep_results.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    if model_column not in df.columns:
        raise SystemExit(
            f"--model_column '{model_column}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}")
    return df[["Ur", "CFD", model_column]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--primary_coupled_eval_dir", required=True)
    p.add_argument("--primary_model_column", required=True)
    p.add_argument("--primary_label", required=True)
    p.add_argument("--secondary_coupled_eval_dir", required=True)
    p.add_argument("--secondary_model_column", required=True)
    p.add_argument("--secondary_label", required=True)
    p.add_argument("--secondary_color", default="orange", choices=["orange", "green"])
    p.add_argument("--output_dir", required=True,
                   help="results/-relative directory to write thesis_figures/ under "
                        "(this is a merge of two existing runs, so it doesn't belong "
                        "to either one's own directory).")
    p.add_argument("--out_name", default="amplitude_response_comparison")
    p.add_argument("--dataset_note", default=None)
    args = p.parse_args()

    primary = _load_sweep(args.primary_coupled_eval_dir, args.primary_model_column)
    secondary = _load_sweep(args.secondary_coupled_eval_dir, args.secondary_model_column)

    if set(primary["Ur"]) != set(secondary["Ur"]):
        raise SystemExit(
            f"Ur grids differ between the two sweeps: "
            f"primary-only={sorted(set(primary['Ur']) - set(secondary['Ur']))}, "
            f"secondary-only={sorted(set(secondary['Ur']) - set(primary['Ur']))}")

    merged = primary.merge(secondary, on="Ur", suffixes=("", "_secondary"))
    mismatch = merged[(merged["CFD"] - merged["CFD_secondary"]).abs() > 1e-9]
    if not mismatch.empty:
        raise SystemExit(
            f"CFD reference disagrees between the two sweeps at Ur="
            f"{mismatch['Ur'].tolist()} -- they were not run against the same "
            f"reference; refusing to plot them as if they were.")
    # Only "CFD" can collide between the two frames (the two model-column
    # names are always distinct ablation identifiers) -- drop the
    # now-redundant duplicate now that the mismatch check above has passed.
    merged = merged.drop(columns=["CFD_secondary"])

    thesis_dir = PROJECT_ROOT / "results" / args.output_dir / "thesis_figures"
    r = plot_amplitude_response_thesis(
        merged,
        model_column=args.primary_model_column,
        model_label=args.primary_label,
        output_dir=thesis_dir,
        literature_columns={
            args.secondary_model_column: {"label": args.secondary_label, "color": args.secondary_color},
        },
        dataset_note=args.dataset_note,
        out_name=args.out_name,
    )
    print(f"wrote {r['pdf_path']}")


if __name__ == "__main__":
    main()
