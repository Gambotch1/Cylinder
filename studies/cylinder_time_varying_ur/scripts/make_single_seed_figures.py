"""
Thesis figures for the single-seed production time-varying-Ur sweep (see
run_time_varying_sweep.py --single_seed_production). Runs the SAME
per-result functions in plot_time_varying_ur.py that the eventual 3-seed
production run will also use (plot_time_varying_panels, plateau_summary,
plot_transition_window) -- none of those take a seed dimension at all, so
nothing about them changes for single-seed input.

Deliberately does NOT call plot_multiseed_panels: that function's median
envelope + IQR band requires >=2 seeds to mean anything. Computed from a
single seed, np.percentile collapses q25=q75=the one value, so the IQR
band would render with zero width -- reading as "tight seed agreement"
when in fact there is no seed-agreement information at all. Skipping it
here (rather than calling it with a 1-seed dict) avoids ever producing
that misleading figure.

Once the architecture/history-length sensitivity study's Stage 2 freezes
and the real 3-seed sweep runs (run_time_varying_sweep.py without
--single_seed_production/--smoke), a separate driver should call
plot_multiseed_panels for the aggregated main-chapter transition figures;
this script's single-seed output is not a substitute for that, and is
labelled accordingly (see OUTPUT_NOTE below and every receipt's own
single_seed_production=true field).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = STUDY_ROOT.parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(STUDY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT / "scripts"))

from plot_time_varying_ur import (  # noqa: E402
    LOCKIN_REGION_TRANSITIONS, POST_LOCKIN_APPENDIX_TRANSITION,
    plateau_summary, plot_time_varying_panels, plot_transition_window,
    plot_transitions_composite_thesis,
)

REPO_ROOT = STUDY_ROOT.parents[1]
SCHEDULES = ["ascending", "ascending_cosine", "triangular_cosine"]
OUTPUT_NOTE = (
    "single-seed production run (--single_seed_production): NOT the "
    "originally-planned 3-seed median/IQR result -- see this schedule's "
    "own receipt (single_seed_production=true, stage2_gate_bypassed=true)."
)

FIXED_UR_CSV = (REPO_ROOT / "results" / "gru_cylinder200_nd_context_noacc_coupled_eval"
                / "closed_loop_metrics.csv")


def _load_result(schedule: str) -> dict:
    npz_path = STUDY_ROOT / "results" / f"time_varying_ur_{schedule}_single_seed_production.npz"
    d = np.load(npz_path)
    return {k: d[k] for k in ("time", "Ur", "displacement", "velocity", "CL")}


def _load_receipt(schedule: str) -> dict:
    receipt_path = (STUDY_ROOT / "receipts"
                     / f"time_varying_ur_{schedule}_single_seed_production.receipt.json")
    return json.loads(receipt_path.read_text())


def _load_fixed_ur_references() -> tuple[dict, dict]:
    """{Ur: A_star} from the independent fixed-Ur coupled-eval sweep --
    cfd_A_star (CFD reference) and surrogate_A_star (the same GRU
    checkpoint run at each Ur independently, i.e. NOT from this
    time-varying run) -- for plateau_summary's cross-check."""
    if not FIXED_UR_CSV.exists():
        print(f"[warn] {FIXED_UR_CSV} not found -- plateau_summary will "
              f"have no fixed-Ur reference overlay.")
        return {}, {}
    df = pd.read_csv(FIXED_UR_CSV)
    cfd = dict(zip(df["Ur"], df["cfd_A_star"]))
    gru = dict(zip(df["Ur"], df["surrogate_A_star"]))
    return cfd, gru


def main():
    fixed_ur_cfd, fixed_ur_gru = _load_fixed_ur_references()

    for schedule_name in SCHEDULES:
        print(f"\n=== {schedule_name} ===")
        result = _load_result(schedule_name)
        receipt = _load_receipt(schedule_name)
        schedule = receipt["schedule"]
        D, fn, dt = receipt["D"], receipt["fn"], receipt["dt"]
        Tn = 1.0 / fn

        out_dir = STUDY_ROOT / "results" / "thesis_figures" / schedule_name
        appendix_dir = out_dir / "appendix"
        out_dir.mkdir(parents=True, exist_ok=True)
        appendix_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "README_single_seed.txt").write_text(OUTPUT_NOTE + "\n")

        pdf, png = plot_time_varying_panels(
            result, schedule, D, fn, out_dir / "overview")
        print(f"  wrote {pdf.name}")

        fixed_xlim = (-5.0 * Tn, 15.0 * Tn)
        for region, tr in LOCKIN_REGION_TRANSITIONS.items():
            pdf, png = plot_transition_window(
                result, schedule, D, fn,
                ur_before=tr["ur_before"], ur_after=tr["ur_after"],
                out_path_stem=out_dir / f"transition_{region}",
                fixed_xlim=fixed_xlim,
            )
            print(f"  wrote {pdf.name}  ({tr['ur_before']:g}->{tr['ur_after']:g})")

        # One-page composite of all 3 main-chapter transitions (h/D +
        # envelope only, Ur(t) stated in each panel's title) -- for
        # placing three-up on a single portrait page, where the full
        # per-transition figures above are too tall.
        pdf, png = plot_transitions_composite_thesis(
            result, schedule, D, fn, LOCKIN_REGION_TRANSITIONS,
            out_path_stem=out_dir / "transitions_composite",
            fixed_xlim=fixed_xlim,
        )
        print(f"  wrote {pdf.name}  (all 3 main transitions, one page)")

        tr = POST_LOCKIN_APPENDIX_TRANSITION
        pdf, png = plot_transition_window(
            result, schedule, D, fn,
            ur_before=tr["ur_before"], ur_after=tr["ur_after"],
            out_path_stem=appendix_dir / "transition_post_lockin",
            fixed_xlim=fixed_xlim,
        )
        print(f"  wrote appendix/{pdf.name}  ({tr['ur_before']:g}->{tr['ur_after']:g})")

        df, fig = plateau_summary(result, schedule, D, fn, dt, fixed_ur_cfd, fixed_ur_gru)
        df.to_csv(out_dir / "plateau_summary.csv", index=False)
        fig.savefig(out_dir / "plateau_summary.pdf")
        fig.savefig(out_dir / "plateau_summary.png", dpi=200)
        import matplotlib.pyplot as plt
        plt.close(fig)
        n_stable = int(df["stabilised"].sum())
        print(f"  wrote plateau_summary.pdf/csv  ({n_stable}/{len(df)} plateaus stabilised)")

    print(f"\nAll figures written under "
          f"{STUDY_ROOT / 'results' / 'thesis_figures'}/<schedule>/ "
          f"(main-chapter) and .../appendix/ (post-lock-in only). "
          f"Every schedule directory carries a README_single_seed.txt "
          f"reiterating this is single-seed, gate-bypassed output.")


if __name__ == "__main__":
    main()
