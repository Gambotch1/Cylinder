"""
Thesis figures for the differentiable curriculum rollout fine-tune
(sec:bridge_rollout_curriculum): the training-loss trajectory across the
4 period-based phases, and a closed-loop response comparison (CFD vs the
pre-curriculum warm-start vs the post-curriculum checkpoint) at the
lock-in peak.

Reads already-saved artifacts only -- no model inference, no retraining:
  results/gru_bridge_rollout_curriculum/train_history.json
  results/gru_bridge_nd_context_noacc_final22_peaktrain_coupled_eval/*.npz
  results/gru_bridge_rollout_curriculum_coupled_eval/*.npz
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from viv_analysis.evaluate_all import load_full_cfd_df
from viv_analysis.plot_style import (
    CFD_COLOR, ERROR_COLOR, MODEL_COLOR, TEXT_WIDTH_IN, apply_thesis_style,
)
from viv_analysis.utils import PROJECT_ROOT, format_ur_label

CURRICULUM_DIR = PROJECT_ROOT / "results" / "gru_bridge_rollout_curriculum"
PEAKTRAIN_EVAL_DIR = PROJECT_ROOT / "results" / "gru_bridge_nd_context_noacc_final22_peaktrain_coupled_eval"
CURRICULUM_EVAL_DIR = PROJECT_ROOT / "results" / "gru_bridge_rollout_curriculum_coupled_eval"
OUT_DIR = CURRICULUM_DIR / "thesis_figures"


def plot_training_curves() -> tuple[Path, Path]:
    apply_thesis_style()
    import matplotlib.pyplot as plt

    history = json.loads((CURRICULUM_DIR / "train_history.json").read_text())
    phases = history["phases"]

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.6))

    cum_epoch = 0
    phase_boundaries = [0]
    all_x, all_train, all_val = [], [], []
    for p in phases:
        for e in p["epochs"]:
            cum_epoch += 1
            all_x.append(cum_epoch)
            all_train.append(e["train_loss"])
            all_val.append(e["val_loss"])
        phase_boundaries.append(cum_epoch)

    ax.plot(all_x, all_train, color=MODEL_COLOR, marker="o", ms=3, lw=1.1, label="Train loss")
    ax.plot(all_x, all_val, color=ERROR_COLOR, marker="s", ms=3, lw=1.1, ls=(0, (4, 2)), label="Val loss")

    labels = [r"$0.25\,T_n$", r"$0.5\,T_n$", r"$1.0\,T_n$", r"$2.0\,T_n$"]
    for i, (lo, hi) in enumerate(zip(phase_boundaries[:-1], phase_boundaries[1:])):
        if i > 0:
            ax.axvline(lo + 0.5, color="gray", lw=0.6, ls=":")
        mid = (lo + hi + 1) / 2.0
        ax.annotate(f"Phase {i+1}\n{labels[i]}", xy=(mid, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7.5, color="dimgray")

    ax.set_xlabel("Cumulative epoch")
    ax.set_ylabel(r"Scaled $C_L$ MSE")
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlim(0.5, cum_epoch + 0.5)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "rollout_curriculum_training_curves"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)

    best_vals = [p["best_val_loss"] for p in phases]
    caption = (
        r"Training and validation loss (scaled $C_L$ MSE, "
        r"Eq.~\ref{eq:rollout_loss}) across the four period-based curriculum "
        r"phases, warm-started from the resonance-enriched one-step "
        r"checkpoint (\texttt{gru\_bridge\_nd\_context\_noacc\_final22\_"
        r"peaktrain}). Vertical dotted lines mark phase transitions, where "
        r"the learning rate is halved and training resumes from the "
        r"preceding phase's best-validation checkpoint. Best validation "
        rf"loss per phase: {best_vals[0]:.3f} ($0.25\,T_n$), "
        rf"{best_vals[1]:.3f} ($0.5\,T_n$), {best_vals[2]:.3f} ($1.0\,T_n$), "
        rf"{best_vals[3]:.3f} ($2.0\,T_n$). Validation loss is noisy within "
        r"each phase (each epoch draws a fresh random start point per "
        r"case) but does not diverge as the horizon grows, indicating "
        r"stable optimisation throughout -- see "
        r"Sec.~\ref{sec:bridge_rollout_curriculum_result} for why this "
        r"stable training does not translate into stable closed-loop "
        r"response."
    )
    stem.with_suffix(".caption.txt").write_text(caption + "\n")
    return stem.with_suffix(".pdf"), stem.with_suffix(".png")


def _load_case(npz_path: Path, cfd_df):
    d = np.load(npz_path, allow_pickle=True)
    receipt = json.loads(npz_path.with_suffix(".receipt.json").read_text())
    Ur = float(d["Ur"])
    case_label = format_ur_label(Ur)
    case_df = cfd_df[cfd_df["case"].astype(str) == case_label]
    return dict(
        t=d["t"], h=d["h"], cl=d["cl"], D=float(d["D"]),
        t_handoff=receipt.get("handoff_time_s"),
        CFD_t=case_df["time"].to_numpy(), CFD_h=case_df["disp"].to_numpy(),
        CFD_cl=case_df["cl"].to_numpy(),
    )


def plot_closed_loop_comparison(ur_tag: str = "Ur6.7385") -> tuple[Path, Path]:
    apply_thesis_style()
    import matplotlib.pyplot as plt

    print("Loading bridge CFD dataset...")
    cfd_df = load_full_cfd_df("bridge")

    peak_npz = next(PEAKTRAIN_EVAL_DIR.glob(f"coupled_bridge_{ur_tag}_*.npz"))
    curr_npz = next(CURRICULUM_EVAL_DIR.glob(f"coupled_bridge_{ur_tag}_*.npz"))

    peak = _load_case(peak_npz, cfd_df)
    curr = _load_case(curr_npz, cfd_df)
    D = peak["D"]

    mask_p = peak["t"] - peak["t"][0] <= 200
    mask_c = curr["t"] - curr["t"][0] <= 200
    cfd_mask = peak["CFD_t"] - peak["CFD_t"][0] <= 200

    fig, (ax_h, ax_cl) = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN, 5.2), sharex=False)

    t0 = peak["CFD_t"][cfd_mask][0]
    ax_h.plot(peak["CFD_t"][cfd_mask] - t0, (peak["CFD_h"][cfd_mask] - np.mean(peak["CFD_h"][cfd_mask])) / D,
              color=CFD_COLOR, lw=1.0, label="CFD reference")
    ax_h.plot(peak["t"][mask_p] - peak["t"][0], (peak["h"][mask_p] - np.mean(peak["h"][mask_p])) / D,
              color=MODEL_COLOR, lw=1.0, label="Pre-curriculum (peaktrain)")
    ax_h.plot(curr["t"][mask_c] - curr["t"][0], (curr["h"][mask_c] - np.mean(curr["h"][mask_c])) / D,
              color=ERROR_COLOR, lw=1.0, ls=(0, (4, 2)), label="Post-curriculum")
    ax_h.axvline(0.0, color="gray", lw=0.6, ls=":")
    ax_h.set_ylabel(r"$(h-\bar h)/D$")
    ax_h.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, frameon=False, borderaxespad=0.0)

    ax_cl.plot(peak["CFD_t"][cfd_mask] - t0, peak["CFD_cl"][cfd_mask], color=CFD_COLOR, lw=1.0)
    ax_cl.plot(peak["t"][mask_p] - peak["t"][0], peak["cl"][mask_p], color=MODEL_COLOR, lw=1.0)
    ax_cl.plot(curr["t"][mask_c] - curr["t"][0], curr["cl"][mask_c], color=ERROR_COLOR, lw=1.0, ls=(0, (4, 2)))
    ax_cl.axvline(0.0, color="gray", lw=0.6, ls=":")
    ax_cl.set_ylabel(r"$C_L$")
    ax_cl.set_xlabel(r"$t-t_{\mathrm{h}}$ [s]")

    fig.align_ylabels([ax_h, ax_cl])
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"rollout_curriculum_closed_loop_{ur_tag.replace('.', 'p')}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)

    caption = (
        rf"Closed-loop response at the lock-in peak ({ur_tag}, $U=16$\,m/s), "
        r"pre- vs post-curriculum fine-tune, both evaluated with the "
        r"unmodified closed-loop evaluator under identical settings "
        r"(\texttt{forcing\_mode=v1\_additive}, \texttt{handoff\_offset=2000}). "
        r"Both surrogates collapse relative to the CFD reference; the "
        r"post-curriculum checkpoint's collapse is total (exactly zero "
        r"predicted amplitude at every one of the 22 in-scope validation "
        r"cases, not just this one), compared to the pre-curriculum "
        r"checkpoint's small residual response."
    )
    stem.with_suffix(".caption.txt").write_text(caption + "\n")
    return stem.with_suffix(".pdf"), stem.with_suffix(".png")


def plot_open_vs_closed_loop_contrast() -> tuple[Path, Path]:
    """Bar chart: open-loop (teacher-forced) R^2 before vs after the
    curriculum rollout fine-tune, per validation case plus aggregate --
    contrasted in the caption with the closed-loop result (exact-zero
    amplitude, 0/22 pass) for the SAME post-curriculum checkpoint."""
    apply_thesis_style()
    import matplotlib.pyplot as plt

    peak_metrics = json.loads((PROJECT_ROOT / "results" / "gru_bridge_nd_context_noacc_final22_peaktrain"
                                / "metrics_gru.json").read_text())
    curr_metrics = json.loads((CURRICULUM_DIR / "open_loop_val_metrics.json").read_text())

    peak_r2 = peak_metrics["val_metrics"]["r2"]
    curr_agg_r2 = curr_metrics["aggregate_val_metrics"]["r2"]
    per_case = curr_metrics["per_case_val_metrics"]
    cases = sorted(per_case.keys())

    labels = ["Aggregate\n(pre-curriculum)", "Aggregate\n(post-curriculum)"] + \
        [f"{c}\n(post-curriculum)" for c in cases]
    values = [peak_r2, curr_agg_r2] + [per_case[c]["r2"] for c in cases]
    colors = [MODEL_COLOR, ERROR_COLOR] + [ERROR_COLOR] * len(cases)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.2))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.6)
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(r"Open-loop (teacher-forced) $R^2$")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "rollout_curriculum_openloop_vs_closedloop"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)

    caption = (
        rf"Open-loop (teacher-forced) $R^2$ on the validation partition, "
        rf"pre- vs post-curriculum: {peak_r2:.3f} $\to$ {curr_agg_r2:.3f} "
        r"aggregate (per-case range "
        rf"{min(per_case[c]['r2'] for c in cases):.3f}--"
        rf"{max(per_case[c]['r2'] for c in cases):.3f}). Open-loop accuracy "
        r"degraded substantially during the curriculum fine-tune but did not "
        r"collapse to a trivial conditional-mean solution (which would read "
        r"near $R^2\approx 0$) -- contrasted with the SAME post-curriculum "
        r"checkpoint's closed-loop evaluation "
        r"(Fig.~\ref{fig:rollout_curriculum_closed_loop}), which collapses "
        r"to exactly zero predicted amplitude on all 22 in-scope validation "
        r"cases (0/22 pass). The curriculum therefore caused partial "
        r"forgetting of open-loop skill, not the two cleaner alternatives "
        r"(fully preserved open-loop accuracy despite closed-loop "
        r"instability, or total collapse to a trivial solution) that would "
        r"each individually give a single-cause explanation."
    )
    stem.with_suffix(".caption.txt").write_text(caption + "\n")
    return stem.with_suffix(".pdf"), stem.with_suffix(".png")


def main():
    p1 = plot_training_curves()
    print(f"Wrote {p1[0]}\nWrote {p1[1]}")
    p2 = plot_closed_loop_comparison()
    print(f"Wrote {p2[0]}\nWrote {p2[1]}")
    p3 = plot_open_vs_closed_loop_contrast()
    print(f"Wrote {p3[0]}\nWrote {p3[1]}")


if __name__ == "__main__":
    main()
