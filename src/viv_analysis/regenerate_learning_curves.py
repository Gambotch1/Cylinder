#!/usr/bin/env python3
"""
Regenerate a thesis-quality learning-curve figure from an LSF training
job's captured stdout log -- per-epoch train/val loss is NOT persisted
anywhere in results/<model>/ (train_gru.py only holds it in memory to draw
the original learning_curve.png, then discards it), so retraining would
otherwise be the only way to reproduce this figure. The original bsub log
(logs/<job>_<index>.log) already has every "Epoch N/M  train=...  val=..."
line printed during that same run -- parsing it recovers the exact
historical values with no retraining and no risk of drifting from what was
actually selected as gru_best.pt.

Usage:
    python -m src.viv_analysis.regenerate_learning_curves \
        --log logs/gen_train_noacc_cylinder_1.log --model_subdir gru_cylinder200_dim_context_noacc \
        --log logs/gen_train_noacc_cylinder_2.log --model_subdir gru_cylinder200_nd_context_noacc
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from viv_analysis.thesis_plots import plot_learning_curve_thesis
from viv_analysis.utils import PROJECT_ROOT

_EPOCH_RE = re.compile(r"Epoch\s+(\d+)/\d+\s+train=([\d.]+)\s+val=([\d.]+)")


def parse_epoch_log(log_path: Path) -> tuple[list[float], list[float]]:
    train_losses, val_losses = [], []
    for line in Path(log_path).read_text().splitlines():
        m = _EPOCH_RE.search(line)
        if m:
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))
    if not train_losses:
        raise ValueError(f"No 'Epoch N/M train=... val=...' lines found in {log_path}")
    return train_losses, val_losses


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", action="append", required=True, help="Path to an LSF stdout log (repeatable).")
    p.add_argument("--model_subdir", action="append", required=True,
                  help="Matching results/<model_subdir>/ per --log, same order (repeatable).")
    p.add_argument("--condition_label", action="append", default=None,
                  help="Optional caption description per model, same order as --log "
                       "(defaults to a generic description built from model_subdir).")
    args = p.parse_args()

    if len(args.log) != len(args.model_subdir):
        raise SystemExit("--log and --model_subdir must be given the same number of times, in matching order")
    labels = args.condition_label or [None] * len(args.log)
    if len(labels) != len(args.log):
        raise SystemExit("--condition_label must match --log count if given at all")

    for log_path, model_subdir, label in zip(args.log, args.model_subdir, labels):
        train_losses, val_losses = parse_epoch_log(Path(log_path))
        output_dir = PROJECT_ROOT / "results" / model_subdir / "thesis_figures"
        r = plot_learning_curve_thesis(
            train_losses, val_losses, output_dir, case_label=model_subdir,
            condition_label=label or f"the {model_subdir} model",
        )
        print(f"{model_subdir}: {len(train_losses)} epochs, best_epoch={r['best_epoch']} -> {r['pdf_path']}")


if __name__ == "__main__":
    main()
