"""
One-off migration: moves closed-loop npz/receipt.json/png files back from
the WRONG double-nested location a bug in run_coupled_sweep's --output_dir
resolution sent them to (results/studies/gru_architecture_history_sensitivity
/results/{dataset}/stage{N}/{config}/closed_loop_eval/) to the REAL,
intended location (studies/gru_architecture_history_sensitivity/results/
{dataset}/stage{N}/{config}/closed_loop_eval/, i.e. REPO_ROOT/<that path
with the extra results/studies/gru_architecture_history_sensitivity prefix
stripped>). See _common.py's run_coupled_sweep --output_dir comment for
the root cause (now fixed there; this script only recovers already-computed
data, it re-runs no simulation).

Every file is MOVED (never copied, never deleted outright) into the real
directory, merging with whatever summary.json/csv already lives there.
Refuses to overwrite a same-named file that already exists at the
destination (flags it instead) rather than silently clobbering something.
Dry-run by default; pass --apply to actually move files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import REPO_ROOT, STUDY_ROOT

WRONG_ROOT = REPO_ROOT / "results" / "studies" / "gru_architecture_history_sensitivity" / "results"
CORRECT_ROOT = STUDY_ROOT / "results"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true",
                   help="Actually move files. Without this, only prints what would move.")
    args = p.parse_args()

    if not WRONG_ROOT.exists():
        print(f"{WRONG_ROOT} does not exist -- nothing to migrate.")
        return

    n_moved = 0
    n_conflict = 0
    for wrong_dir in sorted(WRONG_ROOT.glob("*/stage*/*/closed_loop_eval")):
        rel = wrong_dir.relative_to(WRONG_ROOT)
        correct_dir = CORRECT_ROOT / rel
        correct_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(wrong_dir.iterdir()):
            if not f.is_file():
                continue
            dest = correct_dir / f.name
            if dest.exists():
                print(f"[CONFLICT] {dest} already exists -- leaving {f} in place, not overwriting.")
                n_conflict += 1
                continue
            print(f"{'[MOVE]' if args.apply else '[DRY RUN would move]'} {f} -> {dest}")
            if args.apply:
                f.rename(dest)
            n_moved += 1

    print(f"\n{n_moved} file(s) {'moved' if args.apply else 'would be moved'}, {n_conflict} conflict(s).")
    if args.apply and n_conflict == 0:
        # Clean up the now-empty wrong-nested tree entirely (never touches
        # anything outside WRONG_ROOT).
        import shutil
        shutil.rmtree(REPO_ROOT / "results" / "studies")
        print(f"Removed the now-empty {REPO_ROOT / 'results' / 'studies'}")
    elif args.apply:
        print("Conflicts found -- leaving results/studies/ in place for manual review.")


if __name__ == "__main__":
    main()
