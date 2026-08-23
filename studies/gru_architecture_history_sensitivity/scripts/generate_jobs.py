"""
Freezes the study manifest and generates LSF (bsub) job arrays into jobs/.

Two modes:
  --freeze-manifest   Combine configs/*.json + manifests/stage0_audit.json
                       into the single frozen manifests/study_manifest.json.
                       Must be run (and audit_pipeline.py must have already
                       run) before job generation.
  (default)            Read manifests/study_manifest.json and emit bsub
                       scripts into jobs/. Never reconstructs the grid from
                       configs/*.json directly -- always goes through the
                       frozen manifest, per the task's explicit requirement.

Cluster conventions (queue, GPU request, module/venv activation, env vars)
are copied from the repo's existing working submission scripts
(gen_train_missing_ablations.bsub / gen_eval_missing_ablations_batch*.bsub),
not re-derived.

Stage 2 job scripts: each dataset gets its OWN selection manifest file,
manifests/selection_manifest_{dataset}_stage1.json (never a shared/
unqualified filename -- cylinder200 and bridge freeze independently, and a
shared filename would let freezing one silently overwrite the other's,
which is exactly what happened once during this study before this was
fixed). At generation time, this script checks whether that dataset's own
manifest exists and is frozen:
  - NOT frozen yet: writes jobs/stage2_train_eval_{dataset}_DISABLED.bsub --
    #BSUB directives commented out, and the script body itself also refuses
    to run (exits 1) if the manifest still isn't frozen by the time anyone
    tries to run it directly.
  - Frozen: writes jobs/stage2_train_eval_{dataset}.bsub -- a genuinely
    submittable array (real #BSUB directives), with the selected
    (hidden_size, num_layers) baked in from that manifest. The runtime
    frozen-check is kept in BOTH cases as a second line of defense, not
    just skipped once generation-time sees frozen=true.
Nothing in this script submits any job -- it only writes files.

The generated stage1_{open,closed}_loop_validation job arrays call
evaluate_open_loop.py / evaluate_closed_loop.py only -- both restricted to
the canonical VALIDATION partition by hard assertion. This script never
generates a job for evaluate_closed_loop_train_diagnostic.py (training-case
diagnostics, including the bridge Ur=6.7385 mechanistic control) or
unlock_test_evaluation.py (test-partition evaluation) -- both are run
manually, by name, outside any generated array, and have no effect on
selection.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from _common import STUDY_ROOT, REPO_ROOT, load_json, write_json

BSUB_HEADER = """#!/bin/bash
#BSUB -J "{job_name}[1-{n}]"
#BSUB -q BatchGPU
#BSUB -gpu "num=1:mode=shared"
#BSUB -R "rusage[mem=32000]"
#BSUB -W 48:00
#BSUB -o {study_rel}/logs/{job_name}_%I.log

set -euo pipefail
mkdir -p {study_rel}/logs
cd {repo_root}
source .venv/bin/activate
export VIV_NUM_WORKERS=0
export OMP_NUM_THREADS=4
: "${{LSB_JOBINDEX:?LSB_JOBINDEX is not set}}"
"""


def freeze_manifest():
    audit_path = STUDY_ROOT / "manifests" / "stage0_audit.json"
    if not audit_path.exists():
        raise SystemExit(f"{audit_path} not found -- run scripts/audit_pipeline.py first.")

    manifest = {
        "study": "gru_architecture_history_sensitivity",
        "frozen_at_git_commit": load_json(STUDY_ROOT / "manifests" / "stage0_audit.json"),
        "architecture_grid": load_json(STUDY_ROOT / "configs" / "architecture_grid.json"),
        "cylinder_history_grid": load_json(STUDY_ROOT / "configs" / "cylinder_history_grid.json"),
        "bridge_history_grid": load_json(STUDY_ROOT / "configs" / "bridge_history_grid.json"),
    }
    out = STUDY_ROOT / "manifests" / "study_manifest.json"
    write_json(out, manifest)
    print(f"Frozen study manifest written to {out}")


def _stage1_points(manifest: dict):
    grid = manifest["architecture_grid"]
    for H, L, seed in itertools.product(grid["hidden_sizes"], grid["num_layers"], grid["seeds"]):
        yield H, L, seed


_HISTORY_GRID_KEY = {"cylinder200": "cylinder_history_grid", "bridge": "bridge_history_grid"}


def _stage2_points(manifest: dict, dataset: str):
    grid = manifest[_HISTORY_GRID_KEY[dataset]]
    for point, seed in itertools.product(grid["history_points"], grid["seeds"]):
        yield point, seed


def _write_train_array(manifest: dict, dataset: str) -> Path:
    grid = manifest["architecture_grid"]
    seq_len = grid["dataset_history"][dataset]["seq_len"]
    points = list(_stage1_points(manifest))
    job_name = f"stage1_train_{dataset}"
    study_rel = STUDY_ROOT.relative_to(REPO_ROOT)

    lines = [BSUB_HEADER.format(job_name=job_name, n=len(points),
                                 study_rel=study_rel, repo_root=REPO_ROOT)]
    lines.append("case \"$LSB_JOBINDEX\" in")
    for i, (H, L, seed) in enumerate(points, 1):
        lines.append(
            f'  {i}) H={H}; L={L}; SEED={seed} ;;'
        )
    lines.append('  *) echo "Invalid LSB_JOBINDEX=$LSB_JOBINDEX" >&2; exit 2 ;;')
    lines.append("esac")
    lines.append(
        f'python3 -u {study_rel}/scripts/train_sensitivity.py '
        f'--dataset {dataset} --hidden_size "$H" --num_layers "$L" '
        f'--seq_len {seq_len} --seed "$SEED" --stage 1'
    )
    out = STUDY_ROOT / "jobs" / f"{job_name}.bsub"
    out.write_text("\n".join(lines) + "\n")
    return out


def _write_eval_array(manifest: dict, kind: str) -> Path:
    """kind in {'open_loop', 'closed_loop'}; one array spanning both
    datasets' Stage 1 points (cylinder200 first, then bridge)."""
    points_cyl = [("cylinder200", H, L, seed, manifest["architecture_grid"]
                    ["dataset_history"]["cylinder200"]["seq_len"])
                  for H, L, seed in _stage1_points(manifest)]
    points_bri = [("bridge", H, L, seed, manifest["architecture_grid"]
                    ["dataset_history"]["bridge"]["seq_len"])
                  for H, L, seed in _stage1_points(manifest)]
    points = points_cyl + points_bri

    job_name = f"stage1_{kind}_validation"
    script_name = "evaluate_open_loop.py" if kind == "open_loop" else "evaluate_closed_loop.py"
    study_rel = STUDY_ROOT.relative_to(REPO_ROOT)

    lines = [BSUB_HEADER.format(job_name=job_name, n=len(points),
                                 study_rel=study_rel, repo_root=REPO_ROOT)]
    lines.append("case \"$LSB_JOBINDEX\" in")
    for i, (dataset, H, L, seed, seq_len) in enumerate(points, 1):
        run_dir = f"{study_rel}/results/{dataset}/stage1/H{H}_L{L}_seq{seq_len}_seed{seed}"
        lines.append(f'  {i}) RUN_DIR="{run_dir}" ;;')
    lines.append('  *) echo "Invalid LSB_JOBINDEX=$LSB_JOBINDEX" >&2; exit 2 ;;')
    lines.append("esac")
    lines.append(f'python3 -u {study_rel}/scripts/{script_name} --run_dir "$RUN_DIR"')
    out = STUDY_ROOT / "jobs" / f"{job_name}.bsub"
    out.write_text("\n".join(lines) + "\n")
    return out


def _write_stage2_disabled(manifest: dict, dataset: str) -> Path:
    """Writes jobs/stage2_train_eval_{dataset}.bsub (real, submittable
    #BSUB directives) if this dataset's OWN selection_manifest_{dataset}_
    stage1.json is already frozen at generation time, otherwise jobs/
    stage2_train_eval_{dataset}_DISABLED.bsub (directives commented out).
    Either way the script body keeps its own runtime frozen-check as a
    second line of defense -- generation-time and run-time can disagree
    (e.g. this was regenerated before freezing, then the file got frozen
    later without regenerating), so the runtime check is never skipped
    just because generation-time already saw frozen=true.
    """
    points = list(_stage2_points(manifest, dataset))
    job_name = f"stage2_train_eval_{dataset}"
    study_rel = STUDY_ROOT.relative_to(REPO_ROOT)

    selection_manifest_path = STUDY_ROOT / "manifests" / f"selection_manifest_{dataset}_stage1.json"
    is_frozen = False
    if selection_manifest_path.exists():
        is_frozen = bool(load_json(selection_manifest_path).get("frozen"))

    if is_frozen:
        header = [
            "#!/bin/bash",
            f'#BSUB -J "{job_name}[1-{len(points)}]"',
            '#BSUB -q BatchGPU',
            '#BSUB -gpu "num=1:mode=shared"',
            '#BSUB -R "rusage[mem=32000]"',
            '#BSUB -W 48:00',
            f'#BSUB -o {study_rel}/logs/{job_name}_%I.log',
        ]
        out_name = f"{job_name}.bsub"
    else:
        header = [
            "#!/bin/bash",
            f'# DISABLED -- Stage 2 ({dataset}) history-length sensitivity.',
            f'# BSUB -J "{job_name}[1-{len(points)}]"   # <- directives commented out on purpose',
            '# BSUB -q BatchGPU',
            '# BSUB -gpu "num=1:mode=shared"',
            '# BSUB -R "rusage[mem=32000]"',
            '# BSUB -W 48:00',
            f'# BSUB -o {study_rel}/logs/{job_name}_%I.log',
        ]
        out_name = f"{job_name}_DISABLED.bsub"

    lines = header + [
        "",
        "set -euo pipefail",
        f'SELECTION_MANIFEST="{study_rel}/manifests/selection_manifest_{dataset}_stage1.json"',
        'if [ ! -f "$SELECTION_MANIFEST" ]; then',
        '  echo "Stage 2 refused: no frozen Stage 1 selection manifest at $SELECTION_MANIFEST." >&2',
        '  exit 1',
        "fi",
        'if ! python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get(\'frozen\') else 1)" "$SELECTION_MANIFEST"; then',
        '  echo "Stage 2 refused: selection manifest exists but is not frozen." >&2',
        '  exit 1',
        "fi",
        "",
        f"cd {REPO_ROOT}",
        "source .venv/bin/activate",
        "export VIV_NUM_WORKERS=0",
        "export OMP_NUM_THREADS=4",
        ': "${LSB_JOBINDEX:?LSB_JOBINDEX is not set}"',
        "",
        f'ARCH=$(python3 -c "import json; sel=json.load(open(\'$SELECTION_MANIFEST\')); '
        f'c=sel[\'selected_configuration\']; print(c[\'hidden_size\'], c[\'num_layers\'])")',
        'H=$(echo $ARCH | cut -d" " -f1); L=$(echo $ARCH | cut -d" " -f2)',
        "",
        'case "$LSB_JOBINDEX" in',
    ]
    for i, (point, seed) in enumerate(points, 1):
        lines.append(f'  {i}) SEQ_LEN={point["seq_len"]}; SEED={seed}; '
                      f'HIST_LABEL="{point["label"]}" ;;')
    lines.append('  *) echo "Invalid LSB_JOBINDEX=$LSB_JOBINDEX" >&2; exit 2 ;;')
    lines.append("esac")
    lines.append(
        f'python3 -u {study_rel}/scripts/train_sensitivity.py '
        f'--dataset {dataset} --hidden_size "$H" --num_layers "$L" '
        f'--seq_len "$SEQ_LEN" --seed "$SEED" --stage 2 --history_label "$HIST_LABEL"'
    )
    lines.append(
        f'python3 -u {study_rel}/scripts/evaluate_open_loop.py --run_dir '
        f'"{study_rel}/results/{dataset}/stage2/${{HIST_LABEL}}_H${{H}}_L${{L}}_seq${{SEQ_LEN}}_seed${{SEED}}"'
    )
    lines.append(
        f'python3 -u {study_rel}/scripts/evaluate_closed_loop.py --run_dir '
        f'"{study_rel}/results/{dataset}/stage2/${{HIST_LABEL}}_H${{H}}_L${{L}}_seq${{SEQ_LEN}}_seed${{SEED}}"'
    )

    out = STUDY_ROOT / "jobs" / out_name
    out.write_text("\n".join(lines) + "\n")
    # If this dataset was previously ungenerated (or generated while still
    # unfrozen) and is now frozen, an old *_DISABLED.bsub for it would
    # otherwise linger alongside the new enabled one -- remove it so
    # there's exactly one, unambiguous script per dataset.
    stale = STUDY_ROOT / "jobs" / f"{job_name}_DISABLED.bsub"
    if is_frozen and stale.exists() and stale != out:
        stale.unlink()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze-manifest", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                    help="Print what would be generated without writing files.")
    args = p.parse_args()

    if args.freeze_manifest:
        freeze_manifest()
        return

    manifest_path = STUDY_ROOT / "manifests" / "study_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found -- run "
                          f"'python generate_jobs.py --freeze-manifest' first.")
    manifest = load_json(manifest_path)

    (STUDY_ROOT / "jobs").mkdir(parents=True, exist_ok=True)
    generated = []
    if not args.dry_run:
        generated.append(_write_train_array(manifest, "cylinder200"))
        generated.append(_write_train_array(manifest, "bridge"))
        generated.append(_write_eval_array(manifest, "open_loop"))
        generated.append(_write_eval_array(manifest, "closed_loop"))
        generated.append(_write_stage2_disabled(manifest, "cylinder200"))
        generated.append(_write_stage2_disabled(manifest, "bridge"))
    else:
        n1 = len(list(_stage1_points(manifest)))
        n2c = len(list(_stage2_points(manifest, "cylinder200")))
        n2b = len(list(_stage2_points(manifest, "bridge")))
        print(f"[DRY RUN] would generate: stage1_train_cylinder200[1-{n1}], "
              f"stage1_train_bridge[1-{n1}], stage1_open_loop_validation[1-{2*n1}], "
              f"stage1_closed_loop_validation[1-{2*n1}], "
              f"stage2_train_eval_cylinder200_DISABLED[1-{n2c}], "
              f"stage2_train_eval_bridge_DISABLED[1-{n2b}]")
        return

    print("Generated:")
    for g in generated:
        print(f"  {g}")


if __name__ == "__main__":
    main()
