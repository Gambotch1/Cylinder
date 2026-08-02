#!/bin/bash
#BSUB -J "cylinder200_sweep"
#BSUB -q BatchGPU
#BSUB -gpu "num=1:mode=shared"
#BSUB -R "rusage[mem=120000]"
#BSUB -W 48:00
#BSUB -o cylinder200_sweep_%J_%I.log

set -euo pipefail

cd /scratch/jote9827/Github/Cylinder
source .venv/bin/activate

# Regenerates coupled-inference plots/tables/metrics (evaluate_all.py
# requirement #10) for the cylinder200 model trained by
# submit_cylinder200_train.sh. Run AFTER that job completes --
# gru_cylinder200_ctx_noise0.05/gru_best.pt must exist.
# Writes to a new results/cylinder200_sweep/ subdir; never touches any
# legacy cylinder1000 sweep output.
PYTHONPATH=src python3 -u -m viv_analysis.evaluate_all \
    --dataset cylinder200 \
    --model_subdir gru_cylinder200_ctx_noise0.05 \
    --output_dir cylinder200_sweep
