#!/bin/bash
#BSUB -J "cylinder200_training_ctx_noise0.05"
#BSUB -q BatchGPU
#BSUB -gpu "num=1:mode=shared"
#BSUB -R "rusage[mem=120000]"
#BSUB -W 48:00
#BSUB -o cylinder200_training_ctx_noise0.05_%J_%I.log

set -euo pipefail

cd /scratch/jote9827/Github/Cylinder
source .venv/bin/activate

# Standard (dimensional) cylinder200 GRU baseline -- new artifact dir,
# never collides with any legacy gru_cylinder1000/gru_cylinder_re_1000*
# directory (see tests/test_cylinder200_migration.py::TestNoLegacyOverwrite).
PYTHONPATH=src python3 -u -m viv_analysis.train_gru \
    --cfd_dataset cylinder200 \
    --use_ur_context \
    --noise_std 0.05 \
    --exp_subdir gru_cylinder200_ctx_noise0.05
