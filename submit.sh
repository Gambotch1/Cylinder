#!/bin/bash
#BSUB -J "bridge_inference"
#BSUB -q BatchGPU
#BSUB -gpu "num=1:mode=shared"
#BSUB -R "rusage[mem=120000]"
#BSUB -W 48:00
#BSUB -o logs/diagnose/bridge_nd_diagnose%J_%I.log

set -euo pipefail

cd /scratch/jote9827/Github/Cylinder
source .venv/bin/activate

PYTHONPATH=src python3 -u -m viv_analysis.diagnose_off_manifold \
      --Ur 6.0 --handoff_offset 2000 \
      --checkpoint gru_best_.pt \
      --model_subdir gru_bridge_nd_context_noise0.05 \
      --cfd_dataset cylinder_re_1000 --total_time 500