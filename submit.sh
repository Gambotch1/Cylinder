#!/bin/bash
#BSUB -J direct_k10                # Job name
#BSUB -q BatchGPU                  # Target the massive GPU queue you found!
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32000]"       # Request 32GB of RAM
#BSUB -W 12:00                     # Max run time (12 hours)
#BSUB -o k10_train_%J.log          # Output log file (%J is the LSF Job ID)


echo "Starting LSF Control Job on node: $HOSTNAME"

# 1. Activate your Conda environment

# 2. Navigate to your project folder
cd /scratch/jote9827/Github/Cylinder
source .venv/bin/activate

export VIV_NUM_WORKERS=0

# 3. Run the Python script 
python -u -m src.viv_analysis.coupled_inference --Ur 6.7385\
    --cfd_dataset bridge --model_subdir gru_bridge_noise0.05 \
    --total_time 200 --handoff_offset 2000