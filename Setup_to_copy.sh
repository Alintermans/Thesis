#!/bin/bash
# Create and activate a new Python virtual environment
cd $VSC_SCRATCH
module load Python/3.10.8-GCCcore-12.2.0
export HF_HOME="/scratch/leuven/361/vsc36141/HF/"
source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/ParodieGenLBL.py