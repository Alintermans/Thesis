#!/bin/bash -l
#SBATCH --time=03:00:00
module load cluster/wice/gpu_h100

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py generate no_constraints_with_guardrails None ${SLURM_ARRAY_TASK_ID}