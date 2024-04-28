#!/bin/bash -l
#SBATCH --time=04:30:00
module load cluster/wice/gpu_h100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"
source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py generate pos Llama2_70BChat ${SLURM_ARRAY_TASK_ID}