#!/bin/bash -l

#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --time=03:15:00
#SBATCH --nodes=1 
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=1
#SBATCH --account=lp_dtai1

set -e 

module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"
source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py generate syllable Mistral8x7BItV01 ${SLURM_ARRAY_TASK_ID}
