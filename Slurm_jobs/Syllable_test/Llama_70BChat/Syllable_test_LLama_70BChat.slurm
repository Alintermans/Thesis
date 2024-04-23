#!/bin/bash -l
#SBATCH --job-name=Syllable_test_Llama2_chat70B
#SBATCH --output=Syllable_test_Llama2_chat70B.out
#SBATCH --error=Syllable_test_Llama2_chat70B.err

#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1 
#SBATCH --ntasks=4 
#SBATCH --gpus-per-node=1
#SBATCH --time=02:15:00
#SBATCH --mem-per-cpu=8000
#SBATCH --account=lp_dtai1

set -e 

module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"
source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py generate syllable Llama2_70BChat ${SLURM_ARRAY_TASK_ID}
