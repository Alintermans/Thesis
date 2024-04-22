#!/bin/bash -l
#SBATCH --job-name=Syllable_test_Mistral8x7BItV01
#SBATCH --output=Syllable_test_Mistral8x7BItV01.out
#SBATCH --error=Syllable_test_Mistral8x7BItV01.err

#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1 
#SBATCH --ntasks=8 
#SBATCH --gpus-per-node=1
#SBATCH --time=02:15:00
#SBATCH --mem-per-cpu=8000
#SBATCH --account=intro_vsc36141

set -e 

module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"
source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py syllable Mistral8x7BItV01 ${SLURM_ARRAY_TASK_ID}
