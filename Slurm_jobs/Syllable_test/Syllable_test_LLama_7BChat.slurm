#!/bin/bash -l
#SBATCH --job-name=Syllable_test_Llama2_chat7B_${SLURM_ARRAY_TASK_ID}
#SBATCH --output=Syllable_test_Llama2_chat7B_${SLURM_ARRAY_TASK_ID}.out
#SBATCH --error=Syllable_test_Llama2_chat7B_${SLURM_ARRAY_TASK_ID}.err

set -e 

#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1 
#SBATCH --ntasks=8 
#SBATCH --gpus-per-node=1
#SBATCH --time=02:15:00
#SBATCH --mem-per-cpu=8000
#SBATCH --account=intro_vsc36141


module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"
source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py syllable Llama2_7BChat ${SLURM_ARRAY_TASK_ID}
