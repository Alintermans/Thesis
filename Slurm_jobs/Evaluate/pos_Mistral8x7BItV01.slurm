#!/bin/bash -l
#SBATCH --time=01:30:00

module load cluster/genius/gpu_v100

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate pos Mistral8x7BItV01 Experiments/ConstrainedParodieGenerator/CallibrationExperiments/PosConstraint/0