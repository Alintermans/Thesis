#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate no_constraints_with_guardrails None Experiments/ConstrainedParodieGenerator/CallibrationExperiments/NoConstraintsWithGuardrails/1