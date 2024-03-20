#!/bin/bash
# Create and activate a new Python virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install Python libraries
#pip install --upgrade https://github.com/Alintermans/transformers/tree/main
pip install git+https://github.com/Alintermans/transformers.git
pip install accelerate bitsandbytes sentencepiece evaluate

# Download the code
git clone https://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@github.com/Alintermans/Thesis.git
cd Thesis

# Run the code
python3 Experiments/ConstrainedParodieGenerator/ParodieGenLBL.py