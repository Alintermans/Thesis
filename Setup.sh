#!/bin/bash
# Create and activate a new Python virtual environment
python3.8 -m venv myenv
source myenv/bin/activate

# Install Python libraries
#pip install --upgrade https://github.com/Alintermans/transformers/tree/main
pip install git+https://github.com/Alintermans/transformers.git
pip install accelerate bitsandbytes sentencepiece evaluate torch nltk pronouncing

# Download the code
git clone https://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@github.com/Alintermans/Thesis.git
cd Thesis

# Run the code
python3 Experiments/ConstrainedParodieGenerator/ParodieGenLBL.py