# Thesis 

This repository contains the source code and related documents to my master thesis: Enforcing creative constraints in autoregressive language models during generation for musical parodies.

This repository contains:
- The source code to generate the parodies using any autoregressive models.
- The experiments and results for the conducted tests, as described in the thesis.

As described in the thesis, there was a bug when the tests were conducted, which resulted in the generation stopping whenever one of the beams had reached the correct syllable count and not completing all beams. Secondly, an improvement was made that allows punctation marks and capitalized line starts. Both are implemented together in a second version of the code.

## Structure
- [Version With Bug](./Experiments/ConstrainedParodieGenerator): This contains all the source code needed to recreate all the experiments and to generate parodies using any autoregressive model.
- [Version Witt Bug Fix and Improvement](./Experiments/ConstrainedParodieGenerator_2): This contains the same as the previous folder, but with the bug fix and improvement implemented.
- [Songs](./Songs/): This contains all the songs used in the experiments.
- [Parodies](./Experiments/ConstrainedParodieGenerator/CallibrationExperiments): This contains all the generated parodies with a json and txt format. The json format also includes all parameter settings used for the generation.
- [Results](./Experiments/ConstrainedParodieGenerator/CallibrationExperiments): This contains all the evaluations from the experiments.
- [Survey Results](Experiments/ConstrainedParodieGenerator/Survey): This contains all the results from the human evaluation.
