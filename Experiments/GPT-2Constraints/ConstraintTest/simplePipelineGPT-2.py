from transformers import pipeline, set_seed, TextGenerationPipeline
from ConstrainedPipeline import ConstrainedPipeline 


generator = pipeline(model='gpt2', pipeline_class=ConstrainedPipeline)
set_seed(42)
result = generator("Hello, I'm a a parody lyrics maker and I make beatifull parodies,", max_length=1000, num_return_sequences=1)
print(result[0]['generated_text'])