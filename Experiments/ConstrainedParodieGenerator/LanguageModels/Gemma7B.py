from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Gemma7B(LM):
    def __init__(self,  use_quantization=False, use_cuda=True):
        super().__init__(use_quantization, use_cuda)
        self.model_url = "google/gemma-7b"
        self.setup_language_model()
        self.name = 'Gemma 7B'
        return None
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name
    
    def prepare_prompt(self, system_prompt, context_prompt, assistant_prompt):

        prompt = system_prompt + '\n' + context_prompt + '\n' + assistant_prompt
        tokenized_prompt = self.tokenizer.encode(prompt, return_tensors="pt")   
        
        return prompt, tokenized_prompt