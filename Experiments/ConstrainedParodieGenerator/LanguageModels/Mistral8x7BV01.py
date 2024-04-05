from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Mistral8x7BV01(LM):
    def __init__(self,  use_quantization=False, use_cuda=True):
        super().__init__(use_quantization, use_cuda)
        self.model_url = "mistralai/Mixtral-8x7B-v0.1"
        self.setup_language_model()
        self.name = 'Mistral 8x7B v0.1'
        return None
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name
    
    def prepare_prompt(self, system_prompt, context_prompt):

        prompt = system_prompt + '\n' + context_prompt
        tokenized_prompt = self.tokenizer.encode(prompt, return_tensors="pt")   
        
        return prompt, tokenized_prompt