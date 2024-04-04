from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Gemma2B(LM):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b",  token= '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')
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b",  token= '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')
        self.name = 'Gemma 2B'
        return None
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name
    
    def prepare_prompt(self, system_prompt, context_prompt):
        return system_prompt + '\n' + context_prompt