from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Gemma7BIt(LM):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it",  token= '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')
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it",  token= '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')
        self.name = 'Gemma 7B It'
        return None
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name