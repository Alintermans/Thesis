from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Mistral8x7BV01(LM):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
        self.name = 'Mistral 8x7B v0.1'
        return None
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name