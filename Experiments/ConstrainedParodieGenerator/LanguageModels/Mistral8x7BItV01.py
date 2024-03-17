from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Mistral8x7BItV01(LM):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.name = 'Mistral 8x7B Instruct v0.1'
        return None
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name