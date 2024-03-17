from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Llama2_70B(LM):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf", token= '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')
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf", token= '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')
        self.name = 'Llama 2 70B'
        return None
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name