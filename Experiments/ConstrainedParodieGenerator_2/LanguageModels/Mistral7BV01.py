from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Mistral7BV01(LM):
    def __init__(self,  use_quantization=False, use_cuda=True):
        super().__init__(use_quantization, use_cuda)
        self.model_url = "mistralai/Mistral-7B-v0.1"
        self.quantized_model_url = "TheBloke/Mistral-7B-v0.1-AWQ"
        self.setup_language_model()
        self.name = 'Mistral 7B v0.1'
        return None
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_name():
        return 'Mistral 7B Instruct v0.2'
    
    def prepare_prompt(self, system_prompt, context_prompt, assistant_prompt):

        prompt = system_prompt + '\n' + context_prompt 
        if assistant_prompt != '':
            prompt += '\n' + assistant_prompt
            if not prompt.endswith("\n"):
                prompt += "\n"
        tokenized_prompt = self.tokenizer.encode(prompt, return_tensors="pt")   
        
        return prompt, tokenized_prompt