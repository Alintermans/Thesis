from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Llama2_7B(LM):
    def __init__(self,  use_quantization=False, use_cuda=True):
        super().__init__(use_quantization, use_cuda)
        self.model_url = "meta-llama/Llama-2-7b-hf"
        self.quantized_model_url = "TheBloke/Llama-2-7B-AWQ"
        self.setup_language_model()
        self.name = 'Llama 2 7B'
        return None
    

    def special_new_line_tokens(self):
        return [13]

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name
    
    def prepare_prompt(self, system_prompt, context_prompt, assistant_prompt):

        prompt = system_prompt + '\n' + context_prompt 
        if assistant_prompt != '':
            prompt += '\n' + assistant_prompt
            if not prompt.endswith("\n"):
                prompt += "\n"
        tokenized_prompt = self.tokenizer.encode(prompt, return_tensors="pt")   
        
        return prompt, tokenized_prompt