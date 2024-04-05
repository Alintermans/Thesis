from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

class GPT2(LM):
    def __init__(self,  use_quantization=False, use_cuda=True):
        super().__init__(use_quantization, use_cuda)
        if self.use_cuda and not self.use_quantization:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", token= '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')
            self.model = AutoModelForCausalLM.from_pretrained("gpt2", token= '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', device_map='auto')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", token= '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')
            self.model = AutoModelForCausalLM.from_pretrained("gpt2", token= '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')
        self.name = 'GPT2'
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