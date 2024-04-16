from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Mistral7BItV02(LM):
    def __init__(self,  use_quantization=False, use_cuda=True):
        super().__init__(use_quantization, use_cuda)
        self.model_url = "mistralai/Mistral-7B-Instruct-v0.2"
        self.quantized_model_url = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
        self.setup_language_model()
        self.name = 'Mistral 7B Instruct v0.2'
        return None
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name
    
    def prepare_prompt(self, system_prompt, context_prompt, assistant_prompt):
        messages = [
            {"role": "user", "content": system_prompt + context_prompt}
        ]

        untokenized_prompt =  self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
        if assistant_prompt != '':
            untokenized_prompt += '\n' + assistant_prompt
            if not untokenized_prompt.endswith("\n"):
                untokenized_prompt += "\n"
            
        tokenized_prompt = self.tokenizer.encode(untokenized_prompt, return_tensors="pt") 
        prompt = self.tokenizer.decode(tokenized_prompt[0], skip_special_tokens=True)

        return prompt, tokenized_prompt