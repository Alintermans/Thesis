from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Gemma2BIt(LM):
    def __init__(self,  use_quantization=False, use_cuda=True):
        super().__init__(use_quantization, use_cuda)
        self.model_url = "google/gemma-2b-it"
        self.setup_language_model()
        self.name = 'Gemma 2B It'
        return None
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name
    
    def prepare_prompt(self, system_prompt, context_prompt, assistant_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_prompt},
            {"role": "assistant", "content": assistant_prompt}
        ]

        tokenized_prompt =  self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        prompt = self.tokenizer.decode(tokenized_prompt[0], skip_special_tokens=True)

        return prompt, tokenized_prompt