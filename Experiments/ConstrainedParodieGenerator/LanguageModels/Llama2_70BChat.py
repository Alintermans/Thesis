from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Llama2_70BChat(LM):
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            use_4bit = True
            # Compute dtype for 4-bit base models
            bnb_4bit_compute_dtype = "float16"
            # Quantization type (fp4 or nf4)
            bnb_4bit_quant_type = "nf4"
            use_nested_quant = False

            compute_dtype = getattr(torch, "float16")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_nested_quant,
            )
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf", token= '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')
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", 
            token= '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',
            quantization_config=bnb_config,
            device_map = 'auto'
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf", token= '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')
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", token= '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')
        self.name = 'Llama 2 70B Chat'
        return None

    def special_new_line_tokens(self):
        return [13]
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name
    
    def prepare_prompt(self, system_prompt, context_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_prompt}
        ]

        tokenized_prompt =  self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        prompt = self.tokenizer.decode(tokenized_prompt, skip_special_tokens=True)

        return prompt, tokenized_prompt