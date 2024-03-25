from LM import LM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class Llama2_7BChat(LM):
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
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token= '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')
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", 
            token= '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',
            quantization_config=bnb_config
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token= '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')
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token= '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')
        self.name = 'Llama 2 7B Chat'
        return None
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_name(self):
        return self.name