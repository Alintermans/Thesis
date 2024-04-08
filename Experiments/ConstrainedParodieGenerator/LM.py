from abc import ABC, abstractmethod
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import inspect
import torch

class LM(ABC):

    def __init__(self, use_quantization=False, use_cuda=True):
        self.tokenizer = None
        self.model = None
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.use_quantization = use_quantization and torch.cuda.is_available()
        self.model_url = None

        ## quantization config
        use_4bit = True
        # Compute dtype for 4-bit base models
        bnb_4bit_compute_dtype = "float16"
        # Quantization type (fp4 or nf4)
        bnb_4bit_quant_type = "nf4"
        use_nested_quant = False

        compute_dtype = getattr(torch, "float16")

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            #bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            #bnb_4bit_use_double_quant=use_nested_quant,
        )

        return None
    
    def setup_language_model(self):
        if self.use_cuda and self.use_quantization:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_url, token= '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')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_url, 
            token= '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',
            quantization_config=self.bnb_config,
            device_map = 'auto'
            )
        elif self.use_cuda and not self.use_quantization:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_url, token= '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')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_url, token= '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', device_map='auto')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_url, token= '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')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_url, token= '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')
    
    def get_start_token(self):
        return self.tokenizer.encode('')[0] if (len(self.tokenizer.encode(''))>0)  else None
    
    def set_to_gpu_if_possible(self):
        if self.model is not None and torch.cuda.is_available():
            self.model.to('cuda')
        return None
    
    
        
    
    def special_new_line_tokens(self):
        return []
    
    @abstractmethod
    def prepare_prompt(self, system_prompt, context_prompt):
        raise NotImplementedError("prepare_prompt not implemented yet")

    @abstractmethod
    def get_tokenizer(self):
        raise NotImplementedError("tokenizer not implemented yet")
    
    @abstractmethod
    def get_model(self):
        raise NotImplementedError("model not implemented yet")
    
    @abstractmethod
    def get_name(self):
        raise NotImplementedError("name not implemented yet")
    
    def accepts_attention_mask(self):
        return "attention_mask" in set(inspect.signature(self.model.forward).parameters.keys())
    


