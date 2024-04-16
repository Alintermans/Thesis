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
        self.quantized_model_url = None
        self.quantized_revision = "main"

        return None
    
    def setup_language_model(self):
        if self.quantized_model_url is None:
            self.quantized_model_url = self.model_url

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_url, token= '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', use_fast=True)
        if self.use_cuda and self.use_quantization:
            self.model = AutoModelForCausalLM.from_pretrained(self.quantized_model_url, revision=self.quantized_revision,
            token= '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',
            device_map = 'auto'
            )
        elif self.use_cuda and not self.use_quantization:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_url, token= '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', device_map='auto')
        else:
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
    


