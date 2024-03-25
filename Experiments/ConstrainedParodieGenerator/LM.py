from abc import ABC, abstractmethod
import inspect
import torch

class LM(ABC):

    def __init__(self):
        self.tokenizer = None
        self.model = None
        return None
    
    def get_start_token(self):
        return self.tokenizer.encode('')[0] if (len(self.tokenizer.encode(''))>0)  else None
    
    def set_to_gpu_if_possible(self):
        if self.model is not None and torch.cuda.is_available():
            self.model.to('cuda')
        return None
    
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
    


