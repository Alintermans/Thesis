from abc import ABC, abstractmethod

class LM(ABC):

    def __init__(self):
        self.tokenizer = None
        self.model = None
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
    


