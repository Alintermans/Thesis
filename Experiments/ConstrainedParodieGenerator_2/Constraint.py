from abc import ABC, abstractmethod
from transformers import StoppingCriteriaList, LogitsProcessorList, LogitsProcessor
import torch

class Constraint(ABC):
    #Abstract class for constraints

    @abstractmethod
    def get_name(self):
        raise NotImplementedError("get_name not implemented")
        return ""
    
    @abstractmethod
    def get_hyperparameters_in_dict(self):
        raise NotImplementedError("get_name not implemented")
        return {}

    @abstractmethod
    def set_hyperparameters(self, **kwargs):
        raise NotImplementedError("set_hyperparameters not implemented")
    
    
    @staticmethod
    def hyperparameters_config(**kwargs):
        raise NotImplementedError("hyperparameters_config not implemented")
    
    #Returns a score for the next token, given the current token, the current score and the current input_ids within the beam search, the next score is the cumulated neg log likelihood of the beam search
    def apply_beam_constraint(self, nexttoken, next_score, input_ids, cur_len, length_penalty):
        if self.is_beam_constraint_active():
            raise NotImplementedError("apply_beam_constraint not implemented")
        else:
            return next_score
    

    
    def stopping_criteria(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        raise NotImplementedError("stopping_criteria not implemented")
    
    
    def logits_processor(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.is_logits_processor_active():
            raise NotImplementedError("logits_processor not implemented")
        else:
            return scores
    
    @abstractmethod
    def is_constrained_satisfied(self, generated_text):
        raise NotImplementedError("is_constrained_satisfied not implemented")
        return True
    
    
    
    @abstractmethod
    def is_beam_constraint_active(self):
        raise NotImplementedError("is_beam_constraint_active not implemented")
    
    @abstractmethod
    def is_stopping_criteria_active(self):
        raise NotImplementedError("get_stopping_criteria not implemented")
    
    @abstractmethod
    def is_logits_processor_active(self):
        raise NotImplementedError("get_logits_processor not implemented")


class ConstraintLogitsProcessor(LogitsProcessor):  
    def __init__(self, constraint):
        self.constraint = constraint
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.constraint.logits_processor(input_ids, scores)

class ConstraintList:
    def __init__(self, constraints=[]):
        self.constraints = constraints
    
    def add_constraint(self, constraint):
        self.constraints.append(constraint)
    
    def get_stopping_criteria_list(self):
        stopping_criteria_list = []
        for constraint in self.constraints:
            if constraint.is_stopping_criteria_active():
                stopping_criteria_list.append(constraint.stopping_criteria)
        return stopping_criteria_list
    
    def get_logits_processor_list(self):
        logits_processor_list = []
        for constraint in self.constraints:
            if constraint.is_logits_processor_active():
                logits_processor_list.append(ConstraintLogitsProcessor(constraint))
        return logits_processor_list
    
    def are_constraints_satisfied(self, generated_text):
        nb_constraints_satisfied = 0
        result = True
        for constraint in self.constraints:
            if not constraint.is_constrained_satisfied(generated_text):
                result = False
            else:   
                nb_constraints_satisfied += 1
        return result, nb_constraints_satisfied
    
    def __iter__(self):
        return iter(self.constraints)


    

