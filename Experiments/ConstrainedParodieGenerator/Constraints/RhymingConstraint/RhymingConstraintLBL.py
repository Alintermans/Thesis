from Constraint import Constraint
from SongUtils import  get_syllable_count_of_sentence
import torch
################################################ CONSTRAINT CLASS ################################################


class RhymingConstraintLBL(Constraint):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def apply_beam_constraint(self, next_token, next_score, input_ids, cur_len, length_penalty):
        
        return next_score
    
    def stopping_criteria(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        

        return False
    
    def logits_processor(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores
                




        return scores
    
    def is_beam_constraint_active(self):
        return True
    
    def is_stopping_criteria_active(self):
        return True
    
    def is_logits_processor_active(self):
        return True


