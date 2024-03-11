from Constraint import Constraint
from SongUtils import get_pos_tags_of_line
import torch
################################################ CONSTRAINT CLASS ################################################


class RhymingConstraintLBL(Constraint):
    def __init__(self, tokenizer, start_token=None):
        self.tokenizer = tokenizer
        self.start_token = start_token


    
    def apply_beam_constraint(self, next_token, next_score, input_ids, cur_len, length_penalty):
        
        previous_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        current_token = self.tokenizer.decode(next_token, skip_special_tokens=True)
        candidate_text = previous_text + current_token
        last_line = candidate_text.split('\n')[-1]
        
        return next_score

        
    
        

    
    def stopping_criteria(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        return False
    
    def logits_processor(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        


        for i in range(len(input_ids)):
            pass
                    
        return scores
    
    def is_beam_constraint_active(self):
        return True
    
    def is_stopping_criteria_active(self):
        return False
    
    def is_logits_processor_active(self):
        return True


