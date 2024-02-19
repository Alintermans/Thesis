from Constraint import Constraint
from SongUtils import tokenize_sentence, count_syllables, get_syllable_count_of_sentence
import torch
################################################ CONSTRAINT CLASS ################################################


class SyllableConstraint(Constraint):
    def __init__(self, syllable_amount, tokenizer):
        self.syllable_amount = syllable_amount
        self.tokenizer = tokenizer
    
    def set_syllable_amount(self, syllable_amount):
        self.syllable_amount = syllable_amount
    
    def apply_beam_constraint(self, next_token, next_score, input_ids, cur_len, length_penalty):
        previous_text = self.tokenizer.decode(input_ids)
        current_token_text = self.tokenizer.decode(next_token)
        candidate_text = previous_text + current_token_text

        words = tokenize_sentence(candidate_text)
        result = sum(count_syllables(word) for word in words)
        
        current_length = input_ids.shape[-1] + 1
        

        if result > self.syllable_amount or (result == self.syllable_amount and count_syllables(current_token_text) == 0):
            next_score = next_score + next_score*(10) * ( current_length ** length_penalty)
            #next_score = float('-inf')
        elif result == self.syllable_amount:
            next_score = next_score - next_score*0.1 * ( current_length ** length_penalty)
        #print(candidate_text,' count: ' ,result, ' score: ', next_score)
        return next_score
    
    def stopping_criteria(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        for input in input_ids:
            sentence = self.tokenizer.decode(input, skip_special_tokens=True)
            words = tokenize_sentence(sentence)
            sum = 0
            for word in words:
                sum += count_syllables(word) 
            if sum >=self.syllable_amount:
                #print('sum: ',sum, 'sentence: ', sentence)
                return True

        return False
    
    def is_beam_constraint_active(self):
        return True
    
    def is_stopping_criteria_active(self):
        return True
    
    def is_logits_processor_active(self):
        return False


