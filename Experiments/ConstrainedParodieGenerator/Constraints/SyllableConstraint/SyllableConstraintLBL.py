from Constraint import Constraint
from SongUtils import  get_syllable_count_of_sentence, does_string_contain_newline
import torch
################################################ CONSTRAINT CLASS ################################################


class SyllableConstraintLBL(Constraint):
    def __init__(self, tokenizer, start_token=None):
        self.new_syllable_amount = None
        self.syllable_amount_prompt = None
        self.tokenizer = tokenizer
        self.start_token = start_token
        self.special_new_line_tokens = []
        self.new_line_tokens = tokenizer.encode('\n')
        if self.start_token is not None and self.start_token in self.new_line_tokens:
            self.new_line_tokens.remove(self.start_token)
        self.original_prompt = None
        #Hyperparameters
        self.good_beamscore_multiplier = 0.1 
        self.bad_beamscore_multiplier = 10
        self.top_k_tokens_to_consider = 30
        self.disable_constraint = False

    def set_special_new_line_tokens(self, special_new_line_tokens):
        self.special_new_line_tokens += special_new_line_tokens
    
    def set_original_prompt(self, original_prompt):
        self.original_prompt = original_prompt
    
    def disable(self):
        self.disable_constraint = True
    
    def enable(self):
        self.disable_constraint = False
    
    def set_hyperparameters(self, good_beamscore_multiplier=0.1, bad_beamscore_multiplier=10):
        self.good_beamscore_multiplier = good_beamscore_multiplier
        self.bad_beamscore_multiplier = bad_beamscore_multiplier
    
    def set_new_syllable_amount(self, new_syllable_amount):
        self.new_syllable_amount = new_syllable_amount
    
    def set_syllable_amount_prompt(self, syllable_amount_prompt):
        self.syllable_amount_prompt = syllable_amount_prompt
    
    def apply_beam_constraint(self, next_token, next_score, input_ids, cur_len, length_penalty):
        if self.disable_constraint:
            return next_score

        if self.new_syllable_amount is None:
            raise Exception('Syllable amount not set')
        # if self.syllable_amount_prompt is None:
        #     raise Exception('Syllable amount prompt not set')
        previous_text = self.tokenizer.decode(input_ids)
        current_token_text = self.tokenizer.decode(next_token)
        candidate_text = previous_text + current_token_text

        last_line = candidate_text[len(self.original_prompt):]
        result = get_syllable_count_of_sentence(last_line)
        
        current_length = input_ids.shape[-1] + 1
        
        # if does_string_contain_newline(last_line):
        #     next_score = next_score + next_score*self.bad_beamscore_multiplier* ( current_length ** length_penalty)
            

        if result > self.new_syllable_amount or (result == self.new_syllable_amount and get_syllable_count_of_sentence(current_token_text) == 0):
            next_score = next_score + next_score*self.bad_beamscore_multiplier
            #next_score = float('-inf')
        elif result == self.new_syllable_amount:
            next_score = next_score - next_score*self.good_beamscore_multiplier
        #print(candidate_text,' count: ' ,result, ' score: ', next_score)
        return next_score
    
    def stopping_criteria(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        if self.disable_constraint:
            return False

        if self.new_syllable_amount is None:
            raise Exception('Syllable amount not set')
        # if self.syllable_amount_prompt is None:
        #     raise Exception('Syllable amount prompt not set')
        result = []

        for input in input_ids:
            sentences = self.tokenizer.decode(input, skip_special_tokens=True)
            last_line = sentences[len(self.original_prompt):]
            sum = get_syllable_count_of_sentence(last_line)
            if sum >=self.new_syllable_amount:
                print('sum: ',sum, 'sentence: ', sentences[len(self.original_prompt):])
                result.append(True)
            else:
                result.append(False)
        
        if (len([x for x in result if x]) == len(result)):
            return True
        



        return False
    
    def logits_processor(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.disable_constraint:
            return scores

        if self.new_syllable_amount is None:
            raise Exception('Syllable amount not set')
        
        new_line_token = self.new_line_tokens
        # if (len(new_line_token) > 1):
        #     for i in range(len(input_ids)):
        #         input = input_ids[i]
        #         last_token = input[-1].item()
        #         if last_token == new_line_token[-2]:
        #             scores[i][new_line_token[-1]] = float('-inf')

        for i in range(len(input_ids)):
            input = input_ids[i]
            sentences = self.tokenizer.decode(input, skip_special_tokens=True)
            last_line = sentences[len(self.original_prompt):]
            sum = get_syllable_count_of_sentence(last_line)

            if sum < self.new_syllable_amount:
                

                _, best_tokens = scores[i].topk(self.top_k_tokens_to_consider)

                for token in best_tokens:
                    word = self.tokenizer.decode(token, skip_special_tokens=True)
                    syllable_count = get_syllable_count_of_sentence(word)
                    if syllable_count + sum > self.new_syllable_amount:
                        scores[i][token] = torch.finfo(scores.dtype).min
                    
                _, best_tokens = scores[i].topk(self.top_k_tokens_to_consider)
                for token in self.special_new_line_tokens:
                    scores[i][token] = torch.finfo(scores.dtype).min
                scores[i][self.tokenizer.eos_token_id] = torch.finfo(scores.dtype).min
            else:
                scores[i] = torch.finfo(scores.dtype).min
                scores[i][self.tokenizer.eos_token_id] = torch.finfo(scores.dtype).max

            

            
                
            
                




        return scores
    
    def is_beam_constraint_active(self):
        return True
    
    def is_stopping_criteria_active(self):
        return True
    
    def is_logits_processor_active(self):
        return True


