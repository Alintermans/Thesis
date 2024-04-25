from Constraint import Constraint
from SongUtils import  get_syllable_count_of_sentence, does_string_contain_newline, only_adds_regular_characters, last_word_only_has_consontants, does_not_contain_special_characters
import torch
import time
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
        self.disable_constraint = False
        self.eos_string = self.tokenizer.decode(self.tokenizer.eos_token_id)
        self.tokenized_prompt_length = None


        #Hyperparameters
        self.good_beamscore_multiplier = None
        self.bad_beamscore_multiplier = None
        self.top_k_tokens_to_consider = None
        self.all_beams_have_syllable_amount = None
        
    
    def get_name(self):
        return 'SyllableConstraintLBL'

    def get_hyperparameters_in_dict(self):
        return {
            self.get_name(): {
                'good_beamscore_multiplier': self.good_beamscore_multiplier,
                #'bad_beamscore_multiplier': self.bad_beamscore_multiplier,
                'top_k_tokens_to_consider': self.top_k_tokens_to_consider,
                'all_beams_have_syllable_amount': self.all_beams_have_syllable_amount
            }
        }

    @staticmethod
    def hyperparameters_config(good_beamscore_multiplier=0.1, top_k_tokens_to_consider=30, all_beams_have_syllable_amount=False):
        return {
            'good_beamscore_multiplier': good_beamscore_multiplier,
            #'bad_beamscore_multiplier': bad_beamscore_multiplier,
            'top_k_tokens_to_consider': top_k_tokens_to_consider,
            'all_beams_have_syllable_amount': all_beams_have_syllable_amount
        }

    def set_hyperparameters(self, config):
        self.good_beamscore_multiplier = config['good_beamscore_multiplier']
       #self.bad_beamscore_multiplier = config['bad_beamscore_multiplier']
        self.top_k_tokens_to_consider = config['top_k_tokens_to_consider']
        self.all_beams_have_syllable_amount = config['all_beams_have_syllable_amount']


    def set_special_new_line_tokens(self, special_new_line_tokens):
        self.special_new_line_tokens += special_new_line_tokens
    
    def set_original_prompt(self, original_prompt, tokenized_prompt_length=0):
        self.original_prompt = original_prompt
        self.tokenized_prompt_length = tokenized_prompt_length
    
    def disable(self):
        self.disable_constraint = True
    
    def enable(self):
        self.disable_constraint = False
    
    
    
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


        next_token_tensor = torch.tensor([next_token.item()], device = next_token.device)
        candidate_text = self.tokenizer.decode(torch.cat([input_ids, next_token_tensor], dim=0), skip_special_tokens=True)
        current_token_text = self.tokenizer.decode(next_token, skip_special_tokens=True)
        # candidate_text = previous_text + current_token_text

        last_line = candidate_text[len(self.original_prompt):]
        if last_line.endswith(self.eos_string):
            #print('eos token')
            last_line = last_line.replace(self.eos_string, '')

        result = get_syllable_count_of_sentence(last_line)
        
        current_length = input_ids.shape[-1] + 1
        
        # if does_string_contain_newline(last_line):
        #     next_score = next_score + next_score*self.bad_beamscore_multiplier* ( current_length ** length_penalty)

        if next_score != next_score:
            next_score = torch.tensor(0.0, device=next_score.device)
        
        # if result > self.new_syllable_amount or (result == self.new_syllable_amount and get_syllable_count_of_sentence(current_token_text) == 0) or does_string_contain_newline(last_line):
            
        #     next_score = next_score + next_score*self.bad_beamscore_multiplier
        #     #next_score = float('-inf')
        elif result == self.new_syllable_amount:
            next_score = next_score - next_score*self.good_beamscore_multiplier
            #print("next score=", next_score)
        #print(candidate_text[len(self.original_prompt):],' count: ' ,result, ' score: ', next_score)
        
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
            if last_line.endswith(self.eos_string):
                last_line = last_line.replace(self.eos_string, '')
            sum = get_syllable_count_of_sentence(last_line)
            last_token = input[-1].item()
            if sum >=self.new_syllable_amount or last_token == self.tokenizer.eos_token_id:
                #print('sum: ',sum, 'sentence: ', sentences[len(self.original_prompt):])
                result.append(True)
            elif input.shape[-1] - self.tokenized_prompt_length > 50:
                result.append(True)
            else:
                result.append(False)
            
            
        
        # if (len([x for x in result if x]) == len(result)):
        #     return True
        if self.all_beams_have_syllable_amount:
            if len([x for x in result if x]) == len(result):
                return True
        elif len([x for x in result if x]) >0:
                return True


        return False
    
    # def logits_processor(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    #     if self.disable_constraint:
    #         return scores

    #     if self.new_syllable_amount is None:
    #         raise Exception('Syllable amount not set')

    #     # Prepare for operations
    #     min_score = torch.finfo(scores.dtype).min
    #     eos_token_id = self.tokenizer.eos_token_id
    #     bos_token_id = self.tokenizer.bos_token_id
    #     prompt_length = len(self.original_prompt)
    #     original_prompt_length = len(self.original_prompt)

    #     # Process each input sequence
    #     for i in range(len(input_ids)):
    #         input_seq = input_ids[i]
    #         decoded_text = self.tokenizer.decode(input_seq, skip_special_tokens=True)
    #         last_line = decoded_text[original_prompt_length:]

    #         # Handle special tokens and end of string
    #         if last_line.endswith(self.eos_string):
    #             last_line = last_line.replace(self.eos_string, '')

    #         syllable_count = get_syllable_count_of_sentence(last_line)
    #         scores[i][bos_token_id] = min_score

    #         # Main logic based on the syllable count
    #         if syllable_count < self.new_syllable_amount:
    #             scores = self.process_token_selection(scores, i, input_seq, last_line, decoded_text, syllable_count, min_score, eos_token_id, prompt_length)
    #         elif syllable_count == self.new_syllable_amount:
    #             scores = self.process_completion(scores, i, input_seq, last_line, decoded_text, min_score, eos_token_id, prompt_length)
    #         else:
    #             scores[i] = torch.abs(scores[i]) * min_score
    #             scores[i][eos_token_id] = torch.tensor(-1, device=scores.device)

    #     return scores

    # def process_token_selection(self, scores, index, input_seq, last_line, decoded_text, current_syllable_count, min_score, eos_token_id, prompt_length):
    #     best_scores, best_tokens = scores[index].topk(self.top_k_tokens_to_consider)
    #     scores[index] = torch.abs(scores[index]) * min_score

    #     if best_scores[0].item() == float('-inf'):
    #         scores[index][eos_token_id] = torch.tensor(-1, device=scores.device)

    #     for score, token in zip(best_scores, best_tokens):
    #         next_token_tensor = torch.tensor([token], device=scores.device)
    #         candidate_text = self.tokenizer.decode(torch.cat([input_seq, next_token_tensor], dim=0), skip_special_tokens=True)[prompt_length:]
    #         syllable_count = get_syllable_count_of_sentence(candidate_text)

    #         if self.is_valid_candidate(syllable_count, candidate_text, current_syllable_count):
    #             scores[index][token] = score
        
    #     return scores

    # def process_completion(self, scores, index, input_seq, last_line, decoded_text, min_score, eos_token_id, prompt_length):
    #     best_scores, best_tokens = scores[index].topk(2)
    #     activated = False
    #     scores[index] = torch.abs(scores[index]) * min_score

    #     for score, token in zip(best_scores, best_tokens):
    #         next_token_tensor = torch.tensor([token], device=scores.device)
    #         candidate_text = self.tokenizer.decode(torch.cat([input_seq, next_token_tensor], dim=0), skip_special_tokens=True)[prompt_length:]
    #         syllable_count = get_syllable_count_of_sentence(candidate_text)

    #         if self.is_valid_completion(candidate_text, syllable_count):
    #             scores[index][token] = score
    #             activated = True

    #     if not activated:
    #         scores[index][eos_token_id] = torch.tensor(-1, device=scores.device)
        
    #     return scores

    # def is_valid_candidate(self, syllable_count, candidate_text, current_syllable_count):
    #     if  syllable_count <= self.new_syllable_amount and not does_string_contain_newline(candidate_text) and does_not_contain_special_characters(candidate_text):
    #         if syllable_count == self.new_syllable_amount:
    #             if not last_word_only_has_consontants(candidate_text):
    #                 return True
    #         else:
    #             return True
    #     else:
    #         return False

    # def is_valid_completion(self, candidate_text, syllable_count):
    #     return syllable_count <= self.new_syllable_amount and not does_string_contain_newline(candidate_text) and only_adds_regular_characters(candidate_text)

    
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
            if last_line.endswith(self.eos_string):
                last_line = last_line.replace(self.eos_string, '')
            sum = get_syllable_count_of_sentence(last_line)
            scores[i][self.tokenizer.bos_token_id] = torch.finfo(scores.dtype).min
            if sum < self.new_syllable_amount:
                

                best_scores, best_tokens = scores[i].topk(self.top_k_tokens_to_consider)
                if best_scores[0].item() == float('-inf'):
                    scores[i][self.tokenizer.eos_token_id] = torch.tensor(-1, device = scores.device)
                
                scores[i] = abs(scores[i])*torch.finfo(scores.dtype).min
                #print(best_scores)
                for score, token in zip(best_scores, best_tokens):
                    next_token_tensor = torch.tensor([token], device = scores[i].device)
                    candidate_text = self.tokenizer.decode(torch.cat([input, next_token_tensor], dim=0), skip_special_tokens=True)
                    syllable_count = get_syllable_count_of_sentence(candidate_text[len(self.original_prompt):])
                    if syllable_count <= self.new_syllable_amount and not does_string_contain_newline(candidate_text[len(self.original_prompt):])and does_not_contain_special_characters(candidate_text[len(self.original_prompt):]):
                        if syllable_count == self.new_syllable_amount:
                            if not last_word_only_has_consontants(candidate_text[len(self.original_prompt):]):
                                scores[i][token] = score
                        else:
                            scores[i][token] = score
                    
                
                for token in self.special_new_line_tokens:
                    scores[i][token] = torch.finfo(scores.dtype).min
                scores[i][self.tokenizer.eos_token_id] = torch.finfo(scores.dtype).min
            elif sum == self.new_syllable_amount:
                best_scores, best_tokens = scores[i].topk(2)
                if best_scores[0].item() == float('-inf'):
                    scores[i][self.tokenizer.eos_token_id] = torch.tensor(-1, device = scores.device)
                
                scores[i] = abs(scores[i])*torch.finfo(scores.dtype).min

                activated = False

                #print(best_scores)
                for score, token in zip(best_scores, best_tokens):
                    next_token_tensor = torch.tensor([token], device = scores[i].device)
                    candidate_text = self.tokenizer.decode(torch.cat([input, next_token_tensor], dim=0), skip_special_tokens=True)
                    syllable_count = get_syllable_count_of_sentence(candidate_text[len(self.original_prompt):])
                    if syllable_count <= self.new_syllable_amount and not does_string_contain_newline(candidate_text[len(self.original_prompt):]) and only_adds_regular_characters(last_line,candidate_text[len(self.original_prompt):]):
                        scores[i][token] = score
                        #print(last_line,candidate_text[len(self.original_prompt):])
                        activated = True 

                
                if not activated:
                    scores[i][self.tokenizer.eos_token_id] = torch.tensor(-1, device = scores.device)


            else:
                eos_score = scores[i][self.tokenizer.eos_token_id]
                scores[i] = abs(scores[i])*torch.finfo(scores.dtype).min
                scores[i][self.tokenizer.eos_token_id] = torch.tensor(-1, device = scores.device)
            


        return scores

    def apply_optimzed_logit_processor(self, input_ids, scores, best_tokens, last_line, new_lines):
        if self.disable_constraint:
            return scores

        if self.new_syllable_amount is None:
            raise Exception('Syllable amount not set')
        
        new_line_token = self.new_line_tokens

       
        if last_line.endswith(self.eos_string):
            last_line = last_line.replace(self.eos_string, '')
        sum = get_syllable_count_of_sentence(last_line)
        scores[self.tokenizer.bos_token_id] = torch.finfo(scores.dtype).min
        if sum < self.new_syllable_amount:
            prev_scores = scores.clone()
            scores = abs(scores)*torch.finfo(scores.dtype).min

            for i in range(len(best_tokens)):
                token = best_tokens[i]
                score = prev_scores[token]
                new_line = new_lines[i]
                syllable_count = get_syllable_count_of_sentence(new_line)
                if syllable_count <= self.new_syllable_amount and not does_string_contain_newline(new_line)and does_not_contain_special_characters(new_line):
                    if syllable_count == self.new_syllable_amount:
                        if not last_word_only_has_consontants(new_line):
                            scores[token] = score
                    else:
                        scores[token] = score
                
            
            for token in self.special_new_line_tokens:
                scores[token] = torch.finfo(scores.dtype).min
            scores[self.tokenizer.eos_token_id] = torch.finfo(scores.dtype).min
        elif sum == self.new_syllable_amount:
            best_scores, best_tokens = scores.topk(2)
            if best_scores[0].item() == float('-inf'):
                scores[self.tokenizer.eos_token_id] = torch.tensor(-1, device = scores.device)
            
            scores = abs(scores)*torch.finfo(scores.dtype).min

            activated = False

            for score, token in zip(best_scores, best_tokens):
                index = torch.where(best_tokens == token)[0].item()
                new_line = new_lines[index]
                syllable_count = get_syllable_count_of_sentence(new_line)
                if syllable_count <= self.new_syllable_amount and not does_string_contain_newline(new_line) and only_adds_regular_characters(last_line,new_line):
                    scores[token] = score
                    #print(last_line,candidate_text[len(self.original_prompt):])
                    activated = True 

            
            if not activated:
                scores[self.tokenizer.eos_token_id] = torch.tensor(-1, device = scores.device)


        else:
            scores = abs(scores)*torch.finfo(scores.dtype).min
            scores[self.tokenizer.eos_token_id] = torch.tensor(-1, device = scores.device)
            


        return scores



    def is_constrained_satisfied(self, generated_text):
        if self.new_syllable_amount is None:
            raise Exception('Syllable amount not set')
        if self.disable_constraint:
            return True
        sum = get_syllable_count_of_sentence(generated_text)
        if sum == self.new_syllable_amount:
            return True
        return False
    
    def is_beam_constraint_active(self):
        return True
    
    def is_stopping_criteria_active(self):
        return True
    
    def is_logits_processor_active(self):
        return True


