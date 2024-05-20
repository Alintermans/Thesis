from Constraint import Constraint
from SongUtils import get_syllable_count_of_sentence
import torch
################################################ CONSTRAINT CLASS ################################################


class SyllableConstraintFull(Constraint):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.paragraphs = None
        self.num_beams = None
        self.next_line_token = tokenizer.encode('\n')[0]
        self.syllble_amount_per_line = None
        self.nb_of_lines = None
        self.prompt_length = None
    
    def set_paragraphs(self, paragraphs):
        self.paragraphs = paragraphs
        self.syllble_amount_per_line = []
        for (_,paragraph) in paragraphs:
            for line in paragraph:
                self.syllble_amount_per_line.append(get_syllable_count_of_sentence(line))
        self.nb_of_lines = len(self.syllble_amount_per_line)
    
    def set_num_beams(self, num_beams):
        self.num_beams = num_beams
    
    def set_prompt_length(self, prompt_length):
        self.prompt_length = prompt_length
        
    
    def apply_beam_constraint(self, next_token, next_score, input_ids, cur_len, length_penalty):
        if self.paragraphs is None:
            raise Exception('paragraph not set')
        
        if self.num_beams is None:
            raise Exception('num_beams not set')
        
        if self.prompt_length is None:
            raise Exception('prompt_length not set')
        
        
        already_generated_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)[self.prompt_length:]
        already_generated_lines = already_generated_text.split('\n')
        current_line = already_generated_lines[-1]
        current_line_syllable_count = get_syllable_count_of_sentence(current_line)
        len_already_generated_lines = len(already_generated_lines)

        if (len_already_generated_lines > self.nb_of_lines):
            return next_score
        

        if current_line_syllable_count > self.syllble_amount_per_line[len_already_generated_lines-1]:
            next_score = next_score + next_score*0.1
            
        # elif current_line_syllable_count == self.syllble_amount_per_line[len_already_generated_lines-1]:
        #      next_score = next_score - next_score*0.1 
            
        return next_score


    def stopping_criteria(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        if self.paragraphs is None:
            raise Exception('paragraph not set')
        
        if self.num_beams is None:
            raise Exception('num_beams not set')
        
        if self.prompt_length is None:
            raise Exception('prompt_length not set')

        for input in input_ids:
            already_generated_text = self.tokenizer.decode(input, skip_special_tokens=True)[self.prompt_length:]
            already_generated_lines = already_generated_text.split('\n')

            if (len(already_generated_lines) > self.nb_of_lines):
                print('returned True')
                return True
            
        return False
    
    def logits_processor(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.paragraphs is None:
            raise Exception('paragraph not set')
        
        if self.num_beams is None:
            raise Exception('num_beams not set')
        
        if self.prompt_length is None:
            raise Exception('prompt_length not set')
        

        for i in range(self.num_beams):
            already_generated_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)[self.prompt_length:]
            already_generated_lines = already_generated_text.split('\n')
            current_line = already_generated_lines[-1]
            current_line_syllable_count = get_syllable_count_of_sentence(current_line)
            len_already_generated_lines = len(already_generated_lines)

            if (len_already_generated_lines > self.nb_of_lines):
                scores[i] = abs(scores[i]) * float('-inf')
                scores[i][self.next_line_token] = 0
                continue

            if current_line_syllable_count >= self.syllble_amount_per_line[len_already_generated_lines-1]:
                #print("current_line_syllable_count: ", current_line_syllable_count , "| expected syllble amount: ", self.syllble_amount_per_line[len(already_generated_lines)-1], " sentence: ", current_line)
                scores[i] = abs(scores[i]) * float('-inf')
                scores[i][self.next_line_token] = 0
        return scores
    
    def is_beam_constraint_active(self):
        return True
    
    def is_stopping_criteria_active(self):
        return True
    
    def is_logits_processor_active(self):
        return True 


