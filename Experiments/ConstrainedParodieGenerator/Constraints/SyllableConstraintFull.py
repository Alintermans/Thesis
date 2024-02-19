from Constraint import Constraint
from SongUtils import tokenize_sentence, count_syllables, get_syllable_count_of_sentence
import torch
################################################ CONSTRAINT CLASS ################################################


class SyllableConstraint(Constraint):
    def __init__(self, paragraphs, tokenizer,next_line_token = 198, num_beams=1, prompt = ""):
        self.tokenizer = tokenizer
        self.paragraphs = paragraphs
        self.num_beams = num_beams
        self.next_line_token = next_line_token
        self.syllble_amount_per_line = []
        for (_,paragraph) in paragraphs:
            for line in paragraph:
                self.syllble_amount_per_line.append(get_syllable_count_of_sentence(line))
        self.nb_of_lines = len(self.syllble_amount_per_line)
        self.len_prompt = len(prompt)
    
    def apply_beam_constraint(self, next_token, next_score, input_ids, cur_len, length_penalty):
        
        already_generated_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)[self.len_prompt:]
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
        for input in input_ids:
            already_generated_text = self.tokenizer.decode(input, skip_special_tokens=True)[self.len_prompt:]
            already_generated_lines = already_generated_text.split('\n')

            if (len(already_generated_lines) > self.nb_of_lines):
                print('returned True')
                return True
            
        return False
    
    def logits_processor(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i in range(self.num_beams):
            already_generated_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)[self.len_prompt:]
            already_generated_lines = already_generated_text.split('\n')
            current_line = already_generated_lines[-1]
            current_line_syllable_count = get_syllable_count_of_sentence(current_line)
            len_already_generated_lines = len(already_generated_lines)

            if (len_already_generated_lines > self.nb_of_lines):
                scores[i] = abs(scores[i]) * float('-inf')
                scores[i][self.next_line_token] = 0
                continue

            if current_line_syllable_count >= self.syllble_amount_per_line[len_already_generated_lines-1]:
                print("current_line_syllable_count: ", current_line_syllable_count , "| expected syllble amount: ", self.syllble_amount_per_line[len(already_generated_lines)-1], " sentence: ", current_line)
                scores[i] = abs(scores[i]) * float('-inf')
                scores[i][self.next_line_token] = 0
        return scores
    
    def is_beam_constraint_active(self):
        return True
    
    def is_stopping_criteria_active(self):
        return True
    
    def is_logits_processor_active(self):
        return True 


