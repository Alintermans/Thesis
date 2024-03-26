from Constraint import Constraint
from SongUtils import get_pos_tags_of_line, similarity_of_pos_tags_sequences
import torch
################################################ CONSTRAINT CLASS ################################################


class PosConstraintLBL(Constraint):
    def __init__(self, tokenizer, start_token=None, top_k_words_to_consider=100):
        self.tokenizer = tokenizer
        self.start_token = start_token
        self.expected_pos_tags = None
        self.top_k_words_to_consider = top_k_words_to_consider
        #hyperparameters
        self.good_beamscore_multiplier = 0.1
        self.pos_similarity_limit_to_boost = 0.5
        self.good_token_multiplier = 0.6
        self.margin_of_similarity_with_new_token = 0.1

        self.disable_constraint = False
    
    def disable(self):
        self.disable_constraint = True
    
    def enable(self):
        self.disable_constraint = False
    
    def set_hyperparameters(self, good_beamscore_multiplier=0.1, pos_similarity_limit_to_boost=0.5, good_token_multiplier=0.6, margin_of_similarity_with_new_token=0.1):
        self.good_beamscore_multiplier = good_beamscore_multiplier
        self.pos_similarity_limit_to_boost = pos_similarity_limit_to_boost
        self.good_token_multiplier = good_token_multiplier
        self.margin_of_similarity_with_new_token = margin_of_similarity_with_new_token
    

    def set_expected_pos_tags(self, expected_pos_tags):
        self.expected_pos_tags = expected_pos_tags


    
    def apply_beam_constraint(self, next_token, next_score, input_ids, cur_len, length_penalty):
        if self.disable_constraint:
            return next_score
        
        previous_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        current_token = self.tokenizer.decode(next_token, skip_special_tokens=True)
        candidate_text = previous_text + current_token
        last_line = candidate_text.split('\n')[-1]

        pos_tags = get_pos_tags_of_line(last_line)

        if pos_tags is None:
            return next_score
        
        if self.expected_pos_tags is None:
            raise Exception('Expected pos tags not set')

        min_length = min(len(pos_tags), len(self.expected_pos_tags))
        similarity = similarity_of_pos_tags_sequences(pos_tags[:min_length], self.expected_pos_tags[:min_length])
        #print('similarity: ', similarity, 'pos_tags: ', pos_tags, 'expected_pos_tags: ', self.expected_pos_tags)
        if similarity > self.pos_similarity_limit_to_boost:
            #return next_score - next_score*(similarity)*self.good_beamscore_multiplier * ( cur_len ** length_penalty)
            return next_score - next_score*(similarity)*self.good_beamscore_multiplier 

        
        return next_score

        
    
        

    
    def stopping_criteria(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        return False
    
    def logits_processor(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.disable_constraint:
            return scores


        for i in range(len(input_ids)):
            input = input_ids[i]
            text = self.tokenizer.decode(input, skip_special_tokens=True)
            last_line = text.split('\n')[-1]

            _, best_tokens = scores[i].topk(self.top_k_words_to_consider)
            for token in best_tokens:
                pos_tags_last_line = get_pos_tags_of_line(last_line)
                min_length = min(len(pos_tags_last_line), len(self.expected_pos_tags))
                similarity_of_last_line = similarity_of_pos_tags_sequences(pos_tags_last_line[:min_length], self.expected_pos_tags[:min_length])
                candidate_text = last_line + self.tokenizer.decode(token, skip_special_tokens=True)
                pos_tags = get_pos_tags_of_line(candidate_text)
                min_length = min(len(pos_tags), len(self.expected_pos_tags))
                similarity_with_new_token = similarity_of_pos_tags_sequences(pos_tags[:min_length], self.expected_pos_tags[:min_length])
                if pos_tags is not None:
                    
                    if similarity_with_new_token > min(similarity_of_last_line + self.margin_of_similarity_with_new_token, 0.99) or similarity_with_new_token == 1.0 :
                        #print('similarity_with_new_token: ', similarity_with_new_token, 'similarity_of_last_line: ', similarity_of_last_line)
                        scores[i][token] = scores[i][token] - scores[i][token]*self.good_token_multiplier*(similarity_with_new_token - similarity_of_last_line)
                    
            

            
                    
        return scores
    
    def is_beam_constraint_active(self):
        return True
    
    def is_stopping_criteria_active(self):
        return False
    
    def is_logits_processor_active(self):
        return True


