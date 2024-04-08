from Constraint import Constraint
from SongUtils import get_pos_tags_of_line, similarity_of_pos_tags_sequences
import torch
################################################ CONSTRAINT CLASS ################################################


class PosConstraintLBL(Constraint):
    def __init__(self, tokenizer, start_token=None):
        self.tokenizer = tokenizer
        self.start_token = start_token
        self.expected_pos_tags = None
        self.disable_constraint = False

        #hyperparameters
        self.top_k_tokens_to_consider = None
        self.good_beamscore_multiplier = None
        self.good_token_multiplier = None
        self.limit_of_pos_similarity_to_satisfy_constraint = None
        
    
    def get_hyperparameters_in_dict(self):
        return {
            self.get_name(): {
                'top_k_tokens_to_consider': self.top_k_tokens_to_consider,
                'good_beamscore_multiplier': self.good_beamscore_multiplier,
                'good_token_multiplier': self.good_token_multiplier,
                'limit_of_pos_similarity_to_satisfy_constraint': self.limit_of_pos_similarity_to_satisfy_constraint
            }
        }

    def get_name(self):
        return 'PosConstraintLBL'
    
    def disable(self):
        self.disable_constraint = True
    
    def enable(self):
        self.disable_constraint = False
    
    @staticmethod
    def hyperparameters_config(good_beamscore_multiplier=0.1, good_token_multiplier=0.6, limit_of_pos_similarity_to_satisfy_constraint=0.5, top_k_tokens_to_consider=100):
        return {
            'good_beamscore_multiplier': good_beamscore_multiplier,
            'good_token_multiplier': good_token_multiplier,
            'limit_of_pos_similarity_to_satisfy_constraint': limit_of_pos_similarity_to_satisfy_constraint,
            'top_k_tokens_to_consider': top_k_tokens_to_consider
        }
        
    
    def set_hyperparameters(self, config):
        self.good_beamscore_multiplier = config['good_beamscore_multiplier']
        self.good_token_multiplier = config['good_token_multiplier']
        self.limit_of_pos_similarity_to_satisfy_constraint = config['limit_of_pos_similarity_to_satisfy_constraint']
        self.top_k_tokens_to_consider = config['top_k_tokens_to_consider']


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
        similarity = similarity_of_pos_tags_sequences(pos_tags, self.expected_pos_tags[:min_length])
        #print('similarity: ', similarity, 'pos_tags: ', pos_tags, 'expected_pos_tags: ', self.expected_pos_tags)
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

            _, best_tokens = scores[i].topk(self.top_k_tokens_to_consider)
            for token in best_tokens:
                pos_tags_last_line = get_pos_tags_of_line(last_line)
                #The min length is only ysed for the expected pos tags, because the last line may be shorter than the expected pos tags but dtw can handle multiple lengths, but to esnure a valid score when the full line is being generted, the min length is used to cap the length of the expected pos tags, but when the full length is reached both are compared as is
                min_length = min(len(pos_tags_last_line), len(self.expected_pos_tags))
                similarity_of_last_line = similarity_of_pos_tags_sequences(pos_tags_last_line, self.expected_pos_tags[:min_length])
                candidate_text = last_line + self.tokenizer.decode(token, skip_special_tokens=True)
                pos_tags = get_pos_tags_of_line(candidate_text)
                min_length = min(len(pos_tags), len(self.expected_pos_tags))
                similarity_with_new_token = similarity_of_pos_tags_sequences(pos_tags, self.expected_pos_tags[:min_length])
                if pos_tags is not None:
                    #print('similarity_with_new_token: ', similarity_with_new_token, 'similarity_of_last_line: ', similarity_of_last_line)
                    if similarity_with_new_token >= similarity_of_last_line:
                        #print('similarity_with_new_token: ', similarity_with_new_token, 'similarity_of_last_line: ', similarity_of_last_line)
                        scores[i][token] = scores[i][token] - scores[i][token]*self.good_token_multiplier*similarity_with_new_token/(max(1-(similarity_with_new_token - similarity_of_last_line), 0.01))
                    
            

            
                    
        return scores
    

    def is_constrained_satisfied(self, generated_text):
        if self.disable_constraint:
            return True
        pos_tags_generated_text = get_pos_tags_of_line(generated_text)
        if pos_tags_generated_text is None:
            return False
        pos_similarity = similarity_of_pos_tags_sequences(pos_tags_generated_text, self.expected_pos_tags)
        if pos_similarity > self.limit_of_pos_similarity_to_satisfy_constraint:
            #print('pos_similarity: ', pos_similarity)
            return True
        return False
        
    
    def is_beam_constraint_active(self):
        return True
    
    def is_stopping_criteria_active(self):
        return False
    
    def is_logits_processor_active(self):
        return True


