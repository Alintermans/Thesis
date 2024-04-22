from transformers import LogitsProcessor
import torch

class Backtracking: 
    def __init__(self, original_input_length, constraints, backtracking_logits_processor, eos_token_id=None, use_backtracking=True):
        self.original_input_length = original_input_length
        self.constraints = constraints
        self.does_loop_continue = True
        self.latest_input_ids = None
        self.best_result = None
        self.highest_nb_constraints_satisfied = 0
        self.best_score = None
        self.retries = [5,5,5]
        self.retry_points = [0.75, 0.5, 0.25]
        self.backtracking_logits_processor = backtracking_logits_processor
        self.eos_token_id = eos_token_id
        self.use_backtracking = use_backtracking

    
    def continue_loop(self):
        return self.does_loop_continue

    def get_result(self):
        return self.best_result
    

    

    def calculate_nb_tokens_to_remove(self, new_line_token_length, eos_token_present):
        if eos_token_present:
            new_line_token_length -= 1

        index = None
        for i in range(len(self.retries)):
            if self.retries[i] > 0:
                index = i
                break
        if index is None:
            return 0
        print("retrying with index: ", index)
        self.retries[index] -= 1
        return int(new_line_token_length * (1-self.retry_points[index])) + 1 if eos_token_present else int(new_line_token_length * (1-self.retry_points[index]))



    def validate_result(self, decoded_results, output_ids, scores):
        index = 0
        best_nb_constraints_satisfied = 0
        best_index_not_satisfied = 0
        for i in range(len(decoded_results)):
            decoded_result = decoded_results[i]
            constraints_satisfied, nb_satisfied_constraints = self.constraints.are_constraints_satisfied(decoded_result)
            if constraints_satisfied:
                index = i
                break
            if nb_satisfied_constraints > best_nb_constraints_satisfied:
                best_nb_constraints_satisfied = nb_satisfied_constraints
                best_index_not_satisfied = i
        
        if best_nb_constraints_satisfied > 0:
            index = best_index_not_satisfied
        
        decoded_result = decoded_results[index]
        constraints_satisfied, nb_satisfied_constraints = self.constraints.are_constraints_satisfied(decoded_result)
        score = scores[index].item()
        
        if constraints_satisfied or not self.use_backtracking:
            self.best_result = decoded_result
            self.does_loop_continue = False
            return True
        
        if (nb_satisfied_constraints > self.highest_nb_constraints_satisfied and self.best_score is None) or (nb_satisfied_constraints > self.highest_nb_constraints_satisfied and score > self.best_score):
            self.highest_nb_constraints_satisfied = nb_satisfied_constraints
            self.best_result = decoded_result
        print(decoded_result)
        new_line_token_length = output_ids[index].shape[-1] - self.original_input_length
        eos_token_present = False
        if output_ids[index][-1].item == self.eos_token_id:
            eos_token_present = True
        print("new_line_token_length: ", new_line_token_length)
        
        nb_tokens_to_remove = self.calculate_nb_tokens_to_remove(new_line_token_length, eos_token_present)
        print("nb_tokens_to_remove: ", nb_tokens_to_remove)
        if nb_tokens_to_remove == 0:
            self.does_loop_continue = False
            return False
        
        self.latest_input_ids = output_ids[index:index+1, :-nb_tokens_to_remove].clone()
        self.backtracking_logits_processor.add_output_sequence_to_ignore(output_ids[index])


    def get_updated_input_ids(self):
        return self.latest_input_ids

    
class BacktrackingLogitsProcessor(LogitsProcessor):
    def __init__(self, original_input_length):
        self.sequences_to_ignore = []
        self.original_input_ids_length = original_input_length
    
    def add_output_sequence_to_ignore(self, sequence):
        self.sequences_to_ignore.append(sequence[self.original_input_ids_length:])
        
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if len(self.sequences_to_ignore) == 0:
            return scores
        
        for i in range(len(input_ids)):
            input = input_ids[i][self.original_input_ids_length:]
            for sequence in self.sequences_to_ignore:
                #print("input: ", input, "sequence: ", sequence)
                if torch.equal(input, sequence[:len(input)]) and len(input) < len(sequence):
                    scores[i][sequence[len(input)].item()] = torch.finfo(scores.dtype).min
        
        return scores