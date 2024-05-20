from transformers import LogitsProcessor
import torch

class OptimizedConstraint(LogitsProcessor):
    def __init__(self, constraints, tokenizer,  top_k=100):
        self.constraints = constraints
        self.top_k = top_k
        self.prompt_length = 0
        self.tokenizer = tokenizer
    
    def set_prompt_length(self, prompt_length):
        self.prompt_length = prompt_length
        

    def __call__(self, input_ids, scores):
        for i in range(len(input_ids)):
            last_line = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)[self.prompt_length:]
            best_scores, best_tokens = scores[i].topk(self.top_k)
            new_lines = []
            scores[i] = abs(scores[i])*torch.finfo(scores.dtype).min
            for score, token in zip(best_scores, best_tokens):
                next_token_tensor = torch.tensor([token], device = scores[i].device)
                candidate_text = self.tokenizer.decode(torch.cat([input_ids[i], next_token_tensor], dim=0), skip_special_tokens=True)
                new_lines.append(candidate_text[self.prompt_length:])
                scores[i][token] = score
            
            for constraint in self.constraints:
                scores[i] = constraint.apply_optimzed_logit_processor(input_ids[i], scores[i], best_tokens, last_line, new_lines)

            

        return scores
    