from transformers import LogitsProcessor
from torch import nn
import torch 

class PostProcessor:
    def __init__(self, tokenizer):
        self.logits_processor = PostProcessorLogitsProcessor(self)
        self.tokenizer = tokenizer

    def apply_beam_post_processing(self, nexttoken, next_score, input_ids, cur_len, length_penalty):
        if next_score.item() != next_score.item():
            raise Exception('next_score is nan')
        

        if next_score.item() == float('inf'):
            raise Exception('next_score is inf')
        
        if next_score.item() == float('-inf'):
            raise Exception('next_score is -inf')


        return next_score 
    
    def get_logits_processor(self):
        return self.logits_processor
    
    def logits_processor_call(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i in range(len(input_ids)):
            input = scores[i]
            result = [x for x in input if x != float("inf") and x != float("-inf") and x != float("nan") and x != 0.0 and x != torch.finfo(scores.dtype).min]
            
            if len(result) == 0:
                scores[i][self.tokenizer.eos_token_id] = torch.finfo(scores.dtype).max

        scores = nn.functional.log_softmax(
                scores, dim=-1
            )


        


        return scores 








class PostProcessorLogitsProcessor(LogitsProcessor):  
    def __init__(self, post_processor):
        self.post_processor = post_processor
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.post_processor.logits_processor_call(input_ids, scores) 

    