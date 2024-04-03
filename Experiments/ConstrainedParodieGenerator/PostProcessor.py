from transformers import LogitsProcessor
from torch import nn
import torch 

class PostProcessor:
    def __init__(self):
        self.logits_processor = PostProcessorLogitsProcessor(self)

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
        # for i in range(len(input_ids)):
        #     input = input_ids[i]
        #     result = [x for x in input if x != float("inf") or x != float("-inf") or x != float("nan") or x != 0.0]
        #     print("nb_of_possible_tokens: ", len(result))

        scores = nn.functional.log_softmax(
                scores, dim=-1
            )


        return scores 








class PostProcessorLogitsProcessor(LogitsProcessor):  
    def __init__(self, post_processor):
        self.post_processor = post_processor
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.post_processor.logits_processor_call(input_ids, scores) 

    