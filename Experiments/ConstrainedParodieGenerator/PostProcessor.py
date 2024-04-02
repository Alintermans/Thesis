from transformers import LogitsProcessor
import torch 
class PostProcessor:
    def __init__(self):
        self.logits_processor = PostProcessorLogitsProcessor(self)

    def apply_beam_post_processing(self, nexttoken, next_score, input_ids, cur_len, length_penalty):

        return next_score 
    
    def get_logits_processor(self):
        return self.logits_processor





class PostProcessorLogitsProcessor(LogitsProcessor):  
    def __init__(self, post_processor):
        self.post_processor = post_processor
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores

    