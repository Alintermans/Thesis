from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, StoppingCriteriaList, LogitsProcessorList, NoBadWordsLogitsProcessor, BeamSearchScorer, LogitsProcessor
from SyllableConstrainedBeamSearch import BeamSearchScorerFilterConstrained, count_syllables, tokenize_sentence
import transformers
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

sequence = "The following song is a parody on a Taylor swift song\n\n[Verse 1]\nOnce upon a time in the history of this\ncomprehensive society there has been one\n"
set_seed(4)

forbidden_charachters = ['[', ']', '(', ')', '{', '}', '<', '>', '|', '\\', '/', '_', '-', '+', '=', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`', ';', ':', '"', "'", ',', '.', '?', '\n']
forbidden_tokens = [[tokenizer.encode(c)[0]] for c in forbidden_charachters]

def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
    for input in input_ids:
        sentence = tokenizer.decode(input, skip_special_tokens=True)
        words = tokenize_sentence(sentence)
        sum = 0
        for word in words:
            sum += count_syllables(word) 
        if sum >syllable_count:
            print('sum: ',sum, 'sentence: ', sentence)
            return True

    return False

class ExactSyllableCountLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, syllables_amount=8, best_k_tokens=2):
        super().__init__()
        self.tokenizer = tokenizer
        self.syllables_amount = syllables_amount
        self.best_k_tokens = best_k_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        for i in range(len(input_ids)):
            score = scores[i]
            #get the indeces of the best k tokens
            best_k_tokens = torch.topk(score, self.best_k_tokens)[1]
            for token in best_k_tokens:
                #print("token: ", token)    
                sentence = self.tokenizer.decode(torch.cat((input_ids[i], torch.tensor([token]))) , skip_special_tokens=True)
                print("sentence: ", sentence)   
                words = tokenize_sentence(sentence)
                sum = 0
                
                for word in words:
                    sum += count_syllables(word) 
                print("syllable count: ", sum)
                if sum == self.syllables_amount:
                    print("found exact syllable count")
                    scores[i][token] *= 0
                elif sum > self.syllables_amount:
                    scores[i][token] = scores[i][token] + scores[i][token]*0.5
                    
                

        return scores


syllable_count = 80
num_beams = 50

stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
logit_processor = LogitsProcessorList([NoBadWordsLogitsProcessor(forbidden_tokens, eos_token_id=tokenizer.eos_token_id), ExactSyllableCountLogitsProcessor(tokenizer, syllables_amount=syllable_count, best_k_tokens=10)])



model_inputs = tokenizer(sequence, return_tensors="pt")
#print(model_inputs)

beam_scorer = BeamSearchScorer(
    batch_size= model_inputs['input_ids'].shape[0],
    max_length=100,
    num_beams=num_beams,
    device=model.device,
    # tokenizer=tokenizer,
    # target_syllables=syllable_count,
)

generated = model.beam_search(
    torch.cat([model_inputs['input_ids']] * num_beams),
    beam_scorer,
    logits_processor=logit_processor,
    stopping_criteria=stopping_criteria
    )

#print(generated)

# print(output[0])
sentence = tokenizer.decode(generated[0], skip_special_tokens=True)
print(sentence)
words = tokenize_sentence(sentence)
sum = 0
for word in words:
    sum += count_syllables(word)
print(sum)




