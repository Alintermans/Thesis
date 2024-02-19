from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, StoppingCriteriaList, LogitsProcessorList, NoBadWordsLogitsProcessor
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
        if sum >=syllable_count:
            return True

    return False

stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
logit_processor = LogitsProcessorList([NoBadWordsLogitsProcessor(forbidden_tokens, eos_token_id=tokenizer.eos_token_id)])

syllable_count = 53
num_beams = 5

model_inputs = tokenizer(sequence, return_tensors="pt")
#print(model_inputs)

beam_scorer = BeamSearchScorerFilterConstrained(
    batch_size= model_inputs['input_ids'].shape[0],
    max_length=100,
    num_beams=num_beams,
    device=model.device,
    tokenizer=tokenizer,
    target_syllables=syllable_count,
)

generated = model.beam_search(
    torch.cat([model_inputs['input_ids']] * num_beams),
    beam_scorer,
    logits_processor=logit_processor,
    stopping_criteria=stopping_criteria
    )

print(generated)

# print(output[0])
print(tokenizer.decode(generated[0], skip_special_tokens=True))




