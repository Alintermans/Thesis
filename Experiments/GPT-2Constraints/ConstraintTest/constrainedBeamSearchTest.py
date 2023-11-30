from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, StoppingCriteriaList
from SyllableConstrainedBeamSearch import BeamSearchScorerFilterConstrained, count_syllables, tokenize_sentence
import transformers
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

sequence = "The following song is a parody on a Taylor swift song\n\n[Verse 1]\nOnce upon a time in the history of this\ncomprehensive society there has been one\n"
set_seed(4)

syllable_count = 50

forbidden_charachters = ['[', ']', '(', ')', '{', '}', '<', '>', '|', '\\', '/', '_', '-', '+', '=', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`', ';', ':', '"', "'", ',', '.', '?']
forbidden_tokens = [tokenizer.encode(c)[0] for c in forbidden_charachters]

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


model_inputs = tokenizer(sequence, return_tensors="pt") 
print(model_inputs)
output = model.generate(
    **model_inputs, 
    max_new_tokens=80, 
    num_return_sequences=1, 
    num_beams=50, 
    no_repeat_ngram_size=1, 
    early_stopping=True,
    suppress_tokens=forbidden_tokens,
    BeamSearcher=BeamSearchScorerFilterConstrained,
    target_syllables=syllable_count,
    tokenizer=tokenizer,
    stopping_criteria=stopping_criteria,
    
    )

print(output[0])
print(tokenizer.decode(output[0], skip_special_tokens=True))



