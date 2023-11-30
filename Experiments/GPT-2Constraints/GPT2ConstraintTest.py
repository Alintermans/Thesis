from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, StoppingCriteriaList, MaxLengthCriteria, LogitsProcessorList, NoBadWordsLogitsProcessor
from BeamSearchScorerConstrained import BeamSearchScorerConstrained
from Constraint import ConstraintList, Constraint
from SyllableConstraint import SyllableConstraint,get_syllable_count_of_sentence
from RhyminConstraint import EndRhymeWith
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

forbidden_charachters = ['[', ']', '(', ')', '{', '}', '<', '>', '|', '\\', '/', '_', '-', '+', '=', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`', ';', ':', '"', "'", ',', '.', '?', '\n']
forbidden_tokens = [[tokenizer.encode(c)[0]] for c in forbidden_charachters]

sequence = "The following song is a parody on a Taylor swift song\n\n[Verse 1]\nOnce upon a time in the history of this\ncomprehensive society there has been one\n"
set_seed(5)

num_beams = 20

syllable_amount = 52

model_inputs = tokenizer(sequence, return_tensors="pt")
constraints = ConstraintList([SyllableConstraint(syllable_amount, tokenizer), EndRhymeWith("this", tokenizer, syllable_amount)])
stopping_criteria_list = constraints.get_stopping_criteria_list()
stopping_criteria = StoppingCriteriaList(stopping_criteria_list)
logits_processor_list = constraints.get_logits_processor_list()
logits_processor_list.append(NoBadWordsLogitsProcessor(forbidden_tokens, eos_token_id=tokenizer.eos_token_id))
logits_processor = LogitsProcessorList(logits_processor_list)




beam_scorer = BeamSearchScorerConstrained(
    batch_size= model_inputs['input_ids'].shape[0],
    max_length=100,
    num_beams=num_beams,
    device=model.device,
    constraints = constraints,
)

generated = model.beam_search(
    torch.cat([model_inputs['input_ids']] * num_beams),
    beam_scorer,
    stopping_criteria=stopping_criteria,
    logits_processor = logits_processor,
    )
sentence = tokenizer.decode(generated[0], skip_special_tokens=True)
print("generated: ", sentence)
print("Syllable count: ", get_syllable_count_of_sentence(sentence))
