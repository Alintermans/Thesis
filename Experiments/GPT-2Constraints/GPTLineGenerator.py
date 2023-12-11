from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, StoppingCriteriaList, MaxLengthCriteria, LogitsProcessorList, NoBadWordsLogitsProcessor, NoRepeatNGramLogitsProcessor
from BeamSearchScorerConstrained import BeamSearchScorerConstrained
from Constraint import ConstraintList, Constraint
from SyllableConstraint import SyllableConstraint
from SongUtils import get_syllable_count_of_sentence
import torch


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                torch_dtype=torch.bfloat16, 
                                                low_cpu_mem_usage=True, 
                                                token= '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',
                                                # max_memory={"cpu": "11GIB"},
                                                # offload_state_dict=True,
                                                # offload_folder = '/Volumes/Samsung\ SSD/offload'
                                                load_in_8bits=True
                                                )    

set_seed(42)
num_beams = 2

forbidden_charachters = ['[', ']', '(', ')', '{', '}', '<', '>', '|', '\\', '/', '_', '——', ' — ', '..' '+', '=', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`', ';', ':', '"', "'", ',', '.', '?', '\n', '...']
forbidden_tokens = [[tokenizer.encode(c)[0]] for c in forbidden_charachters]

def generate_parodie_line(prompt, line):
    model_inputs = tokenizer(prompt, return_tensors="pt")

    syllable_amount_prompt = get_syllable_count_of_sentence(prompt)
    syllable_amount_line = get_syllable_count_of_sentence(line)
    print('Syllable count prompt: ', syllable_amount_prompt)
    print("Line: ", line, '| Syllable count: ', syllable_amount_line)
    syllable_amount = syllable_amount_prompt + syllable_amount_line
    constraints = ConstraintList([SyllableConstraint(syllable_amount, tokenizer)])
    stopping_criteria_list = constraints.get_stopping_criteria_list()
    stopping_criteria = StoppingCriteriaList(stopping_criteria_list)
    logits_processor_list = constraints.get_logits_processor_list()
    logits_processor_list.append(NoRepeatNGramLogitsProcessor(2))
    logits_processor_list.append(NoBadWordsLogitsProcessor(forbidden_tokens, eos_token_id=tokenizer.eos_token_id))
    logits_processor = LogitsProcessorList(logits_processor_list)

    beam_scorer = BeamSearchScorerConstrained(
        batch_size= model_inputs['input_ids'].shape[0],
        max_length=1000,
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

    sentence = tokenizer.decode(generated[0], skip_special_tokens=True)[len(prompt):]
    print('syllable count: ', get_syllable_count_of_sentence(sentence))
    return sentence
    