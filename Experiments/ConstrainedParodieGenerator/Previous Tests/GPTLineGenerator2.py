from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, StoppingCriteriaList, MaxLengthCriteria, LogitsProcessorList, NoBadWordsLogitsProcessor, NoRepeatNGramLogitsProcessor,  TopKLogitsWarper, TemperatureLogitsWarper, TopPLogitsWarper
from BeamSearchScorerConstrained import BeamSearchScorerConstrained
from Constraint import ConstraintList, Constraint
from SyllableConstraint2 import SyllableConstraint
from SongUtils import get_syllable_count_of_sentence
import torch



def generate_parodie(prompt, paragraphs, tokenizer = AutoTokenizer.from_pretrained("gpt2-medium"),  model = AutoModelForCausalLM.from_pretrained("gpt2-medium") ):

    set_seed(42)
    num_beams = 2

    forbidden_charachters = ['[', ']', '(', ')', '{', '}', '<', '>', '|', '\\', '/', '_', '——', ' — ', '..' '+', '=', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`', ';', ':', '"', "'", ',', '.', '?', '\n', '...']
    forbidden_tokens = [[tokenizer.encode(c)[0]] for c in forbidden_charachters]
    model_inputs = tokenizer(prompt, return_tensors="pt")
    next_line_token = tokenizer.encode('\n')[0]
    constraints = ConstraintList([SyllableConstraint(paragraphs, tokenizer, next_line_token=next_line_token,  num_beams=num_beams, prompt=prompt)])
    stopping_criteria_list = constraints.get_stopping_criteria_list()
    stopping_criteria = StoppingCriteriaList(stopping_criteria_list)
    logits_processor_list = []
    logits_processor_list.append(NoRepeatNGramLogitsProcessor(2))
    logits_processor_list.append(NoBadWordsLogitsProcessor(forbidden_tokens, eos_token_id=tokenizer.eos_token_id))
    logits_processor_list += constraints.get_logits_processor_list()
    logits_processor = LogitsProcessorList(logits_processor_list)

    # logits_warper = LogitsProcessorList(
    #     [
    #         TopPLogitsWarper(0.9),
    #         TemperatureLogitsWarper(0.5),
    #     ]
    # )

    beam_scorer = BeamSearchScorerConstrained(
        batch_size= 1,
        max_length=100000,
        num_beams=num_beams,
        device=model.device,
        constraints = constraints,
        do_early_stopping=False,
        length_penalty=10.0,
    )

    generated = model.beam_search(
        torch.cat([model_inputs['input_ids']] * num_beams, dim=0),
        beam_scorer,
        stopping_criteria=stopping_criteria,
        logits_processor = logits_processor,
        
        )
    sentence = tokenizer.decode(generated[0], skip_special_tokens=True)[len(prompt):]
    return sentence
    