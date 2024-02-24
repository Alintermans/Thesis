from Constraints.SyllableConstraint.SyllableConstraintLBL import SyllableConstraintLBL
from Constraint import ConstraintList
from BeamSearchScorerConstrained import BeamSearchScorerConstrained
from LanguageModels.GPT2 import GPT2
from LanguageModels.Gemma2BIt import Gemma2BIt
from SongUtils import read_song, divide_song_into_paragraphs, get_syllable_count_of_sentence, write_song

from transformers import (
                set_seed, 
                StoppingCriteriaList, 
                MaxLengthCriteria, 
                LogitsProcessorList, 
                NoBadWordsLogitsProcessor, 
                NoRepeatNGramLogitsProcessor,
                TopKLogitsWarper,
                TopPLogitsWarper,
                TemperatureLogitsWarper
                )
import torch


########## LM ##########
#lm = GPT2()
lm = Gemma2BIt()
tokenizer = lm.get_tokenizer()
model = lm.get_model()

######### Settings ##########
set_seed(42)
num_beams = 3


######### Constraints ##########
syllable_constraint = SyllableConstraintLBL(tokenizer)

forbidden_charachters = ['[', ']', '(', ')', '{', '}', '<', '>', '|', '\\', '/', '_', '——', ' — ', '..' '+', '=', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`', ';', ':', '"', "'", ',', '.', '?', '\n', '\n\n', '  ', '...']
forbidden_tokens = [[tokenizer.encode(c)[0]] for c in forbidden_charachters]
forbidden_tokens_logit_processor = NoBadWordsLogitsProcessor(forbidden_tokens, eos_token_id=tokenizer.eos_token_id)

no_ngram_logits_processor = NoRepeatNGramLogitsProcessor(2)

## Combine Constraints
constraints = ConstraintList([syllable_constraint])

stopping_criteria_list = constraints.get_stopping_criteria_list() + []
stopping_criteria = StoppingCriteriaList(stopping_criteria_list)

logits_processor_list = constraints.get_logits_processor_list() + [no_ngram_logits_processor, forbidden_tokens_logit_processor]
logits_processor = LogitsProcessorList(logits_processor_list)


######## Generate Line ########
def generate_line(prompt, **kwargs):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    ## Constraints
    syllable_constraint.set_new_syllable_amount(kwargs['new_syllable_amount'])


    ## Beam Search
    beam_scorer = BeamSearchScorerConstrained(
        batch_size= input_ids.shape[0],
        max_length=1000,
        num_beams=num_beams,
        device=model.device,
        constraints = constraints,
    )

    ## Generate
    if (kwargs.get('do_sample') is not None and kwargs.get('do_sample') == True):
        if kwargs.get('top_k') is None:
            raise Exception('top_k not set')
        if kwargs.get('top_p') is None:
            raise Exception('top_p not set')

        top_k = kwargs['top_k']
        top_p = kwargs['top_p']
        temperature = kwargs['temperature']
        logits_warper = LogitsProcessorList(
             [
                TopPLogitsWarper(top_p),
                TopKLogitsWarper(top_k),
                TemperatureLogitsWarper(temperature),
             ]
        )

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        outputs = model.sample(
            torch.cat([input_ids] * num_beams),
            beam_scorer=beam_scorer,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            pad_token_id=pad_token_id,
        )
    else:
        outputs = model.beam_search(
            torch.cat([input_ids] * num_beams),
            beam_scorer=beam_scorer,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]



########## Generate Parodie  ##########



def generate_parodie(song_file_path, system_prompt, context, **kwargs):
    song = read_song(song_file_path) #expects a json file, where the lyrics is stored in the key 'lyrics'
    song_in_paragraphs = divide_song_into_paragraphs(song)

    prompt = system_prompt + context + "ORIGINAL SONG : \n\n" + song + "\n\nAlready generated PARODIE: \n\n"
    parodie = ""
    state = "Finished Correctly"

    if (kwargs.get('do_sample') is not None and kwargs.get('do_sample') == True):
        if kwargs.get('top_k') is None:
            kwargs['top_k'] = 50
        if kwargs.get('top_p') is None:
            kwargs['top_p'] = 0.95
        if kwargs.get('temperature') is None:
            kwargs['temperature'] = 0.7

    try: 
        for paragraph in song_in_paragraphs:
            parodie += paragraph[0] + "\n"
            for line in paragraph[1]:
                syllable_amount = get_syllable_count_of_sentence(line)
                new_line = generate_line(prompt + parodie, new_syllable_amount=syllable_amount, **kwargs)
                parodie += new_line + "\n"
                print(line, " | ",new_line)
            parodie += "\n"
    except Exception as e:
        raise Exception(e)
        print("Error has occured ", e)
        state = "Error has occured " + str(e) + "\n" + "Not finished correctly"
        parodie += "\n\n" + "[ERROR]: Not finished correctly" + "\n\n"

    decoding_method = "Beam Search"
    if (kwargs.get('do_sample') is not None or kwargs.get('do_sample') == True):
        decoding_method = "Sampling Beam Search" + " | top_k: " + str(kwargs['top_k']) + " | top_p: " + str(kwargs['top_p'] + " | temperature: " + str(kwargs['temperature']))



    print("Parodie: ", parodie)
    write_song('Experiments/ConstrainedParodieGenerator/GeneratedParodies/', 
                original_song_file_path = song_file_path, 
                parodie = parodie, context = context, 
                system_prompt = system_prompt, 
                prompt = prompt, 
                constraints_used = "SyllableConstraintLBL",
                language_model_name = lm.get_name(),
                state = state,
                way_of_generation = "Line by Line",
                decoding_method = decoding_method)



if(__name__ == '__main__'):
    song_file_path = 'Songs/json/Taylor_Swift-Is_It_Over_Now_(Small_Version).json'
    #song_file_path = 'Songs/json/Coldplay-Viva_La_Vida.json'

    system_prompt = "I'm a parodie genrator that will write beatifull parodies and make sure that the syllable count and the rhyming of my parodies are the same as the original song\n"
    context = "The following parodie will be about that pineaple shouldn't be on pizza\n"

    generate_parodie(song_file_path, system_prompt, context, do_sample=True, top_k=50, top_p=0.7, temperature=0.7)
    #print(generate_line("Hello\n", new_syllable_amount=7, do_sample=True, top_k=50, top_p=0.9, temperature=0.7))

