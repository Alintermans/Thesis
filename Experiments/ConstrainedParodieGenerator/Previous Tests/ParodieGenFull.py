from Constraints.SyllableConstraint.SyllableConstraintFull import SyllableConstraintFull
from Constraint import ConstraintList
from BeamSearchScorerConstrained import BeamSearchScorerConstrained
from LanguageModels.GPT2 import GPT2
from SongUtils import read_song, divide_song_into_paragraphs, get_syllable_count_of_sentence, write_song

from transformers import set_seed, StoppingCriteriaList, MaxLengthCriteria, LogitsProcessorList, NoBadWordsLogitsProcessor, NoRepeatNGramLogitsProcessor
import torch


########## LM ##########
lm = GPT2()
tokenizer = lm.get_tokenizer()
model = lm.get_model()

######### Settings ##########
set_seed(42)
num_beams = 3

######### Constraints ##########
syllable_constraint = SyllableConstraintFull(tokenizer)

forbidden_charachters = ['[', ']', '(', ')', '{', '}', '<', '>', '|', '\\', '/', '_', '——', ' — ', '..' '+', '=', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`', ';', ':', '"', "'", ',', '.', '?', '\n', '...']
forbidden_tokens = [[tokenizer.encode(c)[0]] for c in forbidden_charachters]
forbidden_tokens_logit_processor = NoBadWordsLogitsProcessor(forbidden_tokens, eos_token_id=tokenizer.eos_token_id)

no_ngram_logits_processor = NoRepeatNGramLogitsProcessor(2)

## Combine Constraints
constraints = ConstraintList([syllable_constraint])

stopping_criteria_list = constraints.get_stopping_criteria_list() + []
stopping_criteria = StoppingCriteriaList(stopping_criteria_list)

logits_processor_list = [no_ngram_logits_processor, forbidden_tokens_logit_processor] + constraints.get_logits_processor_list() 
logits_processor = LogitsProcessorList(logits_processor_list)


######## Generate Line ########
def generate_lines(prompt, **kwargs):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    ## Constraints
    syllable_constraint.set_num_beams(num_beams)
    syllable_constraint.set_paragraphs(kwargs['song_in_paragraphs'])
    syllable_constraint.set_prompt_length(len(prompt))

    ## Beam Search
    beam_scorer = BeamSearchScorerConstrained(
        batch_size= input_ids.shape[0],
        max_length=1000,
        num_beams=num_beams,
        device=model.device,
        constraints = constraints,
        length_penalty=10.0,
    )

    ## Generate

    outputs = model.beam_search(
        torch.cat([input_ids] * num_beams),
        beam_scorer=beam_scorer,
        stopping_criteria=stopping_criteria,
        logits_processor=logits_processor,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]



########## Generate Parodie  ##########

#print(generate_line("Hello\n", new_syllable_amount=7)

def generate_parodie(song_file_path, system_prompt, context):
    song = read_song(song_file_path) #expects a json file, where the lyrics is stored in the key 'lyrics'
    song_in_paragraphs = divide_song_into_paragraphs(song)

    prompt = system_prompt + context + "ORIGINAL SONG : \n\n" + song + "\n\nAlready generated PARODIE: \n\n"
    parodie = ""
    state = "Finished Correctly"
    try: 
        generated_parodie = generate_lines(prompt, song_in_paragraphs=song_in_paragraphs)

        for paragraph in song_in_paragraphs:
            parodie += paragraph[0] + '\n'
            for line in paragraph[1]:
                parodie += generated_parodie[:generated_parodie.find('\n')+1] 
                generated_parodie = generated_parodie[generated_parodie.find('\n')+1:]
            parodie += '\n'


    except Exception as e:
        print(e)
        state = "Error has occured " + str(e) + "\n" + "Not finished correctly"
        parodie += "\n\n" + "[ERROR]: Not finished correctly" + "\n\n"


    print("Parodie: ", parodie)
    write_song('Experiments/ConstrainedParodieGenerator/GeneratedParodies/', 
                original_song_file_path = song_file_path, 
                parodie = parodie, context = context, 
                system_prompt = system_prompt, 
                prompt = prompt, 
                constraints_used = "SyllableConstraintFull",
                language_model_name = lm.get_name(),
                state = state,
                way_of_generation = "Generated in one go",
                decoding_method = "Beam Search")



if(__name__ == '__main__'):
    song_file_path = 'Songs/json/Taylor_Swift-Is_It_Over_Now_(Small_Version).json'
    #song_file_path = 'Songs/json/Coldplay-Viva_La_Vida.json'

    system_prompt = "I'm a parodie genrator that will write beatifull parodies and make sure that the syllable count and the rhyming of my parodies are the same as the original song\n"
    context = "The following parodie will be about that pineaple shouldn't be on pizza\n"

    generate_parodie(song_file_path, system_prompt, context)

