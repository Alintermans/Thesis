from Constraints.SyllableConstraint.SyllableConstraintLBL import SyllableConstraintLBL
from Constraints.RhymingConstraint.RhymingConstraintLBL import RhymingConstraintLBL
from Constraints.PosConstraint.PosConstraintLBL import PosConstraintLBL
from Constraint import ConstraintList
from BeamSearchScorerConstrained import BeamSearchScorerConstrained
from LanguageModels.GPT2 import GPT2
from LanguageModels.Gemma2BIt import Gemma2BIt
from LanguageModels.Gemma2B import Gemma2B
from LanguageModels.Gemma7B import Gemma7B
from LanguageModels.Gemma7BIt import Gemma7BIt
from LanguageModels.Llama2_7B import Llama2_7B
from LanguageModels.Llama2_7BChat import Llama2_7BChat
from LanguageModels.Llama2_70B import Llama2_70B
from LanguageModels.Llama2_70BChat import Llama2_70BChat
from LanguageModels.Mistral7BV01 import Mistral7BV01
from LanguageModels.Mistral7BItV02 import Mistral7BItV02
from LanguageModels.Mistral8x7BV01 import Mistral8x7BV01
from LanguageModels.Mistral8x7BItV01 import Mistral8x7BItV01
from SongUtils import read_song, divide_song_into_paragraphs, get_syllable_count_of_sentence, write_song, forbidden_charachters_to_tokens, get_final_word_of_line,get_pos_tags_of_line
from SongEvaluator import count_same_nb_lines_and_return_same_paragraphs, count_syllable_difference_per_line, count_nb_line_pairs_match_rhyme_scheme, calculate_pos_tag_similarity
import os

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

########## Global Variables ##########
lm = None
tokenizer = None
model = None
start_token = None
num_beams = None
syllable_constraint = None
rhyming_constraint = None
pos_constraint = None
constraints = None
stopping_criteria = None
logits_processor = None

AVAILABLE_LMS = {'GPT2': GPT2, 'Gemma2BIt': Gemma2BIt, 'Gemma2B': Gemma2B, 'Gemma7B': Gemma7B, 'Gemma7BIt': Gemma7BIt, 'Llama2_7B': Llama2_7B, 'Llama2_7BChat': Llama2_7BChat, 'Llama2_70B': Llama2_70B, 'Llama2_70BChat': Llama2_70BChat, 'Mistral7BV01': Mistral7BV01, 'Mistral7BItV02': Mistral7BItV02, 'Mistral8x7BV01': Mistral8x7BV01, 'Mistral8x7BItV01': Mistral8x7BItV01}

########## LM ##########
def set_language_model(lm_name):
    global lm
    if lm_name in AVAILABLE_LMS:
        lm = AVAILABLE_LMS[lm_name]()
    else:
        raise Exception('Language Model not found')
    global tokenizer
    global model
    global start_token
    tokenizer = lm.get_tokenizer()
    model = lm.get_model()
    start_token = lm.get_start_token()

def set_num_beams(num=2):
    global num_beams
    num_beams = num


######### Constraints ##########
def set_constraints(rhyme_type="assonant", top_k_rhyme_words=10, top_k_words_to_consider_for_pos=200):
    global syllable_constraint
    global rhyming_constraint
    global pos_constraint
    global constraints
    global stopping_criteria
    global logits_processor
    syllable_constraint = SyllableConstraintLBL(tokenizer, start_token=start_token)

    rhyming_constraint = RhymingConstraintLBL(tokenizer, start_token=start_token, top_k_rhyme_words=top_k_rhyme_words, rhyme_type=rhyme_type)

    pos_constraint = PosConstraintLBL(tokenizer, start_token=start_token, top_k_words_to_consider=top_k_words_to_consider_for_pos)

    forbidden_charachters = ['[', ']', '(', ')', '{', '}', '<', '>', '|', '\\', '/', '_', '——', ' — ', '..' '+', '=', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`', ';', ':', '"', "'", ',', '.', '?', '\n', '\n\n', '  ', '...']
    forbidden_tokens = forbidden_charachters_to_tokens(tokenizer, forbidden_charachters)
    forbidden_tokens_logit_processor = NoBadWordsLogitsProcessor(forbidden_tokens, eos_token_id=tokenizer.eos_token_id)

    no_ngram_logits_processor = NoRepeatNGramLogitsProcessor(2)
    ## Combine Constraints
    constraints = ConstraintList([pos_constraint, rhyming_constraint, syllable_constraint])

    stopping_criteria_list = constraints.get_stopping_criteria_list() + []
    stopping_criteria = StoppingCriteriaList(stopping_criteria_list)

    logits_processor_list = constraints.get_logits_processor_list() + [no_ngram_logits_processor, forbidden_tokens_logit_processor]
    logits_processor = LogitsProcessorList(logits_processor_list)








######## Generate Line ########
def generate_line(prompt, **kwargs):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    batch_size = input_ids.shape[0]
    input_ids = input_ids.repeat_interleave(num_beams, dim=0)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    attention_mask = None
    if lm.accepts_attention_mask():
        attention_mask = model._prepare_attention_mask_for_generation(
            input_ids, pad_token_id, eos_token_id
        )
    
    ## Constraints
    syllable_constraint.set_new_syllable_amount(kwargs['new_syllable_amount'])
    rhyming_constraint.set_rhyming_word(kwargs['rhyming_word'])
    rhyming_constraint.set_required_syllable_count(kwargs['new_syllable_amount'])

    pos_constraint.set_expected_pos_tags(kwargs['pos_tags'])
    

    ## Beam Search
    beam_scorer = BeamSearchScorerConstrained(
        batch_size= batch_size,
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
                TemperatureLogitsWarper(temperature),
                #TopKLogitsWarper(top_k),
                TopPLogitsWarper(top_p),
                
             ]
        )
        
        outputs = model.beam_sample(
            input_ids,
            beam_scorer=beam_scorer,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores = True,
            return_dict_in_generate=False,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
            
        )
    else:
        outputs = model.beam_search(
            input_ids,
            beam_scorer=beam_scorer,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores = True,
            return_dict_in_generate=False,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,

        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]



########## Generate Parodie  ##########



def generate_parodie(song_file_path, system_prompt, context, **kwargs):
    song = read_song(song_file_path) #expects a json file, where the lyrics is stored in the key 'lyrics'
    song_in_paragraphs = divide_song_into_paragraphs(song)

    prompt = system_prompt + context + "ORIGINAL SONG : \n\n" + song + "\n\nAlready generated PARODIE: \n\n"
    #prompt = system_prompt + context + "\n\nAlready generated PARODIE: \n\n"
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
            rhyming_lines = rhyming_constraint.get_rhyming_lines(paragraph[1])
            rhyming_constraint.reset_rhyming_words_to_ignore()
            parodie += paragraph[0] + "\n"
            for i in range(len(paragraph[1])):
                line = paragraph[1][i]
                
                ## Constraints Settings
                syllable_amount = get_syllable_count_of_sentence(line)

                rhyming_word = None
                if rhyming_lines[i] is not None:
                    rhyming_line = parodie.split('\n')[rhyming_lines[i]-i-1]
                    rhyming_word = get_final_word_of_line(rhyming_line)

                
                pos_tags = get_pos_tags_of_line(line)

                ##Generate new line
                new_line = generate_line(prompt + parodie, new_syllable_amount=syllable_amount, rhyming_word=rhyming_word, pos_tags=pos_tags, **kwargs)
                rhyming_constraint.add_rhyming_words_to_ignore(rhyming_word)
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
        decoding_method = "Sampling Beam Search" + " | top_p: " + str(kwargs['top_p']) + " | temperature: " + str(kwargs['temperature'])



    print("Parodie: ", parodie)
    write_song('Experiments/ConstrainedParodieGenerator/GeneratedParodies/', 
                original_song_file_path = song_file_path, 
                parodie = parodie, context = context, 
                system_prompt = system_prompt, 
                prompt = prompt, 
                constraints_used = "SyllableConstraintLBL, RhymingConstraintLBL, PosConstraintLBL",
                chosen_hyper_parameters = kwargs['chosen_hyper_parameters'],
                num_beams = kwargs['num_beams'],
                seed = kwargs['seed'],
                language_model_name = lm.get_name(),
                state = state,
                way_of_generation = "Line by Line",
                decoding_method = decoding_method)

    parodie_in_paragraphs = divide_song_into_paragraphs(parodie)
    return song_in_paragraphs, parodie_in_paragraphs



if(__name__ == '__main__'):
    ###### SetUp ######
    set_language_model('GPT2')
    set_seed(42)
    set_num_beams(2)
    set_constraints(rhyme_type="assonant", top_k_rhyme_words=10, top_k_words_to_consider_for_pos=200)

    ########## Ranges For Hyperparameters To Test ##########
    ## General
    num_possible_beams = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    do_samples = [True, False]

    ## Syllable Constraint
    good_beamscore_multipliers_syllable = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
    bad_beamscore_multipliers_syllable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    ## Rhyming Constraint
    rhyme_types = ['assonant', 'perfect', 'near']
    top_k_rhyme_words = [10, 50, 100, 200, 500]
    good_beamscore_multipliers_rhyme = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
    good_beamscore_multipliers_assonant = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
    continue_good_rhyme_multipliers = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    good_rhyming_token_multipliers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
    max_possible_syllable_counts = [1,2,3,4]

    ## POS Constraint
    top_k_words_to_consider_for_pos = [100,200, 500,1000,2000,5000]
    good_beamscore_multipliers_pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
    pos_similarity_limit_to_boosts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
    good_token_multipliers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
    margin_of_similarity_with_new_tokens = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]


    ######### Hyperparameters ##########
    syllable_constraint.set_hyperparameters(good_beamscore_multiplier=0.1, bad_beamscore_multiplier=10)
    rhyming_constraint.set_hyperparameters(max_possible_syllable_count=3, good_beamscore_multiplier_same_rhyme_type=0.95, good_beamscore_multiplier_assonant=0.9, continue_good_rhyme_multiplier=0.99, good_rhyming_token_multiplier=0.9)
    pos_constraint.set_hyperparameters(good_beamscore_multiplier=0.1, pos_similarity_limit_to_boost=0.5, good_token_multiplier=0.6, margin_of_similarity_with_new_token=0.1)
    chosen_hyper_parameters = {
        'SyllableConstraintLBL': {'good_beamscore_multiplier': 0.1, 'bad_beamscore_multiplier': 10},
       'RhymingConstraintLBL': {'max_possible_syllable_count': 3, 'good_beamscore_multiplier_same_rhyme_type': 0.95, 'good_beam_score_multiplier_assonant': 0.9, 'continue_good_rhyme_multiplier': 0.99, 'good_rhyming_token_multiplier': 0.9},
        'PosConstraintLBL': {'good_beamscore_multiplier': 0.1, 'pos_similarity_limit_to_boost': 0.5, 'good_token_multiplier': 0.6, 'margin_of_similarity_with_new_token': 0.1},
        'rhyme_type': 'assonant',
        'top_k_rhyme_words': 10,
       'top_k_words_to_consider_for_pos': 200
    }

    song_directory = 'Songs/json/'
    # song_file_path = 'Songs/json/Taylor_Swift-It_Is_Over_Now_(Very_Small).json'
    song_file_path = 'Songs/json/Coldplay-Viva_La_Vida.json'

    system_prompt = "I'm a parody genrator that will write beatifull parodies and make sure that the syllable count and the rhyming of my parodies are the same as the original song\n"
    context = "The following parodie will be about that pineaple shouldn't be on pizza\n"

    songs = os.listdir(song_directory)
    # for song in songs:
    #     song_file_path = song_directory + song


    ##Test syllable constraints
    # rhyming_constraint.disable()
    # for num_beam in num_beams:
    #     set_num_beams(num_beam)
    #     for do_sample in do_samples:
    #         for good_beamscore_multiplier_syllable in good_beamscore_multipliers_syllable:
    #             for bad_beamscore_multiplier_syllable in bad_beamscore_multipliers_syllable:
    #                 syllable_constraint.set_hyperparameters(good_beamscore_multiplier=good_beamscore_multiplier_syllable, bad_beamscore_multiplier=bad_beamscore_multiplier_syllable)
    #                 chosen_hyper_parameters['SyllableConstraintLBL']['good_beamscore_multiplier'] = good_beamscore_multiplier_syllable
    #                 chosen_hyper_parameters['SyllableConstraintLBL']['bad_beamscore_multiplier'] = bad_beamscore_multiplier_syllable
    #                 original_song, parody = generate_parodie(song_file_path, system_prompt, context, do_sample=do_sample, top_k=100, top_p=0.95, temperature=0.7, chosen_hyper_parameters=chosen_hyper_parameters, num_beams=num_beam, seed=42)


    

    

    

    generate_parodie(song_file_path, system_prompt, context, do_sample=True, top_k=100, top_p=0.95, temperature=0.7, chosen_hyper_parameters=chosen_hyper_parameters, num_beams=2, seed=42)
    
    

