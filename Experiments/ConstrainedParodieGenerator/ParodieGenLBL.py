from Constraints.SyllableConstraint.SyllableConstraintLBL import SyllableConstraintLBL
from Constraints.RhymingConstraint.RhymingConstraintLBL import RhymingConstraintLBL
from Constraints.PosConstraint.PosConstraintLBL import PosConstraintLBL
from Constraint import ConstraintList
from PostProcessor import PostProcessor
from BeamSearchScorerConstrained import BeamSearchScorerConstrained
from Backtracking import Backtracking, BacktrackingLogitsProcessor
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
logits_processor_list = None
eos_token_id = None
pad_token_id = None
post_processor = None

########## Constants ##########

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
    model.bfloat16()
    global eos_token_id
    global pad_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

def set_num_beams(num=2):
    global num_beams
    num_beams = num


######### Constraints ##########
def set_constraints():
    global syllable_constraint
    global rhyming_constraint
    global pos_constraint
    global constraints
    global stopping_criteria
    global logits_processor
    global logits_processor_list
    global post_processor
    syllable_constraint = SyllableConstraintLBL(tokenizer, start_token=start_token)
    syllable_constraint.set_special_new_line_tokens(lm.special_new_line_tokens())

    rhyming_constraint = RhymingConstraintLBL(tokenizer, start_token=start_token)

    pos_constraint = PosConstraintLBL(tokenizer, start_token=start_token)

    forbidden_charachters = ['[', ']', '(', ')', '{', '}', '<', '>', '|', '\\', '/', '_', '——', ' — ', '..' '+', '=', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`', ';', ':', '"', "'", ',', '.', '?', '\n', '\n\n', '  ', '...']
    forbidden_tokens = forbidden_charachters_to_tokens(tokenizer, forbidden_charachters)
    forbidden_tokens_logit_processor = NoBadWordsLogitsProcessor(forbidden_tokens, eos_token_id=tokenizer.eos_token_id)

    no_ngram_logits_processor = NoRepeatNGramLogitsProcessor(2)
    ## Combine Constraints
    constraints = ConstraintList([pos_constraint, rhyming_constraint, syllable_constraint])

    stopping_criteria_list = constraints.get_stopping_criteria_list() + []
    stopping_criteria = StoppingCriteriaList(stopping_criteria_list)

    logits_processor_list = constraints.get_logits_processor_list() + [no_ngram_logits_processor, forbidden_tokens_logit_processor]
    

    ## Initialize Post Processor
    post_processor = PostProcessor()
    logits_processor_list.append(post_processor.get_logits_processor())



def prepare_inputs(input_ids): 
    input_ids = input_ids.to(model.device)
    batch_size = input_ids.shape[0]
    input_ids = input_ids.repeat_interleave(num_beams, dim=0)
    attention_mask = None
    if lm.accepts_attention_mask():
        attention_mask = model._prepare_attention_mask_for_generation(
            input_ids, pad_token_id, eos_token_id
        )
    return input_ids, batch_size, attention_mask



######## Generate Line ########
def generate_line(prompt, **kwargs):
    original_prompt_length = len(prompt)

    ## Encode inputs
    input_ids = tokenizer.encode(prompt, return_tensors="pt")   
    original_input_length = input_ids.shape[-1]
    

    ## Prepare backtracking
    backtracking_logits_processor = BacktrackingLogitsProcessor(original_input_length)
    backtracking = Backtracking(original_input_length, constraints, backtracking_logits_processor)
    
    ## Constraints
    syllable_constraint.set_new_syllable_amount(kwargs['new_syllable_amount'])

    rhyming_constraint.set_rhyming_word(kwargs['rhyming_word'])
    rhyming_constraint.set_required_syllable_count(kwargs['new_syllable_amount'])

    pos_constraint.set_expected_pos_tags(kwargs['pos_tags'])

    ## Generate
    while backtracking.continue_loop():
        ## Prepare inputs
        prepared_input_ids, batch_size, attention_mask = prepare_inputs(input_ids)
        ## Beam Search
        beam_scorer = BeamSearchScorerConstrained(
            batch_size= batch_size,
            max_length=1000,
            num_beams=num_beams,
            device=model.device,
            constraints = constraints,
            num_beam_hyps_to_keep=num_beams,
            length_penalty=10.0,
            post_processor=post_processor
        )

        ## Genearate
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
                    #post_processor.get_logits_processor()
                    
                ]
            )
            
            outputs = model.beam_sample(
                prepared_input_ids,
                beam_scorer=beam_scorer,
                stopping_criteria=stopping_criteria,
                logits_processor=LogitsProcessorList([backtracking_logits_processor] +  logits_processor_list),
                logits_warper=logits_warper,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores = True,
                return_dict_in_generate=True,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=True,
                renormalize_logits=True
            )
        else:
            outputs = model.beam_search(
                prepared_input_ids,
                beam_scorer=beam_scorer,
                stopping_criteria=stopping_criteria,
                logits_processor= LogitsProcessorList([backtracking_logits_processor] +  logits_processor_list),
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores = True,
                return_dict_in_generate=True,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=True,
                renormalize_logits=True 
            )
        
        ## Decode and validate result
        decoded_results = [tokenizer.decode(sequence, skip_special_tokens=True)[original_prompt_length:] for sequence in outputs['sequences']]
        backtracking.validate_result(decoded_results, outputs['sequences'], outputs['sequences_scores'])
        input_ids = backtracking.get_updated_input_ids()

    result = backtracking.get_result()
    return result



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
                syllable_constraint.set_original_prompt(prompt + parodie)
                rhyming_word = None
                if rhyming_lines[i] is not None:
                    rhyming_line = parodie.split('\n')[rhyming_lines[i]-i-1]
                    rhyming_word = get_final_word_of_line(rhyming_line)

                
                pos_tags = get_pos_tags_of_line(line)

                ##Generate new line
                new_line = generate_line(prompt + parodie, new_syllable_amount=syllable_amount, rhyming_word=rhyming_word, pos_tags=pos_tags, **kwargs)
                new_rhyme_word = get_final_word_of_line(new_line)
                rhyming_constraint.add_rhyming_words_to_ignore(new_rhyme_word)
                print("Contraints are satisfied: ", constraints.are_constraints_satisfied(new_line))
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
                constraints_used = kwargs['constrained_used'],
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
    #set_language_model('Llama2_7BChat')
    set_seed(42)
    set_num_beams(2)
    set_constraints()

    

    ######### Hyperparameters ##########
    syllable_constraint.set_hyperparameters(good_beamscore_multiplier=0.5, bad_beamscore_multiplier=5, top_k_tokens_to_consider=30)
    rhyming_constraint.set_hyperparameters(max_possible_syllable_count=3, good_beamscore_multiplier_same_rhyme_type=0.95, good_beamscore_multiplier_assonant=0.9, continue_good_rhyme_multiplier=0.99, good_rhyming_token_multiplier=0.9, top_k_rhyme_words=10, rhyme_type='perfect')
    pos_constraint.set_hyperparameters(good_beamscore_multiplier=0.1, pos_similarity_limit_to_boost=0.5, good_token_multiplier=0.6, margin_of_similarity_with_new_token=0.1, limit_of_pos_similarity_to_satisfy_constraint=0.5, top_k_words_to_consider=200)
    chosen_hyper_parameters = {
        'SyllableConstraintLBL': {'good_beamscore_multiplier': 0.5, 'bad_beamscore_multiplier': 10},
       'RhymingConstraintLBL': {'max_possible_syllable_count': 3, 'good_beamscore_multiplier_same_rhyme_type': 0.95, 'good_beam_score_multiplier_assonant': 0.9, 'continue_good_rhyme_multiplier': 0.99, 'good_rhyming_token_multiplier': 0.9},
        'PosConstraintLBL': {'good_beamscore_multiplier': 0.1, 'pos_similarity_limit_to_boost': 0.5, 'good_token_multiplier': 0.6, 'margin_of_similarity_with_new_token': 0.1, 'limilt_of_pos_similarity_to_satisfy_constraint': 0.5},
        'rhyme_type': 'assonant',
        'top_k_rhyme_words': 10,
       'top_k_words_to_consider_for_pos': 200
    }

    song_directory = 'Songs/json/'
    # song_file_path = 'Songs/json/Taylor_Swift-It_Is_Over_Now_(Very_Small).json'
    #song_file_path = 'Songs/json/Coldplay-Viva_La_Vida.json'
    song_file_path = 'Songs/json/Taylor_Swift-Is_It_Over_Now_(Small_Version).json'

    system_prompt = "I'm a parody genrator that will write beatifull parodies and make sure that the syllable count and the rhyming of my parodies are the same as the original song\n"
    context = "The following parodie will be about that pineaple shouldn't be on pizza\n"

    

    constrained_used = "All"
    
    

    

    generate_parodie(song_file_path, system_prompt, context, do_sample=True, top_k=100, top_p=0.95, temperature=0.7, chosen_hyper_parameters=chosen_hyper_parameters, num_beams=2, seed=42, constrained_used=constrained_used)
    
    

