from Constraints.SyllableConstraint.SyllableConstraintLBL import SyllableConstraintLBL
from Constraints.RhymingConstraint.RhymingConstraintLBL import RhymingConstraintLBL
from Constraints.PosConstraint.PosConstraintLBL import PosConstraintLBL
from Constraints.OptimizedConstraint import OptimizedConstraint
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
from SongUtils import read_song, divide_song_into_paragraphs, get_syllable_count_of_sentence, write_song, forbidden_charachters_to_tokens, get_final_word_of_line,get_pos_tags_of_line, replace_content_for_prompts, cleanup_line, get_song_structure, process_parody
from SongEvaluator import count_same_nb_lines_and_return_same_paragraphs, count_syllable_difference_per_line, count_nb_line_pairs_match_rhyme_scheme, calculate_pos_tag_similarity
import os
import time


from transformers import (
                set_seed, 
                StoppingCriteriaList, 
                MaxLengthCriteria, 
                LogitsProcessorList, 
                NoBadWordsLogitsProcessor, 
                RepetitionPenaltyLogitsProcessor,
                NoRepeatNGramLogitsProcessor,
                TopKLogitsWarper,
                TopPLogitsWarper,
                TemperatureLogitsWarper,
                utils
                )
import torch

######### Supress Warnings ##########
utils.logging.set_verbosity_error()

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
optimized_constraint = None

########## Constants ##########

AVAILABLE_LMS = {'GPT2': GPT2, 'Gemma2BIt': Gemma2BIt, 'Gemma2B': Gemma2B, 'Gemma7B': Gemma7B, 'Gemma7BIt': Gemma7BIt, 'Llama2_7B': Llama2_7B, 'Llama2_7BChat': Llama2_7BChat, 'Llama2_70B': Llama2_70B, 'Llama2_70BChat': Llama2_70BChat, 'Mistral7BV01': Mistral7BV01, 'Mistral7BItV02': Mistral7BItV02, 'Mistral8x7BV01': Mistral8x7BV01, 'Mistral8x7BItV01': Mistral8x7BItV01}

########## LM ##########
def set_language_model(lm_name, use_quantization=False, use_cuda=True):
    torch.cuda.empty_cache()

    global lm
    if lm_name in AVAILABLE_LMS:
        lm = AVAILABLE_LMS[lm_name](use_quantization=use_quantization, use_cuda=use_cuda)
    else:
        raise Exception('Language Model not found')
    global tokenizer
    global model
    global start_token
    tokenizer = lm.get_tokenizer()
    model = lm.get_model()
    start_token = lm.get_start_token()
    #model.bfloat16()
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
    global optimized_constraint
    syllable_constraint = SyllableConstraintLBL(tokenizer, start_token=start_token)
    syllable_constraint.set_special_new_line_tokens(lm.special_new_line_tokens())

    rhyming_constraint = RhymingConstraintLBL(tokenizer, start_token=start_token)

    pos_constraint = PosConstraintLBL(tokenizer, start_token=start_token)

    forbidden_charachters = ['[', ']', '(', ')', '{', '}', '<', '>', '|', '/', '_', '——', '.' '+', '=', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`', ';', ':', '"', "'", ',', '.', '?', '\n', '\n\n']
    forbidden_tokens = forbidden_charachters_to_tokens(tokenizer, forbidden_charachters)
    forbidden_tokens_logit_processor = NoBadWordsLogitsProcessor(forbidden_tokens, eos_token_id=tokenizer.eos_token_id)

    repetition_penalty_logits_processor = RepetitionPenaltyLogitsProcessor(1.2)


    ## Combine Constraints
    constraints = ConstraintList([pos_constraint, rhyming_constraint, syllable_constraint])

    optimized_constraint = OptimizedConstraint(constraints, tokenizer, top_k=100)

    stopping_criteria_list = constraints.get_stopping_criteria_list() + []
    stopping_criteria = StoppingCriteriaList(stopping_criteria_list)
    logits_processor_list = constraints.get_logits_processor_list() + [repetition_penalty_logits_processor, forbidden_tokens_logit_processor]
    #logits_processor_list = [optimized_constraint] + [repetition_penalty_logits_processor, forbidden_tokens_logit_processor]

    

    ## Initialize Post Processor
    post_processor = PostProcessor(tokenizer)
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
def generate_line(prompt, input_ids, **kwargs):
    original_prompt_length = len(prompt)

    ## Encode inputs
    original_input_length = input_ids.shape[-1]
    

    ## Prepare backtracking
    backtracking_logits_processor = BacktrackingLogitsProcessor(original_input_length)
    backtracking = Backtracking(original_input_length, constraints, backtracking_logits_processor, eos_token_id, use_backtracking=kwargs.get('use_backtracking', True))
    
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
            if kwargs.get('top_p') is None:
                raise Exception('top_p not set')

            top_p = kwargs['top_p']
            temperature = kwargs['temperature']
            logits_warper = LogitsProcessorList(
                [  
                    TemperatureLogitsWarper(temperature),
                    TopPLogitsWarper(top_p)
                    
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



def generate_parody(song_file_path, system_prompt, context_prompt, assistant_prompt, **kwargs):
    ## Setup 
    use_cuda = kwargs.get('use_cuda', False)
    use_quantization = kwargs.get('use_quantization', False)
    set_language_model(kwargs['language_model'], use_quantization=use_quantization, use_cuda=use_cuda)
    set_seed(kwargs['seed'])
    set_num_beams(kwargs['num_beams'])

    ## Config constraints
    set_constraints()
    if kwargs.get('syllable_constrained') is None or kwargs['syllable_constrained'] == False:
        syllable_constraint.disable()
    else:
        syllable_constraint.enable()
        syllable_constraint.set_hyperparameters(kwargs['syllable_constraint_hyperparameters'])
    
    if kwargs.get('rhyming_constrained') is None or kwargs['rhyming_constrained'] == False:
        rhyming_constraint.disable()
    else:
        rhyming_constraint.enable()
        rhyming_constraint.set_hyperparameters(kwargs['rhyming_constraint_hyperparameters'])
    
    if kwargs.get('pos_constrained') is None or kwargs['pos_constrained'] == False:
        pos_constraint.disable()
    else:
        pos_constraint.enable()
        pos_constraint.set_hyperparameters(kwargs['pos_constraint_hyperparameters'])
    
    song = read_song(song_file_path) #expects a json file, where the lyrics is stored in the key 'lyrics'
    song_in_paragraphs = divide_song_into_paragraphs(song)
    #print("Original Song: ", song)
    song_in_paragraphs, original_structure = get_song_structure(song_in_paragraphs)
    # print("Original Structure: ", original_structure)
    # print("Song in Paragraphs: ", song_in_paragraphs)

    if system_prompt.endswith('.txt'):
        system_prompt = open(system_prompt, 'r').read()
    if context_prompt.endswith('.txt'):
        context_prompt = open(context_prompt, 'r').read()
    if assistant_prompt.endswith('.txt'):
        assistant_prompt = open(assistant_prompt, 'r').read()
    

    #prompt = system_prompt + context + "ORIGINAL SONG : \n\n" + song + "\n\nAlready generated PARODIE: \n\n" + parodie
    #prompt = system_prompt + context + "\n\nAlready generated PARODIE: \n\n"
    parodie = ""
    state = "Finished Correctly"

    if (kwargs.get('do_sample') is not None and kwargs.get('do_sample') == True):
        if kwargs.get('top_p') is None:
            kwargs['top_p'] = 0.95
        if kwargs.get('temperature') is None:
            kwargs['temperature'] = 0.7
    start_time = time.time()
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
                
                ##Prepare prompt
                prepared_system_prompt, prepared_context_prompt, prepared_assistant_prompt = replace_content_for_prompts(system_prompt, context_prompt, assistant_prompt, parodie, song, rhyming_word, pos_tags, syllable_amount, line)
                prompt, tokenized_prompt = lm.prepare_prompt(prepared_system_prompt, prepared_context_prompt, prepared_assistant_prompt)
                #prompt = system_prompt + context_prompt + "ORIGINAL SONG : \n\n" + song + "\n\nAlready generated PARODIE: \n\n" + parodie
                syllable_constraint.set_original_prompt(prompt)
                optimized_constraint.set_prompt_length(len(prompt))
                ##Generate new line
                
                new_line = generate_line(prompt, tokenized_prompt, new_syllable_amount=syllable_amount, rhyming_word=rhyming_word, pos_tags=pos_tags, **kwargs)
                new_rhyme_word = get_final_word_of_line(new_line)
                rhyming_constraint.add_rhyming_words_to_ignore(new_rhyme_word)
                print("Contraints are satisfied: ", constraints.are_constraints_satisfied(new_line))
                new_line = cleanup_line(new_line)
                parodie += new_line + "\n"
                print(line, " | ",new_line)
            parodie += "\n"
        parodie = process_parody(parodie, original_structure)
    except Exception as e:
        #raise Exception(e)
        print("Error has occured ", e)
        state = "Error has occured " + str(e) + "\n" + "Not finished correctly"
        parodie += "\n\n" + "[ERROR]: Not finished correctly" + "\n\n"

    decoding_method = "Beam Search"
    if (kwargs.get('do_sample') is not None or kwargs.get('do_sample') == True):
        decoding_method = "Sampling Beam Search" + " | top_p: " + str(kwargs['top_p']) + " | temperature: " + str(kwargs['temperature'])

    constraints_used = "" 
    chosen_hyper_parameters = {}
    if kwargs.get('syllable_constrained') is not None and kwargs['syllable_constrained'] == True:
        constraints_used += "Syllable Constraint | "
        chosen_hyper_parameters.update(syllable_constraint.get_hyperparameters_in_dict())
    if kwargs.get('rhyming_constrained') is not None and kwargs['rhyming_constrained'] == True:
        constraints_used += "Rhyming Constraint | "
        chosen_hyper_parameters.update(rhyming_constraint.get_hyperparameters_in_dict())
    if kwargs.get('pos_constrained') is not None and kwargs['pos_constrained'] == True:
        constraints_used += "POS Constraint | "
        chosen_hyper_parameters.update(pos_constraint.get_hyperparameters_in_dict())

    generation_duration = round(time.time() - start_time, 2)

    folder_path_for_generated_parodies = kwargs.get('folder_path_for_generated_parodies', 'Experiments/ConstrainedParodieGenerator/GeneratedParodies/')

    print("Parodie: ", parodie)
    write_song(folder_path_for_generated_parodies, 
                original_song_file_path = song_file_path, 
                parodie = parodie, 
                context = context_prompt, 
                system_prompt = system_prompt, 
                assistant_prompt = assistant_prompt,
                prompt = prompt, 
                constraints_used = constraints_used,
                chosen_hyper_parameters =chosen_hyper_parameters,
                num_beams = kwargs['num_beams'],
                seed = kwargs['seed'],
                duration = generation_duration,
                language_model_name = lm.get_name(),
                state = state,
                way_of_generation = "Line by Line",
                decoding_method = decoding_method)

    parodie_in_paragraphs = divide_song_into_paragraphs(parodie)
    return song_in_paragraphs, parodie_in_paragraphs



if(__name__ == '__main__'):
    ###### Set Up ######
    language_model = 'GPT2'
    #language_model = 'Llama2_7BChat'
    #language_model = "Mistral7BItV02"
    song_directory = 'Songs/json/'
    song_file_path = 'Songs/Taylor_Swift-It_Is_Over_Now_(Very_Small).json'
    #song_file_path = 'Songs/json/Coldplay-Viva_La_Vida.json'
    #song_file_path = 'Songs/json/Taylor_Swift-Is_It_Over_Now_(Small_Version).json'

    # system_prompt = "I'm a parody genrator that will write beatifull parodies and make sure that the syllable count and the rhyming of my parodies are the same as the original song\n"
    # context_prompt = "The following parodie will be about that pineaple shouldn't be on pizza\n"

    #system_prompt = "I'm a parody genrator that will write beatifull parodies and make sure that the syllable count and the rhyming of my parodies are the same as the original song"
    #context_prompt = "The following parodie will be about that pineaple shouldn't be on pizza\nORIGINAL SONG : \n\n{{$SONG}}\n\nAlready generated PARODIE: \n\n{{$PARODY}}"

    prompt_version = "Old"

    system_prompt = "Experiments/ConstrainedParodieGenerator/PromptTexts/"+prompt_version+"/system_prompt.txt"
    context_prompt = "Experiments/ConstrainedParodieGenerator/PromptTexts/"+prompt_version+"/context_prompt.txt"
    assistant_prompt = "Experiments/ConstrainedParodieGenerator/PromptTexts/"+prompt_version+"/assistant_prompt.txt"

    

    ######### Hyperparameters ##########
    syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.5, top_k_tokens_to_consider=200, all_beams_have_syllable_amount=False)
    rhyming_constraint_hyperparameters = RhymingConstraintLBL.hyperparameters_config(max_possible_syllable_count=2, good_beamscore_multiplier_same_rhyme_type=0.95, good_rhyming_token_multiplier=0.9, top_k_rhyme_words=10, rhyme_type='perfect', frequent_words=True)
    pos_constraint_hyperparameters = PosConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.1, good_token_multiplier=0.6, limit_of_pos_similarity_to_satisfy_constraint=0.5, top_k_tokens_to_consider=200)

    
    

    generate_parody(
        song_file_path= song_file_path, 
        system_prompt = system_prompt, 
        context_prompt = context_prompt, 
        assistant_prompt = assistant_prompt,
        language_model = language_model,
        use_cuda=True,
        use_quantization=False,
        do_sample=True, 
        top_p=0.9, 
        temperature=0.75, 
        #temperature=float(2),
        num_beams=2, 
        seed=42, 
        syllable_constrained = True,
        rhyming_constrained = True,
        pos_constrained = True,
        syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
        rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters, 
        pos_constraint_hyperparameters=pos_constraint_hyperparameters
        )

    

