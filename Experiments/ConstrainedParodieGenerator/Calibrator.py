import torch
import os
import sys
import random
import json
import matplotlib.pyplot as plt
from ParodieGenLBL import generate_parody, AVAILABLE_LMS
from Constraints.SyllableConstraint.SyllableConstraintLBL import SyllableConstraintLBL
from Constraints.RhymingConstraint.RhymingConstraintLBL import RhymingConstraintLBL
from Constraints.PosConstraint.PosConstraintLBL import PosConstraintLBL
import platform
import os 
import ray
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from SongEvaluator import evaluate as evaluate_song
import numpy as np
import seaborn as sns
import asyncio
import aiofiles.os
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import re
import statistics

## Constants
GLOBAL_SEED = 42
POSSIBLE_NUM_BEAMS  = [5]
SONG_DIR = "Songs/json/"
SYSTEM_PROMPTS = ["Experiments/ConstrainedParodieGenerator/PromptTexts/1/system_prompt.txt", "Experiments/ConstrainedParodieGenerator/PromptTexts/2/system_prompt.txt", "Experiments/ConstrainedParodieGenerator/PromptTexts/3/system_prompt.txt"]
CONTEXT_PROMPTS = ["Experiments/ConstrainedParodieGenerator/PromptTexts/1/context_prompt.txt", "Experiments/ConstrainedParodieGenerator/PromptTexts/2/context_prompt.txt", "Experiments/ConstrainedParodieGenerator/PromptTexts/3/context_prompt.txt"]
ASSISTANT_PROMPTS = ["Experiments/ConstrainedParodieGenerator/PromptTexts/1/assistant_prompt.txt", "Experiments/ConstrainedParodieGenerator/PromptTexts/2/assistant_prompt.txt", "Experiments/ConstrainedParodieGenerator/PromptTexts/3/assistant_prompt.txt"]

LANGUAGE_MODELS_CHAT = ['Llama2_7BChat', 'Llama2_70BChat', 'Mistral7BItV02', 'Mistral8x7BItV01']
LANGUAGE_MODELS_NON_CHAT = ['Llama2_7B', 'Llama2_70B', 'Mistral7BV01', 'Mistral8x7BV01']


## Init parameters
random.seed(GLOBAL_SEED)
    

START_FOLDER = None

if platform.system() == 'Linux':
    START_FOLDER = os.environ["VSC_DATA"] + "/CallibrationExperiments/"
else:
    START_FOLDER = "Experiments/ConstrainedParodieGenerator/CallibrationExperiments/"



############################################ Generation Functions ############################################

@ray.remote(num_gpus=1, max_calls=1)
def generate_parody_with_ray(**kwargs):
    return generate_parody(**kwargs)

def generate_parody_without_ray(**kwargs):
    return generate_parody(**kwargs)


def calibrate_prompt(song_file_path):
    prompt_nbs = len(SYSTEM_PROMPTS)

    syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.9, top_k_tokens_to_consider=200, all_beams_have_syllable_amount=False)
    rhyming_constraint_hyperparameters = RhymingConstraintLBL.hyperparameters_config(max_possible_syllable_count=3, good_beamscore_multiplier_same_rhyme_type=0.9, good_rhyming_token_multiplier=0.9, top_k_rhyme_words=10, rhyme_type='perfect')
    pos_constraint_hyperparameters = PosConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.6, good_token_multiplier=0.6, limit_of_pos_similarity_to_satisfy_constraint=0.5, top_k_tokens_to_consider=200)

    folder_path_for_generated_parodies = START_FOLDER + "PromptCalibration/"

    if not os.path.exists(folder_path_for_generated_parodies):
        os.makedirs(folder_path_for_generated_parodies)
    
    for prompt_nb in range(prompt_nbs):
        system_prompt = SYSTEM_PROMPTS[prompt_nb]
        context_prompt = CONTEXT_PROMPTS[prompt_nb]
        assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]

        
        for language_model in LANGUAGE_MODELS_CHAT:
            index = 1
            for num_beams in POSSIBLE_NUM_BEAMS:
                if torch.cuda.is_available():
                    ray.get(generate_parody_with_ray.remote(
                    song_file_path= song_file_path, 
                    system_prompt = system_prompt, 
                    context_prompt = context_prompt, 
                    assistant_prompt = assistant_prompt,
                    language_model = language_model,
                    folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(prompt_nb) + "/",
                    use_cuda=True,
                    use_quantization=True,
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=0.75, 
                    num_beams=num_beams, 
                    seed=GLOBAL_SEED, 
                    syllable_constrained = True,
                    rhyming_constrained = True,
                    pos_constrained = True,
                    syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                    rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                    pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                    use_backtracking = True
                    ))
                else:
                    generate_parody_without_ray(
                    song_file_path= song_file_path, 
                    system_prompt = system_prompt, 
                    context_prompt = context_prompt, 
                    assistant_prompt = assistant_prompt,
                    language_model = language_model,
                    folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(prompt_nb) + "/",
                    use_cuda=True,
                    use_quantization=True,
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=0.75, 
                    num_beams=num_beams, 
                    seed=GLOBAL_SEED, 
                    syllable_constrained = False,
                    rhyming_constrained = False,
                    pos_constrained = False,
                    syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                    rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                    pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                    use_backtracking = False
                    )

                index += 1


def calibrate_rhymin_frequency(song_file_path):
    prompt_nb = 0
    system_prompt = SYSTEM_PROMPTS[prompt_nb]
    context_prompt = CONTEXT_PROMPTS[prompt_nb]
    assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]


    syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.9, top_k_tokens_to_consider=200, all_beams_have_syllable_amount=False)
    pos_constraint_hyperparameters = PosConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.9, good_token_multiplier=0.9, limit_of_pos_similarity_to_satisfy_constraint=0.5, top_k_tokens_to_consider=200)

    folder_path_for_generated_parodies = START_FOLDER + "RhymingFrequency/" 

    if not os.path.exists(folder_path_for_generated_parodies):
        os.makedirs(folder_path_for_generated_parodies)


    for language_model in LANGUAGE_MODELS_CHAT:
        index = 1
        for rhyming_top_frequent in [True, False]:
            for num_beams in POSSIBLE_NUM_BEAMS:
                rhyming_constraint_hyperparameters = RhymingConstraintLBL.hyperparameters_config(max_possible_syllable_count=3, good_beamscore_multiplier_same_rhyme_type=0.9, good_rhyming_token_multiplier=0.9, top_k_rhyme_words=10, rhyme_type='perfect', frequent_words=rhyming_top_frequent)
                if torch.cuda.is_available():
                    ray.get(generate_parody_with_ray.remote(
                    song_file_path= song_file_path, 
                    system_prompt = system_prompt, 
                    context_prompt = context_prompt, 
                    assistant_prompt = assistant_prompt,
                    language_model = language_model,
                    folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                    use_cuda=True,
                    use_quantization=True,
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=0.75, 
                    num_beams=num_beams, 
                    seed=GLOBAL_SEED, 
                    syllable_constrained = True,
                    rhyming_constrained = True,
                    pos_constrained = False,
                    syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                    rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                    pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                    use_backtracking = False
                    ))
                else:
                    generate_parody_without_ray(
                    song_file_path= song_file_path,
                    system_prompt = system_prompt,
                    context_prompt = context_prompt,
                    assistant_prompt = assistant_prompt,
                    language_model = language_model,
                    folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                    use_cuda=True,
                    use_quantization=True,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.75,
                    num_beams=num_beams,
                    seed=GLOBAL_SEED,
                    syllable_constrained = True,
                    rhyming_constrained = True,
                    pos_constrained = True,
                    syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                    rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                    pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                    use_backtracking = False
                    )
                    

                index += 1

def calibrate_rhyming_types(song_file_path):
    prompt_nb = 0
    system_prompt = SYSTEM_PROMPTS[prompt_nb]
    context_prompt = CONTEXT_PROMPTS[prompt_nb]
    assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]


    syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.9, top_k_tokens_to_consider=200, all_beams_have_syllable_amount=False)
    pos_constraint_hyperparameters = PosConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.9, good_token_multiplier=0.9, limit_of_pos_similarity_to_satisfy_constraint=0.5, top_k_tokens_to_consider=200)

    folder_path_for_generated_parodies = START_FOLDER + "RhymingTypes/"

    if not os.path.exists(folder_path_for_generated_parodies):
        os.makedirs(folder_path_for_generated_parodies)


    for language_model in LANGUAGE_MODELS_CHAT:
        index = 1
        for rhyme_type in ['perfect', 'near', 'assonant']:
            for num_beams in POSSIBLE_NUM_BEAMS:
                rhyming_constraint_hyperparameters = RhymingConstraintLBL.hyperparameters_config(max_possible_syllable_count=3, good_beamscore_multiplier_same_rhyme_type=0.9, good_rhyming_token_multiplier=0.9, top_k_rhyme_words=10, rhyme_type=rhyme_type)
                if torch.cuda.is_available():
                    ray.get(generate_parody_with_ray.remote(
                    song_file_path= song_file_path, 
                    system_prompt = system_prompt, 
                    context_prompt = context_prompt, 
                    assistant_prompt = assistant_prompt,
                    language_model = language_model,
                    folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                    use_cuda=True,
                    use_quantization=True,
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=0.75, 
                    num_beams=num_beams, 
                    seed=GLOBAL_SEED, 
                    syllable_constrained = True,
                    rhyming_constrained = True,
                    pos_constrained = False,
                    syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                    rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                    pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                    use_backtracking = False
                    ))
                else:
                    generate_parody_without_ray(
                    song_file_path= song_file_path,
                    system_prompt = system_prompt,
                    context_prompt = context_prompt,
                    assistant_prompt = assistant_prompt,
                    language_model = language_model,
                    folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                    use_cuda=True,
                    use_quantization=True,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.75,
                    num_beams=num_beams,
                    seed=GLOBAL_SEED,
                    syllable_constrained = True,
                    rhyming_constrained = True,
                    pos_constrained = True,
                    syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                    rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                    pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                    use_backtracking = False
                    )

                index += 1

def calibrate_backtracking(song_file_path):
    prompt_nb = 0
    system_prompt = SYSTEM_PROMPTS[prompt_nb]
    context_prompt = CONTEXT_PROMPTS[prompt_nb]
    assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]

    

    syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.9, top_k_tokens_to_consider=200, all_beams_have_syllable_amount=False)
    pos_constraint_hyperparameters = PosConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.6, good_token_multiplier=0.6, limit_of_pos_similarity_to_satisfy_constraint=0.5, top_k_tokens_to_consider=200)
    rhyming_constraint_hyperparameters = RhymingConstraintLBL.hyperparameters_config(max_possible_syllable_count=2, good_beamscore_multiplier_same_rhyme_type=0.9, good_rhyming_token_multiplier=0.9, top_k_rhyme_words=10, rhyme_type='perfect')
    folder_path_for_generated_parodies = START_FOLDER + "Backtracking_2/"

    if not os.path.exists(folder_path_for_generated_parodies):
        os.makedirs(folder_path_for_generated_parodies)


    for language_model in LANGUAGE_MODELS_CHAT:
        index = 1
        for use_backtracking in [True, False]:
            for num_beams in POSSIBLE_NUM_BEAMS:
                if torch.cuda.is_available():
                    ray.get(generate_parody_with_ray.remote(
                    song_file_path= song_file_path, 
                    system_prompt = system_prompt, 
                    context_prompt = context_prompt, 
                    assistant_prompt = assistant_prompt,
                    language_model = language_model,
                    folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                    use_cuda=True,
                    use_quantization=True,
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=0.75, 
                    num_beams=num_beams, 
                    seed=GLOBAL_SEED, 
                    syllable_constrained = True,
                    rhyming_constrained = True,
                    pos_constrained = True,
                    syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                    rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                    pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                    use_backtracking = use_backtracking
                    ))
                else:
                    generate_parody_without_ray(
                    song_file_path= song_file_path,
                    system_prompt = system_prompt,
                    context_prompt = context_prompt,
                    assistant_prompt = assistant_prompt,
                    language_model = language_model,
                    folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                    use_cuda=True,
                    use_quantization=True,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.75,
                    num_beams=num_beams,
                    seed=GLOBAL_SEED,
                    syllable_constrained = True,
                    rhyming_constrained = True,
                    pos_constrained = True,
                    syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                    rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                    pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                    use_backtracking = use_backtracking
                    )
                
                index += 1

def generate_all_non_chat(song_file_path):
    prompt_nb = 0
    system_prompt = SYSTEM_PROMPTS[prompt_nb]
    context_prompt = CONTEXT_PROMPTS[prompt_nb]
    assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]

    

    syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.9, top_k_tokens_to_consider=200, all_beams_have_syllable_amount=False)
    pos_constraint_hyperparameters = PosConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.6, good_token_multiplier=0.6, limit_of_pos_similarity_to_satisfy_constraint=0.5, top_k_tokens_to_consider=200)
    rhyming_constraint_hyperparameters = RhymingConstraintLBL.hyperparameters_config(max_possible_syllable_count=3, good_beamscore_multiplier_same_rhyme_type=0.9, good_rhyming_token_multiplier=0.9, top_k_rhyme_words=10, rhyme_type='perfect')
    folder_path_for_generated_parodies = START_FOLDER + "AllNonChat/"

    if not os.path.exists(folder_path_for_generated_parodies):
        os.makedirs(folder_path_for_generated_parodies)


    for language_model in LANGUAGE_MODELS_NON_CHAT:
        index = 1
        for num_beams in POSSIBLE_NUM_BEAMS:
            if torch.cuda.is_available():
                ray.get(generate_parody_with_ray.remote(
                song_file_path= song_file_path, 
                system_prompt = system_prompt, 
                context_prompt = context_prompt, 
                assistant_prompt = assistant_prompt,
                language_model = language_model,
                folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                use_cuda=True,
                use_quantization=True,
                do_sample=True, 
                top_p=0.9, 
                temperature=0.75, 
                num_beams=num_beams, 
                seed=GLOBAL_SEED, 
                syllable_constrained = True,
                rhyming_constrained = True,
                pos_constrained = True,
                syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                use_backtracking = False
                ))
            else:
                generate_parody_without_ray(
                song_file_path= song_file_path,
                system_prompt = system_prompt,
                context_prompt = context_prompt,
                assistant_prompt = assistant_prompt,
                language_model = language_model,
                folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                use_cuda=True,
                use_quantization=True,
                do_sample=True,
                top_p=0.9,
                temperature=0.75,
                num_beams=num_beams,
                seed=GLOBAL_SEED,
                syllable_constrained = True,
                rhyming_constrained = True,
                pos_constrained = True,
                syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                use_backtracking = False
                )


            index += 1

def generate_all_chat(song_file_path):
    prompt_nb = 0
    system_prompt = SYSTEM_PROMPTS[prompt_nb]
    context_prompt = CONTEXT_PROMPTS[prompt_nb]
    assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]

    

    syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.9, top_k_tokens_to_consider=200, all_beams_have_syllable_amount=True)
    pos_constraint_hyperparameters = PosConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.6, good_token_multiplier=0.6, limit_of_pos_similarity_to_satisfy_constraint=0.5, top_k_tokens_to_consider=200)
    rhyming_constraint_hyperparameters = RhymingConstraintLBL.hyperparameters_config(max_possible_syllable_count=2, good_beamscore_multiplier_same_rhyme_type=0.9, good_rhyming_token_multiplier=0.9, top_k_rhyme_words=10, rhyme_type='perfect')
    folder_path_for_generated_parodies = START_FOLDER + "AllChat_with_fixed_beam_search/"

    if not os.path.exists(folder_path_for_generated_parodies):
        os.makedirs(folder_path_for_generated_parodies)


    for language_model in LANGUAGE_MODELS_CHAT:
        index = 1
        for num_beams in POSSIBLE_NUM_BEAMS:
            if torch.cuda.is_available():
                ray.get(generate_parody_with_ray.remote(
                song_file_path= song_file_path, 
                system_prompt = system_prompt, 
                context_prompt = context_prompt, 
                assistant_prompt = assistant_prompt,
                language_model = language_model,
                folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                use_cuda=True,
                use_quantization=True,
                do_sample=True, 
                top_p=0.9, 
                temperature=0.75, 
                num_beams=num_beams, 
                seed=GLOBAL_SEED, 
                syllable_constrained = True,
                rhyming_constrained = True,
                pos_constrained = True,
                syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                use_backtracking = True
                ))
            else:
                generate_parody_without_ray(
                song_file_path= song_file_path,
                system_prompt = system_prompt,
                context_prompt = context_prompt,
                assistant_prompt = assistant_prompt,
                language_model = language_model,
                folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                use_cuda=True,
                use_quantization=True,
                do_sample=True,
                top_p=0.9,
                temperature=0.75,
                num_beams=num_beams,
                seed=GLOBAL_SEED,
                syllable_constrained = True,
                rhyming_constrained = True,
                pos_constrained = True,
                syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                use_backtracking = False
                )



def generate_no_constraints_with_guardrails(song_file_path):
    prompt_nb = 0
    system_prompt = SYSTEM_PROMPTS[prompt_nb]
    context_prompt = CONTEXT_PROMPTS[prompt_nb]
    assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]

    folder_path_for_generated_parodies = START_FOLDER + "NoConstraintsWithGuardrails/"

    if not os.path.exists(folder_path_for_generated_parodies):
        os.makedirs(folder_path_for_generated_parodies)

    for language_model in LANGUAGE_MODELS_CHAT:
        index = 1
        for num_beams in POSSIBLE_NUM_BEAMS:
            if torch.cuda.is_available():
                ray.get(generate_parody_with_ray.remote(
                song_file_path= song_file_path, 
                system_prompt = system_prompt, 
                context_prompt = context_prompt, 
                assistant_prompt = assistant_prompt,
                language_model = language_model,
                folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                use_cuda=True,
                use_quantization=True,
                do_sample=True, 
                top_p=0.9, 
                temperature=0.75, 
                num_beams=num_beams, 
                seed=GLOBAL_SEED, 
                syllable_constrained = False,
                rhyming_constrained = False,
                pos_constrained = False,
                use_backtracking = False,
                use_new_line_stop_criteria=True
                ))
            
    

def calibrate_syllable_constraint(song_file_path, prompt_nb, language_model):
    possible_good_beamscore_multipliers_syllable = [ 0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    possible_top_k_tokens_to_consider = [200]

    folder_path_for_generated_parodies = START_FOLDER + "SyllableConstraint/" + str(prompt_nb)+"/"
    if not os.path.exists(folder_path_for_generated_parodies):
        os.makedirs(folder_path_for_generated_parodies)

    
    system_prompt = SYSTEM_PROMPTS[prompt_nb]
    context_prompt = CONTEXT_PROMPTS[prompt_nb]
    assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]

    index = 1


    for num_beams in POSSIBLE_NUM_BEAMS:
        for good_beamscore_multiplier_syllable in possible_good_beamscore_multipliers_syllable:
            for top_k_tokens_to_consider in possible_top_k_tokens_to_consider:
                syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(good_beamscore_multiplier=good_beamscore_multiplier_syllable, top_k_tokens_to_consider=top_k_tokens_to_consider, all_beams_have_syllable_amount=False)
                if torch.cuda.is_available():
                    ray.get(generate_parody_with_ray.remote(
                    song_file_path= song_file_path, 
                    system_prompt = system_prompt, 
                    context_prompt = context_prompt, 
                    assistant_prompt = assistant_prompt,
                    language_model = language_model,
                    folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                    use_cuda=True,
                    use_quantization=True,
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=0.75, 
                    num_beams=num_beams, 
                    seed=GLOBAL_SEED, 
                    syllable_constrained = True,
                    rhyming_constrained = False,
                    pos_constrained = False,
                    syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                    use_backtracking = False
                    ))
                else:
                    generate_parody_without_ray(
                    song_file_path= song_file_path, 
                    system_prompt = system_prompt, 
                    context_prompt = context_prompt, 
                    assistant_prompt = assistant_prompt,
                    language_model = language_model,
                    folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                    use_cuda=True,
                    use_quantization=True,
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=0.75, 
                    num_beams=num_beams, 
                    seed=GLOBAL_SEED, 
                    syllable_constrained = True,
                    rhyming_constrained = False,
                    pos_constrained = False,
                    syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                    use_backtracking = False
                    )

                index += 1



def calibrate_rhyming_constraint(song_file_path, prompt_nb, language_model, beam_index):
    possible_rhyme_types = [ 'perfect']
    top_k_rhyme_words = [10]
    good_beamscore_multipliers_rhyme = [ 0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    good_rhyming_token_multipliers = [ 0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    max_possible_syllable_counts = [3]

    syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.9, top_k_tokens_to_consider=200, all_beams_have_syllable_amount=False)

    folder_path_for_generated_parodies = START_FOLDER + "RhymingConstraint/" + str(prompt_nb)+"/"

    if not os.path.exists(folder_path_for_generated_parodies):
        os.makedirs(folder_path_for_generated_parodies)
    
    system_prompt = SYSTEM_PROMPTS[prompt_nb]
    context_prompt = CONTEXT_PROMPTS[prompt_nb]
    assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]

    index = 1 + beam_index*6

    for num_beams in POSSIBLE_NUM_BEAMS:
        for rhyme_type in possible_rhyme_types:
            for top_k_rhyme_word in top_k_rhyme_words:
                for good_rhyming_token_multiplier in good_rhyming_token_multipliers:
                    for max_possible_syllable_count in max_possible_syllable_counts:
                        good_beamscore_multiplier_rhyme = good_beamscore_multipliers_rhyme[beam_index]
                        rhyming_constraint_hyperparameters = RhymingConstraintLBL.hyperparameters_config(max_possible_syllable_count= max_possible_syllable_count, good_beamscore_multiplier_same_rhyme_type=good_beamscore_multiplier_rhyme, good_rhyming_token_multiplier=good_rhyming_token_multiplier, top_k_rhyme_words=top_k_rhyme_word, rhyme_type=rhyme_type)

                        if torch.cuda.is_available():
                            ray.get(generate_parody_with_ray.remote(
                                song_file_path= song_file_path, 
                                system_prompt = system_prompt, 
                                context_prompt = context_prompt, 
                                assistant_prompt = assistant_prompt,
                                language_model = language_model,
                                folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                                use_cuda=True,
                                use_quantization=True,
                                do_sample=True, 
                                top_p=0.9, 
                                temperature=0.75, 
                                num_beams=num_beams, 
                                seed=GLOBAL_SEED, 
                                syllable_constrained = True,
                                rhyming_constrained = True,
                                pos_constrained = False,
                                rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                                syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                                use_backtracking = False
                                ))
                        else:
                            generate_parody_without_ray(
                                song_file_path= song_file_path, 
                                system_prompt = system_prompt, 
                                context_prompt = context_prompt, 
                                assistant_prompt = assistant_prompt,
                                language_model = language_model,
                                folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                                use_cuda=True,
                                use_quantization=True,
                                do_sample=True, 
                                top_p=0.9, 
                                temperature=0.75, 
                                num_beams=num_beams, 
                                seed=GLOBAL_SEED, 
                                syllable_constrained = True,
                                rhyming_constrained = True,
                                pos_constrained = False,
                                rhyming_constraint_hyperparameters=rhyming_constraint_hyperparameters,
                                syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                                use_backtracking = False
                                )

                        index += 1

def calibrate_pos_constraint(song_file_path, prompt_nb, language_model, beam_index):
    top_k_tokens_to_consider_for_pos = [200]
    good_beamscore_multipliers_pos = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    good_token_multipliers = [ 0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    limits_of_pos_similarity_to_satisfy_constraint = [0.5]

    syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.9, top_k_tokens_to_consider=200, all_beams_have_syllable_amount=False)

    folder_path_for_generated_parodies = START_FOLDER + "PosConstraint/" + str(prompt_nb)+"/"

    if not os.path.exists(folder_path_for_generated_parodies):
        os.makedirs(folder_path_for_generated_parodies)

    system_prompt = SYSTEM_PROMPTS[prompt_nb]
    context_prompt = CONTEXT_PROMPTS[prompt_nb]
    assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]

    index = 1 + beam_index*6

    for num_beams in POSSIBLE_NUM_BEAMS:
        for top_k_tokens_to_consider in top_k_tokens_to_consider_for_pos:
            for good_token_multiplier in good_token_multipliers:
                for limit_of_pos_similarity_to_satisfy_constraint in limits_of_pos_similarity_to_satisfy_constraint:
                    good_beamscore_multiplier_pos = good_beamscore_multipliers_pos[beam_index]
                    pos_constraint_hyperparameters = PosConstraintLBL.hyperparameters_config(good_beamscore_multiplier=good_beamscore_multiplier_pos, good_token_multiplier=good_token_multiplier, limit_of_pos_similarity_to_satisfy_constraint=limit_of_pos_similarity_to_satisfy_constraint, top_k_tokens_to_consider=top_k_tokens_to_consider)


                    if torch.cuda.is_available():
                        ray.get(generate_parody_with_ray.remote(
                        song_file_path= song_file_path, 
                        system_prompt = system_prompt, 
                        context_prompt = context_prompt, 
                        assistant_prompt = assistant_prompt,
                        language_model = language_model,
                        folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                        use_cuda=True,
                        use_quantization=True,
                        do_sample=True, 
                        top_p=0.95, 
                        temperature=0.7, 
                        num_beams=num_beams, 
                        seed=GLOBAL_SEED, 
                        syllable_constrained = True,
                        rhyming_constrained = False,
                        pos_constrained = True,
                        pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                        syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                        use_backtracking = False
                        ))
                    else:
                        generate_parody_without_ray(
                        song_file_path= song_file_path, 
                        system_prompt = system_prompt, 
                        context_prompt = context_prompt, 
                        assistant_prompt = assistant_prompt,
                        language_model = language_model,
                        folder_path_for_generated_parodies = folder_path_for_generated_parodies + str(index) + "/",
                        use_cuda=True,
                        use_quantization=True,
                        do_sample=True, 
                        top_p=0.95, 
                        temperature=0.7, 
                        num_beams=num_beams, 
                        seed=GLOBAL_SEED, 
                        syllable_constrained = True,
                        rhyming_constrained = False,
                        pos_constrained = True,
                        pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                        syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                        use_backtracking = False
                        )

                    index += 1


#def generate_with_all_constraints(song_file_path, language_model):




############################################ Evaluation Functions ############################################
        

@ray.remote
def calculate_song_evaluation(file, folder_path, rhyme_type='perfect'):
    results = evaluate_song(folder_path + file, rhyme_type=rhyme_type)
    return results

def plot_results(x_data, y_data, xlabel, ylabel, title, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(file_name, dpi=300)

def plot_2d_heatmap(x_data, y_data, z_data, xlabel, ylabel, zlabel, title, file_name):
    n_x = len(np.unique(x_data))
    n_y = len(np.unique(y_data))
    
    z_grid = np.array(z_data).reshape(n_x, n_y)
    
    #mirror y 
    y_data = y_data[::-1]
    z_grid = z_grid[::-1]

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(z_grid, cmap='Blues', cbar_kws={'label': zlabel}, annot=True, xticklabels=x_data, yticklabels=y_data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    folder_path = "/".join(file_name.split("/")[:-1])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save the plot
    plt.savefig(file_name, dpi=300)
    

def evaluate_syllable(language_model_name, folder_path):
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/SyllableConstraint/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/SyllableConstraint/"


    possible_good_beamscore_multipliers_syllable = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]

    original_results = []

    print("Evaluating Syllable Constraint for " + language_model_name)
    constraint_folder_path = "Syllable_Constraint_|_/"
    for index in tqdm(range(len(possible_good_beamscore_multipliers_syllable))):
        temp_folder_path = folder_path + str(index + 1) + "/" + language_model_name + "/" + constraint_folder_path +"json/"
        if os.path.isdir(temp_folder_path):

            # if len(os.listdir(temp_folder_path)) != 20:
            #     raise Exception("Not all songs have been generated only " + str(len(os.listdir(temp_folder_path))))

            results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path) for file in os.listdir(temp_folder_path)])
            
            for file, i in zip(os.listdir(temp_folder_path), range(len(results))):
                with open(temp_folder_path + file, "r") as f:
                    data = json.load(f)
                    results[i]["original_parody_settings"] = data
            original_results.append(results)


            

    if os.path.isdir(dest_folder) == False:
        os.makedirs(dest_folder)
    
    if os.path.isdir(dest_folder+language_model_name.replace(" ", "_")) == False:
        os.makedirs(dest_folder+language_model_name.replace(" ", "_"))
    
    #save results 
    with open(dest_folder+language_model_name.replace(" ", "_")+"/results.json", "w") as f:
        json.dump({
            "results": original_results,
            "good_beamscore_multipliers_syllable": possible_good_beamscore_multipliers_syllable
        }, f, indent=4)

def process_syllable_results(language_model_name):
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/SyllableConstraint/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/SyllableConstraint/"
    
    data = None
    with open(dest_folder+language_model_name.replace(" ", "_")+"/results.json", "r") as f:
        data = json.load(f)
    results = data["results"]
    possible_good_beamscore_multipliers_syllable = data["good_beamscore_multipliers_syllable"]

    avg_perplexities = []
    avg_perplexities_difference = []
    avg_syllable_differences = []
    avg_mean_deviation_syllable_count = []
    avg_correct_syllable_count = []
    avg_duration = []
    avg_correct_rhymes  = []
    avg_rhyme_word_length = []
    avg_pos_similarity = []
    avg_mean_deviation_pos_similarity = []
    avg_correct_pos_lines = []
    avg_overlap = []
    avg_repetition_difference = []

    for result in results:
        perplexities = []
        perplexities_difference = []
        syllable_differences = []
        mean_deviation_syllable_count = []
        correct_syllable_count = []
        duration = []
        correct_rhymes = []
        pos_similarity = []
        mean_deviation_pos_similarity = []
        correct_pos_lines = []
        overlap = []
        repetition_difference = []
        rhyme_word_length = []

        for song in result:
            perplexities.append(song["parody_song_perplexity"])
            perplexities_difference.append(song["parody_song_perplexity"] - song["original_song_perplexity"])
            syllable_differences.append(song["avg_syllable_count_difference"])
            mean_deviation_syllable_count.append(song["mean_deviation_syllable_count"])
            correct_syllable_count.append(song["nb_lines_correct_syllable_count"]/song["correct_nb_lines"])
            duration.append(song["original_parody_settings"]["generation_duration"])
            correct_rhymes.append(song["nb_matching_rhyme_pairs"]/song["nb_expected_rhyming_pairs"])
            pos_similarity.append(song["avg_pos_tag_similarity"])
            mean_deviation_pos_similarity.append(song["mean_deviation_pos_tag_similarity"])
            correct_pos_lines.append(song["nb_correct_pos_similarities"]/song["correct_nb_lines"])
            overlap.append(song["overlap"])
            repetition_difference.append(song["original_song_repetition_score"] - song["parody_song_repetition_score"])
            rhyme_word_length.append(song["avg_rhyme_word_length"])
        
        avg_perplexities.append(statistics.median(perplexities))
        avg_perplexities_difference.append(statistics.median(perplexities_difference))
        avg_syllable_differences.append(statistics.median(syllable_differences))
        avg_mean_deviation_syllable_count.append(statistics.median(mean_deviation_syllable_count))
        avg_correct_syllable_count.append(statistics.median(correct_syllable_count))
        avg_duration.append(statistics.median(duration))
        avg_correct_rhymes.append(statistics.median(correct_rhymes))
        avg_pos_similarity.append(statistics.median(pos_similarity))
        avg_mean_deviation_pos_similarity.append(statistics.median(mean_deviation_pos_similarity))
        avg_correct_pos_lines.append(statistics.median(correct_pos_lines))
        avg_overlap.append(statistics.median(overlap))
        avg_repetition_difference.append(statistics.median(repetition_difference)) 
        avg_rhyme_word_length.append(statistics.median(rhyme_word_length))
    if not os.path.isdir(dest_folder+language_model.replace(" ", "_")):
        os.makedirs(dest_folder+language_model.replace(" ", "_"))
    with open(dest_folder+language_model_name.replace(" ", "_")+"/averages.json", "w") as f:
        json.dump({
            "avg_perplexities": avg_perplexities,
            "avg_perplexities_difference": avg_perplexities_difference,
            "avg_syllable_differences": avg_syllable_differences,
            "avg_mean_deviation_syllable_count": avg_mean_deviation_syllable_count,
            "avg_correct_syllable_count": avg_correct_syllable_count,
            "avg_duration": avg_duration,
            "avg_correct_rhymes": avg_correct_rhymes,
            "avg_rhyme_word_length": avg_rhyme_word_length,
            "avg_pos_similarity": avg_pos_similarity,
            "avg_mean_deviation_pos_similarity": avg_mean_deviation_pos_similarity,
            "avg_correct_pos_lines": avg_correct_pos_lines,
            "avg_overlap": avg_overlap,
            "avg_repetition_difference": avg_repetition_difference,
            "good_beamscore_multipliers_syllable": possible_good_beamscore_multipliers_syllable
        }, f, indent=4)

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_perplexities,
        'Good Beamscore Multiplier',
        'Perplexity',
        'Perplexity vs. Good Beamscore Multiplier',
        dest_folder+language_model_name.replace(" ", "_")+'/perplexity.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_perplexities_difference,
        'Good Beamscore Multiplier',
        'Perplexity Difference With Original Song',
        'Perplexity Difference With Original Song vs. Good Beamscore Multiplier',
        dest_folder+language_model_name.replace(" ", "_")+'/perplexity_difference.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_syllable_differences,
        'Good Beamscore Multiplier',
        'Avg. Syllable Difference',
        'Avg. Syllable Difference vs. Good Beamscore Multiplier',
        dest_folder+language_model_name.replace(" ", "_")+'/syllable_differences.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_mean_deviation_syllable_count,
        'Good Beamscore Multiplier',
        'Mean Deviation Syllable Avg. Difference',
        'Mean Deviation Syllable Avg. Difference vs. Good Beamscore Multiplier',
        dest_folder+language_model_name.replace(" ", "_")+'/mean_deviation_syllable_count.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_correct_syllable_count,
        'Good Beamscore Multiplier',
        'Correct Syllable Count Percentage',
        'Correct Syllable Count Percentage vs. Good Beamscore Multiplier',
        dest_folder+language_model_name.replace(" ", "_")+'/correct_syllable_count.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_duration,
        'Good Beamscore Multiplier',
        'Generation Duration',
        'Generation Duration vs. Good Beamscore Multiplier',
        dest_folder+language_model_name.replace(" ", "_")+'/duration.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_correct_rhymes,
        'Good Beamscore Multiplier',
        'Correct Rhymes Percentage',
        'Correct Rhymes Percentage vs. Good Beamscore Multiplier',
        dest_folder+language_model_name.replace(" ", "_")+'/correct_rhymes.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_pos_similarity,
        'Good Beamscore Multiplier',
        'Avg. POS-tag Similarity',
        'Avg. POS-tag Similarity vs. Good Beamscore Multiplier',
        dest_folder+language_model_name.replace(" ", "_")+'/pos_similarity.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_mean_deviation_pos_similarity,
        'Good Beamscore Multiplier',
        'Mean Deviation Avg. POS-tag Similarity',
        'Mean Deviation Avg. POS-tag Similarity vs. Good Beamscore Multiplier',
        dest_folder+language_model_name.replace(" ", "_")+'/mean_deviation_pos_similarity.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_correct_pos_lines,
        'Good Beamscore Multiplier',
        'Correct POS-tag Lines Percentage',
        'Correct POS-tag Lines Percentage vs. Good Beamscore Multiplier',
        dest_folder+language_model_name.replace(" ", "_")+'/correct_pos_lines.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_overlap,
        'Good Beamscore Multiplier',
        'Overlap with Original Song',
        'Overlap with Original Song vs. Good Beamscore Multiplier',
        dest_folder+language_model_name.replace(" ", "_")+'/overlap.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_repetition_difference,
        'Good Beamscore Multiplier',
        'Repetition Difference with Original Song',
        'Repetition Difference with Original Song vs. Good Beamscore Multiplier',
        dest_folder+language_model_name.replace(" ", "_")+'/repetition_difference.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_rhyme_word_length,
        'Good Beamscore Multiplier',
        'Avg. Rhyme Word Syllable Count',
        'Avg. Rhyme Word Syllable Count vs. Good Beamscore Multiplier',
        dest_folder+language_model_name.replace(" ", "_")+'/rhyme_word_length.png'
    )





async def evaluate_rhyming(language_model_name, folder_path):
    rhyming_folder = 'Experiments/ConstrainedParodieGenerator/CalibrationResults/RhymingConstraint/'

    if platform.system() == 'Linux':
        rhyming_folder = os.environ["VSC_DATA"] + "/CalibrationResults/RhymingConstraint/"

    possible_good_beamscore_multipliers_rhyme = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    good_rhyming_token_multipliers = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    possible_rhyme_types = ['perfect']
    top_k_rhyme_words = [10]
    max_possible_syllable_counts = [3]

    original_results = []

    print("Evaluating Rhyming Constraint for " + language_model_name)
    constraint_folder_path = "Syllable_Constraint_|_Rhyming_Constraint_|_/"
    async for index_beam in atqdm(range(len(possible_good_beamscore_multipliers_rhyme))):
        original_results_per_beam = []



        async for index_token in atqdm(range(len(good_rhyming_token_multipliers))):
            temp_folder_path = folder_path + str(index_beam*6 + index_token + 1) + "/" + language_model_name + "/" + constraint_folder_path +"json/"
            if await aiofiles.os.path.isdir(temp_folder_path):

                dir_list = await aiofiles.os.listdir(temp_folder_path)

                results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path) for file in dir_list])
                for result,file in zip(results, dir_list):
                    with open(temp_folder_path + file, "r") as f:
                        data = json.load(f)
                        result["original_parody_settings"] = data

                original_results_per_beam.append(results)



        original_results.append(original_results_per_beam)


    # More async file operations for saving and plotting results
    if not await aiofiles.os.path.isdir(rhyming_folder):
        await aiofiles.os.makedirs(rhyming_folder)
    
    model_folder = rhyming_folder + language_model_name.replace(" ", "_")
    if not await aiofiles.os.path.isdir(model_folder):
        await aiofiles.os.makedirs(model_folder)
    
    async with aiofiles.open(model_folder + "/results.json", "w") as f:
        await f.write(json.dumps({
            "good_beamscore_multipliers_rhyme": possible_good_beamscore_multipliers_rhyme,
            "good_rhyming_token_multipliers": good_rhyming_token_multipliers,
            "results": original_results
        }, indent=4))

async def evaluate_pos(language_model_name, folder_path):
    pos_folder = 'Experiments/ConstrainedParodieGenerator/CalibrationResults/PosConstraint/'

    if platform.system() == 'Linux':
        pos_folder = os.environ["VSC_DATA"] + "/CalibrationResults/PosConstraint/"

    top_k_tokens_to_consider_for_pos = [200]
    good_beamscore_multipliers_pos = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    good_token_multipliers = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    limits_of_pos_similarity_to_satisfy_constraint = [0.5]

    original_results = []

    print("Evaluating POS Constraint for " + language_model_name)
    constraint_folder_path = "Syllable_Constraint_|_POS_Constraint_|_/"
    
    async for index_beam in atqdm(range(len(good_beamscore_multipliers_pos))):
        original_results_per_beam = []

        async for index_token in atqdm(range(len(good_token_multipliers))):
            temp_folder_path = folder_path + str(index_beam*6 + index_token + 1) + "/" + language_model_name + "/" + constraint_folder_path +"json/"
            if await aiofiles.os.path.isdir(temp_folder_path):

                dir_list = await aiofiles.os.listdir(temp_folder_path)

                results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path) for file in dir_list])
                for result,file in zip(results, dir_list):
                    with open(temp_folder_path + file, "r") as f:
                        data = json.load(f)
                        result["original_parody_settings"] = data

                original_results_per_beam.append(results)
        
        original_results.append(original_results_per_beam)

    # More async file operations for saving and plotting results
    if not await aiofiles.os.path.isdir(pos_folder):
        await aiofiles.os.makedirs(pos_folder)

    model_folder = pos_folder + language_model_name.replace(" ", "_")
    if not await aiofiles.os.path.isdir(model_folder):
        await aiofiles.os.makedirs(model_folder)

    async with aiofiles.open(model_folder + "/results.json", "w") as f:
        await f.write(json.dumps({
            "good_beamscore_multipliers_pos": good_beamscore_multipliers_pos,
            "good_token_multipliers": good_token_multipliers,
            "results": original_results
        }, indent=4))



def process_rhyming_or_pos_results(language_model_name, constraint_type):
    rhyming_folder = 'Experiments/ConstrainedParodieGenerator/CalibrationResults/RhymingConstraint/'
    pos_folder = 'Experiments/ConstrainedParodieGenerator/CalibrationResults/PosConstraint/'

    if platform.system() == 'Linux':
        rhyming_folder = os.environ["VSC_DATA"] + "/CalibrationResults/RhymingConstraint/"
        pos_folder = os.environ["VSC_DATA"] + "/CalibrationResults/PosConstraint/"
    
    if constraint_type == "rhyming":
        folder = rhyming_folder
    else:
        folder = pos_folder
    
    dest_folder = ""
    if constraint_type == "rhyming":
        dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/RhymingConstraint/"
        if platform.system() == 'Linux':
            dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/RhymingConstraint/"
    else:
        dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/PosConstraint/"
        if platform.system() == 'Linux':
            dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/PosConstraint/"
    
    data = None
    with open(folder+language_model_name.replace(" ", "_")+"/results.json", "r") as f:
        data = json.load(f)
    results = data["results"]

    x_data = None
    y_data = None
    x_label = None
    y_label = None

    if constraint_type == "rhyming":
        y_data = data["good_beamscore_multipliers_rhyme"]
        x_data = data["good_rhyming_token_multipliers"]
        y_label = "Good Beamscore Multiplier"
        x_label = "Good Rhyming Token Multiplier"
    else:
        y_data = data["good_beamscore_multipliers_pos"]
        x_data = data["good_token_multipliers"]
        y_label = "Good Beamscore Multiplier"
        x_label = "Good Token Multiplier"
    
    avg_perplexities = []
    avg_perplexities_difference = []
    avg_syllable_differences = []
    avg_mean_deviation_syllable_count = []
    avg_correct_syllable_count = []
    avg_duration = []
    avg_correct_rhymes  = []
    avg_pos_similarity = []
    avg_mean_deviation_pos_similarity = []
    avg_correct_pos_lines = []
    avg_overlap = []
    avg_repetition_difference = []
    avg_rhyme_word_length = []
    nb_songs = []

    for result_beams in results:
        perplexities_per_beam = []
        perplexities_difference_per_beam = []
        syllable_differences_per_beam = []
        mean_deviation_syllable_count_per_beam = []
        correct_syllable_count_per_beam = []
        duration_per_beam = []
        correct_rhymes_per_beam = []
        pos_similarity_per_beam = []
        mean_deviation_pos_similarity_per_beam = []
        correct_pos_lines_per_beam = []
        overlap_per_beam = []
        repetition_difference_per_beam = []
        rhyme_word_length_per_beam = []
        nb_songs_per_beam = []



        for result_token in result_beams:
            perplexities = []
            perplexities_difference = []
            syllable_differences = []
            mean_deviation_syllable_count = []
            correct_syllable_count = []
            duration = []
            correct_rhymes = []
            pos_similarity = []
            mean_deviation_pos_similarity = []
            correct_pos_lines = []
            overlap = []
            repetition_difference = []
            rhyme_word_length = []
            nb_songs_temp = 0

            for song in result_token:
                perplexities.append(song["parody_song_perplexity"])
                perplexities_difference.append(song["parody_song_perplexity"] - song["original_song_perplexity"])
                syllable_differences.append(song["avg_syllable_count_difference"])
                mean_deviation_syllable_count.append(song["mean_deviation_syllable_count"])
                correct_syllable_count.append(song["nb_lines_correct_syllable_count"]/song["correct_nb_lines"])
                duration.append(song["original_parody_settings"]["generation_duration"])
                correct_rhymes.append(song["nb_matching_rhyme_pairs"]/song["nb_expected_rhyming_pairs"])
                pos_similarity.append(song["avg_pos_tag_similarity"])
                mean_deviation_pos_similarity.append(song["mean_deviation_pos_tag_similarity"])
                correct_pos_lines.append(song["nb_correct_pos_similarities"]/song["correct_nb_lines"])
                overlap.append(song["overlap"])
                repetition_difference.append(song["original_song_repetition_score"] - song["parody_song_repetition_score"])
                rhyme_word_length.append(song["avg_rhyme_word_length"])
                nb_songs_temp += 1
            
            perplexities_per_beam.append(statistics.median(perplexities))
            perplexities_difference_per_beam.append(statistics.median(perplexities_difference))
            syllable_differences_per_beam.append(statistics.median(syllable_differences))
            mean_deviation_syllable_count_per_beam.append(statistics.median(mean_deviation_syllable_count))
            correct_syllable_count_per_beam.append(statistics.median(correct_syllable_count))
            duration_per_beam.append(statistics.median(duration))
            correct_rhymes_per_beam.append(statistics.median(correct_rhymes))
            pos_similarity_per_beam.append(statistics.median(pos_similarity))
            mean_deviation_pos_similarity_per_beam.append(statistics.median(mean_deviation_pos_similarity))
            correct_pos_lines_per_beam.append(statistics.median(correct_pos_lines))
            overlap_per_beam.append(statistics.median(overlap))
            repetition_difference_per_beam.append(statistics.median(repetition_difference))
            rhyme_word_length_per_beam.append(statistics.median(rhyme_word_length))
            nb_songs_per_beam.append(nb_songs_temp)

        avg_perplexities.append(perplexities_per_beam)
        avg_perplexities_difference.append(perplexities_difference_per_beam)
        avg_syllable_differences.append(syllable_differences_per_beam)
        avg_mean_deviation_syllable_count.append(mean_deviation_syllable_count_per_beam)
        avg_correct_syllable_count.append(correct_syllable_count_per_beam)
        avg_duration.append(duration_per_beam)
        avg_correct_rhymes.append(correct_rhymes_per_beam)
        avg_pos_similarity.append(pos_similarity_per_beam)
        avg_mean_deviation_pos_similarity.append(mean_deviation_pos_similarity_per_beam)
        avg_correct_pos_lines.append(correct_pos_lines_per_beam)
        avg_overlap.append(overlap_per_beam)
        avg_repetition_difference.append(repetition_difference_per_beam)
        avg_rhyme_word_length.append(rhyme_word_length_per_beam)
        nb_songs.append(nb_songs_per_beam)
    if not os.path.isdir(dest_folder+language_model.replace(" ", "_")):
            os.makedirs(dest_folder+language_model.replace(" ", "_"))
    with open(folder+language_model_name.replace(" ", "_")+"/averages.json", "w") as f:
        json.dump({
            "avg_perplexities": avg_perplexities,
            "avg_perplexities_difference": avg_perplexities_difference,
            "avg_syllable_differences": avg_syllable_differences,
            "avg_mean_deviation_syllable_count": avg_mean_deviation_syllable_count,
            "avg_correct_syllable_count": avg_correct_syllable_count,
            "avg_duration": avg_duration,
            "avg_correct_rhymes": avg_correct_rhymes,
            "avg_rhyme_word_length": avg_rhyme_word_length,
            "avg_pos_similarity": avg_pos_similarity,
            "avg_mean_deviation_pos_similarity": avg_mean_deviation_pos_similarity,
            "avg_correct_pos_lines": avg_correct_pos_lines,
            "avg_overlap": avg_overlap,
            "avg_repetition_difference": avg_repetition_difference,
            "good_beamscore_multipliers_rhyme": x_data,
            "good_rhyming_token_multipliers": y_data,
            "nb_songs": nb_songs

        }, f, indent=4)
    
    plot_2d_heatmap(
        x_data,
        y_data,
        avg_perplexities,
        x_label,
        y_label,
        'Perplexity',
        'Perplexity vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/perplexity.png'
    )

    plot_2d_heatmap(
        x_data,
        y_data,
        avg_perplexities_difference,
        x_label,
        y_label,
        'Perplexity Difference With Original Song',
        'Perplexity Difference With Original Song vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/perplexity_difference.png'
    )

    plot_2d_heatmap(
        x_data,
        y_data,
        avg_syllable_differences,
        x_label,
        y_label,
        'Avg. Syllable Difference',
        'Avg. Syllable Difference vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/syllable_differences.png'
    )

    plot_2d_heatmap(
        x_data,
        y_data,
        avg_mean_deviation_syllable_count,
        x_label,
        y_label,
        'Mean Deviation Syllable Avg. Difference',
        'Mean Deviation Syllable Avg. Difference vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/mean_deviation_syllable_count.png'
    )

    plot_2d_heatmap(
        x_data,
        y_data,
        avg_correct_syllable_count,
        x_label,
        y_label,
        'Correct Syllable Count Percentage',
        'Correct Syllable Count Percentage vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/correct_syllable_count.png'
    )

    plot_2d_heatmap(
        x_data,
        y_data,
        avg_duration,
        x_label,
        y_label,
        'Generation Duration',
        'Generation Duration vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/duration.png'
    )

    plot_2d_heatmap(
        x_data,
        y_data,
        avg_correct_rhymes,
        x_label,
        y_label,
        'Correct Rhymes Percentage',
        'Correct Rhymes Percentage vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/correct_rhymes.png'
    )

    plot_2d_heatmap(
        x_data,
        y_data,
        avg_pos_similarity,
        x_label,
        y_label,
        'Avg. POS-tag Similarity',
        'Avg. POS-tag Similarity vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/pos_similarity.png'
    )

    plot_2d_heatmap(
        x_data,
        y_data,
        avg_mean_deviation_pos_similarity,
        x_label,
        y_label,
        'Mean Deviation Avg. POS-tag Similarity',
        'Mean Deviation Avg. POS-tag Similarity vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/mean_deviation_pos_similarity.png'
    )

    plot_2d_heatmap(
        x_data,
        y_data,
        avg_correct_pos_lines,
        x_label,
        y_label,
        'Correct POS-tag Lines Percentage',
        'Correct POS-tag Lines Percentage vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/correct_pos_lines.png'
    )

    plot_2d_heatmap(
        x_data,
        y_data,
        avg_overlap,
        x_label,
        y_label,
        'Overlap with Original Song',
        'Overlap with Original Song vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/overlap.png'
    )

    plot_2d_heatmap(
        x_data,
        y_data,
        avg_repetition_difference,
        x_label,
        y_label,
        'Repetition Difference with Original Song',
        'Repetition Difference with Original Song vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/repetition_difference.png'
    )

    plot_2d_heatmap(
        x_data,
        y_data,
        avg_rhyme_word_length,
        x_label,
        y_label,
        'Avg. Rhyme Word Syllable Count',
        'Avg. Rhyme Word Syllable Count vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/rhyme_word_length.png'
    )

    plot_2d_heatmap(
        x_data,
        y_data,
        nb_songs,
        x_label,
        y_label,
        'Number of Songs',
        'Number of Songs vs. ' + x_label + ' and ' + y_label,
        folder+language_model_name.replace(" ", "_")+'/nb_songs.png'
    )


    

async def evaluate_prompt(folder_path):
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/PromptEvaluation/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/PromptEvaluation/"
    
    possible_prompts = [1, 2, 3]

    original_results = []

    print("Evaluating Prompt Evaluation")
    constraint_folder_path = "Syllable_Constraint_|_Rhyming_Constraint_|_POS_Constraint_|_/"
    async for language_model in atqdm(LANGUAGE_MODELS_CHAT):
        original_results_per_model = []
        language_model = AVAILABLE_LMS[language_model].get_name()
        async for index in atqdm(range(len(possible_prompts))):
            temp_folder_path = folder_path + str(index) + "/" + language_model + '/' +constraint_folder_path +"json/"
            if await aiofiles.os.path.isdir(temp_folder_path):

                dir_list = await aiofiles.os.listdir(temp_folder_path)

                results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path) for file in dir_list])
                for result,file in zip(results, dir_list):
                    with open(temp_folder_path + file, "r") as f:
                        data = json.load(f)
                        result["original_parody_settings"] = data

                original_results_per_model.append(results)
        
        original_results.append(original_results_per_model)

    if not await aiofiles.os.path.isdir(dest_folder):
        await aiofiles.os.makedirs(dest_folder)

    with open(dest_folder + "results.json", "w") as f:
        json.dump({
            "results": original_results,
            "possible_prompts": possible_prompts
        }, f, indent=4)

    
def process_prompt_results():
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/PromptEvaluation/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/PromptEvaluation/"
    
    data = None
    with open(dest_folder+"results.json", "r") as f:
        data = json.load(f)
    results = data["results"]
    possible_prompts = data["possible_prompts"]

    for i in range(len(LANGUAGE_MODELS_CHAT)):
        language_model = LANGUAGE_MODELS_CHAT[i]
        avg_perplexities = []
        avg_perplexities_difference = []
        avg_syllable_differences = []
        avg_mean_deviation_syllable_count = []
        avg_correct_syllable_count = []
        avg_duration = []
        avg_correct_rhymes  = []
        avg_pos_similarity = []
        avg_mean_deviation_pos_similarity = []
        avg_correct_pos_lines = []
        avg_overlap = []
        avg_repetition_difference = []
        avg_rhyme_word_length = []

        for result in results[i]:
            perplexities = []
            perplexities_difference = []
            syllable_differences = []
            mean_deviation_syllable_count = []
            correct_syllable_count = []
            duration = []
            correct_rhymes = []
            pos_similarity = []
            mean_deviation_pos_similarity = []
            correct_pos_lines = []
            overlap = []
            repetition_difference = []
            rhyme_word_length = []

            for song in result:
                perplexities.append(song["parody_song_perplexity"])
                perplexities_difference.append(song["parody_song_perplexity"] - song["original_song_perplexity"])
                syllable_differences.append(song["avg_syllable_count_difference"])
                mean_deviation_syllable_count.append(song["mean_deviation_syllable_count"])
                correct_syllable_count.append(song["nb_lines_correct_syllable_count"]/song["correct_nb_lines"])
                duration.append(song["original_parody_settings"]["generation_duration"])
                correct_rhymes.append(song["nb_matching_rhyme_pairs"]/song["nb_expected_rhyming_pairs"])
                pos_similarity.append(song["avg_pos_tag_similarity"])
                mean_deviation_pos_similarity.append(song["mean_deviation_pos_tag_similarity"])
                correct_pos_lines.append(song["nb_correct_pos_similarities"]/song["correct_nb_lines"])
                overlap.append(song["overlap"])
                repetition_difference.append(song["original_song_repetition_score"] - song["parody_song_repetition_score"])
                rhyme_word_length.append(song["avg_rhyme_word_length"])
            
            avg_perplexities.append(statistics.median(perplexities))
            avg_perplexities_difference.append(statistics.median(perplexities_difference))
            avg_syllable_differences.append(statistics.median(syllable_differences))
            avg_mean_deviation_syllable_count.append(statistics.median(mean_deviation_syllable_count))
            avg_correct_syllable_count.append(statistics.median(correct_syllable_count))
            avg_duration.append(statistics.median(duration))
            avg_correct_rhymes.append(statistics.median(correct_rhymes))
            avg_pos_similarity.append(statistics.median(pos_similarity))
            avg_mean_deviation_pos_similarity.append(statistics.median(mean_deviation_pos_similarity))
            avg_correct_pos_lines.append(statistics.median(correct_pos_lines))
            avg_overlap.append(statistics.median(overlap))
            avg_repetition_difference.append(statistics.median(repetition_difference))
            avg_rhyme_word_length.append(statistics.median(rhyme_word_length))
        if not os.path.isdir(dest_folder+language_model.replace(" ", "_")):
            os.makedirs(dest_folder+language_model.replace(" ", "_"))
        with open(dest_folder+language_model.replace(" ", "_")+"/averages.json", "w") as f:
            json.dump({
                "avg_perplexities": avg_perplexities,
                "avg_perplexities_difference": avg_perplexities_difference,
                "avg_syllable_differences": avg_syllable_differences,
                "avg_mean_deviation_syllable_count": avg_mean_deviation_syllable_count,
                "avg_correct_syllable_count": avg_correct_syllable_count,
                "avg_duration": avg_duration,
                "avg_correct_rhymes": avg_correct_rhymes,
                "avg_rhyme_word_length": avg_rhyme_word_length,
                "avg_pos_similarity": avg_pos_similarity,
                "avg_mean_deviation_pos_similarity": avg_mean_deviation_pos_similarity,
                "avg_correct_pos_lines": avg_correct_pos_lines,
                "avg_overlap": avg_overlap,
                "avg_repetition_difference": avg_repetition_difference,
                "possible_prompts": possible_prompts
            }, f, indent=4)
        
        plot_results(
            possible_prompts,
            avg_perplexities,
            'Prompt Number',
            'Perplexity',
            'Perplexity vs. Prompt Number',
            dest_folder+language_model.replace(" ", "_")+'/perplexity.png'
        )

        plot_results(
            possible_prompts,
            avg_perplexities_difference,
            'Prompt Number',
            'Perplexity Difference With Original Song',
            'Perplexity Difference With Original Song vs. Prompt Number',
            dest_folder+language_model.replace(" ", "_")+'/perplexity_difference.png'
        )

        plot_results(
            possible_prompts,
            avg_syllable_differences,
            'Prompt Number',
            'Avg. Syllable Difference',
            'Avg. Syllable Difference vs. Prompt Number',
            dest_folder+language_model.replace(" ", "_")+'/syllable_differences.png'
        )

        plot_results(
            possible_prompts,
            avg_mean_deviation_syllable_count,
            'Prompt Number',
            'Mean Deviation Syllable Avg. Difference',
            'Mean Deviation Syllable Avg. Difference vs. Prompt Number',
            dest_folder+language_model.replace(" ", "_")+'/mean_deviation_syllable_count.png'
        )

        plot_results(
            possible_prompts,
            avg_correct_syllable_count,
            'Prompt Number',
            'Correct Syllable Count Percentage',
            'Correct Syllable Count Percentage vs. Prompt Number',
            dest_folder+language_model.replace(" ", "_")+'/correct_syllable_count.png'
        )

        plot_results(
            possible_prompts,
            avg_duration,
            'Prompt Number',
            'Generation Duration',
            'Generation Duration vs. Prompt Number',
            dest_folder+language_model.replace(" ", "_")+'/duration.png'
        )

        plot_results(
            possible_prompts,
            avg_correct_rhymes,
            'Prompt Number',
            'Correct Rhymes Percentage',
            'Correct Rhymes Percentage vs. Prompt Number',
            dest_folder+language_model.replace(" ", "_")+'/correct_rhymes.png'
        )

        plot_results(
            possible_prompts,
            avg_pos_similarity,
            'Prompt Number',
            'Avg. POS-tag Similarity',
            'Avg. POS-tag Similarity vs. Prompt Number',
            dest_folder+language_model.replace(" ", "_")+'/pos_similarity.png'
        )

        plot_results(
            possible_prompts,
            avg_mean_deviation_pos_similarity,
            'Prompt Number',
            'Mean Deviation Avg. POS-tag Similarity',
            'Mean Deviation Avg. POS-tag Similarity vs. Prompt Number',
            dest_folder+language_model.replace(" ", "_")+'/mean_deviation_pos_similarity.png'
        )

        plot_results(
            possible_prompts,
            avg_correct_pos_lines,
            'Prompt Number',
            'Correct POS-tag Lines Percentage',
            'Correct POS-tag Lines Percentage vs. Prompt Number',
            dest_folder+language_model.replace(" ", "_")+'/correct_pos_lines.png'
        )

        plot_results(
            possible_prompts,
            avg_overlap,
            'Prompt Number',
            'Overlap with Original Song',
            'Overlap with Original Song vs. Prompt Number',
            dest_folder+language_model.replace(" ", "_")+'/overlap.png'
        )

        plot_results(
            possible_prompts,
            avg_repetition_difference,
            'Prompt Number',
            'Repetition Difference with Original Song',
            'Repetition Difference with Original Song vs. Prompt Number',
            dest_folder+language_model.replace(" ", "_")+'/repetition_difference.png'
        )

        plot_results(
            possible_prompts,
            avg_rhyme_word_length,
            'Prompt Number',
            'Avg. Rhyme Word Syllable Count',
            'Avg. Rhyme Word Syllable Count vs. Prompt Number',
            dest_folder+language_model.replace(" ", "_")+'/rhyme_word_length.png'
        )

            

async def evaluate_rhyming_frequencies(folder_path):
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/RhymingFrequencies/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/RhymingFrequencies/"

    possible_top_frequent_words = [True, False]

    original_results = []

    print("Evaluating Rhyming Frequencies")
    constraint_folder_path = "Syllable_Constraint_|_Rhyming_Constraint_|_POS_Constraint_|_/"
    async for language_model in atqdm(LANGUAGE_MODELS_CHAT):
        original_results_per_model = []
        language_model = AVAILABLE_LMS[language_model].get_name()
        async for index in atqdm(range(len(possible_top_frequent_words))):
            temp_folder_path = folder_path + str(index) + "/" + language_model + constraint_folder_path + "json/"
            if await aiofiles.os.path.isdir(temp_folder_path):

                dir_list = await aiofiles.os.listdir(temp_folder_path)

                results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path) for file in dir_list])
                for result,file in zip(results, dir_list):
                    with open(temp_folder_path + file, "r") as f:
                        data = json.load(f)
                        result["original_parody_settings"] = data

                original_results_per_model.append(results)
        
        original_results.append(original_results_per_model)

    if not await aiofiles.os.path.isdir(dest_folder):
        await aiofiles.os.makedirs(dest_folder)

    with open(dest_folder + "results.json", "w") as f:
        json.dump({
            "results": original_results,
            "possible_top_frequent_words": possible_top_frequent_words
        }, f, indent=4)

def process_rhyming_frequencies_results():
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/RhymingFrequencies/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/RhymingFrequencies/"
    
    data = None
    with open(dest_folder+"results.json", "r") as f:
        data = json.load(f)
    results = data["results"]
    possible_top_frequent_words = data["possible_top_frequent_words"]

    for language_model in LANGUAGE_MODELS_CHAT:
        avg_perplexities = []
        avg_perplexities_difference = []
        avg_syllable_differences = []
        avg_mean_deviation_syllable_count = []
        avg_correct_syllable_count = []
        avg_duration = []
        avg_correct_rhymes  = []
        avg_pos_similarity = []
        avg_mean_deviation_pos_similarity = []
        avg_correct_pos_lines = []
        avg_overlap = []
        avg_repetition_difference = []
        avg_rhyme_word_length = []

        for result in results:
            perplexities = []
            perplexities_difference = []
            syllable_differences = []
            mean_deviation_syllable_count = []
            correct_syllable_count = []
            duration = []
            correct_rhymes = []
            pos_similarity = []
            mean_deviation_pos_similarity = []
            correct_pos_lines = []
            overlap = []
            repetition_difference = []
            rhyme_word_length = []

            for song in result:
                perplexities.append(song["parody_song_perplexity"])
                perplexities_difference.append(song["parody_song_perplexity"] - song["original_song_perplexity"])
                syllable_differences.append(song["avg_syllable_count_difference"])
                mean_deviation_syllable_count.append(song["mean_deviation_syllable_count"])
                correct_syllable_count.append(song["nb_lines_correct_syllable_count"]/song["correct_nb_lines"])
                duration.append(song["original_parody_settings"]["generation_duration"])
                correct_rhymes.append(song["nb_matching_rhyme_pairs"]/song["nb_expected_rhyming_pairs"])
                pos_similarity.append(song["avg_pos_tag_similarity"])
                mean_deviation_pos_similarity.append(song["mean_deviation_pos_tag_similarity"])
                correct_pos_lines.append(song["nb_correct_pos_similarities"]/song["correct_nb_lines"])
                overlap.append(song["overlap"])
                repetition_difference.append(song["original_song_repetition_score"] - song["parody_song_repetition_score"])
                rhyme_word_length.append(song["avg_rhyme_word_length"])

            avg_perplexities.append(statistics.median(perplexities))
            avg_perplexities_difference.append(statistics.median(perplexities_difference))
            avg_syllable_differences.append(statistics.median(syllable_differences))
            avg_mean_deviation_syllable_count.append(statistics.median(mean_deviation_syllable_count))
            avg_correct_syllable_count.append(statistics.median(correct_syllable_count))
            avg_duration.append(statistics.median(duration))
            avg_correct_rhymes.append(statistics.median(correct_rhymes))
            avg_pos_similarity.append(statistics.median(pos_similarity))
            avg_mean_deviation_pos_similarity.append(statistics.median(mean_deviation_pos_similarity))
            avg_correct_pos_lines.append(statistics.median(correct_pos_lines))
            avg_overlap.append(statistics.median(overlap))
            avg_repetition_difference.append(statistics.median(repetition_difference))
            avg_rhyme_word_length.append(statistics.median(rhyme_word_length))
        if not os.path.isdir(dest_folder+language_model.replace(" ", "_")):
            os.makedirs(dest_folder+language_model.replace(" ", "_"))
        with open(dest_folder+language_model.replace(" ", "_")+"/averages.json", "w") as f:
            json.dump({
                "avg_perplexities": avg_perplexities,
                "avg_perplexities_difference": avg_perplexities_difference,
                "avg_syllable_differences": avg_syllable_differences,
                "avg_mean_deviation_syllable_count": avg_mean_deviation_syllable_count,
                "avg_correct_syllable_count": avg_correct_syllable_count,
                "avg_duration": avg_duration,
                "avg_correct_rhymes": avg_correct_rhymes,
                "avg_rhyme_word_length": avg_rhyme_word_length,
                "avg_pos_similarity": avg_pos_similarity,
                "avg_mean_deviation_pos_similarity": avg_mean_deviation_pos_similarity,
                "avg_correct_pos_lines": avg_correct_pos_lines,
                "avg_overlap": avg_overlap,
                "avg_repetition_difference": avg_repetition_difference,
                "possible_top_frequent_words": possible_top_frequent_words
            }, f, indent=4)

        plot_results(
            possible_top_frequent_words,
            avg_perplexities,
            'Top Frequent Words',
            'Perplexity',
            'Perplexity vs. Top Frequent Words',
            dest_folder+language_model.replace(" ", "_")+'/perplexity.png'
        )

        plot_results(
            possible_top_frequent_words,
            avg_perplexities_difference,
            'Top Frequent Words',
            'Perplexity Difference With Original Song',
            'Perplexity Difference With Original Song vs. Top Frequent Words',
            dest_folder+language_model.replace(" ", "_")+'/perplexity_difference.png'
        )

        plot_results(
            possible_top_frequent_words,
            avg_syllable_differences,
            'Top Frequent Words',
            'Avg. Syllable Difference',
            'Avg. Syllable Difference vs. Top Frequent Words',
            dest_folder+language_model.replace(" ", "_")+'/syllable_differences.png'
        )

        plot_results(
            possible_top_frequent_words,
            avg_mean_deviation_syllable_count,
            'Top Frequent Words',
            'Mean Deviation Syllable Avg. Difference',
            'Mean Deviation Syllable Avg. Difference vs. Top Frequent Words',
            dest_folder+language_model.replace(" ", "_")+'/mean_deviation_syllable_count.png'
        )

        plot_results(
            possible_top_frequent_words,
            avg_correct_syllable_count,
            'Top Frequent Words',
            'Correct Syllable Count Percentage',
            'Correct Syllable Count Percentage vs. Top Frequent Words',
            dest_folder+language_model.replace(" ", "_")+'/correct_syllable_count.png'
        )

        plot_results(
            possible_top_frequent_words,
            avg_duration,
            'Top Frequent Words',
            'Generation Duration',
            'Generation Duration vs. Top Frequent Words',
            dest_folder+language_model.replace(" ", "_")+'/duration.png'
        )

        plot_results(
            possible_top_frequent_words,
            avg_correct_rhymes,
            'Top Frequent Words',
            'Correct Rhymes Percentage',
            'Correct Rhymes Percentage vs. Top Frequent Words',
            dest_folder+language_model.replace(" ", "_")+'/correct_rhymes.png'
        )

        plot_results(
            possible_top_frequent_words,
            avg_pos_similarity,
            'Top Frequent Words',
            'Avg. POS-tag Similarity',
            'Avg. POS-tag Similarity vs. Top Frequent Words',
            dest_folder+language_model.replace(" ", "_")+'/pos_similarity.png'
        )

        plot_results(
            possible_top_frequent_words,
            avg_mean_deviation_pos_similarity,
            'Top Frequent Words',
            'Mean Deviation Avg. POS-tag Similarity',
            'Mean Deviation Avg. POS-tag Similarity vs. Top Frequent Words',
            dest_folder+language_model.replace(" ", "_")+'/mean_deviation_pos_similarity.png'
        )

        plot_results(
            possible_top_frequent_words,
            avg_correct_pos_lines,
            'Top Frequent Words',
            'Correct POS-tag Lines Percentage',
            'Correct POS-tag Lines Percentage vs. Top Frequent Words',
            dest_folder+language_model.replace(" ", "_")+'/correct_pos_lines.png'
        )

        plot_results(
            possible_top_frequent_words,
            avg_overlap,
            'Top Frequent Words',
            'Overlap with Original Song',
            'Overlap with Original Song vs. Top Frequent Words',
            dest_folder+language_model.replace(" ", "_")+'/overlap.png'
        )

        plot_results(
            possible_top_frequent_words,
            avg_repetition_difference,
            'Top Frequent Words',
            'Repetition Difference with Original Song',
            'Repetition Difference with Original Song vs. Top Frequent Words',
            dest_folder+language_model.replace(" ", "_")+'/repetition_difference.png'
        )

        plot_results(
            possible_top_frequent_words,
            avg_rhyme_word_length,
            'Top Frequent Words',
            'Avg. Rhyme Word Syllable Count',
            'Avg. Rhyme Word Syllable Count vs. Top Frequent Words',
            dest_folder+language_model.replace(" ", "_")+'/rhyme_word_length.png'
        )

async def evaluate_rhyming_types(folder_path):
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/RhymingTypes/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/RhymingTypes/"

    possible_rhyming_types = ["perfect", "near", "assonant"]

    original_results = []

    print("Evaluating Rhyming Types")
    constraint_folder_path = "Syllable_Constraint_|_Rhyming_Constraint_|_/"
    async for language_model in atqdm(LANGUAGE_MODELS_CHAT):
        original_results_per_model = []
        language_model = AVAILABLE_LMS[language_model].get_name()
        async for index in atqdm(range(len(possible_rhyming_types))):
            temp_folder_path = folder_path + str(index + 1) + "/" + language_model + '/' + constraint_folder_path + "json/"
            if await aiofiles.os.path.isdir(temp_folder_path):

                dir_list = await aiofiles.os.listdir(temp_folder_path)

                results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path, possible_rhyming_types[index]) for file in dir_list])
                for result,file in zip(results, dir_list):
                    with open(temp_folder_path + file, "r") as f:
                        data = json.load(f)
                        result["original_parody_settings"] = data

                original_results_per_model.append(results)
        
        original_results.append(original_results_per_model)

    if not await aiofiles.os.path.isdir(dest_folder):
        await aiofiles.os.makedirs(dest_folder)

    with open(dest_folder + "results.json", "w") as f:
        json.dump({
            "results": original_results,
            "possible_rhyming_types": possible_rhyming_types
        }, f, indent=4)


def process_rhyming_types_results():
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/RhymingTypes/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/RhymingTypes/"
    
    data = None
    with open(dest_folder+"results.json", "r") as f:
        data = json.load(f)
    results = data["results"]
    possible_rhyming_types = data["possible_rhyming_types"]

    for i in range(len(LANGUAGE_MODELS_CHAT)):
        language_model = LANGUAGE_MODELS_CHAT[i]
        avg_perplexities = []
        avg_perplexities_difference = []
        avg_syllable_differences = []
        avg_mean_deviation_syllable_count = []
        avg_correct_syllable_count = []
        avg_duration = []
        avg_correct_rhymes  = []
        avg_pos_similarity = []
        avg_mean_deviation_pos_similarity = []
        avg_correct_pos_lines = []
        avg_overlap = []
        avg_repetition_difference = []
        avg_rhyme_word_length = []

        for result in results[i]:
            perplexities = []
            perplexities_difference = []
            syllable_differences = []
            mean_deviation_syllable_count = []
            correct_syllable_count = []
            duration = []
            correct_rhymes = []
            pos_similarity = []
            mean_deviation_pos_similarity = []
            correct_pos_lines = []
            overlap = []
            repetition_difference = []
            rhyme_word_length = []

            for song in result:
                perplexities.append(song["parody_song_perplexity"])
                perplexities_difference.append(song["parody_song_perplexity"] - song["original_song_perplexity"])
                syllable_differences.append(song["avg_syllable_count_difference"])
                mean_deviation_syllable_count.append(song["mean_deviation_syllable_count"])
                correct_syllable_count.append(song["nb_lines_correct_syllable_count"]/song["correct_nb_lines"])
                duration.append(song["original_parody_settings"]["generation_duration"])
                correct_rhymes.append(song["nb_matching_rhyme_pairs"]/song["nb_expected_rhyming_pairs"])
                pos_similarity.append(song["avg_pos_tag_similarity"])
                mean_deviation_pos_similarity.append(song["mean_deviation_pos_tag_similarity"])
                correct_pos_lines.append(song["nb_correct_pos_similarities"]/song["correct_nb_lines"])
                overlap.append(song["overlap"])
                repetition_difference.append(song["original_song_repetition_score"] - song["parody_song_repetition_score"])
                rhyme_word_length.append(song["avg_rhyme_word_length"])

            avg_perplexities.append(statistics.median(perplexities))
            avg_perplexities_difference.append(statistics.median(perplexities_difference))
            avg_syllable_differences.append(statistics.median(syllable_differences))
            avg_mean_deviation_syllable_count.append(statistics.median(mean_deviation_syllable_count))
            avg_correct_syllable_count.append(statistics.median(correct_syllable_count))
            avg_duration.append(statistics.median(duration))
            avg_correct_rhymes.append(statistics.median(correct_rhymes))
            avg_pos_similarity.append(statistics.median(pos_similarity))
            avg_mean_deviation_pos_similarity.append(statistics.median(mean_deviation_pos_similarity))
            avg_correct_pos_lines.append(statistics.median(correct_pos_lines))
            avg_overlap.append(statistics.median(overlap))
            avg_repetition_difference.append(statistics.median(repetition_difference))
            avg_rhyme_word_length.append(statistics.median(rhyme_word_length))
        if not os.path.isdir(dest_folder+language_model.replace(" ", "_")):
            os.makedirs(dest_folder+language_model.replace(" ", "_"))
        with open(dest_folder+language_model.replace(" ", "_")+"/averages.json", "w") as f:
            json.dump({
                "avg_perplexities": avg_perplexities,
                "avg_perplexities_difference": avg_perplexities_difference,
                "avg_syllable_differences": avg_syllable_differences,
                "avg_mean_deviation_syllable_count": avg_mean_deviation_syllable_count,
                "avg_correct_syllable_count": avg_correct_syllable_count,
                "avg_duration": avg_duration,
                "avg_correct_rhymes": avg_correct_rhymes,
                "avg_rhyme_word_length": avg_rhyme_word_length,
                "avg_pos_similarity": avg_pos_similarity,
                "avg_mean_deviation_pos_similarity": avg_mean_deviation_pos_similarity,
                "avg_correct_pos_lines": avg_correct_pos_lines,
                "avg_overlap": avg_overlap,
                "avg_repetition_difference": avg_repetition_difference,
                "possible_rhyming_types": possible_rhyming_types
            }, f, indent=4)

        plot_results(
            possible_rhyming_types,
            avg_perplexities,
            'Rhyming Type',
            'Perplexity',
            'Perplexity vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/perplexity.png'
        )

        plot_results(
            possible_rhyming_types,
            avg_perplexities_difference,
            'Rhyming Type',
            'Perplexity Difference With Original Song',
            'Perplexity Difference With Original Song vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/perplexity_difference.png'
        )

        plot_results(
            possible_rhyming_types,
            avg_syllable_differences,
            'Rhyming Type',
            'Avg. Syllable Difference',
            'Avg. Syllable Difference vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/syllable_differences.png'
        )

        plot_results(
            possible_rhyming_types,
            avg_mean_deviation_syllable_count,
            'Rhyming Type',
            'Mean Deviation Syllable Avg. Difference',
            'Mean Deviation Syllable Avg. Difference vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/mean_deviation_syllable_count.png'
        )

        plot_results(
            possible_rhyming_types,
            avg_correct_syllable_count,
            'Rhyming Type',
            'Correct Syllable Count Percentage',
            'Correct Syllable Count Percentage vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/correct_syllable_count.png'
        )

        plot_results(
            possible_rhyming_types,
            avg_duration,
            'Rhyming Type',
            'Generation Duration',
            'Generation Duration vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/duration.png'
        )

        plot_results(
            possible_rhyming_types,
            avg_correct_rhymes,
            'Rhyming Type',
            'Correct Rhymes Percentage',
            'Correct Rhymes Percentage vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/correct_rhymes.png'
        )

        plot_results(
            possible_rhyming_types,
            avg_pos_similarity,
            'Rhyming Type',
            'Avg. POS-tag Similarity',
            'Avg. POS-tag Similarity vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/pos_similarity.png'
        )

        plot_results(
            possible_rhyming_types,
            avg_mean_deviation_pos_similarity,
            'Rhyming Type',
            'Mean Deviation Avg. POS-tag Similarity',
            'Mean Deviation Avg. POS-tag Similarity vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/mean_deviation_pos_similarity.png'
        )

        plot_results(
            possible_rhyming_types,
            avg_correct_pos_lines,
            'Rhyming Type',
            'Correct POS-tag Lines Percentage',
            'Correct POS-tag Lines Percentage vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/correct_pos_lines.png'
        )

        plot_results(
            possible_rhyming_types,
            avg_overlap,
            'Rhyming Type',
            'Overlap with Original Song',
            'Overlap with Original Song vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/overlap.png'
        )

        plot_results(
            possible_rhyming_types,
            avg_repetition_difference,
            'Rhyming Type',
            'Repetition Difference with Original Song',
            'Repetition Difference with Original Song vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/repetition_difference.png'
        )

        plot_results(
            possible_rhyming_types,
            avg_rhyme_word_length,
            'Rhyming Type',
            'Avg. Rhyme Word Syllable Count',
            'Avg. Rhyme Word Syllable Count vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/rhyme_word_length.png'
        )

        plot_results(
            possible_rhyming_types,
            avg_rhyme_word_length,
            'Rhyming Type',
            'Avg. Rhyme Word Syllable Count',
            'Avg. Rhyme Word Syllable Count vs. Rhyming Type',
            dest_folder+language_model.replace(" ", "_")+'/rhyme_word_length.png'
        )


async def evaluate_backtracking(folder_path):
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/Backtracking/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/Backtracking/"

    possible_backtracking = [True, False]

    original_results = []

    print("Evaluating Backtracking")
    constraint_folder_path = "Syllable_Constraint_|_Rhyming_Constraint_|_POS_Constraint_|_/"
    async for language_model in atqdm(LANGUAGE_MODELS_CHAT):
        original_results_per_model = []
        language_model = AVAILABLE_LMS[language_model].get_name()
        async for index in atqdm(range(len(possible_backtracking))):
            temp_folder_path = folder_path + str(index +1) + "/" + language_model + '/' + constraint_folder_path + "json/"
            if await aiofiles.os.path.isdir(temp_folder_path):

                dir_list = await aiofiles.os.listdir(temp_folder_path)

                results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path) for file in dir_list])
                for result,file in zip(results, dir_list):
                    with open(temp_folder_path + file, "r") as f:
                        data = json.load(f)
                        result["original_parody_settings"] = data

                original_results_per_model.append(results)
        
        original_results.append(original_results_per_model)

    if not await aiofiles.os.path.isdir(dest_folder):
        await aiofiles.os.makedirs(dest_folder)

    with open(dest_folder + "results.json", "w") as f:
        json.dump({
            "results": original_results,
            "possible_backtracking": possible_backtracking
        }, f, indent=4)

def process_backtracking_results():
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/Backtracking_2/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/Backtracking/"
    
    data = None
    with open(dest_folder+"results.json", "r") as f:
        data = json.load(f)
    results = data["results"]
    possible_backtracking = data["possible_backtracking"]

    for i in range(len(LANGUAGE_MODELS_CHAT)):
        language_model = LANGUAGE_MODELS_CHAT[i]
        avg_perplexities = []
        avg_perplexities_difference = []
        avg_syllable_differences = []
        avg_mean_deviation_syllable_count = []
        avg_correct_syllable_count = []
        avg_duration = []
        avg_correct_rhymes  = []
        avg_pos_similarity = []
        avg_mean_deviation_pos_similarity = []
        avg_correct_pos_lines = []
        avg_overlap = []
        avg_repetition_difference = []
        avg_rhyme_word_length = []

        for result in results[i]:
            perplexities = []
            perplexities_difference = []
            syllable_differences = []
            mean_deviation_syllable_count = []
            correct_syllable_count = []
            duration = []
            correct_rhymes = []
            pos_similarity = []
            mean_deviation_pos_similarity = []
            correct_pos_lines = []
            overlap = []
            repetition_difference = []
            rhyme_word_length = []

            for song in result:
                perplexities.append(song["parody_song_perplexity"])
                perplexities_difference.append(song["parody_song_perplexity"] - song["original_song_perplexity"])
                syllable_differences.append(song["avg_syllable_count_difference"])
                mean_deviation_syllable_count.append(song["mean_deviation_syllable_count"])
                correct_syllable_count.append(song["nb_lines_correct_syllable_count"]/song["correct_nb_lines"])
                duration.append(song["original_parody_settings"]["generation_duration"])
                correct_rhymes.append(song["nb_matching_rhyme_pairs"]/song["nb_expected_rhyming_pairs"])
                pos_similarity.append(song["avg_pos_tag_similarity"])
                mean_deviation_pos_similarity.append(song["mean_deviation_pos_tag_similarity"])
                correct_pos_lines.append(song["nb_correct_pos_similarities"]/song["correct_nb_lines"])
                overlap.append(song["overlap"])
                repetition_difference.append(song["original_song_repetition_score"] - song["parody_song_repetition_score"])
                rhyme_word_length.append(song["avg_rhyme_word_length"])

            avg_perplexities.append(statistics.median(perplexities))
            avg_perplexities_difference.append(statistics.median(perplexities_difference))
            avg_syllable_differences.append(statistics.median(syllable_differences))
            avg_mean_deviation_syllable_count.append(statistics.median(mean_deviation_syllable_count))
            avg_correct_syllable_count.append(statistics.median(correct_syllable_count))
            avg_duration.append(statistics.median(duration))
            avg_correct_rhymes.append(statistics.median(correct_rhymes))
            avg_pos_similarity.append(statistics.median(pos_similarity))
            avg_mean_deviation_pos_similarity.append(statistics.median(mean_deviation_pos_similarity))
            avg_correct_pos_lines.append(statistics.median(correct_pos_lines))
            avg_overlap.append(statistics.median(overlap))
            avg_repetition_difference.append(statistics.median(repetition_difference))
            avg_rhyme_word_length.append(statistics.median(rhyme_word_length))

        if not os.path.isdir(dest_folder+language_model.replace(" ", "_")):
            os.makedirs(dest_folder+language_model.replace(" ", "_"))

        with open(dest_folder+language_model.replace(" ", "_")+"/averages.json", "w") as f:
            json.dump({
                "avg_perplexities": avg_perplexities,
                "avg_perplexities_difference": avg_perplexities_difference,
                "avg_syllable_differences": avg_syllable_differences,
                "avg_mean_deviation_syllable_count": avg_mean_deviation_syllable_count,
                "avg_correct_syllable_count": avg_correct_syllable_count,
                "avg_duration": avg_duration,
                "avg_correct_rhymes": avg_correct_rhymes,
                "avg_rhyme_word_length": avg_rhyme_word_length,
                "avg_pos_similarity": avg_pos_similarity,
                "avg_mean_deviation_pos_similarity": avg_mean_deviation_pos_similarity,
                "avg_correct_pos_lines": avg_correct_pos_lines,
                "avg_overlap": avg_overlap,
                "avg_repetition_difference": avg_repetition_difference,
                "possible_backtracking": possible_backtracking
            }, f, indent=4)

        plot_results(
            possible_backtracking,
            avg_perplexities,
            'Backtracking',
            'Perplexity',
            'Perplexity vs. Backtracking',
            dest_folder+language_model.replace(" ", "_")+'/perplexity.png'
        )

        plot_results(
            possible_backtracking,
            avg_perplexities_difference,
            'Backtracking',
            'Perplexity Difference With Original Song',
            'Perplexity Difference With Original Song vs. Backtracking',
            dest_folder+language_model.replace(" ", "_")+'/perplexity_difference.png'
        )

        plot_results(
            possible_backtracking,
            avg_syllable_differences,
            'Backtracking',
            'Avg. Syllable Difference',
            'Avg. Syllable Difference vs. Backtracking',
            dest_folder+language_model.replace(" ", "_")+'/syllable_differences.png'
        )

        plot_results(
            possible_backtracking,
            avg_mean_deviation_syllable_count,
            'Backtracking',
            'Mean Deviation Syllable Avg. Difference',
            'Mean Deviation Syllable Avg. Difference vs. Backtracking',
            dest_folder+language_model.replace(" ", "_")+'/mean_deviation_syllable_count.png'
        )

        plot_results(
            possible_backtracking,
            avg_correct_syllable_count,
            'Backtracking',
            'Correct Syllable Count Percentage',
            'Correct Syllable Count Percentage vs. Backtracking',
            dest_folder+language_model.replace(" ", "_")+'/correct_syllable_count.png'
        )

        plot_results(
            possible_backtracking,
            avg_duration,
            'Backtracking',
            'Generation Duration',
            'Generation Duration vs. Backtracking',
            dest_folder+language_model.replace(" ", "_")+'/duration.png'
        )

        plot_results(
            possible_backtracking,
            avg_correct_rhymes,
            'Backtracking',
            'Correct Rhymes Percentage',
            'Correct Rhymes Percentage vs. Backtracking',
            dest_folder+language_model.replace(" ", "_")+'/correct_rhymes.png'
        )

        plot_results(
            possible_backtracking,
            avg_pos_similarity,
            'Backtracking',
            'Avg. POS-tag Similarity',
            'Avg. POS-tag Similarity vs. Backtracking',
            dest_folder+language_model.replace(" ", "_")+'/pos_similarity.png'
        )

        plot_results(
            possible_backtracking,
            avg_mean_deviation_pos_similarity,
            'Backtracking',
            'Mean Deviation Avg. POS-tag Similarity',
            'Mean Deviation Avg. POS-tag Similarity vs. Backtracking',
            dest_folder+language_model.replace(" ", "_")+'/mean_deviation_pos_similarity.png'
        )

        plot_results(
            possible_backtracking,
            avg_correct_pos_lines,
            'Backtracking',
            'Correct POS-tag Lines Percentage',
            'Correct POS-tag Lines Percentage vs. Backtracking',
            dest_folder+language_model.replace(" ", "_")+'/correct_pos_lines.png'
        )

        plot_results(
            possible_backtracking,
            avg_overlap,
            'Backtracking',
            'Overlap with Original Song',
            'Overlap with Original Song vs. Backtracking',
            dest_folder+language_model.replace(" ", "_")+'/overlap.png'
        )

        plot_results(
            possible_backtracking,
            avg_repetition_difference,
            'Backtracking',
            'Repetition Difference with Original Song',
            'Repetition Difference with Original Song vs. Backtracking',
            dest_folder+language_model.replace(" ", "_")+'/repetition_difference.png'
        )

        plot_results(
            possible_backtracking,
            avg_rhyme_word_length,
            'Backtracking',
            'Avg. Rhyme Word Syllable Count',
            'Avg. Rhyme Word Syllable Count vs. Backtracking',
            dest_folder+language_model.replace(" ", "_")+'/rhyme_word_length.png'
        )



async def evaluate_all_non_chat(folder_path):
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/AllNonChat/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/AllNonChat/"

    original_results = []

    print("Evaluating All Non Chat")

    async for language_model in atqdm(LANGUAGE_MODELS_NON_CHAT-1):
        original_results_per_model = []
        language_model = AVAILABLE_LMS[language_model].get_name()
        constraint_folder_path = "Syllable_Constraint_|_Rhyming_Constraint_|_POS_Constraint_|_/"
        temp_folder_path = folder_path + language_model + '/' + constraint_folder_path + "json/"
        if await aiofiles.os.path.isdir(temp_folder_path):

            dir_list = await aiofiles.os.listdir(temp_folder_path)

            results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path) for file in dir_list])
            for result,file in zip(results, dir_list):
                with open(temp_folder_path + file, "r") as f:
                    data = json.load(f)
                    result["original_parody_settings"] = data

            original_results_per_model.append(results)
        
        original_results.append(original_results_per_model)

    if not await aiofiles.os.path.isdir(dest_folder):
        await aiofiles.os.makedirs(dest_folder)

    with open(dest_folder + "results.json", "w") as f:
        json.dump({
            "results": original_results
        }, f, indent=4)

def process_all_non_chat_results():
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/AllNonChat/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/AllNonChat/"
    
    data = None
    with open(dest_folder+"results.json", "r") as f:
        data = json.load(f)
    results = data["results"]

    for i in range(len(LANGUAGE_MODELS_NON_CHAT)-1):
        language_model = LANGUAGE_MODELS_NON_CHAT[i]
        avg_perplexities = 0
        avg_perplexities_difference = 0
        avg_syllable_differences = 0
        avg_mean_deviation_syllable_count = 0
        avg_correct_syllable_count = 0
        avg_duration = 0
        avg_correct_rhymes  = 0
        avg_pos_similarity = 0
        avg_mean_deviation_pos_similarity = 0
        avg_correct_pos_lines = 0
        avg_overlap = 0
        avg_repetition_difference = 0
        avg_rhyme_word_length = 0
        result = results[i][0]
        
        perplexities = []
        perplexities_difference = []
        syllable_differences = []
        mean_deviation_syllable_count = []
        correct_syllable_count = []
        duration = []
        correct_rhymes = []
        pos_similarity = []
        mean_deviation_pos_similarity = []
        correct_pos_lines = []
        overlap = []
        repetition_difference = []
        rhyme_word_length = []



        for song in result:
            perplexities.append(song["parody_song_perplexity"])
            perplexities_difference.append(song["parody_song_perplexity"] - song["original_song_perplexity"])
            syllable_differences.append(song["avg_syllable_count_difference"])
            mean_deviation_syllable_count.append(song["mean_deviation_syllable_count"])
            correct_syllable_count.append(song["nb_lines_correct_syllable_count"]/song["correct_nb_lines"])
            duration.append(song["original_parody_settings"]["generation_duration"])
            correct_rhymes.append(song["nb_matching_rhyme_pairs"]/song["nb_expected_rhyming_pairs"])
            pos_similarity.append(song["avg_pos_tag_similarity"])
            mean_deviation_pos_similarity.append(song["mean_deviation_pos_tag_similarity"])
            correct_pos_lines.append(song["nb_correct_pos_similarities"]/song["correct_nb_lines"])
            overlap.append(song["overlap"])
            repetition_difference.append(song["original_song_repetition_score"] - song["parody_song_repetition_score"])
            rhyme_word_length.append(song["avg_rhyme_word_length"])
        
        avg_perplexities = statistics.median(perplexities)
        avg_perplexities_difference = statistics.median(perplexities_difference)
        avg_syllable_differences = statistics.median(syllable_differences)
        avg_mean_deviation_syllable_count = statistics.median(mean_deviation_syllable_count)
        avg_correct_syllable_count = statistics.median(correct_syllable_count)
        avg_duration = statistics.median(duration)
        avg_correct_rhymes = statistics.median(correct_rhymes)
        avg_pos_similarity = statistics.median(pos_similarity)
        avg_mean_deviation_pos_similarity = statistics.median(mean_deviation_pos_similarity)
        avg_correct_pos_lines = statistics.median(correct_pos_lines)
        avg_overlap = statistics.median(overlap)
        avg_repetition_difference = statistics.median(repetition_difference)
        avg_rhyme_word_length = statistics.median(rhyme_word_length)

        if not os.path.isdir(dest_folder+language_model.replace(" ", "_")):
            os.makedirs(dest_folder+language_model.replace(" ", "_"))

        with open(dest_folder+language_model.replace(" ", "_")+"/averages.json", "w") as f:
            json.dump({
                "avg_perplexities": avg_perplexities,
                "avg_perplexities_difference": avg_perplexities_difference,
                "avg_syllable_differences": avg_syllable_differences,
                "avg_mean_deviation_syllable_count": avg_mean_deviation_syllable_count,
                "avg_correct_syllable_count": avg_correct_syllable_count,
                "avg_duration": avg_duration,
                "avg_correct_rhymes": avg_correct_rhymes,
                "avg_rhyme_word_length": avg_rhyme_word_length,
                "avg_pos_similarity": avg_pos_similarity,
                "avg_mean_deviation_pos_similarity": avg_mean_deviation_pos_similarity,
                "avg_correct_pos_lines": avg_correct_pos_lines,
                "avg_overlap": avg_overlap,
                "avg_repetition_difference": avg_repetition_difference
            }, f, indent=4)


async def evaluate_all_chat(folder_path):
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/AllChat/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/AllChat/"

    original_results = []

    print("Evaluating All Chat")
    constraint_folder_path = "Syllable_Constraint_|_Rhyming_Constraint_|_POS_Constraint_|_/"
    async for language_model in atqdm(LANGUAGE_MODELS_CHAT):
        original_results_per_model = []
        language_model = AVAILABLE_LMS[language_model].get_name()
        temp_folder_path = folder_path + language_model + '/' + constraint_folder_path + "json/"
        if await aiofiles.os.path.isdir(temp_folder_path):

            dir_list = await aiofiles.os.listdir(temp_folder_path)

            results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path) for file in dir_list])
            for result,file in zip(results, dir_list):
                with open(temp_folder_path + file, "r") as f:
                    data = json.load(f)
                    result["original_parody_settings"] = data

            original_results_per_model.append(results)
        
        original_results.append(original_results_per_model)

    if not await aiofiles.os.path.isdir(dest_folder):
        await aiofiles.os.makedirs(dest_folder)

    with open(dest_folder + "results.json", "w") as f:
        json.dump({
            "results": original_results
        }, f, indent=4)

def process_all_chat_results():
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/AllChat/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/AllChat/"
    
    data = None
    with open(dest_folder+"results.json", "r") as f:
        data = json.load(f)
    results = data["results"]

    for i in range(len(LANGUAGE_MODELS_CHAT)):
        language_model = LANGUAGE_MODELS_CHAT[i]
        avg_perplexities = 0
        avg_perplexities_difference = 0
        avg_syllable_differences = 0
        avg_mean_deviation_syllable_count = 0
        avg_correct_syllable_count = 0
        avg_duration = 0
        avg_correct_rhymes  = 0
        avg_pos_similarity = 0
        avg_mean_deviation_pos_similarity = 0
        avg_correct_pos_lines = 0
        avg_overlap = 0
        avg_repetition_difference = 0
        avg_rhyme_word_length = 0
        result = results[i][0]
        
        perplexities = []
        perplexities_difference = []
        syllable_differences = []
        mean_deviation_syllable_count = []
        correct_syllable_count = []
        duration = []
        correct_rhymes = []
        pos_similarity = []
        mean_deviation_pos_similarity = []
        correct_pos_lines = []
        overlap = []
        repetition_difference = []
        rhyme_word_length = []



        for song in result:
            perplexities.append(song["parody_song_perplexity"])
            perplexities_difference.append(song["parody_song_perplexity"] - song["original_song_perplexity"])
            syllable_differences.append(song["avg_syllable_count_difference"])
            mean_deviation_syllable_count.append(song["mean_deviation_syllable_count"])
            correct_syllable_count.append(song["nb_lines_correct_syllable_count"]/song["correct_nb_lines"])
            duration.append(song["original_parody_settings"]["generation_duration"])
            correct_rhymes.append(song["nb_matching_rhyme_pairs"]/song["nb_expected_rhyming_pairs"])
            pos_similarity.append(song["avg_pos_tag_similarity"])
            mean_deviation_pos_similarity.append(song["mean_deviation_pos_tag_similarity"])
            correct_pos_lines.append(song["nb_correct_pos_similarities"]/song["correct_nb_lines"])
            overlap.append(song["overlap"])
            repetition_difference.append(song["original_song_repetition_score"] - song["parody_song_repetition_score"])
            rhyme_word_length.append(song["avg_rhyme_word_length"])
        
        avg_perplexities = statistics.median(perplexities)
        avg_perplexities_difference = statistics.median(perplexities_difference)
        avg_syllable_differences = statistics.median(syllable_differences)
        avg_mean_deviation_syllable_count = statistics.median(mean_deviation_syllable_count)
        avg_correct_syllable_count = statistics.median(correct_syllable_count)
        avg_duration = statistics.median(duration)
        avg_correct_rhymes = statistics.median(correct_rhymes)
        avg_pos_similarity = statistics.median(pos_similarity)
        avg_mean_deviation_pos_similarity = statistics.median(mean_deviation_pos_similarity)
        avg_correct_pos_lines = statistics.median(correct_pos_lines)
        avg_overlap = statistics.median(overlap)
        avg_repetition_difference = statistics.median(repetition_difference)
        avg_rhyme_word_length = statistics.median(rhyme_word_length)

        if not os.path.isdir(dest_folder+language_model.replace(" ", "_")):
            os.makedirs(dest_folder+language_model.replace(" ", "_"))

        with open(dest_folder+language_model.replace(" ", "_")+"/averages.json", "w") as f:
            json.dump({
                "avg_perplexities": avg_perplexities,
                "avg_perplexities_difference": avg_perplexities_difference,
                "avg_syllable_differences": avg_syllable_differences,
                "avg_mean_deviation_syllable_count": avg_mean_deviation_syllable_count,
                "avg_correct_syllable_count": avg_correct_syllable_count,
                "avg_duration": avg_duration,
                "avg_correct_rhymes": avg_correct_rhymes,
                "avg_rhyme_word_length": avg_rhyme_word_length,
                "avg_pos_similarity": avg_pos_similarity,
                "avg_mean_deviation_pos_similarity": avg_mean_deviation_pos_similarity,
                "avg_correct_pos_lines": avg_correct_pos_lines,
                "avg_overlap": avg_overlap,
                "avg_repetition_difference": avg_repetition_difference
            }, f, indent=4)






async def evaluate_no_constraints_no_guardrails(folder_path):
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/NoConstraintsNoGuardrails/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/NoConstraintsNoGuardrails/"

    original_results = []

    print("Evaluating No Constraints No Guardrails")
    async for language_model in atqdm(LANGUAGE_MODELS_CHAT):
        original_results_per_model = []
        language_model = AVAILABLE_LMS[language_model].get_name()
        temp_folder_path = folder_path + language_model + "/None" "/json/"
        print(temp_folder_path)
        if await aiofiles.os.path.isdir(temp_folder_path):

            dir_list = await aiofiles.os.listdir(temp_folder_path)
            
            results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path) for file in dir_list])
            for result,file in zip(results, dir_list):
                with open(temp_folder_path + file, "r") as f:
                    data = json.load(f)
                    result["original_parody_settings"] = data

            original_results_per_model.append(results)
        
        original_results.append(original_results_per_model)

    if not await aiofiles.os.path.isdir(dest_folder):
        await aiofiles.os.makedirs(dest_folder)

    with open(dest_folder + "results.json", "w") as f:
        json.dump({
            "results": original_results
        }, f, indent=4)

def process_no_constraints_no_guardrails_results():
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/NoConstraintsNoGuardrails/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/NoConstraintsNoGuardrails/"
    
    data = None
    with open(dest_folder+"results.json", "r") as f:
        data = json.load(f)
    results = data["results"]

    for i in range(len(LANGUAGE_MODELS_CHAT)):
        language_model = LANGUAGE_MODELS_CHAT[i]
        avg_correct_paragraphs = 0
        avg_correct_lines = 0
        avg_perplexities = 0
        avg_perplexities_difference = 0
        avg_syllable_differences = 0
        avg_mean_deviation_syllable_count = 0
        avg_correct_syllable_count = 0
        avg_duration = 0
        avg_correct_rhymes  = 0
        avg_pos_similarity = 0
        avg_mean_deviation_pos_similarity = 0
        avg_correct_pos_lines = 0
        avg_overlap = 0
        avg_repetition_difference = 0
        avg_rhyme_word_length = 0
        result = results[i][0]
        
        correct_paragraphs = []
        correct_lines = []
        perplexities = []
        perplexities_difference = []
        syllable_differences = []
        mean_deviation_syllable_count = []
        correct_syllable_count = []
        duration = []
        correct_rhymes = []
        pos_similarity = []
        mean_deviation_pos_similarity = []
        correct_pos_lines = []
        overlap = []
        repetition_difference = []
        rhyme_word_length = []

        

        for song in result:
            correct_paragraphs.append(song["correct_nb_paragraphs"]/song["original_song_nb_paragraphs"])
            correct_lines.append(song["correct_nb_lines"]/song["original_song_nb_lines"])
            perplexities.append(song["parody_song_perplexity"])
            perplexities_difference.append(song["parody_song_perplexity"] - song["original_song_perplexity"])
            syllable_differences.append(song["avg_syllable_count_difference"])
            mean_deviation_syllable_count.append(song["mean_deviation_syllable_count"])
            correct_syllable_count.append(song["nb_lines_correct_syllable_count"]/song["correct_nb_lines"])
            duration.append(song["original_parody_settings"]["generation_duration"])
            correct_rhymes.append(song["nb_matching_rhyme_pairs"]/song["nb_expected_rhyming_pairs"])
            pos_similarity.append(song["avg_pos_tag_similarity"])
            mean_deviation_pos_similarity.append(song["mean_deviation_pos_tag_similarity"])
            correct_pos_lines.append(song["nb_correct_pos_similarities"]/song["correct_nb_lines"])
            overlap.append(song["overlap"])
            repetition_difference.append(song["original_song_repetition_score"] - song["parody_song_repetition_score"])
            rhyme_word_length.append(song["avg_rhyme_word_length"])

        

        avg_correct_paragraphs = statistics.median(correct_paragraphs)
        avg_correct_lines = statistics.median(correct_lines)
        avg_perplexities = statistics.median(perplexities)
        avg_perplexities_difference = statistics.median(perplexities_difference)
        avg_syllable_differences = statistics.median(syllable_differences)
        avg_mean_deviation_syllable_count = statistics.median(mean_deviation_syllable_count)
        avg_correct_syllable_count = statistics.median(correct_syllable_count)
        avg_duration = statistics.median(duration)
        avg_correct_rhymes = statistics.median(correct_rhymes)
        avg_pos_similarity = statistics.median(pos_similarity)
        avg_mean_deviation_pos_similarity = statistics.median(mean_deviation_pos_similarity)
        avg_correct_pos_lines = statistics.median(correct_pos_lines)
        avg_overlap = statistics.median(overlap)
        avg_repetition_difference = statistics.median(repetition_difference)
        avg_rhyme_word_length = statistics.median(rhyme_word_length)

        if not os.path.isdir(dest_folder+language_model.replace(" ", "_")):
            os.makedirs(dest_folder+language_model.replace(" ", "_"))
        with open(dest_folder+language_model.replace(" ", "_")+"/averages.json", "w") as f:
            json.dump({
                "avg_correct_paragraphs": avg_correct_paragraphs,
                "avg_correct_lines": avg_correct_lines,
                "avg_perplexities": avg_perplexities,
                "avg_perplexities_difference": avg_perplexities_difference,
                "avg_syllable_differences": avg_syllable_differences,
                "avg_mean_deviation_syllable_count": avg_mean_deviation_syllable_count,
                "avg_correct_syllable_count": avg_correct_syllable_count,
                "avg_duration": avg_duration,
                "avg_correct_rhymes": avg_correct_rhymes,
                "avg_rhyme_word_length": avg_rhyme_word_length,
                "avg_pos_similarity": avg_pos_similarity,
                "avg_mean_deviation_pos_similarity": avg_mean_deviation_pos_similarity,
                "avg_correct_pos_lines": avg_correct_pos_lines,
                "avg_overlap": avg_overlap,
                "avg_repetition_difference": avg_repetition_difference
            }, f, indent=4)

async def evaluate_no_constraints_with_guardrails(folder_path):
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/NoConstraintsWithGuardrails/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/NoConstraintsWithGuardrails/"

    original_results = []

    print("Evaluating No Constraints With Guardrails")
    async for language_model in atqdm(LANGUAGE_MODELS_CHAT):
        original_results_per_model = []
        language_model = AVAILABLE_LMS[language_model].get_name()
        temp_folder_path = folder_path + language_model + "/None" "/json/"
        if await aiofiles.os.path.isdir(temp_folder_path):

            dir_list = await aiofiles.os.listdir(temp_folder_path)

            results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path) for file in dir_list])
            for result,file in zip(results, dir_list):
                with open(temp_folder_path + file, "r") as f:
                    data = json.load(f)
                    result["original_parody_settings"] = data

            original_results_per_model.append(results)
        
        original_results.append(original_results_per_model)

    if not await aiofiles.os.path.isdir(dest_folder):
        await aiofiles.os.makedirs(dest_folder)

    with open(dest_folder + "results.json", "w") as f:
        json.dump({
            "results": original_results
        }, f, indent=4)

def process_no_constraints_with_guardrails_results():
    dest_folder = "Experiments/ConstrainedParodieGenerator/CalibrationResults/NoConstraintsWithGuardrails/"
    if platform.system() == 'Linux':
        dest_folder = os.environ["VSC_DATA"] + "/CalibrationResults/NoConstraintsWithGuardrails/"
    
    data = None
    with open(dest_folder+"results.json", "r") as f:
        data = json.load(f)
    results = data["results"]

    for i in range(len(LANGUAGE_MODELS_CHAT)):
        language_model = LANGUAGE_MODELS_CHAT[i]
        avg_correct_paragraphs = 0
        avg_correct_lines = 0
        avg_perplexities = 0
        avg_perplexities_difference = 0
        avg_syllable_differences = 0
        avg_mean_deviation_syllable_count = 0
        avg_correct_syllable_count = 0
        avg_duration = 0
        avg_correct_rhymes  = 0
        avg_pos_similarity = 0
        avg_mean_deviation_pos_similarity = 0
        avg_correct_pos_lines = 0
        avg_overlap = 0
        avg_repetition_difference = 0
        avg_rhyme_word_length = 0
        result = results[i][0]
        
        correct_paragraphs = []
        correct_lines = []
        perplexities = []
        perplexities_difference = []
        syllable_differences = []
        mean_deviation_syllable_count = []
        correct_syllable_count = []
        duration = []
        correct_rhymes = []
        pos_similarity = []
        mean_deviation_pos_similarity = []
        correct_pos_lines = []
        overlap = []
        repetition_difference = []
        rhyme_word_length = []


        

        for song in result:
            correct_paragraphs.append(song["correct_nb_paragraphs"]/song["original_song_nb_paragraphs"])
            correct_lines.append(song["correct_nb_lines"]/song["original_song_nb_lines"])
            perplexities.append(song["parody_song_perplexity"])
            perplexities_difference.append(song["parody_song_perplexity"] - song["original_song_perplexity"])
            syllable_differences.append(song["avg_syllable_count_difference"])
            mean_deviation_syllable_count.append(song["mean_deviation_syllable_count"])
            correct_syllable_count.append(song["nb_lines_correct_syllable_count"]/song["correct_nb_lines"])
            duration.append(song["original_parody_settings"]["generation_duration"])
            correct_rhymes.append(song["nb_matching_rhyme_pairs"]/song["nb_expected_rhyming_pairs"])
            pos_similarity.append(song["avg_pos_tag_similarity"])
            mean_deviation_pos_similarity.append(song["mean_deviation_pos_tag_similarity"])
            correct_pos_lines.append(song["nb_correct_pos_similarities"]/song["correct_nb_lines"])
            overlap.append(song["overlap"])
            repetition_difference.append(song["original_song_repetition_score"] - song["parody_song_repetition_score"])
            rhyme_word_length.append(song["avg_rhyme_word_length"])
        

        avg_correct_paragraphs = statistics.median(correct_paragraphs)
        avg_correct_lines = statistics.median(correct_lines)
        avg_perplexities = statistics.median(perplexities)
        avg_perplexities_difference = statistics.median(perplexities_difference)
        avg_syllable_differences = statistics.median(syllable_differences)
        avg_mean_deviation_syllable_count = statistics.median(mean_deviation_syllable_count)
        avg_correct_syllable_count = statistics.median(correct_syllable_count)
        avg_duration = statistics.median(duration)
        avg_correct_rhymes = statistics.median(correct_rhymes)
        avg_pos_similarity = statistics.median(pos_similarity)
        avg_mean_deviation_pos_similarity = statistics.median(mean_deviation_pos_similarity)
        avg_correct_pos_lines = statistics.median(correct_pos_lines)
        avg_overlap = statistics.median(overlap)
        avg_repetition_difference = statistics.median(repetition_difference)
        avg_rhyme_word_length = statistics.median(rhyme_word_length)

        if not os.path.isdir(dest_folder+language_model.replace(" ", "_")):
            os.makedirs(dest_folder+language_model.replace(" ", "_"))
        with open(dest_folder+language_model.replace(" ", "_")+"/averages.json", "w") as f:
            json.dump({
                "avg_correct_paragraphs": avg_correct_paragraphs,
                "avg_correct_lines": avg_correct_lines,
                "avg_perplexities": avg_perplexities,
                "avg_perplexities_difference": avg_perplexities_difference,
                "avg_syllable_differences": avg_syllable_differences,
                "avg_mean_deviation_syllable_count": avg_mean_deviation_syllable_count,
                "avg_correct_syllable_count": avg_correct_syllable_count,
                "avg_duration": avg_duration,
                "avg_correct_rhymes": avg_correct_rhymes,
                "avg_rhyme_word_length": avg_rhyme_word_length,
                "avg_pos_similarity": avg_pos_similarity,
                "avg_mean_deviation_pos_similarity": avg_mean_deviation_pos_similarity,
                "avg_correct_pos_lines": avg_correct_pos_lines,
                "avg_overlap": avg_overlap,
                "avg_repetition_difference": avg_repetition_difference
            }, f, indent=4)



def test():
    plot_2d_heatmap([1, 2, 3, 4, 5, 6], [0.2, 0.4, 0.6, 0.8, 0.9, 0.99], [[10,3,4,2,3,8],[1,9,5,9,7,9], [4,5,6,7,8,9], [1,2,3,4,5,6], [2,3,4,5,6,7], [3,4,5,6,7,8]], "Prompt Number", "Good Beamscore Multiplier", "Perplexity", "Title", "Experiments/ConstrainedParodieGenerator/CalibrationResults/test/test.png")



def generate(constraint, language_model, song_nb):
    songs = os.listdir(SONG_DIR)
    print(songs)
    beam_index = 0
    song_nb = int(song_nb) -1
    if constraint == 'rhyming' or constraint == 'pos':
        beam_index = song_nb % 6
        song_nb = song_nb // 6
        print(song_nb, beam_index)
    song = songs[song_nb]
    print(len(songs))
    print(song)
    song_file_path = SONG_DIR + song
    prompt_nb = 0
    if constraint == "syllable":
        calibrate_syllable_constraint(song_file_path, prompt_nb, language_model)
    elif constraint == "rhyming":
        calibrate_rhyming_constraint(song_file_path, prompt_nb, language_model, beam_index)
    elif constraint == "pos":
        calibrate_pos_constraint(song_file_path, prompt_nb, language_model, beam_index)
    elif constraint == "prompt":
        calibrate_prompt(song_file_path)
    elif constraint == "rhymin_frequency":
        calibrate_rhymin_frequency(song_file_path)
    elif constraint == "rhyming_types":
        calibrate_rhyming_types(song_file_path)
    elif constraint == "backtracking":
        calibrate_backtracking(song_file_path)
    elif constraint == "all_non_chat":
        generate_all_non_chat(song_file_path)
    elif constraint == "all_chat":
        generate_all_chat(song_file_path)
    elif constraint == "no_constraints_with_guardrails":
        generate_no_constraints_with_guardrails(song_file_path)
    else:
        print("Invalid constraint")
        sys.exit(1)


def evaluate(constraint, language_model, folder_path):
    ray.init(log_to_driver=False)
    try:
        language_model_name = AVAILABLE_LMS[language_model].get_name()
    except KeyError:
        language_model_name = language_model
    if not folder_path.endswith("/"):
        folder_path += "/"
    if constraint == "syllable":
        evaluate_syllable(language_model_name, folder_path)
    elif constraint == "rhyming":
        asyncio.run(evaluate_rhyming(language_model_name, folder_path))
    elif constraint == "pos":
        asyncio.run(evaluate_pos(language_model_name, folder_path))
    elif constraint == "prompt":
        asyncio.run(evaluate_prompt(folder_path))
    elif constraint == "rhymin_frequency":
        asyncio.run(evaluate_rhyming_frequencies(folder_path))
    elif constraint == "rhyming_types":
        asyncio.run(evaluate_rhyming_types(folder_path))
    elif constraint == "backtracking":
        asyncio.run(evaluate_backtracking(folder_path))
    elif constraint == "all_non_chat":
        asyncio.run(evaluate_all_non_chat(folder_path))
    elif constraint == "all_chat":
        asyncio.run(evaluate_all_chat(folder_path))
    elif constraint == "no_constraints_no_guardrails":
        asyncio.run(evaluate_no_constraints_no_guardrails(folder_path))
    elif constraint == "no_constraints_with_guardrails":
        asyncio.run(evaluate_no_constraints_with_guardrails(folder_path))


def process(constraint, language_model):
    try:
        language_model_name = AVAILABLE_LMS[language_model].get_name()
    except KeyError:
        language_model_name = language_model
    if constraint == "syllable":
        process_syllable_results(language_model_name)
    elif constraint == "rhyming":
        process_rhyming_or_pos_results(language_model_name, "rhyming")
    elif constraint == "pos":
        process_rhyming_or_pos_results(language_model_name, "pos")
    elif constraint == "prompt":
        process_prompt_results()
    elif constraint == "rhymin_frequency":
        process_rhyming_frequencies_results()
    elif constraint == "rhyming_types":
        process_rhyming_types_results()
    elif constraint == "backtracking":
        process_backtracking_results()
    elif constraint == "all_non_chat":
        process_all_non_chat_results()
    elif constraint == "all_chat":
        process_all_chat_results()
    elif constraint == "no_constraints_no_guardrails":
        process_no_constraints_no_guardrails_results()
    elif constraint == "no_constraints_with_guardrails":
        process_no_constraints_with_guardrails_results()
    else:
        print("Invalid constraint")
        sys.exit(1)





slurm_jobs ="""
################# Generate #################
#### prompt ####

#!/bin/bash -l
#SBATCH --time=08:00:00
module load cluster/wice/gpu_h100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"
source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py generate prompt None ${SLURM_ARRAY_TASK_ID}

#### rhyming frequency ####

#!/bin/bash -l
#SBATCH --time=05:45:00
module load cluster/wice/gpu_h100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"
source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py generate rhymin_frequency None ${SLURM_ARRAY_TASK_ID}

#### rhyming types ####

#!/bin/bash -l
#SBATCH --time=08:00:00
module load cluster/wice/gpu_h100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"
source ThesisEnv/bin/activate
cd Thesis

python Experiments/ConstrainedParodieGenerator/Calibrator.py generate rhyming_types None ${SLURM_ARRAY_TASK_ID}

#### backtracking ####

#!/bin/bash -l
#SBATCH --time=08:00:00
module load cluster/wice/gpu_h100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"
source ThesisEnv/bin/activate
cd Thesis

python Experiments/ConstrainedParodieGenerator/Calibrator.py generate backtracking None ${SLURM_ARRAY_TASK_ID}

#### all non chat ####

#!/bin/bash -l
#SBATCH --time=08:00:00
module load cluster/wice/gpu_h100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py generate all_non_chat None ${SLURM_ARRAY_TASK_ID}

#### all chat ####

#!/bin/bash -l
#SBATCH --time=05:45:00
module load cluster/wice/gpu_h100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py generate all_chat None ${SLURM_ARRAY_TASK_ID}

#### no constraints with guardrails ####

#!/bin/bash -l
#SBATCH --time=03:00:00
module load cluster/wice/gpu_h100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py generate no_constraints_with_guardrails None ${SLURM_ARRAY_TASK_ID}

################# Evaluate #################

#### syllable Llama2_7BChat ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate syllable Llama2_7BChat Experiments/ConstrainedParodieGenerator/CallibrationExperiments/SyllableConstraint/0

#### syllable Llama2_70BChat ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate syllable Llama2_70BChat Experiments/ConstrainedParodieGenerator/CallibrationExperiments/SyllableConstraint/0

#### syllable Mistral7BItV02 ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate syllable Mistral7BItV02 Experiments/ConstrainedParodieGenerator/CallibrationExperiments/SyllableConstraint/0

#### syllable Mistral8x7BItV01 ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate syllable Mistral8x7BItV01 Experiments/ConstrainedParodieGenerator/CallibrationExperiments/SyllableConstraint/0

#### rhyming Llama2_7BChat ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate rhyming Llama2_7BChat Experiments/ConstrainedParodieGenerator/CallibrationExperiments/RhymingConstraint/0

#### rhyming Llama2_70BChat ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate rhyming Llama2_70BChat Experiments/ConstrainedParodieGenerator/CallibrationExperiments/RhymingConstraint/0

#### rhyming Mistral7BItV02 ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate rhyming Mistral7BItV02 Experiments/ConstrainedParodieGenerator/CallibrationExperiments/RhymingConstraint/0

#### rhyming Mistral8x7BItV01 ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate rhyming Mistral8x7BItV01 Experiments/ConstrainedParodieGenerator/CallibrationExperiments/RhymingConstraint/0

#### pos Llama2_7BChat ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate pos Llama2_7BChat Experiments/ConstrainedParodieGenerator/CallibrationExperiments/PosConstraint/0

#### pos Llama2_70BChat ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate pos Llama2_70BChat Experiments/ConstrainedParodieGenerator/CallibrationExperiments/PosConstraint/0

#### pos Mistral7BItV02 ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate pos Mistral7BItV02 Experiments/ConstrainedParodieGenerator/CallibrationExperiments/PosConstraint/0

#### pos Mistral8x7BItV01 ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate pos Mistral8x7BItV01 Experiments/ConstrainedParodieGenerator/CallibrationExperiments/PosConstraint/0

#### prompt ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate prompt None Experiments/ConstrainedParodieGenerator/CallibrationExperiments/PromptCalibration

#### rhymin frequency ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate rhymin_frequency None Experiments/ConstrainedParodieGenerator/CallibrationExperiments/RhymingFrequency

#### rhyming types ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate rhyming_types None Experiments/ConstrainedParodieGenerator/CallibrationExperiments/RhymingTypes

#### backtracking ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate backtracking None Experiments/ConstrainedParodieGenerator/CallibrationExperiments/Backtracking

#### all non chat ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate all_non_chat None Experiments/ConstrainedParodieGenerator/CallibrationExperiments/AllNonChat/1

#### all chat ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate all_chat None Experiments/ConstrainedParodieGenerator/CallibrationExperiments/AllChat/1

#### no constraints with guardrails ####

#!/bin/bash -l
#SBATCH --time=02:30:00

module load cluster/genius/gpu_p100
module load Python/3.10.8-GCCcore-12.2.0

cd $VSC_SCRATCH

export HF_HOME="/scratch/leuven/361/vsc36141/HF/"

source ThesisEnv/bin/activate
cd Thesis
python Experiments/ConstrainedParodieGenerator/Calibrator.py evaluate no_constraints_with_guardrails None Experiments/ConstrainedParodieGenerator/CallibrationExperiments/NoConstraintsWithGuardrails/1




"""

def create_slurm_jobs():
    #split the slurmjobs string per section and for each section make the corresponding slurm file
    #sections = re.split(r'#+\s*.*?\s*#+', slurm_jobs)
    section_pattern = r'#################\s*.*?\s*#################\n'

    start_folder = 'Slurm_jobs/'

    # Using re.findall to extract pairs of (header, content)
    heads = re.findall(section_pattern, slurm_jobs, re.DOTALL)
    heads = [match[:-2].strip('# ') for match in heads]
    sections = re.split(section_pattern, slurm_jobs)
    sections = [section.strip() for section in sections if section.strip() != ""]
    assert len(heads) == len(sections)
    
    for i in range(len(sections)):
        job_pattern = r'####\s*.*?\s*####\n'
        matches = re.findall(job_pattern, sections[i], re.DOTALL)
        job_heads = [match[:-2].strip('# ') for match in matches]

        job_sections = re.split(job_pattern, sections[i])
        job_sections = [section.strip() for section in job_sections if section.strip() != ""]

        assert len(job_heads) == len(job_sections)

        for j in range(len(job_heads)):
            if not os.path.isdir(start_folder+ heads[i] + "/"):
                os.makedirs(start_folder + heads[i] + "/")
            with open(start_folder + heads[i] + "/" + job_heads[j].replace(' ', '_') + ".slurm", "w") as f:
                f.write(job_sections[j])
    


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 Calibrator.py <mode> \n mode = generate/evaluate ")
        sys.exit(1)
    
    mode = sys.argv[1]

    if mode == "generate":
        if len(sys.argv) < 5:
            print("Usage: python3 Calibrator.py generate <constraint> <language_model> <song>\n" + "<constraint> <language_model> <song>\n <constraint> = syllable/rhyming/pos\n <language_model> = " + str(AVAILABLE_LMS.keys()) + "\n <song> a number between 1 and 20")
            sys.exit(1)
        constraint = sys.argv[2]
        language_model = sys.argv[3]
        song_nb = sys.argv[4]
        generate(constraint, language_model, song_nb)
    elif mode == "evaluate":
        if len(sys.argv) < 5:
            print("Usage: python3 Calibrator.py evaluate <constraint> <language_model> <folder_path>")
            sys.exit(1)
        
        constraint = sys.argv[2]
        language_model = sys.argv[3]
        folder_path = sys.argv[4]
        evaluate(constraint, language_model, folder_path)
    elif mode == "process":
        if len(sys.argv) < 4:
            print("Usage: python3 Calibrator.py process <constraint> <language_model>")
            sys.exit(1)
        
        constraint = sys.argv[2]
        language_model = sys.argv[3]
        process(constraint, language_model)

    elif mode == "test":
        test()

    elif mode == "create_slurm_jobs":
        create_slurm_jobs()
    else:
        print("Usage: python3 Calibrator.py <mode> \n mode = generate/evaluate ")
        sys.exit(1)