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


## Constants
GLOBAL_SEED = 42
POSSIBLE_NUM_BEAMS  = [2,4,8,12,16]
SONG_DIR = "Songs/json/"
SYSTEM_PROMPTS = []
CONTEXT_PROMPTS = []
ASSISTANT_PROMPTS = []

## Init parameters
random.seed(GLOBAL_SEED)
    



def calibrate_syllable_constraint(song_file_path, prompt_nb, language_model):
    possible_good_beamscore_multipliers_syllable = [0,1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    possible_bad_beamscore_multipliers_syllable = [0, 2, 4, 6, 8, 10],
    possible_top_k_tokens_to_consider = [5, 10, 30, 50, 100, 200],
    possible_all_beams_have_syllable_amount = [True, False]

    folder_path_for_generated_parodies = "Experiments/ConstrainedParodieGenerator/CallibrationExperiments/" + "SyllableConstraint/" + str(prompt_nb)+"/"
    if not os.path.exists(folder_path_for_generated_parodies):
        os.makedirs(folder_path_for_generated_parodies)

    
    system_prompt = SYSTEM_PROMPTS[prompt_nb]
    context_prompt = CONTEXT_PROMPTS[prompt_nb]
    assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]


    for num_beams in POSSIBLE_NUM_BEAMS:
        for good_beamscore_multiplier_syllable in possible_good_beamscore_multipliers_syllable:
            for bad_beamscore_multiplier_syllable in possible_bad_beamscore_multipliers_syllable:
                for top_k_tokens_to_consider in possible_top_k_tokens_to_consider:
                    for all_beams_have_syllable_amount in possible_all_beams_have_syllable_amount:
                        
                        syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(good_beamscore_multiplier=good_beamscore_multiplier_syllable, bad_beamscore_multiplier=bad_beamscore_multiplier_syllable, top_k_tokens_to_consider=top_k_tokens_to_consider, all_beams_have_syllable_amount=all_beams_have_syllable_amount)

                        generate_parody(
                        song_file_path= song_file_path, 
                        system_prompt = system_prompt, 
                        context_prompt = context_prompt, 
                        assistant_prompt = assistant_prompt,
                        language_model = language_model,
                        folder_path_for_generated_parodies = folder_path_for_generated_parodies,
                        use_cuda=True,
                        use_quantization=True,
                        do_sample=True, 
                        top_p=0.95, 
                        temperature=0.7, 
                        num_beams=num_beams, 
                        seed=GLOBAL_SEED, 
                        syllable_constrained = True,
                        rhyming_constrained = False,
                        pos_constrained = False,
                        syllable_constraint_hyperparameters=syllable_constraint_hyperparameters
                        )



def calibrate_rhyming_constraint(song_file_path, prompt_nb, language_model):
    possible_rhyme_types = ['assonant', 'perfect', 'near']
    top_k_rhyme_words = [2,5,10,20]
    good_beamscore_multipliers_rhyme = [0,1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    good_rhyming_token_multipliers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    max_possible_syllable_counts = [1,2,3]

    syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.5, bad_beamscore_multiplier=5, top_k_tokens_to_consider=30, all_beams_have_syllable_amount=False)

    folder_path_for_generated_parodies = "Experiments/ConstrainedParodieGenerator/CallibrationExperiments/" + "RhymingConstraint/" + str(prompt_nb)+"/"

    if not os.path.exists(folder_path_for_generated_parodies):
        os.makedirs(folder_path_for_generated_parodies)
    
    system_prompt = SYSTEM_PROMPTS[prompt_nb]
    context_prompt = CONTEXT_PROMPTS[prompt_nb]
    assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]

    for num_beams in POSSIBLE_NUM_BEAMS:
        for rhyme_type in possible_rhyme_types:
            for top_k_rhyme_word in top_k_rhyme_words:
                for good_beamscore_multiplier_rhyme in good_beamscore_multipliers_rhyme:
                    for good_rhyming_token_multiplier in good_rhyming_token_multipliers:
                        for max_possible_syllable_count in max_possible_syllable_counts:
                                rhyming_constraint_hyperparameters = RhymingConstraintLBL.hyperparameters_config(max_possible_syllable_count= max_possible_syllable_count, good_beamscore_multiplier_same_rhyme_type=good_beamscore_multiplier_rhyme, good_rhyming_token_multiplier=good_rhyming_token_multiplier, top_k_rhyme_words=top_k_rhyme_word, rhyme_type=rhyme_type)

                                generate_parody(
                                song_file_path= song_file_path, 
                                system_prompt = system_prompt, 
                                context_prompt = context_prompt, 
                                assistant_prompt = assistant_prompt,
                                language_model = language_model,
                                folder_path_for_generated_parodies = folder_path_for_generated_parodies,
                                use_cuda=True,
                                use_quantization=True,
                                do_sample=True, 
                                top_p=0.95, 
                                temperature=0.7, 
                                num_beams=num_beams, 
                                seed=GLOBAL_SEED, 
                                syllable_constrained = False,
                                rhyming_constrained = True,
                                pos_constrained = False,
                                rhyme_constraint_hyperparameters=rhyme_constraint_hyperparameters
                                )

def calibrate_pos_constraint(song_file_path, prompt_nb, language_model):
    top_k_tokens_to_consider_for_pos = [100,200, 500,1000,2000,5000]
    good_beamscore_multipliers_pos = [0,1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    good_token_multipliers = [0,1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    limits_of_pos_similarity_to_satisfy_constraint = [0,1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(good_beamscore_multiplier=0.5, bad_beamscore_multiplier=5, top_k_tokens_to_consider=30, all_beams_have_syllable_amount=False)

    folder_path_for_generated_parodies = "Experiments/ConstrainedParodieGenerator/CallibrationExperiments/" + "PosConstraint/" + str(prompt_nb)+"/"

    if not os.path.exists(folder_path_for_generated_parodies):
        os.makedirs(folder_path_for_generated_parodies)

    system_prompt = SYSTEM_PROMPTS[prompt_nb]
    context_prompt = CONTEXT_PROMPTS[prompt_nb]
    assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]

    for num_beams in POSSIBLE_NUM_BEAMS:
        for top_k_tokens_to_consider in top_k_tokens_to_consider_for_pos:
            for good_beamscore_multiplier_pos in good_beamscore_multipliers_pos:
                for good_token_multiplier in good_token_multipliers:
                    for limit_of_pos_similarity_to_satisfy_constraint in limits_of_pos_similarity_to_satisfy_constraint:
                        pos_constraint_hyperparameters = PosConstraintLBL.hyperparameters_config(good_beamscore_multiplier=good_beamscore_multiplier_pos, good_token_multiplier=good_token_multiplier, limit_of_pos_similarity_to_satisfy_constraint=limit_of_pos_similarity_to_satisfy_constraint, top_k_tokens_to_consider=top_k_tokens_to_consider)


                        generate_parody(
                        song_file_path= song_file_path, 
                        system_prompt = system_prompt, 
                        context_prompt = context_prompt, 
                        assistant_prompt = assistant_prompt,
                        language_model = language_model,
                        folder_path_for_generated_parodies = folder_path_for_generated_parodies,
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
                        syllable_constraint_hyperparameters=syllable_constraint_hyperparameters
                        )






if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 Calibrator.py <mode> <constraint> <language_model>\n <mode> = generate/evaluate\n <constraint> = syllable/rhyming/pos\n <language_model> = " + str(AVAILABLE_LMS.keys()) + "\n")
        sys.exit(1)
    

    mode = sys.argv[1]
    
    constraint = sys.argv[2]

    language_model = sys.argv[3]

    if mode == "generate":
    
        songs = os.listdir(SONG_DIR)
        for song in songs:
            song_file_path = SONG_DIR + song
            for prompt_nb in range(len(SYSTEM_PROMPTS)):
                if constraint == "syllable":
                    calibrate_syllable_constraint(song_file_path, prompt_nb, language_model)
                elif constraint == "rhyming":
                    calibrate_rhyming_constraint(song_file_path, prompt_nb, language_model)
                elif constraint == "pos":
                    calibrate_pos_constraint(song_file_path, prompt_nb, language_model)
    elif mode == "evaluate":
        pass





        
    


    

# def divide_into_training_and_testing_sets(songs, training_ratio=0.8):
#     random.shuffle(songs)
#     num_songs = len(songs)
#     training_set = songs[:int(training_ratio*num_songs)]
#     testing_set = songs[int(training_ratio*num_songs):]
#     return training_set, testing_set


# ### Evaluate
# # Plotting
# def plotting():
#     plt.figure(figsize=(10, 6))
#     plt.plot(parameter_values, performance_metrics, marker='o', linestyle='-', color='b')
#     plt.title('Algorithm Performance vs. Parameter Value')
#     plt.xlabel('Parameter Value')
#     plt.ylabel('Performance Metric')
#     plt.grid(True)
#     plt.savefig('Experiments/img.png', dpi=300)





