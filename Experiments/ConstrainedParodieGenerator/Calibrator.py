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

## Constants
GLOBAL_SEED = 42
POSSIBLE_NUM_BEAMS  = [5]
SONG_DIR = "Songs/json/"
SYSTEM_PROMPTS = ["Experiments/ConstrainedParodieGenerator/PromptTexts/1/system_prompt.txt", "Experiments/ConstrainedParodieGenerator/PromptTexts/2/system_prompt.txt", "Experiments/ConstrainedParodieGenerator/PromptTexts/3/system_prompt.txt"]
CONTEXT_PROMPTS = ["Experiments/ConstrainedParodieGenerator/PromptTexts/1/context_prompt.txt", "Experiments/ConstrainedParodieGenerator/PromptTexts/2/context_prompt.txt", "Experiments/ConstrainedParodieGenerator/PromptTexts/3/context_prompt.txt"]
ASSISTANT_PROMPTS = ["Experiments/ConstrainedParodieGenerator/PromptTexts/1/assistant_prompt.txt", "Experiments/ConstrainedParodieGenerator/PromptTexts/2/assistant_prompt.txt", "Experiments/ConstrainedParodieGenerator/PromptTexts/3/assistant_prompt.txt"]

## Init parameters
random.seed(GLOBAL_SEED)
    

START_FOLDER = None

if platform.system() == 'Linux':
    START_FOLDER = os.environ["VSC_DATA"] + "/CallibrationExperiments/"
else:
    START_FOLDER = "Experiments/ConstrainedParodieGenerator/CallibrationExperiments/"

@ray.remote(num_gpus=1, max_calls=1)
def generate_parody_with_ray(**kwargs):
    return generate_parody(**kwargs)

def generate_parody_without_ray(**kwargs):
    return generate_parody(**kwargs)

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

async def calibrate_pos_constraint_async(song_file_path, prompt_nb, language_model, beam_index):
    top_k_tokens_to_consider_for_pos = [200]
    good_beamscore_multipliers_pos = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    good_token_multipliers = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    limits_of_pos_similarity_to_satisfy_constraint = [0.5]

    syllable_constraint_hyperparameters = SyllableConstraintLBL.hyperparameters_config(
        good_beamscore_multiplier=0.9, top_k_tokens_to_consider=200, all_beams_have_syllable_amount=False
    )

    folder_path_for_generated_parodies = START_FOLDER + "PosConstraint_async/" + str(prompt_nb) + "/"

    if not await aiofiles.os.path.exists(folder_path_for_generated_parodies):
        await aiofiles.os.makedirs(folder_path_for_generated_parodies)

    system_prompt = SYSTEM_PROMPTS[prompt_nb]
    context_prompt = CONTEXT_PROMPTS[prompt_nb]
    assistant_prompt = ASSISTANT_PROMPTS[prompt_nb]

    index = 1 + beam_index * 6

    executor = ThreadPoolExecutor()
    loop = asyncio.get_running_loop()

    tasks = []
    for num_beams in POSSIBLE_NUM_BEAMS:
        for top_k_tokens_to_consider in top_k_tokens_to_consider_for_pos:
            for good_token_multiplier in good_token_multipliers:
                for limit_of_pos_similarity_to_satisfy_constraint in limits_of_pos_similarity_to_satisfy_constraint:
                    good_beamscore_multiplier_pos = good_beamscore_multipliers_pos[beam_index]
                    pos_constraint_hyperparameters = PosConstraintLBL.hyperparameters_config(
                        good_beamscore_multiplier=good_beamscore_multiplier_pos,
                        good_token_multiplier=good_token_multiplier,
                        limit_of_pos_similarity_to_satisfy_constraint=limit_of_pos_similarity_to_satisfy_constraint,
                        top_k_tokens_to_consider=top_k_tokens_to_consider
                    )

                    parody_path = folder_path_for_generated_parodies + str(index) + "/"

                    task = loop.run_in_executor(
                        executor,
                        ray.get,
                        generate_parody_with_ray.remote(
                            song_file_path=song_file_path,
                            system_prompt=system_prompt,
                            context_prompt=context_prompt,
                            assistant_prompt=assistant_prompt,
                            language_model=language_model,
                            folder_path_for_generated_parodies=parody_path,
                            use_cuda=True,
                            use_quantization=True,
                            do_sample=True,
                            top_p=0.95,
                            temperature=0.7,
                            num_beams=num_beams,
                            seed=GLOBAL_SEED,
                            syllable_constrained=True,
                            rhyming_constrained=False,
                            pos_constrained=True,
                            pos_constraint_hyperparameters=pos_constraint_hyperparameters,
                            syllable_constraint_hyperparameters=syllable_constraint_hyperparameters,
                            use_backtracking=False
                        )
                    )

                    tasks.append(task)
                    index += 1

    # Await all tasks to complete
    await asyncio.gather(*tasks)



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
    elif constraint == "pos_async":
        asyncio.run(calibrate_pos_constraint_async(song_file_path, prompt_nb, language_model, beam_index))


def evaluate(constraint, language_model, folder_path):
    ray.init(log_to_driver=False)
    language_model_name = AVAILABLE_LMS[language_model].get_name()
    if not folder_path.endswith("/"):
        folder_path += "/"
    if constraint == "syllable":
        evaluate_syllable(language_model_name, folder_path)
    elif constraint == "rhyming":
        asyncio.run(evaluate_rhyming(language_model_name, folder_path))
    elif constraint == "pos":
        constraint_folder_path = "Syllable_Constraint_|_POS_Constraint_|_"
    elif constraint == "all":
        constraint_folder_path = "Syllable_Constraint_|_Rhyming_Constraint_|_POS Constraint_|_"

        

@ray.remote
def calculate_song_evaluation(file, folder_path):
    results = evaluate_song(folder_path + file)
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
    sns.heatmap(z_grid, cmap='coolwarm', cbar_kws={'label': zlabel}, annot=True, xticklabels=x_data, yticklabels=y_data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    folder_path = "/".join(file_name.split("/")[:-1])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save the plot
    plt.savefig(file_name, dpi=300)
    



def evaluate_syllable(language_model_name, folder_path):
    possible_good_beamscore_multipliers_syllable = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]

    avg_perplexities = []
    avg_syllable_differences = []
    avg_mean_deviation_syllable_count = []
    avg_correct_syllable_count = []

    print("Evaluating Syllable Constraint for " + language_model_name)
    constraint_folder_path = "Syllable_Constraint_|_/"
    for index in tqdm(range(len(possible_good_beamscore_multipliers_syllable))):
        temp_folder_path = folder_path + str(index + 1) + "/" + language_model_name + "/" + constraint_folder_path +"/json/"
        if os.path.isdir(temp_folder_path):
            perplexities = []
            syllable_differences = []
            mean_deviation_syllable_count = []
            correct_syllale_count = []

            if len(os.listdir(temp_folder_path)) != 20:
                raise Exception("Not all songs have been generated only " + str(len(os.listdir(temp_folder_path))))

            results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path) for file in os.listdir(temp_folder_path)])
            for result in results:
                perplexities.append(result["parody_song_perplexity"])
                syllable_differences.append(result["avg_syllable_count_difference"])
                mean_deviation_syllable_count.append(result["mean_deviation_syllable_count"])
                correct_syllale_count.append(result["nb_lines_correct_syllable_count"]/result["correct_nb_lines"])

            avg_perplexities.append(sum(perplexities)/len(perplexities))
            avg_syllable_differences.append(sum(syllable_differences)/len(syllable_differences))
            avg_mean_deviation_syllable_count.append(sum(mean_deviation_syllable_count)/len(mean_deviation_syllable_count))
            avg_correct_syllable_count.append(sum(correct_syllale_count)/len(correct_syllale_count))
    print("Perplexities: ", avg_perplexities)
    print("Syllable Differences: ", avg_syllable_differences)
    print("Mean Deviation Syllable Count: ", avg_mean_deviation_syllable_count)
    print("Correct Syllable Count: ", avg_correct_syllable_count)

    if os.path.isdir("Experiments/ConstrainedParodieGenerator/CalibrationResults/SyllableConstraint/") == False:
        os.makedirs("Experiments/ConstrainedParodieGenerator/CalibrationResults/SyllableConstraint/")
    
    if os.path.isdir("Experiments/ConstrainedParodieGenerator/CalibrationResults/SyllableConstraint/"+language_model_name.replace(" ", "_")) == False:
        os.makedirs("Experiments/ConstrainedParodieGenerator/CalibrationResults/SyllableConstraint/"+language_model_name.replace(" ", "_"))
    
    #save results 
    with open("Experiments/ConstrainedParodieGenerator/CalibrationResults/SyllableConstraint/"+language_model_name.replace(" ", "_")+"/results.json", "w") as f:
        json.dump({
            "good_beamscore_multipliers_syllable": possible_good_beamscore_multipliers_syllable,
            "avg_perplexities": avg_perplexities,
            "avg_syllable_differences": avg_syllable_differences,
            "avg_mean_deviation_syllable_count": avg_mean_deviation_syllable_count,
            "avg_correct_syllable_count": avg_correct_syllable_count
        }, f, indent=4)

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_perplexities,
        'Good Beamscore Multiplier',
        'Perplexity',
        'Perplexity vs. Good Beamscore Multiplier',
        'Experiments/ConstrainedParodieGenerator/CalibrationResults/SyllableConstraint/'+language_model_name.replace(" ", '_')+'/perplexity.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_syllable_differences,
        'Good Beamscore Multiplier',
        'Syllable Differences',
        'Syllable Differences vs. Good Beamscore Multiplier',
        'Experiments/ConstrainedParodieGenerator/CalibrationResults/SyllableConstraint/'+language_model_name.replace(" ", '_')+'/syllable_differences.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_mean_deviation_syllable_count,
        'Good Beamscore Multiplier',
        'Mean Deviation Syllable Count',
        'Mean Deviation Syllable Count vs. Good Beamscore Multiplier',
        'Experiments/ConstrainedParodieGenerator/CalibrationResults/SyllableConstraint/'+language_model_name.replace(" ", '_')+'/mean_deviation_syllable_count.png'
    )

    plot_results(
        possible_good_beamscore_multipliers_syllable,
        avg_correct_syllable_count,
        'Good Beamscore Multiplier',
        'Correct Syllable Count',
        'Correct Syllable Count vs. Good Beamscore Multiplier',
        'Experiments/ConstrainedParodieGenerator/CalibrationResults/SyllableConstraint/'+language_model_name.replace(" ", "_")+'/correct_syllable_count.png'
    )

# def evaluate_rhyming(language_model_name, folder_path):
#     rhyming_folder = 'Experiments/ConstrainedParodieGenerator/CalibrationResults/RhymingConstraint/'

    
#     if platform.system() == 'Linux':
#         rhyming_folder = os.environ["VSC_DATA"] + "/CallibrationExperiments/RhymingConstraint/"



#     possible_good_beamscore_multipliers_rhyme = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
#     good_rhyming_token_multipliers = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
#     possible_rhyme_types = ['perfect']
#     top_k_rhyme_words = [10]
#     max_possible_syllable_counts = [3]

#     avg_perplexities = []
#     avg_correct_rhyme = []
#     avg_syllable_differences = []
#     avg_mean_deviation_syllable_count = []
#     avg_correct_syllable_count = []

#     print("Evaluating Rhyming Constraint for " + language_model_name)
#     constraint_folder_path = "Syllable_Constraint_|_Rhyming_Constraint_|_/"
#     for index_beam in tqdm(range(len(possible_good_beamscore_multipliers_rhyme))):
#         avg_perplexities_per_beam = []
#         avg_correct_rhyme_per_beam = []
#         avg_syllable_differences_per_beam = []
#         avg_mean_deviation_syllable_count_per_beam = []
#         avg_correct_syllable_count_per_beam = []

#         for index_token in tqdm(range(len(good_rhyming_token_multipliers))):
#             temp_folder_path = folder_path + str(index_beam + index_token + 1) + "/" + language_model_name + "/" + constraint_folder_path +"/json/"
#             if os.path.isdir(temp_folder_path):
#                 perplexities = []
#                 correct_rhyme = []
#                 syllable_differences = []
#                 mean_deviation_syllable_count = []
#                 correct_syllale_count = []


#                 if len(os.listdir(temp_folder_path)) != 20:
#                     raise Exception("Not all songs have been generated only " + str(len(os.listdir(temp_folder_path))))

#                 results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path) for file in os.listdir(temp_folder_path)])
#                 for result in results:
#                     perplexities.append(result["parody_song_perplexity"])
#                     correct_rhyme.append(result["nb_matching_rhyme_pairs"]/result["nb_expected_rhyming_pairs"])
#                     syllable_differences.append(result["avg_syllable_count_difference"])
#                     mean_deviation_syllable_count.append(result["mean_deviation_syllable_count"])
#                     correct_syllale_count.append(result["nb_lines_correct_syllable_count"]/result["correct_nb_lines"])

#                 avg_perplexities_per_beam.append(sum(perplexities)/len(perplexities))
#                 avg_correct_rhyme_per_beam.append(sum(correct_rhyme)/len(correct_rhyme))
#                 avg_syllable_differences_per_beam.append(sum(syllable_differences)/len(syllable_differences))
#                 avg_mean_deviation_syllable_count_per_beam.append(sum(mean_deviation_syllable_count)/len(mean_deviation_syllable_count))
#                 avg_correct_syllable_count_per_beam.append(sum(correct_syllale_count)/len(correct_syllale_count))

#         avg_perplexities.append(avg_perplexities_per_beam)
#         avg_correct_rhyme.append(avg_correct_rhyme_per_beam)
#         avg_syllable_differences.append(avg_syllable_differences_per_beam)
#         avg_mean_deviation_syllable_count.append(avg_mean_deviation_syllable_count_per_beam)
#         avg_correct_syllable_count.append(avg_correct_syllable_count_per_beam)

#     print("Perplexities: ", avg_perplexities)
#     print("Correct Rhyme: ", avg_correct_rhyme)
#     print("Syllable Differences: ", avg_syllable_differences)
#     print("Mean Deviation Syllable Count: ", avg_mean_deviation_syllable_count)
#     print("Correct Syllable Count: ", avg_correct_syllable_count)

#     if os.path.isdir(rhyming_folder) == False:
#         os.makedirs(rhyming_folder)
    
#     if os.path.isdir(rhyming_folder+language_model_name.replace(" ", "_")) == False:
#         os.makedirs(rhyming_folder+language_model_name.replace(" ", "_"))
    
#     #save results
#     with open(rhyming_folder+language_model_name.replace(" ", "_")+"/results.json", "w") as f:
#         json.dump({
#             "good_beamscore_multipliers_rhyme": possible_good_beamscore_multipliers_rhyme,
#             "good_rhyming_token_multipliers": good_rhyming_token_multipliers,
#             "avg_perplexities": avg_perplexities,
#             "avg_correct_rhyme": avg_correct_rhyme,
#             "avg_syllable_differences": avg_syllable_differences,
#             "avg_mean_deviation_syllable_count": avg_mean_deviation_syllable_count,
#             "avg_correct_syllable_count": avg_correct_syllable_count
#         }, f, indent=4)
    
#     plot_2d_heatmap(
#         possible_good_beamscore_multipliers_rhyme,
#         good_rhyming_token_multipliers,
#         avg_perplexities,
#         'Good Beamscore Multiplier',
#         'Good Rhyming Token Multiplier',
#         'Perplexity',
#         'Perplexity vs. Good Beamscore Multiplier and Good Rhyming Token Multiplier',
#         rhyming_folder+language_model_name.replace(" ", "_")+'/perplexity.png'
#     )

#     plot_2d_heatmap(
#         possible_good_beamscore_multipliers_rhyme,
#         good_rhyming_token_multipliers,
#         avg_correct_rhyme,
#         'Good Beamscore Multiplier',
#         'Good Rhyming Token Multiplier',
#         'Correct Rhyme',
#         'Correct Rhyme vs. Good Beamscore Multiplier and Good Rhyming Token Multiplier',
#         rhyming_folder+language_model_name.replace(" ", "_")+'/correct_rhyme.png'
#     )

#     plot_2d_heatmap(
#         possible_good_beamscore_multipliers_rhyme,
#         good_rhyming_token_multipliers,
#         avg_syllable_differences,
#         'Good Beamscore Multiplier',
#         'Good Rhyming Token Multiplier',
#         'Syllable Differences',
#         'Syllable Differences vs. Good Beamscore Multiplier and Good Rhyming Token Multiplier',
#         rhyming_folder+language_model_name.replace(" ", "_")+'/syllable_differences.png'
#     )

#     plot_2d_heatmap(
#         possible_good_beamscore_multipliers_rhyme,
#         good_rhyming_token_multipliers,
#         avg_mean_deviation_syllable_count,
#         'Good Beamscore Multiplier',
#         'Good Rhyming Token Multiplier',
#         'Mean Deviation Syllable Count',
#         'Mean Deviation Syllable Count vs. Good Beamscore Multiplier and Good Rhyming Token Multiplier',
#         rhyming_folder+language_model_name.replace(" ", "_")+'/mean_deviation_syllable_count.png'
#     )

#     plot_2d_heatmap(
#         possible_good_beamscore_multipliers_rhyme,
#         good_rhyming_token_multipliers,
#         avg_correct_syllable_count,
#         'Good Beamscore Multiplier',
#         'Good Rhyming Token Multiplier',
#         'Correct Syllable Count',
#         'Correct Syllable Count vs. Good Beamscore Multiplier and Good Rhyming Token Multiplier',
#         rhyming_folder+language_model_name.replace(" ", "_")+'/correct_syllable_count.png'
#     )

async def evaluate_rhyming(language_model_name, folder_path):
    rhyming_folder = 'Experiments/ConstrainedParodieGenerator/CalibrationResults/RhymingConstraint/'

    if platform.system() == 'Linux':
        rhyming_folder = os.environ["VSC_DATA"] + "/CalibrationResults/RhymingConstraint/"

    possible_good_beamscore_multipliers_rhyme = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    good_rhyming_token_multipliers = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
    possible_rhyme_types = ['perfect']
    top_k_rhyme_words = [10]
    max_possible_syllable_counts = [3]

    avg_perplexities = []
    avg_correct_rhyme = []
    avg_syllable_differences = []
    avg_mean_deviation_syllable_count = []
    avg_correct_syllable_count = []

    original_perplexities = []
    original_correct_rhyme = []
    original_syllable_differences = []
    original_mean_deviation_syllable_count = []
    original_correct_syllable_count = []

    print("Evaluating Rhyming Constraint for " + language_model_name)
    constraint_folder_path = "Syllable_Constraint_|_Rhyming_Constraint_|_/"
    async for index_beam in atqdm(range(len(possible_good_beamscore_multipliers_rhyme))):
        avg_perplexities_per_beam = []
        avg_correct_rhyme_per_beam = []
        avg_syllable_differences_per_beam = []
        avg_mean_deviation_syllable_count_per_beam = []
        avg_correct_syllable_count_per_beam = []
        original_perplexities_per_beam = []
        original_correct_rhyme_per_beam = []
        original_syllable_differences_per_beam = []
        original_mean_deviation_syllable_count_per_beam = []
        original_correct_syllable_count_per_beam = []



        async for index_token in atqdm(range(len(good_rhyming_token_multipliers))):
            temp_folder_path = folder_path + str(index_beam*6 + index_token + 1) + "/" + language_model_name + "/" + constraint_folder_path +"/json/"
            if await aiofiles.os.path.isdir(temp_folder_path):
                perplexities = []
                correct_rhyme = []
                syllable_differences = []
                mean_deviation_syllable_count = []
                correct_syllale_count = []

                dir_list = await aiofiles.os.listdir(temp_folder_path)
                if len(dir_list) != 20:
                    raise Exception("Not all songs have been generated only " + str(len(dir_list)))

                results = ray.get([calculate_song_evaluation.remote(file, temp_folder_path) for file in dir_list])
                for result in results:
                    perplexities.append(result["parody_song_perplexity"])
                    correct_rhyme.append(result["nb_matching_rhyme_pairs"]/result["nb_expected_rhyming_pairs"])
                    syllable_differences.append(result["avg_syllable_count_difference"])
                    mean_deviation_syllable_count.append(result["mean_deviation_syllable_count"])
                    correct_syllale_count.append(result["nb_lines_correct_syllable_count"]/result["correct_nb_lines"])

                avg_perplexities_per_beam.append(sum(perplexities)/len(perplexities))
                avg_correct_rhyme_per_beam.append(sum(correct_rhyme)/len(correct_rhyme))
                avg_syllable_differences_per_beam.append(sum(syllable_differences)/len(syllable_differences))
                avg_mean_deviation_syllable_count_per_beam.append(sum(mean_deviation_syllable_count)/len(mean_deviation_syllable_count))
                avg_correct_syllable_count_per_beam.append(sum(correct_syllale_count)/len(correct_syllale_count))

                original_perplexities_per_beam.append(perplexities)
                original_correct_rhyme_per_beam.append(correct_rhyme)
                original_syllable_differences_per_beam.append(syllable_differences)
                original_mean_deviation_syllable_count_per_beam.append(mean_deviation_syllable_count)
                original_correct_syllable_count_per_beam.append(correct_syllale_count)



        avg_perplexities.append(avg_perplexities_per_beam)
        avg_correct_rhyme.append(avg_correct_rhyme_per_beam)
        avg_syllable_differences.append(avg_syllable_differences_per_beam)
        avg_mean_deviation_syllable_count.append(avg_mean_deviation_syllable_count_per_beam)
        avg_correct_syllable_count.append(avg_correct_syllable_count_per_beam)

        original_perplexities.append(original_perplexities_per_beam)
        original_correct_rhyme.append(original_correct_rhyme_per_beam)
        original_syllable_differences.append(original_syllable_differences_per_beam)
        original_mean_deviation_syllable_count.append(original_mean_deviation_syllable_count_per_beam)
        original_correct_syllable_count.append(original_correct_syllable_count_per_beam)


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
            "avg_perplexities": avg_perplexities,
            "avg_correct_rhyme": avg_correct_rhyme,
            "avg_syllable_differences": avg_syllable_differences,
            "avg_mean_deviation_syllable_count": avg_mean_deviation_syllable_count,
            "avg_correct_syllable_count": avg_correct_syllable_count,
            "original_perplexities": original_perplexities,
            "original_correct_rhyme": original_correct_rhyme,
            "original_syllable_differences": original_syllable_differences,
            "original_mean_deviation_syllable_count": original_mean_deviation_syllable_count,
            "original_correct_syllable_count": original_correct_syllable_count
        }, indent=4))
    
    plot_2d_heatmap(
        possible_good_beamscore_multipliers_rhyme,
        good_rhyming_token_multipliers,
        avg_perplexities,
        'Good Rhyming Token Multiplier',
        'Good Beamscore Multiplier',
        'Perplexity',
        'Perplexity vs. Good Beamscore Multiplier and Good Rhyming Token Multiplier',
        rhyming_folder+language_model_name.replace(" ", "_")+'/perplexity.png'
    )

    plot_2d_heatmap(
        possible_good_beamscore_multipliers_rhyme,
        good_rhyming_token_multipliers,
        avg_correct_rhyme,
        'Good Rhyming Token Multiplier',
        'Good Beamscore Multiplier',
        'Correct Rhyme',
        'Correct Rhyme vs. Good Beamscore Multiplier and Good Rhyming Token Multiplier',
        rhyming_folder+language_model_name.replace(" ", "_")+'/correct_rhyme.png'
    )

    plot_2d_heatmap(
        possible_good_beamscore_multipliers_rhyme,
        good_rhyming_token_multipliers,
        avg_syllable_differences,
        'Good Rhyming Token Multiplier',
        'Good Beamscore Multiplier',
        'Syllable Differences',
        'Syllable Differences vs. Good Beamscore Multiplier and Good Rhyming Token Multiplier',
        rhyming_folder+language_model_name.replace(" ", "_")+'/syllable_differences.png'
    )

    plot_2d_heatmap(
        possible_good_beamscore_multipliers_rhyme,
        good_rhyming_token_multipliers,
        avg_mean_deviation_syllable_count,
        'Good Rhyming Token Multiplier',
        'Good Beamscore Multiplier',
        'Mean Deviation Syllable Count',
        'Mean Deviation Syllable Count vs. Good Beamscore Multiplier and Good Rhyming Token Multiplier',
        rhyming_folder+language_model_name.replace(" ", "_")+'/mean_deviation_syllable_count.png'
    )

    plot_2d_heatmap(
        possible_good_beamscore_multipliers_rhyme,
        good_rhyming_token_multipliers,
        avg_correct_syllable_count,
        'Good Beamscore Multiplier',
        'Good Rhyming Token Multiplier',
        'Correct Syllable Count',
        'Correct Syllable Count vs. Good Beamscore Multiplier and Good Rhyming Token Multiplier',
        rhyming_folder+language_model_name.replace(" ", "_")+'/correct_syllable_count.png'
    )
    


def test():
    plot_2d_heatmap([1, 2, 3, 4, 5, 6], [0.2, 0.4, 0.6, 0.8, 0.9, 0.99], [[10,3,4,2,3,8],[1,9,5,9,7,9], [4,5,6,7,8,9], [1,2,3,4,5,6], [2,3,4,5,6,7], [3,4,5,6,7,8]], "Prompt Number", "Good Beamscore Multiplier", "Perplexity", "Title", "Experiments/ConstrainedParodieGenerator/CalibrationResults/test/test.png")




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
    elif mode == "test":
        test()
    else:
        print("Usage: python3 Calibrator.py <mode> \n mode = generate/evaluate ")
        sys.exit(1)

    
    
    

    
    
    
    





        
    


    

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





