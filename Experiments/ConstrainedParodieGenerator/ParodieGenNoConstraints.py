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
                utils
                )
import torch

######### Supress Warnings ##########
utils.logging.set_verbosity_error()

AVAILABLE_LMS = {'GPT2': GPT2, 'Gemma2BIt': Gemma2BIt, 'Gemma2B': Gemma2B, 'Gemma7B': Gemma7B, 'Gemma7BIt': Gemma7BIt, 'Llama2_7B': Llama2_7B, 'Llama2_7BChat': Llama2_7BChat, 'Llama2_70B': Llama2_70B, 'Llama2_70BChat': Llama2_70BChat, 'Mistral7BV01': Mistral7BV01, 'Mistral7BItV02': Mistral7BItV02, 'Mistral8x7BV01': Mistral8x7BV01, 'Mistral8x7BItV01': Mistral8x7BItV01}

folder_path_for_generated_parodies = os.environ["VSC_DATA"] +"NoConstraints/"

############### Variables ###############
lm = None
tokenizer = None
model = None



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
    tokenizer = lm.get_tokenizer()
    model = lm.get_model()
    



def generate_parody(**kwargs):
    song_file_path = kwargs['song_file_path']

    set_seed(kwargs['seed'])

    system_prompt = kwargs['system_prompt']
    context_prompt = kwargs['context_prompt']
    assistant_prompt = kwargs['assistant_prompt']


    use_cuda = kwargs.get('use_cuda', False)
    use_quantization = kwargs.get('use_quantization', False)
    set_language_model(kwargs['language_model'], use_quantization=use_quantization, use_cuda=use_cuda)

    song = read_song(song_file_path) #expects a json file, where the lyrics is stored in the key 'lyrics'

    if system_prompt.endswith('.txt'):
        system_prompt = open(system_prompt, 'r').read()
    if context_prompt.endswith('.txt'):
        context_prompt = open(context_prompt, 'r').read()
    if assistant_prompt.endswith('.txt'):
        assistant_prompt = open(assistant_prompt, 'r').read()

    
    parodie = ""
    state = "Finished Correctly"
    try:
        start_time = time.time()
        prepared_system_prompt, prepared_context_prompt, prepared_assistant_prompt = replace_content_for_prompts(system_prompt, context_prompt, assistant_prompt, parodie, song, "", "", 0, "")
        prompt, tokenized_prompt = lm.prepare_prompt(prepared_system_prompt, prepared_context_prompt, prepared_assistant_prompt)
        tokenized_prompt = tokenized_prompt.to(model.device)
        parodie = model.generate(tokenized_prompt, do_sample=kwargs['do_sample'], top_p=kwargs['top_p'], temperature=kwargs['temperature'], num_beams=kwargs['num_beams'],repetition_penalty=1.2, max_length=4096, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        parodie = tokenizer.decode(parodie[0], skip_special_tokens=True)[len(prompt):]
        duration = time.time() - start_time


    except Exception as e:
        #raise Exception(e)
        print("Error has occured ", e)
        state = "Error has occured " + str(e) + "\n" + "Not finished correctly"
        parodie += "\n\n" + "[ERROR]: Not finished correctly" + "\n\n"
    
    decoding_method = "Beam Search"
    if (kwargs.get('do_sample') is not None or kwargs.get('do_sample') == True):
        decoding_method = "Sampling Beam Search" + " | top_p: " + str(kwargs['top_p']) + " | temperature: " + str(kwargs['temperature'])
    
    print("Parodie: ", parodie)
    write_song(folder_path_for_generated_parodies, 
                original_song_file_path = song_file_path, 
                parodie = parodie, 
                context = context_prompt, 
                system_prompt = system_prompt, 
                assistant_prompt = assistant_prompt,
                prompt = prompt, 
                constraints_used = "None",
                chosen_hyper_parameters ={},
                num_beams = kwargs['num_beams'],
                seed = kwargs['seed'],
                language_model_name = lm.name,
                state = state,
                way_of_generation = "No Constraints",
                duration = duration,
                decoding_method = decoding_method)







if(__name__ == '__main__'):
    ###### Set Up ######
    language_model = 'GPT2'
    #language_model = 'Llama2_7BChat'
    #language_model = "Mistral7BItV02"
    song_file_path = 'Songs/json/Taylor_Swift-It_Is_Over_Now_(Very_Small).json'
    #song_file_path = 'Songs/json/Coldplay-Viva_La_Vida.json'
    #song_file_path = 'Songs/json/Taylor_Swift-Is_It_Over_Now_(Small_Version).json'

    prompt_version = "1"

    system_prompt = "Experiments/ConstrainedParodieGenerator/PromptTexts/"+prompt_version+"/system_prompt.txt"
    context_prompt = "Experiments/ConstrainedParodieGenerator/PromptTexts/"+prompt_version+"/context_prompt.txt"
    assistant_prompt = "Experiments/ConstrainedParodieGenerator/PromptTexts/"+prompt_version+"/assistant_prompt.txt"

    LMs = ['Llama2_7BChat', 'Llama2_70BChat', 'Mistral7BItV02', 'Mistral8x7BItV01']

    for lm in LMs:
        language_model = lm
        for song in os.listdir("Songs/json/"):
            song_file_path = "Songs/json/" + song

            generate_parody(song_file_path= song_file_path, 
                    system_prompt = system_prompt, 
                    context_prompt = context_prompt, 
                    assistant_prompt = assistant_prompt,
                    language_model = language_model,
                    use_cuda=True,
                    use_quantization=True,
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=0.75, 
                    #temperature=float(2),
                    num_beams=5, 
                    seed=42
                    )
            