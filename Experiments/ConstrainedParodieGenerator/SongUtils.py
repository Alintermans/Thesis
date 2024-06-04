from nltk.corpus import cmudict
import nltk
import torch
import os 
import json
from datetime import date, datetime
import platform
from g2p_en import G2p

g2P = G2p()

################################################## Global Parameters ################################################
if platform.system() == 'Linux':
    folder_path = "/data/leuven/361/vsc36141"
    nltk.data.path.append(folder_path)
    nltk.download('punkt', download_dir=folder_path)
    nltk.download('cmudict', download_dir=folder_path)
    nltk.download('averaged_perceptron_tagger', download_dir=folder_path)
    nltk.download('universal_tagset', download_dir=folder_path)
else:
    nltk.download('punkt')
    nltk.download('cmudict')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')
d = cmudict.dict()

################################################## Language Model Functions ##################################################
def forbidden_charachters_to_tokens(tokenizer, forbidden_charachters):
    result = []
    start_token = tokenizer.encode("")
    if len(start_token) > 0:
        start_token = start_token[0]
    else:
        start_token = None
    for c in forbidden_charachters:
        tokens = tokenizer.encode(c)
        if len(tokens) > 1 and tokens[0] == start_token:
                tokens = tokens[1:]
        result.append(tokens)
    return result

def does_string_contain_newline(string):
    return "\n" in string
        
################################################ TEXT HELPER FUNCTIONS ################################################

def divide_song_into_paragraphs(song):
    paragraphs = []
    current_paragraph = []
    current_paragraph_name = ""
    first_done = False
    for line in song.split("\n"):
        if line.startswith("["):
            if first_done:
                paragraphs.append((current_paragraph_name, current_paragraph))
                current_paragraph = []
            else:
                first_done = True
            current_paragraph_name = line
        elif line == '':
                continue
        else:
            current_paragraph.append(line)
    paragraphs.append((current_paragraph_name, current_paragraph))
    result = []
    for paragraph in paragraphs:
        if not paragraph[0].startswith( "[ERROR]"):
            result.append(paragraph)
    return result

def read_song(file_path):
    song = ""
    song_file = open(file_path, 'r')
    song_json = json.load(song_file)
    song = song_json['lyrics']
    song_file.close()
    return song

def write_song(folder_path, **kwargs):
    original_song_file_path = kwargs['original_song_file_path']
    LM = kwargs['language_model_name']
    system_prompt = kwargs['system_prompt']
    context = kwargs['context']
    assistant_prompt = kwargs['assistant_prompt']
    prompt = kwargs['prompt']
    parodie = kwargs['parodie']
    constraints_used = kwargs['constraints_used']
    state = kwargs['state']
    way_of_generation = kwargs['way_of_generation']
    decoding_method = kwargs['decoding_method']
    chosen_hyper_parameters = kwargs['chosen_hyper_parameters']
    num_beams = kwargs['num_beams']
    seed = kwargs['seed']
    duration = kwargs['duration']

    date_today = date.today().strftime("%d-%m-%Y")
    time = datetime.now().strftime("%Hh-%Mm-%Ss")

    constrained_used_dir = constraints_used.replace(" ", "_")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if not os.path.exists(folder_path + LM):
        os.makedirs(folder_path + LM)

    if not os.path.exists(folder_path + LM + "/" + constrained_used_dir):
        os.makedirs(folder_path + LM + "/" + constrained_used_dir)
    
    if not os.path.exists(folder_path + LM + "/" + constrained_used_dir + "/json"):
        os.makedirs(folder_path + LM + "/" + constrained_used_dir + "/json")
    
    if not os.path.exists(folder_path + LM  + "/" + constrained_used_dir + "/text"):
        os.makedirs(folder_path + LM + "/" + constrained_used_dir + "/text")

    
    original_song = json.load(open(original_song_file_path, 'r'))
    original_song_title = original_song['title']
    original_song_artist = original_song['artist']
    
    file_name_json = original_song_title.replace(' ', '_') + "_parodie_" + date_today + "_" + time + ".json"
    file_path_json = folder_path + LM + "/" + constrained_used_dir + "/json/" + file_name_json

    file_name_txt = original_song_title.replace(' ', '_') + "_parodie_" + date_today + "_" + time + ".txt"
    file_path_txt = folder_path + LM + "/" + constrained_used_dir + "/text/" + file_name_txt
    
    with open(file_path_txt, 'w') as file:
        file.write(parodie)
    
    song = {
        "original_song_title": original_song_title,
        "original_song_artist": original_song_artist,
        "language_model_name": LM,
        "system_prompt": system_prompt,
        "context": context,
        "assistant_prompt": assistant_prompt,
        "prompt": prompt,
        "constraints_used": constraints_used,
        "chosen_hyper_parameters": chosen_hyper_parameters,
        "num_beams": num_beams,
        "seed": seed,
        "way_of_generation": way_of_generation,
        "decoding_method": decoding_method,
        "state": state,
        "date": date_today,
        "time": time,
        "generation_duration": duration,
        "parodie": parodie
    }

    with open(file_path_json, 'w') as file:
        json.dump(song, file, indent=4)
    
    return file_path_json


def get_final_word_of_line(line):
    line = line.replace("’", "'")
    words = line.split(" ")
    if words == []:
        return None
    
    if words[-1] == "":
        words = words[:-1]
        if words == []:
            return None

        
    #remove all punctuation marks and symbols that don't belong to a word
    not_to_end_with = ["’","'",".", ",", "!", "?", ";", ":", "-", "'", "\"", "(", ")", "[", "]", "{", "}",'``' , '&', '#', '*', '$', '£', '`', '+', '\n', '_', ""]
    while words[-1] in not_to_end_with:
        words = words[:-1]
        if words == []:
            return None
    #remove punctutation marks that are at the end of the word
    while words[-1][-1] in not_to_end_with:
        words[-1] = words[-1][:-1]
        if words[-1] == "":
            words = words[:-1]
            if words == []:
                return None

    return words[-1]


def replace_content_for_prompts(system, context, assistant_prompt, parody, song, rhyming_word, pos_tags, syllable_amount, line):
    result = [system, context, assistant_prompt]
    for i in range(len(result)):
        result[i] = result[i].replace("{{$SONG}}", song)
        result[i] = result[i].replace("{{$PARODY}}", parody)
        if rhyming_word is not None:
            result[i] = result[i].replace("{{$RHYMING_WORD}}", rhyming_word)
        else: 
            result[i] = result[i].replace("{{$RHYMING_WORD}}", "None")
        
        if pos_tags is not None:
            result[i] = result[i].replace("{{$POS_TAGS}}", ", ".join(str(element) for element in pos_tags))
        if syllable_amount is not None:
            result[i] = result[i].replace("{{$SYLLABLE_AMOUNT}}", str(syllable_amount))
        result[i] = result[i].replace("{{$LINE}}", line)
    return result[0], result[1], result[2]

def cleanup_line(line):
    #remove al leading and trailing spaces
    line = line.strip()
    line = line.replace("’", "'")
    line = " ".join(line.split())
    line = line.lower()
    if line.startswith("i "):
        line = line.replace("i ", "I ")
    if line.endswith(" i"):
        line = line.replace(" i", " I")
    line= line.replace(" i ", " I ")
    line = line.replace(" i'", " I'")
    line = line.replace("i'", "I'")
    line = line.replace(",,", ",")

    #delete all non regular characters
    line = ''.join(e for e in line if e.isalpha() or e in [" ", "'", "-", ",", ".", "!", "?", "(", ")"])

    return line




def get_song_structure(song_in_paragraphs):
    new_song_in_paragraphs = []
    structure = []
    for i in range(len(song_in_paragraphs)):
        paragraph = song_in_paragraphs[i]
        if paragraph not in new_song_in_paragraphs:
            new_song_in_paragraphs.append(paragraph)
            structure.append(len(new_song_in_paragraphs)-1)
        else:
            structure.append(new_song_in_paragraphs.index(paragraph))
    
    return new_song_in_paragraphs, structure

def process_parody(parody, song_structure):
    parody_in_paragraphs = divide_song_into_paragraphs(parody)
    new_parody = []
    for index in song_structure:
        new_parody.append( parody_in_paragraphs[index])
    
    new_parody = [x+"\n"+"\n".join(y)+"\n" for x,y in new_parody]
    new_parody = "\n".join(new_parody)

    return new_parody

    
        

    

################################################## SYLLABLE COUNTER FUNCTIONS ##################################################

def only_adds_regular_characters(original_line, new_line):
    new_line = new_line.replace("’", "'")
    original_line = original_line.replace("’", "'")
    original_line = ''.join(e for e in original_line if e.isalnum() )
    new_line = ''.join(e for e in new_line if e.isalnum() )
    return original_line != new_line

def does_not_contain_special_characters(line):
    line = line.replace("’", "'")
    new_line = ''.join(e for e in line if e.isalpha() or e in [" ", "'"])
    return line == new_line

def last_word_only_has_consontants(line):
    line = line.replace("’", "'")
    words = line.split(" ")
    if words == []:
        return False
    if words[-1] == "":
        words = words[:-1]
        if words == []:
            return False
    word = words[-1]
    return not any(e in ['a','e','i','o','u'] for e in word)



def tokenize_sentence(sentence):
    if sentence == "":
        return []
    
    first_round = nltk.word_tokenize(sentence, language='english', preserve_line=False)
    temp = []
    for i in range(len(first_round)):
        if first_round[i].endswith("-"):
            temp.append(first_round[i][:-1])
        elif not first_round[i] in [".", ",", "!", "?", ";", ":", "-", "\"", "(", ")", "[", "]", "{", "}", '&', '#', '*', '$', '£', '`', '+', '\n', '_', '``']:
            temp.append(first_round[i])
        
    first_round = temp
    if first_round == []:
        return []
    result = []
    last_one_appended = False
    for i in range(len(first_round)-1):
        if first_round[i+1].startswith("'"):
            result.append(first_round[i]+first_round[i+1])
            if i == len(first_round)-2:
                last_one_appended = True
        elif first_round[i].startswith("'"):
            continue
        else:
            result.append(first_round[i])

    if not last_one_appended:
        result.append(first_round[-1])
    
    if result[-1].endswith("'"):
        result[-1] = result[-1][:-1]
    
    return result


def count_syllables(word):
    if word == "":
        return 0
    #if the string is a puntcuation mark, return 0
    if word in [".", ",", "!", "?", ";", ":", "-", "'", "\"", "(", ")", "[", "]", "{", "}",'``' , '&', '#', '*', '$', '£', '`', '+', '\n', '_']:
        return 0
    try:
        result =  [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
        if result == 0:
            result = 1
    except KeyError:
        # if word not found in cmudict
        result =  count_syllables_hard(word)
    return result


def count_syllables_hard(word):
    word = word.lower()
    # pron = get_pronounciation_of_unknown_word(word)
    # count = len([x for x in pron if x[-1].isdigit()])
    # The following code comes from https://stackoverflow.com/questions/46759492/syllable-count-in-python
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

def get_syllable_count_of_sentence(sentence):
    sentence = sentence.replace("’", "'")
    sentence = cleanup_line(sentence)
    words = tokenize_sentence(sentence)
    result = sum(count_syllables(word) for word in words)
    return result

# Count the number of matching lines in respect to the same number of syllables in two paragraphs, where each paragraph is a list of lines
def count_matching_lines_on_syllable_amount(args):
    paragraph1, paragraph2 = args

    # Remove empty lines from the lists of lines
    lines1 = [line for line in paragraph1 if line != ""]
    lines2 = [line for line in paragraph2 if line != ""]

    # Print the number of lines in each paragraph
    # print("Number of lines in paragraph of original song:", len(lines1))
    # print("Number of lines in paragraph of parodie song:", len(lines2))

    syllable_counts1 = list(map(get_syllable_count_of_sentence, lines1))
    syllable_counts2 = list(map(get_syllable_count_of_sentence, lines2))
    # Count the number of matching lines in the paragraph
    matching_lines = sum(
        1 for syllables1, syllables2 in zip(syllable_counts1, syllable_counts2) if syllables1 == syllables2
    )

    syllable_count_differences = []
    
    for syllables1, syllables2 in zip(syllable_counts1, syllable_counts2):
        syllable_count_differences.append(abs(syllables1 - syllables2))
        
    
    return matching_lines, syllable_count_differences


################################################## Rhyming Functions ##################################################
## The following functions are implemented based upon the idea and code from https://github.com/heig-iict-ida/crpo/blob/main/rhyming
## The code is licensed under the Apache License 2.0
## The code comes from the paper Popescu-Belis A., Atrio A.R., Bernath B., Boisson E., Ferrari T., Theimer-Lienhard X., & Vernikos G. 2023. GPoeT: a Language Model Trained for Rhyme Generation on Synthetic Data. Proceedings of the 6th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL), EACL 2023, Dubrovnik, Croatia.
## The code is modified to fit the needs of the project
## We follow the same approach that for a perfect rhyme you have to look at the last vowel and the following consonants, and for an assonant rhyme you have to look at the last vowel. The defenition of a perfect rhyme states that it has to be the same vowels and consonants starting from the last stressed vowel. 

phonemic_vowels = ["AA","AE","AH","AO","AW","AY","EH","EY","IH","IY","OW","OY","UH","UW","W","Y"] + ["ER"]
folder_path_rhyming_dicts = "Experiments/ConstrainedParodieGenerator/Constraints/RhymingConstraint/RhymingDictionaries/"   
import pickle
import pronouncing
import requests

PERF_RHYMES_DICT = None
ASSONANT_RHYMES_DICT = None
NEAR_RHYME_CLASSES = None
INVERTED_PERF_RHYMES_DICT = None
INVERTED_ASSONANT_RHYMES_DICT = None

def get_top_frequent_words():
    file_path = 'Experiments/ConstrainedParodieGenerator/Constraints/RhymingConstraint/Most_frequent_english_words/frequents_words.txt'
    with open(file_path, 'r') as f:
        text = f.readlines()
    words = []
    for line in text:
        line = line.strip()
        line = line.replace("\n", "")
        words += line.split(" ")
    return words

def create_rhyming_dicts():
    global PERF_RHYMES_DICT
    global ASSONANT_RHYMES_DICT
    global NEAR_RHYME_CLASSES
    global INVERTED_PERF_RHYMES_DICT
    global INVERTED_ASSONANT_RHYMES_DICT
    perf_rhymes = {}
    assonant_rhymes = {}
    #frequents_words = get_top_frequent_words()
    for word, prons in d.items():
        # if word not in frequents_words:
        #     continue
        perf_rhymes[word] = []
        assonant_rhymes[word] = []
        for pron in prons:
            index = -1
            for i in reversed(range(len(pron))):
                if pron[i][-1].isdigit():
                    index = i
                    break
            if index == -1:
                continue
            
            vowel = pron[index][:-1]
            last_consonants = pron[i+1:] if i+1 < len(pron) else []
            key = tuple([vowel] + last_consonants)
            perf_rhymes[word].append(key)
            assonant_rhymes[word].append(vowel)
                    
    
    inverted_perf_rhymes = {}
    inverted_assonant_rhymes = {}
    for word, prons in perf_rhymes.items():
        for pron in prons:
            key = pron
            if key not in inverted_perf_rhymes:
                inverted_perf_rhymes[key] = []
            inverted_perf_rhymes[key].append(word)
    for word, pron in assonant_rhymes.items():
        for vowel in pron:
            key = vowel
            if key not in inverted_assonant_rhymes:
                inverted_assonant_rhymes[key] = []
            inverted_assonant_rhymes[key].append(word)
    


    near_rhyme_classes = {}
    #Create the near rhyme dictionary starting from the perfect rhyme dictionary
    for pron_1 in inverted_perf_rhymes.keys():
        near_rhyme_classes[pron_1] = [pron_1]
        for pron_2 in inverted_perf_rhymes.keys():
            if pron_1 == pron_2:
                continue
            if do_two_end_phon_seq_near_rhyme(list(pron_1), list(pron_2)):
                near_rhyme_classes[pron_1].append(pron_2)
    


    
    if not os.path.exists(folder_path_rhyming_dicts):
        os.makedirs(folder_path_rhyming_dicts)

    with open(folder_path_rhyming_dicts + 'perf_rhymes.pkl', 'wb') as f:
        pickle.dump(perf_rhymes, f,protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder_path_rhyming_dicts + 'assonant_rhymes.pkl', 'wb') as f:
        pickle.dump(assonant_rhymes, f,protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder_path_rhyming_dicts + 'near_rhyme_classes.pkl', 'wb') as f:
        pickle.dump(near_rhyme_classes, f,protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder_path_rhyming_dicts + 'inverted_perf_rhymes.pkl', 'wb') as f:
        pickle.dump(inverted_perf_rhymes, f,protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder_path_rhyming_dicts + 'inverted_assonant_rhymes.pkl', 'wb') as f:
        pickle.dump(inverted_assonant_rhymes, f,protocol=pickle.HIGHEST_PROTOCOL)
    PERF_RHYMES_DICT = perf_rhymes
    ASSONANT_RHYMES_DICT = assonant_rhymes
    NEAR_RHYME_CLASSES = near_rhyme_classes
    INVERTED_PERF_RHYMES_DICT = inverted_perf_rhymes
    INVERTED_ASSONANT_RHYMES_DICT = inverted_assonant_rhymes




def load_rhyming_dicts(use_frequent_words = False):
    global PERF_RHYMES_DICT
    global ASSONANT_RHYMES_DICT
    global NEAR_RHYME_CLASSES
    global INVERTED_PERF_RHYMES_DICT
    global INVERTED_ASSONANT_RHYMES_DICT
    perf_rhymes = None
    assonant_rhymes = None
    near_rhyme_classes = None
    inverted_perf_rhymes = None
    inverted_assonant_rhymes = None
    if use_frequent_words:
        folder_path_rhyming_dicts_frequent = folder_path_rhyming_dicts + "frequent_top_words/"
        with open(folder_path_rhyming_dicts_frequent + 'perf_rhymes.pkl', 'rb') as f:
            perf_rhymes = pickle.load(f)
        with open(folder_path_rhyming_dicts_frequent + 'assonant_rhymes.pkl', 'rb') as f:
            assonant_rhymes = pickle.load(f)
        with open(folder_path_rhyming_dicts_frequent + 'near_rhyme_classes.pkl', 'rb') as f:
            near_rhyme_classes = pickle.load(f)
        with open(folder_path_rhyming_dicts_frequent + 'inverted_perf_rhymes.pkl', 'rb') as f:
            inverted_perf_rhymes = pickle.load(f)
        with open(folder_path_rhyming_dicts_frequent + 'inverted_assonant_rhymes.pkl', 'rb') as f:
            inverted_assonant_rhymes = pickle.load(f)
    else:
        with open(folder_path_rhyming_dicts + 'perf_rhymes.pkl', 'rb') as f:
            perf_rhymes = pickle.load(f)
        with open(folder_path_rhyming_dicts + 'assonant_rhymes.pkl', 'rb') as f:
            assonant_rhymes = pickle.load(f)
        with open(folder_path_rhyming_dicts + 'near_rhyme_classes.pkl', 'rb') as f:
            near_rhyme_classes = pickle.load(f)
        with open(folder_path_rhyming_dicts + 'inverted_perf_rhymes.pkl', 'rb') as f:
            inverted_perf_rhymes = pickle.load(f)
        with open(folder_path_rhyming_dicts + 'inverted_assonant_rhymes.pkl', 'rb') as f:
            inverted_assonant_rhymes = pickle.load(f)
    PERF_RHYMES_DICT = perf_rhymes
    ASSONANT_RHYMES_DICT = assonant_rhymes
    NEAR_RHYME_CLASSES = near_rhyme_classes
    INVERTED_PERF_RHYMES_DICT = inverted_perf_rhymes
    INVERTED_ASSONANT_RHYMES_DICT = inverted_assonant_rhymes


def get_pronounciation_of_unknown_word(word):
    #remove special characters
    word = word.replace("’", "'")
    word = ''.join(e for e in word if e.isalpha()  or e in ["'", "-"])



    return g2P(word)


    #We use the legois tool from CMU
    # from io import BytesIO
    # import re
    # url = "http://www.speech.cs.cmu.edu/cgi-bin/tools/logios/lextool2.pl"

    # #filter word for special characters
    # word = re.sub(r'[^A-Za-z]', '', word)

    # word = word.upper()
    # word_bytes=BytesIO(word.encode('utf-8'))


    # files = {'wordfile': ('wordfile', word_bytes)}
    # try:
    #     request = requests.post(url, files=files)
    #     request.raise_for_status()
    # except requests.exceptions.HTTPError as err:
    #     return []
    # matches = re.findall(r'\<!-- DICT .*?\  -->', request.text)
    # url_dict = ""
    # if len(matches) >0:
    #     match = matches[0]
    #     url_dict = re.sub(r'\<!-- DICT ', '', match)
    #     url_dict = re.sub(r'  -->', '', url_dict)


    # try:
    #     pron_dict_request = requests.get(url_dict)
    #     pron_dict_request.raise_for_status()
    # except requests.exceptions.HTTPError as err:
    #     return []
    # try:
    #     pron_dict = pron_dict_request.text.split("\n")[0]
    #     pron = pron_dict.split('\t')[1].split(" ")
    # except IndexError:
    #     return []

    # #add stress
    # for i in range(len(pron)):
    #     if pron[i] in phonemic_vowels:
    #         pron[i] = pron[i] + "1"
    return pron
    

def get_pronounciation_of_word(word):
    if word in d:
        return d[word][0]
    else:
        return get_pronounciation_of_unkown_word(word)

def get_rhyme_ending(word, rhyme_type = "perfect"):
    try:
        if rhyme_type == "perfect":
            try:
                return PERF_RHYMES_DICT[word]
            except KeyError:
                
                pron = get_pronounciation_of_unknown_word(word)
                return get_perfect_rhyme_ending_from_pron(pron)
        elif rhyme_type == "assonant":
            try:
                return ASSONANT_RHYMES_DICT[word]
            except KeyError:
                pron = get_pronounciation_of_unknown_word(word)
                return get_assonant_rhyme_ending_from_pron(pron)
        else:
            return ""
    except Exception as e:
        return ""


def get_perfect_rhyme_ending_from_pron(pron):
    index = -1

    for i in reversed(range(len(pron))):
        if pron[i] != '' and pron[i][-1].isdigit():
            index = i
            break
    if index == -1:
        return ("")
    vowel = pron[index][:-1]
    last_consonants = pron[i+1:] if i+1 < len(pron) else []
    return [tuple([vowel] + last_consonants)]

def get_assonant_rhyme_ending_from_pron(pron):
    index = -1
    for i in reversed(range(len(pron))):
        if pron[i] != '' and pron[i][-1].isdigit():
            index = i
            break
    if index == -1:
        return ""
    vowel = pron[index][:-1]
    return [vowel]

def do_two_end_phon_seq_near_rhyme(phon_seq1, phon_seq2):
    #The following sequence is adopted from the code from weirdAI on github by Riedl M. (https://github.com/markriedl/weirdai/blob/master/weird_ai.ipynb)
    NEAR_SETS = [
        ['T', 'D'],
        ['P', 'B'],
        ['K', 'G'],
        ['F', 'V'],
        ['TH', 'DH'],
        ['S', 'Z'],
        ['SH', 'ZH'],
        ['CH', 'JH'],
        ['M', 'N'],
        ['NG', 'L'],

             ]

    #filter out all empty strings
    phon_seq1 = [x for x in phon_seq1 if x != '']
    phon_seq2 = [x for x in phon_seq2 if x != '']

    #The end sequence starts with the last vowel and goes to the end of the word
    if len(phon_seq1) == 0 or len(phon_seq2) == 0:
        return False
    
    vowel_1 = phon_seq1[0]
    vowel_2 = phon_seq2[0]

    consonants_1 = phon_seq1[1:]
    consonants_2 = phon_seq2[1:]

    if vowel_1[-1].isdigit():
        vowel_1 = vowel_1[:-1]
    if vowel_2[-1].isdigit():
        vowel_2 = vowel_2[:-1]

    #If the vowel is not the same retrun false
    if vowel_1 != vowel_2:
        if vowel_1 == 'ER' and vowel_2 == 'EH' and len(consonants_2) >0 and consonants_2[0] == 'R':
            consonants_2 = consonants_2[1:]
        elif vowel_1 == 'EH' and vowel_2 == 'ER' and len(consonants_1) >0 and consonants_1[0] == 'R':
            consonants_1 = consonants_1[1:]
        else:
            return False
    
    #Perfect Rhyme
    if consonants_1 == consonants_2:
        return True
    
    if len(consonants_1) == 0 or len(consonants_2) == 0:
        return False

    #If they have the same or near same end consonant, but different consonants between the vowel and the end consonant 
    if consonants_1[-1] == consonants_2[-1] or [consonants_1[-1], consonants_2[-1]] in NEAR_SETS or [consonants_2[-1], consonants_1[-1]] in NEAR_SETS:
        return True
    
    #If they have the same  or near same first consonant but different end consonants
    # if consonants_1[0] == consonants_2[0] or [consonants_1[0], consonants_2[0]] in NEAR_SETS or [consonants_2[0], consonants_1[0]] in NEAR_SETS:
    #     return True
    

    return False


def do_two_words_near_rhyme(word1, word2):
    word1_rhyming = get_rhyme_ending(word1, "perfect")
    word2_rhyming = get_rhyme_ending(word2, "perfect")
    for ending1 in word1_rhyming:
        for ending2 in word2_rhyming:
            if do_two_end_phon_seq_near_rhyme(list(ending1), list(ending2)):
                return True

    return False


def do_two_words_rhyme_perfectly(word1, word2):
    word1_rhyming = None
    try:
        word1_rhyming = PERF_RHYMES_DICT[word1]
    except KeyError:
        word1_pron = get_pronounciation_of_unknown_word(word1)
        word1_rhyming = get_perfect_rhyme_ending_from_pron(word1_pron)
    
    word2_rhyming = None
    try:
        word2_rhyming = PERF_RHYMES_DICT[word2]
    except KeyError:   
        word2_pron = get_pronounciation_of_unknown_word(word2)
        word2_rhyming = get_perfect_rhyme_ending_from_pron(word2_pron)
    
    for ending1 in word1_rhyming:
        for ending2 in word2_rhyming:
            if ending1 == ending2:
                return True
    
    return False

def do_two_words_rhyme_assonantly(word1, word2):
    word1_rhyming = None
    try:
        word1_rhyming = ASSONANT_RHYMES_DICT[word1]
    except KeyError:
        word1_pron = get_pronounciation_of_unknown_word(word1)
        word1_rhyming = get_assonant_rhyme_ending_from_pron(word1_pron)

    word2_rhyming = None
    try:
        word2_rhyming = ASSONANT_RHYMES_DICT[word2]
    except KeyError:
        word2_pron = get_pronounciation_of_unknown_word(word2)
        word2_rhyming = get_assonant_rhyme_ending_from_pron(word2_pron)
    
    for ending1 in word1_rhyming:
        for ending2 in word2_rhyming:
            if ending1 == ending2:
                return True
    
    return False

def do_two_lines_rhyme(sentence1, sentence2, rhyme_type = "perfect"):
    words1 = tokenize_sentence(sentence1)
    words2 = tokenize_sentence(sentence2)
    if len(words1) == 0 or len(words2) == 0:
        return False
    word1 = words1[-1]
    word2 = words2[-1]
    not_to_end_with = [".", ",", "!", "?", ";", ":", "-", "'", "\"", "(", ")", "[", "]", "{", "}",'``' , '&', '#', '*', '$', '£', '`', '+', '\n', '_']
    for c in not_to_end_with:
        if word1.endswith(c):
            word1 = word1[:-(len(c))]
        if word2.endswith(c):
            word2 = word2[:-(len(c))]
    
    
    if rhyme_type == "perfect":
        return do_two_words_rhyme_perfectly(word1, word2)
    elif rhyme_type == "assonant":
        return do_two_words_rhyme_assonantly(word1, word2)
    elif rhyme_type == "near":
        return do_two_words_near_rhyme(word1, word2)
    else:
        return False

def get_perfect_rhyming_words(word):
    rhymes = []
    try:
        rhymes = PERF_RHYMES_DICT[word]
    except KeyError:
        word_pron = get_pronounciation_of_unknown_word(word)
        rhymes = get_perfect_rhyme_ending_from_pron(word_pron)
        
    result = []
    for rhyme in rhymes:
        result += INVERTED_PERF_RHYMES_DICT[rhyme]
    return result

def get_assonant_rhyming_words(word):
    rhymes = []
    try:
        rhymes = ASSONANT_RHYMES_DICT[word]
    except KeyError:
        word_pron = get_pronounciation_of_unknown_word(word)
        rhymes = get_assonant_rhyme_ending_from_pron(word_pron)
        
    result = []
    for rhyme in rhymes:
        result += INVERTED_ASSONANT_RHYMES_DICT[rhyme]
    return result

def get_near_rhyming_words(word):
    end_pron = get_rhyme_ending(word, "perfect")
    near_rhyme_endings = []
    for ending in end_pron:
        near_rhyme_endings = NEAR_RHYME_CLASSES[ending]
    result = []
    for ending in near_rhyme_endings:
        result += INVERTED_PERF_RHYMES_DICT[ending]
    return result

def _do_two_words_rhyme(word1, word2, rhyme_type = "perfect"):
    if rhyme_type == "perfect":
        return do_two_words_rhyme_perfectly(word1, word2)
    elif rhyme_type == "assonant":
        return do_two_words_rhyme_assonantly(word1, word2)
    elif rhyme_type == "near":
        return do_two_words_near_rhyme(word1, word2)
    else:
        return False

def _get_rhyming_words(word, rhyme_type = "perfect"):
    if word == "":
        return []

    word = word.lower()

    try:
        if rhyme_type == "perfect":
            return get_perfect_rhyming_words(word)
        elif rhyme_type == "assonant":
            return get_assonant_rhyming_words(word)
        elif rhyme_type == "near":
            return get_near_rhyming_words(word)
        else:
            return []
    except KeyError:
        return []


def _get_rhyming_lines(paragraph, rhyme_type = "perfect"):
    
    #it will return a list and indicate for each line in the paragraph, to which other lines it rhymes
    all_rhyming_lines = []
    lines_to_exlude = []
    for i in range(len(paragraph)):
        if i in lines_to_exlude:
            continue
        line = paragraph[i]
        rhyming_lines = [i]
        for j in range(i+1, len(paragraph)):
            if j in lines_to_exlude:
                continue
            if do_two_lines_rhyme(line, paragraph[j], rhyme_type):
                rhyming_lines.append(j)
                lines_to_exlude.append(j)
        all_rhyming_lines.append(rhyming_lines)
    result = []
    for i in range(len(paragraph)):
        added = False
        for rhyming_lines in all_rhyming_lines:
            if i in rhyming_lines and rhyming_lines[0] != i:
                result.append(rhyming_lines[0])
                added = True
                break
        if not added:
            result.append(None)

    return result


def rhyming_words_to_tokens_and_syllable_count(tokenizer, rhyming_words, start_token = None):
    result = []
    max_syllable_count = 0
    space_token = tokenizer.encode(" ", add_special_tokens=False)[0]
    for word in rhyming_words:
        tokens_with_space = tokenizer.encode(" "+word, add_special_tokens=False)
        if start_token is None and tokens_with_space[0] == space_token:
            tokens_with_space = tokens_with_space[1:]
        elif start_token is not None and start_token == tokens_with_space[0] and tokens_with_space[1] == space_token:
            tokens_with_space = tokens_with_space[2:]
        
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if start_token is not None and tokens[0] == start_token:
            tokens = tokens[1:]

        syllable_count = count_syllables(word)
        if syllable_count > max_syllable_count:
            max_syllable_count = syllable_count
        dict_with_space = { 
            "tokens": tokens_with_space,
            "syllable_count": syllable_count,
            "word": word
        }
        dict_without_space = { 
            "tokens": tokens,
            "syllable_count": syllable_count,
            "word": word
        }
        result.append(dict_with_space)
        result.append(dict_without_space)
    return result, max_syllable_count


def count_syllables_of_rhyme_words_in_songs():

    songs_dir = "Songs/json/"
    rhyming_words = []
    for file in os.listdir(songs_dir):
        
        if file.endswith(".json"):
            song = read_song(songs_dir + file)
            song = divide_song_into_paragraphs(song)
            for (title, paragraph) in song:
                rhyming_lines = _get_rhyming_lines(paragraph, "perfect")
                rhyming_words_temp = []
                for i in range(len(paragraph)):
                    if (rhyming_lines[i] is not None):
                        rhyming_words_temp.append(get_final_word_of_line(paragraph[i]))
                        rhyming_words_temp.append(get_final_word_of_line(paragraph[rhyming_lines[i]]))
                rhyming_words_temp = list(set(rhyming_words_temp))
                rhyming_words += rhyming_words_temp
    rhyming_words = list(set(rhyming_words))
    number_of_syllables = dict()
    for word in rhyming_words:
        number_of_word_syllables= count_syllables(word)
        if number_of_word_syllables not in number_of_syllables:
            number_of_syllables[number_of_word_syllables] = 1
        else:
            number_of_syllables[number_of_word_syllables] += 1
    print(number_of_syllables)
    # print the number of syllables of the rhyming words in percentage
    total = sum(number_of_syllables.values())
    number_of_syllables_percentage = dict()
    for key in number_of_syllables:
        number_of_syllables_percentage[key] = number_of_syllables[key]/total
    
    print("Total: " + str(total))
    print(number_of_syllables_percentage)
    


################################################## POS TAGGING FUNCTIONS ##################################################

def get_pos_tags_of_line(line):
    line = line.replace("’", "'")
    #remove all signs that don't belong to a word
    line = ''.join(e for e in line if e.isalnum() or e in [" ", "'"])


    words = nltk.word_tokenize(line, language='english', preserve_line=False)
    tags = nltk.pos_tag(words, tagset='universal')

    #Put them in a list
    result = []
    for word, tag in tags:
        result.append(tag)
    
    return result

def dtw_distance(s1, s2):
    dist = lambda x, y: 0 if x == y else 1
    n, m = len(s1), len(s2)
    dtw_matrix = [[float('inf')] * (m + 1) for x in range(n + 1)]
    dtw_matrix[0][0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist(s1[i - 1], s2[j - 1])
            dtw_matrix[i][j] = cost + min(dtw_matrix[i-1][j],    # insertion
                                          dtw_matrix[i][j-1],    # deletion
                                          dtw_matrix[i-1][j-1])  # match

    return dtw_matrix[n][m]


def similarity_of_pos_tags_sequences(seq1, seq2):
    if len(seq1) == 0 or len(seq2) == 0:
        return 0
    # Calculate DTW distance
    dtw_dist = dtw_distance(seq1, seq2)
    
    normalized_dtw_similarity = 1 - (dtw_dist / (len(seq1) + len(seq2)))

    return normalized_dtw_similarity




################################################## TEST FUNCTIONS ##################################################


############################# SYLLABLE COUNT TESTS
def test_tokenize_sentence():
    sentence = "I'm a test sentence"
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]
    
    sentence = "I'm a test sentence."
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]
    
    sentence = "I'm a test sentence!"
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]
    
    sentence = "I'm a test sentence?"
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]
    
    sentence = "I'm a test sentence;"
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]
    
    sentence = "I'm a test sentence:"
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]
    
    sentence = "I'm a test sentence-"
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]
    
    sentence = "I'm a test sentence'"
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]
    
    sentence = "I'm a test sentence\n"
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]
    
    sentence = "I'm a test sentence("
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]
    
    sentence = "I'm a test sentence)"
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]
    
    sentence = "I'm a test sentence["
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]
    
    sentence = "I'm a test sentence]"
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]
    
    sentence = "I'm a test sentence{"
    result = tokenize_sentence(sentence )
    assert result == ["I'm", "a", "test", "sentence"]

    sentence = "I'm a test sentence}"
    result = tokenize_sentence(sentence)
    assert result == ["I'm", "a", "test", "sentence"]

    sentence = "I'm "
    result = tokenize_sentence(sentence)
    assert result == ["I'm"]


def test_count_syllables():
    assert count_syllables("test") == 1
    assert count_syllables("sentence") == 2
    assert count_syllables("I'm") == 1
    assert count_syllables("I've") == 1
    assert count_syllables("I'll") == 1
    assert count_syllables("I'd") == 1
    assert count_syllables("I") == 1
    assert count_syllables("I don't have") == 3
    assert count_syllables("-- I haven't had a") == 5

def test_count_syllables_sentences():
    assert get_syllable_count_of_sentence("I'm a test sentence") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence.") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence!") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence?") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence;") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence:") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence-") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence'") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence\n") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence(") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence)") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence[") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence]") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence{") == 5
    assert get_syllable_count_of_sentence("I'm a test sentence}") == 5
    assert get_syllable_count_of_sentence("I'm ") == 1
    assert get_syllable_count_of_sentence("I'm") == 1
    assert get_syllable_count_of_sentence("I'm a") == 2
    assert get_syllable_count_of_sentence("I'm a ") == 2
    assert get_syllable_count_of_sentence("I'm a test") == 3
    assert get_syllable_count_of_sentence("I'm a test ") == 3
    assert get_syllable_count_of_sentence("-- I haven't had a") == 5
    assert get_syllable_count_of_sentence("-- ") == 0
    assert get_syllable_count_of_sentence("") == 0
    
    assert get_syllable_count_of_sentence("I'm a test sentence\nI'm a test sentence") == 10
    assert get_syllable_count_of_sentence("I'm a test sentence\nI'm a test sentence\nI'm a test sentence") == 15
    
def test_syllable_counter_functions():
    test_tokenize_sentence()
    test_count_syllables()
    test_count_syllables_sentences()


################################################## MAIN ##################################################

if __name__ == "__main__":
    #test_syllable_counter_functions()
    #print("All tests passed")
    #create_rhyming_dicts()
    load_rhyming_dicts(use_frequent_words=False)

    # pron_1 = get_pronounciation_of_word("paddle")
    # pron_2 = get_pronounciation_of_word("mull")
    # print(pron_1, pron_2)

    # print(do_two_end_phon_seq_near_rhyme(pron_1, pron_2))
    #print(get_perfect_rhyming_words("alone"))
    #print(get_pos_tags_of_line("It is "))
   #print(do_two_words_rhyme_perfectly("hello","meeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"))
    
    # print(_do_two_words_rhyme("dream", "ims", "assonant"))
    #print(get_assonant_rhyming_words("Great"))
    #print(cleanup_line("now in  300 kitchen,                     I chills alone �"))
    #print(only_adds_regular_characters("I'm a test sentenc", "I'm a test sentenc've"))
    #print(get_syllable_count_of_sentence("Let's fast forward to three hundred takeout coffees later"))
    #print(get_top_frequent_words())
    #count_syllables_of_rhyme_words_in_songs()
    print(d['myette'])
        
    
