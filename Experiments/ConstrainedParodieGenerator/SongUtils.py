from nltk.corpus import cmudict
import nltk
import torch
import os 
import json
from datetime import date, datetime

################################################## Global Parameters ################################################
#nltk.download('punkt')
nltk.download('cmudict')
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
    return paragraphs

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
    prompt = kwargs['prompt']
    parodie = kwargs['parodie']
    constraints_used = kwargs['constraints_used']
    state = kwargs['state']
    way_of_generation = kwargs['way_of_generation']
    decoding_method = kwargs['decoding_method']

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
    
    file_name_json = original_song_title + "_parodie_" + date_today + "_" + time + ".json"
    file_path_json = folder_path + LM + "/" + constrained_used_dir + "/json/" + file_name_json

    file_name_txt = original_song_title + "_parodie_" + date_today + "_" + time + ".txt"
    file_path_txt = folder_path + LM + "/" + constrained_used_dir + "/text/" + file_name_txt
    
    with open(file_path_txt, 'w') as file:
        file.write(parodie)
    
    song = {
        "original_song_title": original_song_title,
        "original_song_artist": original_song_artist,
        "language_model_name": LM,
        "system_prompt": system_prompt,
        "context": context,
        "prompt": prompt,
        "constraints_used": constraints_used,
        "way_of_generation": way_of_generation,
        "decoding_method": decoding_method,
        "state": state,
        "date": date_today,
        "time": time,
        "parodie": parodie
    }

    with open(file_path_json, 'w') as file:
        json.dump(song, file, indent=4)
    
    return file_path_json

################################################## SYLLABLE COUNTER FUNCTIONS ##################################################




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
    except KeyError:
        # if word not found in cmudict
        result =  count_syllables_hard(word)
    return result
def count_syllables_hard(word):
    word = word.lower()
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

phonemic_vowels = ["AA","AE","AH","AO","AW","AY","EH","EY","IH","IY","OW","OY","UH","UW","W","Y"] + ["ER"]
folder_path_rhyming_dicts = "Experiments/ConstrainedParodieGenerator/Constraints/RhymingConstraint/RhymingDictionaries/"   
import pickle

PERF_RHYMES_DICT = None
ASSONANT_RHYMES_DICT = None
INVERTED_PERF_RHYMES_DICT = None
INVERTED_ASSONANT_RHYMES_DICT = None

def create_rhyming_dicts():
    global PERF_RHYMES_DICT
    global ASSONANT_RHYMES_DICT
    global INVERTED_PERF_RHYMES_DICT
    global INVERTED_ASSONANT_RHYMES_DICT
    perf_rhymes = {}
    assonant_rhymes = {}
    for word, prons in d.items():
        perf_rhymes[word] = []
        assonant_rhymes[word] = []
        for pron in prons:
            index = -1
            for i in reversed(range(len(pron))):
                if pron[i][-1].isdigit():
                    index = i
                    break
            
            vowel = pron[index][:-1]
            last_consonants = pron[i+1:] if i+1 < len(pron) else []
            string = vowel +"".join(last_consonants)
            perf_rhymes[word].append(string)
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

    
    if not os.path.exists(folder_path_rhyming_dicts):
        os.makedirs(folder_path_rhyming_dicts)

    with open(folder_path_rhyming_dicts + 'perf_rhymes.pkl', 'wb') as f:
        pickle.dump(perf_rhymes, f,protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder_path_rhyming_dicts + 'assonant_rhymes.pkl', 'wb') as f:
        pickle.dump(assonant_rhymes, f,protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder_path_rhyming_dicts + 'inverted_perf_rhymes.pkl', 'wb') as f:
        pickle.dump(inverted_perf_rhymes, f,protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder_path_rhyming_dicts + 'inverted_assonant_rhymes.pkl', 'wb') as f:
        pickle.dump(inverted_assonant_rhymes, f,protocol=pickle.HIGHEST_PROTOCOL)
    PERF_RHYMES_DICT = perf_rhymes
    ASSONANT_RHYMES_DICT = assonant_rhymes
    INVERTED_PERF_RHYMES_DICT = inverted_perf_rhymes
    INVERTED_ASSONANT_RHYMES_DICT = inverted_assonant_rhymes


def load_rhyming_dicts():
    global PERF_RHYMES_DICT
    global ASSONANT_RHYMES_DICT
    global INVERTED_PERF_RHYMES_DICT
    global INVERTED_ASSONANT_RHYMES_DICT
    with open(folder_path_rhyming_dicts + 'perf_rhymes.pkl', 'rb') as f:
        perf_rhymes = pickle.load(f)
    with open(folder_path_rhyming_dicts + 'assonant_rhymes.pkl', 'rb') as f:
        assonant_rhymes = pickle.load(f)
    with open(folder_path_rhyming_dicts + 'inverted_perf_rhymes.pkl', 'rb') as f:
        inverted_perf_rhymes = pickle.load(f)
    with open(folder_path_rhyming_dicts + 'inverted_assonant_rhymes.pkl', 'rb') as f:
        inverted_assonant_rhymes = pickle.load(f)
    PERF_RHYMES_DICT = perf_rhymes
    ASSONANT_RHYMES_DICT = assonant_rhymes
    INVERTED_PERF_RHYMES_DICT = inverted_perf_rhymes
    INVERTED_ASSONANT_RHYMES_DICT = inverted_assonant_rhymes



def do_two_words_rhyme_perfectly(word1, word2):
    if word1 not in PERF_RHYMES_DICT or word2 not in PERF_RHYMES_DICT:
        return False
    return PERF_RHYMES_DICT[word1] == PERF_RHYMES_DICT[word2]

def do_two_words_rhyme_assonantly(word1, word2):
    if word1 not in ASSONANT_RHYMES_DICT or word2 not in ASSONANT_RHYMES_DICT:
        return False
    return ASSONANT_RHYMES_DICT[word1] == ASSONANT_RHYMES_DICT[word2]

def do_two_lines_rhyme(sentence1, sentence2):
    words1 = tokenize_sentence(sentence1)
    words2 = tokenize_sentence(sentence2)
    if len(words1) == 0 or len(words2) == 0:
        return False
    return do_two_words_rhyme_perfectly(words1[-1], words2[-1])

def get_perfect_rhyming_words(word):
    if word not in PERF_RHYMES_DICT:
        return []
    rhymes = PERF_RHYMES_DICT[word]
    result = []
    for rhyme in rhymes:
        result += INVERTED_PERF_RHYMES_DICT[rhyme]
    return result

def get_assonant_rhyming_words(word):
    if word not in ASSONANT_RHYMES_DICT:
        return []
    rhymes = ASSONANT_RHYMES_DICT[word]
    result = []
    for rhyme in rhymes:
        result += INVERTED_ASSONANT_RHYMES_DICT[rhyme]
    return result

def get_rhyming_lines(paragraph):
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
            if do_two_lines_rhyme(line, paragraph[j]):
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
    space_token = tokenizer.encode(" ")[0]
    for word in rhyming_words:
        tokens_with_space = tokenizer.encode(" "+word)
        if start_token is None and tokens_with_space[0] == space_token:
            tokens_with_space = tokens_with_space[1:]
        elif start_token is not None and start_token == tokens_with_space[0] and tokens_with_space[1] == space_token:
            tokens_with_space = tokens_with_space[2:]
        
        tokens = tokenizer.encode(word)
        if start_token is not None and tokens[0] == start_token:
            tokens = tokens[1:]

        syllable_count = count_syllables(word)
        if syllable_count > max_syllable_count:
            max_syllable_count = syllable_count
        dict_with_space = { 
            "tokens": tokens_with_space,
            "syllable_count": syllable_count
        }
        dict_without_space = { 
            "tokens": tokens,
            "syllable_count": syllable_count
        }
        result.append(dict_with_space)
        #result.append(dict_without_space)
    return result, max_syllable_count



        





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
    load_rhyming_dicts()

    # test_lines = ["I used to roll the dice", "Feel the fear in my enemy's eyes", "Listen as the crowd would sing", "Now the old king is dead, long live the king", "One minute, I held the key", "Next, the walls were closed on me", "And I discovered that my castles stand", "Upon pillars of salt and pillars of sand"]
    # print(get_rhyming_lines(test_lines))

    print(get_perfect_rhyming_words("dice"))