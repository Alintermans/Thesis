from nltk.corpus import cmudict
import nltk
import torch
import os 
import json
from datetime import date, datetime

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

    date_today = date.today().strftime("%d-%m-%Y")
    time = datetime.now().strftime("%Hh-%Mm-%Ss")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if not os.path.exists(folder_path + LM):
        os.makedirs(folder_path + LM)
    
    if not os.path.exists(folder_path + LM + "/json"):
        os.makedirs(folder_path + LM + "/json")
    
    if not os.path.exists(folder_path + LM + "/text"):
        os.makedirs(folder_path + LM + "/text")

    
    original_song = json.load(open(original_song_file_path, 'r'))
    original_song_title = original_song['title']
    
    file_name_json = original_song_title + "_parodie_" + date_today + "_" + time + ".json"
    file_path_json = folder_path + LM + "/json/" + file_name_json

    file_name_txt = original_song_title + "_parodie_" + date_today + "_" + time + ".txt"
    file_path_txt = folder_path + LM + "/text/" + file_name_txt
    
    with open(file_path_txt, 'w') as file:
        file.write(parodie)
    
    song = {
        "original_song_title": original_song_title,
        "language_model_name": LM,
        "system_prompt": system_prompt,
        "context": context,
        "prompt": prompt,
        "constraints_used": constraints_used,
        "way_of_generation": way_of_generation,
        "state": state,
        "date": date_today,
        "time": time,
        "parodie": parodie
    }

    with open(file_path_json, 'w') as file:
        json.dump(song, file, indent=4)
    
    return file_path_json

################################################## SYLLABLE COUNTER FUNCTIONS ##################################################

nltk.download('punkt')
nltk.download('cmudict')
d = cmudict.dict()


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
    test_syllable_counter_functions()
    print("All tests passed")
