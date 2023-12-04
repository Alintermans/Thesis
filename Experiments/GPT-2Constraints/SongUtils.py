from nltk.corpus import cmudict
import nltk
import torch

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
