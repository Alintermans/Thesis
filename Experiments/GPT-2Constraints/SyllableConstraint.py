from Constraint import Constraint
from nltk.corpus import cmudict

import nltk
import torch

nltk.download('punkt')
nltk.download('cmudict')
d = cmudict.dict()


def tokenize_sentence(sentence):
    
    
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


class SyllableConstraint(Constraint):
    def __init__(self, syllable_amount, tokenizer):
        self.syllable_amount = syllable_amount
        self.tokenizer = tokenizer
    
    def apply_beam_constraint(self, next_token, next_score, input_ids, cur_len, length_penalty):
        previous_text = self.tokenizer.decode(input_ids)
        current_token_text = self.tokenizer.decode(next_token)
        candidate_text = previous_text + current_token_text

        words = tokenize_sentence(candidate_text)
        result = sum(count_syllables(word) for word in words)
        
        current_length = input_ids.shape[-1] + 1
        

        if result > self.syllable_amount or (result == self.syllable_amount and count_syllables(current_token_text) == 0):
            next_score = next_score + next_score*(10) * ( current_length ** length_penalty)
            #next_score = float('-inf')
        elif result == self.syllable_amount:
            next_score = next_score - next_score*0.1 * ( current_length ** length_penalty)
        #print(candidate_text,' count: ' ,result, ' score: ', next_score)
        return next_score
    
    def stopping_criteria(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        for input in input_ids:
            sentence = self.tokenizer.decode(input, skip_special_tokens=True)
            words = tokenize_sentence(sentence)
            sum = 0
            for word in words:
                sum += count_syllables(word) 
            if sum >=self.syllable_amount:
                #print('sum: ',sum, 'sentence: ', sentence)
                return True

        return False
    
    def is_beam_constraint_active(self):
        return True
    
    def is_stopping_criteria_active(self):
        return True
    
    def is_logits_processor_active(self):
        return False

#write tests to test the tokenize_sentence function so it will correctly put all words like I'm, I've etc. together
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
    assert count_syllables("I've") == 1
    assert count_syllables("I'll") == 1
    assert count_syllables("I'd") == 1
    assert count_syllables("I") == 1
    assert count_syllables("I've") == 1
    assert count_syllables("I'll") == 1
    assert count_syllables("I'd") == 1
    assert count_syllables("I") == 1
    assert count_syllables("I've") == 1
    assert count_syllables("I'll") == 1
    assert count_syllables("I'd") == 1
    assert count_syllables("I") == 1
    assert count_syllables("I've") == 1
    assert count_syllables("I'll") == 1
    assert count_syllables("I'd") == 1
    assert count_syllables("I") == 1
    assert count_syllables("I've") == 1
    assert count_syllables("I'll") == 1
    assert count_syllables("I'd") == 1
    assert count_syllables("I") == 1
    assert count_syllables("I've") == 1
    assert count_syllables("I'll") == 1
    assert count_syllables("I'd") == 1
    assert count_syllables("I") == 1
    assert count_syllables("I've") == 1
    assert count_syllables("I'll") == 1
    assert count_syllables("I'd") == 1
    assert count_syllables("I") == 1
    assert count_syllables("I've") == 1

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
    
    assert get_syllable_count_of_sentence("I'm a test sentence\nI'm a test sentence") == 10
    assert get_syllable_count_of_sentence("I'm a test sentence\nI'm a test sentence\nI'm a test sentence") == 15
    


if __name__ == "__main__":
    test_tokenize_sentence()
    test_count_syllables()
    test_count_syllables_sentences()
    print("All tests passed")