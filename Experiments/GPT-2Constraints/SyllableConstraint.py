from Constraint import Constraint
from nltk.corpus import cmudict

import nltk
import torch


d = cmudict.dict()
nltk.download('punkt')
nltk.download('cmudict')

def tokenize_sentence(sentence):
    first_round = nltk.word_tokenize(sentence)
    result = []
    last_one_appended = False
    for i in range(len(first_round)-2):
        if first_round[i+1].startswith("'"):
            result.append(first_round[i]+first_round[i+1])
            if i == len(first_round)-2:
                last_one_appended = True
        elif first_round[i].startswith("'"):
            continue
        else:
            result.append(first_round[i])

    if not last_one_appended:
        result.append(first_round[-2])
    result.append(first_round[-1])
    return result


def count_syllables(word):
    #if the string is a puntcuation mark, return 0
    if word in [".", ",", "!", "?", ";", ":", "-", "'", "\"", "(", ")", "[", "]", "{", "}",'``' , '&', '#', '*', '$', '£', '`', '+', '\n']:
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
