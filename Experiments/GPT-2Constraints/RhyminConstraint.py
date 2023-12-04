from Constraint import Constraint
from SongUtils import tokenize_sentence, get_syllable_count_of_sentence
import torch
import pronouncing

def do_two_words_rhyme(word1, word2):
    phonetics1 = pronouncing.phones_for_word(word1)
    phonetics2 = pronouncing.phones_for_word(word2)

    # Check if any of the phonetic representations of word1 rhymes with any of word2
    for phonetic1 in phonetics1:
        for phonetic2 in phonetics2:
            if pronouncing.rhyming_part(phonetic1) == pronouncing.rhyming_part(phonetic2):
                return True

    return False

class EndRhymeWith(Constraint):
    def __init__(self, word, tokenizer, syllable_amount):
        self.word = word
        self.tokenizer = tokenizer
        self.syllable_amount = syllable_amount

    
    def apply_beam_constraint(self, next_token, next_score, input_ids, cur_len, length_penalty):
        if next_token != self.tokenizer.eos_token_id:
            return next_score
        previous_text = self.tokenizer.decode(input_ids)
        current_token_text = self.tokenizer.decode(next_token)
        candidate_text = previous_text + current_token_text

        words = tokenize_sentence(candidate_text)
        if len(words) == 0:
            return next_score
        
        last_word = words[-1]
        print(last_word)
        if do_two_words_rhyme(last_word, self.word):
            next_score = next_score - next_score*0.1 * ( cur_len ** length_penalty)
        else:
            next_score = next_score - next_score*(100) ** ( cur_len ** length_penalty)
        return next_score
    
    def logits_processor(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i in range(len(input_ids)):
            previous_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            if (get_syllable_count_of_sentence(previous_text) < self.syllable_amount-5):
                continue
            #get 10 best tokens
            best_tokens = torch.topk(scores[i], 1000)
            for token in best_tokens[1]:
                candidate_text = previous_text + self.tokenizer.decode(token)
                if get_syllable_count_of_sentence(candidate_text) == self.syllable_amount:
                    words = tokenize_sentence(candidate_text)
                    if len(words) == 0:
                        continue
                    last_word = words[-1]
                    
                    if do_two_words_rhyme(last_word, self.word):
                        print(last_word, scores[i][token])
                        scores[i][token] = scores[i][token] - scores[i][token]*0.8
                        print(last_word, scores[i][token])
                    # else:
                    #     scores[i][token] = scores[i][token] + scores[i][token]*0.1

        return scores
    
    def is_beam_constraint_active(self):
        return False
    
    def is_stopping_criteria_active(self):
        return False
    
    def is_logits_processor_active(self):
        return True