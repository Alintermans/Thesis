from Constraint import Constraint
from SongUtils import  get_perfect_rhyming_words, rhyming_words_to_tokens_and_syllable_count, load_rhyming_dicts, get_syllable_count_of_sentence, get_assonant_rhyming_words
import torch
################################################ CONSTRAINT CLASS ################################################


class RhymingConstraintLBL(Constraint):
    def __init__(self, tokenizer, start_token=None):
        self.tokenizer = tokenizer
        self.rhyming_word = None
        self.required_syllable_count = None
        self.rhyming_words = None
        self.rhyming_words_tokens = None
        self.rhyming_words_to_ignore = []
        self.max_syllable_count = None
        self.start_token = start_token
        load_rhyming_dicts()

    def set_rhyming_word(self, rhyming_word):
        print('Setting rhyming word to: ', rhyming_word)
        self.rhyming_word = rhyming_word
        self.rhyming_words = get_perfect_rhyming_words(rhyming_word)
        if rhyming_word in self.rhyming_words:
            self.rhyming_words.remove(rhyming_word)
        for word in self.rhyming_words_to_ignore:
            if word in self.rhyming_words:
                self.rhyming_words.remove(word)
        self.rhyming_words_tokens, self.max_syllable_count = rhyming_words_to_tokens_and_syllable_count(self.tokenizer, self.rhyming_words, start_token=self.start_token)

        if self.max_syllable_count > 4:
            self.max_syllable_count = 4
    
    def set_required_syllable_count(self, required_syllable_count):
        self.required_syllable_count = required_syllable_count
    
    def add_rhyming_words_to_ignore(self, word_to_ignore):
        self.rhyming_words_to_ignore.append(word_to_ignore)
    
    def reset_rhyming_words_to_ignore(self):
        self.rhyming_words_to_ignore = []

    

        


    
    def apply_beam_constraint(self, next_token, next_score, input_ids, cur_len, length_penalty):
        
        return next_score
    
    def stopping_criteria(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        return False
    
    def logits_processor(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.rhyming_word is None:
            return scores
        if self.required_syllable_count is None:
            raise Exception('Required syllable count not set')


        for i in range(len(input_ids)):
            input = input_ids[i]
            sentence = self.tokenizer.decode(input, skip_special_tokens=True)
            last_line = sentence.split('\n')[-1]
            syllable_count = get_syllable_count_of_sentence(last_line)
            syllables_left = self.required_syllable_count - syllable_count

            if syllables_left <= 0 or syllables_left > self.max_syllable_count:
                return scores

            #first check if the rhyming word is already initialized and the first token of one of the rhyming words is the last token of the sentence
            for rhyming_word in self.rhyming_words_tokens:
                if input[-1] in rhyming_word['tokens']:
                    current_token_index = rhyming_word['tokens'].index(input[-1])
                    if current_token_index + 1 == len(rhyming_word['tokens']):
                        return scores
                    next_token = rhyming_word['tokens'][current_token_index + 1]
                    scores[i] = abs(scores[i]) * float('-inf')
                    scores[i][next_token] = 0
                    return scores
            
            
            #get the rhyming words that have the same syllable count as the syllables left
            rhyming_words_with_syllables_left = [word for word in self.rhyming_words_tokens if word['syllable_count'] == syllables_left]
            if len(rhyming_words_with_syllables_left) == 0:
                return scores
            
            # #get the rhyming words that have the same syllable count as the syllables left and the first token of one of the rhyming words is the last token of the sentence
            for rhyming_word in rhyming_words_with_syllables_left:
                first_token = rhyming_word['tokens'][0]
                score = scores[i][first_token]
                #print('first token: ', first_token, ' score: ', score + 1*abs(score))
                if score != float('-inf'):
                    scores[i][first_token] = score + 0.8*abs(score)
        

                

        return scores
                




        return scores
    
    def is_beam_constraint_active(self):
        return False
    
    def is_stopping_criteria_active(self):
        return False
    
    def is_logits_processor_active(self):
        return True


