from Constraint import Constraint
from SongUtils import  rhyming_words_to_tokens_and_syllable_count, load_rhyming_dicts, get_syllable_count_of_sentence, _get_rhyming_words, _get_rhyming_lines, _do_two_words_rhyme
import torch
################################################ CONSTRAINT CLASS ################################################


class RhymingConstraintLBL(Constraint):
    def __init__(self, tokenizer, start_token=None, top_k_rhyme_words=100, rhyme_type='perfect'):
        self.tokenizer = tokenizer
        self.rhyming_word = None
        self.required_syllable_count = None
        self.rhyming_words = None
        self.rhyming_words_tokens = None
        self.rhyming_words_to_ignore = []
        self.max_syllable_count = None
        self.start_token = start_token
        self.top_k_rhyme_words = top_k_rhyme_words
        self.rhyme_type = rhyme_type
        load_rhyming_dicts()
    
    def get_rhyming_lines(self, lines):
        return _get_rhyming_lines(lines, self.rhyme_type)

    def set_rhyming_word(self, rhyming_word):
        print('Setting rhyming word to: ', rhyming_word)
        if rhyming_word is None:
            self.rhyming_word = None
            self.rhyming_words = None
            self.rhyming_words_tokens = None
            self.max_syllable_count = None
            return
        self.rhyming_word = rhyming_word
        self.rhyming_words = _get_rhyming_words(rhyming_word, self.rhyme_type)
        if rhyming_word in self.rhyming_words:
            self.rhyming_words.remove(rhyming_word)
        for word in self.rhyming_words_to_ignore:
            if word in self.rhyming_words:
                self.rhyming_words.remove(word)
        self.rhyming_words_tokens, self.max_syllable_count = rhyming_words_to_tokens_and_syllable_count(self.tokenizer, self.rhyming_words, start_token=self.start_token)

        if self.max_syllable_count > 3:
            self.max_syllable_count = 3
    
    def set_required_syllable_count(self, required_syllable_count):
        self.required_syllable_count = required_syllable_count
    
    def add_rhyming_words_to_ignore(self, word_to_ignore):
        self.rhyming_words_to_ignore.append(word_to_ignore)
    
    def reset_rhyming_words_to_ignore(self):
        self.rhyming_words_to_ignore = []

    

        


    
    def apply_beam_constraint(self, next_token, next_score, input_ids, cur_len, length_penalty):
        if self.rhyming_word is None:
            return next_score
        
        if self.required_syllable_count is None:
            raise Exception('Required syllable count not set')
        
        previous_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        current_token = self.tokenizer.decode(next_token, skip_special_tokens=True)
        candidate_text = previous_text + current_token
        last_line = candidate_text.split('\n')[-1]
        current_syllable_count = get_syllable_count_of_sentence(last_line)
        
        if current_syllable_count == self.required_syllable_count:
            
            last_word = last_line.split(' ')[-1]

            if _do_two_words_rhyme(last_word, self.rhyming_word, self.rhyme_type):
                
                next_score = next_score - next_score*0.95
                return next_score
            
            ## Because the syllable constraints stops the generation when the syllable count is reached,it can happpen sometimes that last consonants are not generated, as no extra syllable is added when adding consonants
            ## This may cause it be a near perfect rhyme even though perfect rhymes were chosen. Therefore we check if the last vowels are the same.

            if _do_two_words_rhyme(last_word, self.rhyming_word, "assonant"):
                
                next_score = next_score - next_score*0.9
                
                return next_score
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
            started_with_rhyming_word = False
            prev_scores = scores[i].clone()
            for rhyming_word in self.rhyming_words_tokens:
                if input[-1] in rhyming_word['tokens']:
                    
                    current_token_index = rhyming_word['tokens'].index(input[-1])
                    
                    #verify the whole word is in the input
                    # print(input[-current_token_index - 1:])
                    # print(rhyming_word['tokens'][:current_token_index + 1])
                    if input[-current_token_index - 1:].tolist() != rhyming_word['tokens'][:current_token_index + 1]:
                        continue
                    
                    input_without_rhyme = input[:len(input)-current_token_index - 1]
                    text = self.tokenizer.decode(input_without_rhyme, skip_special_tokens=True)
                    last_line = text.split('\n')[-1]
                    syllable_count_without_rhyme = get_syllable_count_of_sentence(last_line)
                    if syllable_count_without_rhyme + rhyming_word['syllable_count'] != self.required_syllable_count:
                        continue
                    next_token = rhyming_word['tokens'][current_token_index + 1]
                    score = prev_scores[next_token]
                    if not started_with_rhyming_word:
                        scores[i] = abs(scores[i]) * float('-inf')
                        started_with_rhyming_word = True
                    scores[i][next_token] = score + 0.99*abs(score)


                    
            if started_with_rhyming_word:
                return scores
            
            
            #get the rhyming words that have the same syllable count as the syllables left
            rhyming_words_with_syllables_left = [word for word in self.rhyming_words_tokens if word['syllable_count'] == syllables_left]
            if len(rhyming_words_with_syllables_left) == 0:
                return scores
            
            
            
            #Get the top k best tokens
            scores_rhyme_word = [scores[i][word['tokens'][0]] for word in rhyming_words_with_syllables_left]
            ordered_rhyming_words = [x for _, x in sorted(zip(scores_rhyme_word, rhyming_words_with_syllables_left), key=lambda pair: pair[0], reverse=True)]
            

            
            # #get the rhyming words that have the same syllable count as the syllables left and the first token of one of the rhyming words is the last token of the sentence
            for rhyming_word in ordered_rhyming_words[:self.top_k_rhyme_words]:
                first_token = rhyming_word['tokens'][0]
                
                #if the first token isn't in the best tokens, we continue
                score = prev_scores[first_token]
                
                #print('first token: ', first_token, ' score: ', score + 1*abs(score))
                
                if score != float('-inf'):
                    scores[i][first_token] = score + 0.9*abs(score)
                    
        return scores
    
    def is_beam_constraint_active(self):
        return True
    
    def is_stopping_criteria_active(self):
        return False
    
    def is_logits_processor_active(self):
        return True


