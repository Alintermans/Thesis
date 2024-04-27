from SongUtils import does_string_contain_newline

class NewLineStoppingCriteria():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.original_prompt_length = 0
        self.disable = True
    
    def set_prompt_length(self, prompt_length):
        self.original_prompt_length = prompt_length

    def stopping_criteria(self, input_ids, score, **kwargs):
        if self.disable:
            return False
        for input in input_ids:
            decoded_text = self.tokenizer.decode(input, skip_special_tokens=True)[self.original_prompt_length:]
            if does_string_contain_newline(decoded_text):
                return True
        return False
