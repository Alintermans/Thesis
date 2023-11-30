from transformers import Constraint



# this is a constraint that checcks that the number of syllables per generated line is 8. 
class SyllableConstraint(Constraint):
    # def __init__(self, tokenizer, model, max_syllables=8):
    #     super(Constraint, self).__init__()

    #     self.tokenizer = tokenizer
    #     self.max_syllables = max_syllables

    def __init__(self, tokenizer, max_syllables=8):
        super(Constraint, self).__init__()

        self.tokenizer = tokenizer
        self.max_syllables = max_syllables
        self.current_sequence = []
        self.current_sequence_syllables = 0
        self.last_good_token_id = None
        self.completed = False
        self.seqlen = 100
        self.max_seqlen = 100
    


    def advance(self):
        return [self.last_good_token_id]

    def does_advance(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")
        if self.completed:
            return False
        self.last_good_token_id = token_id
        return True

    def update(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        self.current_sequence.append(token_id)
        self.current_sequence_syllables += 1

        print(self.current_sequence)

        if self.current_sequence_syllables > self.max_syllables:
            reset = True
            self.reset()

        if self.does_advance(token_id):
            stepped = True
            if self.current_sequence_syllables == self.max_syllables:
                completed = True
                self.completed = True
            
        return stepped, completed, reset

    def reset(self):
        self.completed = False
        self.current_sequence = []
        self.current_sequence_syllables = 0
        self.last_good_token_id = None


    def remaining(self):
        return self.max_syllables - self.current_sequence_syllables

    def copy(self, stateful=False):
        new_constraint = SyllableConstraint(tokenizer=self.tokenizer, max_syllables=self.max_syllables)

        if stateful:
            new_constraint.completed = self.completed
            new_constraint.max_seqlen = self.max_seqlen
            new_constraint.current_sequence = self.current_sequence
            new_constraint.current_sequence_syllables = self.current_sequence_syllables
            new_constraint.last_good_token_id = self.last_good_token_id
            new_constraint.seqlen = self.seqlen
            



        return new_constraint