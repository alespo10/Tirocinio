class SequenceValidator:
    def __init__(self, activity_a, activity_b):
        self.activity_a = activity_a
        self.activity_b = activity_b

    #Existence templates
    def check_existence(self, sequence):
        return sequence.find(self.activity_a) != -1

    def init(self, sequence):
        return sequence.find(self.activity_b) == 0

    #Relation templates
    def check_response_constraint(self, sequence):
        index_a = sequence.find(self.activity_a)
        index_b = sequence.find(self.activity_b)
        if index_a == -1:
            return index_b != -1 or index_b == -1
        if index_b == -1:
            return False
        return index_b > index_a

    def check_alternate_precedence(self, sequence):
        last_seen_a = -1
        for i, event in enumerate(sequence):
            if event == self.activity_a:
                last_seen_a = i
            elif event == self.activity_b:
                if last_seen_a == -1 or any(e == self.activity_b for e in sequence[last_seen_a + 1: i]):
                    return False
        return True

    def check_chain_precedence(self, sequence):
        for i in range(1, len(sequence)):
            if sequence[i] == self.activity_b and sequence[i - 1] != self.activity_a:
                return False
        return True

    def check_not_coexistence(self, sequence):
        contains_a = self.activity_a in sequence
        contains_b = self.activity_b in sequence
        return not (contains_a and contains_b)


