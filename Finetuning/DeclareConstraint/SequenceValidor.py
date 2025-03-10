class SequenceValidator:
    def __init__(self, activity_a, activity_b):
        self.activity_a = activity_a
        self.activity_b = activity_b

    def check_response_constraint(self, sequence):
        index_a = sequence.find(self.activity_a)
        index_b = sequence.find(self.activity_b)
        if index_a == -1:
            return index_b != -1 or index_b == -1
        if index_b == -1:
            return False
        return index_b > index_a




