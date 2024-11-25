from Finetuning.support import support
from Finetuning.support.support import DELIM_SOS, DELIM_EOS
from textrl import TextRLEnv


class ReinforcementLearningEnvironment(TextRLEnv):

    def get_reward(self, input_item, predicted_list, is_finish):
        input = input_item["prompt"]
        constraints = input_item["constraints"]
        sequence = support.get_string_between(DELIM_SOS, DELIM_EOS, input + " " + " ".join([item for sublist in predicted_list for item in sublist]))
        is_consistent = self._check_consistency(sequence)
        if not is_consistent:
            reward = -10

        else:
            compilancy_score = self._check_constraints(sequence, constraints)
            # all constraints satisfied
            if compilancy_score == 1:
                if is_finish:
                    reward = 2 * 2

                else:
                    reward = 1 * 2

            # no one constraint satisfied
            elif compilancy_score == 0:
                reward = -7

            # some constraint satisfied
            else:
                reward = compilancy_score * 2

        return [reward]

    def _check_constraints(self, sequence, constraints):
        if len(constraints) == 0:
            return 1

        n_fails = 0
        for constraint in constraints:
            if not support.satisfies(sequence, constraint):
                n_fails += 1

        return 1 - ( n_fails / len(constraints))

    def _check_consistency(self, sequence):
        return support.check_consistency(sequence)
