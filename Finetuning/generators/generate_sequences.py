import os
import torch

from Finetuning.support import support
from Finetuning.support.support import choose_from_top, DELIM_EOS, DELIM_SOP, DELIM_SOC, DELIM_EOC, DELIM_SOS, cprint, Color


def check_response_constraint(sequence, activity_a, activity_b):
    activities = sequence.split(" ")
    try:
        index_a = activities.index(activity_a)
        index_b = activities.index(activity_b, index_a + 1)
        return index_b > index_a
    except ValueError:
        return False


def generate_sequences(model, tokenizer, validation_list, n_to_generate, output_file_path, path_table=None, constraints=None, n_min_constraints=0, n_max_constraints=3, verbose=True, avoid_cfls_calculation=False, seed=None):
    model.eval()
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    activity_a = "Assign seriousness"
    activity_b = "Take in charge ticket"
    counter_response_satisfied = 0

    np_validation_list = support.sequences2numpy(validation_list)
    np_generated_list = []
    result_table = []
    with torch.no_grad():
        sequence_num = 0
        counter_satisfied = 0
        counter_generated = n_to_generate
        counter_detailed_satisfied = 0
        counter_detailed_satisfied_total = 0
        while counter_generated > 0:
            # Building constraints for the sequence
            if constraints is not None:
                selected_constraints, stringed_constraints = support.generate_subset_stringed_constraints(n_min_constraints, n_max_constraints, constraints)
                if seed is not None:
                    input = seed + DELIM_SOC + stringed_constraints[:-1] + DELIM_EOC + DELIM_SOS

                else:
                    input = DELIM_SOP + DELIM_SOC + stringed_constraints[:-1] + DELIM_EOC + DELIM_SOS

            else:
                if seed is not None:
                    input = seed + DELIM_SOC + DELIM_EOC + DELIM_SOS

                else:
                    input = DELIM_SOP + DELIM_SOC + DELIM_EOC + DELIM_SOS

            cur_ids = torch.tensor(tokenizer.encode(input)).unsqueeze(0).to(support.device)
            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0, -1], dim=0)
                if i < 3:
                    n = 20

                else:
                    n = 3

                next_token_id = choose_from_top(softmax_logits.to("cpu").numpy(), n=n)
                cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(support.device) * next_token_id], dim=1)
                if next_token_id in tokenizer.encode(DELIM_EOS):
                    break

            counter_generated -= 1
            sequence_num = sequence_num + 1
            output_list = list(cur_ids.squeeze().to("cpu").numpy())
            output_text = tokenizer.decode(output_list)
            print(f"Generated sequence: {output_text}")  # <-- Aggiungi questa riga qui!

            if check_response_constraint(output_text, activity_a, activity_b):
                counter_response_satisfied += 1

            np_generated_list.append(support.sequence2numpy(output_text))
            if constraints is not None:
                result_detailed, counter_current_detailed = support.check_constraints(output_text, selected_constraints, completed=True, detailed=True)
                result = support.check_constraints(output_text, selected_constraints, completed=True, detailed=False)
                counter_satisfied += 1 if result else 0
                counter_detailed_satisfied += counter_current_detailed
                counter_detailed_satisfied_total += len(selected_constraints)

            with open(output_file_path, "a") as f:
                if verbose:
                    print("-" * 150)

                sentence = f"{output_text}"
                f.write(sentence)
                if verbose:
                    print(sentence)

                if constraints is not None:
                    if path_table is not None:
                        result_table.append(support.build_result_line(input, output_text, selected_constraints))

                    sentence = f"Satisfied: {result}\nStatus constraints (on the generated sequence): {result_detailed}"
                    if verbose:
                        cprint(sentence, Color.CYAN)

                    f.write(sentence)

        cprint(f"Sequences satisfying Response({activity_a}, {activity_b}): {counter_response_satisfied}")

        if not avoid_cfls_calculation:
            cfld_metric = support.get_log_similarity(np_validation_list, np_generated_list)
            cprint(f"CFLD metric between the original test set and generated set of sequences (lower is better): {cfld_metric:9.4f}", Color.MAGENTA)
            cprint(f"CFLS metric between the original test set and generated set of sequences (higher is better): {1 - cfld_metric:9.4f}", Color.MAGENTA)

