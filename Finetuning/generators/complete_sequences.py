import torch
import csv

from Finetuning.support import support
from Finetuning.support.support import find_middle_index, MAX_SEQ_LEN, choose_from_top, DELIM_EOS, get_string_between, DELIM_SOC, DELIM_EOC, DELIM_SOS, Color, cprint


def check_response_constraint(sequence, activity_a, activity_b):
    activities = sequence.split(" ")
    try:
        index_a = activities.index(activity_a)
        index_b = activities.index(activity_b, index_a + 1)
        return index_b > index_a
    except ValueError:
        return False


def complete_sequences(model, tokenizer, validation_list, test_dataset, output_file_path, path_table=None, supply_constraints=True, verbose=True, avoid_cfls_calculation=False):
    counter_response_satisfied = 0
    activity_a = "Assign seriousness"
    activity_b = "Take in charge ticket"
    np_validation_list = support.sequences2numpy(validation_list)
    np_generated_list = []

    with torch.no_grad():
        with open(output_file_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["constraints", "", "org:resource", "lifecycle:transition", "concept:name", "time:timestamp", "case:REG_DATE", "case:concept:name", "case:AMOUNT_REQ"])
            for idx, value in enumerate(test_dataset):
                if isinstance(value, dict):
                    input = value["prompt"]
                    output = value["chosen"]

                else:
                    middle_index = find_middle_index(value[0])
                    input = value[0][:middle_index]
                    output = value[0][middle_index + 1:]

                if not supply_constraints:
                    input = support.remove_substring_between(input, DELIM_SOC, DELIM_EOC)

                if verbose:
                    print("-" * 150)
                    print("Sequence expected:  " + output)
                    print("Sequence supplied:  " + input)
                    print("Sequence complete:  " + input + " " + output)

                sequence_finished = False
                cur_ids = torch.tensor(tokenizer.encode(input)).unsqueeze(0).to(support.device)
                # Skipping sample from dataset if it is longer than MAX_SEQ_LEN
                if cur_ids.size()[1] > MAX_SEQ_LEN:
                    if verbose:
                        print("Sequence too much long, skipping!")

                    continue

                for i in range(100):
                    outputs = model(cur_ids, labels=cur_ids)
                    loss, logits = outputs[:2]
                    # Taking the first(from only one in this case) batch and the last predicted embedding
                    softmax_logits = torch.softmax(logits[0, -1], dim=0)
                    if i < 3:
                        n = 20

                    else:
                        n = 3

                    # Randomly (from the top_n probability distribution) select the next word
                    next_token_id = choose_from_top(softmax_logits.to("cpu").numpy(), n=n)
                    # Adding the last word to the running sequence
                    cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(support.device) * next_token_id], dim=1)

                    if next_token_id in tokenizer.encode(DELIM_EOS):
                        sequence_finished = True
                        break

                output_list = list(cur_ids.squeeze().to("cpu").numpy())
                output_text = tokenizer.decode(output_list)
                print(f"Generated sequence: {output_text}")  # <-- Aggiungi questa riga qui!

                if check_response_constraint(output_text, activity_a, activity_b):
                    counter_response_satisfied += 1

                np_generated_list.append(support.sequence2numpy(output_text))
                constraints = get_string_between(DELIM_SOC, DELIM_EOC, output_text)
                events = get_string_between(DELIM_SOS, DELIM_EOS, output_text)
                for tk in events.split(" "):
                    if not tk == "":
                        splitted_event = tk.split("_")
                        lifecycle_transition = splitted_event[0]
                        concept_name = '_'.join(splitted_event[1:])
                        csv_writer.writerow([constraints, "", "", lifecycle_transition, concept_name, "", "", idx, ""])

                if sequence_finished:
                    if verbose:
                        cprint("Sequence generated: " + output_text, Color.GREEN)

                else:
                    if verbose:
                        cprint("Sequence generated: " + output_text, Color.RED)

            cprint(f"Sequences satisfying Response({activity_a}, {activity_b}): {counter_response_satisfied}")

        if not avoid_cfls_calculation:
            cfld_metric = support.get_log_similarity(np_validation_list, np_generated_list)
            cprint(f"CFLD metric between the original test set and generated set of sequences (lower is better): {cfld_metric:9.4f}", Color.MAGENTA)
            cprint(f"CFLS metric between the original test set and generated set of sequences (higher is better): {1 - cfld_metric:9.4f}", Color.MAGENTA)

