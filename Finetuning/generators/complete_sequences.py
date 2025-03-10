import os
import torch
import csv

from Finetuning.support import support
from Finetuning.support.support import (
    find_middle_index, MAX_SEQ_LEN, choose_from_top, DELIM_EOS,
    get_string_between, DELIM_SOC, DELIM_EOC, DELIM_SOS, Color, cprint
)


def check_response_constraint(sequence, activity_a, activity_b):
    # Trova la posizione della prima occorrenza dell'attività A
    index_a = sequence.find(activity_a)
    if index_a == -1:
        return False  # Attività A non trovata

    # Trova la posizione della prima occorrenza dell'attività B dopo A
    index_b = sequence.find(activity_b, index_a + len(activity_a))
    if index_b == -1:
        return False  # Attività B non trovata

    # Se l'indice di B è maggiore di A, il vincolo è rispettato
    return index_b > index_a


def complete_sequences(model, tokenizer, validation_list, test_dataset, output_file_path, path_table=None, supply_constraints=True, verbose=True, avoid_cfls_calculation=False):
    counter_satisfied_all = 0
    counter_satisfied_input = 0
    counter_detailed_satisfied = 0
    counter_detailed_satisfied_total = 0
    counter_response_satisfied = 0

    activity_a = "Assign seriousness"
    activity_b = "Take in charge ticket"

    np_validation_list = support.sequences2numpy(validation_list)
    np_generated_list = []
    result_table = []

    with torch.no_grad():
        with open(output_file_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["constraints", "", "org:resource", "lifecycle:transition", "concept:name", "time:timestamp", "case:REG_DATE", "case:concept:name", "case:AMOUNT_REQ"])
            for idx, value in enumerate(test_dataset):
                if isinstance(value, dict):
                    input = value["prompt"]
                    output = value["chosen"]
                    raw_constraints = value["constraints"]

                else:
                    middle_index = find_middle_index(value[0])
                    input = value[0][:middle_index]
                    output = value[0][middle_index + 1:]
                    raw_constraints = None

                if not supply_constraints:
                    input = support.remove_substring_between(input, DELIM_SOC, DELIM_EOC)

                sequence_finished = False
                cur_ids = torch.tensor(tokenizer.encode(input)).unsqueeze(0).to(support.device)
                # Skipping sample from dataset if it is longer than MAX_SEQ_LEN
                if cur_ids.size()[1] > MAX_SEQ_LEN:
                    continue

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
                        sequence_finished = True
                        break

                output_list = list(cur_ids.squeeze().to("cpu").numpy())
                output_text = tokenizer.decode(output_list)

                print(f"Completed sequence: {output_text}")  # Aggiunto per vedere le sequenze completate

                if check_response_constraint(output_text, activity_a, activity_b):
                    counter_response_satisfied += 1

            cprint(f"Sequences satisfying Response({activity_a}, {activity_b}): {counter_response_satisfied}")

        if not avoid_cfls_calculation:
            cfld_metric = support.get_log_similarity(np_validation_list, np_generated_list)
            cprint(f"CFLD metric between the original test set and generated set of sequences (lower is better): {cfld_metric:9.4f}", Color.MAGENTA)
            cprint(f"CFLS metric between the original test set and generated set of sequences (higher is better): {1 - cfld_metric:9.4f}", Color.MAGENTA)

