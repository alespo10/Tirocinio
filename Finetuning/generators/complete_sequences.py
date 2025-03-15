import os
import torch
import csv

from Finetuning.DeclareConstraint.SequenceValidor import SequenceValidator
from Finetuning.support import support
from Finetuning.support.support import (
    find_middle_index, MAX_SEQ_LEN, choose_from_top, DELIM_EOS,
    get_string_between, DELIM_SOC, DELIM_EOC, DELIM_SOS, Color, cprint
)



def complete_sequences(model, tokenizer, validation_list, test_dataset, output_file_path, path_table=None, supply_constraints=True, verbose=True, avoid_cfls_calculation=True):
    counter_satisfied_all = 0
    counter_satisfied_input = 0
    counter_detailed_satisfied = 0
    counter_detailed_satisfied_total = 0
    counter_response_satisfied = 0

    OSMO = "O_Sent (mail and online)"
    OR = "O_Returned"
    OCO = "O_Create Offer"

    counter_response_satisfied = 0
    counter_chain_precedence = 0
    counter_init = 0
    counter_not_coexistence = 0

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

                for i in range(200):  # Aumenta il numero massimo di iterazioni
                    outputs = model(cur_ids, labels=cur_ids)
                    loss, logits = outputs[:2]
                    softmax_logits = torch.softmax(logits[0, -1], dim=0)

                    if i < 10:  # Estendi la fase esplorativa ai primi 10 token
                        n = 20
                    else:
                        n = 5  # Permetti più libertà nella generazione

                    next_token_id = choose_from_top(softmax_logits.to("cpu").numpy(), n=n)
                    cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(support.device) * next_token_id], dim=1)

                    if next_token_id in tokenizer.encode(DELIM_EOS) and len(cur_ids[0]) > 150:
                        sequence_finished = True
                        break

                output_list = list(cur_ids.squeeze().to("cpu").numpy())
                output_text = tokenizer.decode(output_list)

                print(f"Completed sequence: {output_text}")

                validator = SequenceValidator(OSMO, OR)
                if validator.check_response_constraint(output_text):
                    counter_response_satisfied += 1

                validator = SequenceValidator(OSMO, OR)
                if validator.check_chain_precedence(output_text):
                    counter_chain_precedence += 1

                validator = SequenceValidator(OCO)
                if validator.init(output_text):
                    counter_init += 1

                validator = SequenceValidator(OR, OCO)
                if validator.check_not_coexistence(output_text):
                    counter_not_coexistence += 1

        cprint(f"Sequences satisfying Response({OSMO}, {OR}): {counter_response_satisfied}")
        cprint(f"Sequences satisfying Chain Precedence({OSMO}, {OR}): {counter_chain_precedence}")
        cprint(f"Sequences satisfying Init({OCO}): {counter_init}")
        cprint(f"Sequences satisfying Not CoExistence({OR}, {OCO}): {counter_not_coexistence}")

        if not avoid_cfls_calculation:
            cfld_metric = support.get_log_similarity(np_validation_list, np_generated_list)
            cprint(f"CFLD metric between the original test set and generated set of sequences (lower is better): {cfld_metric:9.4f}", Color.MAGENTA)
            cprint(f"CFLS metric between the original test set and generated set of sequences (higher is better): {1 - cfld_metric:9.4f}", Color.MAGENTA)

