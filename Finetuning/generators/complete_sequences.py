import torch
import csv

from Finetuning.support import support
from Finetuning.support.support import find_middle_index, MAX_SEQ_LEN, choose_from_top, DELIM_EOS, get_string_between, DELIM_SOC, DELIM_EOC, DELIM_SOS, Color, cprint

def complete_sequences(model, tokenizer, validation_list, test_dataset, output_file_path, path_table=None, supply_constraints=True, verbose=True, avoid_cfls_calculation=False):
    counter_satisfied_all = 0
    counter_satisfied_input = 0
    counter_detailed_satisfied = 0
    counter_detailed_satisfied_total = 0
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

                if raw_constraints is not None:
                    if path_table is not None:
                        result_table.append(support.build_result_line(input, output_text, raw_constraints))

                    result_on_current_detailed, counter_current_detailed = support.check_constraints(input + output_text, raw_constraints, completed=True, detailed=True)
                    result_on_input_detailed, _ = support.check_constraints(input, raw_constraints, completed=False, detailed=True)
                    result_on_current = support.check_constraints(input + output_text, raw_constraints, completed=True, detailed=False)
                    result_on_input = support.check_constraints(input, raw_constraints, completed=False, detailed=False)
                    counter_satisfied_all += 1 if result_on_current else 0
                    counter_satisfied_input += 1 if result_on_input else 0
                    counter_detailed_satisfied += counter_current_detailed
                    counter_detailed_satisfied_total += len(raw_constraints)
                    if verbose:
                        cprint(f"All constraints satisfied (on the completed sequence): {result_on_current}", Color.CYAN)
                        cprint(f"All constraints satisfied (on the input sequence): {result_on_input}", Color.CYAN)
                        cprint(f"Status constraints (on the completed sequence): {result_on_current_detailed}", Color.CYAN)
                        cprint(f"Status constraints (on the input sequence): {result_on_input_detailed}", Color.CYAN)

        if raw_constraints is not None:
            if not supply_constraints:
                cprint("Constraints not supplied in the input!", Color.MAGENTA)

            else:
                if path_table is not None:
                    support.save_csv_result(result_table, path_table)

            if counter_detailed_satisfied_total == 0:
                cprint(f"Constraints satisfied (on the completed sequences): {counter_detailed_satisfied}/{counter_detailed_satisfied_total} - 0%", Color.MAGENTA)

            else:
                cprint(f"Constraints satisfied (on the completed sequences): {counter_detailed_satisfied}/{counter_detailed_satisfied_total} - {int((counter_detailed_satisfied/counter_detailed_satisfied_total)*100)}%", Color.MAGENTA)

            cprint(f"Sequences completed with all constraints satisfied (on the completed sequences): {counter_satisfied_all}/{len(test_dataset)} - {int((counter_satisfied_all/len(test_dataset))*100)}%", Color.MAGENTA)
            cprint(f"Sequences completed with all constraints satisfied (on the input sequences): {counter_satisfied_input}/{len(test_dataset)} - {int((counter_satisfied_input/len(test_dataset))*100)}%", Color.MAGENTA)

        if not avoid_cfls_calculation:
            cfld_metric = support.get_log_similarity(np_validation_list, np_generated_list)
            cprint(f"CFLD metric between the original test set and generated set of sequences (lower is better): {cfld_metric:9.4f}", Color.MAGENTA)
            cprint(f"CFLS metric between the original test set and generated set of sequences (higher is better): {1 - cfld_metric:9.4f}", Color.MAGENTA)