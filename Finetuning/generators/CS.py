import torch

from Finetuning.support import support
from Finetuning.support.support import find_middle_index, MAX_SEQ_LEN, choose_from_top, DELIM_EOS, get_string_between, DELIM_SOC, DELIM_EOC, DELIM_SOS, Color, cprint


def complete_sequences(model, tokenizer, validation_list, test_dataset, output_file_path, verbose=True,
                       avoid_cfls_calculation=False):
    """
    Completa sequenze parziali generando solo i nomi delle attività e calcola le metriche CFLS e CFLD.
    """
    np_validation_list = support.sequences2numpy(validation_list)  # Converte le sequenze originali
    np_generated_list = []  # Lista per salvare le sequenze generate

    with torch.no_grad():
        with open(output_file_path, "w") as f:
            for idx, value in enumerate(test_dataset):
                # Ottieni la sequenza di input
                if isinstance(value, dict):
                    input_sequence = value["prompt"]  # Sequenza parziale da completare
                else:
                    middle_index = find_middle_index(value[0])
                    input_sequence = value[0][:middle_index]  # Ottieni la parte iniziale della sequenza

                # Controllo per input vuoti o malformati
                if not input_sequence.strip():  # Controlla se l'input è vuoto o solo spazi
                    if verbose:
                        print(f"Skipping empty or invalid input at index {idx}")
                    continue

                if verbose:
                    print("-" * 150)
                    print(f"Input sequence: {input_sequence}")

                # Tokenizzazione e controllo
                cur_ids = tokenizer.encode(input_sequence)
                if not cur_ids:  # Controllo per token vuoti
                    if verbose:
                        print(f"Skipping invalid tokenized input at index {idx}: {input_sequence}")
                    continue

                cur_ids = torch.tensor(cur_ids).unsqueeze(0).to(support.device)

                # Skipping samples if they exceed MAX_SEQ_LEN
                if cur_ids.size()[1] > MAX_SEQ_LEN:
                    if verbose:
                        print("Sequence too long, skipping!")
                    continue

                # Generazione sequenza
                for i in range(100):  # Limitiamo la lunghezza della sequenza completata
                    outputs = model(cur_ids, labels=cur_ids)
                    _, logits = outputs[:2]
                    softmax_logits = torch.softmax(logits[0, -1], dim=0)

                    n = 20 if i < 3 else 3  # N cambia in base al numero di token generati finora
                    next_token_id = choose_from_top(softmax_logits.to("cpu").numpy(), n=n)
                    cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(support.device) * next_token_id], dim=1)

                    if next_token_id in tokenizer.encode(
                            DELIM_EOS):  # Se viene generato il token di fine sequenza, interrompe
                        break

                # Decodifica e post-elaborazione
                output_list = list(cur_ids.squeeze().to("cpu").numpy())
                output_text = tokenizer.decode(output_list)

                # Estrai solo i nomi delle attività
                activities = []
                for sentence in output_text.split(". "):
                    if "the activity" in sentence:
                        parts = sentence.split(" ")
                        activity_name = " ".join(parts[2:4])  # Estrarre solo il nome dell'attività
                        activities.append(activity_name)

                # Aggiungi la sequenza di attività completata alla lista
                np_generated_list.append(support.sequence2numpy(" → ".join(activities)))

                # Salva la sequenza generata nel file
                f.write(" → ".join(activities) + "\n")

                if verbose:
                    print(f"Completed sequence: {' → '.join(activities)}")

    # Calcolo delle metriche CFLS e CFLD
    if not avoid_cfls_calculation:
        cfld_metric = support.get_log_similarity(np_validation_list, np_generated_list)
        cprint(f"CFLD metric (lower is better): {cfld_metric:9.4f}", Color.MAGENTA)
        cprint(f"CFLS metric (higher is better): {1 - cfld_metric:9.4f}", Color.MAGENTA)
