import os
import torch

from Finetuning.support import support
from Finetuning.support.support import choose_from_top, DELIM_EOS, DELIM_SOP, DELIM_SOC, DELIM_EOC, DELIM_SOS, cprint, Color

def generate_sequences(model, tokenizer, validation_list, n_to_generate, output_file_path, verbose=True, avoid_cfls_calculation=False, seed=None):
    model.eval()
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    np_validation_list = support.sequences2numpy(validation_list)
    np_generated_list = []

    with torch.no_grad():
        counter_generated = n_to_generate

        while counter_generated > 0:
            # Costruzione dell'input per il modello
            input = seed + DELIM_SOS if seed else DELIM_SOP + DELIM_SOS
            cur_ids = torch.tensor(tokenizer.encode(input)).unsqueeze(0).to(support.device)

            # Generazione della sequenza
            for _ in range(100):  # Limitiamo la lunghezza massima
                outputs = model(cur_ids, labels=cur_ids)
                _, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0, -1], dim=0)

                next_token_id = choose_from_top(softmax_logits.to("cpu").numpy(), n=3)
                cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(support.device) * next_token_id], dim=1)

                if next_token_id in tokenizer.encode(DELIM_EOS):  # Fine della sequenza
                    break

            counter_generated -= 1

            # Decodifica della sequenza generata
            output_list = list(cur_ids.squeeze().to("cpu").numpy())
            output_text = tokenizer.decode(output_list)

            # Estrai solo i nomi delle attività
            activities = []
            for sentence in output_text.split(". "):
                if "the activity" in sentence:
                    parts = sentence.split(" ")
                    activity_name = " ".join(parts[2:4])  # Estrai solo il nome dell'attività
                    activities.append(activity_name)

            np_generated_list.append(support.sequence2numpy(" → ".join(activities)))

            # Salva la sequenza nel file
            with open(output_file_path, "a") as f:
                f.write(" → ".join(activities) + "\n")

            if verbose:
                print(f"Generated activities: {' → '.join(activities)}")

    # Calcolo delle metriche CFLS e CFLD
    if not avoid_cfls_calculation:
        cfld_metric = support.get_log_similarity(np_validation_list, np_generated_list)
        cprint(f"CFLD metric (lower is better): {cfld_metric:9.4f}", Color.MAGENTA)
        cprint(f"CFLS metric (higher is better): {1 - cfld_metric:9.4f}", Color.MAGENTA)
