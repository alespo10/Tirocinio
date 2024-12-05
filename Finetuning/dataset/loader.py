import gzip
import random
import pickle
import os

from progress.bar import FillingCirclesBar

from Finetuning.support import support
from Finetuning.support.support import base_directory, DELIM_SOC, DELIM_EOC, DELIM_SOS, DELIM_EOS, find_middle_index, DELIM_SOP


def _get_dataset_file_name(dataset_name, extension):
    return f"{base_directory}/Data/{dataset_name}/{dataset_name}.{extension}"


def _unzip_dataset(dataset_name_path):
    with open(dataset_name_path + ".gz", "rb") as inf, open(dataset_name_path, "w", encoding="utf8") as tof:
        decom_str = gzip.decompress(inf.read()).decode("utf-8")
        tof.write(decom_str)


def _read_generic_log(case_concept_name, concept_name, data_frame, seed):
    sequence_id = data_frame[case_concept_name].unique().tolist()
    sequences_list = []
    start_of_constraints_token = DELIM_SOC
    end_of_constraints_token = DELIM_EOC
    start_of_text_token = DELIM_SOS
    end_of_text_token = DELIM_EOS
    current_sequence_id = sequence_id[0]
    if seed is not None:
        current_sequence = seed + start_of_constraints_token + end_of_constraints_token + start_of_text_token

    else:
        current_sequence = DELIM_SOP + start_of_constraints_token + end_of_constraints_token + start_of_text_token

    for i in data_frame.index:
        if data_frame[case_concept_name][i] != current_sequence_id:
            # completing previous sequence
            current_sequence += end_of_text_token
            sequences_list.append(current_sequence)

            # starting new sequence
            if seed is not None:
                current_sequence = seed + start_of_constraints_token + end_of_constraints_token + start_of_text_token

            else:
                current_sequence = DELIM_SOP + start_of_constraints_token + end_of_constraints_token + start_of_text_token

            current_sequence_id = data_frame[case_concept_name][i]
            current_event = data_frame[concept_name][i].replace(" ", "_")
            current_sequence += current_event + " "

        else:
            current_event = data_frame[concept_name][i].replace(" ", "_")
            current_sequence += current_event + " "

    return sequences_list


def _read_bpi(data_frame):
    # preprocessing dataset structure
    data_frame["detailed_event"] = data_frame["lifecycle:transition"] + "_" + data_frame["concept:name"]
    data_frame = support.standardize_log(data_frame, "detailed_event")
    case_concept_name = "case:concept:name"
    concept_name = "detailed_event"
    # building result
    return _read_generic_log(case_concept_name, concept_name, data_frame)


def _read_bpiw(data_frame, seed):
    # preprocessing dataset structure
    case_concept_name = "case:concept:name"
    concept_name = "concept:name"
    data_frame = support.standardize_log(data_frame, "concept:name")
    # building result
    return _read_generic_log(case_concept_name, concept_name, data_frame, seed)


def _read_synthesized_dXX(data_frame, seed):
    # preprocessing dataset structure
    case_concept_name = "case:concept:name"
    concept_name = "concept:name"
    # building result
    return _read_generic_log(case_concept_name, concept_name, data_frame, seed)



def load_dataset(dataset_name, decl_path=None, decl_min_support=0.1, decl_max_support=0.8, decl_itemsets_support=0.9, decl_max_declare_cardinality=3, shuffle=True, decl_required=False, activities_to_remove=None, seed=None):
    # Percorso del file txt basato sul nome del dataset
    dataset_file_name = f"/content/Tirocinio/Data/{dataset_name}.txt"  # Assumendo che i file txt siano in una cartella "Dataset"

    # Controlla l'esistenza del file txt
    if not os.path.exists(dataset_file_name):
        raise Exception(f"Dataset file ({dataset_file_name}) not found!")

    # Lettura del file txt e caricamento delle sequenze
    with open(dataset_file_name, "r") as file:
        sequences_list = [line.strip() for line in file if line.strip()]  # Rimuove spazi e righe vuote

    # Opzione per mescolare le sequenze
    if shuffle:
        random.shuffle(sequences_list)

    # Nota: la parte di scoperta di vincoli dichiarativi (Declare constraints) rimane facoltativa.
    discovered_model = None

    return sequences_list, discovered_model

#
# def build_observation_list(dataset, load=False, path=None):
#     if load:
#         result = pickle.load(open(path, "rb"))
#         for sample in result:
#             for constraint in sample["constraints"]:
#                 support.add_available_templates(constraint["template"])
#
#         return result
#
#     else:
#         with FillingCirclesBar("Building observation list (dataset for RL):", max=len(dataset)) as bar:
#             observation_list = []
#             for sample in iter(dataset):
#                 input = sample[:find_middle_index(sample)]
#                 output = sample[find_middle_index(sample) + 1:]
#                 observation_list.append({"prompt": input, "input": input, "chosen": output, "constraints": "", "rejected": ""})
#                 bar.next()
#
#             bar.finish()
#             pickle.dump(observation_list, open(path, "wb"))
#             return observation_list


def build_observation_list(dataset, load=False, path=None):
    if load:
        # Carica la lista di osservazioni salvata
        result = pickle.load(open(path, "rb"))
        return result

    else:
        with FillingCirclesBar("Building observation list (dataset for RL):", max=len(dataset)) as bar:
            observation_list = []

            for sample in dataset:  # Itera sulle frasi semantiche del dataset
                activities = []  # Lista per memorizzare le attività estratte

                # Estrai le attività dalla frase semantica
                for sentence in sample.split(". "):  # Dividi in frasi
                    if "the activity" in sentence:  # Cerca le frasi con "the activity"
                        parts = sentence.split(" ")
                        activity_name = " ".join(parts[2:4])  # Ottieni solo il nome dell'attività
                        activities.append(activity_name)

                # Costruisci l'input e l'output basandoti sulle attività
                middle_index = find_middle_index(activities)
                input_sequence = " → ".join(activities[:middle_index])  # Attività iniziali
                output_sequence = " → ".join(activities[middle_index:])  # Attività successive

                # Aggiungi l'osservazione alla lista
                observation_list.append({
                    "prompt": input_sequence,  # Input per il modello
                    "input": input_sequence,  # Una copia di "prompt"
                    "chosen": output_sequence,  # Output previsto
                    "constraints": "",  # Campo vuoto per vincoli
                    "rejected": ""  # Campo vuoto per sequenze scartate
                })

                bar.next()  # Avanza la barra di progresso

            bar.finish()  # Completa la barra di progresso

            # Salva la lista in un file pickle
            if path:
                pickle.dump(observation_list, open(path, "wb"))

            return observation_list
