import numpy
import torch
import pm4py
import pandas
import random
import string
import re
import os

from accelerate.utils import MODEL_NAME

#from Finetuning.trainers.standard_trainer import save_loss_to_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import csv
import requests

from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from datasets import Dataset
from Finetuning.support.declare.Checkers import TemplateConstraintChecker
from Declare4Py.Utils.Declare.TraceStates import TraceState
from Declare4Py.D4PyEventLog import D4PyEventLog
from pm4py.objects.conversion.log import converter as log_converter
from scipy.optimize import linear_sum_assignment
from enum import Enum

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model



base_directory = "/kaggle/working/Tirocinio"
models_folder = f"{base_directory}/trained_models"

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

seed = 0
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


#MODEL_NAME = "facebook/opt-1.3b"
#saving_model_name = "opt_sequencer_ft_clean"
#MODEL_NAME = "gpt2-large"
#saving_model_name = "gpt2large_sequencer_ft_clean"

MODEL_NAME = "gpt2"
saving_model_name = "gpt2_sequencer_ft_clean"

#MODEL_NAME = "EleutherAI/pythia-160m"
#saving_model_name = "pythia-160m_sequencer_ft_clean"

MIN_SEQ_LEN = 100
MAX_SEQ_LEN = 400

DELIM_SOC = "<|startofconstraints|>"
DELIM_EOC = "<|endofconstraints|>"
DELIM_SOS = "<|startoftext|>"
DELIM_EOS = "<|endoftext|>"
DELIM_SOP = "PROCESS: "

dataset_name = ""
activities_to_keep = []
delimiters = [DELIM_SOC, DELIM_EOC, DELIM_SOS, DELIM_EOS, DELIM_SOP]
available_templates = []

class Color(Enum):
    BLUE = 2
    GREEN = 3
    LIGHT_GREEN = 4
    RED = 5
    MAGENTA = 6
    CYAN = 7
    BLACK = 8


def collator(data):
    sub1 = []
    sub2 = []
    result = (sub1, sub2)
    for sequence in data:
        middle_index = find_middle_index(sequence)
        input = sequence[:middle_index]
        output = sequence[middle_index + 1:]
        sub1.append(input)
        sub2.append(output)

    return result


def cprint(text, color=Color.BLACK):
    if color == Color.BLUE:
        code_color = "\033[94m"

    elif color == Color.GREEN:
        code_color = "\033[32m"

    elif color == Color.LIGHT_GREEN:
        code_color = "\033[92m"

    elif color == Color.RED:
        code_color = "\033[91m"

    elif color == Color.MAGENTA:
        code_color = "\033[95m"

    elif color == Color.CYAN:
        code_color = "\033[96m"

    else:
        code_color = "\033[0m"

    print(code_color + str(text) + "\033[0m")


def choose_from_top(probs, n=5):
    ind = numpy.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / numpy.sum(top_prob)
    choice = numpy.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


def find_middle_index(sequence, randomized_lower=True):
    if not randomized_lower:
        absolute_middle = int(len(sequence)/2)

    else:
        absolute_middle = int(len(sequence)/random.randint(2, 5))

    for i in range(absolute_middle, len(sequence)):
        if sequence[i] == " ":
            return i

    return absolute_middle


def get_string_between(start, end, main):
    if start not in main:
        return main

    start_index = main.index(start)
    if end is None or end not in main:
        end_index = len(main)

    else:
        end_index = main.index(end)

    result = ""
    for index in range(start_index + len(start), end_index):
        result = result + main[index]

    return result


def remove_substring_between(text, start, end):
    return re.sub(f"{re.escape(start)}(.*?){re.escape(end)}", f"{start}{end}", text, flags=re.DOTALL)


def satisfies(sequence, constraint, detailed=False, completed=True):
    case_concept_name = []
    concept_name = []
    timestamp = []
    for token in sequence.split(" "):
        if not token == "":
            case_concept_name.append(0)
            concept_name.append(token.replace("_", " "))
            timestamp.append('2024-07-17 09:00:00')

    log_data = {'case:concept:name': case_concept_name, 'concept:name': concept_name, 'time:timestamp': timestamp}
    data_frame = pm4py.format_dataframe(pandas.DataFrame(log_data), case_id='case:concept:name', activity_key='concept:name')
    pm4py_event_log = log_converter.apply(data_frame)
    # Converting pm4py Event Log as Declare Event Log
    event_log = D4PyEventLog(case_name="case:concept:name")
    event_log.log = pm4py.convert_to_event_log(pm4py_event_log)
    event_log.log_length = len(event_log.log)
    event_log.timestamp_key = event_log.log._properties['pm4py:param:timestamp_key']
    event_log.activity_key = event_log.log._properties['pm4py:param:activity_key']
    # Building constraint for checker
    consider_vacuity = False
    rules = {"vacuous_satisfaction": consider_vacuity, "activation": constraint['condition'][0]}
    if constraint['template'].supports_cardinality:
        rules["n"] = constraint['n']

    if constraint['template'].is_binary:
        rules["correlation"] = constraint['condition'][1]

    rules["time"] = constraint['condition'][-1]
    # Checking constraint
    complete_result = (TemplateConstraintChecker(event_log.get_log()[0], completed, constraint['activities'], rules).get_template(constraint['template'])()).state
    if detailed:
        return complete_result

    else:
        return complete_result == TraceState.SATISFIED


def check_constraints(sequence, constraints, detailed, completed):
    if detailed:
        result = []
        counter = 0

    clean_sequence = " ".join(_extract_activities(get_string_between(DELIM_SOS, DELIM_EOS, sequence)))
    if clean_sequence.replace(" ", "") == "":
        if detailed:
            return [TraceState.POSSIBLY_SATISFIED] * len(constraints), len(constraints)

        else:
            return True

    for constraint in constraints:
        if detailed:
            result_on_input = satisfies(clean_sequence, constraint, detailed=True, completed=completed)
            result.append(result_on_input)
            if result_on_input == TraceState.SATISFIED:
                counter += 1

        else:
            result_on_input = satisfies(clean_sequence, constraint)
            if not result_on_input:
                return False

    if detailed:
        return result, counter

    else:
        return True


def check_consistency(sequence):
    for token in sequence.split(" "):
        if not (token in activities_to_keep or token in delimiters):
            return False

    return True


def check_conformance_distance(sequence, original_sequences):
    # Finding minimum distance sequence
    min_distance = 9999999
    for idx, current_sequence in enumerate(original_sequences):
        distance = normalized_damerau_levenshtein_distance(current_sequence.squeeze(0), sequence)
        if distance < min_distance:
            min_distance = distance

    return distance


def _clean_activity(activity):
    if (len(activity) == 11 or len(activity) == 10) and activity in activities_to_keep:
        return activity

    elif len(activity) > 11:
        if activity[:11] in activities_to_keep:
            return activity[:11]

        elif activity[:10] in activities_to_keep:
            return activity[:10]

    return ""

def extract_unique_activities(file_path):
    unique_activities = set()
    # Apri il file CSV e leggi riga per riga
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Salta l'intestazione se presente

        for row in reader:
            if len(row) >= 2:  # Assicurati che ci sia una seconda colonna
                activity = row[1].strip()  # Rimuovi spazi superflui
                unique_activities.add(activity)

    # Converti il set in una lista ordinata
    return sorted(unique_activities)

attDaCsv = "/kaggle/working/Tirocinio/Preprocessing/Input/helpdesk.csv"



def _extract_activities(text):
    text = text.replace("Activity_", " Activity_")
    return [_clean_activity(i) for i in text.split(" ") if i != ""]

def extract_activities_from_text(text, activities_list):
    extracted_activities = []
    for match in re.finditer(rf"\b({'|'.join(map(re.escape, activities_list))})\b", text):
        extracted_activities.append(match.group(0))
    return extracted_activities


def sequence2numpy(sequence):
    #activities = _extract_activities(get_string_between(DELIM_SOS, DELIM_EOS, sequence))
    lista = extract_unique_activities(attDaCsv)
    activities = extract_activities_from_text(sequence,lista)
    np_sequence = numpy.zeros(len(activities))
    for i in range(len(activities)):
        if activities[i] != "":
            np_sequence[i] =  lista.index(activities[i])
    return np_sequence


def sequences2numpy(sequences):
    np_sequences = []
    for i, value in enumerate(sequences):
        np_sequences.append(sequence2numpy(value["prompt"] + value["chosen"] if isinstance(value, dict) else value))

    return np_sequences


def constraint2string(constraint):
    activities = constraint["activities"]
    type = get_string_between("<Template.", ":", str(constraint["template"]))
    all_activities = "".join(map(str, ["" + activities[i].replace(" ", "_") + "," for i in range(0, len(activities))]))[:-1]
    return f'{type.replace(" ", "")}({all_activities})'


def put_constraints(sequence, constraints):
    start_pos = sequence.find(DELIM_SOC) + len(DELIM_SOC)
    end_pos = sequence.find(DELIM_EOC)
    return sequence[:start_pos] + constraints + sequence[end_pos:]


def prune_log(dataframe, activities_to_remove):
    for activity_to_remove in activities_to_remove:
        dataframe = dataframe.drop(dataframe[dataframe["concept:name"].str.strip() == activity_to_remove.replace("_", " ")].index)

    return dataframe


def generate_subset_stringed_constraints(n_min_constraints, n_max_constraints, constraints):
    n_constraints = random.randint(n_min_constraints, n_max_constraints)
    constraints_to_keep = [random.randint(0, len(constraints)) for _ in range(n_constraints)]
    selected_constraints = [constraints[i] for i in constraints_to_keep if i < len(constraints)]
    stringed_constraints = ""
    for constraint in selected_constraints:
        stringed_constraints += constraint2string(constraint) + " "

    return selected_constraints, stringed_constraints


def _pair_traces(normalized_distances, original_log, generated_log):
    cost_matrix = numpy.array(normalized_distances).reshape(len(original_log), len(generated_log))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix, row_ind, col_ind


def _compute_pair_distances(original_log, generated_log):
    distances = []
    for i, original_trace in enumerate(original_log):
        for j, generated_trace in enumerate(generated_log):
            distance = normalized_damerau_levenshtein_distance(original_trace, generated_trace)
            distances.append(distance)

    return distances


def _compute_cfld(row_ind, col_ind, cost_matrix):
    total_distance = 0
    for i, j in zip(row_ind, col_ind):
        total_distance += cost_matrix[i][j]

    cfld = total_distance / len(row_ind)
    return cfld

#generated_log = np_generated_list; original_log = np_validation_list
def get_log_similarity(original_log, generated_log):
    normalized_distances = _compute_pair_distances(original_log, generated_log)
    cost_matrix, row_ind, col_ind = _pair_traces(normalized_distances, original_log, generated_log)
    cfld_metric = _compute_cfld(row_ind, col_ind, cost_matrix)
    return cfld_metric


def _generate_activity_code(index):
    letters = string.ascii_uppercase
    if index < len(letters):
        return f"Activity_{letters[index]}"

    else:
        first_letter_index = (index // len(letters)) - 1
        second_letter_index = index % len(letters)
        return f"Activity_{letters[first_letter_index]}{letters[second_letter_index]}"


def standardize_log(data_frame, activity_column_name):
    # Getting unique activity names
    unique_activities = data_frame[activity_column_name].unique()
    # Generating activity codes dynamically (with double letters if needed)
    activity_map = {activity: _generate_activity_code(i) for i, activity in enumerate(unique_activities)}
    # Converting to categorical and rename
    data_frame[activity_column_name] = data_frame[activity_column_name].astype("category").cat.rename_categories(activity_map)
    return data_frame


def _remove_random_substring(text, substring):
    if substring not in text:
        return text

    start_positions = [i for i in range(len(text) - len(substring) + 1) if text[i:i+len(substring)] == substring]
    random_position = random.choice(start_positions)
    return text[:random_position] + text[random_position + len(substring):]


def _violate_constraint(input, output, constraint):
    # Init and Last skipped
    if "Absence" in constraint["template"].templ_str:
        indices = [index for index, char in enumerate(output) if char == " "]
        random_index = random.choice(indices)
        return f"{output[:random_index]} {constraint['activities'][0]} {output[random_index + len(' '):]}"

    elif "Not Chain Succession" in constraint["template"].templ_str:
        return output.replace(constraint["activities"][1].replace(" ", "_"), f"{constraint['activities'][0]} {constraint['activities'][1]}")

    elif "Not Succession" in constraint["template"].templ_str:
        indices = [index for index, char in enumerate(output) if char == " "]
        random_index = random.choice(indices)
        return f"{output[:random_index]} {constraint['activities'][0]} {output[random_index + len(' '):]}"

    elif "Exclusive" in constraint["template"].templ_str or "Not Co-Existence" in constraint["template"].templ_str:
        if constraint["activities"][0].replace(" ", "_") in (input + output):
            return f"{constraint['activities'][1]} {output}"

        else:
            return f"{constraint['activities'][0]} {output}"

    elif "Choice" in constraint["template"].templ_str:
        return output.replace(constraint["activities"][0].replace(" ", "_"), " ").replace(constraint["activities"][1].replace(" ", "_"), " ")

    # This violates also Alternate Succession and Chain Succession
    elif "Alternate Response" in constraint["template"].templ_str or "Chain Response" in constraint["template"].templ_str or "Chain Precedence" in constraint["template"].templ_str or "Succession" in constraint["template"].templ_str:
        return _remove_random_substring(output, constraint["activities"][1].replace(" ", "_"))

    elif "Responded Existence" in constraint["template"].templ_str or "Response" in constraint["template"].templ_str:
        return output.replace(constraint["activities"][1].replace(" ", "_"), " ")

    elif "Co-Existence" in constraint["template"].templ_str:
        return output.replace(constraint["activities"][random.randint(0, 1)], " ")

    elif "Existence" in constraint["template"].templ_str:
        return output.replace(constraint["activities"][0].replace(" ", "_"), " ")

    elif "Alternate Precedence" in constraint["template"].templ_str:
        return output.replace(constraint["activities"][1].replace(" ", "_"), f"{constraint['activities'][1]} {constraint['activities'][1]}", 1)

    elif "Precedence" in constraint["template"].templ_str:
        tokens = output.split(" ")
        result = ""
        found = False
        for token in tokens:
            if not found:
                if token == constraint["activities"][1].replace(" ", "_"):
                    found = True
                    result += token + " "

                elif not token == constraint["activities"][0].replace(" ", "_"):
                    result += token + " "

            else:
                result += token + " "

        return result

    return output


def build_rejected_output(input, output, constraints):
    result = output
    if len(constraints) > 0:
        #result = _violate_constraint(input, result, constraints[random.randint(0, len(constraints) - 1)])
        for constraint in constraints:
             result = _violate_constraint(input, result, constraint)

        return result

    else:
        return "This is a stub string not allowed!"


def save_model(model, tokenizer, type):
    model_path = os.path.join(models_folder, f"{saving_model_name}_{type}_{dataset_name}")
    model.save_pretrained(model_path)
    #tokenizer.save_pretrained(model_path)




#GPT2 e LARGe
def load_model(type):
    if type == "base":
        # pretrained model
        #model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        #pretrained with adapter model
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        peft_config = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["c_attn"])
        model = get_peft_model(model, peft_config)

        # untrained model
        #config = AutoConfig.from_pretrained(MODEL_NAME)
        #model = AutoModelForCausalLM.from_config(config)

        # custom configured model
        #config = GPT2Config(n_layer=5, n_head=3)
        #model = AutoModelForCausalLM.from_config(config)

    else:
        model_path = os.path.join(models_folder, f"{saving_model_name}_{type}_{dataset_name}")
        model = AutoModelForCausalLM.from_pretrained(model_path)

    model = model.to(device)
    model.config.use_cache = False
    model.train()
    return model



# ###FACEBOOK
# from transformers import AutoModelForCausalLM
# from peft import LoraConfig, get_peft_model, PeftModel
#
# def load_model(type):
#     if type == "base":
#         # Carica il modello base pre-addestrato
#         model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
#
#         # Configura l'adattatore LoRA
#         peft_config = LoraConfig(
#             r=16,
#             lora_alpha=16,
#             lora_dropout=0.05,
#             bias="none",
#             task_type="CAUSAL_LM",
#             target_modules=["fc1", "fc2"]  # Target modules specifici di OPT
#         )
#         # Applica l'adattatore LoRA al modello
#         model = get_peft_model(model, peft_config)
#
#     else:
#         # Percorso del modello fine-tuned
#         model_path = os.path.join(models_folder, f"{saving_model_name}_{type}_{dataset_name}")
#         # Carica il modello base
#         base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
#         # Carica il modello fine-tuned con LoRA
#         model = PeftModel.from_pretrained(base_model, model_path)
#
#     # Sposta il modello sul dispositivo e configura i parametri
#     model = model.to(device)
#     model.config.use_cache = False  # Disabilita la cache (utile per il training)
#     model.train()  # Imposta il modello in modalità di addestramento
#     return model
#


# Pythia
# def load_model(type):
#     if type == "base":
#         model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
#         peft_config = LoraConfig(
#             r=16,
#             lora_alpha=16,
#             lora_dropout=0.05,
#             bias="none",
#             task_type="CAUSAL_LM",
#             target_modules=["query_key_value"]
#         )
#         model = get_peft_model(model, peft_config)
#     else:
#         model_path = os.path.join(models_folder, f"{saving_model_name}_{type}_{dataset_name}")
#         model = AutoModelForCausalLM.from_pretrained(model_path)
#     model = model.to(device)
#     model.config.use_cache = False
#     model.train()
#     return model


def load_tokenizer(observation_list):

    tokenizer_path = os.path.join(models_folder, f"tokenizer_{saving_model_name}_{dataset_name}.tk")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    else:
        dataset = Dataset.from_list(observation_list)

        def get_training_corpus():
            for start_idx in range(0, len(dataset)):
                samples = dataset[start_idx]
                yield samples["prompt"] + samples["chosen"]

        old_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        training_corpus = get_training_corpus()

        tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 500)

        special_tokens_dict = {"additional_special_tokens": [DELIM_SOC, DELIM_EOC, DELIM_SOS, DELIM_EOS, DELIM_SOP]}
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.save_pretrained(tokenizer_path)

    tokenizer.pad_token = DELIM_EOS
    return tokenizer


def add_available_templates(template):
    stringed_template = get_string_between("<Template.", ":", str(template))
    if stringed_template not in available_templates:
        available_templates.append(stringed_template)


def build_result_line(input, output_text, raw_constraints):
    line = {"prompt": input, "output": output_text, "prompt_size": get_string_between(DELIM_SOS, None, input).count("Activity_"), "n_constraints": len(raw_constraints)}
    # iterate over all constraints available
    for template in available_templates:
        # check if constraint is present in the prompt
        present = False
        for raw_constraint in raw_constraints:
            if template == get_string_between("<Template.", ":", str(raw_constraint["template"])):
                present = True
                break

        if present:
            # check if constraint is satisfied
            if check_constraints(input + output_text, [raw_constraint], detailed=False, completed=True):
                line[template] = 2

            else:
                line[template] = 1

        else:
            line[template] = 0

    return line


def save_csv_result(data, path):
    keys = data[0].keys()
    with open(path, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys, delimiter=";")
        dict_writer.writeheader()
        dict_writer.writerows(data)


def send_telegram_ending_notification():
    TOKEN = "7531410690:AAERJ_0H8THYS098xpSMvzVfPrflMr3iaW8"
    chat_id = "255950847"
    message = "The experiment was finished!"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json())
