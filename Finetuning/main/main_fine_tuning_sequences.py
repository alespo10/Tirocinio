import numpy
import os
import logging
import warnings

from codecarbon import EmissionsTracker
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import Finetuning.dataset.loader
from Finetuning.dataset.pm_dataset import PMDataset
from Finetuning.generators.complete_sequences import complete_sequences
from Finetuning.generators.generate_sequences import generate_sequences
from Finetuning.support import support
from Finetuning.support.support import cprint, Color, base_directory, models_folder
from Finetuning.trainers.reinforcement_learning_trainer import reinforcement_learning_train_model
from Finetuning.trainers.standard_trainer import standard_train_model, plot_loss

logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


dataset_name = "split_part1"
test_size = 100
n_to_generate = 100
verbose = False
avoid_cfls_calculation = False
fast_approximated_cfls_calculation = False

std_train = True
std_evaluate = True
std_epochs = 14
std_batch_size = 16
std_learning_rate = 3e-5
std_warmup_steps = 5000
std_use_constraints = False

rl_train = False
rl_evaluate = False
rl_load_dataset_from_file = False
rl_algorithm = "ppo"                # "dpo" "ppo"
rl_epochs = 500
rl_steps = 300 #50 * 500
rl_n_min_constraints_train = 1
rl_n_max_constraints_train = 1
rl_n_min_constraints_test = 1
rl_n_max_constraints_test = 1
rl_n_copies = 1

decl_min_support = 0.1
decl_max_support = 0.9
decl_itemsets_support = 0.9
decl_max_declare_cardinality = 1

##############################################################################################

if dataset_name == "synthesized_d62":
    activities_to_keep = ["Activity_A", "Activity_AF", "Activity_AH", "Activity_AI", "Activity_AJ", "Activity_AG", "Activity_B", "Activity_Q", "Activity_S", "Activity_T", "Activity_U", "Activity_V", "Activity_W", "Activity_AD", "Activity_X", "Activity_AE", "Activity_Y", "Activity_AA", "Activity_AB", "Activity_AC", "Activity_Z", "Activity_R", "Activity_C", "Activity_D", "Activity_E", "Activity_F", "Activity_N", "Activity_H", "Activity_P", "Activity_K", "Activity_L", "Activity_M", "Activity_J", "Activity_I", "Activity_O", "Activity_G"]
    activities_to_remove = ["Activity AK", "Activity AM", "Activity AN", "Activity AL", "Activity AT", "Activity AV", "Activity CU", "Activity CW", "Activity DA", "Activity CX", "Activity CV", "Activity DG", "Activity DI", "Activity DH", "Activity AW", "Activity AU", "Activity AO", "Activity AQ", "Activity AP", "Activity DK", "Activity DL", "Activity CZ", "Activity AR", "Activity AS", "Activity DB", "Activity DD", "Activity DC", "Activity BS", "Activity CC", "Activity BU", "Activity BW", "Activity BX", "Activity BV", "Activity BY", "Activity CA", "Activity BZ", "Activity BT", "Activity CD", "Activity CO", "Activity CQ", "Activity CP", "Activity CR", "Activity CS", "Activity CE", "Activity AX", "Activity AZ", "Activity BB", "Activity BA", "Activity AY", "Activity BF", "Activity BN", "Activity BL", "Activity BH", "Activity BQ", "Activity BP", "Activity BR", "Activity BM", "Activity BJ", "Activity BO", "Activity BI", "Activity BG", "Activity CB", "Activity DJ", "Activity BC", "Activity BD", "Activity CY", "Activity CT", "Activity BK", "Activity BE", "Activity CF", "Activity CJ", "Activity CH", "Activity CN", "Activity CL", "Activity CM", "Activity CI", "Activity CK", "Activity CG", "Activity DF", "Activity DE"]

elif dataset_name == "synthesized_d53":
    activities_to_keep = ["Activity A", "Activity AF", "Activity AH", "Activity AI", "Activity AJ", "Activity AK", "Activity AL", "Activity AM", "Activity AO", "Activity AN", "Activity AG", "Activity B", "Activity AQ", "Activity BU", "Activity AR", "Activity Q", "Activity S", "Activity T", "Activity U", "Activity V", "Activity W", "Activity AD", "Activity X", "Activity AE", "Activity Y", "Activity AA", "Activity AB", "Activity AC", "Activity Z", "Activity R", "Activity BV", "Activity BW", "Activity BX", "Activity BY", "Activity CA", "Activity CB", "Activity BZ", "Activity CC", "Activity C", "Activity D", "Activity E", "Activity F", "Activity N", "Activity H", "Activity P", "Activity K", "Activity L", "Activity M", "Activity J", "Activity I", "Activity O", "Activity G", "Activity AS", "Activity AU", "Activity AY", "Activity BC", "Activity AZ", "Activity AV", "Activity BJ", "Activity BQ", "Activity BL", "Activity BS", "Activity BR", "Activity BO", "Activity BP", "Activity BN", "Activity BT", "Activity BM", "Activity BK", "Activity BB", "Activity AT", "Activity BD", "Activity BG", "Activity BE", "Activity BI", "Activity AW", "Activity AX", "Activity BA", "Activity AP", "Activity BF", "Activity BH"]
    activities_to_remove = []

else:
    activities_to_keep = ["Activity_A", "Activity_B", "Activity_C", "Activity_D", "Activity_E", "Activity_F", "Activity_G", "Activity_H", "Activity_I", "Activity_J", "Activity_K", "Activity_L", "Activity_M", "Activity_N", "Activity_O", "Activity_P", "Activity_Q", "Activity_R", "Activity_S", "Activity_T", "Activity_U", "Activity_V", "Activity_W", "Activity_X", "Activity_Y", "Activity_Z", "Activity_AA", "Activity_AB", "Activity_AC", "Activity_AD", "Activity_AE", "Activity_AF", "Activity_AG", "Activity_AH", "Activity_AI", "Activity_AJ"]
    activities_to_remove = []

completed_sequences_std_woc_output_file_path = f"{base_directory}/outputs/completed_std_woc_{dataset_name}_ep_{std_epochs}.csv" #
completed_sequences_std_wc_output_file_path = f"{base_directory}/outputs/completed_std_wc_{dataset_name}_ep_{std_epochs}.csv"
generated_sequences_std_woc_output_file_path = f"{base_directory}/outputs/generated_std_woc_{dataset_name}_ep_{std_epochs}.txt" #
generated_sequences_std_wc_output_file_path = f"{base_directory}/outputs/generated_std_wc_{dataset_name}_ep_{std_epochs}.txt"
generated_sequences_std_wc_table_output_file_path = f"{base_directory}/outputs/table_generated_std_wc_{dataset_name}_ep_{std_epochs}_al_{rl_algorithm}.csv"
completed_sequences_std_wc_table_output_file_path = f"{base_directory}/outputs/table_completed_std_wc_{dataset_name}_ep_{std_epochs}_al_{rl_algorithm}.csv"
completed_sequences_rnf_woc_output_file_path = f"{base_directory}/outputs/completed_rnf_woc_{dataset_name}_ep_{std_epochs}_al_{rl_algorithm}.csv"
completed_sequences_rnf_wc_output_file_path = f"{base_directory}/outputs/completed_rnf_wc_{dataset_name}_ep_{std_epochs}_al_{rl_algorithm}.csv"
generated_sequences_rnf_woc_output_file_path = f"{base_directory}/outputs/generated_rnf_woc_{dataset_name}_ep_{std_epochs}_al_{rl_algorithm}.txt"
generated_sequences_rnf_wc_output_file_path = f"{base_directory}/outputs/generated_rnf_wc_{dataset_name}_ep_{std_epochs}_al_{rl_algorithm}.txt"
generated_sequences_rnf_wc_table_output_file_path = f"{base_directory}/outputs/table_generated_rnf_wc_{dataset_name}_ep_{std_epochs}_al_{rl_algorithm}.csv"
completed_sequences_rnf_wc_table_output_file_path = f"{base_directory}/outputs/table_completed_rnf_wc_{dataset_name}_ep_{std_epochs}_al_{rl_algorithm}.csv"

observation_list_train_path = f"{base_directory}/outputs/observation_list_train_{dataset_name}.p"
observation_list_test_path = f"{base_directory}/outputs/observation_list_test_{dataset_name}.p"
decl_path = support.base_directory + f"/outputs/declare_model_{dataset_name}.decl"

support.activities_to_keep = activities_to_keep
support.dataset_name = dataset_name

################### Loading dataset ###################

cprint(f"Loading dataset {dataset_name}...", Color.BLUE)
if rl_train or rl_evaluate:
    test_size = int(test_size / rl_n_copies + 1)

dataset = PMDataset(dataset_name=dataset_name, decl_path=decl_path, decl_required=True, decl_min_support=decl_min_support, decl_max_support=decl_max_support, decl_itemsets_support=decl_itemsets_support, decl_max_declare_cardinality=decl_max_declare_cardinality, activities_to_remove=activities_to_remove)
train_set = Subset(dataset, numpy.arange(len(dataset) - test_size))
test_set = Subset(dataset, numpy.arange(len(dataset) - test_size, len(dataset)))
sequence_loader_train = DataLoader(train_set, batch_size=1, shuffle=True)
sequence_loader_test = DataLoader(test_set, batch_size=1, shuffle=True)

print(f"Building observation list (train)...")
train_observation_list = Finetuning.dataset.loader.build_observation_list(train_set, rl_load_dataset_from_file, observation_list_train_path)
print(f"Building observation list (test)...")
test_observation_list = Finetuning.dataset.loader.build_observation_list(test_set, rl_load_dataset_from_file, observation_list_test_path)
rl_test_set = PMDataset(sequences=test_observation_list, decl_path=decl_path)

################### Loading model ###################

cprint(f"Loading model '{support.MODEL_NAME}'...", Color.BLUE)
model = support.load_model("base")
model.train()
optimizer = AdamW(model.parameters(), lr=std_learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=std_warmup_steps, num_training_steps=-1)
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

cprint(f"Loading tokenizer built on observation list...", Color.BLUE)
tokenizer = support.load_tokenizer(train_observation_list)

################### Training model ###################

if std_train:
    cprint("Training model (std)...", Color.BLUE)


    # Specifica il percorso per il file di output della loss
    loss_file_path = f"{base_directory}/outputs/loss_data_{dataset_name}.csv"

    standard_train_model(
        model,
        tokenizer,
        optimizer,
        scheduler,
        sequence_loader_train,
        sequence_loader_test,  # Passa il test set per il calcolo della loss
        std_epochs,
        std_batch_size,
        loss_file_path
    )

    # Crea il grafico dalla loss salvata
    plot_loss(loss_file_path)
else:
    cprint("Skipping standard training model...", Color.MAGENTA)


if rl_train:
    model = support.load_model("fine_tuned")
    cprint("Training model (rl)...", Color.BLUE)
    reinforcement_learning_train_model(rl_algorithm, model, tokenizer, train_observation_list, rl_epochs, rl_steps, models_folder)

else:
    cprint("Skipping reinforcement learning training model...", Color.MAGENTA)

################### Evaluating model ###################

if fast_approximated_cfls_calculation:
    evaluation_observation_list = test_observation_list

else:
    evaluation_observation_list = train_observation_list + test_observation_list

if std_evaluate:
    cprint("Evaluating standard model...", Color.GREEN)
    model = support.load_model("fine_tuned")
    if not avoid_cfls_calculation:
        cprint("Generating sequences from scratch without constraints...", Color.BLUE)
        generate_sequences(model, tokenizer, evaluation_observation_list, n_to_generate, generated_sequences_std_woc_output_file_path, verbose=verbose, avoid_cfls_calculation=avoid_cfls_calculation)

    cprint("Generating sequences from scratch with constraints...", Color.BLUE)
    #generate_sequences(model, tokenizer, evaluation_observation_list, n_to_generate, generated_sequences_std_wc_output_file_path, generated_sequences_std_wc_table_output_file_path, constraints=dataset.get_constraints(), verbose=verbose, avoid_cfls_calculation=avoid_cfls_calculation)
    cprint("Completing sequences without constraints...", Color.BLUE)
    complete_sequences(model, tokenizer, evaluation_observation_list, rl_test_set, completed_sequences_std_woc_output_file_path, supply_constraints=False, verbose=verbose, avoid_cfls_calculation=avoid_cfls_calculation)
    cprint("Completing sequences with constraints...", Color.BLUE)
    #complete_sequences(model, tokenizer, evaluation_observation_list, rl_test_set, completed_sequences_std_woc_output_file_path, completed_sequences_std_wc_table_output_file_path, verbose=verbose, avoid_cfls_calculation=avoid_cfls_calculation)

if rl_evaluate:
    cprint("Evaluating reinforced model...", Color.GREEN)
    model = support.load_model("reinforced")
    if not avoid_cfls_calculation:
        cprint("Generating sequences from scratch without constraints...", Color.BLUE)
        generate_sequences(model, tokenizer, evaluation_observation_list, n_to_generate, generated_sequences_rnf_woc_output_file_path, verbose=verbose, avoid_cfls_calculation=avoid_cfls_calculation)

    cprint("Generating sequences from scratch with constraints...", Color.BLUE)
    generate_sequences(model, tokenizer, evaluation_observation_list, n_to_generate, generated_sequences_rnf_wc_output_file_path, generated_sequences_rnf_wc_table_output_file_path, constraints=dataset.get_constraints(), verbose=verbose, avoid_cfls_calculation=avoid_cfls_calculation)
    cprint("Completing sequences without constraints...", Color.BLUE)
    complete_sequences(model, tokenizer, evaluation_observation_list, rl_test_set, completed_sequences_rnf_woc_output_file_path, supply_constraints=False, verbose=verbose, avoid_cfls_calculation=avoid_cfls_calculation)
    cprint("Completing sequences with constraints...", Color.BLUE)
    complete_sequences(model, tokenizer, evaluation_observation_list, rl_test_set, completed_sequences_rnf_woc_output_file_path, completed_sequences_rnf_wc_table_output_file_path, verbose=verbose, avoid_cfls_calculation=avoid_cfls_calculation)

#support.send_telegram_ending_notification()
