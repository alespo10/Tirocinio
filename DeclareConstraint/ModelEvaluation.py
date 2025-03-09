import os
import numpy as np
from Finetuning.support import support
from Finetuning.generators.generate_sequences import generate_sequences
from Finetuning.generators.complete_sequences import complete_sequences
from Finetuning.support.support import cprint, Color
from Finetuning.dataset.loader import build_observation_list
from Finetuning.dataset.pm_dataset import PMDataset


class ModelEvaluation:
    def __init__(self,
                 model_path=None,
                 tokenizer_path=None,
                 dataset_path=None,
                 test_size=0.2,
                 rl_load_dataset_from_file=False,
                 observation_list_train_path="./observation_list_train.pkl",
                 observation_list_test_path="./observation_list_test.pkl",
                 decl_path=None,
                 n_to_generate=100,
                 verbose=True,
                 avoid_cfls_calculation=False):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.rl_load_dataset_from_file = rl_load_dataset_from_file
        self.observation_list_train_path = observation_list_train_path
        self.observation_list_test_path = observation_list_test_path
        self.decl_path = decl_path
        self.n_to_generate = n_to_generate
        self.verbose = verbose
        self.avoid_cfls_calculation = avoid_cfls_calculation
        self.model = None
        self.tokenizer = None
        self.evaluation_observation_list = []
        self.rl_test_set = None
        self.train_set = []
        self.test_set = []

    def load_model(self):
        if self.model_path and self.tokenizer_path:
            cprint(f"Loading the model from {self.model_path}...", Color.GREEN)
            self.model = support.load_model(self.model_path)

            cprint(f"Loading the tokenizer from {self.tokenizer_path}...", Color.GREEN)
            self.tokenizer = support.load_tokenizer(self.tokenizer_path)
        else:
            cprint("Error: Model and Tokenizer paths must be provided.", Color.RED)

    def prepare_datasets(self):
        if self.dataset_path and os.path.exists(self.dataset_path):
            cprint(f"Loading dataset from {self.dataset_path}...", Color.BLUE)
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                dataset = f.read().splitlines()

            test_size = int(len(dataset) * self.test_size)
            self.train_set = dataset[:len(dataset) - test_size]
            self.test_set = dataset[len(dataset) - test_size:]
            cprint(f"Dataset loaded: {len(self.train_set)} training samples, {len(self.test_set)} testing samples.",
                   Color.GREEN)
        else:
            cprint("Error: Dataset path is not valid or file does not exist.", Color.RED)
            return

        cprint("Building observation list (train)...", Color.BLUE)
        train_observation_list = build_observation_list(self.train_set,
                                                        self.rl_load_dataset_from_file,
                                                        self.observation_list_train_path)

        cprint("Building observation list (test)...", Color.BLUE)
        test_observation_list = build_observation_list(self.test_set,
                                                       self.rl_load_dataset_from_file,
                                                       self.observation_list_test_path)

        self.evaluation_observation_list = train_observation_list + test_observation_list
        self.rl_test_set = PMDataset(sequences=test_observation_list, decl_path=self.decl_path)
        cprint(f"Prepared {len(self.evaluation_observation_list)} evaluation observations.", Color.GREEN)

    def run_evaluation(self, output_dir="./evaluation_results"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        generated_sequences_output_file = os.path.join(output_dir, "generated_sequences.txt")
        completed_sequences_output_file = os.path.join(output_dir, "completed_sequences.txt")

        cprint("Generating sequences from scratch without constraints...", Color.BLUE)
        generate_sequences(
            self.model,
            self.tokenizer,
            self.evaluation_observation_list,
            self.n_to_generate,
            generated_sequences_output_file,
            verbose=self.verbose,
            avoid_cfls_calculation=self.avoid_cfls_calculation
        )

        cprint("Completing sequences without constraints...", Color.BLUE)
        complete_sequences(
            self.model,
            self.tokenizer,
            self.evaluation_observation_list,
            self.rl_test_set,
            completed_sequences_output_file,
            supply_constraints=False,
            verbose=self.verbose,
            avoid_cfls_calculation=self.avoid_cfls_calculation
        )

        cprint("Evaluation completed.", Color.GREEN)


# Esempio di utilizzo
if __name__ == '__main__':
    evaluation = ModelEvaluation(
        model_path="/Users/alessandro/Downloads/OptBpic2017ParimaParte4EpocheFinale/working/Tirocinio/trained_models/opt_sequencer_ft_clean_fine_tuned_split_part1",
        tokenizer_path="/Users/alessandro/Downloads/OptBpic2017ParimaParte4EpocheFinale/working/Tirocinio/trained_models/tokenizer_opt_sequencer_ft_clean_split_part1.tk",
        dataset_path="/Users/alessandro/PycharmProjects/Tirocinio/Data/split_part1.txt",
        test_size=0.2,
        rl_load_dataset_from_file=True,
        decl_path=None,
        n_to_generate=100
    )
    evaluation.load_model()
    evaluation.prepare_datasets()
    evaluation.run_evaluation(output_dir="./evaluation_results_custom")
