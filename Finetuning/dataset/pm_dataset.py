from torch.utils.data.dataset import Dataset
from Finetuning.dataset import loader


class PMDataset(Dataset):

    def __init__(self, dataset_name=None, decl_path=None, decl_min_support=None, decl_max_support=None, decl_itemsets_support=None, decl_max_declare_cardinality=None, decl_required=None, sequences=None, activities_to_remove=[], seed=None):
        super().__init__()
        if sequences is None:
            self.sequences, self.declare_model = loader.load_dataset(dataset_name=dataset_name, decl_path=decl_path, decl_min_support=decl_min_support, decl_max_support=decl_max_support, decl_itemsets_support=decl_itemsets_support, decl_max_declare_cardinality=decl_max_declare_cardinality, decl_required=decl_required, activities_to_remove=activities_to_remove, seed=seed)

        else:
            if isinstance(sequences[0], dict):
                self.sequences = []
                for val in sequences:
                    self.sequences.append(val)

            else:
                self.sequences = sequences

    def get_constraints(self):
        return self.declare_model.constraints

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        return self.sequences[item]
