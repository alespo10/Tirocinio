from transformers import AutoModelForCausalLM, AutoTokenizer

from Finetuning.dataset.pm_dataset import PMDataset
from Finetuning.generators.complete_sequences import complete_sequences
from peft import PeftModel


model_path = "/kaggle/working/Tirocinio/trained_models"

tokenizer = AutoTokenizer.from_pretrained(model_path)

base_model_name = "facebook/opt-1.3b"
model = AutoModelForCausalLM.from_pretrained(base_model_name)

model = PeftModel.from_pretrained(model, model_path, is_trainable=False)

model.eval()

print("✅ Modello LoRA caricato correttamente!")

test_file_path = "/kaggle/working/Tirocinio/Data/split_part1.txt"
with open(test_file_path, "r", encoding="utf-8") as f:
    test_lines = f.readlines()[:100]

# Creare un dataset di test con queste righe
test_dataset = PMDataset(sequences=test_lines, decl_path=None)

# Eseguire il completamento delle sequenze solo su queste 100 righe
complete_sequences(
    model, tokenizer, validation_list=[], test_dataset=test_dataset,
    output_file_path="test_results.csv", supply_constraints=True, verbose=True
)
