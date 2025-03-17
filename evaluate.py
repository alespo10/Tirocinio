import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from Finetuning.dataset.pm_dataset import PMDataset
from Finetuning.generators.complete_sequences import complete_sequences
from Finetuning.generators.generate_sequences import generate_sequences

from peft import PeftModel
import os

# Percorsi dei modelli e del tokenizer
model_path = "/kaggle/working/Tirocinio/tr/opt_sequencer_ft_clean_fine_tuned_helpdesk"
tokenizer_path = "/kaggle/working/Tirocinio/tr/tokenizer_opt_sequencer_ft_clean_helpdesk.tk"
base_model_name = "facebook/opt-1.3b"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Eseguendo su: {device}")

base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
model = PeftModel.from_pretrained(base_model, model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

model.eval()

print("✅ Modello LoRA caricato correttamente!")

# Percorso del file di test
test_file_path = "/kaggle/working/Tirocinio/Data/helpdesk.txt"

# Verifica se il file esiste
if not os.path.exists(test_file_path):
    raise FileNotFoundError(f"❌ Il file {test_file_path} non esiste! Controlla il percorso.")

# Leggere solo le prime 100 righe del file di test
with open(test_file_path, "r", encoding="utf-8") as f:
    test_lines = f.readlines()[:100]

# Creare un dataset di test con queste righe
test_dataset = PMDataset(sequences=test_lines, decl_path=None)

# Eseguire il completamento delle sequenze su GPU/CPU
complete_sequences(
    model, tokenizer, validation_list=[], test_dataset=test_dataset,
    output_file_path="test_results.csv", supply_constraints=True, verbose=True
)


#
# # Numero di sequenze da generare
# n_to_generate = 100  # Cambia questo valore a seconda delle esigenze
#
# # Percorso di output
# output_file_path = "generate_results.csv"
# # Eseguire la generazione di nuove sequenze
# generate_sequences(
#     model, tokenizer, validation_list=[],
#     n_to_generate=n_to_generate,
#     output_file_path=output_file_path,
#     verbose=True
# )
#
# print("✅ Completamento sequenze terminato e salvato in test_results.csv!")

