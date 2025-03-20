from syncode import Syncode
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Percorsi del modello e del tokenizer
base_model_name = "facebook/opt-1.3b"
fine_tuned_model_path = "/kaggle/working/Tirocinio/trained_models/opt_sequencer_ft_clean_fine_tuned_split_part1"
save_path = "/kaggle/working/"
tok_path = "/kaggle/working/Tirocinio/trained_models/tokenizer_opt_sequencer_ft_clean_split_part1.tk"

# Controllo GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Eseguendo su: {device}")

# Carica il modello base
base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)

# Applica il modello fine-tuned con PEFT
model = PeftModel.from_pretrained(base_model, fine_tuned_model_path).to(device)

# Carica il tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Salva il modello e il tokenizer
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Definizione della grammatica
# grammar = '''
#     start: activity (" , " activity)*
#
#     activity: "The monthly cost" NUMBER "for the loan"
#              | "calculated considering" TEXT
#              | "based on the credit score" NUMBER
#              | "the offered amount" NUMBER
#              | "the first withdrawal amount" NUMBER
#
#     NUMBER: /[0-9]+(\.[0-9]+)?/
#     TEXT: /[a-zA-Z0-9 ]+/
# '''

grammar = """
    declare Response(A, B)

    start: Response("O_Sent (mail and online)", "O_Returned")

    Response: "The process started with " ACTIVITY " and then " MIDDLE* " it transitioned to " END_ACTIVITY "."

    ACTIVITY: "0_Sent (mail and online)"
    MIDDLE: "the system processing the request" | "a customer verification step" | "an intermediate review" | "further checks"
    END_ACTIVITY: "0_Returned"
"""

# Caricamento di SynCode
syn_llm = Syncode(model=save_path, grammar=grammar, parse_output_only=True)

# Input per testare la generazione
inp = "The monthly cost 408.29 for the loan, determined based on the credit score 679"
output = syn_llm.infer(inp)

print(f"Syncode augmented LLM output:\n{output}")