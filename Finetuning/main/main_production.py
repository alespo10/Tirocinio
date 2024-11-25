import torch
import logging
import warnings

from torch.utils.data import Dataset, DataLoader
from Finetuning.dataset.pm_dataset import PMDataset
from Finetuning.support import support
from Finetuning.support.support import cprint, Color, base_directory, MAX_SEQ_LEN, choose_from_top, DELIM_EOS

logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


#prompt = "edit this sequence in order to the Activity_B is followed by the Activity_F): PROCESS: Activity_A Activity_B Activity_D Activity_E Activity_F Activity_D Activity_E Activity_F Activity_D"
prompt = "Who won the superbowl?"
quantity_drafts = 1
dataset_name = ""
model_type = "base"

##############################################################################################

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

cprint("Loading model...", Color.BLUE)
model, tokenizer, _ = support.load_model(model_type)
model = model.to(device)

cprint("Creating dataset...", Color.BLUE)
dataset = PMDataset(sequences=[prompt] * quantity_drafts)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

cprint("Completing sequences...", Color.BLUE)
with torch.no_grad():
    for idx, sequence in enumerate(dataloader):
        print("-" * 150)
        sequence_finished = False
        cur_ids = torch.tensor(tokenizer.encode(sequence[0])).unsqueeze(0).to(device)
        # Skip sample from dataset if it is longer than MAX_SEQ_LEN
        if cur_ids.size()[1] > MAX_SEQ_LEN:
            print("Sequence too much long, skipping!")
            continue

        for i in range(100):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0, -1], dim=0)
            if i < 3:
                n = 20

            else:
                n = 3

            next_token_id = choose_from_top(softmax_logits.to("cpu").numpy(), n=n)
            cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id], dim=1)
            if next_token_id in tokenizer.encode(DELIM_EOS):
                sequence_finished = True
                break

        output_list = list(cur_ids.squeeze().to("cpu").numpy())
        output_text = tokenizer.decode(output_list)
        print("Sequence generated: ")
        if sequence_finished:
            cprint(output_text, Color.GREEN)

        else:
            cprint(output_text, Color.RED)
