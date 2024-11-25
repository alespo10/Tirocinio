import torch
from progress.bar import FillingCirclesBar
from Finetuning.support import support
from Finetuning.support.support import MAX_SEQ_LEN


def standard_train_model(model, tokenizer, optimizer, scheduler, dataloader, epochs, batch_size):
    tmp_sequences_tens = None
    proc_seq_count = 0
    sum_loss = 0.0
    current_loss = "+infinity"
    batch_count = 0
    model.train()
    loss_plot = []

    with FillingCirclesBar("Training model:", max=epochs) as bar:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs} started...")  # Print for epoch start
            epoch_loss = 0.0  # To track loss per epoch
            for idx, sequence in enumerate(dataloader):
                # "Fit as many sequence sequences into MAX_SEQ_LEN sequence as possible" logic start
                sequence_tens = torch.tensor(tokenizer.encode(sequence[0])).unsqueeze(0).to(support.device)

                # Debugging: Print sequence length
                print(f"Processing sequence {idx + 1}, sequence length: {sequence_tens.size()[1]}")

                # Skip sample from dataset if it is longer than MAX_SEQ_LEN
                if sequence_tens.size()[1] > MAX_SEQ_LEN:
                    print(f"Skipping sequence {idx + 1} as its length exceeds MAX_SEQ_LEN")
                    continue

                # The first sequence in the sequence
                if not torch.is_tensor(tmp_sequences_tens):
                    tmp_sequences_tens = sequence_tens
                    continue

                else:
                    # The next sequence does not fit in so we process the sequence and leave the last sequence
                    # as the start for next sequence
                    if tmp_sequences_tens.size()[1] + sequence_tens.size()[1] > MAX_SEQ_LEN:
                        work_sequences_tens = tmp_sequences_tens
                        tmp_sequences_tens = sequence_tens
                        print(
                            f"Batch exceeds MAX_SEQ_LEN, processing previous batch of size {work_sequences_tens.size()[1]}")

                    else:
                        # Add the sequence to sequence, continue and try to add more
                        tmp_sequences_tens = torch.cat([tmp_sequences_tens, sequence_tens[:, 1:]], dim=1)
                        continue

                # Sequence ready, process it through the model
                outputs = model(work_sequences_tens, labels=work_sequences_tens)
                loss, logits = outputs[:2]
                loss.backward()
                sum_loss += loss.detach().data

                proc_seq_count += 1
                if proc_seq_count == batch_size:
                    proc_seq_count = 0
                    batch_count += 1
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()

                    # Debugging: Print after each batch update
                    print(f"Batch {batch_count}: Loss {loss.item():.4f}")
                    loss_plot.append(round(loss.item(), 4))

                if batch_count == 100:
                    current_loss = sum_loss
                    batch_count = 0
                    sum_loss = 0.0
                    # Debugging: Print loss every 100 batches
                    print(f"Epoch {epoch + 1}: Loss after 100 batches: {current_loss:.4f}")

            # Print loss after each epoch
            print(f"Epoch {epoch + 1} completed. Loss: {sum_loss:.4f}")
            bar.next()
            bar.suffix = f"{int(bar.percent)}%% Current loss: {current_loss}"

            # Stopping training if loss is sufficiently low
            if (not type(current_loss) is str) and (current_loss < 500):
                print("Stopping training early due to low loss!")
                break

        bar.finish()

        # Storing the model
        print("Training completed. Saving the model...")
        support.save_model(model, tokenizer, "fine_tuned")
