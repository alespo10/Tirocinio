import torch
import csv
import matplotlib.pyplot as plt
from progress.bar import FillingCirclesBar
from Finetuning.support import support
from Finetuning.support.support import MAX_SEQ_LEN


def evaluate_loss(model, tokenizer, dataloader):
    model.eval()  # Metti il modello in modalità valutazione
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disabilita il calcolo dei gradienti
        for idx, sequence in enumerate(dataloader):
            # Converte la sequenza in tensore
            sequence_tens = torch.tensor(tokenizer.encode(sequence[0])).unsqueeze(0).to(support.device)

            # Salta le sequenze troppo lunghe
            if sequence_tens.size()[1] > MAX_SEQ_LEN:
                continue

            # Calcola l'output e la loss
            outputs = model(sequence_tens, labels=sequence_tens)
            loss = outputs[0]  # La loss è il primo elemento dell'output

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss


def save_loss_to_file(loss_data, file_path):
    """
    Salva i valori di loss in un file CSV.
    """
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Test Loss"])
        writer.writerows(loss_data)


def plot_loss(file_path):
    """
    Legge i valori di loss da un file CSV e crea un grafico.
    """
    epochs, train_losses, test_losses = [], [], []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Salta l'intestazione
        for row in reader:
            epochs.append(int(row[0]))
            train_losses.append(float(row[1]))
            test_losses.append(float(row[2]))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, test_losses, label="Test Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss per Epoch")
    plt.legend()
    plt.grid()
    plt.show()


def standard_train_model(model, tokenizer, optimizer, scheduler, train_dataloader, test_dataloader, epochs, batch_size, loss_file):
    tmp_sequences_tens = None
    proc_seq_count = 0
    loss_data = []  # Per salvare i valori di loss

    with FillingCirclesBar("Training model:", max=epochs) as bar:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs} started...")
            model.train()  # Torna in modalità training

            for idx, sequence in enumerate(train_dataloader):
                # Converti la sequenza in tensore
                sequence_tens = torch.tensor(tokenizer.encode(sequence[0])).unsqueeze(0).to(support.device)

                # Salta campioni troppo lunghi
                if sequence_tens.size()[1] > MAX_SEQ_LEN:
                    continue

                # Preparazione batch
                if not torch.is_tensor(tmp_sequences_tens):
                    tmp_sequences_tens = sequence_tens
                    continue

                else:
                    if tmp_sequences_tens.size()[1] + sequence_tens.size()[1] > MAX_SEQ_LEN:
                        work_sequences_tens = tmp_sequences_tens
                        tmp_sequences_tens = sequence_tens
                    else:
                        tmp_sequences_tens = torch.cat([tmp_sequences_tens, sequence_tens[:, 1:]], dim=1)
                        continue

                # Calcola la loss e aggiorna i pesi
                outputs = model(work_sequences_tens, labels=work_sequences_tens)
                loss = outputs[0]
                loss.backward()

                proc_seq_count += 1
                if proc_seq_count == batch_size:
                    proc_seq_count = 0
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()

            # Calcolo delle perdite per epoca
            train_loss = evaluate_loss(model, tokenizer, train_dataloader)
            test_loss = evaluate_loss(model, tokenizer, test_dataloader)
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

            # Aggiungi i valori al dataset
            loss_data.append([epoch + 1, train_loss, test_loss])

            bar.next()
        bar.finish()

    # Salva i dati della loss su file
    save_loss_to_file(loss_data, loss_file)
    print(f"Loss data saved to {loss_file}")
