import csv
import os


def split_csv(file_path, output_dir):
    # Assicurati che la cartella di output esista
    os.makedirs(output_dir, exist_ok=True)

    with open(file_path, 'r') as file:
        reader = list(csv.reader(file))

        # Calcola il numero di righe per ciascun file
        total_rows = len(reader)
        chunk_size = total_rows // 4
        remainder = total_rows % 4

        # Determina gli indici per suddividere il file
        split_1 = chunk_size + (1 if remainder > 0 else 0)
        split_2 = split_1 + chunk_size + (1 if remainder > 1 else 0)
        split_3 = split_2 + chunk_size + (1 if remainder > 2 else 0)

        # Suddivide il file in quattro parti
        data1 = reader[:split_1]
        data2 = reader[split_1:split_2]
        data3 = reader[split_2:split_3]
        data4 = reader[split_3:]

    # Percorsi dei file di output
    output_file1 = os.path.join(output_dir, 'split_part1.csv')
    output_file2 = os.path.join(output_dir, 'split_part2.csv')
    output_file3 = os.path.join(output_dir, 'split_part3.csv')
    output_file4 = os.path.join(output_dir, 'split_part4.csv')

    # Scrive i dati nei file di output
    for data, output_file in zip([data1, data2, data3, data4],
                                 [output_file1, output_file2, output_file3, output_file4]):
        with open(output_file, 'w', newline='') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(data)


# Percorso del file CSV iniziale e della cartella di output
file_path = 'C:/Users/nicol/PycharmProjects/Tirocinio/Preprocessing/Input/bpic2017_o.csv'
output_dir = 'C:/Users/nicol/PycharmProjects/Tirocinio/Data'

# Esegui la funzione
split_csv(file_path, output_dir)




