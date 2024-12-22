import pandas as pd
import matplotlib.pyplot as plt


class LossPlotter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Carica i dati dal file CSV."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dati caricati con successo da {self.file_path}.")
        except Exception as e:
            print(f"Errore nel caricare il file CSV: {e}")

    def plot_train_loss(self):
        """Crea un grafico della Train Loss."""
        if self.df is None:
            print("I dati non sono stati caricati. Assicurati di chiamare load_data() prima di plot_train_loss().")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.df['Epoch'], self.df['Train Loss'], label='Train Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Train Loss')
        plt.title('Variazione di Train Loss')
        plt.legend()
        plt.show()

    def plot_test_loss(self):
        """Crea un grafico della Test Loss."""
        if self.df is None:
            print("I dati non sono stati caricati. Assicurati di chiamare load_data() prima di plot_test_loss().")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.df['Epoch'], self.df['Test Loss'], label='Test Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Test Loss')
        plt.title('Variazione di Test Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def main(file_path):
        """Funzione principale che esegue il programma."""
        plotter = LossPlotter(file_path)
        plotter.load_data()
        plotter.plot_train_loss()  # Prima il grafico della Train Loss
        plotter.plot_test_loss()  # Poi il grafico della Test Loss


# Per eseguire il programma:
if __name__ == "__main__":
    file_path = '/Users/alessandro/PycharmProjects/Tirocinio/Finetuning/loss_data_helpdesk.csv'  # Sostituisci con il percorso del tuo file CSV
    LossPlotter.main(file_path)
