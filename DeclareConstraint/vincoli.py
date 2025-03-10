import pandas as pd
from collections import defaultdict
from Preprocessing.Utility.log_help import log

#Da qui capisco pattern comuni da testare.
def extract_common_patterns(file_path: str, dataset: str, min_support: float = 0.05):
    # Ottiene le colonne rilevanti per il dataset specifico
    event_columns = log[dataset]['event_attribute']
    target_column = log[dataset]['target']  # Tipicamente 'activity'

    # Legge il file CSV usando solo le colonne rilevanti
    df = pd.read_csv(file_path, usecols=event_columns)

    # Converte il timestamp in formato datetime e ordina per case e timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')
    df = df.sort_values(by=['case', 'timestamp'])

    # Raccoglie le sequenze di attività per ogni caso
    sequences = df.groupby('case')[target_column].apply(list).tolist()
    total_cases = len(sequences)  # Numero totale di tracce uniche

    # Dizionari per memorizzare le occorrenze dei vincoli
    init_counts = defaultdict(int)
    response_counts = defaultdict(int)

    # Estrazione dei pattern Init, Precedence, Response e Not Coexistence
    for seq in sequences:
        if seq:
            # Conta l'attività di inizio (Init)
            init_counts[seq[0]] += 1

        seen_activities = set()
        for i, activity in enumerate(seq):
            seen_activities.add(activity)

            # Conta i vincoli di tipo Response (A → B)
            if i < len(seq) - 1:
                next_activity = seq[i + 1]
                response_counts[(activity, next_activity)] += 1





    # Calcolo dell'attività più frequente come prima attività (Init)
    most_frequent_init = max(init_counts, key=init_counts.get)
    init_support = init_counts[most_frequent_init] / total_cases
    init_pattern = f'Init("{most_frequent_init}") ✅ Seguito nel {init_support:.2%} delle tracce'



    response_patterns = {
        f'Response("{a}" → "{b}")': min(count / total_cases, 1.0)
        for (a, b), count in response_counts.items()
    }



    # Creazione del report finale con i pattern e le percentuali di supporto
    pattern_report = [init_pattern]


    for pattern, support in response_patterns.items():
        pattern_report.append(f'{pattern} ✅ Seguito nel {support:.2%} delle tracce')


    return pattern_report




# Esegui l'estrazione dei pattern dal file CSV per il dataset 'helpdesk'
file_path = '/Users/alessandro/PycharmProjects/Tirocinio/Preprocessing/Input/helpdesk.csv'  # Sostituisci con il percorso reale del file CSV
patterns = extract_common_patterns(file_path, 'helpdesk', min_support=0.05)

# Mostra i pattern estratti
for pattern in patterns:
    print(pattern)
