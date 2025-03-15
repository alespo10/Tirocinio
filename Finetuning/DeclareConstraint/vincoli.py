import pandas as pd
from collections import defaultdict
from Preprocessing.Utility.log_help import log


# Da qui capisco pattern comuni da testare.
def extract_common_patterns(file_path: str, dataset: str, min_support: float = 0.05):
    event_columns = log[dataset]['event_attribute']
    target_column = log[dataset]['target']
    df = pd.read_csv(file_path, usecols=event_columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')
    df = df.sort_values(by=['case', 'timestamp'])
    sequences = df.groupby('case')[target_column].apply(list).tolist()
    total_cases = len(sequences)

    init_counts = defaultdict(int)
    response_counts = defaultdict(int)
    chain_precedence_counts = defaultdict(int)
    not_coexistence_counts = defaultdict(int)

    for seq in sequences:
        if seq:
            init_counts[seq[0]] += 1
        seen_activities = set()
        for i, activity in enumerate(seq):
            seen_activities.add(activity)
            if i < len(seq) - 1:
                next_activity = seq[i + 1]
                response_counts[(activity, next_activity)] += 1
                if i > 0 and seq[i - 1] != activity:
                    chain_precedence_counts[(seq[i - 1], activity)] += 1

            for other_activity in seen_activities:
                if other_activity != activity:
                    not_coexistence_counts[(activity, other_activity)] += 1

    most_frequent_init = max(init_counts, key=init_counts.get)
    init_support = init_counts[most_frequent_init] / total_cases
    init_pattern = f'Init("{most_frequent_init}") ✅ Seguito nel {init_support:.2%} delle tracce'

    response_patterns = {
        f'Response("{a}" → "{b}")': min(count / total_cases, 1.0)
        for (a, b), count in response_counts.items()
    }

    chain_precedence_patterns = {
        f'Chain Precedence("{a}" → "{b}")': min(count / total_cases, 1.0)
        for (a, b), count in chain_precedence_counts.items()
    }

    not_coexistence_patterns = {
        f'Not CoExistence("{a}", "{b}")': min(count / total_cases, 1.0)
        for (a, b), count in not_coexistence_counts.items()
    }

    pattern_report = [init_pattern]
    for pattern, support in response_patterns.items():
        pattern_report.append(f'{pattern} ✅ Seguito nel {support:.2%} delle tracce')
    for pattern, support in chain_precedence_patterns.items():
        pattern_report.append(f'{pattern} ✅ Seguito nel {support:.2%} delle tracce')
    for pattern, support in not_coexistence_patterns.items():
        pattern_report.append(f'{pattern} ✅ Seguito nel {support:.2%} delle tracce')

    return pattern_report


file_path = '/Users/alessandro/PycharmProjects/Tirocinio/Preprocessing/Input/split_part1.csv'  # Sostituisci con il percorso reale del file CSV
patterns = extract_common_patterns(file_path, 'split_part1', min_support=0.05)

for pattern in patterns:
    print(pattern)
