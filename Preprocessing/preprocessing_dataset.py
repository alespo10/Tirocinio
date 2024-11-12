import pandas as pd
import os
import random
from Utility.log_help import log

transition_phrases = [
    "Later",
    "Afterward",
    "Following this",
    "Next",
    "Subsequently",
    "Then",
    "After this event",
    "In the following moments",
    "Immediately after",
    "As the next step"
]


def load_and_convert_log_to_txt(input_csv_path, activities_to_remove=None):
    if not os.path.exists(input_csv_path):
        raise Exception(f"CSV file ({input_csv_path}) not found!")
    data_frame = pd.read_csv(input_csv_path, sep=",")

    dataset_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    if dataset_name not in log:
        raise ValueError(f"FileCSV '{dataset_name}' not supported! Choose from {list(log.keys())}.")


    template_info = log[dataset_name]
    event_template = template_info['event_template']
    trace_template = template_info['trace_template']
    event_attributes = template_info['event_attribute']
    trace_attributes = template_info['trace_attribute']

    if activities_to_remove is not None:
        data_frame = data_frame[~data_frame["activity"].isin(activities_to_remove)]

    sentences = []
    current_case = None
    trace_events = []  # Temporary list


    for index, row in data_frame.iterrows():
        if row['case'] != current_case:
            # If there are accumulated events, concatenate them and add as a single trace sentence
            if trace_events:
                trace_sentence = " ".join(trace_events)
                sentences.append(trace_sentence)
                trace_events = []  # Reset the list

            current_case = row['case']

            if trace_template:
                trace_data = {attr: row[attr] for attr in trace_attributes if attr in row}
                trace_sentence = trace_template.format(**trace_data)
                trace_events.append(trace_sentence)  # Start the new case with the trace sentence


        event_data = {attr: row[attr] for attr in event_attributes if attr in row}
        event_sentence = event_template.format(**event_data)


        is_last_event_in_trace = (
                index == len(data_frame) - 1 or data_frame.iloc[index + 1]['case'] != current_case
        )


        if not is_last_event_in_trace:
            event_sentence += f" {random.choice(transition_phrases)}"

        trace_events.append(event_sentence)

    if trace_events:
        trace_sentence = " ".join(trace_events)
        sentences.append(trace_sentence)


    output_path = create_output_file(dataset_name)
    with open(output_path, "w") as file:
        for sentence in sentences:
            file.write(sentence + "\n")

    print(f"Log file converted into semantic sentences and saved to {output_path}.")


def create_output_file(dataset_name):
    directory_path = "/Users/alessandro/PycharmProjects/Tirocinio/Dataset"
    file_path = os.path.join(directory_path, f"{dataset_name}.txt")
    print(f"File successfully created at: {file_path}")
    return file_path


load_and_convert_log_to_txt(
    input_csv_path="/Users/alessandro/PycharmProjects/Tirocinio/Input/mip.csv",
    # activities_to_remove=["Activity A", "Activity B"]
)
