import pandas as pd
import os
from Utility.log_help import log


def load_and_convert_log_to_txt(input_csv_path, activities_to_remove=None):
    dataset_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    if dataset_name not in log:
        raise ValueError(f"FileCSV '{dataset_name}' not supported! Choose from {list(log.keys())}.")

    # Extract template and configurations for the dataset
    template_info = log[dataset_name]
    event_template = template_info['event_template']
    trace_template = template_info['trace_template']
    event_attributes = template_info['event_attribute']
    trace_attributes = template_info['trace_attribute']


    if not os.path.exists(input_csv_path):
        raise Exception(f"CSV file ({input_csv_path}) not found!")
    data_frame = pd.read_csv(input_csv_path, sep=",")

    if activities_to_remove is not None:
        data_frame = data_frame[~data_frame["activity"].isin(activities_to_remove)]

    sentences = []
    current_case = None

    # Loop through each event in the DataFrame
    for _, row in data_frame.iterrows():
        # Check if the case has changed
        if row['case'] != current_case:
            current_case = row['case']
            # Generate trace-level description using trace_template
            if trace_template:
                # Create a dictionary with only the trace attributes present in the DataFrame
                trace_data = {attr: row[attr] for attr in trace_attributes if attr in row}
                # Format the trace-level sentence
                trace_sentence = trace_template.format(**trace_data)
                sentences.append(trace_sentence)  # Add the trace sentence at the start of the new case

        event_data = {attr: row[attr] for attr in event_attributes if attr in row}
        sentence = event_template.format(**event_data)
        sentences.append(sentence)

    output_path = create_output_file(dataset_name)
    with open(output_path, "w") as file:
        for sentence in sentences:
            file.write(sentence + "\n")

    print(f"Log file converted into semantic sentences and saved to {output_path}.")


def create_output_file(dataset_name):
    directory_path = "/Dataset"
    file_path = os.path.join(directory_path, f"{dataset_name}.txt")
    print(f"File successfully created at: {file_path}")
    return file_path


load_and_convert_log_to_txt(
    input_csv_path="/FileCSV/sepsis.csv",
    # activities_to_remove=["Activity A", "Activity B"]
)

