import os
import json
import csv

# Specify the four parent folders and the path to the CSV file
parent_folders = [
    '/Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_1_raw_data/run_1/battery/battery_onboarding/',
    '/Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_1_raw_data/run_1/battery/battery_offboarding/',
    '/Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_1_raw_data/run_2/battery/run2_battery_onboarding/',
    '/Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_1_raw_data/run_2/battery/run2_battery_offboarding/'
]
csv_path = '/path/to/your/csvfile.csv'

# Log file to store results
log_file = 'missing_string_log.txt'

# Read the CSV file and create a mapping of file names to search strings
file_search_string_mapping = {}
with open(csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row if it exists
    for row in reader:
        file_name, search_string = row
        file_search_string_mapping[file_name] = search_string

# Function to check if the string is present in the JSON file
def is_string_present(file_path, search_string):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return search_string in str(data)

# Open the log file for writing
with open(log_file, 'w') as log:
    # Iterate through the parent folders
    for parent_folder in parent_folders:
        # Iterate through sub-folders
        for subdir, _, files in os.walk(parent_folder):
            for file_name in files:
                # Check if the file is a JSON file
                if file_name.endswith('.json'):
                    file_path = os.path.join(subdir, file_name)
                    search_string = file_search_string_mapping.get(file_name)
                    # Check if the string is present in the file
                    if search_string and not is_string_present(file_path, search_string):
                        log.write(f"Missing string in folder: {subdir}, file: {file_name}\n")
                        print(f"Missing string in folder: {subdir}, file: {file_name}")

print(f"Log file created: {log_file}")
