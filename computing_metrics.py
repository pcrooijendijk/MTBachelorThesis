#%% Import all modules
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from third_party.midi_processor.processor import encode_midi
import pretty_midi

#%% Saving NLL into a matrix
folder_path = os.getcwd()  # Get the current working directory (opened folder)
directory_path = folder_path + '/rpr/results/output/probs/'
directory_path2 = folder_path + '/rpr/results/output/raw/'
n_files = 57
notes = 128
length_matrix = 57000
matrix_all_files = np.empty((n_files, length_matrix, notes))
matrix_all_files_raw = np.empty((n_files, length_matrix, notes))
n_iter = 0
n_iter_2 = 0

# Loop over files in the directory
for file_name in os.listdir(directory_path):
    if os.path.isfile(os.path.join(directory_path, file_name)):
        # Process the file
        file_path = directory_path + file_name
        df = pd.read_csv(file_path, header = None)
        matrix_temp = df.values
        matrix_all_files[n_iter] = matrix_temp
        n_iter += 1

# Loop over files in the directory
for file_name in os.listdir(directory_path2):
    if os.path.isfile(os.path.join(directory_path2, file_name)):
        # Process the file
        file_path = directory_path2 + file_name
        df = pd.read_csv(file_path, header = None)
        matrix_temp = df.values
        matrix_all_files_raw[n_iter_2] = matrix_temp
        n_iter_2 += 1

#%% Cleaning the matrices such that there are no NaN values in them anymore
filenames = set()
dir_midi = folder_path + '/custom_midis/'
file_path = folder_path + "/filenames.txt"

with open(file_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    line = line.strip()  # Remove leading/trailing whitespace and newlines
    filenames.add(line)

note_positions = []

for fn in sorted(filenames):  
# Encode midi data of current composition
    primer_file = dir_midi + fn
    raw_mid = encode_midi(primer_file)

    # Loop over notes (note onset events only)
    note_positions.append([i for i, j in enumerate(raw_mid) if j <= 127]) # find note onset events

# Removing the NaN values from the matrix with all files
clean_matrix_all_files = []
clean_matrix_all_files_raw = []
for row in range(len(matrix_all_files)):
    clean_matrix_all_files.append(matrix_all_files[row][~np.isnan(matrix_all_files[row]).any(axis=1)])  
    clean_matrix_all_files_raw.append(matrix_all_files_raw[row][~np.isnan(matrix_all_files_raw[row]).any(axis=1)])

#%% Getting the durations of each note in the MIDI files
# Loading the MIDI files
pretty_midi_files = []
for fn in sorted(filenames): 
    pretty_midi_files.append(pretty_midi.PrettyMIDI(dir_midi+fn))

# Calculate the duration of each event
event_durations = []
for midi_file in pretty_midi_files:
    for instrument in midi_file.instruments:
        durations = []
        for note in instrument.notes:
            duration = note.end - note.start
            durations.append(duration)
        event_durations.append(durations)

#%% Remove events which are above the 127
filtered_list_note_positions = []
removed_indices = []
for midi_file in note_positions:
    filtered_list_note_positions.append([num for num in midi_file if num <= 127])
    removed_indices.append([idx for idx, num in enumerate(midi_file) if num > 127])

filtered_list_clean_matrix = []
filtered_list_event_durations = []
for midi_file in range(len(note_positions)):
    filtered_list_event_durations.append([elem for i, elem in enumerate(event_durations[midi_file]) if i not in removed_indices[midi_file]])
    filtered_list_clean_matrix.append([elem for i, elem in enumerate(clean_matrix_all_files_raw[midi_file]) if i not in removed_indices[midi_file]])

#%% Computing the "mean weighted" cross-entropy 
real_distribution = []
for notes in filtered_list_note_positions:
    real_distribution.append(np.eye(128)[notes])

cross_entropy = []
for i, midi_file in enumerate(filtered_list_clean_matrix):
    temp_matrix = []
    for notes in range(len(midi_file)):
        temp_matrix.append(-np.sum(real_distribution[i][notes]*np.log(filtered_list_clean_matrix[i][notes])))
    cross_entropy.append(temp_matrix)

mean_weighted_cross_entropy = []
for midi_files in range(len(cross_entropy)):
    mean_weighted_cross_entropy.append(np.mean(np.array(cross_entropy[midi_files]) * np.array(filtered_list_event_durations[midi_files])))

print(mean_weighted_cross_entropy)
df = pd.DataFrame(mean_weighted_cross_entropy)

# Specify the output Excel file path
output_path_ent = folder_path + "/rpr/results/metrics/mdw_ent.xlsx"

# Write the DataFrame to an Excel file
df.to_excel(output_path_ent, index=False)

#%% Computing the "mean weighted" information content
mean_weighted_information_content = []
for midi_files in range(len(clean_matrix_all_files)):
    mean = np.mean(clean_matrix_all_files[midi_files], axis=1)
    mean_weighted_information_content.append(np.mean(mean * event_durations[midi_files]))

print(mean_weighted_information_content)
df = pd.DataFrame(mean_weighted_information_content)

# Specify the output Excel file path
output_path_ic = folder_path + "/rpr/results/metrics/mdw_ic.xlsx"

# Write the DataFrame to an Excel file
df.to_excel(output_path_ic, index=False)

#%%
# Read the Excel file
df = pd.read_excel(folder_path + '\IDyOM values.xlsx')

# Storing the IDYOM values from Gold et al. (2019)
mDW_IC = df.iloc[:, 1].values
mDW_Ent = df.iloc[:, 2].values

#%% Plotting the data
# Create scatter plot
plt.scatter(mean_weighted_information_content, mean_weighted_cross_entropy, label='My data')
plt.scatter(mDW_Ent, mDW_IC, label='Gold data')

# Set plot title and labels
plt.title("Scatter Plot")
plt.xlabel("Cross-Entropy")
plt.ylabel("Information Content")
plt.legend()

# Display the plot
plt.show()
# %%
