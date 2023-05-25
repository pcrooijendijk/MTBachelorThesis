#%% Import all modules
import torch
import torch.nn as nn

import numpy as np
import pickle
import os
import pandas as pd
import sys
import math
import random
# For plotting
import pypianoroll
from pypianoroll import Multitrack, Track
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('SVG')
#%matplotlib inline
#matplotlib.get_backend()
import mir_eval.display
import librosa
import librosa.display
# For rendering output audio
import pretty_midi
from midi2audio import FluidSynth
# from google.colab import output
from IPython.display import display, Javascript, HTML, Audio

#%% Saving NLL into a matrix
directory_path = 'C:/Users/Pien/Documents/Documenten/Radboud/AI_third_year/Thesis/MTKERN/MusicTransformer-Pytorch-private/rpr/results/output/'
n_files = 57
notes = 128
length_matrix = 56999
matrix_all_files = np.empty((n_files, length_matrix, notes))
n_iter = 0

# Loop over files in the directory
for file_name in os.listdir(directory_path):
    if os.path.isfile(os.path.join(directory_path, file_name)):
        # Process the file
        file_path = directory_path + file_name
        df = pd.read_csv(file_path)
        matrix_temp = df.values
        matrix_all_files[n_iter] = matrix_temp
        n_iter += 1

#%% Information content computation
information_content = []
for midi_files in range(n_files):
    information_content.append(np.nansum(np.unique(matrix_all_files[midi_files], axis=0)))
print(information_content)

# %% Cross-entropy computation
cross_entropy = []
for midi_files in range(n_files):
    cross_entropy.append(-np.nansum(np.unique(np.log2(matrix_all_files[midi_files])*matrix_all_files[midi_files], axis=0)))
print(cross_entropy)

# %%
