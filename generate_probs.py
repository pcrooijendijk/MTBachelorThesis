#%% Import packges, modules, functions
import torch
import torch.nn as nn

import numpy as np
import pickle
import os
import pandas as pd
from third_party.midi_processor.processor import encode_midi

from model.music_transformer import MusicTransformer
from dataset.e_piano_analysis import process_midi
from utilities.constants import TORCH_LABEL_TYPE
from utilities.device import get_device

#%% Define functions
def norm_neg_log_probs(neg_log_probs):
    
    probs = np.exp(-neg_log_probs)
    probs = probs/probs.sum(axis=1)[:,None]
    
    return -np.log(probs)

#%% Load data
dir_midi    = 'C:/Users/Pien/Documents/Documenten/Radboud/AI_third_year/Thesis/MTKERN/MusicTransformer-Pytorch-private/custom_midis/'

filenames = set()

file_path = "C:/Users/Pien/Documents/Documenten/Radboud/AI_third_year/Thesis/MTKERN/MusicTransformer-Pytorch-private/filenames.txt"

with open(file_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    line = line.strip()  # Remove leading/trailing whitespace and newlines
    filenames.add(line)
     
#%% Set model
model = 'MT'
len_context_list = [1] #  list of context lengths

for len_context in len_context_list:        
    # Arguments
    model_weights = 'C:/Users/Pien/Documents/Documenten/Radboud/AI_third_year/Thesis/MTKERN/MusicTransformer-Pytorch-private/rpr/results/best_acc_weights.pickle' 

    rpr            = True
    max_sequence   = 2048
    primer_len     = len_context
        
    # Loop over compositions
    for fn in filenames:
        probs_data = np.empty(shape=(len(filenames)*1000,128))
        probs_data[:] = np.NaN
        raw_probs_data = np.empty(shape=(len(filenames)*1000,128))
        raw_probs_data[:] = np.NaN
        i_row = 0
        # Encode midi data of current composition
        primer_file = dir_midi + fn
        raw_mid = encode_midi(primer_file)
        
        # loop over notes (note onset events only)
        note_positions = [i for i, j in enumerate(raw_mid) if j <= 127] # find note onset events
        print(fn, len(note_positions))
        for i_note, note_pos in enumerate(note_positions):
                   
                if i_note < primer_len:
                    sequence_start = 0
                    num_prime = note_pos
                else:
                    sequence_start = note_positions[i_note - primer_len]
                    num_prime = note_pos - sequence_start
                    
                primer, _  = process_midi(raw_mid, num_prime, sequence_start, random_seq=False)
                primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())

                # Variables
                n_layers = 6
                num_heads = 8
                d_model = 512
                dim_feedforward = 1024

                model = MusicTransformer(n_layers=n_layers, num_heads=num_heads,
                                         d_model=d_model, dim_feedforward=dim_feedforward,
                                         max_sequence=max_sequence, rpr=rpr).to(get_device())
        
                model.load_state_dict(torch.load(model_weights,map_location=torch.device('cpu'))) 
                # For music analysis we do not need to use GPU 
                # or we do benefit greatly from and can better run multiple cluster jobs using CPU
                
                # Length of target sequence
                target_seq_length =  num_prime + 1
                
                # random sampling generation
                model.eval()
                with torch.set_grad_enabled(False):
                    probs, raw_probs = model.get_probs(primer, i_note, target_seq_length = target_seq_length, raw_mid = raw_mid)
                
                probs_data[i_row, :] = probs
                raw_probs_data[i_row, :] = raw_probs
                
                i_row += 1
        
        # Normalize probabilites to sum up to 1f
        probs_data = norm_neg_log_probs(probs_data)
    
        # Assume uniform distribution for the first note in each composition
        probs_data[0] = -np.log(np.repeat(1/128, 128))
        
        fn_out = 'C:/Users/Pien/Documents/Documenten/Radboud/AI_third_year/Thesis/MTKERN/MusicTransformer-Pytorch-private/rpr/results/output/probs/probs_%s.csv' % (fn)
        fn_out_raw = 'C:/Users/Pien/Documents/Documenten/Radboud/AI_third_year/Thesis/MTKERN/MusicTransformer-Pytorch-private/rpr/results/output/raw/probs_%sraw.csv' % (fn)

        df_out = pd.DataFrame(probs_data)
        df_out_raw = pd.DataFrame(raw_probs_data)

        df_out.to_csv(fn_out, index=False, header=False)
        df_out_raw.to_csv(fn_out_raw, index=False, header=False)
