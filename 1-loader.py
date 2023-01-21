# -*- coding: utf-8 -*-
"""
Created on Aug 17 2021

@author: Joao

Objetive: 
    
    Gathers the data of each scenario in a numpy file.
    Each scenario has a CSV file indicating which files are associated with 
    each unit and each modality (mmWave power, GPS data, ...). 
    For each scenario, unit and modality, this script accesses every path 
    specified in the CSV, reads the files, and concatenates the information
    of each sample in a convinient numpy array. 
    
    Note: this script only gathers all the existing data. It does not change
    the data in any way.
    
Script inputs:
    data_folder: folder where the data for scenario is. 
                 Use a path relative to the current working directory (cwd)
                 E.g. 'ViWiReal Scenarios' if *cwd*/ViWiReal Scenarios
    scenario_idx: index of scenario. E.g. 5 corresponds to 'Scenario5_Tyler'
    max_samples: read at most X samples from the scenario.
    n_pos_vals: number of values per position sample
    n_pwr_vals: number of values per mmWave power measurement sample
    output_folder: relative path to where to output the processed CSV file.

Script outputs:
    - a CSV file with the position and power data for a given scenario    

"""


import os
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.image as mplimage

# Inputs:
#   Input and output data folders (absolute paths)
data_folder = r'E:\DeepSense-scenarios' # os.path.join(os.getcwd(), 'DeepSense')
output_folder = os.path.join(os.getcwd(), 'Gathered_data3')

scenario_idx = 1

unit = 2
first_sample = 1 # 1 to start on the first
max_samples = int(1e5) #9

modalities = ['loc', 'loc-cal', 'pwr_60ghz'] # Select multiple modalities! 

labels = ['seq_index']

# Available modalities: 
#   - 'loc': Localization
#   - 'pwr_60ghz': Power
#   - 'rgb': RGB Images
#   - 'lidar': Lidar data
#   - 'radar': Radar data

#%% Set constants

CAM_DIMENSIONS = {'cam1': (540,960,3),
                  'cam2': (720,1280,3)}

CAMS_PER_SCENARIO = ['cam1'] * 15 + ['cam2'] * 1 + ['cam1'] * 7

PWRS_PER_SCENARIO = [(64,1)] * 15 + [(1,1)] * 1 + [(64,1)] * 7

DATA_DIMENSIONS = {'loc': (2, 1),
                   'loc_cal': (2, 1),
                   'pwr_60ghz': PWRS_PER_SCENARIO[scenario_idx-1],
                   'rgb': CAM_DIMENSIONS[CAMS_PER_SCENARIO[scenario_idx-1]],
                   'lidar': (460, 2),
                   'radar': (4, 256, 128),
                   'seq_index': (1, 1),
                   'blockage_label': (1, 1),
                   'bbox': (2,4)}

SUPPORTED_MODALITIES_PER_UNIT_PER_SCENARIO = \
    [{1: ['loc', 'pwr_60ghz', 'rgb'],                   2: ['loc']},            # 1
     {1: ['loc', 'pwr_60ghz', 'rgb'],                   2: ['loc']},            # 2
     {1: ['loc', 'pwr_60ghz', 'rgb'],                   2: ['loc', 'loc_cal']}, # 3
     {1: ['loc', 'pwr_60ghz', 'rgb'],                   2: ['loc', 'loc_cal']}, # 4
     {1: ['loc', 'pwr_60ghz', 'rgb'],                   2: ['loc']},            # 5
     {1: ['loc', 'pwr_60ghz', 'rgb'],                   2: ['loc']},            # 6
     {1: ['loc', 'pwr_60ghz', 'rgb'],                   2: ['loc']},            # 7
     {1: ['loc', 'pwr_60ghz', 'rgb', 'lidar'],          2: ['loc', 'loc_cal']}, # 8
     {1: ['loc', 'pwr_60ghz', 'rgb', 'lidar', 'radar'], 2: ['loc', 'loc_cal']}, # 9
     {1: ['pwr_60ghz', 'rgb']},                                                 # 10
     {1: ['pwr_60ghz', 'rgb']},                                                 # 11
     {1: ['pwr_60ghz', 'rgb']},                                                 # 12
     {1: ['loc', 'pwr_60ghz', 'rgb']},                                          # 13
     {1: ['loc', 'pwr_60ghz', 'rgb'],                   2: ['loc']},            # 14
     {1: ['loc', 'pwr_60ghz', 'rgb']},                                          # 15
     {1: ['pwr_60ghz', 'rgb', 'blockage']},                                     # 16
     {1: ['loc', 'pwr_60ghz', 'rgb']},                                           # 17
     {1: ['loc', 'pwr_60ghz', 'rgb']},                                           # 18
     {1: ['loc', 'pwr_60ghz', 'rgb']},                                           # 19
     {1: ['pwr_60ghz', 'rgb']},                                                 # 20
     {1: ['loc', 'pwr_60ghz', 'rgb']},                             # 21
     {1: ['loc', 'pwr_60ghz', 'rgb']},                             # 22
     {1: ['loc', 'pwr_60ghz', 'rgb'], 
      2: ['loc', 'speed', 'altitude', 'distance', 'height', 
          'x-speed', 'y-speed', 'z-speed', 'pitch', 'roll']},                   # 23
     ]
    
SUPPORTED_FORMATS = ['.npy', '.mat']

SUPPORTED_UNITS = SUPPORTED_MODALITIES_PER_UNIT_PER_SCENARIO[scenario_idx-1].keys()
if unit not in SUPPORTED_UNITS:
    raise Exception(f"Unit '{unit}' not supported for scenario {scenario_idx}. "
                    f"Supported units are: {SUPPORTED_UNITS}.")

SUPPORTED_MODALITIES = \
    SUPPORTED_MODALITIES_PER_UNIT_PER_SCENARIO[scenario_idx-1][unit]

if modalities == 'all':
    modalities = SUPPORTED_MODALITIES
    
allowed_modalities = []
for mod in modalities:
    if mod in SUPPORTED_MODALITIES:
        allowed_modalities.append(mod)
    else:
        print(f"WARNING: Modality '{mod}' not supported for unit {unit} in "
              f"scenario {scenario_idx}. Removing from the modalities list...")

SUPPORTED_LABELS_PER_SCENARIO = [
    ['seq_index'],  # Scenario 1
    ['seq_index'],  # Scenario 2
    ['seq_index'],  # Scenario 3
    ['seq_index'],  # Scenario 4
    ['seq_index'],  # Scenario 5
    ['seq_index'],  # Scenario 6
    ['seq_index'],  # Scenario 7
    ['seq_index'],  # Scenario 8
    ['seq_index'],  # Scenario 9
    ['bbox'],  # Scenario 10
    ['bbox'],  # Scenario 11
    ['bbox'],  # Scenario 12
    ['seq_index', 'bbox'],  # Scenario 13
    ['seq_index', 'bbox'],  # Scenario 14
    ['seq_index', 'bbox'],  # Scenario 15
    ['seq_index', 'unit1_blockage'],  # Scenario 16
    ['seq_index', 'unit1_blockage'],  # Scenario 17
    ['seq_index', 'unit1_blockage'],  # Scenario 18
    ['seq_index', 'unit1_blockage'],  # Scenario 19
    ['seq_index', 'unit1_blockage'],  # Scenario 20
    ['seq_index', 'unit1_blockage'],  # Scenario 21
    ['seq_index', 'unit1_blockage'],  # Scenario 22
    ['seq_index'],  # Scenario 23
    ]

# Finally, some useful and constant strings
UNIT_STR = 'unit' + str(unit) + '_'

#%% Task A - Get the path to the csv of the target scenario

# Task A.1 - list all scenarios in the data folder
all_files_in_data_folder = os.listdir(data_folder)


# Task A.2 - obtain the all scenario directories
#            (They start with "Scenario", as in "Scenario5_Tyler")
all_scenario_dir = [i for i in all_files_in_data_folder
                    if i[:8].lower() == 'scenario']

# Task A.3 - select the scenario we want to read
scenario_dir = [scen_dir for scen_dir in all_scenario_dir
                if scen_dir.split('_')[0].lower() == 'scenario' + str(scenario_idx)][0]

print(f'Opening folder: {scenario_dir}')

scen_path = os.path.join(data_folder, scenario_dir)

# Task A.4 - select the csv inside that directory: starts with "scenarioX",
#            where X is the scenario_index
csv_file = [file for file in os.listdir(scen_path) 
            if file == 'scenario' + str(scenario_idx) + '.csv'][0]

print(f'Reading: {csv_file}')

csv_path = os.path.join(scen_path, csv_file)


#%% Task B: Open CSV with the files of each sample and load relevant data 
# Task B.1 - Load csv containing the files for each sample
dataframe = pd.read_csv(csv_path)
print(f'Columns: {dataframe.columns.values}')
n_rows = dataframe.shape[0]
print(f'Number of Rows: {n_rows}')

# Task B.2 - Load data from each file
n_samples = min(max_samples, n_rows)
last_sample = n_samples
print(f'Reading {n_samples} samples... ', end='')

problematic_samples = []
data = {}
for mod in allowed_modalities:
    data[mod] = np.squeeze(np.zeros((n_samples, *DATA_DIMENSIONS[mod])))
    if mod == 'radar':
        data[mod] = data[mod].astype(np.complex64)
    files = dataframe[UNIT_STR + mod].values
    sample_range = np.arange(first_sample-1, last_sample)
    print(f'Reading modality: {mod}')
    for i in tqdm(sample_range):
        try:
            file_parts = files[i].split('/')[1:]
        except AttributeError:
            print('Path is improperly formated. Rectify CSV.')
            problematic_samples.append(i)
            
        file_path = os.path.join(scen_path, *file_parts)
        
        new_idx = np.where(sample_range == i)
        if mod in ['loc', 'loc_cal', 'pwr_60ghz']:
            data[mod][new_idx] = np.loadtxt(file_path)
        
        if mod == 'rgb':
            data[mod][new_idx] = mplimage.imread(file_path)
            
        if mod in ['lidar', 'radar']:
            data[mod][new_idx] = np.load(file_path)
        
if problematic_samples:
    print(problematic_samples)
    raise Exception('Cannot read some samples.')

problematic_samples = []
label_data = {}
for label in labels:
    label_data[label] = np.squeeze(np.zeros((n_samples, *DATA_DIMENSIONS[label])))
    # sample_range = np.arange(first_sample-1, last_sample)
    # Different labels have different ways of being read
    if label in ['seq_index']:
        label_data[label] = dataframe[label].values
    # if label in ['blockage_label']:
    #     label_data[label] = dataframe[label].values
        
    # if label in ['bbox']:
    #     for i in tqdm(sample_range):
    #         label_data[label][ = dataframe[label].values
            

#%% Task C: Create output folder and get an output path
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

for mod in allowed_modalities:
    output_file = ('scenario' + str(scenario_idx) + '_' + 
                   UNIT_STR + mod + '_' + str(first_sample) + '-' + 
                   str(last_sample))
                   
    output_path = os.path.join(output_folder, output_file)

    print(f'Writing to {output_path}... ', end='')
    
    np.save(output_path, data[mod])
    

for label in labels:    
    output_file = ('scenario' + str(scenario_idx) + '_' +
                   label + '_' + str(first_sample) + '-' + 
                   str(last_sample))
                   
    output_path = os.path.join(output_folder, output_file)

    print(f'Writing to {output_path}... ', end='')
    
    np.save(output_path, label_data[label])

print('Done.')
