# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 19:39:19 2021

@author: jamorais
"""

import os
import train_test_func as func
# data probing, KNN and Lookup table intensive

# This folder should contain the outputs from loader.py
data_folder = os.path.join(os.getcwd(), 'Ready_data_norm1')

# Folder that will have this particular set of experiments 
experiment_folder = '' # '' uses the Hyperparemeters defined below on the name.


#%% Heaving KNN  Look-up table testing 

test_KNN_all_n = False
test_lookup_all_n = True

scen_idxs = np.arange(1,9+1)

for scen_idx in scen_idxs:
    
    # The saved folder will have all experiments conducted. 
    saved_path = join_paths([os.getcwd(), 'saved_folder', f'scenario {scen_idx}'])
    
    # Create dir if doesn't exist
    if not os.path.isdir(saved_path):
        os.mkdir(saved_path)
        
    # Load data from data_folder
    x_train = np.load(os.path.join(data_folder, f"scenario{scen_idx}_x_train.npy"))
    y_train = np.load(os.path.join(data_folder, f"scenario{scen_idx}_y_train.npy"))
    x_test = np.load(os.path.join(data_folder, f"scenario{scen_idx}_x_test.npy"))
    y_test = np.load(os.path.join(data_folder, f"scenario{scen_idx}_y_test.npy"))
    
    if test_KNN_all_n :
        vals_to_test = np.arange(1,100+1)
        n_vals_to_test = len(vals_to_test)
        
        # This is exactly as above, see the comments there.
        test_KNN_accs = np.zeros((n_vals_to_test, n_top_stats))
        for n_idx in range(n_vals_to_test):
            n = vals_to_test[n_idx]
            print(f'Doing KNN for n = {n:<2}')
            
            pred_beams = []
            
            np.random.seed(0)
            total_hits = np.zeros(n_top_stats)
            for sample_idx in range(n_test_samples):
                test_sample = x_test[sample_idx]
                test_label = y_test[sample_idx]
                
                distances = np.sqrt(np.sum((x_train - test_sample)**2, axis=1))
                
                neighbors_sorted_by_dist = np.argsort(distances)
                
                best_beams_n_neighbors = y_train[neighbors_sorted_by_dist[:n]]
                pred_beams.append(mode_list(best_beams_n_neighbors))
                
                for i in range(n_top_stats):
                    hit = np.any(pred_beams[-1][:top_beams[i]] == test_label)
                    total_hits[i] += 1 if hit else 0
            
            test_KNN_accs[n_idx] = total_hits / n_test_samples
        
        best_n = np.argmax(test_KNN_accs[:,0]) + 1
        # Plot the accuracy for each value of n
        f = plt.figure(figsize=(7,4), constrained_layout=True) # PUT BACK TO 7,4
        plt.plot(vals_to_test, np.round(test_KNN_accs*100,2))
        plt.legend([f"Top-{i} Accuracy" for i in top_beams], loc='upper right',
                    bbox_to_anchor=(1.36, 1.025))
        plt.xlabel('Number of neighbors')
        plt.ylabel('Accuracy')
        plt.title(f'Scenario {scen_idx} KNN Performance for all N (best N = {best_n})')
        plt.minorticks_on()
        plt.grid()
        plt.savefig(os.path.join(saved_path, f'KNN_test_all_N_scen{scen_idx}.pdf'),
                    bbox_inches = "tight")
        # bbox_inches = "tight" is needed if we are putting things outside the 
        # normal canvas size. This is what 'inline' in Spyder uses when displaying
    
    ########
    
    if test_lookup_all_n:
        
        # List of predicted beams for the test set
        n_labels = 64
        top_beams = [1,2,3,5]
        n_top_stats = len(top_beams)
        n_test_samples = len(x_test)
        acc = np.zeros(n_top_stats)
            
        vals_to_test = np.arange(1,50+1)
        n_vals_to_test = len(vals_to_test)
        test_LT_accs =  np.zeros((n_vals_to_test, n_top_stats))
        
        for n_idx, n in enumerate(vals_to_test):
            print(f'Doing Look-up Table for n = {n:<2}')
            
            pred_beams = []
            
            np.random.seed(0)
            total_hits = np.zeros(n_top_stats)
            
            # 1- Define bins
            n_bins_across_x1 = n
            n_bins_across_x2 = n
            bin_size = np.array([1,1]) / [n_bins_across_x1, n_bins_across_x2]
            n_bins = n_bins_across_x1 * n_bins_across_x2
            
            # 2- Create a list with the samples per bin
            samples_per_bin = [[] for bin_idx in range(n_bins)]
            
            # 3- Map each input to a bin
            for x_idx, x in enumerate(x_train):
                samples_per_bin[func.pos_to_bin(x, bin_size, n_bins)].append(x_idx)
            
            # 4- Define the values to predict for samples in that bin
            prediction_per_bin = [mode_list(y_train[samples_per_bin[bin_idx]])
                                  for bin_idx in range(n_bins)]
            
            # 5- Evaluation phase. Map each test sample to a bin and get the 
            #    prediction.
            for idx, x in enumerate(x_test):
                pred_beams.append(prediction_per_bin[func.pos_to_bin(x, bin_size, n_bins)])
            
                # 6- Get top-1, top-2, top-3 and top-5 accuracies
                for i in range(n_top_stats):
                    hit = np.any(pred_beams[-1][:top_beams[i]] == y_test[idx])
                    total_hits[i] += 1 if hit else 0
        
            # Average the number of correct guesses (over the total samples)
            test_LT_accs[n_idx] = np.round(total_hits / n_test_samples, 4)
        
        
        best_n = np.argmax(test_LT_accs[:,0]) + 1
        
        # Plot the accuracy for each value of n
        f = plt.figure(figsize=(6,4), constrained_layout=True)
        plt.plot(vals_to_test, np.round(test_LT_accs*100,2))
        # plt.legend([f"Top-{i} Accuracy" for i in top_beams], loc='upper right',
        #             bbox_to_anchor=(1.36, 1.025))
        plt.xlabel('Number of Quantization bins of each coordinate')
        plt.ylabel('Accuracy [%]')
        plt.title(f'Scenario {scen_idx} Look-up Table Performance for all N (best N = {best_n})')
        plt.minorticks_on()
        plt.grid()
        plt.savefig(os.path.join(saved_path, f'LookupTable_test_all_N_scen{scen_idx}.pdf'),
                    bbox_inches = "tight")
        # bbox_inches = "tight" is needed if we are putting things outside the 
        # normal canvas size. This is what 'inline' in Spyder uses when displaying
        