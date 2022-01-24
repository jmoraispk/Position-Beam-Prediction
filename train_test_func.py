# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 13:02:42 2021

@author: jamorais
"""

import os
import utm
import csv
import time
import shutil
import numpy as np
import scipy.io as scipyio
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mplimage

import torch as t
import torch.cuda as cuda
import torch.optim as optimizer
import torch.nn as nn
import torchvision.transforms as transf
from torch.utils.data import Dataset, DataLoader

def get_experiment_name(scen_idx, n_beams, norm_type, noise):
    return f'scenario {scen_idx} beams {n_beams} norm {norm_type} noise {noise}'


def min_max(arr, ax=0):
    """ Computes min-max normalization of array <arr>. """
    return (arr - arr.min(axis=ax)) / (arr.max(axis=ax) - arr.min(axis=ax))
    

def xy_from_latlong(lat_long):
    """ Assumes lat and long along row. Returns same row vec/matrix on 
    cartesian coords."""
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat_long[:,0], lat_long[:,1])
    return np.stack((x,y), axis=1)


def add_pos_noise(pos, noise_variance_in_m=1):
    
    n_samples = pos.shape[0]
    
    # Get noise in xy coordinates
    dist = np.random.normal(0, noise_variance_in_m, n_samples)
    ang = np.random.uniform(0, 2*np.pi, n_samples)
    xy_noise = np.stack((dist * np.cos(ang), dist * np.sin(ang)), axis=1)
    
    # Get position in xy coordinates
    x, y, zn, zl = utm.from_latlon(pos[:,0], pos[:,1])
    xy_pos = np.stack((x,y), axis=1)

    # Apply noise to position and return conversion to lat_long coordinates
    xy_pos_noise = xy_pos + xy_noise
    
    lat,long = utm.to_latlon(xy_pos_noise[:,0], xy_pos_noise[:,1], zn, zl)
    pos_with_noise = np.stack((lat,long), axis=1)
    return pos_with_noise


def normalize_pos(pos1, pos2, norm_type):
    """
    Normalizations:
    1- lat&long -> min_max
    2- lat&long -> min_max (north-aware)
    3- lat&long -> cartesian -> min_max    
    4- lat&long -> cartesian -> rotation -> min_max
    5- lat&long -> cartesian -> distance & angle -> center angle at 90º -> 
                -> normalize to 0-1: divide distance and a angle by max values
    
    Advantages of each normalization:
    1- the simplest...
    2- better for transfer learning (TL)
    3- reference is the earth axis
    4- common reference: the BS. --> Should improve in Transfer Learning!
    5- same as 4, but using polar coordinates with more 
       transferable normalizations (not min_max)
    """
    
    if norm_type == 1:
        pos_norm = min_max(pos2)
    
    if norm_type == 2:
        # Check where the BS is and flip axis
        pos_norm = min_max(pos2)
        
        avg_pos2 = np.mean(pos2, axis=0)
        
        if pos1[0,0] > avg_pos2[0]:
            pos_norm[:,0] = 1 - pos_norm[:,0]
        if pos1[0,1] > avg_pos2[1]:
            pos_norm[:,1] = 1 - pos_norm[:,1]
        
    if norm_type == 3:
        pos_norm = min_max(xy_from_latlong(pos2))
        
        
    if norm_type  == 4:
        # For relative positions, rotate axis, and min_max it.
        pos2_cart = xy_from_latlong(pos2)
        pos_bs_cart = xy_from_latlong(pos1)
        avg_pos2 = np.mean(pos2_cart, axis=0)
        
        vect_bs_to_ue = avg_pos2 - pos_bs_cart
        
        theta = np.arctan2(vect_bs_to_ue[1], vect_bs_to_ue[0])
        rot_matrix = np.array([[ np.cos(theta), np.sin(theta)],
                               [-np.sin(theta), np.cos(theta)]])
        pos_transformed =  np.dot(rot_matrix, pos2.T).T
        pos_norm = min_max(pos_transformed)
        
    
    if norm_type == 5:
        pos2_cart = xy_from_latlong(pos2)
        pos_bs_cart = xy_from_latlong(pos1)
        pos_diff = pos2_cart - pos_bs_cart
        
        # get distances and angle from the transformed position
        dist = np.linalg.norm(pos_diff, axis=1)
        ang = np.arctan2(pos_diff[:,1], pos_diff[:,0])
        
        # Normalize distance + normalize and offset angle
        dist_norm = dist / max(dist)
        
        # 1- Get the angle to the average position
        avg_pos = np.mean(pos_diff, axis=0)
        avg_pos_ang = np.arctan2(avg_pos[1], avg_pos[0])
        
        # A small transformation to the angle to avoid having breaks 
        # between -pi and pi
        ang2 = np.zeros(ang.shape)
        for i in range(len(ang)):
            ang2[i] = ang[i] if ang[i] > 0 else ang[i] + 2 * np.pi
        
        avg_pos_ang2 = \
            avg_pos_ang + 2 * np.pi if avg_pos_ang < 0 else avg_pos_ang
        
        # 2- Offset angle avg position at 90º
        offset2 = np.pi/2 - avg_pos_ang2
        ang_final = ang2 + offset2
        
        # MAP VALUES OF 0-PI TO 0-1
        ang_norm = ang_final / np.pi
        
        pos_norm = np.stack((dist_norm,ang_norm), axis=1)
    
    return pos_norm


def save_data(split, filename,
              x_train, x_val, x_test, y_train, y_val, y_test, y_test_pwr):
    
    np.save(filename + '_x_train', x_train)
    np.save(filename + '_y_train', y_train)
    np.save(filename + '_x_val', x_val)
    np.save(filename + '_y_val', y_val)
    np.save(filename + '_x_test', x_test)
    np.save(filename + '_y_test', y_test)
    np.save(filename + '_y_test_pwr', y_test_pwr)
    


class DataFeed(Dataset):
    def __init__(self, x_train, y_train, transform=None):
        all_data = np.hstack((x_train, np.reshape(y_train, (len(y_train),1) )))
        self.samples = all_data.tolist()
        self.transform = transform
        self.seq_len = all_data.shape[-1]

    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pos_data = t.zeros((self.seq_len,))
        for i,s in enumerate(sample):
            x = s
            pos_data[i] = t.tensor(x, requires_grad=False)
        return pos_data


# Model Definition (Fully connected, ReLu)
class NN_FCN(nn.Module):
    def __init__(self, num_features, num_output, nodes_per_layer, n_layers):
        super(NN_FCN, self).__init__()
        self.n_layers = n_layers
        
        if n_layers < 2:
            raise Exception('A NN must include at least input and output layers.')
        
        self.layer_in = nn.Linear(num_features, nodes_per_layer)
        if n_layers > 2:
            self.std_layer = nn.Linear(nodes_per_layer, nodes_per_layer)
        
        self.layer_out = nn.Linear(nodes_per_layer, num_output)
        self.relu = nn.ReLU()
        
        
    def forward(self, inputs):
        x = self.relu(self.layer_in(inputs))
        if self.n_layers > 2:
            for n in range(self.n_layers-2):
                x = self.relu(self.std_layer(x))
        
        x = self.layer_out(x)
        return (x)


def train_net(x_train, y_train, x_val, y_val, backup_folder, 
              num_epochs, model, train_batch_size, lr, decay_L2, 
              top_stats=[1,2,3,5], rnd_seed=0, 
              fixed_GPU=True, backup_best_model=True, 
              save_all_pred_labels=True, make_plots=True,
              print_top_stats_per_epoch=False):
    
    # Make dir if doesn't exist
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)    
        
    # Copy the training files
    try:
        shutil.copy(os.path.basename(__file__), backup_folder)
    except:
        try:
            shutil.copy('4-train_test.py', backup_folder)
            shutil.copy('train_test_func.py', backup_folder)
        except:
            print('One can only copy when executed in a terminal.')
            input('Press any key to continue without a backup of the code...')
    
    # Save CSV with the predicted labels of each epoch?
    if save_all_pred_labels:
        save_directory = os.path.join(backup_folder,'saved_analysis_files')
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
    
    # Save the model of best epoch
    if backup_best_model:
        checkpoint_directory = os.path.join(backup_folder, 'model_checkpoint')    
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
            
        net_name = os.path.join(checkpoint_directory, 'nn_position_beam_pred')
    
    # Before the loaders since they will shuffle data!
    t.manual_seed(rnd_seed)
    t.backends.cudnn.deterministic = True
    
    # Data Input to Torch
    proc_pipe = transf.Compose([transf.ToTensor()])
    
    # Create torch DataLoaders
    train_loader = DataLoader(DataFeed(x_train, y_train, transform=proc_pipe),
                              batch_size=train_batch_size, #num_workers=8,
                              shuffle=True)
    
    val_loader =  DataLoader(DataFeed(x_val, y_val, transform=proc_pipe),
                             batch_size=1, #num_workers=8,
                             shuffle=False)
    # shuffle False to match the y_val (in case of save pred labels=True)
    
    # We're collecting top1, top2, top3, and top5 statistics: in top_stats
    n_top_stats = len(top_stats)
    n_val_samples = len(y_val)
    n_labels = model.layer_out.out_features
    
    # Select GPU
    if fixed_GPU:
        cuda_device_id = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    else: # pick random GPU
        gpu_id = np.random.choice(np.arange(0,cuda.device_count()))
        cuda_device_id = t.device(f"cuda:{gpu_id}") 
       
    
    # Per batch running losses
    running_training_loss = np.zeros(num_epochs)
    running_val_loss = np.zeros(num_epochs)
    
    # Accuracies
    running_accs = np.zeros((num_epochs, n_top_stats))
    best_accs = np.zeros(n_top_stats)
    
    # All labels predicted in the test set
    all_pred_labels = np.zeros((num_epochs, n_val_samples, n_labels))
    # all_test_labels = np.zeros((num_epochs, n_val_samples))
        
    # Model Training
    t_0 = time.time()
    
    # For reproducibility
    with cuda.device(cuda_device_id):
        
        # Build the network:
        net = model.cuda()

        #  Optimization parameters:
        criterion = nn.CrossEntropyLoss()
        opt = optimizer.Adam(net.parameters(), lr=lr, weight_decay=decay_L2)
        LR_sch = optimizer.lr_scheduler.MultiStepLR(opt, [20,40], gamma=0.2)
        # "Decays the learning rate of each parameter group by 
        #  gamma once the number of epoch reaches one of the milestones."
        
        # Converges slower, less accurate, but is more robust 
        # (i.e. less variability across runs)
        # opt = optimizer.AdamW(net.parameters(), lr=lr, weight_decay=decay_L2)
        # LR_sch = optimizer.lr_scheduler.ReduceLROnPlateau(opt, 'min')
        
        for epoch in range(num_epochs):
            print(f'========== Epoch No. {epoch+1: >2} ==========')
            t_1 = time.time()
            
            # Cummulative training/test validation losses
            training_cum_loss = 0
            val_cum_loss = 0
            
            # Data aspect: X is in first (2) positions, Label is on last (3rd)
            for tr_count, y in enumerate(train_loader):
                net.train()
                x = y[:, :-1].type(t.Tensor).cuda()
                label = y[:, -1].long().cuda()
                opt.zero_grad()
                out = net.forward(x)
                batch_loss = criterion(out, label)
                batch_loss.backward()
                opt.step()                    
                training_cum_loss += batch_loss.item()
            
            # Each batch loss is the average loss of each sample in it. 
            # Avg. over batches to obtain the per sample training loss avg.
            running_training_loss[epoch] = training_cum_loss / (tr_count + 1)
            
            print('Start validation')
            
            # List of best 5 Predicted Beams for each test sample
            total_hits = np.zeros(n_top_stats)
            
            for idx, data in enumerate(val_loader):
                net.eval()
                x = data[:, :-1].type(t.Tensor).cuda()
                label = data[:, -1].long()
                opt.zero_grad()
                out = net.forward(x)
                val_cum_loss += criterion(out, label.cuda()).item()
                label = label.cpu().numpy()
                
                # Sort labels according to activation strength
                all_pred_labels[epoch, idx] = \
                    t.argsort(out, dim=1, descending=True).cpu().numpy()[0]
                
                # If the best beam is in the topX, then +1 hit for that batch
                for i in range(n_top_stats):
                    hit = np.any(all_pred_labels[epoch, idx, :top_stats[i]] == label)
                    total_hits[i] += 1 if hit else 0
            
            # Average the number of correct guesses (over the total samples)
            running_accs[epoch,:] = total_hits / n_val_samples
            
            # Gather avg. loss of each test sample
            running_val_loss[epoch] = val_cum_loss / n_val_samples
            
            if print_top_stats_per_epoch:
                for i in range(n_top_stats):
                    print(f'Average Top-{top_stats[i]} accuracy '
                          f'{running_accs[epoch,i]*100:.2f}')
            
            # Check if current accuracy of top-1 beam surpasses the best so far
            if running_accs[epoch, 0] > best_accs[0]:
                print("NEW BEST!")
                if backup_best_model:
                    t.save(net.state_dict(), net_name)
                best_accs[:] = running_accs[epoch, :]
                best_epoch = epoch + 1
                
            print(f'Curr (top-1) accuracy: {running_accs[epoch, 0]*100:2.2f}%')
            print(f'Best (top-1) accuracy: {best_accs[0]*100:2.2f}%')
            
            # Take a learning step
            LR_sch.step()
            # With ReduceLROnPlateau: LR_sch.step(running_val_loss[epoch])
            
            print(f'Time taken for epoch {epoch+1}: {time.time() - t_1:.2f} s.')
                  
            
    # Write all predicted beams, for each sample to a CSV
    if save_all_pred_labels:
        print("Saving the predicted value in a csv file")  
        for epoch in range(num_epochs):
            predicted_csv_name = f"pred_beams_epoch_{epoch+1}.csv"
            csv_path = os.path.join(save_directory, predicted_csv_name)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(zip(y_val, all_pred_labels[epoch]))
    
    print('--------------------------------------------')
    print(f'Total time taken for training: {(time.time() - t_0):.2f} s.')
    
    print(f'Best Epoch: {best_epoch}')
    print('Best Validation Results:')
    for i in range(n_top_stats):
        print(f'\tAverage Top-{top_stats[i]} validation accuracy '
              f'{best_accs[i]*100:.2f}')
          
    # Save what was best epoch
    with open(os.path.join(backup_folder, f'best_epoch={best_epoch}.txt'), 'w'):
        pass
    
    # Save best accuracies
    np.savetxt(os.path.join(backup_folder, 'best_val_accs.txt'), 
               best_accs * 100, fmt='%.2f')
    
    if make_plots:
        # Plot Top-1, Top-2, Top-3 validation accuracies across epochs
        epochs = np.arange(1, num_epochs+1)
        plt.figure()
        plt.plot(epochs, running_accs[:,0], 'g*-', lw=2.0, label='Top-1 Accuracy')
        plt.plot(epochs, running_accs[:,1], 'b*-', lw=2.0, label='Top-2 Accuracy')
        plt.plot(epochs, running_accs[:,2], 'r*-', lw=2.0, label='Top-3 Accuracy')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Validation Accuracy [%]')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(backup_folder, 'position_beam_val_acc.pdf'))
    
        # Plot Training vs Validation loss across epochs
        plt.figure()
        plt.plot(epochs, running_training_loss, 'g*-', lw=2.0, label='Training Loss')
        plt.plot(epochs, running_val_loss, 'b*-', lw=2.0, label='Validation Loss')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Average Loss per sample')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(backup_folder, 'training_vs_validation_loss.pdf'))
    
    return net_name


def test_net(x_test, model):
    
    # Data Input to Torch
    proc_pipe = transf.Compose([transf.ToTensor()])
    n_test_samples = x_test.shape[0]
    test_loader = DataLoader(DataFeed(x_test, np.zeros(n_test_samples), 
                                      transform=proc_pipe),
                             batch_size=1, shuffle=False)
    # shuffle = False is important! This way, test labels have the same order.
    
    n_labels = model.layer_out.out_features
    all_pred_labels = np.zeros((n_test_samples, n_labels))
    
    cuda_device_id = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    with cuda.device(cuda_device_id):
        net = model.cuda()
        net.eval() # a evaluation switch: turns off Dropouts, BatchNorms, ...
           
        for idx, data in enumerate(test_loader):
            x = data[:, :2].type(t.Tensor).cuda()
            out = net.forward(x)
            
            # Sort labels according to activation strength
            all_pred_labels[idx] = \
                t.argsort(out, dim=1, descending=True).cpu().numpy()[0]
        
    return all_pred_labels.astype(int)


def join_paths(path_list):
    """ Joins paths with os.path.join(). """
    n_path_parts = len(path_list)
    
    if n_path_parts < 2:
        raise Exception('Path list must have 2 or more elements to join.')
    
    s = os.path.join(path_list[0], path_list[1])
    
    if n_path_parts > 2:
        for path_idx in range(2, n_path_parts):
            s = os.path.join(s, path_list[path_idx])
            
    return s

def mode_list(arr):
    """ Returns ordered list based on # of occurences in 1D array. """
    vals, counts = np.unique(arr, return_counts=True)
    return vals[np.flip(np.argsort(counts))]


def pos_to_bin(pos, bin_size, n_bins):
    # The bin indices will be flattened out
    # 
    # x2
    # ^
    # | d e f
    # | a b c
    # --------> x1
    # 
    # Will be mapped to: a b c d e f
    if pos[0] == 1:
        pos[0] -= 1e-9

    if pos[1] == 1:
        pos[1] -= 1e-9
        
    bin_idx = int(np.floor(pos[0] / bin_size[0]) + 
                  1 / bin_size[0] * np.floor(pos[1] / bin_size[1]))    
    
    return max(min(bin_idx, n_bins-1), 0)

def print_number_of_samples(x_train, x_val, x_test, y_train, y_val, y_test):
    
    n_train_samples = len(y_train)
    n_val_samples = len(y_val)
    n_test_samples = len(y_test)
    n_samples = n_train_samples + n_val_samples + n_test_samples
    
    print(f'Samples in Training: {n_train_samples}\n'
          f'Samples in Validation: {n_val_samples}\n'
          f'Samples in Test: {n_test_samples}\n'
          f'Total samples: {n_samples}')
    
    print(f'x_train is {x_train.shape}\n'
          f'y_train is {y_train.shape}\n'
          f'x_val   is {x_val.shape}\n'
          f'y_val   is {y_val.shape}\n'
          f'x_test  is {x_test.shape}\n'
          f'x_test  is {y_test.shape}')
    
    
def deg_to_dec(d, m, s, direction='N'):
    if direction in ['N', 'E']:
        mult = 1
    elif direction in ['S', 'W']:
        mult = -1
    else:
        raise Exception('Invalid direction.')
        
    return mult * (d + m/60 + s/3600)


def get_corners_for_GPS_pic(scen_idx):
    # Load corners and image from Google Earth
    if scen_idx in [1,2]:
        gps_bottom_left = [deg_to_dec(33,25,14.49, 'N'),
                           deg_to_dec(111,55,45.06, 'W')]
        gps_top_right   = [deg_to_dec(33,25,12.35, 'N'),
                           deg_to_dec(111,55,43.67, 'W')]
    if scen_idx in [3,4]:
        gps_bottom_left = [deg_to_dec(33,25,4.31, 'N'),
                           deg_to_dec(111,55,33.85, 'W')]
        gps_top_right   = [deg_to_dec(33,25, 6.72, 'N'),
                           deg_to_dec(111,55,35.96, 'W')]
    if scen_idx == 5:
        gps_bottom_left = [deg_to_dec(33,25,15.62, 'N'), 
                           deg_to_dec(111,55,45.17, 'W')]
        gps_top_right   = [deg_to_dec(33,25,13.63, 'N'), 
                           deg_to_dec(111,55,43.93, 'W')]
    if scen_idx == 6:
        gps_bottom_left = [deg_to_dec(33,25,36.25, 'N'), 
                           deg_to_dec(111,55,46.89, 'W')]
        gps_top_right   = [deg_to_dec(33,25,33.17, 'N'),
                           deg_to_dec(111,55,44.87, 'W')]
    if scen_idx == 7:
        gps_bottom_left = [deg_to_dec(33,15,29.88, 'N'), 
                           deg_to_dec(111,51,32.76, 'W')]
        gps_top_right   = [deg_to_dec(33,15,31.96, 'N'),  
                           deg_to_dec(111,51,34.10, 'W')]
    if scen_idx in [8,9]:
        gps_bottom_left = [deg_to_dec(33,25,10.54, 'N'),
                           deg_to_dec(111,55,44.62, 'W')]
        gps_top_right   = [deg_to_dec(33,25,8.62, 'N'), 
                           deg_to_dec(111,55,43.45, 'W')]

    return (gps_bottom_left, gps_top_right)


def get_stats_of_data(stats, pos1, pos2, pwr1, scen_idx):
    
    n_samples = len(pwr1)
    n_labels = pwr1.shape[-1]
    beam_labels = np.argmax(pwr1, axis=1)
    
    # 1- Avg SNR, noise and maximum power
    if 1 in stats:
        max_min_pwr_ratio_per_sample = [pwr1[i,beam_labels[i]] / np.min(pwr1[i,:]) 
                                        for i in range(n_samples)]
        avg_clearance_db = 10 * np.log10(np.mean(max_min_pwr_ratio_per_sample))
        print(f"avg SNR = {avg_clearance_db:.2f} dB.")
        print(f"avg_noise_floor = {np.mean(np.min(pwr1, axis=1)):.4f}")
        print(f"avg max power = {np.mean(np.max(pwr1, axis=1)):.4f}")
    
    # 2- Avg Distance between BS and UE
    if 2 in stats:
        pos1_cart = xy_from_latlong(pos1)
        pos2_cart = xy_from_latlong(pos2)
        dist = np.linalg.norm(pos2_cart - pos1_cart, axis=1)
        # dist_avg_pos = np.linalg.norm(np.mean(pos2_cart - pos1_cart, axis=0))
        print(f"avg distance = {np.mean(dist):.2f} m.")
        # print(f"distance of avg position = {dist_avg_pos:.2f} m.")
        # these two are 98% correlated.... not very useful.
    
    # 3- Count how many beams( on avg.) have powers within 70% of the max power
    if 3 in stats: 
        thres = 0.7
        max_power_per_sample = np.max(pwr1, axis=1)
        n_beams_within_thres = [np.sum(pwr1[i] > thres * max_power_per_sample[i])
                                for i in range(n_samples)]
        avg_n_beams = np.mean(n_beams_within_thres)
        print(f"avg # beams within {thres*100:.0f}% of max = {avg_n_beams:.2f} beams.")
    
    # 4- Power noise: 
    # Check previous and next adjacent samples. 
    # If they have the same "best beam", quantify the maximum 
    # variability (largest-smallest)/pwr of curr sample
    if 4 in stats:
        normed_variability = np.zeros(n_samples-2)
        for sample_idx in range(n_samples):
            if sample_idx not in [0, n_samples-1]:
                # best beam index
                bb = beam_labels[sample_idx]
                vals = np.stack((pwr1[sample_idx-1, bb],
                                 pwr1[sample_idx,bb],
                                 pwr1[sample_idx+1, bb]))
                normed_variability[sample_idx-1] = \
                    (max(vals) - min(vals)) / pwr1[sample_idx,bb]
                    
        print(f'Beam power variability {np.mean(normed_variability):.4f}')

    if 5 in stats:
        # plot and save the array of the average power profile
        
        # There are two ways of normalizing the power for this plot. 
        # 1- min max across all data
        # norm_pwr = min_max(pwr1)
        # 2- divide by max of each sample
        norm_pwr = pwr1 / np.max(pwr1, axis=1)[:,None]
        
        max_idxs = np.argmax(pwr1, axis=1)
        
        # aggregate samples in array to average at the end
        acc_pwrs = np.zeros(pwr1.shape) # n_beams
        
        idx_in_middle = int(np.floor(n_labels/2))
        
        # center the powers before accumulating
        for sample_idx in range(n_samples):
            for idx_in_acc in range(n_labels):
                if idx_in_acc < idx_in_middle: # left part
                    diff = idx_in_middle - idx_in_acc # always positive
                    original_idx = max_idxs[sample_idx] - diff
                    if original_idx < 0:
                        original_idx += 64
                else: # right part
                    diff = idx_in_acc - idx_in_middle
                    original_idx = (max_idxs[sample_idx] + diff) % 63
                acc_pwrs[sample_idx, idx_in_acc] = norm_pwr[sample_idx, original_idx]
        
        pwr_footprint = np.mean(acc_pwrs, axis=0)
        
        plt.plot(pwr_footprint, 'x-', label=f'scen-{scen_idx}')
        # et_xticks([idx_in_middle])
        # ax.set_xticklabels([idx_in_middle])
        # plt.plot(pwr_footprint, label=f'scen-{scen_idx}')
        plt.xlim([25, 38])
        plt.ylim([0.7, 1])
        
        plt.legend(loc='upper left', ncol=1)
        if scen_idx == 9:
            plt.savefig('test3.svg')
        
        # fig, ax = plt.subplots()
        # ax.plot(pwr_footprint, label=f'scen-{scen_idx}')
        # ax.set_xticks([idx_in_middle])
        # ax.set_xticklabels([idx_in_middle])
        # ax.legend(loc='upper left', ncol=1)
        # ax.set_xlim([25, 38])
        # ax.set_ylim([0.7, 1])
        np.save(f'scen_{scen_idx}', pwr_footprint)
        scipyio.savemat(f'scen_{scen_idx}.mat', {'data': pwr_footprint})
    
    
    if 6 in stats:

        pics_folder = 'GPS_pics'
        if not os.path.isdir(pics_folder):
            raise Exception(f'{pics_folder} does not exists. '
                            'Create folder with GPS pictures.')
        
        pic_name = f'{scen_idx}.png'
        GPS_img = mplimage.imread(os.path.join(pics_folder,pic_name))
        
        gps_bottom_left, gps_top_right = get_corners_for_GPS_pic(scen_idx)
        fig = plt.figure()
        ax = fig.add_subplot()
        
        # ax.set_xticks([idx_in_middle])
        # ax.set_xticklabels([idx_in_middle])
        # ax.legend(loc='upper left', ncol=1)
        # ax.set_xlim([25, 38])
        # ax.set_ylim([0.7, 1])
        
        ax.scatter(pos1[0,0], pos1[0,1], s=180, marker='h', c='grey',
                   edgecolors='black', zorder=1)
        # ax.scatter(pos1[0,0], pos1[0,1], s=20, c='black', zorder=1)
        
        ax.imshow(GPS_img, aspect='equal', # 'auto'
                   zorder=0, extent=[gps_bottom_left[0], gps_top_right[0],
                                     gps_bottom_left[1], gps_top_right[1]])

        scat = ax.scatter(pos2[:,0], pos2[:,1], vmin=1, vmax=n_labels,
                          c=beam_labels, s=13, edgecolor='black', linewidth=0.2, 
                          cmap=plt.cm.jet)
        
        # ax.set_title(f'Scenario {scen_idx} [{n_samples} datapoints]')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        # cbar = fig.colorbar(scat, fraction=0.0305, pad=0.04)
        # cbar.set_ticks([1,16,32,48,64])
        # cbar.ax.set_ylabel('Beam index', rotation=270, labelpad=15)
        plt.tight_layout()
        base_name = f'scen_{scen_idx}_position_on_GPS'
        # plt.savefig(base_name + '.svg')
        plt.savefig(base_name + '.eps', bbox_inches='tight')
        
    
def write_results_together(ai_strategy, top_beams, runs_folder, n_runs,
                           val_accs, test_accs, mean_power_losses):

    """Writes results to file like this:
    Validation Results: 
    Top-1 average accuracy 89.15 % and standard deviation 3.2212 %.
    Top-2 average accuracy 97.64 % and standard deviation 1.4771 %.
    Top-3 average accuracy 98.96 % and standard deviation 1.6585 %.
    Top-5 average accuracy 99.72 % and standard deviation 0.4308 %.
    
    Test Results: 
    Top-1 average accuracy 85.18 % and standard deviation 2.0283 %.
    Top-2 average accuracy 96.94 % and standard deviation 0.1171 %.
    Top-3 average accuracy 99.09 % and standard deviation 0.4145 %.
    Top-5 average accuracy 99.73 % and standard deviation 0.0857 %.
    
    Power Loss results
    Mean:0.53, STD: 0.2021 
    """
    results_file = os.path.join(runs_folder, f'{n_runs}-runs_results_summary.txt')
    with open(results_file, 'w') as fp:
        if ai_strategy == 'NN':
            fp.write('Validation Results: \n')
            for i in range(len(top_beams)):
                s = f'Top-{top_beams[i]} average accuracy ' + \
                    f'{np.mean(val_accs[:,i]):.2f} % and ' + \
                    f'standard deviation {np.std(val_accs[:,i]):.4f} %.\n'
                print(s, end='')
                fp.write(s)
            fp.write('\n')
        fp.write('Test Results: \n')
        # For test accuracy results
        for i in range(len(top_beams)):
            s = f'Top-{top_beams[i]} average accuracy ' + \
                 f'{np.mean(test_accs[:,i]):.2f} % and ' + \
                 f'standard deviation {np.std(test_accs[:,i]):.4f} %.\n'
            print(s, end='')
            fp.write(s)
        fp.write('\n')
        # For test Power loss results.
        fp.write('Power Loss results\n')
        fp.write(f"Mean:{np.mean(mean_power_losses):.2f}, "
                 f"STD: {np.std(mean_power_losses):.4f} ")
        
    
def write_results_separate(top_beams, results_folder, n_runs,
                           val_accs, test_accs, mean_power_losses):

    """See example of previous function. 
    This function writes mean and standard deviation results like this:
    top1_val_acc.txt:
    89.15
    3.2212
    top5_test_acc.txt:
    99.73
    0.0857
    mean_pwr_loss_db.txt
    0.53
    0.2021 
    """
    variables = [val_accs, test_accs, mean_power_losses]
    
    for idx, var in enumerate(variables):
        if idx != 2: # (power loss doesn't have top-X results)
            for i, top_beam in enumerate(top_beams):
                mean_and_std = np.array([np.mean(var[:,i]), np.std(var[:,i])])
                acc_str = 'val' if idx == 0 else 'test'
                fname = os.path.join(results_folder,
                                     f'{n_runs}-runs_top-{top_beam}_'
                                     f'{acc_str}_acc.txt',)
                np.savetxt(fname, mean_and_std, fmt='%.2f')
        else:
            mean_and_std = np.array([np.mean(var), np.std(var)])
            fname = os.path.join(results_folder,
                                 f'{n_runs}-runs_mean_power_loss_db.txt')
            np.savetxt(fname, mean_and_std, fmt='%.2f')
    
        
##############################################################################
################################# PLOTS ######################################
##############################################################################


def lookuptab_pred(data, background_pic_label, bin_size, x_train, y_train, n_beams,
                   scat_size, n_bins_across_x1, n_bins_across_x2, color_map,
                   opacity, output_folder, title, plt_name):
    """ Plots the prediction cells of the look-up table against a scatter plot
    of some data: 
        - train data can be used to check whether this is working
        - test data can be used to justify predictions
    """
    
    fig, ax = plt.subplots()
    h_lines = np.arange(0,1+1e-9, bin_size[1])
    v_lines = np.arange(0,1+1e-9, bin_size[0])
    
    m = 0 # 2e-2 # margin
    ax.set_xlabel('$X_1$ Normalized')
    ax.set_ylabel('$X_2$ Normalized')
    ax.set_xlim([0-m,1+m])
    ax.set_ylim([0-m,1+m])
    ax.vlines(v_lines, ymin=0, ymax=1, linewidth=0.8)
    ax.hlines(h_lines, xmin=0, xmax=1, linewidth=0.8)
    
    data = np.reshape(np.array(data), 
                      (n_bins_across_x1, n_bins_across_x2))

    im = ax.imshow(np.flipud(data), vmin=np.nanmin(data), vmax=np.nanmax(data), 
                   cmap=color_map, extent=[0,1,0,1], alpha=opacity)
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel(background_pic_label, rotation=270, labelpad=15)
    plt.scatter(x_train[:,0], x_train[:,1], vmin=1, vmax=n_beams, c=y_train, 
                s=scat_size, cmap=plt.cm.jet, edgecolor='black', linewidth=0.5)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Beam index of Sample', rotation=270, labelpad=15)
    
    plt.title(title)
    
    # fig.tight_layout()
    plt.savefig(os.path.join(output_folder, plt_name))
    
    
def lookup_table_plots(lt_plots, n_beams, scen_idx, run_folder, 
                       n_bins_across_x1, n_bins_across_x2, bin_size, n_bins, n,
                       prediction_per_bin, samples_per_bin, 
                       x_train, y_train, x_test, y_test):
    
    if lt_plots == 'all':
         lt_plots = ['beam_beam_prediction', 
                     'certainty_of_prediction_scatter_train',
                     'certainty_of_prediction_scatter_test',
                     'histogram_samples_per_bin', 'image_samples_per_bin']
    
    n_test_samples = len(x_test)
    
    # Create an image where each 'pixel' will be a square on the
    # grid. Each pixel should represent a) or b).
    if 'beam_beam_prediction' in lt_plots:
        # a) the best beam
        data = [pred[0] if pred.size > 0 else np.nan 
                for pred in prediction_per_bin]
            
        # % of test sampls outside of current table
        count = 0
        for x in x_test:
            if prediction_per_bin[pos_to_bin(x, bin_size, n_bins)].size == 0:
                count += 1
        print(f'{round(count/n_test_samples * 100, 2)} % of test samples '
               'lie outside the table and are predicted randomly.')
        
        lookuptab_pred(data, 'Beam index of prediction',
                       bin_size, x_train, y_train, n_beams, 10,
                       n_bins_across_x1, n_bins_across_x2,
                       'jet', 0.5, 
                       run_folder, 
                       f"Scenario {scen_idx} - Look-up Table Prediction "
                       f"vs Training Data (N = {n})",
                       f"scen{scen_idx}_lookup_pred_vs_training_data_n={n}.pdf")
        
    if 'certainty_of_prediction_scatter_train' in lt_plots or \
       'certainty_of_prediction_scatter_test' in lt_plots:
    
        # b) the percentage of certainty we have for that cell 
        #    (by assessing the relative percentage of the most common
        #     best beam among all contenders from each sample)
        #    Note: this measures 'how sure' the Lookup-table it's
        #    answer, it doesn't mean the answer is correct.
        certainty_of_best = np.zeros(n_bins)
        for bin_idx in range(n_bins):
            vals, counts = np.unique(y_train[samples_per_bin[bin_idx]], 
                                     return_counts=True)
            if vals.size == 0:
                certainty_of_best[bin_idx] = np.nan
            else:
                n_samples = sum(counts)
                n_samples_for_most_common = np.max(counts)
                certainty_of_best[bin_idx] = \
                    (n_samples_for_most_common / n_samples)
            
        if 'certainty_of_prediction_scatter_train' in lt_plots:
            lookuptab_pred(certainty_of_best, "Certainty in the table",
                           bin_size, x_train, y_train, n_beams, 5,
                           n_bins_across_x1, n_bins_across_x2,
                           'viridis', 0.6,
                           run_folder,
                           f"Scenario {scen_idx} - Look-up Table "
                           f"Certainty vs Train Data (N = {n})",
                           f"scen{scen_idx}_lookup_certainty_vs_train_n={n}.pdf")
            
        if 'certainty_of_prediction_scatter_test' in lt_plots:
            lookuptab_pred(certainty_of_best, "Certainty in the table",
                           bin_size, x_test, y_test, n_beams, 5,
                           n_bins_across_x1, n_bins_across_x2,
                           'viridis', 0.6,
                           run_folder,
                           f"Scenario {scen_idx} - Look-up Table "
                           f"Certainty vs Test Data (N = {n})",
                           f"scen{scen_idx}_lookup_certainty_vs_test_n={n}.pdf")
        
    if 'histogram_samples_per_bin' in lt_plots or \
       'image_samples_per_bin' in lt_plots:    
        n_samples_per_bin = []
        for bin_idx in range(n_bins):
            n_samples_per_bin.append(len(samples_per_bin[bin_idx]))    
    
    if 'histogram_samples_per_bin' in lt_plots:
    # Histogram for the number of samples in each bin
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.bar(np.arange(n_bins), n_samples_per_bin, edgecolor='black', linewidth=1)
        for rect in ax.patches:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x()+rect.get_width()/2, height), 
                        xytext=(0, 5), textcoords='offset points', 
                        ha='center', va='bottom') 
        # ax.set_xticks(label_range[::3])
        plt.title(f'Scenario {scen_idx} - Beam frequency in train set')
        plt.xlabel('Beam index')
        plt.ylabel('Frequency')
        fig.tight_layout()
        plt.savefig(os.path.join(run_folder, 'beam_freq_in_training_set.pdf'))
        
    if 'image_samples_per_bin' in lt_plots:
        lookuptab_pred(n_samples_per_bin, 'Number of samples per cell',
                       bin_size, x_train, y_train, n_beams, 10,
                       n_bins_across_x1, n_bins_across_x2,
                       'jet', 0.5, 
                       run_folder, 
                       f"Scenario {scen_idx} - Number of samples per bin (N = {n})",
                       f"scen{scen_idx}_lookup_samples_per_bin_vs_"
                       f"training_data_n={n}.pdf")
        
    
def evaluate_predictors(evaluations, pred_beams, x_test, y_test, n_beams, 
                        scen_idx, ai_strategy, n, run_folder):
    
    if evaluations == 'all':
        evaluations = ['confusion_matrix', 'prediction_error',
                       'prediction_error2', 'positions_colored_by_error']
    
    best_pred_beam_per_sample = [prediction[0] \
         if prediction.size > 0 else round(np.random.uniform(1,64))
         for prediction in pred_beams]
    
    n_test_samples = len(y_test)
    true_labels = y_test
    pred_labels = best_pred_beam_per_sample 
    pred_errors = pred_labels - true_labels
    if 'confusion_matrix' in evaluations:
        # Plot Confusion Matrix
        fig = plt.figure()
        conf_matrix = np.zeros((n_beams, n_beams))
        for i in range(n_test_samples):
            # true labels across rows, pred across cols
            conf_matrix[true_labels[i]-1, pred_labels[i]-1] += 1
        ax = sns.heatmap(conf_matrix / np.max(conf_matrix), cmap='jet')
        ax.invert_yaxis()
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.locator_params(axis='x', nbins=8)
        plt.title(f'Scenario {scen_idx} - {ai_strategy} Confusion Matrix (N={n})')
        plt_name = f'scen_{scen_idx}_{ai_strategy}_confusion_matrix_N={n}.pdf'
        plt.savefig(os.path.join(run_folder, plt_name))
        
    if 'prediction_error' in evaluations:
        # Plot Prediction Error
        plt.figure()
        plt.scatter(true_labels, pred_errors, s=13, color='red')
        plt.xlabel('Ground-Truth Beam')
        plt.ylabel('Prediction Error')
        plt.grid(linestyle='--')
        max_lim = np.max((pred_errors.max(), np.abs(pred_errors.min())))
        plt.ylim([-max_lim-1, max_lim+1])
        plt.title(f'Scenario {scen_idx} - {ai_strategy} Prediction Error (N={n})')
        plt_name = f'scen_{scen_idx}_{ai_strategy}_pred_errors_N={n}.pdf'
        plt.savefig(os.path.join(run_folder, plt_name))
    
    
    if 'prediction_error2' in evaluations:
        # Plot Prediction Error with area proportional to the 
        # number of errors.
        
        plt.figure()
        unique_true_labels = np.unique(true_labels)
        
        # variables for the 'special scatter plot'
        true_labels_repeated = []
        pred_errors_repeated = []
        pred_error_count = []
        # count number of different prediction errors
        for true_label in unique_true_labels:
            # repeat label for each error found
            errors_for_label = \
                np.unique(pred_errors[true_labels == true_label])
            
            n_errors = len(errors_for_label)
            true_labels_repeated.extend([true_label] * n_errors )
            
            pred_errors_repeated.extend(errors_for_label)
            
            for pred_error in errors_for_label:
                pred_error_count.append(
                    np.sum(pred_errors[true_labels == true_label] == pred_error))
        
        x_arr = np.array(true_labels_repeated)
        y_arr = np.array(pred_errors_repeated)
        s_arr = np.sqrt(np.array(pred_error_count)**2)
        plt.scatter(x_arr, y_arr, s=s_arr, color='red')
        
        plt.xlabel('Ground-Truth Beam')
        plt.ylabel('Prediction Error')
        plt.grid(linestyle='--')
        max_lim = np.max((pred_errors.max(), np.abs(pred_errors.min())))
        plt.ylim([-max_lim-1, max_lim+1])
        plt.title(f'Scenario {scen_idx} - {ai_strategy} Prediction Error (N={n})')
        plt_name = f'scen_{scen_idx}_{ai_strategy}_pred_errors_N={n}.pdf'
        plt.savefig(os.path.join(run_folder, plt_name))
    
    
    if 'positions_colored_by_error' in evaluations:
        # Plot Position of Test Samples evaluated based on the error (# of beams difference)
        # diff = np.abs(3-2)
        fig = plt.figure()
        scat = plt.scatter(x_test[:,0], x_test[:,1], 
                           vmin=0, vmax=np.max(np.abs(pred_errors)), 
                           c=np.abs(pred_errors), 
                           s=15, cmap='jet')
        # cbar = plt.colorbar(scat)
        plt.xlabel('$X_1$ Normalized')
        plt.ylabel('$X_2$ Normalized')
        cbar = fig.colorbar(scat)
        cbar.ax.set_ylabel('# beams off', rotation=270, labelpad=15)
        plt.grid()
        m = 1e-2 # margin
        plt.xlim([0-m, 1+m])
        plt.ylim([0-m, 1+m])
        plt.title(f'Scenario {scen_idx} - {ai_strategy} ' + \
                  f'Prediction Error of each test sample (N={n})')
        plt_name = f'scen_{scen_idx}_{ai_strategy}_pos_map_error_N={n}.pdf'
        plt.savefig(os.path.join(run_folder, plt_name))
    
    
def prediction_map(N, ai_strategy, n_beams, scen_idx, run_folder,
                   x_train, y_train, x_test, n, prediction_per_bin, 
                   bin_size, n_bins, trained_nn):
    
    """
    Prediction Map
    1- Create a vector with N samples uniformly spread across the 
        input feature space
    2- Pass samples through predictors
    3- Map the responses in an image plot
    """
    
    # 1- Spreads uniformely and deterministically at least N points 
    #    throughout the feature space. 
    #    Assumes all dimensions are normalized between 0 and 1.
    #    Returns a [N, x_train.ndim] array with each point.
    
    n_per_dim = int(np.ceil(np.sqrt(N)))
    actual_N = n_per_dim**x_train.ndim
    vals_across_one_dim = np.linspace(0,1,n_per_dim+2)[1:-1]
    replicated_tup = tuple([vals_across_one_dim] * x_train.ndim)
    new_x = \
        np.stack(np.meshgrid(*replicated_tup), -1).reshape(-1, x_train.ndim)
    
    # 2- Apply to each method...
    new_y_pred = np.zeros(actual_N)
    if ai_strategy == 'KNN':
        
        for idx, test_sample in enumerate(new_x):
            # Distances to each sample in training set
            distances = np.sqrt(np.sum((x_train - test_sample)**2, axis=1))
            
            # Find the indices of the closest neighbors
            neighbors_sorted_by_dist = np.argsort(distances)
            
            # Take the mode of the best beam of the n closest neighbors
            best_beams_n_neighbors = y_train[neighbors_sorted_by_dist[:n]]
            
            new_y_pred[idx] = mode_list(best_beams_n_neighbors)[0]
    
    if ai_strategy == 'LT':
        # The table is computed already, we just need to apply it.
        for idx, x in enumerate(new_x):
            pred = prediction_per_bin[pos_to_bin(x, bin_size, n_bins)]
            if pred.size == 0:
                pred = int(np.random.uniform(0, n_beams))
            else:
                pred = pred[0]
            new_y_pred[idx] = pred
                

    if ai_strategy == 'NN':
        # Get results from that model
        pred_beams = test_net(new_x, trained_nn)
        new_y_pred = pred_beams[:,0]

    # 3- Turn predictions array to an image and plot the image.
    img_data = np.reshape(np.array(new_y_pred), (n_per_dim, n_per_dim))
    fig, ax = plt.subplots()
    im = ax.imshow(np.flipud(img_data), vmin=1, vmax=n_beams, 
                   cmap='jet', extent=[0,1,0,1])
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('Beam Prediction', rotation=270, labelpad=15)
    plt.xlabel('g$_{lat}$ normalized', fontsize=13)
    plt.ylabel('g$_{long}$  normalized', fontsize=13)
    # plt.title(f'Scenario {scen_idx} - {ai_strategy} '
    #           f'Prediction Map_N={n}_beams={n_beams}')
    plt_name = (f'scen_{scen_idx}_{ai_strategy}_pos_map_error_'
                f'N={n}_beams={n_beams}')
    scipyio.savemat(plt_name[:-4], {'data': img_data})
    plt.savefig(plt_name + '.eps')
    # plt.savefig(os.path.join(run_folder, plt_name + .'pdf'))
    # plt.savefig(os.path.join(run_folder, plt_name + 'svg'))
    
    
def plot_data_probing(training_or_testing_sets, data_plots, ai_strategy, 
                      n_beams, runs_folder, scen_idx, norm_type, 
                      x_train, y_train, x_val, y_val, x_test, y_test):

    label_range = np.arange(n_beams) + 1 # 1-64    
    
    for chosen_set in training_or_testing_sets:

        if chosen_set == 'train':
            x_set = x_train
            y_set = y_train
        elif chosen_set == 'val':
            if ai_strategy in ['KNN', 'LT']:
                continue
            x_set = x_val
            y_set = y_val
        elif chosen_set == 'test':
            x_set = x_test
            y_set = y_test
        elif chosen_set == 'full':
            if ai_strategy in ['KNN', 'LT']:
                x_set = np.vstack((x_train, x_test))
                y_set = np.concatenate((y_train, y_test))
            else:
                x_set = np.vstack((x_train, x_val, x_test))
                y_set = np.concatenate((y_train, y_val, y_test))
        else:
            raise Exception(f"'{chosen_set}' is not recognized "
                             "as a possible set.")
            
        
        if 'position_color_best_beam' in data_plots:
            # Plot Normalized position and respective beam color
            fig = plt.figure(figsize=(6,4))
            
            plt.scatter(x_set[:,0], x_set[:,1], vmin=1, vmax=n_beams,
                        c=y_set, s=13, edgecolor='black', linewidth=0.3, 
                        cmap=plt.cm.jet)
            
            m = 2e-2 # margin
            plt.xlabel('$X_1$ Normalized')
            plt.ylabel('$X_2$ Normalized')
            plt.title(f'Scenario {scen_idx} - Normalized position {chosen_set} set')
            plt.xlim([0-m,1+m])
            plt.ylim([0-m,1+m])
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Beam index', rotation=270, labelpad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(runs_folder, 
                                     f'norm_position_{chosen_set}_set.pdf'))


        # PLOT NORMALIZED DISTANCE AND OFFSETTED ANGLE
        if norm_type == 5 and 'position_color_best_beam_polar' in data_plots:
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(polar=True)
            #           # angle                distance
            ax.scatter(x_set[:,1] * np.pi, x_set[:,0], c=y_set, 
                       s=10, cmap='jet', edgecolor='black', linewidth=0.5)
            
            ax.set_xticks(np.pi/180 * np.linspace(0, 360, 12, endpoint=False))
            ax.grid(True)
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            plt.tight_layout()
            plt.savefig(os.path.join(runs_folder, 
                         f'norm5_final_scenario_{scen_idx}.pdf'))
            
        
        if 'beam_freq_histogram' in data_plots:
            # Plot Histogram of beam frequency
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.hist(y_set, bins=label_range, edgecolor='black', linewidth=1)
            for rect in ax.patches:
                height = rect.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x()+rect.get_width()/2, height), 
                            xytext=(0, 5), textcoords='offset points', 
                            ha='center', va='bottom') 
            
            ax.set_xticks(label_range[::3])
            plt.title(f'Scenario {scen_idx} - Beam frequency in {chosen_set} set')
            plt.xlabel('Beam index')
            plt.ylabel('Frequency')
            fig.tight_layout()
            plt.savefig(os.path.join(runs_folder, 
                                     f'beam_freq_{chosen_set}_set.pdf'))
    