# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:13:34 2021

@author: Isabelle
"""
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def plot_weights(weights, xlabel="", ylabel="", cb = False, matrix = ''):
    cmap = plt.set_cmap('RdBu_r')
    if matrix == 'in':
        f = plt.figure(figsize=(2,6))
        ax1 = plt.subplot(111)
        ax1.matshow(weights, norm=Normalize(vmin=-.5, vmax=.5))
        ax1.set_yticks([0, 20, 40, 60, 80])
        ax1.set_xticks([0,1,2])
        ax1.set_xticklabels([1,2,3])
    if matrix == 'out':
        f = plt.figure(figsize=(6,2))
        ax1 = plt.subplot(111)
        ax1.matshow(weights, norm=Normalize(vmin=-.5, vmax=.5))
        ax1.set_xticks([0, 20, 40, 60, 80])
        ax1.set_yticks([0,1])
        ax1.set_yticklabels([1,2])
    if matrix == 'rec':
        f = plt.figure(figsize=(6,6))
        ax1 = plt.subplot(111)
        ax1.matshow(weights, norm=Normalize(vmin=-.5, vmax=.5))
        ax1.set_xticks([0, 20, 40, 60, 80])
        ax1.set_xticks([0, 20, 40, 60, 80])
    if cb==True:
        plt.colorbar(ax1.matshow(weights, norm=Normalize(vmin=-.5, vmax=.5)))
    ax1.set_title(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    
    
    
    
def initialise_params(name, N_batch):
    params = {}
    params['dt'] = 10 # The simulation timestep.
    params['tau'] = 100 # The intrinsic time constant of neural state decay.
    params['T'] = 2500 # The trial length.
    params['N_rec'] = 100 # The number of recurrent units in the network.
    params['N_in'] = 3
    params['N_out'] = 2
    params['Name'] = name
    params['N_batch'] = N_batch   
    params['dale_ratio'] = .8
    
    return params


def initialise_connectivity(params, N_callosal, P_in, P_rec, P_out):
    
    N_rec = params['N_rec']
    N_in = params['N_in']
    N_out = params['N_out']
    
    nb_excn = int(N_rec*0.4)
    nb_inhn = int(N_rec*0.1)

    #initialise sparse input connectivity depending on the defined probability of connectivity

    input_connectivity = np.zeros((N_rec * N_in))
    input_connectivity[0:int(P_in*N_rec*N_in)] = 1
    np.random.shuffle(input_connectivity)
    input_connectivity = input_connectivity.reshape(N_rec, N_in)
    if N_in==4:
        input_connectivity[:, 3] = np.ones(N_rec)
    
    rec_connectivity = np.zeros((N_rec * N_rec))
    rec_connectivity[0:int(P_rec*N_rec*N_rec)] = 1
    np.random.shuffle(rec_connectivity)
    rec_connectivity = rec_connectivity.reshape(N_rec, N_rec)


    output_connectivity = np.zeros((N_out * N_rec))
    output_connectivity[0:int(P_out*N_out*N_rec)] = 1
    np.random.shuffle(output_connectivity)
    output_connectivity = output_connectivity.reshape(N_out, N_rec)
    
    
    #set basic connectivity with differences in the number of neurons with colossal projections
    
    output_connectivity[:, nb_excn*2:N_rec] = 0
    output_connectivity[0, 40:80] = 0
    output_connectivity[1, 0:40] = 0
    
    rec_connectivity[nb_excn:nb_excn*2,:nb_excn-N_callosal] = 0
    rec_connectivity[:nb_excn,nb_excn:nb_excn*2-N_callosal] = 0
    rec_connectivity[N_rec-nb_inhn:N_rec, :nb_excn-N_callosal] = 0
    rec_connectivity[nb_excn*2:N_rec-nb_inhn, nb_excn:nb_excn*2-N_callosal] = 0
    rec_connectivity[nb_excn:nb_excn*2, nb_excn*2:N_rec-nb_inhn] = 0
    rec_connectivity[:nb_excn, N_rec-nb_inhn:N_rec] = 0
    rec_connectivity[nb_excn*2:N_rec-nb_inhn, N_rec-nb_inhn:N_rec] = 0
    rec_connectivity[N_rec-nb_inhn:N_rec, nb_excn*2:N_rec-nb_inhn] = 0
    
    input_connectivity[0:nb_excn, 1] = 0
    input_connectivity[nb_excn:nb_excn*2, 0] = 0
    input_connectivity[nb_excn*2:N_rec-nb_inhn, 1] = 0
    input_connectivity[N_rec-nb_inhn:N_rec, 0] = 0    
    
    
    return input_connectivity, rec_connectivity, output_connectivity


def gen_pol_trials(daleModel, pd, j, k):
    trials = {}
    
    for i in range(len(j)):
        trials[f'hem{k[i]}stim'] = {}
        params_single_trial = {'intensity_0': j[i][0], 
                                'intensity_1': j[i][1], 
                                'random_output': 1, 
                                'stim_noise': 0.1, 
                                'onset_time': 0, 
                                'stim_duration': 500, 
                                'go_cue_onset': 1500, 
                                'go_cue_duration': 25.0, 
                                'post_go_cue': 125.0,
                                'intensity_opto': 0.4,
                                'end_opto':500}
    
        x, y, mask = pd.generate_trial(params_single_trial) #generate input and output
        x, y, mask = np.array([x]), np.array([y]), np.array([mask]) #add dimension to shape of x, y, mask to fit the test() function and the figure format

        model_output, model_state = daleModel.test(x) # run the model on input x
        
        trials[f'hem{k[i]}stim']['x'] = x[0]
        trials[f'hem{k[i]}stim']['y'] = y[0]
        trials[f'hem{k[i]}stim']['mask'] = mask[0]
        trials[f'hem{k[i]}stim']['model_output'] = model_output[0]
        trials[f'hem{k[i]}stim']['model_state'] = model_state[0]
        
    return trials

def count_pref(array1, array2, indices=False):
    list_of_indices = []
    for i in range(1,len(array1)):
        if array1[i] >= 0 and array1[i] > array2[i]:
            list_of_indices.append(i)
    n_pref = len(list_of_indices) 
    if indices==True:
        return list_of_indices
    
    else:
        return n_pref


def stim_pref_(trials):
    stim_pref_dict = {}
    
    stim_pref_dict['max_hem1stim'] = trials['hem1stim']['model_state'][50,:]-trials['hem1stim']['model_state'][0,:] #save the state of excitatory neurons right after stimulus fore either a stim to hemi 1 or 2
    stim_pref_dict['max_hem2stim'] = trials['hem2stim']['model_state'][50,:]-trials['hem2stim']['model_state'][0,:] #save the state of excitatory neurons right after stimulus fore either a stim to hemi 1 or 2

    return stim_pref_dict

def get_average(trials):
    stim_pref_dict = stim_pref(trials)
    n = [0,40,80]
    hem = [1,2,1]
    average_trajectory  = {}
    
    for i in range(2):
        for j in range(2):
            indices = count_pref(stim_pref_dict[f'max_hem{hem[j]}stim'][n[i]:n[i+1]], stim_pref_dict[f'max_hem{hem[j+1]}stim'][n[i]:n[i+1]], indices=True)
            target = trials[f'hem{hem[j]}stim']['model_state'][:, n[i]:n[i+1]]
            average_trajectory[f'hem{i+1}_hem{hem[j]}stim'] = np.mean(target[:, indices], axis=1)
    
    return average_trajectory


def adapt_for_opto(weights):
    N_rec =  weights['W_in'].shape[0]
    a = np.zeros(N_rec)
    a = a.reshape(N_rec, 1)
    
    weights['input_connectivity'] = np.append(weights['input_connectivity'], a, axis=1)
    weights['W_in'] = np.append(weights['W_in'], a, axis=1)
    return weights

def change_opto_stim(weights, indices):
    N_rec =  weights['W_in'].shape[0]
    a = np.zeros(N_rec)
    a[indices] = 1

    weights['input_connectivity'][:,3] = a
    weights['W_in'][:,3] = a*0.3
    return weights