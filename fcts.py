# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:13:34 2021

@author: Isabelle
"""
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def plot_weights(weights, plot = ""):
    cmap = plt.set_cmap('RdBu_r')
    
    if plot == 'weights':
        data = ['W_rec', 'W_in', 'W_out']
    if plot == 'connectivity':
        data = ['rec_connectivity', 'input_connectivity', 'output_connectivity']
        
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2,  width_ratios=(10,1), height_ratios=(10,1),
                          left=0.2, right=0.8, bottom=0.2, top=0.8,
                          wspace=0.05, hspace=0.05)
    
    w_rec = fig.add_subplot(gs[0, 0])
    w_in = fig.add_subplot(gs[0, 1], sharey=w_rec)
    w_out = fig.add_subplot(gs[1, 0])

    w_rec.matshow(weights[data[0]], norm=Normalize(vmin=-.5, vmax=.5))
    w_in.matshow(weights[data[1]], norm=Normalize(vmin=-.5, vmax=.5), aspect='auto')
    w_out.matshow(weights[data[2]], norm=Normalize(vmin=-.5, vmax=.5), aspect='auto')

    w_rec.set_yticks([0, 20, 40, 60, 80])
    w_rec.set_xticks([0, 20, 40, 60, 80])
    
    w_in.set_xticks([0,1,2])
    w_in.set_xticklabels(['I1','I2','G'])
    w_out.set_yticks([0,1])
    w_out.set_yticklabels(['O1','O2'])
    
    w_out.tick_params(top = False, labeltop = False, bottom=False)
    w_in.tick_params(left = False, labelleft = False, bottom=False)
    w_rec.tick_params(bottom=False)
    
    w_rec.set_title('From', fontsize=12)
    w_rec.set_ylabel('To', fontsize=12)
    
    
    
    
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
    input_connectivity[:, 3] = 0
    
    return input_connectivity, rec_connectivity, output_connectivity


def gen_pol_trials(daleModel, pd, j, opto_stim, sim=False):
    trials = {}
    k = np.array(range(len(j)))+1
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
                                'end_opto':500}
    
        x, y, mask = pd.generate_trial(params_single_trial) #generate input and output
        x, y, mask = np.array([x]), np.array([y]), np.array([mask]) #add dimension to shape of x, y, mask to fit the test() function and the figure format
        if sim==False:
            model_output, model_state = daleModel.test(x) # run the model on input x
        elif sim==True:
            model_output, model_state = daleModel.run_trials(x)
        
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
    key = list(trials.keys())
    stim_pref_dict['max_hem1stim'] = trials[key[1]]['model_state'][50,:]-trials[key[1]]['model_state'][0,:] #save the state of excitatory neurons right after stimulus fore either a stim to hemi 1 or 2
    stim_pref_dict['max_hem2stim'] = trials[key[0]]['model_state'][50,:]-trials[key[0]]['model_state'][0,:] #save the state of excitatory neurons right after stimulus fore either a stim to hemi 1 or 2

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