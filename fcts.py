# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:13:34 2021

@author: Isabelle
"""
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import os

def plot_weights(weights, plot = ""):
    cmap = plt.set_cmap('RdBu_r')
 
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2,  width_ratios=(10,1), height_ratios=(10,1),
                          left=0.2, right=0.8, bottom=0.2, top=0.8,
                          wspace=0.05, hspace=0.05)
    
    w_rec = fig.add_subplot(gs[0, 0])
    w_in = fig.add_subplot(gs[0, 1], sharey=w_rec)
    w_out = fig.add_subplot(gs[1, 0])

    w_rec.matshow(weights['W_rec'], norm=Normalize(vmin=-.5, vmax=.5))
    w_in.matshow(weights['W_in'], norm=Normalize(vmin=-.5, vmax=.5), aspect='auto')
    w_out.matshow(weights['W_out'], norm=Normalize(vmin=-.5, vmax=.5), aspect='auto')

    w_rec.set_yticks([0, 20, 40, 60, 80])
    w_rec.set_xticks([0, 20, 40, 60, 80])
    
    w_in.set_xticks([0,1,2])
    w_in.set_xticklabels(['1','2','G'])
    w_out.set_yticks([0,1])
    w_out.set_yticklabels(['O1','O2'])
    
    w_out.tick_params(top = False, labeltop = False, bottom=False)
    w_in.tick_params(left = False, labelleft = False, bottom=False)
    w_rec.tick_params(bottom=False)
    
    w_rec.set_title('From', fontsize=12)
    w_rec.set_ylabel('To', fontsize=12)



def initialise_connectivity(params, N_callosal, P_in, P_rec, P_out):
    
    N_rec = params['N_rec']
    N_in = params['N_in']
    N_out = params['N_out']
    opto = params['opto']
    
    nb_excn = int(N_rec*0.4)
    nb_inhn = int(N_rec*0.1)

    #initialise sparse input connectivity depending on the defined probability of connectivity

    input_connectivity = np.zeros((N_rec * N_in))
    input_connectivity[0:int(P_in*N_rec*N_in)] = 1
    np.random.shuffle(input_connectivity)
    input_connectivity = input_connectivity.reshape(N_rec, N_in)
    
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


def gen_pol_trials(daleModel, sd, inputs, sim=False):
    
    "Creates a dictionary of trials with personalised parameters of input intensity given by 'inputs'"
    "This parameter is a list in the shape Nx2, where n is the number of trials to generate, each trial defined by the values taken by intensity_0 and intensity_1"
    "The function returns the input, expected output, mask, output and state of the network for each trial."
    
    trials = {}
    for i in range(len(inputs)):
        trials[f'1:{inputs[i][0]}_2:{inputs[i][1]}'] = {}
        params_single_trial = sd.generate_trial_params(0,0)
        params_single_trial['intensity_0'] = inputs[i][0]
        params_single_trial['intensity_1'] = inputs[i][1]
    
        x, y, mask = sd.generate_trial(params_single_trial) #generate input and output
        x, y, mask = np.array([x]), np.array([y]), np.array([mask]) #add dimension to shape of x, y, mask to fit the test() function and the figure format
        if sim==False:
            model_output, model_state = daleModel.test(x) # run the model on input x
        elif sim==True:
            model_output, model_state = daleModel.run_trials(x)
        
        trials[f'1:{inputs[i][0]}_2:{inputs[i][1]}'] = {'x':x[0], 'y':y[0], 'model_output': model_output[0], 'model_state':model_state[0]}
        
    return trials

def count_pref(array1, array2, indices=False):
    
    "This function compares array1 and array2 and assesses at when an element in array1 is greater than that in array2"
    "the function can return the number of elements to be higher in array 1 or the indices of the elements in question"
    "The element also has to be non negative in array 1."
    
    list_of_indices = []
    for i in range(1,len(array1)):
        if array1[i] >= 0 and array1[i] > array2[i]:
            list_of_indices.append(i)
    n_pref = len(list_of_indices) 
    if indices==True:
        return list_of_indices
    
    else:
        return n_pref

def stim_pref(trials):
    "save the state of excitatory neurons right after stimulus fore either a stim to hemi 1 or 2"
    
    stim_pref_dict = {}
    key = list(trials.keys())
    stim_pref_dict['max_hem1stim'] = trials[key[0]]['model_state'][50,:]-trials[key[0]]['model_state'][0,:] 
    stim_pref_dict['max_hem2stim'] = trials[key[1]]['model_state'][50,:]-trials[key[1]]['model_state'][0,:] 
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
    "Adapts the weight matrix to trials with simulated optogenetic stimulation"
    "Adds a 4th input channel to the input weight matrix (W_in) and to the input connectivity matrix (input_connectivity)"
    "returns all the weights of the network"
    
    N_rec =  weights['W_in'].shape[0]
    a = np.zeros(N_rec)
    a = a.reshape(N_rec, 1)
    
    weights['input_connectivity'] = np.append(weights['input_connectivity'], a, axis=1)
    weights['W_in'] = np.append(weights['W_in'], a, axis=1)
    return weights


def change_opto_stim(weights, indices):
    "This function adds optogenetic stimulation to any input neuron. The target neurons are defined by the indeces parameter."
    "The input weights are set to 0.3"
    
    N_rec =  weights['W_in'].shape[0]
    a = np.zeros(N_rec)
    a[indices] = 1

    weights['input_connectivity'][:,3] = a
    weights['W_in'][:,3] = a*0.3
    return weights


def get_model_info(Path, Models, File):
    
    first_set = pd.DataFrame(columns = ['filename', 'P_in', 'P_rec', 'N_cal', 'seed', 'loss',
                                          'mean_hem1_ipsi', 'mean_hem1_contra', 'mean_hem2_ipsi', 'mean_hem2_contra',
                                          'var_hem1_ipsi', 'var_hem1_contra', 'var_hem2_ipsi', 'var_hem2_contra',
                                          'nb_hem1_ipsi_pref', 'nb_hem2_ipsi_pref', 'nb_hem1_contra_pref', 'nb_hem2_contra_pref',
                                          'total_active', 'fraction_ipsi_pref'])
                                        
    for item in os.listdir(Path+Models):
        
        dalemodel_test = dict(np.load(f'/UserFolder/neur0003/first_set_models/{item}', allow_pickle=True))
        if list(dalemodel_test.keys())[0] == 'arr_0':
            dalemodel_test = dalemodel_test['arr_0'].reshape(-1)[0]
            stim_pref = dalemodel_test['stim_pref'].reshape(-1)[0]
            
            params_conv = {0.0:0.08, 0.1:0.1, 0.2:0.25, 0.5:0.5, 0.7:0.75, 1.0:1.0}
            P_in = params_conv[round(float(item[13:15])*0.1, 2)]
            P_rec = params_conv[round(float(item[19:21])*0.1, 2)]
            N_cal = int(item[25:27])
            seed = int(item[29])
            
        else:
            trials = dalemodel_test['trials'].reshape(-1)[0]
            stim_pref = stim_pref(trials)
            params = dalemodel_test['params'].reshape(1)[0]  
            P_in = params['P_in']
            P_rec = params['P_rec']
            N_cal = params['N_cal']
            seed = params['seed']
        
        loss = dalemodel_test['losses'][-1]
        
    
        mean_hem1_ipsi = np.mean(stim_pref['1:0.0_2:0.6'][0:40])
        mean_hem1_contra = np.mean(stim_pref['1:0.6_2:0.0'][0:40])
        mean_hem2_ipsi = np.mean(stim_pref['1:0.6_2:0.0'][40:80])
        mean_hem2_contra = np.mean(stim_pref['1:0.0_2:0.6'][40:80])
        
        var_hem1_ipsi = np.std(stim_pref['1:0.0_2:0.6'][0:40])
        var_hem1_contra = np.std(stim_pref['1:0.6_2:0.0'][0:40])
        var_hem2_ipsi = np.std(stim_pref['1:0.6_2:0.0'][40:80])
        var_hem2_contra = np.std(stim_pref['1:0.0_2:0.6'][40:80])
        
        nb_hem1_ipsi_pref = count_pref(stim_pref['1:0.0_2:0.6'][0:40], stim_pref['1:0.6_2:0.0'][0:40], indices=False)
        nb_hem2_ipsi_pref = count_pref(stim_pref['1:0.6_2:0.0'][40:80], stim_pref['1:0.0_2:0.6'][40:80], indices=False)
        nb_hem1_contra_pref = count_pref(stim_pref['1:0.6_2:0.0'][0:40], stim_pref['1:0.0_2:0.6'][0:40], indices=False)
        nb_hem2_contra_pref = count_pref(stim_pref['1:0.0_2:0.6'][40:80], stim_pref['1:0.6_2:0.0'][40:80], indices=False)
    
        total_active = nb_hem1_ipsi_pref + nb_hem2_ipsi_pref + nb_hem1_contra_pref + nb_hem2_contra_pref
        fraction_ipsi_pref = (nb_hem1_ipsi_pref+nb_hem2_ipsi_pref)/total_active
        
        new_row = {'filename':item, 'P_in':P_in, 'P_rec':P_rec, 'N_cal':N_cal, 'seed':seed, 'loss': loss,
                    'mean_hem1_ipsi':mean_hem1_ipsi, 'mean_hem1_contra':mean_hem1_contra, 'mean_hem2_ipsi':mean_hem2_ipsi, 'mean_hem2_contra':mean_hem2_contra,
                    'var_hem1_ipsi':var_hem1_ipsi, 'var_hem1_contra':var_hem1_contra, 'var_hem2_ipsi':var_hem2_ipsi, 'var_hem2_contra':var_hem2_contra,
                    'nb_hem1_ipsi_pref':nb_hem1_ipsi_pref, 'nb_hem2_ipsi_pref':nb_hem2_ipsi_pref, 'nb_hem1_contra_pref':nb_hem1_contra_pref, 'nb_hem2_contra_pref':nb_hem2_contra_pref, 
                    'total_active':total_active, 'fraction_ipsi_pref':fraction_ipsi_pref}
                   
        first_set = first_set.append(new_row, ignore_index = True)
    
    first_set.to_pickle(Path+File+'.pkl')

