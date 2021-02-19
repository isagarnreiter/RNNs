# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:13:34 2021

@author: Isabelle
"""
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def plot_weights(weights, title="", xlabel= "", ylabel=""):
    cmap = plt.set_cmap('RdBu_r')
    img = plt.matshow(weights, norm=Normalize(vmin=-.5, vmax=.5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    
    
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

def initialise_connectivity(params, N_colossal, N_exc_no_in, N_inh_no_in, Input_to_colossal):
    N_rec = params['N_rec']
    N_in = params['N_in']
    N_out = params['N_out']
    
    nb_excn = int(N_rec*0.4)
    nb_inhn = int(N_rec*0.1)

    input_connectivity = np.ones((N_rec, N_in))
    rec_connectivity = np.ones((N_rec, N_rec))
    output_connectivity = np.ones((N_out, N_rec))
    
    output_connectivity[:, nb_excn*2:N_rec] = 0
    
    rec_connectivity[nb_excn:nb_excn*2,:nb_excn-N_colossal] = 0
    rec_connectivity[:nb_excn,nb_excn:nb_excn*2-N_colossal] = 0
    rec_connectivity[N_rec-nb_inhn:N_rec, :nb_excn-N_colossal] = 0
    rec_connectivity[nb_excn*2:N_rec-nb_inhn, nb_excn:nb_excn*2-N_colossal] = 0
    rec_connectivity[nb_excn:nb_excn*2, nb_excn*2:N_rec-nb_inhn] = 0
    rec_connectivity[:nb_excn, N_rec-nb_inhn:N_rec] = 0
    rec_connectivity[nb_excn*2:N_rec-nb_inhn, N_rec-nb_inhn:N_rec] = 0
    rec_connectivity[N_rec-nb_inhn:N_rec, nb_excn*2:N_rec-nb_inhn] = 0
    
    input_connectivity[0:nb_excn, 1] = 0
    input_connectivity[nb_excn:nb_excn*2, 0] = 0
    input_connectivity[nb_excn*2:N_rec-nb_inhn, 1] = 0
    input_connectivity[N_rec-nb_inhn:N_rec, 0] = 0

    #initialise sparse input connectivity
    
    if Input_to_colossal==False:
        nb_excn = nb_excn - N_colossal
        
    input_sparseness_excn = np.ones((nb_excn))
    input_sparseness_inhn = np.ones((nb_inhn))
    
    input_sparseness_excn[0:N_exc_no_in]=0
    input_sparseness_inhn[0:N_inh_no_in]=0
    
    np.random.shuffle(input_sparseness_excn)
    np.random.shuffle(input_sparseness_inhn)
    
    input_connectivity[0:40, 0] = input_sparseness_excn
    input_connectivity[80:90, 0] = input_sparseness_inhn
    input_connectivity[90:100, 1] = input_sparseness_inhn
    input_connectivity[40:80, 1] = input_sparseness_excn
    
    
    return input_connectivity, rec_connectivity, output_connectivity