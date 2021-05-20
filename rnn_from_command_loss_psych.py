# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:49:12 2020

@author: Isabelle
"""

from oli_task import PerceptualDiscrimination
from psychrnn.backend.models.basic import Basic
import tensorflow as tf
import numpy as np
import random
import fcts
import sys

for i in range(8):

    P_in = 0.50
    P_rec = 0.5
    N_callosal = 20
    seed = i
    
    params = {'P_in':P_in, 'P_rec':P_rec, 'N_cal':N_callosal, 'seed':seed}
    
    tf.compat.v2.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # ---------------------- Set up a basic model ---------------------------
    
    pd = PerceptualDiscrimination(dt = 10, # The simulation timestep
                                  tau = 100, # The intrinsic time constant of neural state decay.
                                  T = 2500, # The trial length, 
                                  N_batch = 50,
                                  N_in = 3,
                                  N_rec = 100,
                                  N_out = 2) # Initialize the task object
    
    dale_network_params = pd.get_task_params() # get the params passed in and defined in pd
    dale_network_params['name'] = 'dale_network_'
    dale_network_params['N_rec'] = 100 # set the number of recurrent units in the model
    dale_network_params['N_in'] = 3 # number of input channels
    dale_network_params['N_out'] = 2 # number of output channels
    dale_network_params['dale_ratio'] = 0.8 # ratio of excitatory to inhibitory neurons
    
    # define connectivity of the network
    # the initial achitecture is predefined in the function initialise_connectivity.
    # the probability of connection can be defined for the input matrix, the recurrent matrix and the output matrix
    # N_callosal corresponds to the number of neurons with callosal projections.
    
    P_out = 1
    in_connect, rec_connect, out_connect, = fcts.initialise_connectivity(dale_network_params, N_callosal, P_in, P_rec, P_out)
    
    dale_network_params['input_connectivity'] = in_connect
    dale_network_params['rec_connectivity'] = rec_connect
    dale_network_params['output_connectivity'] = out_connect
    
    daleModel = Basic(dale_network_params)
    
    # Initiate training parameters ---------------------------
    train_params = {}
    train_params['training_iters'] = 100000 # number of iterations to train for Default: 50000
    train_params['verbosity'] = True # If true, prints information as training progresses. Default: True
    
    # Train a basic model ---------------------------
    losses, initialTime, trainTime = daleModel.train(pd, train_params)
    
    weights = daleModel.get_weights() # Fetch the weights of the network 
    
    daleModel.destruct()
    
    # Test the trained model and save the test batch in a dictionary ---------------------------
    pd = PerceptualDiscrimination(dt = 10, # The simulation timestep
                                  tau = 100, # The intrinsic time constant of neural state decay.
                                  T = 2500, # The trial length, 
                                  N_batch = 200,
                                  N_in = 3,
                                  N_rec = 100,
                                  N_out = 2)
    
    dale_2 = dale_network_params.copy()
    dale_2['N_batch'] = 200
    dale_2['rec_noise'] = 0.015
    dale_2['name'] = 'dale_2'
    dale_network_params.update(weights)
    
    dale_2_Model = Basic(dale_2)
    
    x, y,mask, train_params = pd.get_trial_batch() # get pd task inputs and outputs
    model_output, model_state = dale_2_Model.test(x) # run the model on input 

    bin_means, bins, frac_choice = pd.psychometric_curve(model_output, train_params)
    psychometric = np.array([bin_means, bins, frac_choice], dtype='object')
    
        # Save the loss, stimulus preference and weights of the network in npz file ---------------------------
    np.savez(f'models/model_loss_psychometric_{i}',  
             losses=losses, 
             weights=weights, 
             psychometric=psychometric)
    
    print(f'model saved as model_loss_psychometric_{i}')
    
    dale_2_Model.destruct()
    
    
    
    
