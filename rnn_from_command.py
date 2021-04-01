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

# P_in = float(sys.argv[1])
# P_rec = float(sys.argv[2])
# N_callosal = int(sys.argv[3])
# seed = int(sys.argv[4])

P_in = 0.5
P_rec = 0.1
N_callosal = 10
seed = 1

tf.compat.v2.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ---------------------- Set up a basic model ---------------------------

pd = PerceptualDiscrimination(dt = 10, # The simulation timestep
                              tau = 100, # The intrinsic time constant of neural state decay.
                              T = 2500, # The trial length, 
                              N_batch = 50) # Initialize the task object

dale_network_params = pd.get_task_params() # get the params passed in and defined in pd
dale_network_params['name'] = 'dale_network'
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
train_params['training_iters'] = 2000 # number of iterations to train for Default: 50000
train_params['verbosity'] = True # If true, prints information as training progresses. Default: True

# Train a basic model ---------------------------
losses, initialTime, trainTime = daleModel.train(pd, train_params)

# Test the trained model and save the test batch in a dictionary ---------------------------
# only first 10 trials are saved because otherwise the file becomes too big
x, y,mask, train_params = pd.get_trial_batch() # get pd task inputs and outputs
model_output, model_state = daleModel.test(x) # run the model on input x
Test_batch = {'x':x[0:10], 'y':y[0:10], 'mask':mask[0:10], 'model_output':model_output[0:10], 'model_state':model_state[0:10]} 


stim_pref_dict = fcts.stim_pref(daleModel, pd) # Get the weights of the network
weights = daleModel.get_weights() # Fetch the weights of the network 


# Save the loss, stimulus preference and weights of the network in npz file ---------------------------
np.savez(f'IpsiContra_In{str(P_in)[0]+str(P_in)[2]}_Rec{str(P_rec)[0]+str(P_rec)[2]}_Col{N_callosal}_s{seed}',  
         losses=losses, 
         weights=weights, 
         stim_pref=stim_pref_dict,
         test_batch=Test_batch)

print(f'model saved under IpsiContra_IN{str(P_in)[0]+str(P_in)[2]}_REC{str(P_rec)[0]+str(P_rec)[2]}_Col{N_callosal}_s{seed}')


daleModel.destruct()




