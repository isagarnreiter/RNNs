# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:49:12 2020

@author: Isabelle
"""

from psychrnn.backend.models.basic import Basic
from oli_task_perturb import SensoryDiscrimination
import tensorflow as tf
import numpy as np
import random
import fcts
import sys

# P_in = float(sys.argv[1])
# P_rec = float(sys.argv[2])
# N_callosal = int(sys.argv[3])
# seed = int(sys.argv[4])

P_in = 0.1
P_rec = 1.0
N_callosal = 30
seed = 5

params = {'P_in':P_in, 'P_rec':P_rec, 'N_cal':N_callosal, 'seed':seed}

tf.compat.v2.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ---------------------- Set up a basic model ---------------------------

sd = SensoryDiscrimination(dt = 10,
                              tau = 100, 
                              T = 2500, 
                              N_batch = 50,
                              N_rec=100,
                              N_out=2,
                              opto=0.0)# Initialize the task object

dale_network_params = sd.get_task_params() # get the params passed in and defined in sd
dale_network_params['name'] = 'dale_network'
dale_network_params['dale_ratio'] = 0.8 #define the ratio of excitatory to inhibitory neurons
dale_network_params['rec_noise'] = 0.02 #define the level of noise within the recurrent network

#=============================================================================
#define connectivity of the network
#the initial achitecture is predefined in the function initialise_connectivity.
#the probability of connection can be defined for the input matrix, the recurrent matrix and the output matrix
#N_callosal corresponds to the number of neurons with callosal projections.

P_out = 1.0
in_connect, rec_connect, out_connect, = fcts.initialise_connectivity(dale_network_params, N_callosal, P_in, P_rec, P_out)

dale_network_params['input_connectivity'] = in_connect
dale_network_params['rec_connectivity'] = rec_connect
dale_network_params['output_connectivity'] = out_connect


daleModel = Basic(dale_network_params)

# Initiate training parameters ---------------------------
train_params = {} # set up the training parameters
train_params['training_iters'] = 10000 # number of iterations to train for Default: 50000
train_params['verbosity'] = True # If true, prints information as training progresses. Default: True

#performance measure takes the loss at each epoch and if the loss reaches a certain threshold (defined by the performance_cutoff), the training stops
def performance_measure(trial_batch, trial_y, output_mask, output, epoch, losses, verbosity):
    if epoch%10==0:
        return 1-losses[-1]
    else:
        return 0.1
    
train_params['performance_measure'] = performance_measure
train_params['performance_cutoff'] = 1-.02

# Train a basic model ---------------------------
losses, initialTime, trainTime = daleModel.train(sd, train_params)

#generate 2 test trials for the network in which either input 1 or input 2 is 0.6 and input 1 or input 2 is 0.
trials = fcts.gen_pol_trials(daleModel, sd, [[0.6, 0.0], [0.0, 0.6]], [0,1])
weights = daleModel.get_weights() # Fetch the weights of the network 


#get the psychometric curve of the performance of the network on a batch of trials
x, y,mask, train_params = sd.get_trial_batch() # get pd task inputs and outputs
model_output, model_state = daleModel.test(x) # run the model on input 
psychometric = sd.psychometric_curve(model_output, train_params)


# Save the loss, stimulus preference and weights of the network in npz file ---------------------------
np.savez(f'IpsiContra_In{str(P_in)[0]+str(P_in)[2:4]}_Rec{str(P_rec)[0]+str(P_rec)[2:4]}_Cal{N_callosal}_s{seed}',  
         params=params,
         losses=losses, 
         weights=weights, 
         trials=trials,
         psychometric=psychometric)

print(f'model saved as IpsiContra_In{str(P_in)[0]+str(P_in)[2]}_Rec{str(P_rec)[0]+str(P_rec)[2]}_Cal{N_callosal}_s{seed}')


daleModel.destruct()




