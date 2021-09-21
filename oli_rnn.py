# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:49:12 2020

@author: Isabelle
"""
from oli_task_perturb import SensoryDiscrimination
from psychrnn.backend.models.basic import Basic
import tensorflow as tf
import fcts

from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd

from scipy.stats import lognorm, norm
from matplotlib.colors import Normalize

%matplotlib inline
seed=0

tf.compat.v2.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

#%%

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
# P_in, P_rec, P_out are the probability of connection for the input, recurrent and output matrix

N_callosal = 20
P_in = 0.4
P_rec = 0.4
P_out = 0.4
in_connect, rec_connect, out_connect, = fcts.initialise_connectivity(dale_network_params, N_callosal, P_in, P_rec, P_out)

dale_network_params['input_connectivity'] = in_connect
dale_network_params['rec_connectivity'] = rec_connect
dale_network_params['output_connectivity'] = out_connect


daleModel = Basic(dale_network_params)

#%%

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


# ---------------------- Train a basic model ---------------------------
losses, initialTime, trainTime = daleModel.train(sd, train_params)


# plot the losses of the network
fig1= plt.figure()
plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Training Iteration")
plt.xticks(np.linspace(0,int(train_params['training_iters']/100), 5), labels=np.linspace(0,int(train_params['training_iters']), 5))
plt.ylim(0, max(losses))

#%%
# ---------------------- Test the trained model ---------------------------
x, y,mask, train_params = sd.get_trial_batch() # get sd task inputs and outputs
model_output, model_state = daleModel.test(x) # run the model on input x

#%%
#plot the psychometric curve for the test batch

bin_means, bins = sd.psychometric_curve(model_output, train_params)

plt.plot(bins, bin_means,marker='o', label='choice 1')
plt.xlabel('difference input 1 - input 2')
plt.legend()
plt.ylabel('%')

#%%
# ---------------------- Plot the results ---------------------------
# determine which trial to plot
trial_nb = 4

# represent the lack of mask by an interruption in the expected output
for i in range(len(mask[trial_nb])):
    if mask[trial_nb][i][0] == 0:
        y[trial_nb][i] =+ np.nan

dt = 10

fig2, ax = plt.subplots(2,2,figsize=(20,8))

z=0
zipp = [x,y,model_state, model_output]
titles = ['Input', 'Target Output', 'States', 'Output']
for i in range(2):
    for j in range(2):
        ax[i,j].plot(range(0, len(zipp[z][trial_nb,:,:])*dt,dt), zipp[z][trial_nb,:,:], linewidth=3)
        ax[i,j].set_title(titles[z], fontsize=16)
        ax[i,j].set_ylim(-0.1,1.1)
        ax[i,j].set_yticks([0,1])
        ax[1,j].set_xlabel("Time (ms)", fontsize = 12)
        z+=1
        
ax[0,0].legend(["Input Channel 1", "Input Channel 2", 'go cue'])
ax[1,0].set_ylim(-0.8, 0.8)
ax[1,0].set_yticks([-0.5, 0, 0.5])

fig2.tight_layout()

#%%
# ---------------------- Save and plot the weights of the network ---------------------------

weights = daleModel.get_weights()
#%%
fcts.plot_weights(weights, plot='connectivity')
plt.colorbar(plt.matshow(weights['W_rec'], norm=Normalize(vmin=-.5, vmax=.5)))

daleModel.save("weights/model_example_write_up_partial_connectivity_5")


#%%
daleModel.destruct()

