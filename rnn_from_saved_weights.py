# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:47:02 2020

@author: Isabelle
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from weights_fct import plot_weights
from psychrnn.backend.models.basic import Basic
import tensorflow as tf0
from oli_task import PerceptualDiscrimination

#%%

weights = dict(np.load('./weights/saved_weights_1_2_output.npz', allow_pickle = True))
# =============================================================================
# changes to be made to the weights
# =============================================================================
weights['W_in'][:,1] = 0

# weights['W_rec'][40:80, 20:40] = 0
# weights['W_rec'][90:100, 20:40] = 0

np.savez('./weights/modified_saved_weights.npz', **weights)

#%%
dt = 10 # The simulation timestep.
tau = 100 # The intrinsic time constant of neural state decay.
T = 2500 # The trial length.
N_batch = 50 # The number of trials per training update.
N_rec = 100 # The number of recurrent units in the network.
N_in = 3
N_out = 2
name = 'modif_model' #  Unique name used to determine variable scope for internal use.

pd = PerceptualDiscrimination(dt = dt, tau = tau, T = T, N_batch = N_batch) # Initialize the task object
modif_network_params = pd.get_task_params() # get the params passed in and defined in pd
modif_network_params['N_rec'] = N_rec # set the number of recurrent units in the model
modif_network_params['name'] = name
modif_network_params['N_in'] = N_in
modif_network_params['N_out'] = N_out
modif_network_params['dale_ratio'] = .8


modif_network_params['load_weights_path'] = './weights/modified_saved_weights.npz'

fileModel = Basic(modif_network_params)

#%%
weights = fileModel.get_weights()
plot_weights(weights['W_rec'], xlabel = 'From', ylabel = 'To')
plot_weights(weights['W_in'])

#%%
x, y,mask, train_params = pd.get_trial_batch() # get pd task inputs and outputs
model_output, model_state = fileModel.test(x) # run the model on input x

Accuracy = pd.accuracy_function(y, model_output, mask)

#%%
trial_nb = 7
for i in range(len(mask[trial_nb])):
    if mask[trial_nb][i][0] == 0:
        y[trial_nb][i] =+ np.nan

fig2 = plt.figure(figsize=(20,8))

ax1 = plt.subplot(221)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), x[trial_nb,:,:])
ax1.set_title("Input", fontsize = 16)
ax1.legend(["Input Channel 1", "Input Channel 2", 'go cue'])

ax2 = plt.subplot(222)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), y[trial_nb,:,:])
ax2.set_title("Target output", fontsize = 16)

ax3 = plt.subplot(224)
ax3.plot(range(0, len(x[0,:,:])*dt,dt), model_output[trial_nb,:,:])
ax3.set_xlabel("Time (ms)", fontsize = 16)
ax3.set_title("Output", fontsize = 16)

ax4 = plt.subplot(223)
ax4.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,:])
ax4.set_xlabel("Time (ms)", fontsize = 16)
ax4.set_title("State of each neuron", fontsize = 16)

fig2.tight_layout()

#%%
fig3 = plt.figure()
ax1 = plt.subplot(211)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,80:90], c = 'blue', alpha=0.6)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,0:30], c = 'red', alpha=0.6)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,30:40], c = 'black', alpha=0.6)
ax1.set_title("State of each neuron in H1", fontsize = 10)
ax1.set_ylim(-1, 1.5)

ax2 = plt.subplot(212)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,40:70], c='red', alpha=0.6)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,90:100], c='blue', alpha=0.6)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,70:80], c='black', alpha=0.6)
ax2.set_xlabel("Time (ms)", fontsize = 10)
ax2.set_title("State of each neuron in H2", fontsize = 10)
ax2.set_ylim(-1, 1.5)

plt.tight_layout()

#%%
fileModel.destruct()
