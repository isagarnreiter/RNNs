# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:49:12 2020

@author: Isabelle
"""

from oli_task import PerceptualDiscrimination
from psychrnn.backend.models.basic import Basic

import tensorflow as tf
from matplotlib import pyplot as plt

import numpy as np
import random
from matplotlib.colors import Normalize
%matplotlib inline
seed=2020

tf.compat.v2.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

#%%

# ---------------------- Set up a basic model ---------------------------
dt = 10 # The simulation timestep.
tau = 100 # The intrinsic time constant of neural state decay.
T = 2500 # The trial length.
N_batch = 50 # The number of trials per training update.
N_rec = 100 # The number of recurrent units in the network.
N_in = 3
N_out = 2
name = 'dale_model_' #  Unique name used to determine variable scope for internal use.

pd = PerceptualDiscrimination(dt = dt, tau = tau, T = T, N_batch = N_batch) # Initialize the task object
dale_network_params = pd.get_task_params() # get the params passed in and defined in pd
dale_network_params['N_rec'] = N_rec # set the number of recurrent units in the model
dale_network_params['name'] = name
dale_network_params['N_in'] = N_in
dale_network_params['N_out'] = N_out

dale_network_params['dale_ratio'] = .8

#=============================================================================
#build array depicting connections between 2 seperate excitatory + inhibitory domains. 
#domains receive input from either of the 2 channels
#10% of connexions are weighted randomly between domains

nb_excn = int(N_rec*0.4)
nb_inhn = int(N_rec*0.1)

input_connectivity = np.ones((N_rec, N_in))
rec_connectivity = np.ones((N_rec, N_rec))
output_connectivity = np.ones((N_out, N_rec))

input_connectivity[0:nb_excn, 1] = 0
input_connectivity[nb_excn:nb_excn*2, 0] = 0
input_connectivity[nb_excn*2:N_rec-nb_inhn, 1] = 0
input_connectivity[N_rec-nb_inhn:N_rec, 0] = 0

pre_connect = np.zeros((nb_excn*(nb_excn+nb_inhn)))
nb_of_random_connections = int(0.05*(len(pre_connect)))
pre_connect[:nb_of_random_connections] = 1

np.random.shuffle(pre_connect)
rand_connect1 = np.array(np.split(pre_connect, indices_or_sections = nb_excn+nb_inhn))
np.random.shuffle(pre_connect)
rand_connect2 = np.array(np.split(pre_connect, indices_or_sections = nb_excn+nb_inhn))



rec_connectivity[nb_excn:nb_excn*2,:nb_excn] = rand_connect1[:nb_excn]
rec_connectivity[:nb_excn,nb_excn:nb_excn*2] = rand_connect2[:nb_excn]
rec_connectivity[N_rec-nb_inhn:N_rec, :nb_excn] = rand_connect1[nb_excn:]
rec_connectivity[nb_excn*2:N_rec-nb_inhn, nb_excn:nb_excn*2] = rand_connect2[nb_excn:]
rec_connectivity[nb_excn:nb_excn*2, nb_excn*2:N_rec-nb_inhn] = 0
rec_connectivity[:nb_excn, N_rec-nb_inhn:N_rec] = 0
rec_connectivity[nb_excn*2:N_rec-nb_inhn, N_rec-nb_inhn:N_rec] = 0
rec_connectivity[N_rec-nb_inhn:N_rec, nb_excn*2:N_rec-nb_inhn] = 0

dale_network_params['input_connectivity'] = input_connectivity
dale_network_params['rec_connectivity'] = rec_connectivity
dale_network_params['output_connectivity'] = output_connectivity
#=============================================================================

daleModel = Basic(dale_network_params)

#%%

train_params = {}
train_params['save_weights_path'] =  None # Where to save the model after training. Default: None
train_params['training_iters'] = 50000 # number of iterations to train for Default: 50000
train_params['learning_rate'] = .001 # Sets learning rate if use default optimizer Default: .001
train_params['loss_epoch'] = 10 # Compute and record loss every 'loss_epoch' epochs. Default: 10
train_params['verbosity'] = True # If true, prints information as training progresses. Default: True
train_params['save_training_weights_epoch'] = 100 # save training weights every 'save_training_weights_epoch' epochs. Default: 100
train_params['training_weights_path'] = None # where to save training weights as training progresses. Default: None
train_params['optimizer'] = tf.compat.v1.train.AdamOptimizer(learning_rate=train_params['learning_rate']) # What optimizer to use to compute gradients. Default: tf.train.AdamOptimizer(learning_rate=train_params['learning_rate'])
train_params['clip_grads'] = True # If true, clip gradients by norm 1. Default: True

# ---------------------- Train a basic model ---------------------------
losses, initialTime, trainTime = daleModel.train(pd, train_params)

fig1= plt.figure()
plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Training Iteration")
plt.title("Loss During Training")

#%%

# ---------------------- Test the trained model ---------------------------
x, y,mask, train_params = pd.get_trial_batch() # get pd task inputs and outputs
model_output, model_state = daleModel.test(x) # run the model on input x


# ---------------------- Plot the results ---------------------------
trial_nb = 1
for i in range(len(mask[trial_nb])):
    if mask[trial_nb][i][0] == 0:
        y[trial_nb][i] =+ np.nan

fig2 = plt.figure(figsize=(20,8))

ax1 = plt.subplot(221)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), x[trial_nb,:,:])
ax1.set_title("Input", fontsize = 16)
ax1.legend(["Output Channel 1", "Output Channel 2", 'go cue'])

ax2 = plt.subplot(222)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), y[trial_nb,:,:])
ax2.set_title("Expected output", fontsize = 16)

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
# ---------------------- Save and plot the weights of the network ---------------------------

def plot_weights(weights, title="", xlabel= "", ylabel=""):
    cmap = plt.set_cmap('RdBu_r')
    img = plt.matshow(weights, norm=Normalize(vmin=-.5, vmax=.5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    
weights = daleModel.get_weights()
fig12 = plt.figure(figsize = (10,10))
ax1 = plt.subplot(111)
ax1 = plot_weights(weights['W_rec'],  
             xlabel = 'From', 
             ylabel = 'To')

#daleModel.save("weights/saved_weights_100")

plot_weights(weights['W_in'])
plot_weights(weights['W_out'])


#%%
length = len(train_params)
Acc
for i in range(length):        
    if train_params[i]["direction"] == 0:
        train_params[i]["coherence"] = -train_params[i]["coherence"]
    train_params[i]['choice'] = np.argmax(model_output[i][199])
    
    
coherence = [train_params[i]['coherence'] for i in range (length)]
choice = [train_params[i]['choice'] for i in range (length)]

fig4 = plt.figure()
plt.scatter(coherence, choice)

#%%

fig3 = plt.figure()
ax1 = plt.subplot(211)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,0:40])
ax1.set_xlabel("Time (ms)", fontsize = 16)
ax1.set_title("State of each neuron", fontsize = 16)

ax2 = plt.subplot(212)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,40:80])
ax2.set_xlabel("Time (ms)", fontsize = 16)
ax2.set_title("State of each neuron", fontsize = 16)


#%%
daleModel.destruct()


#%%

fig7 = plt.figure()

ax1 = plt.subplot(2,1,1)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), x[trial_nb,:,:])
ax1.set_title("Input")
ax1.legend(["Output Channel 1", "Output Channel 2", 'go cue'])

ax2 = plt.subplot(2,1,2)
ax2.plot(range(0, len(y[0,:,:])*dt,dt), y[trial_nb,:,:])
ax2.set_xlabel("Tme (ms)")
ax2.legend(["stimulation"])
fig7.tight_layout()

#%%
trial_nb = 7
fig8 = plt.figure(figsize=(10,4))
ax1 = plt.subplot(111)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), mask[trial_nb,:,:], c = 'grey')
ax1.set_title('Mask')



