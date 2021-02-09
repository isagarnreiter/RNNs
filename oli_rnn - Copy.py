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
from weights_fct import plot_weights
from scipy.stats import lognorm
from scipy.stats import norm

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

#network with more precise connectivity. Make a difference between neurons receiving a direct imput

nb_excn = int(N_rec*0.4)
nb_inhn = int(N_rec*0.1)

input_connectivity = np.ones((N_rec, N_in))
rec_connectivity = np.ones((N_rec, N_rec))
output_connectivity = np.ones((N_out, N_rec))

input_connectivity[0:50, 1] = 0
input_connectivity[nb_excn:nb_excn*2, 0] = 0
input_connectivity[0:10, 0] = 0
input_connectivity[nb_excn*2:N_rec-nb_inhn, 1] = 0
input_connectivity[N_rec-nb_inhn:N_rec, 0] = 0

output_connectivity[:, nb_excn*2:N_rec] = 0


rec_connectivity[nb_excn:nb_excn*2,:nb_excn-10] = 0
rec_connectivity[:nb_excn,nb_excn:nb_excn*2-10] = 0
rec_connectivity[N_rec-nb_inhn:N_rec, :nb_excn-10] = 0
rec_connectivity[nb_excn*2:N_rec-nb_inhn, nb_excn:nb_excn*2-10] = 0
rec_connectivity[nb_excn:nb_excn*2, nb_excn*2:N_rec-nb_inhn] = 0
rec_connectivity[:nb_excn, N_rec-nb_inhn:N_rec] = 0
rec_connectivity[nb_excn*2:N_rec-nb_inhn, N_rec-nb_inhn:N_rec] = 0
rec_connectivity[N_rec-nb_inhn:N_rec, nb_excn*2:N_rec-nb_inhn] = 0
rec_connectivity[10:20, 30:40] = 0
rec_connectivity[10:20, 70:80] = 0
rec_connectivity[50:60, 30:40] = 0
rec_connectivity[50:60, 70:80] = 0


dale_network_params['input_connectivity'] = input_connectivity
dale_network_params['rec_connectivity'] = rec_connectivity
dale_network_params['output_connectivity'] = output_connectivity


daleModel = Basic(dale_network_params)

#%%

train_params = {}
train_params['save_weights_path'] =  None # Where to save the model after training. Default: None
train_params['training_iters'] = 100000 # number of iterations to train for Default: 50000
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

#%%
# ---------------------- Plot the results ---------------------------
trial_nb = 1
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

#compare states of different neural populations
fig3 = plt.figure()
ax1 = plt.subplot(211)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,80:90], c = 'blue', alpha=0.6)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,0:30], c = 'red', alpha=0.6)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,30:40], c = 'black', alpha=0.6)
ax1.set_title("State of each neuron in H1", fontsize = 10)

ax2 = plt.subplot(212)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,40:70], c='red', alpha=0.6)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,90:100], c='blue', alpha=0.6)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,70:80], c='black', alpha=0.6)
ax2.set_xlabel("Time (ms)", fontsize = 10)
ax2.set_title("State of each neuron in H2", fontsize = 10)

plt.tight_layout()


#%%
        
bins = pd.psychometric_curve(y, mask, train_params, 8)

plt.plot(bins)
plt.xticks(ticks = np.linspace(0, 8, 9), labels=np.linspace(-1, 1, 9))

#%%
# ---------------------- Save and plot the weights of the network ---------------------------
   
weights = daleModel.get_weights()
plot_weights(weights['W_rec'],  
             xlabel = 'From', 
             ylabel = 'To')

daleModel.save("weights/saved_weights_1")

plot_weights(weights['W_in'])
plot_weights(weights['W_out'])

#%%
#trying to fit a lognormal curve to the distriubtion of weights

weight_distrib = np.concatenate(weights['W_rec'][:, 0:nb_excn*2])
weight_distrib = [i for i in weight_distrib if i != 0.0]

stdev = np.std(weight_distrib)
mean = np.mean(weight_distrib)

fig = plt.figure()
ax = plt.subplot(111)
from scipy.stats import gaussian_kde
density = gaussian_kde(weight_distrib)
xs = np.linspace(0,0.6,150)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax.plot(xs,density(xs))
ax.hist(weight_distrib, bins = 10, density = True)

log_weights = np.log(weight_distrib)

shape, loc, scale = lognorm.fit(weight_distrib, floc = -1)
estimated_mu = np.log(scale)
estimated_sigma = shape

plt.hist(weight_distrib, bins=50, density=True)
xmin = np.min(weight_distrib)
xmax = np.max(weight_distrib)
x = np.linspace(xmin, xmax, 200)
pdf = lognorm.pdf(x, shape, scale = scale, loc = 0)
plt.plot(x, pdf, 'k')
plt.show()

#%%
daleModel.destruct()


