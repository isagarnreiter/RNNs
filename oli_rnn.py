# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:49:12 2020

@author: Isabelle
"""

from oli_task_modif import PerceptualDiscrimination
from psychrnn.backend.models.basic import Basic

import tensorflow as tf
from matplotlib import pyplot as plt

import numpy as np
import random

from scipy.stats import lognorm
from scipy.stats import norm

from fcts import initialise_params
from fcts import plot_weights
from fcts import initialise_connectivity


%matplotlib inline
seed=2020

tf.compat.v2.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

#%%

# ---------------------- Set up a basic model ---------------------------

params = initialise_params('dale_network', 50)

pd = PerceptualDiscrimination(dt = params['dt'],
                              tau = params['tau'], 
                              T = params['T'], 
                              N_batch = params['N_batch']) # Initialize the task object

dale_network_params = pd.get_task_params() # get the params passed in and defined in pd
dale_network_params['N_rec'] = params['N_rec'] # set the number of recurrent units in the model
dale_network_params['name'] = params['Name']
dale_network_params['N_in'] = params['N_in']
dale_network_params['N_out'] = params['N_out']
dale_network_params['dale_ratio'] =params['dale_ratio']

#=============================================================================
#build array depicting connections between 2 seperate excitatory + inhibitory domains. 
#domains receive input from either of the 2 channels
#10% of connexions are weighted randomly between domains

in_connect, rec_connect, out_connect, bv = initialise_connectivity(params, 
                                                               N_colossal = 20, 
                                                               N_exc_no_in = 20, 
                                                               N_inh_no_in = 5,
                                                               Input_to_colossal = True)

dale_network_params['input_connectivity'] = in_connect
dale_network_params['rec_connectivity'] = rec_connect
dale_network_params['output_connectivity'] = out_connect


daleModel = Basic(dale_network_params)

#%%

train_params = {}
train_params['save_weights_path'] =  None # Where to save the model after training. Default: None
train_params['training_iters'] = 200000 # number of iterations to train for Default: 50000
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
trial_nb = 3
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
ax2.set_ylim(-0.1,1.1)

ax3 = plt.subplot(224)
ax3.plot(range(0, len(x[0,:,:])*dt,dt), model_output[trial_nb,:,:])
ax3.set_xlabel("Time (ms)", fontsize = 16)
ax3.set_title("Output", fontsize = 16)
ax3.set_ylim(-0.1,1.1)

ax4 = plt.subplot(223)
ax4.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,:])
ax4.set_xlabel("Time (ms)", fontsize = 16)
ax4.set_title("State of each neuron", fontsize = 16)
ax4.set_ylim(-0.5,0.5)

fig2.tight_layout()

#%%

#compare states of different neural populations
fig3 = plt.figure()
ax1 = plt.subplot(211)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,80:90], c = 'blue', alpha=0.6)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,0:30], c = 'red', alpha=0.6)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,30:40], c = 'black', alpha=0.6)
ax1.set_ylim(-0.5,0.5)
ax1.set_title("State of each neuron in H1", fontsize = 10)

ax2 = plt.subplot(212)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,40:70], c='red', alpha=0.6)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,90:100], c='blue', alpha=0.6)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,70:80], c='black', alpha=0.6)
ax2.set_xlabel("Time (ms)", fontsize = 10)
ax2.set_ylim(-0.5,0.5)
ax2.set_title("State of each neuron in H2", fontsize = 10)

plt.tight_layout()


#%%
        
bins = pd.psychometric_curve(y, mask, train_params,9)

plt.plot(bins)
plt.xticks(ticks = np.linspace(0, 8, 9), labels=np.linspace(-1, 1, 9))

#%%
# ---------------------- Save and plot the weights of the network ---------------------------
 
weights = daleModel.get_weights()

plot_weights(weights['W_rec'],  
            xlabel = 'From', 
            ylabel = 'To')

plot_weights(weights['W_in'])
plot_weights(weights['W_out'])

#daleModel.save("weights/saved_weights_new_task_sparse_connectivity_5")

#%%
#trying to fit a lognormal curve to the distriubtion of weights

weight_distrib = np.concatenate(weights['W_rec'][:, 0:nb_excn*2])
weight_distrib = [i for i in weight_distrib if i != 0.0]

stdev = np.std(weight_distrib)
mean = np.mean(weight_distrib)


#just a gaussian fit
fig = plt.figure()
ax = plt.subplot(111)
from scipy.stats import gaussian_kde
density = gaussian_kde(weight_distrib)
xs = np.linspace(0,0.6,150)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax.plot(xs,density(xs))
ax.hist(weight_distrib, bins = 10, density = True)

#the lognormal distrib fitted to the data but does not work
shape, loc, scale = lognorm.fit(weight_distrib, floc = -1)
estimated_mu = np.log(scale)
estimated_sigma = shape

plt.hist(weight_distrib, bins=50, density=True)
xmin = np.min(weight_distrib)
xmax = np.max(weight_distrib)
x = np.linspace(xmin, xmax, 200)
pdf = lognorm.pdf(x, 1.8, scale = estimated_mu)
plt.plot(x, pdf, 'k')
plt.show()

#normal distribution of the logarithms of the weights

log_weights = np.log(weight_distrib)

lmean = np.mean(log_weights)
lstd = np.std(log_weights)
x = np.linspace(-25, 0, 100)
y = norm.pdf(x,lmean,lstd)

plt.hist(log_weights, bins=50, alpha = 0.75, density=True)
plt.plot(x,y, 'k', color='coral')

#%%
daleModel.destruct()


