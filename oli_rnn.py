# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:49:12 2020

@author: Isabelle
"""

from oli_task_perturb import PerceptualDiscrimination
from psychrnn.backend.models.basic import Basic

import tensorflow as tf
from matplotlib import pyplot as plt

import numpy as np
import random

from scipy.stats import lognorm
from scipy.stats import norm

import fcts
%matplotlib inline
seed=18

import pandas as pa
tf.compat.v2.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

#%%


# ---------------------- Set up a basic model ---------------------------

params = fcts.initialise_params('dale_network', 50)

pd = PerceptualDiscrimination(dt = params['dt'],
                              tau = params['tau'], 
                              T = params['T'], 
                              N_batch = params['N_batch'],
                              N_in=4,
                              N_rec=100,
                              N_out=2,
                              opto = True) # Initialize the task object


dale_network_params = pd.get_task_params() # get the params passed in and defined in pd
dale_network_params['N_rec'] = params['N_rec'] # set the number of recurrent units in the model
dale_network_params['name'] = params['Name']
dale_network_params['N_in'] = 4
dale_network_params['N_out'] = params['N_out']
dale_network_params['dale_ratio'] = 0.8
dale_network_params['rec_noise'] = 0.02

params['N_in'] = 4
#=============================================================================
#define connectivity of the network
#the initial achitecture is predefined in the function initialise_connectivity.
#the probability of connection can be defined for the input matrix, the recurrent matrix and the output matrix
#N_callosal corresponds to the number of neurons with callosal projections.

N_callosal = 20
P_in = 0.5
P_rec = 0.5
P_out = 1.0
in_connect, rec_connect, out_connect, = fcts.initialise_connectivity(params, N_callosal, P_in, P_rec, P_out)


dale_network_params['input_connectivity'] = in_connect
dale_network_params['rec_connectivity'] = rec_connect
dale_network_params['output_connectivity'] = out_connect


daleModel = Basic(dale_network_params)

#%%

train_params = {}
train_params['training_iters'] = 100000 # number of iterations to train for Default: 50000
train_params['verbosity'] = True # If true, prints information as training progresses. Default: True

# ---------------------- Train a basic model ---------------------------
losses, initialTime, trainTime = daleModel.train(pd, train_params)

fig1= plt.figure()
plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Training Iteration")
plt.xticks([0, 20, 40, 60, 80, 100], labels=[0, 20000, 40000, 60000, 80000, 100000])
plt.yticks([0, 0.05, 0.1, .15])
#%%

f = open("file.pkl","wb")
pickle.dump(Tloss,f)
f.close()

#%%
#plot of losses
Tloss_load = pa.read_pickle('file.pkl')

T_loss_array = np.array(Tloss_load.values())
T_loss_SEM = np.std(T_loss_array, axis=1)/np.sqrt(T_loss_array.shape[1])
T_loss_mean = np.mean(T_loss_array, axis=1)

x=np.linspace()

plt.errorbar(x, T_loss_mean, yerr=T_loss_SEM, alpha=0.2)
plt.plot(x, T_loss_mean)

#%%
# ---------------------- Test the trained model ---------------------------
x, y,mask, train_params = pd.get_trial_batch() # get pd task inputs and outputs
model_output, model_state = daleModel.test(x) # run the model on input x

#%%

bin_means, bins, frac_choice = pd.psychometric_curve(model_output, train_params)

plt.plot(bins, bin_means,marker='o', label='choice 1')
plt.xlabel('coherence')
plt.legend()
plt.ylabel('%')
#%%
# ---------------------- Plot the results ---------------------------
trial_nb = 4
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
        ax[i,j].plot(range(0, len(zipp[z][0,:,:])*dt,dt), zipp[z][trial_nb,:,3], linewidth=3, c='red')
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
#plot the relationship between reponse to stim 1 and stim2 for each neurons
unity_line = [-1, 0, 1]

figure = plt.figure(figsize=(6,6))
ax1 = plt.subplot(111)
ax1.scatter(max_hem1_hem1stim, max_hem1_hem2stim, c = 'coral', label = 'hemisphere 1', alpha=0.6)
ax1.scatter(max_hem2_hem1stim, max_hem2_hem2stim, c = 'green', label = 'hemisphere 2', alpha=0.6)
ax1.plot(unity_line, unity_line, c='black')
ax1.set_xlim(-1, 1)
ax1.set_xticks([-1,-0.5,0, 0.5,1])
ax1.set_ylim(-1,1)
ax1.set_yticks([-1,-0.5,0, 0.5,1])
ax1.set_title('states of excitatory neuron in hemisphere 1 and 2 at T = 500 ms')
ax1.legend()
ax1.set_xlabel('stim in hem 1')
ax1.set_ylabel('stim in hem 2')

#%%
# ---------------------- Save and plot the weights of the network ---------------------------

weights = daleModel.get_weights()
#%%
fcts.plot_weights(weights, plot='weights')
plt.colorbar(w_rec.matshow(weights['w_rec'], norm=Normalize(vmin=-.5, vmax=.5)))

daleModel.save("weights/model_example_write_up_partial_connectivity_5")

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

