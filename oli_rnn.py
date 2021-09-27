# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:49:12 2020

@author: Isabelle
"""
from sensory_discrimination_task import SensoryDiscrimination
from psychrnn.backend.models.basic import Basic
import tensorflow as tf
import fcts
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d 
from sklearn.preprocessing import StandardScaler

%matplotlib inline
seed=0

tf.compat.v2.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

#%%
# complete if model loaded from existing weights

weights = dict(np.load('./weights/', allow_pickle = True))
#Change the weights as needed
np.savez('./weights/', **weights)

weights = dict(np.load('', allow_pickle = True))
weights = weights['weights'].reshape(1)[0]
weights = fcts.adapt_for_opto(weights)
weights = fcts.change_opto_stim(weights, indices)

#%%
load_from_weights = False
N_callosal = 20
P_in = 0.4
P_rec = 0.4
P_out = 0.4

# ---------------------- Set up a basic model ---------------------------
sd = SensoryDiscrimination(dt = 10,
                              tau = 100, 
                              T = 2500, 
                              N_batch = 9,
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

if load_from_weights == True:
    dale_network_params.update(weights)
else:
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
# ---------------------- Plot example trial ---------------------------
trial = fcts.gen_pol_trials(daleModel, sd, [[0.6, 0.2]], sim=False)
a = fcts.visualise_trial(trial)

#%%
# ---------------------- Save and plot the weights of the network ---------------------------
weights = daleModel.get_weights()

fcts.plot_weights(weights, plot='connectivity')
plt.colorbar(plt.matshow(weights['W_rec'], norm=Normalize(vmin=-.5, vmax=.5)))

daleModel.save("weights/model_example_write_up_partial_connectivity_5")

#%%
#get contra and ipsi preferring cells in the model

trials = fcts.gen_pol_trials(fileModel, task, [[0.6,0.0], [0.0,0.6]], 0.0, sim=False)
arr = fcts.stim_pref_(trials)
arr1 = arr['max_hem1stim'][0:40]
arr2 = arr['max_hem2stim'][0:40]
contra_pref = fcts.count_pref(arr1, arr2, indices=True)
ipsi_pref = fcts.count_pref(arr2, arr1, indices=True)

#%%
#trials for PCA
nb_trials = 3
x,y,mask = np.array([]), np.array([]), np.array([]) 
stimuli = [[0.6,0.0], [0.0, 0.6], [0.2, 0.2]]
for i in range(len(stimuli)):
# ---------------------- Test the trained model ---------------------------
    params = sd.generate_trial_params(batch=0, trial=i)
    params['intensity_0'] = stimuli[i][0]
    params['intensity_1'] = stimuli[i][1]
    xt, yt, maskt = sd.generate_trial(params)
    
    xt = np.tile(xt, nb_trials)
    yt = np.tile(yt, nb_trials)
    maskt = np.tile(maskt, nb_trials)
    
    x = np.append(x, xt)
    y = np.append(y, yt)
    mask = np.append(mask, maskt)
        
x=np.transpose(x.reshape(len(stimuli)*nb_trials,250,3), (0,1,2))
y=np.transpose(y.reshape(len(stimuli)*nb_trials,250,2), (0,1,2))
mask=np.transpose(mask.reshape(len(stimuli)*nb_trials,250,2), (0,1,2))

model_output, model_state = daleModel.test(x) # run the model on input x

model_output = np.transpose(model_output.reshape(nb_trials, len(stimuli), 250, 2), (1,0,2,3))
model_state = np.transpose(model_state.reshape(nb_trials, len(stimuli), 250, 100), (1,0,2,3))

#calculate the PCA of the states of the network for multiple trials
#get average trajectory for trials with the same parameters
init_state = daleModel.get_weights()['init_state'].reshape(100)
model_state_init = np.insert(model_state, 0, init_state, axis=2)
PCA_data = np.concatenate(model_state_init, axis=0)
PCA_data = np.concatenate(PCA_data, axis=0)
#stdPCA_data = StandardScaler().fit_transform(PCA_data) # no normaisation ??

pca_states = PCA(n_components=3)
pca_states.fit(PCA_data)
PCA_states = pca_states.transform(PCA_data)
PCA_states = PCA_states.reshape(len(stimuli),nb_trials,251,3)
PCA_states = np.mean(PCA_states, axis=1)
#PCA_init_state = pca_states.fit_transform(init_state.reshape(1,100))

#%%
#plot the first 3 principal components

3D = True

figure = plt.figure(figsize=(6,6))
ax1 = plt.subplot(111) 

labels = []
for i in range(len(stimuli)):
    Diff= int((stimuli[i][0]-stimuli[i][1])/0.6*100)
    labels.append(Diff)
    
cmap = cm.get_cmap('RdYlBu')
rgba = cmap(np.linspace(0,1,len(labels)))

if 3D = True:
    ax1 = plt.axes(projection ='3d') 
    ax1.grid(False)

    for i in range(len(stimuli)):
        ax1.plot(PCA_states[i,0,0], PCA_states[i,0,2], PCA_states[i,0,1], c='black', marker = 'x')
        ax1.plot(PCA_states[i,0:51,0], PCA_states[i,0:51,2], PCA_states[i,0:51,1], linestyle='dashed', c=rgba[i])
        ax1.plot(PCA_states[i,51:151,0], PCA_states[i,51:151,2], PCA_states[i,51:151,1], linestyle='solid', c=rgba[i], label=labels[i])
        ax1.plot(PCA_states[i,151:,0], PCA_states[i,151:,2], PCA_states[i,151:,1], linestyle='dashdot', c=rgba[i])
    
    ax1.plot(PCA_states[2,0:51,0], PCA_states[2,0:51,2], PCA_states[2,0:51,1], linestyle='dashed', c='grey')
    ax1.plot(PCA_states[2,51:151,0], PCA_states[2,51:151,2], PCA_states[2,51:151,1], linestyle='solid', c='grey', label=labels[2])
    ax1.plot(PCA_states[2,151:,0], PCA_states[2,151:,2], PCA_states[2,151:,1], linestyle='dashdot', c='grey')

    ax1.set_zlabel('PC2')
    ax1.set_zlim(min(PCA_states[:,:,2].flatten()),max(PCA_states[:,:,2].flatten()))


else:
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)
        
    for i in range(len(stimuli)):
        ax1.plot(PCA_states[i,0,0], PCA_states[i,0,2], c='black', marker = 'x')
        ax1.plot(PCA_states[i,0:51,0], PCA_states[i,0:51,2], linestyle='dashed', c=rgba[i])
        ax1.plot(PCA_states[i,51:151,0], PCA_states[i,51:151,2], linestyle='solid', c=rgba[i], label=labels[i])
        ax1.plot(PCA_states[i,151:,0], PCA_states[i,151:,2], linestyle='dashdot', c=rgba[i])
    
    ax1.plot(PCA_states[2,0:51,0], PCA_states[2,0:51,2], linestyle='dashed', c='grey')
    ax1.plot(PCA_states[2,51:151,0], PCA_states[2,51:151,2], linestyle='solid', c='grey', label=labels[2])
    ax1.plot(PCA_states[2,151:,0], PCA_states[2,151:,2], linestyle='dashdot', c='grey')
    
    # ax1.plot(opto_states[15,0:51,0], opto_states[15,0:51,1], linestyle='dashed', c='darkorange')
    # ax1.plot(opto_states[15,51:151,0], opto_states[15,51:151,1], linestyle='solid', c='darkorange', label='ipsi stim')
    # ax1.plot(opto_states[15,151:,0], opto_states[15,151:,1], linestyle='dashdot', c='darkorange')
    
    # ax1.plot(contra_stim[23,0:51,0], contra_stim[23,0:51,1], linestyle='dashed', c='dodgerblue')
    # ax1.plot(contra_stim[23,51:151,0], contra_stim[23,51:151,1], linestyle='solid', c='dodgerblue', label='contra stim')
    # ax1.plot(contra_stim[23,151:,0], contra_stim[23,151:,1], linestyle='dashdot', c='dodgerblue')

# ax1.plot(opto_states[32,0:51,0], opto_states[32,0:51,2], opto_states[32,0:51,1], linestyle='dashed', c='darkorange')
# ax1.plot(opto_states[32,51:151,0], opto_states[32,51:151,2], opto_states[32,51:151,1], linestyle='solid', c='darkorange', label='ipsi stim')
# ax1.plot(opto_states[32,151:,0], opto_states[32,151:,2], opto_states[32,151:,1], linestyle='dashdot', c='darkorange')

# ax1.plot(contra_stim[23,0:51,0], contra_stim[23,0:51,2], contra_stim[23,0:51,1], linestyle='dashed', c='dodgerblue')
# ax1.plot(contra_stim[23,51:151,0], contra_stim[23,51:151,2], contra_stim[23,51:151,1], linestyle='solid', c='dodgerblue', label='contra stim')
# ax1.plot(contra_stim[23,151:,0], contra_stim[23,151:,2], contra_stim[23,151:,1], linestyle='dashdot', c='dodgerblue')

ax1.set_xlim(min(PCA_states[:,:,0].flatten()),max(PCA_states[:,:,0].flatten()))
ax1.set_ylim(min(PCA_states[:,:,1].flatten()),max(PCA_states[:,:,1].flatten()))
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC3')

ax1.legend(frameon=False, loc='upper left')

#%%
daleModel.destruct()

