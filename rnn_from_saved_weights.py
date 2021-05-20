# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:47:02 2020

@author: Isabelle
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from fcts import plot_weights
from psychrnn.backend.models.basic import Basic
import tensorflow as tf0
from oli_task import PerceptualDiscrimination
from oli_task_perturb import PerceptualDiscrimination
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d 
from sklearn.preprocessing import StandardScaler

%matplotlib inline
#%%

#access weights directly

weights = dict(np.load('./weights/model_example_write_up.npz.npz', allow_pickle = True))

np.savez('./weights/model_example_write_up_full_connectivity', **weights)

#%%
dt = 10 # The simulation timestep.
tau = 100 # The intrinsic time constant of neural state decay.
T = 2500 # The trial length.
N_batch = 50 # The number of trials per training update.
N_rec = 100 # The number of recurrent units in the network.
N_in = 4
N_out = 2
name = 'dale_model' #  Unique name used to determine variable scope for internal use.

#model = dict(np.load('models/model_example_write_up.npz', allow_pickle=True))
#weights = model['weights'].reshape(1)[0]
#weights_modif = adapt_for_opto(weights)

task = oli_task_perturb.PerceptualDiscrimination(dt = dt, tau = tau, T = T, N_batch = N_batch, N_in=N_in, N_rec=N_rec, N_out=N_out, opto=True) # Initialize the task object
file_network_params = task.get_task_params() # get the params passed in and defined in pd
file_network_params['N_rec'] = N_rec # set the number of recurrent units in the model
file_network_params['name'] = name
file_network_params['N_in'] = N_in
file_network_params['N_out'] = N_out
file_network_params['rec_noise'] = 0.0
#file_network_params.update(weights)
#load weights 
weights = dict(np.load('IpsiContra_In075_Rec10_Cal20_s15.npz', allow_pickle = True))
weights = weights['weights'].reshape(1)[0]
weights = fcts.adapt_for_opto(weights)
weights = fcts.change_opto_stim(weights, ipsi_pref)
file_network_params.update(weights)
fileModel = Basic(file_network_params)


#%%
#plot the weights of the network

weights = fileModel.get_weights()

fig = plt.figure(figsize=(8,8))
ax1= fcts.plot_weights(weights, plot="weights")

#plt.colorbar(w_rec.matshow(weights[data[0]], norm=Normalize(vmin=-.5, vmax=.5)))

#%%
#get contra and ipsi preferring cells in the model

trials = fcts.gen_pol_trials(fileModel, task, [[0.0,0.6], [0.6,0.0]], 0.0, sim=False)
arr = fcts.stim_pref_(trials)
arr1 = arr['max_hem1stim'][0:40]
arr2 = arr['max_hem2stim'][0:40]
contra_pref = fcts.count_pref(arr1, arr2, indices=True)
ipsi_pref = fcts.count_pref(arr2, arr1, indices=True)

#%%
#trials for PCA
av = 10
x = np.array([])
y = np.array([])
mask = np.array([])
params_list = {}
for i in range(50):
# ---------------------- Test the trained model ---------------------------
    params = task.generate_trial_params(batch=0, trial=i)
    params_list[i] = params
    for i in range(av):
        xt,yt,maskt = task.generate_trial(params)
        x = np.append(x, xt)
        y = np.append(y, yt)
        mask = np.append(mask, maskt)

x=np.transpose(x.reshape(50,av,250,4), (1,0,2,3))
y=np.transpose(y.reshape(50,av,250,2), (1,0,2,3))
mask=np.transpose(mask.reshape(50,av,250,2), (1,0,2,3))

model_output = np.array([])
model_state = np.array([])
for i in range(av):
    model_output_temp, model_state_temp = fileModel.test(x[i]) # run the model on input x
    model_output = np.append(model_output, model_output_temp)
    model_state = np.append(model_state, model_state_temp)
    
model_output = np.transpose(model_output.reshape(av, 50, 250, 2), (1,0,2,3))
model_state = np.transpose(model_state.reshape(av, 50, 250, 100), (1,0,2,3))
#%% 
#calculate the PCA of the states of the network for multiple trials
#get average trajectory for trials with the same parameters
av=10
init_state = fileModel.get_weights()['init_state'].reshape(100)
model_state_init = np.insert(model_state, 0, init_state, axis=2)
PCA_data = np.concatenate(model_state_init, axis=0)
PCA_data = np.concatenate(PCA_data, axis=0)
#stdPCA_data = StandardScaler().fit_transform(PCA_data) # no normaisation ??


pca_states = PCA(n_components=3)
pca_states.fit(PCA_data)
PCA_states = pca_states.transform(PCA_data)
PCA_states = PCA_states.reshape(50,av,251,3)
PCA_states = np.mean(PCA_states, axis=1)
#PCA_init_state = pca_states.fit_transform(init_state.reshape(1,100))

#%%

#plot the first 3 principal components

figure = plt.figure(figsize=(6,6))
ax1 = plt.subplot(111) 

ax1 = plt.axes(projection ='3d') 
ax1.grid(False)
trial_nb = [47,21,49,43]
colour = ['blue', 'green', 'orange', 'yellow']
labels = ['ch1', 'ch2', 'equ', 'no']
ax1.plot(PCA_states[trial_nb[0],0,0], PCA_states[trial_nb[0],0,1], PCA_states[trial_nb[0],0,2], c='black', marker = 'x')
for i in range(len(trial_nb)):
    ax1.plot(PCA_states[trial_nb[i],:,0], PCA_states[trial_nb[i],:,1], PCA_states[trial_nb[i],:,2], c=colour[i], label=labels[i])
ax1.set_xlim(min(PCA_states[:,:,0].flatten()),max(PCA_states[:,:,0].flatten()))
ax1.set_ylim(min(PCA_states[:,:,1].flatten()),max(PCA_states[:,:,1].flatten()))
ax1.set_zlim(min(PCA_states[:,:,2].flatten()),max(PCA_states[:,:,2].flatten()))
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')
ax1.legend(frameon=False, loc='upper left')
#%%
#plot the first two principal components

figure = plt.figure(figsize=(5,5))
ax1 = plt.subplot(111) 
labels = {'-100':49, '-66':46, '-33':34, '0':48, '33':41, '66':22, '100':40}
cmap = cm.get_cmap('RdYlBu')
rgba = cmap(np.linspace(0,1,7))
keys = list(labels.keys())
ind = [0,6]
for i in ind:
    ax1.plot(PCA_states[labels[keys[i]],:,0], PCA_states[labels[keys[i]],:,1], c=rgba[i], label=keys[i])
    ax1.plot(PCA_states[labels[keys[i]],0,0], PCA_states[labels[keys[i]],0,1], c='black', marker = 'x')
    ax1.plot(PCA_states[labels[keys[i]],51,0], PCA_states[labels[keys[i]],51,1], c=rgba[i], marker = 'o', markersize=4)
    ax1.plot(PCA_states[labels[keys[i]],151,0], PCA_states[labels[keys[i]],151,1], c=rgba[i], marker = 'o', markersize=4)

ax1.plot(PCA_states[labels[keys[3]],:,0], PCA_states[labels[keys[3]],:,1], c='grey', label='0')

ax1.plot(opto_states[32,:,0], opto_states[32,:,1], c='black', label='opto')

ax1.set_xlim(min(PCA_states[:,:,0].flatten())-0.05,max(PCA_states[:,:,0].flatten())+0.05)
ax1.set_ylim(min(PCA_states[:,:,1].flatten())-0.05,max(PCA_states[:,:,1].flatten())+0.05)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.legend(frameon=False, loc='upper right', fontsize=9)

#%%
#test network on batch of random trials

x, y,mask, train_params = task.get_trial_batch() # get pd task inputs and outputs
model_output, model_state = fileModel.test(x) # run the model on input x
#%%

#generate a single test trial

#initialise parameters manually
params_single_trial = {'intensity_0': 0.2, 
                       'intensity_1': 0.4, 
                       'random_output': 1, 
                       'stim_noise': 0.1, 
                       'onset_time': 0, 
                       'stim_duration': 500, 
                       'go_cue_onset': 1500, 
                       'go_cue_duration': 25.0, 
                       'post_go_cue': 125.0,
                       'intensity_opto':0.4,
                       'end_opto':500}

x, y, mask = task_pert.generate_trial(params_single_trial) #generate input and output

#add dimension to shape of x, y, mask to fit the test() function and the figure format
x = np.array([x[:]])
mask = np.array([mask])
y = np.array([y])

model_output, model_state = fileModel.test(x) # run the model on input x



#%%
bin_means, bins, frac_choice = psychometric_curve(task, model_output, train_params)

plt.plot(bins, bin_means,marker='o', label='choice 1')
plt.xlabel('coherence')
plt.ylabel('% choice 1')
plt.ylim(-5,105)


#%%
trial_nb = 0
for i in range(len(mask[0])):
    if mask[0,i][0] == 0:
        y[0,i] =+ np.nan

dt = 10

fig2, ax = plt.subplots(2,2,figsize=(10,5))

means = np.array([np.mean(model_state[0,:,contra_pref], axis = 0), np.mean(model_state[0,:,ipsi_pref], axis = 0)]).T
std_upper = np.array([means[:,0] + np.std(model_state[0,:,contra_pref], axis = 0), means[:,1] + np.std(model_state[0,:,ipsi_pref], axis = 0)]).T
std_lower = np.array([means[:,0] - np.std(model_state[0,:,contra_pref], axis = 0), means[:,1] - np.std(model_state[0,:,ipsi_pref], axis = 0)]).T


z=0
zipp = [x[0,0:250,0],y[0],means[0:100], model_output[0,:,:]]
titles = ['Input', 'Target Output', 'Hemisphere 1', 'Output']
for i in range(2):
    for j in range(2):
        ax[i,j].plot(range(0, len(zipp[z])*dt,dt), zipp[z])
        ax[i,j].set_title(titles[z], fontsize=14)
        ax[i,j].set_ylim(-0.1,1.1)
        ax[i,j].set_yticks([0,1])
        ax[i,j].set_yticklabels([0,1], fontsize=12)
        ax[1,j].set_xlabel("Time (ms)", fontsize = 12)
        z+=1
ax[1,0].fill_between(range(0, 100*dt,dt), std_lower[0:100,0], std_upper[0:100,0], alpha=0.2, color='C0')
ax[1,0].fill_between(range(0, 100*dt,dt), std_lower[0:100,1], std_upper[0:100,1], alpha=0.2, color='C1')
ax[0,0].legend(["Input 1", "Input  2", 'go cue', 'stim'], frameon=False)
ax[1,0].legend(['contra', 'ipsi'], frameon=False)
ax[1,0].set_ylim(-0.3, 0.6)
ax[1,0].set_yticks([-0.2, 0, 0.2, 0.4, 0.6])
ax[1,0].set_yticklabels([-0.2, 0, 0.2, 0.4, 0.6], fontsize=12)

fig2.tight_layout()

#%% 
#plot the relationship between reponse to stim 1 and stim2 for each neurons
unity_line = [-1, 0, 1]

figure = plt.figure(figsize=(6,6))
ax1 = plt.subplot(111)
ax1.scatter(max_hem1_hem1stim, max_hem1_hem2stim, c = 'coral', label = 'hemisphere 1', alpha=0.6)
ax1.scatter(max_hem2_hem1stim, max_hem2_hem2stim, c = 'green', label = 'hemisphere 2', alpha=0.6)
ax1.plot(unity_line, unity_line, c='black')
ax1.set_xlim(-1,1)
ax1.set_xticks([-1,-0.5,0, 0.5,1])
ax1.set_ylim(-1,1)
ax1.set_yticks([-1,-0.5,0, 0.5,1])
ax1.set_title('states of excitatory neuron in hemisphere 1 and 2 at T = 500 ms')
ax1.legend()
ax1.set_xlabel('stim in hem 1')
ax1.set_ylabel('stim in hem 2')

#%%
fileModel.destruct()
