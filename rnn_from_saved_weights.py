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

# =============================================================================
# add changes to be made to the weights
# =============================================================================

np.savez('./weights/modified_saved_weights.npz', **weights)

#%%
dt = 10 # The simulation timestep.
tau = 100 # The intrinsic time constant of neural state decay.
T = 2500 # The trial length.
N_batch = 50 # The number of trials per training update.
N_rec = 100 # The number of recurrent units in the network.
N_in = 3
N_out = 2
name = 'dale_model' #  Unique name used to determine variable scope for internal use.

model = dict(np.load('/UserFolder/neur0003/third_set_models/IpsiContra_In05_Rec025_Cal20_s0.npz', allow_pickle=True))
weights = model['weights'].reshape(1)[0]
#weights_modif = adapt_for_opto(weights)

task = oli_task.PerceptualDiscrimination(dt = dt, tau = tau, T = T, N_batch = N_batch) # Initialize the task object
file_network_params = task.get_task_params() # get the params passed in and defined in pd
file_network_params['N_rec'] = N_rec # set the number of recurrent units in the model
file_network_params['name'] = name
file_network_params['N_in'] = N_in
file_network_params['N_out'] = N_out
file_network_params.update(weights)
#load weights 
#file_network_params['load_weight_path'] = 'weights/model_example_write_up.npz.'

fileModel = Basic(file_network_params)


#%%
#plot the weights of the network

weights = fileModel.get_weights()
plot_weights(weights['W_rec'], xlabel = 'From', ylabel = 'To', matrix='rec')
plot_weights(weights['W_in'])
plot_weights(weights['W_out'])

#%%

# ---------------------- Test the trained model ---------------------------
x, y,mask, train_params = task.get_trial_batch() # get pd task inputs and outputs
model_output, model_state = fileModel.test(x) # run the model on input x

#%%

#generate a single test trial

#initialise parameters manually
params_single_trial = {'intensity_0': 0.0, 
                       'intensity_1': 0.0, 
                       'random_output': 1, 
                       'stim_noise': 0.1, 
                       'onset_time': 0, 
                       'stim_duration': 500, 
                       'go_cue_onset': 1500, 
                       'go_cue_duration': 25.0, 
                       'post_go_cue': 125.0,}

x, y, mask = task_pert.generate_trial(params_single_trial) #generate input and output

#add dimension to shape of x, y, mask to fit the test() function and the figure format
x = np.array([x])
mask = np.array([mask])
y = np.array([y])

model_output, model_state = fileModel.test(x) # run the model on input x


#save the state of excitatory neurons right after stimulus fore either a stim to hemi 1 or 2
if params_single_trial['intensity_0'] == 0:
    max_hem2_hem2stim = model_state[trial_nb,50,40:80]  
    max_hem1_hem2stim = model_state[trial_nb,50,0:40]
elif params_single_trial['intensity_1'] == 0:    
    max_hem2_hem1stim = model_state[trial_nb,50,40:80]
    max_hem1_hem1stim = model_state[trial_nb,50,0:40]


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
bin_means, bins, frac_choice = task.psychometric_curve(model_output, train_params)

plt.plot(bins, bin_means[0],marker='o', label='choice 1')
plt.plot(bins, bin_means[1],marker='o', c='orange', label = 'choice 2')
plt.xlabel('U1-U2')
plt.legend()
plt.ylabel('%')


#%% 
#calculate the PCA of the states of the network for a given trial
#a = model_state[trial_n]
init_state = fileModel.get_weights()['init_state'].reshape(100)
# init_state = np.repeat(init_state, 50)
# init_state = init_state.reshape(100, 50).T

model_state_init = np.insert(model_state, 0, init_state, axis=1)
model_state_init = np.round(model_state_init, 5)
PCA_data = np.concatenate(model_state_init, axis=0)
#stdPCA_data = StandardScaler().fit_transform(PCA_data) # no normaisation ??


pca_states = PCA(n_components=3)
PCA_states = pca_states.fit_transform(PCA_data)
#PCA_init_state = pca_states.fit_transform(init_state.reshape(1,100))

#%%
figure = plt.figure(figsize=(6,6))
ax1 = plt.subplot(111) 

ax1 = plt.axes(projection ='3d') 
ax1.grid(False)
trial_nb = [0,2,3,11]
colour = ['blue', 'green', 'orange', 'yellow']
labels = ['ch1', 'ch2', 'equ', 'no']
for i in range(len(trial_nb)):
    start = 251*trial_nb[i]
    stop = 251*trial_nb[i]+251
    ax1.plot(PCA_states[start:stop,0], PCA_states[start:stop,1], PCA_states[start:stop,2], c=colour[i], label=labels[i])
ax1.plot(PCA_states[start,0], PCA_states[start,1], PCA_states[start,2], c='black', marker = 'x')
ax1.set_xlim(min(PCA_states[:,0]),max(PCA_states[:,0]))
ax1.set_ylim(min(PCA_states[:,1]),max(PCA_states[:,1]))
ax1.set_zlim(min(PCA_states[:,2]),max(PCA_states[:,2]))
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')
ax1.legend(frameon=False, loc='upper left')
#%%

figure = plt.figure(figsize=(6,6))
ax1 = plt.subplot(111) 
trial_nb = [0,2,3,11]
colour = ['blue', 'green', 'orange', 'yellow']
labels = ['ch1', 'ch2', 'equ', 'no']
for i in range(len(trial_nb)):
    start = 251*trial_nb[i]
    stop = 251*trial_nb[i]+251
    ax1.plot(PCA_states[start:stop,0], PCA_states[start:stop,1], c=colour[i], label=labels[i])
    ax1.plot(PCA_states[start,0], PCA_states[start,1], c='black', marker = 'x')


ax1.set_xlim(min(PCA_states[:,0]),max(PCA_states[:,0]))
ax1.set_ylim(min(PCA_states[:,1]),max(PCA_states[:,1]))
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.legend(frameon=False, loc='upper right')

#%%
trial_nb = 1
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
        ax[i,j].plot(range(0, len(zipp[z][0,:,:])*dt,dt), zipp[z][trial_nb,:,:])
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

#plot the states of neurons depending on whether they are inhibtory excitatory, or project to the other hemisphere
fig3 = plt.figure()
ax1 = plt.subplot(211)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,80:90], c = 'blue', alpha=0.6)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,0:30], c = 'red', alpha=0.6)
ax1.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,30:40], c = 'black', alpha=0.6)
ax1.set_title("State of each neuron in H1", fontsize = 10)
ax1.set_ylim(-0.5, 0.5)

ax2 = plt.subplot(212)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,40:70], c = 'red', alpha=0.6)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,90:100], c='blue', alpha=0.6)
ax2.plot(range(0, len(x[0,:,:])*dt,dt), model_state[trial_nb,:,70:80], c='black', alpha=0.6)
ax2.set_xlabel("Time (ms)", fontsize = 10)
ax2.set_title("State of each neuron in H2", fontsize = 10)
ax2.set_ylim(-0.5, 0.5)

plt.tight_layout()



#%%
fileModel.destruct()
