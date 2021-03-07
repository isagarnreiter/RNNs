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
from oli_task_modif import PerceptualDiscrimination
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d 


%matplotlib inline
#%%

#access weights directly

weights = dict(np.load('./weights/saved_weights_new_task_sparse_connectivity_4.npz', allow_pickle = True))

# =============================================================================
# changes to be made to the weights
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

pd = PerceptualDiscrimination(dt = dt, tau = tau, T = T, N_batch = N_batch) # Initialize the task object
file_network_params = pd.get_task_params() # get the params passed in and defined in pd
file_network_params['N_rec'] = N_rec # set the number of recurrent units in the model
file_network_params['name'] = name
file_network_params['N_in'] = N_in
file_network_params['N_out'] = N_out

#load weights 
file_network_params['load_weights_path'] = './weights/probabs_20_02_08_08.npz'

fileModel = Basic(file_network_params)

#%%
#plot the weights of the network

weights = fileModel.get_weights()
plot_weights(weights['W_rec'], xlabel = 'From', ylabel = 'To')
plot_weights(weights['W_in'])
plot_weights(weights['W_out'])

#%%

#generate a single test trial

#initialise parameters manually
params_single_trial = {'intensity_0': 0.0, 
                       'intensity_1': 0.6, 
                       'random_output': 1, 
                       'stim_noise': 0.1, 
                       'onset_time': 0, 
                       'stim_duration': 500, 
                       'go_cue_onset': 1500, 
                       'go_cue_duration': 25.0, 
                       'post_go_cue': 125.0}

x, y, mask = pd.generate_trial(params_single_trial) #generate input and output

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
#calculate the PCA of the states of the network for a given trial

pca_states = PCA(n_components=3)
PCA_states = pca_states.fit_transform(model_state[trial_nb])

figure = plt.figure()
ax1 = plt.subplot(111) 

ax1 = plt.axes(projection ='3d') 

ax1.scatter(PCA_states[0:150,0], PCA_states[0:150,1], PCA_states[0:150,2], c='blue', s = 3)
ax1.scatter(PCA_states[150:250,0], PCA_states[150:250,1], PCA_states[150:250, 2], c='red', s = 3)
ax1.scatter(PCA_states[0,0], PCA_states[0,1], PCA_states[0,2], c='black', s = 10, marker = 'x')

ax1.set_ylim(-1.5,1.5)
ax1.set_xlim(-1.5,1.5)
ax1.set_zlim(-1.5, 1.5)
#plt.legend(('PC1', 'PC2'))

#%%
# ---------------------- Plot the results ---------------------------
trial_nb = 0
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
