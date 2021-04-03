# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:13:02 2021

@author: Isabelle
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import lognorm
from scipy.stats import norm
import os
from matplotlib import cm
import pandas as pd
from matplotlib import colors

import csv

%matplotlib inline

def count_ipsi_pref(array1, array2):
    list_of_indices = []
    for i in range(1,len(array1)):
        if array1[i] <= 0:
            list_of_indices.append(i)
          
    array1 = np.delete(array1, list_of_indices)
    array2 = np.delete(array2, list_of_indices)
    
    array3 = array1 - array2
    array3 = array3[array3>0]
    nb_ipsi_pref = array3.size
    
    return nb_ipsi_pref

#%%

dalemodel_test = dict(np.load("IpsiContra_In02_Rec02_Col10_s0.npz", allow_pickle=True))
#%%


for item in os.listdir('outputs'):
    dalemodel_test = dict(np.load(f'outputs/{item}', allow_pickle=True))

    stim_pref = dalemodel_test['stim_pref'].reshape(1)[0]
    loss = dalemodel_test['losses'][-1]
    
    filename = item[0:-4]
    
    P_in = round(float(item[13]),2) + round(float(item[14])*(.1), 2)
    P_rec = round(float(item[19]),2) + round(float(item[20])*(.1), 2)
    N_cal = int(item[25:27])
    seed = int(item[29])
    variance = np.std(stim_pref['max_hem2stim'][0:40]) * np.std(stim_pref['max_hem1stim'][40:80])



#%%
#plots the positions of the connection probability
figure = plt.figure()
ax1 = plt.subplot(111) 

ax1 = plt.axes(projection ='3d')
ax1.scatter(P_in, P_rec, P_out)
ax1.set_xlabel('P_in')
ax1.set_ylabel('P_rec')
ax1.set_zlabel('P_out')
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax1.set_zlim(0,1)

#%%
#plot the loss during training

fig1= plt.figure()
plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Training Iteration")
plt.title("Loss During Training")

#%%

#test network on test batch

# ---------------------- Test the trained model ---------------------------
x, y,mask, train_params = pd.get_trial_batch() # get pd task inputs and outputs
model_output, model_state = daleModel.test(x) # run the model on input x

# ---------------------- Plot the results ---------------------------
trial_nb = 0
for i in range(len(mask[trial_nb])):
    if mask[trial_nb][i][0] == 0:
        y[trial_nb][i] =+ np.nan

dt = params['dt']

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
#plot the relationship between reponse to stim 1 and stim2 for each neurons

model_info = pd.DataFrame(columns = ['filename', 'P_in', 'P_rec', 'N_cal', 'seed', 'loss',
                                     'mean_hem1_ipsi', 'mean_hem1_contra', 'mean_hem2_ipsi', 'mean_hem2_contra',
                                     'var_hem1_ipsi', 'var_hem1_contra', 'var_hem2_ipsi', 'var_hem2_contra',
                                     'max_hem1_ipsi', 'max_hem1_contra', 'max_hem2_ipsi', 'max_hem2_contra',
                                     'nb_hem1_ipsi_pref', 'nb_hem2_ipsi_pref'])

for item in os.listdir('outputs'):
    
    dalemodel_test = dict(np.load(f'outputs/{item}', allow_pickle=True))

    stim_pref = dalemodel_test['stim_pref'].reshape(1)[0]
    
    loss = dalemodel_test['losses'][-1]
    
    filename = item[0:-4]
    
    P_in = round(float(item[13]),2) + round(float(item[14])*(.1), 2)
    P_rec = round(float(item[19]),2) + round(float(item[20])*(.1), 2)
    N_cal = int(item[25:27])
    seed = int(item[29])

    mean_hem1_ipsi = np.mean(stim_pref['max_hem2stim'][0:40])
    mean_hem1_contra = np.mean(stim_pref['max_hem1stim'][0:40])
    mean_hem2_ipsi = np.mean(stim_pref['max_hem1stim'][40:80])
    mean_hem2_contra = np.mean(stim_pref['max_hem2stim'][40:80])
    
    var_hem1_ipsi = np.std(stim_pref['max_hem2stim'][0:40])
    var_hem1_contra = np.std(stim_pref['max_hem1stim'][0:40])
    var_hem2_ipsi = np.std(stim_pref['max_hem1stim'][40:80])
    var_hem2_contra = np.std(stim_pref['max_hem2stim'][40:80])
    
    max_hem1_ipsi = np.max(stim_pref['max_hem2stim'][0:40])
    max_hem1_contra = np.max(stim_pref['max_hem1stim'][0:40])
    max_hem2_ipsi = np.max(stim_pref['max_hem1stim'][40:80])
    max_hem2_contra = np.max(stim_pref['max_hem2stim'][40:80])
    
    
    nb_hem1_ipsi_pref = count_ipsi_pref(stim_pref['max_hem2stim'][0:40], stim_pref['max_hem1stim'][0:40])
    nb_hem2_ipsi_pref = count_ipsi_pref(stim_pref['max_hem1stim'][40:80], stim_pref['max_hem2stim'][40:80])
    
    
    new_row = {'filename':item, 'P_in':P_in, 'P_rec':P_rec, 'N_cal':N_cal, 'seed':seed, 'loss': loss,
               'mean_hem1_ipsi':mean_hem1_ipsi, 'mean_hem1_contra':mean_hem1_contra, 'mean_hem2_ipsi':mean_hem2_ipsi, 'mean_hem2_contra':mean_hem2_contra,
               'var_hem1_ipsi':var_hem1_ipsi, 'var_hem1_contra':var_hem1_contra, 'var_hem2_ipsi':var_hem2_ipsi, 'var_hem2_contra':var_hem2_contra,
               'max_hem1_ipsi':max_hem1_ipsi, 'max_hem1_contra':max_hem1_contra, 'max_hem2_ipsi':max_hem2_ipsi, 'max_hem2_contra':max_hem2_contra,
               'nb_hem1_ipsi_pref':nb_hem1_ipsi_pref, 'nb_hem2_ipsi_pref':nb_hem2_ipsi_pref}

    model_info = model_info.append(new_row, ignore_index = True)
    
    # figure = plt.figure(figsize=(6,6))
    # ax1 = plt.subplot(111)
    # ax1.scatter(stim_pref['max_hem1stim'][0:40], stim_pref['max_hem2stim'][0:40], c = 'coral', label = 'hemisphere 1', alpha=0.6)
    # ax1.scatter(stim_pref['max_hem1stim'][40:80], stim_pref['max_hem2stim'][40:80], c = 'green', label = 'hemisphere 2', alpha=0.6)
    # ax1.plot([-1, 1], [-1, 1], c='black')
    # ax1.set_xlim(-2, 2)
    # ax1.set_xticks([-1,-0.5,0, 0.5,1])
    # ax1.set_ylim(-2,2)
    # ax1.set_yticks([-1,-0.5,0, 0.5,1])
    # ax1.set_title(f'loss = {loss}')
    # ax1.legend()
    # ax1.set_xlabel('stim in hem 1')
    # ax1.set_ylabel('stim in hem 2')
   
model_info.to_csv('model_info.csv')
#%%
def take_first(elem):
    return elem[0]


columns = ['P_in', 'P_rec', 'N_cal', 'seed', 'nb_hem1_ipsi_pref', 'nb_hem1_ipsi_pref']
var = model_info[columns].to_numpy()


    
var_sort0 = np.array(sorted(var, key=take_first))
var_sort0 = var_sort0.reshape(4, 48, len(columns))
for i in [0,1,2,3]:
    var_sort0[i] = np.array(sorted(var_sort0[i], key=take_second))

var_sort0 = var_sort0.reshape(4, 4, 12, len(columns))
for i in [0,1,2,3]:
    for j in [0,1,2,3]:
        var_sort0[i][j] = np.array(sorted(var_sort0[i][j], key=take_third))
        
var_sort0 = var_sort0.reshape(4, 4, 4, 3, len(columns))
for i in [0,1,2,3]:
    for j in [0,1,2,3]:
        for k in [0,1,2,3]:
            var_sort0[i][j][k] = np.array(sorted(var_sort0[i][j][k], key=take_fourth))

#%%

ax = 1
fig4 = plt.figure(figsize = (10,10))
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, ax)
        plt.imshow(float(var_sort0[i,j,:,:,4]), cmap='viridis', norm=colors.Normalize(vmin=0.003, vmax=0.2))
        ax=ax+1
        
fig4.colorbar(mappable=float(var_sort0[0,0,:,:,4], cmap='viridis', fraction=.1, orientation='horizontal',)
#%%

N = 4

fig, axs = plt.subplots(N, N)
fig.suptitle('Multiple images')

images = []
for i in range(N):
    for j in range(N):
        # Generate data with a range that varies from one plot to the next.
        data = var_sort0[i,j,:,:,4]
        images.append(axs[i, j].imshow(data))
        axs[i, j].label_outer()

# Find the min and max of all colors for use in setting the color scale.
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)

fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.05)


# Make images respond to changes in the norm of other images (e.g. via the
# "edit axis, curves and images parameters" GUI on Qt), but be careful not to
# recurse infinitely!
def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())


for im in images:
    im.callbacksSM.connect('changed', update)

plt.show()

#%%
# ---------------------- Save and plot the weights of the network ---------------------------

weights = dalemodel_test['weights'].reshape(1)[0]

plot_weights(weights['W_rec'],  
            xlabel = 'From', 
            ylabel = 'To')

plot_weights(weights['W_in'])
plot_weights(weights['W_out'])


#%%
#try and do psychometric curve, need to update code for this
   
bins = pd.psychometric_curve(y, mask, train_params,9)

plt.plot(bins)
plt.xticks(ticks = np.linspace(0, 8, 9), labels=np.linspace(-1, 1, 9))

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
