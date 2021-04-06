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


def take_first(elem):
    return elem[0]

def take_second(elem):
    return elem[1]

def take_third(elem):
    return elem[2]

def take_fourth(elem):
    return elem[3]

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
plt.plot(model_best[5])
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
#produce dataframe of with info about all models

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
#make seperate dataframe with defined number of ipsi preferring cells

model_best = model_info[model_info.columns[:]].to_numpy()
columns = model_info.columns[:]

indexes=[]
for i in range(model_best.shape[0]):
    if model_best[i][18]<=2 or model_best[i][18]>=10 or model_best[i][19]<=2 or model_best[i][19]>=10 or abs(model_best[i][18]-model_best[i][19]) >= 3:
        indexes.append(i)

model_best = np.delete(model_best, indexes, axis=0)

model_best_df = pd.DataFrame(model_best, columns = columns)

#model_best_df.to_csv('model_best.csv')#%%

#%%
#load dataframes

model_best = pd.read_csv('model_best.csv', index_col='Unnamed: 0')
model_info = pd.read_csv('model_info.csv', index_col='Unnamed: 0')

#%%
arr_info = model_info[model_info.columns[1:]].to_numpy()
columns = model_info.columns[1:]

file = np.array(sorted(arr_info, key=take_first))
file = file.reshape(4, 48, len(columns))
for i in range(4):
    file[i] = np.array(sorted(file[i], key=take_second))
file = file.reshape(4, 4, 12, len(columns))
for i in range(4):
    for j in range(4):
        file[i][j] = np.array(sorted(file[i][j], key=take_third))        
file = file.reshape(4, 4, 4, 3, len(columns))
for i in range(4):
    for j in range(4):
        for k in range(4):
            file[i][j][k] = np.array(sorted(file[i][j][k], key=take_fourth))

#%%
# mean_total = np.mean(model_best[:,6:10], axis=1)
# mean_var = np.mean(model_best[:, 10:14], axis =1)

fig, ax = plt.subplots(2, 2, figsize=(5,5))

x=0
labels = ['P_in', 'P_rec', 'N_cal', 'seed']

for i in [0,1]:
    for j in [0,1]:        
        ax[i,j].hist(model_best[[labels[x]]])
        ax[i,j].set_xlabel(labels[x])
        x=x+1
        
ax[0,0].set_xticks([0.2, 0.5, 0.7, 1])
ax[0,1].set_xticks([0.2, 0.5, 0.7, 1])
ax[1,0].set_xticks([10, 20, 30, 40])
ax[1,1].set_xticks([0,1,2])
ax[1,1].set_yticks([0,3,6,9,12])

plt.tight_layout()

#%%

N = 4

fig, axs = plt.subplots(N, N, figsize=(4,6))
fig.suptitle('number of neurons with a preference for ipsilateral stim')
images = []
for i in range(N):
    for j in range(N):
        # Generate data with a range that varies from one plot to the next.
        data = np.sum(var_sort0[i,j,:,:,17:19], axis=2)
        axs[i, j].imshow(data, aspect='auto')
        images.append(axs[i, j].imshow(data, aspect='auto'))
        axs[i, j].set_yticks([0,1,2,3])
        axs[i, j].set_yticklabels([10,20,30,40], size=8)
        axs[i, j].set_xticks([0,1,2])
        axs[i, j].set_xticklabels([0,1,2], size=8)
        axs[i, j].label_outer()

# Find the min and max of all colors for use in setting the color scale.
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)

fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.07)

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

#%% compare distribution of variance and mean response

mean_resp_all = np.array(np.mean(model_info[list(columns[5:9])], axis=1))
mean_resp_best = np.array(np.mean(model_best[list(columns[5:9])], axis=1))

from scipy.stats import gaussian_kde

resp_all_dens = gaussian_kde(list(mean_resp_all))
resp_best_dens = gaussian_kde(list(mean_resp_best))

#xs = np.linspace(-0.3, 0.02, 100)
xs = np.linspace(0.1, 0.5, 100)


dens, ax = plt.subplots(1,1)
ax.set_title('mean response distribution')
ax.hist([mean_resp_all, mean_resp_best],  bins=10 , color=['red', 'blue'], label = ['all models', 'best models'])
#ax.plot(xs, resp_all_dens(xs), c='red', label='all')
#ax.plot(xs, resp_best_dens(xs), c='blue', label='best')
ax.legend()
plt.show()
#%%

axes =plt.figure(figsize = (4,6))
ax2 = axes.add_axes([0.05,0.25,0.9,0.65])
ax2.set_ylim(1,0)
ax2.set_yticks([0.15, 0.4,0.65,0.9])
ax2.set_yticklabels([0.2,0.5,0.7,1])
ax2.set_ylabel('P(in)')
ax2.set_xticks([0.15, 0.4, 0.65, 0.9])
ax2.set_xticklabels([0.2, 0.5, 0.7, 1])
ax2.set_xlabel('P(rec)')
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
