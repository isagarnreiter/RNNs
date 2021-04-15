# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:13:02 2021

@author: Isabelle
"""

from oli_task import PerceptualDiscrimination
from matplotlib import pyplot as plt
from psychrnn.backend.models.basic import Basic
import numpy as np
from scipy.stats import lognorm, norm, gaussian_kde
import os
from matplotlib import cm, colors, colorbar
import pandas as pd
import shutil
import csv
import fcts
import seaborn as sns
from pylab import text
from mpl_toolkits.axes_grid1 import make_axes_locatable



%matplotlib inline

def count_pref(array1, array2):
    list_of_indices = []
    for i in range(1,len(array1)):
        if array1[i] <= 0:
            list_of_indices.append(i)
    array1 = np.delete(array1, list_of_indices)
    array2 = np.delete(array2, list_of_indices)
    
    array3 = array1 - array2
    array3 = array3[array3>0]
    nb_pref = array3.size
    
    return nb_pref


def take_first(elem):
    return elem[0]

def take_second(elem):
    return elem[1]

def take_third(elem):
    return elem[2]

def take_fourth(elem):
    return elem[3]

#%%
#load dataframes

model_best = pd.read_csv('model_best.csv', index_col='Unnamed: 0')
model_info = pd.read_csv('model_info.csv', index_col='Unnamed: 0')

#%% 
#produce dataframe of with info about all models

# model_info = pd.DataFrame(columns = ['filename', 'P_in', 'P_rec', 'N_cal', 'seed', 'loss',
#                                      'mean_hem1_ipsi', 'mean_hem1_contra', 'mean_hem2_ipsi', 'mean_hem2_contra',
#                                      'var_hem1_ipsi', 'var_hem1_contra', 'var_hem2_ipsi', 'var_hem2_contra',
#                                      'max_hem1_ipsi', 'max_hem1_contra', 'max_hem2_ipsi', 'max_hem2_contra',
#                                      'nb_hem1_ipsi_pref', 'nb_hem2_ipsi_pref'])

for item in os.listdir('outputs2'):
    
    dalemodel_test = dict(np.load(f'outputs2/{item}', allow_pickle=True))
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
    
    nb_hem1_ipsi_pref = count_pref(stim_pref['max_hem2stim'][0:40], stim_pref['max_hem1stim'][0:40])
    nb_hem2_ipsi_pref = count_pref(stim_pref['max_hem1stim'][40:80], stim_pref['max_hem2stim'][40:80])
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
    # figure.savefig(f'stimpref_figs/{item[0:-4]}')
    
hem1_contra_pref = np.array([])
hem2_contra_pref = np.array([])
for item in os.listdir('outputs'):
    dalemodel_test = dict(np.load(f'outputs/{item}', allow_pickle=True))
    stim_pref = dalemodel_test['arr_0'][0]['stim_pref'].reshape(1)[0] 
    hem1_contra_pref = np.append(hem1_contra_pref, count_pref(stim_pref['max_hem1stim'][0:40], stim_pref['max_hem2stim'][0:40]))
    hem2_contra_pref = np.append(hem2_contra_pref, count_pref(stim_pref['max_hem2stim'][40:80], stim_pref['max_hem1stim'][40:80]))
    
model_info['hem1_contra_pref'] = hem1_contra_pref
model_info['hem2_contra_pref'] = hem2_contra_pref
model_info['total_active_hem1'] = model_info['hem1_contra_pref'] + model_info['nb_hem1_ipsi_pref']
model_info['total_active_hem2'] = model_info['hem2_contra_pref'] + model_info['nb_hem2_ipsi_pref']

model_info.to_csv('model_info.csv')

#%%
#make seperate dataframe with defined number of ipsi preferring cells

model_best = model_info[model_info.columns[:]].to_numpy()
columns = model_info.columns[:]

indexes=[]
for i in range(model_best.shape[0]):
    if model_best[i][18]<=1 or model_best[i][18]>=12 or model_best[i][19]<=2 or model_best[i][19]>=12 or abs(model_best[i][18]-model_best[i][19]) >= 3:
        indexes.append(i)

model_best = np.delete(model_best, indexes, axis=0)
model_best_df = pd.DataFrame(model_best, columns = columns)
model_best_df.to_csv('\model_best_new.csv')

#%%

#add all figures generated for stim preference to a seperate folder
for item in os.listdir('trials'):
    if f'{item[:-13]}.npz' in np.array(model_best[['filename']])[:,0]:
        newPath = shutil.copy(f'trials\{item}', 'trials_select')


#%%
#save weights
for item in os.listdir('outputs'):
    dalemodel_test = dict(np.load(f'outputs/{item}', allow_pickle=True))
    weights = dalemodel_test['weights'].reshape(1)[0]
    np.savez(f'weights\{item}', **weights)

#%%

#test network on test batch

task = PerceptualDiscrimination(dt = 10, # The simulation timestep
                              tau = 100, # The intrinsic time constant of neural state decay.
                              T = 2500, # The trial length, 
                              N_batch = 50) # Initialize the task object

network_params = task.get_task_params() # get the params passed in and defined in pd
network_params['N_rec'] = 100 # set the number of recurrent units in the model
network_params['name'] = 'model'
network_params['N_in'] = 3
network_params['N_out'] = 2

dt = 10
results = ['x', 'y', 'model_state', 'model_output']
labels = ['Input', 'Expected Output', 'State of each Neuron', 'Output']
lims = [(), (-0.1, 1.1), (-0.1, 1.1), (-0.5, 0.5)]

for item in os.listdir('weights'):
        
    network_params['load_weights_path'] = f'weights/{item}'
    Model = Basic(network_params)
    trials = fcts.gen_pol_trials(Model, task)
    
    dalemodel_test = dict(np.load(f'outputs/{item}', allow_pickle=True))
    dalemodel_test = np.append(dalemodel_test, trials)
    
    np.savez(f'outputs/{item[0:-4]}', dalemodel_test)
    
    for l in list(trials.keys()):
    #make a figure of the trials
    
        for i in range(len(trials[l]['mask'])):
            if trials[l]['mask'][i][0] == 0:
                trials[l]['y'][i][0] =+ np.nan
        
        x_len = range(0,len(trials[l]['x'])*dt,dt)
        data = {'H1':trials[l]['model_state'][:,0:40], 'H2':trials[l]['model_state'][:,40:80]}
        keys = list(data.keys())
        
        fig2, ax = plt.subplots(2, 3, figsize=(30,8))
        fig2.suptitle(f'{l} Trial for: {item}', fontsize=16)
        x=0
        for i in range(2):
            for j in range(2):
                ax[i,j].plot(x_len, trials[l][results[x]])
                ax[i,j].set_title(labels[x], fontsize = 14)
                x= x+1
                
            ax[i,2].plot(x_len, data[keys[i]], alpha=0.9)
            ax[i,2].set_ylim(-0.8,0.8)
            ax[i,2].set_title(f"{keys[i]}", fontsize = 14)
        
        for i in range(3):
            ax[1,i].set_xlabel("Time (ms)", fontsize = 10)
            
        ax[0,0].legend(["Input Channel 1", "Input Channel 2", 'go cue'])
        
        fig2.tight_layout()
        fig2.savefig(f'trials/{item[0:-4]}_{l}')
    
    Model.destruct()
    

#%%

#create structured np.array to produce a heat map for different variables 

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
#plot histogram of parameters in model_best file

fig, ax = plt.subplots(2, 2, figsize=(5,5))

x=0
labels = ['P_in', 'P_rec', 'N_cal', 'seed']

for i in [0,1]:
    for j in [0,1]:        
        ax[i,j].hist(model_best[[labels[x]]])
        ax[i,j].set_xlabel(labels[x])
        x=x+1
        
ax[0,0].set_xticks([0.08, 0.1, 0.25, 0.5, 0.75, 1])
ax[0,1].set_xticks([0.08, 0.1, 0.25, 0.5, 0.75, 1])
ax[1,0].set_xticks([10, 20, 30, 40])
ax[1,1].set_xticks([0,1,2])

plt.tight_layout()

#check relation between different parameters and if different associations are more likely
#%%
p_in = [0.1, 0.2, 0.5, 0.7, 1]
p_rec = [0.0, 0.1, 0.2, 0.5, 0.7, 1]
N_cal = [10, 20, 30, 40]

x = model_best['seed']
y =  model_best['N_cal']

# list_x = {0.0:[0,0], 0.1:[0,1], 0.2:[1,2], 0.5:[2,3], 0.7:[3,4], 1.0:[4,5]}
# other_x=[list_x[i][0] for i in x]

cmap = sns.color_palette("viridis",  as_cmap=True )
norm = colors.Normalize(vmin=model_best['fraction that prefer ipsi stim'].min(), vmax=model_best['fraction that prefer ipsi stim'].max())
colours = {}
for cval in model_best['fraction that prefer ipsi stim']:
    colours.update({cval : cmap(norm(cval))})


figure4, ax = plt.subplots(1,1, figsize=(6,6))
ax = sns.swarmplot(x, y, size=10, hue=model_best['fraction that prefer ipsi stim'], palette=colours)

plt.gca().legend_.remove()

divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
figure4.add_axes(ax_cb)
cb1 = colorbar.ColorbarBase(ax_cb, cmap=cmap,
                                norm=norm,
                                orientation='vertical')




#ax.set_xticks(p_in)
#ax.set_yticks(N_cal)

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

#%% compare distribution of variance and mean response between model_info and model_best file

mean_resp_all = np.array(np.mean(model_info[list(columns[9:13])], axis=1))
mean_resp_best = np.array(np.mean(model_best[list(columns[9:13])], axis=1))

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
