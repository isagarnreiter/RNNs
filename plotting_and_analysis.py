# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:13:02 2021

@author: Isabelle
"""

import oli_task
import oli_task_perturb
from matplotlib import pyplot as plt
from psychrnn.backend.simulation import BasicSimulator
from psychrnn.backend.models.basic import Basic
import numpy as np
from scipy.stats import lognorm, norm, gaussian_kde
import os
from matplotlib import cm, colors, colorbar, markers
import pandas as pd
import shutil
import csv
import fcts
import random
from fcts import count_pref
import seaborn as sns
from pylab import text
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
%matplotlib inline

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

model_best = pd.read_csv('/UserFolder/neur0003/model_best.csv', index_col='Unnamed: 0')
first_set = pd.read_csv('/UserFolder/neur0003/first_set_model.csv', index_col='Unnamed: 0')
first_set = pd.read_pickle('/UserFolder/neur0003/first_set_model.pkl')

second_set = pd.read_csv('/UserFolder/neur0003/second_set_model.csv', index_col='Unnamed: 0')
model_best1 = pd.read_csv('/UserFolder/neur0003/first_set_best.csv', index_col='Unnamed: 0')
third_set = pd.read_pickle('/UserFolder/neur0003/third_set_model.pkl')

#%% 
#produce dataframe of with info about all models

third_set = pd.DataFrame(columns = ['filename', 'P_in', 'P_rec', 'N_cal', 'seed', 'loss',
                                      'mean_hem1_ipsi', 'mean_hem1_contra', 'mean_hem2_ipsi', 'mean_hem2_contra',
                                      'var_hem1_ipsi', 'var_hem1_contra', 'var_hem2_ipsi', 'var_hem2_contra',
                                      'nb_hem1_ipsi_pref', 'nb_hem2_ipsi_pref', 'nb_hem1_contra_pref', 'nb_hem2_contra_pref',
                                      'total_active', 'fraction_ipsi_pref', 
                                      'stim_pref_hem1stim_hem1', 'stim_pref_hem1stim_hem2', 'stim_pref_hem2stim_hem2', 'stim_pref_hem2stim_hem1'])

for item in os.listdir('/UserFolder/neur0003/third_set_models'):
    
    dalemodel_test = dict(np.load(f'/UserFolder/neur0003/third_set_models/{item}', allow_pickle=True))
    trials = dalemodel_test['trials'].reshape(-1)[0]
    stim_pref = stim_pref_(trials)
    
    stim_pref_hem1stim_hem1 = sorted(stim_pref['max_hem1stim'][0:40])
    stim_pref_hem1stim_hem2 = sorted(stim_pref['max_hem1stim'][40:80])
    stim_pref_hem2stim_hem2 = sorted(stim_pref['max_hem2stim'][40:80])
    stim_pref_hem2stim_hem1 = sorted(stim_pref['max_hem2stim'][0:40])
    
    loss = dalemodel_test['losses'][-1]
    filename = item[0:-4]
    
    params = dalemodel_test['params'].reshape(1)[0]  
    params_conv = {0.0:0.08, 0.1:0.1, 0.2:0.25, 0.5:0.5, 0.7:0.75, 1.0:1.0}
    P_in = params['P_in']
    P_rec = params['P_rec']
    N_cal = params['N_cal']
    seed = params['seed']

    mean_hem1_ipsi = np.mean(stim_pref['max_hem2stim'][0:40])
    mean_hem1_contra = np.mean(stim_pref['max_hem1stim'][0:40])
    mean_hem2_ipsi = np.mean(stim_pref['max_hem1stim'][40:80])
    mean_hem2_contra = np.mean(stim_pref['max_hem2stim'][40:80])
    
    var_hem1_ipsi = np.std(stim_pref['max_hem2stim'][0:40])
    var_hem1_contra = np.std(stim_pref['max_hem1stim'][0:40])
    var_hem2_ipsi = np.std(stim_pref['max_hem1stim'][40:80])
    var_hem2_contra = np.std(stim_pref['max_hem2stim'][40:80])
    
    nb_hem1_ipsi_pref = count_pref(stim_pref['max_hem2stim'][0:40], stim_pref['max_hem1stim'][0:40], indices=False)
    nb_hem2_ipsi_pref = count_pref(stim_pref['max_hem1stim'][40:80], stim_pref['max_hem2stim'][40:80], indices=False)
    nb_hem1_contra_pref = count_pref(stim_pref['max_hem1stim'][0:40], stim_pref['max_hem2stim'][0:40], indices=False)
    nb_hem2_contra_pref = count_pref(stim_pref['max_hem2stim'][40:80], stim_pref['max_hem1stim'][40:80], indices=False)

    total_active = nb_hem1_ipsi_pref + nb_hem2_ipsi_pref + nb_hem1_contra_pref + nb_hem2_contra_pref
    fraction_ipsi_pref = (nb_hem1_ipsi_pref+nb_hem2_ipsi_pref)/total_active
    
    new_row = {'filename':item, 'P_in':P_in, 'P_rec':P_rec, 'N_cal':N_cal, 'seed':seed, 'loss': loss,
                'mean_hem1_ipsi':mean_hem1_ipsi, 'mean_hem1_contra':mean_hem1_contra, 'mean_hem2_ipsi':mean_hem2_ipsi, 'mean_hem2_contra':mean_hem2_contra,
                'var_hem1_ipsi':var_hem1_ipsi, 'var_hem1_contra':var_hem1_contra, 'var_hem2_ipsi':var_hem2_ipsi, 'var_hem2_contra':var_hem2_contra,
                'nb_hem1_ipsi_pref':nb_hem1_ipsi_pref, 'nb_hem2_ipsi_pref':nb_hem2_ipsi_pref, 'nb_hem1_contra_pref':nb_hem1_contra_pref, 'nb_hem2_contra_pref':nb_hem2_contra_pref, 
                'total_active':total_active, 'fraction_ipsi_pref':fraction_ipsi_pref, 
                'stim_pref_hem1stim_hem1':stim_pref_hem1stim_hem1, 'stim_pref_hem1stim_hem2':stim_pref_hem1stim_hem2, 'stim_pref_hem2stim_hem2':stim_pref_hem2stim_hem2, 'stim_pref_hem2stim_hem1':stim_pref_hem2stim_hem1}
    
    third_set = third_set.append(new_row, ignore_index = True)
    
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
    # figure.savefig(f'/UserFolder/neur0003/stim_pref_second_set/{item[0:-4]}')

third_set.to_pickle('/UserFolder/neur0003/third_set_model.pkl')

#%%
#make seperate dataframe with defined number of ipsi preferring cells

model_best = first_set[first_set.columns[:]].to_numpy()
columns = first_set.columns[:]

indexes=[]
for i in range(model_best.shape[0]):
    if model_best[i][18]<=0 or model_best[i][19]<=1 or abs(model_best[i][18]-model_best[i][19]) >= 3:
        indexes.append(i)
    # elif abs(model_best[i][20]-model_best[i][21]) > 5:
    #     indexes.append(i)
    # elif model_best[i][23] > 0.4:
    #     indexes.append(i)
model_best = np.delete(model_best, indexes, axis=0)
model_best = pd.DataFrame(model_best, columns = columns)


#drop models with some kind of problem
# exclusions = np.array(['IpsiContra_In05_Rec02_Col10_s2',
#               'IpsiContra_In02_Rec07_Col40_s2',
#               'IpsiContra_In02_Rec10_Col10_s2',
#               'IpsiContra_In07_Rec07_Col10_s2',
#               'IpsiContra_In10_Rec10_Col20_s1',
#               'IpsiContra_In07_Rec01_Col20_s1'])

# index=[]
# for i in range(len(model_best['filename'])):
#     if model_best['filename'][i][:-4] in exclusions:
#         index.append(i)
# model_best = model_best.drop(index)

model_best.to_csv('UserFolder/neur0003/first_set_best.csv')

#%%
#Save trials
n = [0,40]
pourc = 100
coh = 'both'
dt = 10
results = ['x', 'y', 'model_state', 'model_output']
labels = ['Input', 'Expected Output', 'State of each Neuron', 'Output']
lims = [(), (-0.1, 1.1), (-0.1, 1.1), (-0.5, 0.5)]

for item in os.listdir('/UserFolder/neur0003/third_set_models')[36:37]:
    
    dalemodel_test = dict(np.load(f'/UserFolder/neur0003/third_set_models/{item}', allow_pickle=True))
    weights = adapt_for_opto(dalemodel_test['weights'].reshape(1)[0])
    trials = dalemodel_test['trials'].reshape(1)[0]
    stim_pref_dict = fcts.stim_pref_(trials)
    
    if coh == 'ipsi' or coh=='both':
        arr1 = stim_pref_dict['max_hem1stim'][n_range[0]:n_range[1]]
        arr2 = stim_pref_dict['max_hem2stim'][n_range[0]:n_range[1]]
    if coh == 'contra':
        arr1 = stim_pref_dict['max_hem2stim'][n_range[0]:n_range[1]]
        arr2 = stim_pref_dict['max_hem1stim'][n_range[0]:n_range[1]]
    
    indices = count_pref(arr1, arr2, indices=True)
    if coh =='both':
        indices += count_pref(arr2, arr1, indices=True)
    
    if n_range == [40,80]:
        indices = np.array(np.zeros(40), indices).flatten()
    
    indices = indices[:int(pourc/100*len(indices))]
    
    weights_modif = change_opto_stim(weights, indices)
    simulator = BasicSimulator(weights = weights_modif , params = {'dt': 10, 'tau': 100})
    trial_equal = gen_pol_trials(simulator, task_pert, [[0.4,0.2]], opto_stim=0.4, sim=True)
    
    #make a figure of the trials
    
    for i in range(len(trial_equal['hem1stim']['mask'])):
        if trial_equal['hem1stim']['mask'][i][0] == 0:
            trial_equal['hem1stim']['y'][i] =+ np.nan
    
    x_len = range(0,len(trial_equal['hem1stim']['x'])*dt,dt)
    data = {'H1':trial_equal['hem1stim']['model_state'][:,0:40], 'H2':trial_equal['hem1stim']['model_state'][:,40:80]}
    keys = list(data.keys())
    
    fig2, ax = plt.subplots(2, 3, figsize=(30,8))
    fig2.suptitle(f'Trial for: {item} with opto stim to {pourc}% {coh} cells in Hem1', fontsize=16)
    x=0
    for i in range(2):
        for j in range(2):
            ax[i,j].plot(x_len, trial_equal['hem1stim'][results[x]])
            ax[i,j].set_title(labels[x], fontsize = 14)
            x= x+1
            
        ax[i,2].plot(x_len, data[keys[i]], alpha=0.9)
        ax[i,2].set_ylim(-0.8,0.8)
        ax[i,2].set_title(f"{keys[i]}", fontsize = 14)
    
    for i in range(3):
        ax[1,i].set_xlabel("Time (ms)", fontsize = 10)
        
    ax[0,0].legend(["Input Channel 1", "Input Channel 2", 'go cue'])
    ax[1,1].set_ylim(-0.02, 1.02)
    fig2.tight_layout()
    
    #fig2.savefig(f'UserFolder/neur0003/trial_third_set/{item[0:-4]}_{l}')


#%%

#create structured np.array to produce a heat map for different variables 
columns = first_set.columns[1:]
arr_info = first_set[columns].to_numpy()

file = np.array(sorted(arr_info, key=take_first))
file = file.reshape(5, int(file.shape[0]/5) , len(columns))
for i in range(4):
    file[i] = np.array(sorted(file[i], key=take_second))
file = file.reshape(5, 6, int(file.shape[0]/5), len(columns))
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
        ax[i,j].hist(first_set[[labels[x]]])
        ax[i,j].set_xlabel(labels[x])
        x=x+1
        
# ax[0,0].set_xticks([0.08, 0.1, 0.25, 0.5, 0.75, 1])
# ax[0,1].set_xticks([0.08, 0.1, 0.25, 0.5, 0.75, 1])
# ax[1,0].set_xticks([10, 20, 30, 40])
# ax[1,1].set_xticks([0,1,2])

plt.tight_layout()

#check relation between different parameters and if different associations are more likely
#%%
#plot average fraction of ipsi preferring cells depending on P_rec and P_in

P_rec = [0.08, 0.1, 0.25, 0.5, 0.75, 1.0]
P_in = [0.1, 0.25, 0.5, 0.75, 1.0]
av_fraction = np.array([])
for i in P_rec:
    for j in P_in:
        index = []
        for k in list(first_set.index):
            if first_set['P_rec'][k] == i and first_set['P_in'][k] == j:
                index.append(k)
        frac = np.mean(first_set['fraction_ipsi_pref'][index])
        active = np.mean(first_set['total_active'][index])
        av_fraction = np.append(av_fraction, [i, j, frac, active])
av_fraction = av_fraction.reshape(30, 4)


av_fraction[:,0] = np.where(av_fraction[:,0]!=0.08, av_fraction[:,0],-0.25)
av_fraction[:,0] = np.where(av_fraction[:,0]!=0.1, av_fraction[:,0],0)
av_fraction[:,1]= np.where(av_fraction[:,1]!=0.1, av_fraction[:,1],0)


file = first_set
title = 'fraction of ipsi-preferrinf cells in models, categorised by P_in and P_rec'

columns = file.columns[1:]
mean_resp = np.mean(file[list(columns[6:10])], axis=1)
var_resp = np.mean(file[list(columns[10:14])], axis=1)
P_in_rec = file[columns[1]]*file[columns[2]]
# diff_ipsi = np.abs(file[columns[20]]-file[columns[21]])

x = np.array(file['P_rec'])
y = np.array(file['P_in'])
c = file['fraction_ipsi_pref']

x = np.where(x!=0.08, x,-0.25)
x = np.where(x!=0.1, x,0)
y= np.where(y!=0.1, y,0)

np.random.seed(1)
x= x+np.random.uniform(-0.05, 0.05, len(x))
y=y+np.random.uniform(-0.05,0.05, len(y))


norm = colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = colors.Colormap('rainbow')

figure4, ax = plt.subplots(1,1, figsize=(6,6))
figure4.suptitle(title)
ax.scatter(av_fraction[:,0], av_fraction[:,1], c=av_fraction[:,2], s=1000, norm=norm, alpha=0.8)
ax.scatter(x, y, c=c, s=30, alpha=0.9, norm=norm)

ax.set_yticks([0, .25,.5, .75,1.])
ax.set_xticks([-0.25, 0., .25,.5, .75,1.])
ax.set_yticklabels([0.1, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels([0.08, 0.1, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel('P_rec')
ax.set_ylabel('P_in')

divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
figure4.add_axes(ax_cb)
cb1 = colorbar.ColorbarBase(ax_cb, norm=norm, orientation='vertical', label=c.name)

#%%
columns = file.columns[1:]
mean_resp = np.mean(file[list(columns[6:10])], axis=1)

figure = plt.figure(figsize=(12,12))

x = {'P_in':np.array(file['P_in']), 'P_rec':np.array(file['P_rec']), 'N_cal':np.array(file['N_cal'])}
y = {'Total active neurons':np.array(file['total_active']), 'fraction_ipsi_preferring neurons':np.array(file['fraction_ipsi_pref']), 'mean_activity':mean_resp}

k=1
for j in range(3):
    for i in range(3):
        ax = plt.subplot(3,3,k)
        sns.boxplot(x[list(x.keys())[i]],y[list(y.keys())[j]], fliersize=4, width=.7, flierprops=dict(marker='o'), linewidth=2, color='white')
        k+=1
        #ax[j,0].set_ylabel(list(y.keys())[j], size=12)
        #ax[2,i].set_xlabel(list(x.keys())[i], size=12)
        
# plt.xlabel('P_in')
# plt.ylabel('fraction of ipsi preferring neurons')
# plt.ylim(-0.02,1.02)

#%%
N = 5
M = 6


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
col = {'contra':['stim_pref_hem2stim_hem2', 'stim_pref_hem2stim_hem2'], 'ipsi':['stim_pref_hem1stim_hem2', 'stim_pref_hem2stim_hem1']}
n= 40

indices = {0.08:[], 0.10:[], 0.25:[], 0.50:[], 0.75:[], 1.0:[]}
n_cals = {10:[], 20:[], 30:[], 40:[]}
for i in list(first_set.index):
    if first_set['P_rec'][i] == 0.25:
        indices[first_set['P_rec'][i]].append(i)
        n_cals[first_set['N_cal'][i]].append(i)

figure, ax = plt.subplots(1,1)
figure.suptitle('distribution of activity for a stimulus as a function of P_rec')
x = np.linspace(1, n, n)
for j in list(n_cals.keys()):
    for i in [list(col.keys())[0]]:   
        a = np.array(first_set[col[i]].values.tolist())[n_cals[j]]
        a = np.mean(a, axis=0)
        mean_resp_ord = np.mean(a, axis=0)
        sem = np.std(a, axis=0, ddof=1)/np.sqrt(np.size(a))
        ax.plot(x, mean_resp_ord,label = j)
    
ax.hlines(0, 0, n, colors='grey', linestyles='--', alpha=0.8)
ax.set_xlabel('neuron number')

plt.legend()

#%%
# ---------------------- Save and plot the weights of the network ---------------------------

weights = dalemodel_test['weights'].reshape(1)[0]

fcts.plot_weights(weights['W_rec'],  
            xlabel = 'From', 
            ylabel = 'To', cb=False, matrix='rec')

fcts.plot_weights(weights['W_in'], cb=False, matrix='in')
fcts.plot_weights(weights['W_out'], cb=False, matrix='out')

#%%
#look at fraction of neurons preferring choice 1 with and without optogenetic stimulation (target neurons can be modified)
task = oli_task.PerceptualDiscrimination(dt = 10,
                              tau = 100, 
                              T = 2500, 
                              N_batch = 100,
                              N_in = 3,
                              N_rec = 100,
                              N_out = 2) # Initialize the task object

task_pert = oli_task_perturb.PerceptualDiscrimination(dt = 10,
                              tau = 100, 
                              T = 2500, 
                              N_batch = 100,
                              N_in = 4,
                              N_rec = 100,
                              N_out = 2) # Initialize the task object

n_range = [40,80]
 
#%%
#create dictionary to compare the output of the models in response to equal stimuli for different levels of optogenetic stimulation

coherences = {'none_00':{}, 'ipsi_25':{}, 'ipsi_50':{}, 'ipsi_75':{}, 'ipsi_100':{},
              'cont_25':{}, 'cont_50':{}, 'cont_75':{}, 'cont_100':{}, 
              'both_25':{}, 'both_50':{}, 'both_75':{}, 'both_100':{}}

for coh in coherences:
    coherences[coh] = {'0.0':[], '0.2':[],'0.4':[], '0.6':[]}
    
intensity = ['0.0', '0.2','0.4', '0.6']

for item in os.listdir('/UserFolder/neur0003/third_set_models'):
    
    dalemodel_test = dict(np.load(f'/UserFolder/neur0003/third_set_models/{item}', allow_pickle=True))
    weights = fcts.adapt_for_opto(dalemodel_test['weights'].reshape(1)[0])
    trials = dalemodel_test['trials'].reshape(1)[0]
    stim_pref_dict = fcts.stim_pref_(trials)
    
    for coh in coherences:
        if coh[0:4]=='cont' or coh[0:4]=='none' or coh[0:4]=='both':
            stim1 = 'max_hem1stim'
            stim2 = 'max_hem2stim'
            
        if coh[0:4]=='ipsi':
            stim1 = 'max_hem2stim'
            stim2 = 'max_hem1stim'
        
        # network_params = task.get_task_params() # get the params passed in and defined in pd
        # network_params['N_rec'] = 100 # set the number of recurrent units in the model
        # network_params['name'] = 'basic'
        # network_params['N_in'] = 3
        # network_params['N_out'] = 2
        # network_params.update(weights)
        
        # Model = Basic(network_params)
        # x, y,mask, train_params = task.get_trial_batch() # get pd task inputs and outputs
        # model_output, model_state = Model.test(x) # run the model on input x
        # choices, diff, z = task.psychometric_curve(model_output, train_params)
        # acc = task.accuracy_function(y, model_output, mask)
        # frac_ch1.append(z)
        # accuracy.append(acc)
        # Model.destruct() 
        
        arr1 = stim_pref_dict[stim1][n_range[0]:n_range[1]]
        arr2 = stim_pref_dict[stim2][n_range[0]:n_range[1]]
        indices = count_pref(arr1, arr2, indices=True)
        if coh[0:4]=='both':
            indices += fcts.count_pref(arr2, arr1, indices=True)
            random.shuffle(indices)
            
        if n_range == [40,80]:
            indices = np.array(indices)
            indices = indices+40
        
        pourc = int(coh[5:])
        indices = indices[:int(pourc/100*len(indices))]
        indices=list(indices)
        weights_modif = fcts.change_opto_stim(weights, indices)
        print(coh, indices)
        
        simulator = BasicSimulator(weights=weights_modif , params = {'dt': 10, 'tau':100})
        
        trials = fcts.gen_pol_trials(simulator, task_pert, [[0.0, 0.0],[0.2,0.2],[0.4,0.4],[0.6,0.6]], opto_stim=0.4, sim=True)
        
        for i in range(1,5):
            coherences[coh][intensity[i-1]].append(trials[f'hem{i}stim']['model_output'][249])
            
        #acc = task_pert.accuracy_function(y, model_output, mask)
        #accuracy_opto.append(acc)

#%%
# find choice of network depending on ratio and total number of cells activated
# iterations determined by nb_trials per model, with varying number of stimulation of ipsi preferring and contra preferring cells
# for each model outputs an array (nb_trials*4) with each subarray containing in order:
# number of stimulated cells that are ipsi preferring
# number of stimulated cells
# value of output 1 at t = 2500 ms
# value of output 2 at t = 2500 ms

coherences = {}
nb_trials = 30
n_range = [0,40]
for item in os.listdir('/UserFolder/neur0003/third_set_models'):
    coherences[item] = np.array([])
    print(item)
    dalemodel_test = dict(np.load(f'/UserFolder/neur0003/third_set_models/{item}', allow_pickle=True))
    weights = fcts.adapt_for_opto(dalemodel_test['weights'].reshape(1)[0])
    trials = dalemodel_test['trials'].reshape(1)[0]
    stim_pref_dict = stim_pref_(trials)
    
    arr1 = stim_pref_dict['max_hem1stim'][n_range[0]:n_range[1]]
    arr2 = stim_pref_dict['max_hem2stim'][n_range[0]:n_range[1]]
    
    for i in range(nb_trials):
        indices_contra_pref = fcts.count_pref(arr1, arr2, indices=True)
        indices_ipsi_pref = fcts.count_pref(arr2, arr1, indices=True)
        
        if len(indices_contra_pref) != 0:
            n = np.random.choice(range(len(indices_contra_pref)+1))
        else:
            n = 0
        indices_contra_pref = random.sample(indices_contra_pref, n)
        
        if len(indices_ipsi_pref) != 0:
            n = np.random.choice(range(len(indices_ipsi_pref)+1))
        else:
            n = 0
        indices_ipsi_pref = random.sample(indices_ipsi_pref, n)
        
        indices= indices_ipsi_pref + indices_contra_pref
        weights_modif = fcts.change_opto_stim(weights, indices)
        simulator = BasicSimulator(weights=weights_modif , params = {'dt': 10, 'tau':100})
        trials = fcts.gen_pol_trials(simulator, task_pert, [[0.4,0.4]], opto_stim=0.4, sim=True)
        
        coherences[item] = np.append(coherences[item], np.array([len(indices_ipsi_pref), len(indices), trials['hem1stim']['model_output'][249][0], trials['hem1stim']['model_output'][249][1]]))
    
    coherences[item] = coherences[item].reshape(nb_trials, 4)
#%%
repla = coherences
for i in coherences:
    for j in coherences[i]:
        coherences[i][j] = np.array(coherences[i][j])

f = open("/UserFolder/neur0003/coherences_ratio.pkl","wb")
pickle.dump(coherences,f)
f.close()

#%%
coherences = pd.read_pickle('/UserFolder/neur0003/coherences_ratio.pkl')
items = list(coherences.keys())

for i in range(len(items)):
    if int(third_set[third_set['filename']==items[i]]['nb_hem1_ipsi_pref']) == 0 or int(third_set[third_set['filename']==items[i]]['nb_hem1_ipsi_pref']) == 0:
        coherences.pop(items[i])
coherences.pop('IpsiContra_In05_Rec025_Cal20_s19.npz')
#%%
total_hem1 = np.array(third_set['nb_hem1_ipsi_pref'] + third_set['nb_hem1_contra_pref'])
norm = colors.Normalize(vmin=0, vmax=total_hem1.max())

fig_ratio,ax = plt.subplots(1,1,figsize=(6,6))

for i in list(coherences.keys()):
    if i[14] == '7' or i[14]=='5':
        ratio = coherences[i][:,0]/(coherences[i][:,1]-coherences[i][:,0])
        percent = (coherences[i][:,0]/coherences[i][:,1])*100
        percent[np.isnan(percent)] = 0
        ax.scatter(ratio, coherences[i][:,2]-coherences[i][:,3], s=20, alpha=0.8)

ax.set_xlabel('ratio ipsi/contra')
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig_ratio.add_axes(ax_cb)
cb1 = colorbar.ColorbarBase(ax_cb, norm=norm, orientation='vertical', label='total activated cells')
#ax.set_xlim(-0.04,2.04)
ax.set_ylabel('<- choice 2 - choice 1 ->')
#%%
bins = np.linspace(0,1,11)

fig_ratio_means,ax = plt.subplots(1,1,figsize=(6,6))

mean_all_models = pd.DataFrame()

for i in list(coherences.keys()): 
    if i[14]=='7':
        percent = (coherences[i][:,0]/coherences[i][:,1])
        percent[np.isnan(percent)] = 0
        
        choice = coherences[i][:,2]-coherences[i][:,3]
        dig = np.digitize(percent, bins)
        bin_means = np.array([choice[dig == k].mean() for k in range(1, len(bins)+1)])
        bin_means = pd.Series(bin_means)
        mean_all_models[i] = bin_means
        bin_means = bin_means.drop(bin_means[np.isnan(bin_means)].index)
        
        ax.plot(bin_means)

ax.plot(mean_all_models.mean(axis=1), c='black', linewidth=3)
ax.set_xticks([0,2,4,6,8,10])
ax.set_xticklabels([0,20,40,60,80,100])
ax.set_xlabel('% stimulated ipsi preferring cells')
ax.set_title('dense models')

#%% 
coherences_hem1_opt = pd.read_pickle('/UserFolder/neur0003/coherences_hem1_dict.pkl')
coherences_hem2_opt = pd.read_pickle('/UserFolder/neur0003/coherences_hem1_dict.pkl')

coherences = {'hem1':coherences_hem1_opt, 'hem2':coherences_hem2_opt}

file_order = [] #15 still in there, but was removed from coherence dictionary because didn't train
for item in os.listdir('/UserFolder/neur0003/third_set_models'):
    file_order.append(item)
    
coh_averages = {'hem1':np.array([]), 'hem2':np.array([])}
for k in coherences:
    for i in coherences[k]:
        for j in coherences[k][i]:
            coh_averages[k] = np.append(coh_averages[k], np.mean(coherences[k][i][j], axis=0))
    coh_averages[k] = coh_averages[k].reshape(13, 4, 2)    

#%%

Hems = {'hem1':0, 'hem2':1}
Stims = {'ipsi':0, 'contra':1, 'both':2}

ind = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12]]
plot_data = np.ones((2,3,5,4))
for k in range(len(coherences)):
    for i in range(3):
        plot_data[:,i,:,0] = coh_averages[hems[k]][ind[i],0,0]
        plot_data[:,i,:,1] = np.mean(coh_averages[hems[k]][ind[i],1:,0], axis=1)
        plot_data[:,i,:,2] = coh_averages[hems[k]][ind[i],0,1]
        plot_data[:,i,:,3] = np.mean(coh_averages[hems[k]][ind[i],1:,1], axis=1)
        

stim = 'ipsi'
hem = 'hem1'

labels = ['hem1 no input', 'hem1 input', 'hem2 no input', 'hem2 input']
for i in range(4):
    plt.plot(plot_data[Hems[hem], Stims[stim],:,i], marker='o', label=labels[i])
plt.ylim(-0.05, 1)
plt.xticks([0,1,2,3,4], labels=[0,25,50,75,100])
plt.xlabel('% stimulated cells')
plt.title(f'average output for opto stim of {stim} cells in {hem}')
plt.legend()

#%%
coherence_array = coherences
for i in coherence_array:
    for j in coherence_array[i]:
        coherence_array[i][j] = np.array(list(coherence_array[i][j].values()))
for i in coherence_array:
    coherence_array[i] = np.array(list(coherence_array[i].values()))
coherence_array = np.array(list(coherence_array.values()))

#%%
#relative amplitude for a single model depending on the target cell population

model_nb = 36
hem_stim = 'hem1'
hem_record = 'hem1'
labels = list(coherences['hem1']['none_00'].keys())

for i in range(4):
    plt.plot(coherence_array[Hems[hem_stim],:,i,model_nb,Hems[hem_record]], marker='o', label=labels[i])

plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12], ['n', 'i25', 'i50', 'i75', 'i100', 'c25', 'c50', 'c75', 'c100', 'b25', 'b50', 'b75', 'b100'])
plt.title(f'{file_order[model_nb]}')
plt.legend()

#%%
#mean for all models with stim and record hemisphere

hem_stim = 'hem1'
hem_record = 'hem1'

labels = list(coherences['hem1']['none_00'].keys())
data=np.mean(coherences_array_75[Hems[hem_stim],:,:,:,Hems[hem_record]], axis=2)
for i in range(4):
    plt.plot(data[:,i], marker='o', label=labels[i])
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12], ['n', 'i25', 'i50', 'i75', 'i100', 'c25', 'c50', 'c75', 'c100', 'b25', 'b50', 'b75', 'b100'])
plt.title(f'mean for sparse model')
plt.legend()

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
