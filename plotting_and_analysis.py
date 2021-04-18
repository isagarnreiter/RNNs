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
from fcts import count_pref
import seaborn as sns
from pylab import text
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


#%% 
#produce dataframe of with info about all models

first_set = pd.DataFrame(columns = ['filename', 'P_in', 'P_rec', 'N_cal', 'seed', 'loss',
                                      'mean_hem1_ipsi', 'mean_hem1_contra', 'mean_hem2_ipsi', 'mean_hem2_contra',
                                      'var_hem1_ipsi', 'var_hem1_contra', 'var_hem2_ipsi', 'var_hem2_contra',
                                      'nb_hem1_ipsi_pref', 'nb_hem2_ipsi_pref', 'nb_hem1_contra_pref', 'nb_hem2_contra_pref',
                                      'total_active', 'fraction_ipsi_pref', 'stim_pref_hem1stim', 'stim_pref_hem2stim'])

for item in os.listdir('/UserFolder/neur0003/first_set_models'):
    
    dalemodel_test = dict(np.load(f'/UserFolder/neur0003/first_set_models/{item}', allow_pickle=True))
    trials = dalemodel_test['arr_0'][1]
    stim_pref = fcts.stim_pref(trials)    
    stim_pref_hem1stim = sorted(stim_pref['max_hem1stim'][0:80])
    stim_pref_hem2stim = sorted(stim_pref['max_hem2stim'][0:80])
    loss = dalemodel_test['arr_0'][0]['losses'][-1]
    #params = dalemodel_test['params'].reshape(1)[0]    
    filename = item[0:-4]
    
    params_conv = {0.0:0.08, 0.1:0.1, 0.2:0.25, 0.5:0.5, 0.7:0.75, 1.0:1.0}
    P_in = params_conv[round(float(item[13]),2) + round(float(item[14])*(.1), 2)]
    P_rec = params_conv[round(float(item[19]),2) + round(float(item[20])*(.1), 2)]
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
                'stim_pref_hem1stim':stim_pref_hem1stim, 'stim_pref_hem2stim':stim_pref_hem2stim}
    
    first_set = first_set.append(new_row, ignore_index = True)
    
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

first_set.to_pickle('/UserFolder/neur0003/first_set_model.pkl')

#first_set.to_csv('/UserFolder/neur0003/first_set_model.csv')

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

#add all figures generated for stim preference to a seperate folder
for item in os.listdir('/UserFolder/neur0003/trials'):
    if f'{item[:-13]}.npz' in np.array(model_best[['filename']])[:,0]:
        newPath = shutil.copy(f'/UserFolder/neur0003/trials/{item}', '/UserFolder/neur0003/trials_select')


#%%
#save weights
for item in os.listdir('outputs'):
    dalemodel_test = dict(np.load(f'outputs/{item}', allow_pickle=True))
    weights = dalemodel_test['weights'].reshape(1)[0]
    np.savez(f'weights\{item}', **weights)

#%%

dt = 10
results = ['x', 'y', 'model_state', 'model_output']
labels = ['Input', 'Expected Output', 'State of each Neuron', 'Output']
lims = [(), (-0.1, 1.1), (-0.1, 1.1), (-0.5, 0.5)]

for item in os.listdir('UserFolder/neur0003/second_set_models'):
    
    dalemodel_test = dict(np.load(f'/UserFolder/neur0003/second_set_models/{item}', allow_pickle=True))
    trials = dalemodel_test['trials'].reshape(1)[0]
    for l in list(trials.keys()):
    #make a figure of the trials
    
        for i in range(len(trials[l]['mask'])):
            if trials[l]['mask'][i][0] == 0:
                trials[l]['y'][i] =+ np.nan
        
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
        fig2.savefig(f'UserFolder/neur0003/trials_second_set/{item[0:-4]}_{l}')
    

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
mean_resp = np.mean(file[list(columns[6:10])], axis=1)
var_resp = np.mean(file[list(columns[10:14])], axis=1)
P_in_rec = file[columns[1]]*file[columns[2]]
diff_ipsi = np.abs(file[columns[20]]-file[columns[21]])

file = first_set
title = 'fraction of ipsi-preferring cells as a function of N_cal'

x = file['seed']
y = file['total_active']
c = file['P_rec']


cmap = sns.color_palette("viridis",  as_cmap=True )
norm = colors.Normalize(vmin=c.min(), vmax=c.max())
colours = {}
for cval in c:
    colours.update({cval : cmap(norm(cval))})


figure4, ax = plt.subplots(1,1, figsize=(6,6))
figure4.suptitle(title)
#ax = sns.swarmplot(x, y, size=5, hue=c, palette=colours)
ax = sns.boxplot(x, y)

plt.gca().legend_.remove()

divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
figure4.add_axes(ax_cb)
cb1 = colorbar.ColorbarBase(ax_cb, cmap=cmap,
                                norm=norm,
                                orientation='vertical', label='total active')




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

indices = []
for i in range(len(first_set['P_in'])):
    if first_set['P_rec'][i] == 0.25 :
        indices.append(i)

mean_resp_ord = np.mean(np.array(first_set['stim_pref_hem1stim'][indices].values.tolist()), axis=0)

plt.plot(mean_resp_ord)
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
