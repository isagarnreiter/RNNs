# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:13:02 2021

@author: Isabelle
"""

from sensory_discrimination_task import SensoryDiscrimination
from matplotlib import pyplot as plt
from psychrnn.backend.simulation import BasicSimulator
from psychrnn.backend.models.basic import Basic
import numpy as np
from scipy.stats import lognorm, norm, gaussian_kde, stats
import os
from matplotlib import cm, colors, colorbar, markers
import pandas as pd
import shutil
import csv
import fcts
import random
import seaborn as sns
from pylab import text
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
%matplotlib inline
from sklearn.linear_model import LinearRegression
from scipy.io import loadmat
import researchpy as rp

#%%
#load dataframes

first_set = pd.read_pickle('/UserFolder/neur0003/first_set_models.pkl')
third_set = pd.read_pickle('/UserFolder/neur0003/third_set_model.pkl')
coherences = pd.read_pickle('/UserFolder/neur0003/coherences_ratio.pkl')

#%%
sd = SensoryDiscrimination(dt = 10,
                              tau = 100, 
                              T = 2500, 
                              N_batch = 100,
                              N_rec = 100,
                              N_out = 2,
                              opto=0.0) # Initialize the task object

sd_opto = SensoryDiscrimination(dt = 10,
                              tau = 100, 
                              T = 2500, 
                              N_batch = 100,
                              N_rec = 100,
                              N_out = 2,
                              opto=0.4) # Initialize the task object

#%%
#plot of mean loss (+/- SEM) of multiple models
losses = []
for item in os.listdir('models'):
    model = dict(np.load(f'models/{item}', allow_pickle=True))
    loss = model['losses']
    losses.append(loss)

fig=plt.figure(figsize=(4,3))
losses = np.array(losses)
losses = np.array([np.mean(losses, axis=0), np.std(losses, axis=0)])
losses_upper = losses[0]+losses[1]
losses_lower = losses[0]-losses[1]
x = np.linspace(500, 99500, 199)
plt.plot(x, losses[0], color='dimgrey')
plt.fill_between(x, losses_upper, losses_lower, alpha=0.4, color='grey')
plt.xlabel('training iteration')
plt.ylabel('loss')

#%%

#get info to make the psychometric curve and choice matrix

input_choice = np.array([])
psychometric = np.array([], dtype='float64')
for item in os.listdir('models'):
    model = dict(np.load(f'models/{item}', allow_pickle=True))
    weights = model['weights'].reshape(1)[0]
    simulator = BasicSimulator(weights = weights , params = {'dt': 10, 'tau': 100, 'rec_noise':0.02})
    x, y, mask, params = task_pert.get_trial_batch()
    outputs, states = simulator.run_trials(x)
    bin_means, bins = task_pert.psychometric_curve(outputs, params)
    psychometric = np.append(psychometric, bin_means)
    
    output = np.greater(outputs[:,249,1], outputs[:,249,0])
    choice = np.ones(100)
    choice[output] = 2
    
    ints = np.array([[params[i]['intensity_0'], params[i]['intensity_1']] for i in range(100)])
    choice = np.column_stack((ints, choice))
    input_choice = np.append(input_choice, choice)

input_choice = input_choice.reshape(800,3)
psychometric = np.array([np.mean(psychometric, axis=0), np.std(psychometric, axis=0)])

#%%

#psychometric curve

data = loadmat('C:/Users/Isabelle/Documents/GitHub/RNNs/delay_task_symmetric_psychometric_data.mat')
trial_code, trial_choice = data['for_brendan']['trial_code'][0][0][0], data['for_brendan']['trial_choice'][0][0][0]
bins = np.array([])

for i in range(len(trial_code)):
    for k in range(1,6):
        frac = trial_choice[i][0][trial_code[i][0] == k]
        frac = np.count_nonzero(frac==2)/len(frac)
        bins = np.append(bins, frac)
    
bins = bins.reshape(30,5)
bins_mean = np.array([np.mean(bins, axis=0), np.std(bins, axis=0)])

x_model = np.linspace(-100, 100, 7)
x_mouse = np.array([-100, -50, 0, 50, 100])

fig=plt.figure(figsize=(4,3.5))
ax = plt.subplot(1,1,1)
ax.errorbar(x_mouse, bins_mean[0]*100, bins_mean[1]*100, color='black', marker='o', label='mouse')
ax.plot(x_model, psychometric[0], color='blue', marker='o', label = 'RNN')
ax.fill_between(x_model, psychometric[0]+psychometric[1], psychometric[0]-psychometric[1], alpha=0.3, color='blue')
ax.legend(frameon=False)
ax.set_xlabel('input difference')
ax.set_ylabel('% choice 1')

#%%

#choice matrix
#plot on x axis input 2 and y axis input 1, with color coded according to choice probability

inputs = [0.6, 0.4, 0.2, 0.0]
mean_input_choice = []
for i in inputs:
    a = input_choice[input_choice[:,0]==i]
    for j in [0.0, 0.2, 0.4, 0.6]:
        b = a[a[:,1]==j]
        pour = b.shape[0]/input_choice.shape[0]
        b = np.count_nonzero(b[:,2]==1)/len(b[:,2])
        b = [i,j,b,pour]
        mean_input_choice.append(b)

mean_input_choice = np.array(mean_input_choice).reshape(4,4,4)

plt.imshow(mean_input_choice[:,:,2], cmap=cm.get_cmap('RdBu'), vmin=0, vmax=1)
plt.yticks([0,1,2,3], labels=inputs)
plt.xticks([0,1,2,3], labels=[0.0,0.2,0.4,0.6])
plt.xlabel('input 2')
plt.ylabel('input 1')
plt.colorbar(plt.imshow(mean_input_choice[:,:,2],cmap=cm.get_cmap('RdBu'), vmin=0, vmax=1))

#%%
#boxplots for different parameters: 
# on x axis: P_in, P_rec, P_out
# on y axis: Total active neurons, fraction of ipsi-preferring cells and mean activity of the network

file= first_set

columns = file.columns[1:]
mean_resp = np.mean(file[list(columns[6:10])], axis=1)

figure = plt.figure(figsize=(12,12))

x = {'P_in':np.array(file['P_in']), 
     'P_rec':np.array(file['P_rec']), 
     'N_cal':np.array(file['N_cal'])}

y = {'Total active neurons':np.array(file['total_active']), 
     'fraction_ipsi_preferring neurons':np.array(file['fraction_ipsi_pref']), 
     'mean_activity':mean_resp}

k=1
for j in range(3):
    for i in range(3):
        ax = plt.subplot(3,3,k)
        sns.boxplot(x[list(x.keys())[i]],y[list(y.keys())[j]], fliersize=4, width=.7, flierprops=dict(marker='o'), linewidth=2, color='white')
        k+=1

stats.f_oneway(file['total_active'][file['N_cal'] == 10],
               file['total_active'][file['N_cal'] == 20], 
               file['total_active'][file['N_cal'] == 30],
               file['total_active'][file['N_cal'] == 40])

#%%
#boxplot of the number of active cells and the fraction of ipsi preferring cells depending on sparse of dense networks.
#uses as input the info. file (as pickle)

figure = plt.figure(figsize=(5,3))
ax1 = plt.subplot(121)
sns.boxplot(third_set['P_rec'], third_set['total_active'], fliersize=4, width=.7, flierprops=dict(marker='o'), linewidth=2, color='white')
ax2 = plt.subplot(122)
sns.boxplot(third_set['P_rec'], third_set['fraction_ipsi_pref'], fliersize=4, width=.7, flierprops=dict(marker='o'), linewidth=2, color='white')
plt.tight_layout()

ax1.set_xticklabels(['sparse', 'dense'])
ax2.set_xticklabels(['sparse', 'dense'])

#%%
# get the accuracy of models 
Path = 'UserFolder/neur0003/third_set_models'
n = [0,40]
state = pd.DataFrame(columns=['contra_I1', 'contra_I2', 'ipsi_I1', 'ipsi_I2', 'nb_ipsi', 'nb_contra'])

for item in os.listdir(Path):
    
    dalemodel_test = dict(np.load(f'{Path}/{item}', allow_pickle=True))
    if list(dalemodel_test.keys())[0] == 'arr_0':
        dalemodel_test = dalemodel_test['arr_0'].reshape(-1)[0]

    weights = dalemodel_test['weights'].reshape(-1)[0]        
    weights = fcts.adapt_for_opto(weights)
    simulator = BasicSimulator(weights = weights , params = {'dt': 10, 'tau': 100})
    trials = fcts.gen_pol_trials(simulator, task_pert, [[0.6,0.0], [0.0, 0.6]],sim=True)
    stim_pref = fcts.stim_pref_(trials)
    
    hem1 = stim_pref['max_hem1stim'][n_range[0]:n_range[1]]
    hem2 = stim_pref['max_hem2stim'][n_range[0]:n_range[1]]

    indices_ipsi = fcts.count_pref(hem2, hem1, indices=True) 
    indices_contra = fcts.count_pref(hem1, hem2, indices=True) 
    
    new_row = {'contra_I1':np.mean(stim_pref['max_hem1stim'][indices_contra]),
               'contra_I2':np.mean(stim_pref['max_hem2stim'][indices_contra]),
               'ipsi_I1':np.mean(stim_pref['max_hem1stim'][indices_ipsi]),
               'ipsi_I2':np.mean(stim_pref['max_hem2stim'][indices_ipsi]),
               'nb_ipsi': len(indices_ipsi), 'nb_contra':len(indices_contra)}
    
    state = state.append(new_row, ignore_index = True)


#states[coh] = np.array(list(states[coh].values()))
#%%
#boxplot of the ipsi and contra preferring cells

fi, ax = plt.subplots(1,1, figsize=(4,4))


a = ax.boxplot(state.loc[[1,2,3,5,6,7],['contra_I1', 'contra_I2']], labels = ['',''])
b = ax.boxplot(state.loc[[1,2,3,5,6,7],['ipsi_I1', 'ipsi_I2']], labels = ['I1>I2', 'I2>I1'])
for i in ['boxes', 'caps', 'medians', 'whiskers', 'fliers']:
    for j in a[i]:
        j.set(color='C0')
    for j in b[i]:
        j.set(color='C1')

for j in a['fliers']:
    j.set(markeredgecolor='C0')
for j in b['fliers']:
    j.set(markeredgecolor='C1')
    
plt.ylim(-0.21, 0.21)
plt.yticks([0.2, 0.1, 0, -0.1, -0.2])
plt.ylabel('state')

#t_stat, p = stats.wilcoxon(state.loc[[1,2,3,5,6,7],['contra_I1']].values.reshape(6), state.loc[[1,2,3,5,6,7],['ipsi_I2']].values.reshape(6))

#%%
#Save trial / visualise trial
n_range = [0,40]
coh = 'ipsi'
s1 = 0.2
s2 = 0.6
model = 'IpsiContra_In05_Rec025_Cal20_s0.npz'

dalemodel_test = dict(np.load(f'/UserFolder/neur0003/third_set_models/{model}', allow_pickle=True))
if list(dalemodel_test.keys())[0] == 'arr_0':
    dalemodel_test = dalemodel_test['arr_0'].reshape(-1)[0]
weights = fcts.adapt_for_opto(dalemodel_test['weights'].reshape(1)[0])
trials = dalemodel_test['trials'].reshape(1)[0]
stim_pref_dict = fcts.stim_pref_(trials)

arr1 = stim_pref_dict['max_hem1stim'][n_range[0]:n_range[1]]
arr2 = stim_pref_dict['max_hem2stim'][n_range[0]:n_range[1]]

if coh == 'ipsi' or coh=='both':
    indices = fcts.count_pref(arr1, arr2, indices=True)
if coh == 'contra':
    indices = fcts.count_pref(arr2, arr1, indices=True)
if coh =='both':
    indices += fcts.count_pref(arr2, arr1, indices=True)

weights_modif = fcts.change_opto_stim(weights, indices)
simulator = BasicSimulator(weights = weights_modif , params = {'dt': 10, 'tau': 100})
trial = fcts.gen_pol_trials(simulator, task_pert, [[s1,s2]], sim=True)

#make a figure of the trials

fig = fcts.visualise_trial(trial)
    
#fig2.savefig(f'UserFolder/neur0003/trial_third_set/{item[0:-4]}_{l}')

#%%
#plot average fraction of ipsi preferring cells depending on P_rec and P_in
#issue with the colormap

file = first_set
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
# plot activity with and without stimulation
# would need to run above code twice - once with and wihtout stimulation to be able to plot this
states_opto_arr = np.array(list(states_opto.values())).reshape(4000,250)
states_ctrl_arr = np.array(list(states_ctrl.values())).reshape(4000,250)

states_opto_arr = np.array([np.mean(states_opto_arr, axis=0), np.std(states_opto_arr, axis=0)/np.sqrt(40)])
states_ctrl_arr = np.array([np.mean(states_ctrl_arr, axis=0), np.std(states_ctrl_arr, axis=0)/np.sqrt(40)])
x = np.linspace(0,990,100)
plt.plot(x, states_opto_arr[0][0:100], color='red')
plt.plot(x, states_ctrl_arr[0][0:100], color='grey')
plt.fill_between(x, states_ctrl_arr[0][0:100]-states_ctrl_arr[1][0:100], states_ctrl_arr[0][0:100]+states_ctrl_arr[1][0:100], alpha=0.4, color='grey')
plt.fill_between(x, states_opto_arr[0][0:100]-states_opto_arr[1][0:100], states_opto_arr[0][0:100]+states_opto_arr[1][0:100], alpha=0.4, color='red')

plt.ylim(-0.12, 0.12)
plt.yticks([-0.1, -0.05, 0, 0.05, 0.1])
plt.xlabel('time (ms)')

probs_opto = np.mean(states_opto_arr.reshape(40, 100, 250), axis=1)[:,50]
probs_ctrl = np.mean(states_ctrl_arr.reshape(40, 100, 250), axis=1)[:,50]
stat, p = stats.ttest_ind(probs_opto, probs_ctrl)


#%%

#code for %choice ipsi vs %stimulated ipsi cells
#shows the average for individual models as colored lines and the average for all models as thicker black line

bins = np.linspace(0,1,11)

fig_ratio_means,ax = plt.subplots(1,1,figsize=(6,6))

mean_all_models = pd.DataFrame()

for i in list(coherences.keys()): 
    if i[14]=='5':
        percent = (coherences[i][:,0]/coherences[i][:,1])
        percent[np.isnan(percent)] = 0
        
        choice = np.array([])
        for j in coherences[i]:
            if j[2] > j[3]:
                choice = np.append(choice, 1)
            if j[3] > j[2]:
                choice = np.append(choice,2)
        
        bin_means = np.array([])
        dig = np.digitize(percent, bins)
        for k in range(1, len(bins)+1):
            ch = choice[dig == k]
            if len(ch) != 0:
                ch = np.count_nonzero(ch==2)/len(ch)
            else:
                ch = np.nan
            bin_means = np.append(bin_means, ch)
            
        bin_means = pd.Series(bin_means)
        mean_all_models[i] = bin_means
        bin_means = bin_means.drop(bin_means[np.isnan(bin_means)].index)

        ax.plot(bin_means)

ax.plot(mean_all_models.mean(axis=1), c='black', linewidth=3)
ax.set_xticks([0,2,4,6,8,10])
ax.set_xticklabels([0,20,40,60,80,100])
ax.set_xlabel('% stimulated ipsi preferring cells')
ax.set_title('dense models')
ax.set_ylabel('% ipsi choice')

#%%
# plot on x axis nb of ipsi pref cells stimulated (normalised to the total nb of ipsi pref cells in the model)
# plot on y axis nb of contra pref cells stimulated (normalised to the total nb of contra pref cells in the model)
# color code for choice preference
#fig type is either 'nb/ratio' or 'ipsi/contra'

coherences_copy = coherences.copy()
key = list(coherences_copy.keys())
coherence_arr = np.array([])

model = [['sparse','5'], ['dense','7']]
fig_type = 'nb/ratio'
k=0
for i in range(len(key)):
    if key[i][14] == model[k][1]:
        total_active = int(third_set[third_set['filename']==key[i]]['nb_hem1_ipsi_pref'])+int(third_set[third_set['filename']==key[i]]['nb_hem1_contra_pref'])
        ipsi = np.round(coherences[key[i]][:,0]/total_active, decimals=1)
        contra = np.round((coherences_copy[key[i]][:,1]-coherences_copy[key[i]][:,0])/total_active, decimals=1)
        total_stim = np.round(coherences_copy[key[i]][:,1]/total_active, decimals = 1)
        ratio_ipsi_contra = np.round(np.log(coherences[key[i]][:,0]/(coherences_copy[key[i]][:,1]-coherences_copy[key[i]][:,0])), decimals=1)

        choice = coherences_copy[key[i]][:,2]-coherences_copy[key[i]][:,3]
        if fig_type == 'nb/ratio':
            coherence_arr = np.append(coherence_arr, np.array([total_stim, ratio_ipsi_contra, choice]))
        elif fig_type == 'ipsi/contra':
            coherence_arr = np.append(coherence_arr, np.array([ipsi, contra, choice]))
    
coherence_arr = np.transpose(coherence_arr.reshape(20,3,30), (0,2,1))
coherence_arr = coherence_arr.reshape(coherence_arr.shape[0]*30,3)

mean_coh = []
if fig_type == 'nb/ratio':  
    x = np.round(np.linspace(1,0.1,10),2)
    y = np.round(np.linspace(np.log(0.1),np.log(10),47),1)
elif fig_type == 'ipsi/contra':
    x = np.round(np.linspace(1,0,11),1)
    y = np.round(np.linspace(0,1,11),1)
for i in x:
    a = coherence_arr[coherence_arr[:,0]==i]
    for j in y:
        b = a[a[:,1]==j]
        pour = b.shape[0]/coherence_arr.shape[0]
        if len(b) != 0:
            #b = np.count_nonzero(b[:,2]==2)/len(b[:,2])
            b = np.mean(b[:,2])
        else:
            b = np.nan
        b = [i,j,b,pour]
        mean_coh.append(b)

mean_coh = np.array(mean_coh)
if fig_type == 'ipsi/contra':
    mean_coh = mean_coh.reshape(11,11,4)
    plt.imshow(mean_coh[:,:,2], cmap=cm.get_cmap('RdBu'), vmin=-1, vmax=1)
    plt.yticks([0,1,2,3,4,5,6,7,8,9,10], labels=x)
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10], labels=y)
    plt.xlabel('contra')
    plt.ylabel('ipsi')
    plt.title(model[k][0])
    plt.colorbar(plt.imshow(mean_coh[:,:,2],cmap=cm.get_cmap('RdBu'), vmin=-1, vmax=1))

elif fig_type == 'nb/ratio':
    fig = plt.figure(figsize=(6,4))
    mean_coh = mean_coh.reshape(10,47,4)
    plt.imshow(mean_coh[:,:,2], cmap=cm.get_cmap('RdBu'), vmin=-1, vmax=1, aspect='auto')
    plt.xticks([0,5,10,15,20,25,30,35,40,45], labels=y[[0,5,10,15,20,25,30,35,40,45]])
    plt.yticks([0,1,2,3,4,5,6,7,8,9], labels=x)
    plt.xlabel('ln(ipsi/contra)')
    plt.ylabel('total stimulated')
    plt.title(model[k][0])
    plt.colorbar(plt.imshow(mean_coh[:,:,2], cmap=cm.get_cmap('RdBu'), vmin=-1, vmax=1, aspect='auto'))
    
#%%
#choice matrix for ipsi/contra stim corrected for the control matrix 

cq = mean_weak[:,:,2] - mean_control[:,:,2]
plt.imshow(cq, cmap=cm.get_cmap('RdBu'), vmin=-1, vmax=1)
plt.yticks([0,1,2,3,4,5,6,7,8,9,10], labels=y)
plt.xticks([0,1,2,3,4,5,6,7,8,9,10], labels=x)
plt.xlabel('contra')
plt.ylabel('ipsi')
plt.title('strong stim-control for dense network')

#%% 
# stimulate all random amount of cells, all but Ipsi or contra preferring cells and see effect on choice.
#issue here

key = list(coherences.keys())
coherence_arr = np.array([])
model = [['sparse','5'], ['dense','7']]
k=1
for i in range(len(key)):
    if key[i][14] == model[k][1]:
        total = 40-int(third_set[third_set['filename']==key[i]]['nb_hem1_ipsi_pref'])-int(third_set[third_set['filename']==key[i]]['nb_hem1_contra_pref'])
        #total = 40
        activated = coherences[key[i]][:,1]
        choice = coherences[key[i]][:,2]-coherences[key[i]][:,3]
        choice[choice>1.0] = 1
        choice[choice<-1.0] = -1

        coherence_arr = np.append(coherence_arr, np.array([activated, choice]))

coherence_arr = np.transpose(coherence_arr.reshape(20,2, 30), (0,2,1))
#coherence_arr = coherence_arr.reshape(coherence_arr.shape[0]*200,2)
bins = np.linspace(0,1,11)
digitized = np.digitize(coherence_arr[:,:,1], bins)
meancoh = []
for i in range(coherence_arr.shape[0]):
    m = np.array([np.mean(coherence_arr[i,:,1][digitized[i] == k]) for k in range(1, len(bins)+1)])
    meancoh.append(m)

meancoh=np.array(meancoh)
mean = np.nanmean(meancoh, axis=0)
bins=bins.reshape(-1,1)
x = np.repeat(bins, 20).reshape(20,11).T
x_regr = x.reshape(20*11).reshape(-1,1)
y_regr = meancoh.reshape(20*11)
x_regr = x_regr[np.isnan(y_regr) == False]
y_regr = y_regr[np.isnan(y_regr) == False]

Regr = LinearRegression()
Regr.fit(bins, mean)
Regr = LinearRegression().fit(bins, mean)
r_sq = Regr.score(bins, mean)
print(r_sq)
print('intercept:', Regr.intercept_)
print('slope:', Regr.coef_)
y_pred = Regr.predict(bins)

plt.scatter(x, meancoh,c=x.T, cmap='rainbow', alpha=0.5)
plt.scatter(bins, mean, c='black', marker='x', s=20)
plt.plot(bins, y_pred, c='black',label = f'{round(Regr.coef_[0],3)}x{round(Regr.intercept_, 3)}')

plt.xlabel('number of cells stimulated (norm)')
plt.ylabel('output difference')
plt.ylim(-1.1, 1.1)
#plt.legend()
plt.title(f'stimulating inactive cells in {model[k][0]} networks')
#%%
# plot the weight distribution for the sparse model and the dense model

weight_distrib = {}
for item in os.listdir('/UserFolder/neur0003/third_set_models'):
    dalemodel_test = dict(np.load(f'/UserFolder/neur0003/third_set_models/{item}', allow_pickle=True))
    weigths_pre = dalemodel_test['weights'].reshape(1)[0]
    weight_distrib[item] = weigths_pre['W_rec']


weight_distrib_arr = [np.array([]),np.array([])]
for i in list(weight_distrib.keys()):
    if i[14]=='5':
        weight_distrib_arr[0] = np.append(weight_distrib_arr[0], weight_distrib[i])
    elif i[14]=='7':
        weight_distrib_arr[1] = np.append(weight_distrib_arr[1], weight_distrib[i])

weight_distrib_arr = -np.sort(np.array(weight_distrib_arr).reshape(2,20,10000),axis=2)

weight_distrib_arr_mean = np.mean(weight_distrib_arr, axis=1)
weight_distrib_arr_std_lower = weight_distrib_arr_mean-np.std(weight_distrib_arr, axis=1)
weight_distrib_arr_std_upper = weight_distrib_arr_mean+np.std(weight_distrib_arr, axis=1)

fig,ax = plt.subplots(1,2, figsize=(8,3))
nz = weight_distrib_arr_mean[0]>0
ax[0].hist(weight_distrib_arr_mean[0][weight_distrib_arr_mean[0]>=0], color='grey')
ax[0].set_title('sparse')

nz = weight_distrib_arr_mean[1]>0
ax[1].hist(weight_distrib_arr_mean[1][weight_distrib_arr_mean[1]>=0], color='grey')
ax[1].set_title('dense')

