# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:33:37 2021

@author: Isabelle
"""
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
data = loadmat('delay_task_symmetric_psychometric_data.mat')
trial_code, trial_choice = data['for_brendan']['trial_code'][0][0][0], data['for_brendan']['trial_choice'][0][0][0]
bins = np.array([])

for i in range(len(trial_code)):
    for k in range(1,6):
        frac = trial_choice[i][0][trial_code[i][0] == k]
        frac = np.count_nonzero(frac==2)/len(frac)
        bins = np.append(bins, frac)
    
bins = bins.reshape(30,5)
bins_mean = np.array([np.mean(bins, axis=0), np.std(bins, axis=0)])
x = np.array([-100, -50, 0, 50, 100])
bins_mean = np.array(bins_mean)
bins_mean_upper = bins_mean[0]+bins_mean[1]
bins_mean_lower = bins_mean[0]-bins_mean[1]

fig=plt.figure(figsize=(4,3.5))
ax = plt.subplot(1,1,1)
ax.errorbar(x, bins_mean[0], bins_mean[1], color='black', marker='o')

ax.set_xlabel('coherence')
ax.set_ylabel('% choice 1')
ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
ax.set_yticklabels([0,20,40,60,80,100])
