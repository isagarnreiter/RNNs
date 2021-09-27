# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:11:20 2021

@author: Isabelle
"""
from psychrnn.backend.simulation import BasicSimulator
from sensory_discrimination_task import SensoryDiscrimination
import numpy as np
import random
import fcts
import os

# opto = float(sys.argv[1])
# path = float(sys.argv[2])
# nb_trial = float(sys.argv[3])
# opto_target = float(sys.argv[4])
# stimuli = float(sys.argv[5])
# states = float(sys.argv[6])

opto = 0.0
path = '/UserFolder/neur0003/third_set_models'
nb_trials = 1
opto_target = 'none'
stimuli = [0.2, 0.2]
states = True


sd = SensoryDiscrimination(dt = 10,
                              tau = 100, 
                              T = 2500, 
                              N_batch = 50,
                              N_rec=100,
                              N_out=2,
                              opto= opto)# Initialize the task object

network_params = sd.get_task_params() # get the params passed in and defined in sd
network_params['name'] = 'dale_network'
network_params['dale_ratio'] = 0.8 #define the ratio of excitatory to inhibitory neurons
network_params['rec_noise'] = 0.02 #define the level of noise within the recurrent network


for item in os.listdir(path):
    
    coherences = np.array([])
    model = dict(np.load(f'{path}/{item}', allow_pickle=True))
    weights = model['weights'].reshape(1)[0]
    
    if opto != 0:
        trials = model['trials'].reshape(1)[0]
        stim_pref_dict = fcts.stim_pref(trials)
        
        arr1 = stim_pref_dict['max_hem1stim'][0:40]
        arr2 = stim_pref_dict['max_hem2stim'][0:40]
        
        weights = fcts.adapt_for_opto(weights)
        
        indices_contra_pref = fcts.count_pref(arr1, arr2, indices=True)
        indices_ipsi_pref = fcts.count_pref(arr2, arr1, indices=True)
        
    for i in range(nb_trials):
    
        if len(indices_contra_pref) != 0:
            n = np.random.choice(range(len(indices_contra_pref)+1))
        else:
            n=0
            #raise ValueError(f'no contra preferring cells in model {item}')
            
        if len(indices_ipsi_pref) != 0:
            n = np.random.choice(range(len(indices_ipsi_pref)+1))
        else:
            n=0
            #raise ValueError(f'no ipsi preferring cells in model {item}')
        
        
        if opto_target == 'random':
            indices = np.linspace(0,39,40).astype(int)
            n = np.random.choice(range(40))
            indices = random.sample(list(indices), np.random.choice(range(40)))
            weights = fcts.change_opto_stim(weights, indices)
            
        elif opto_target == 'ipsicontra':
            indices_contra_pref = random.sample(list(indices_contra_pref), np.random.choice(range(len(indices_contra_pref))))
            indices_ipsi_pref = random.sample(list(indices_ipsi_pref), np.random.choice(range(len(indices_ipsi_pref))))
            indices = indices_contra_pref + indices_ipsi_pref
            weights = fcts.change_opto_stim(weights, indices)
        

        simulator = BasicSimulator(weights=weights , params = {'dt': 10, 'tau':100, 'rec_noise':0.02})
        trial = fcts.gen_pol_trials(simulator, sd, [stimuli], sim=True)
        
        if states == True:
            coherences = np.append(coherences, trial[f'{stimuli[0]}_{stimuli[1]}']['model_state'])
            
        else:
            coherences = np.append(coherences, np.array([len(indices_ipsi_pref), len(indices), trials['hem1stim']['model_output'][249][0], trials['hem1stim']['model_output'][249][1]]))
    
    print(f'{item}')
    
    if states==True:
        coherences.reshape(nb_trials, 250, 100)
        np.savez(f'/{item[:-4]}_{opto_target}_States', data = coherences)
    else:
        coherences.reshape(nb_trials, 4)
        np.savez(f'/{item[:-4]}_{opto_target}_Choices', data = coherences)
        
print('done !')
