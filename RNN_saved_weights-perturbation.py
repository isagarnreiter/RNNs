# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:47:02 2020

@author: Isabelle
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from psychrnn.backend.models.basic import Basic
import tensorflow as tf
from task_self_defined import PerceptualDiscrimination

dt = 10 # The simulation timestep.
tau = 100 # The intrinsic time constant of neural state decay.
T = 20000 # The trial length.
N_batch = 50 # The number of trials per training update.
N_rec = 50 # The number of recurrent units in the network.
name = 'dale_model_' #  Unique name used to determine variable scope for internal use.

pd = PerceptualDiscrimination(dt = dt, tau = tau, T = T, N_batch = N_batch) # Initialize the task object
modif_network_params = pd.get_task_params() # get the params passed in and defined in pd
modif_network_params['N_rec'] = N_rec # set the number of recurrent units in the model
modif_network_params['name'] = name

weights = dict(np.load('./weights/saved_weights_biol.npz', allow_pickle = True))
#weights['W_rec'][40:50,:] = 0
#weights['W_rec'][:, 40:50] = 0
np.savez('./weights/modified_saved_weights.npz', **weights)

def plot_weights(weights, title="", xlabel= "", ylabel=""):
    cmap = plt.set_cmap('RdBu_r')
    img = plt.matshow(weights, norm=Normalize(vmin=-.5, vmax=.5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    
    
modif_network_params['load_weights_path'] = './weights/modified_saved_weights.npz'

fileModel = Basic(modif_network_params)


x, y,mask, train_params = pd.get_trial_batch() # get pd task inputs and outputs
model_output, model_state = fileModel.test(x) # run the model on input x

