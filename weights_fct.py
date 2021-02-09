# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:13:34 2021

@author: Isabelle
"""
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
def plot_weights(weights, title="", xlabel= "", ylabel=""):
    cmap = plt.set_cmap('RdBu_r')
    img = plt.matshow(weights, norm=Normalize(vmin=-.5, vmax=.5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()