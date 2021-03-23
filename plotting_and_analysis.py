# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:13:02 2021

@author: Isabelle
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import lognorm
from scipy.stats import norm

#%%

dalemodel_test = dict(np.load("IpsiContra_IN05_REC01_Col10_s2020.npz", allow_pickle=True))
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
unity_line = [-1, 0, 1]

figure = plt.figure(figsize=(6,6))
ax1 = plt.subplot(111)
ax1.scatter(max_hem1_hem1stim, max_hem1_hem2stim, c = 'coral', label = 'hemisphere 1', alpha=0.6)
ax1.scatter(max_hem2_hem1stim, max_hem2_hem2stim, c = 'green', label = 'hemisphere 2', alpha=0.6)
ax1.plot(unity_line, unity_line, c='black')
ax1.set_xlim(-1, 1)
ax1.set_xticks([-1,-0.5,0, 0.5,1])
ax1.set_ylim(-1,1)
ax1.set_yticks([-1,-0.5,0, 0.5,1])
ax1.set_title('states of excitatory neuron in hemisphere 1 and 2 at T = 500 ms')
ax1.legend()
ax1.set_xlabel('stim in hem 1')
ax1.set_ylabel('stim in hem 2')


#%%
# ---------------------- Save and plot the weights of the network ---------------------------

weights = daleModel.get_weights()

plot_weights(weights['W_rec'],  
            xlabel = 'From', 
            ylabel = 'To')

plot_weights(weights['W_in'])
plot_weights(weights['W_out'])

daleModel.save("weights/seg_output_20_1_01_01")


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
