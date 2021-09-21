# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:53:42 2020

@author: Isabelle
"""

from __future__ import division

from psychrnn.tasks.task import Task
import numpy as np

class SensoryDiscrimination(Task):
    """Simulated bilateral sensory discrimination task
    On each trial the network receives two simultaneous noisy inputs into each of two input channels. 
    The network must determine which channel has the higher mean input and respond by driving the corresponding output unit to 1 after the go cue, which occurs 1 second after the end of the stimulation.
    When both inputs are the same, the target output is randomly generated and when both inputs = 0, then both outputs take on a value of 0.
    Takes two channels of noisy input (:attr:`N_in` = 2).
    Two channels output (:attr:`N_out` = 2) with a one hot encoding (high value is 1, low value is 0) towards the higher mean channel.

    There is also the option of adding simulated optogenetic stimulation to the task. The stimulation is applied at the same time as the stimulation through the 2 inputs.
    This task is based on the bilateral sensory discrimination task developped by Oliver Gauld in the Neural Computation Lab in the Wolfson Institute for Biomedical Research.
    
    Args:
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.
        N_in (int) : The number of input nodes
        N_rec (int) : The number of recurrent neurons
        N_out (int) : The number of output neurons
        opto (float) : The strength of the optogenetic stimulation
    """

    def __init__(self, dt, tau, T, N_batch, N_rec, N_out, opto):
        super(SensoryDiscrimination,self).__init__(2, 2, dt, tau, T, N_batch)
        
        self.N_rec = N_rec
        self.N_out = N_out
        self.opto = opto

        if self.opto == 0.0:
            self.N_in = 3
        else:
            self.N_in = 4
        
        self.lo = 0.0 # Low value for one hot encoding
        self.hi = 1.0 # High value for one hot encoding

    def generate_trial_params(self, batch, trial):
        """Define parameters for each trial.
        Implements :func:`~psychrnn.tasks.task.Task.generate_trial_params`.
        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch *batch*.
            
        Returns:
            dict: Dictionary of trial parameters including the following keys:
            :Dictionary Keys: 
                * **intensity** (*float*) (x2) -- the intensity of the 1st (intensity_0) and the 2nd stimulus (intensity_1). Both stimuli are scaled independently and take on values 0.0, 0.2, 0.4 or 0.6.
                * **stim_noise** (*float*) -- Scales the stimlus noise. Set to .1.
                * **onset_time** (*float*) -- Stimulus onset time at the onset of the trial
                * **stim_duration** (*float*) -- Stimulus duration of 1/4th the trial time.
                * **go_cue_onset** (*float*) -- Go cue onset after 3/4th the trial time.
                * **go_cue_duration** (*float*) -- Duration of the go cue of 100th the trial time.
                * **post_go_cue** (*float*) -- deactivation of the mask for 20th of the trial time.
                
        
        """

        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
        params = dict()
        prob_catch_trial = np.random.random()
        if prob_catch_trial > 0.85:    #Catch trials (trials where both inputs are null) occur with a likelihood of at least 15%
           params['intensity_0'] = 0.0
           params['intensity_1'] = 0.0
        else:
            choices = [0.0, 0.2, 0.4, 0.6]
            params['intensity_0'] = np.random.choice(choices)
            params['intensity_1'] = np.random.choice(choices)
        params['random_output'] = np.random.choice([0,1])
        params['stim_noise'] = 0.1
        params['onset_time'] = 0
        params['stim_duration'] = self.T/5
        params['go_cue_onset'] = self.T*(3/5)
        params['go_cue_duration'] = self.T/100
        params['post_go_cue'] = self.T / 20
        
        return params

    def trial_function(self, t, params):
        """Compute the trial properties at :data:`time`.
        Implements :func:`~psychrnn.tasks.task.Task.trial_function`.
        Based on the :data:`params` compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.
        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`generate_trial_params`.
        Returns:
            tuple:
            * **x_t** (*ndarray(dtype=float, shape=(*:attr:`N_in` *,))*) -- Trial input at :data:`time` given :data:`params`. For ``params['onset_time'] < time < params['onset_time'] + params['stim_duration']`` , 1 is added to the noise in both channels, and :data:`params['coherence']` is also added in the channel corresponding to :data:`params[dir]`.
            * **y_t** (*ndarray(dtype=float, shape=(*:attr:`N_out` *,))*) -- Correct trial output at :data:`time` given :data:`params`. From ``time > params['onset_time'] + params[stim_duration] + 20`` onwards, the correct output is encoded using one-hot encoding. Until then, y_t is 0 in both channels.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) -- True if the network should train to match the y_t, False if the network should ignore y_t when training. The mask is True for ``time > params['onset_time'] + params['stim_duration']`` and False otherwise.
        """
        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        int_0 = params['intensity_0']
        int_1 = params['intensity_1']
        rand_out = params['random_output']
        stim_onset = params['onset_time']
        stim_dur = params['stim_duration']
        noise = params['stim_noise']
        go_onset = params['go_cue_onset']
        go_duration = params['go_cue_duration']
        post_cue = params['post_go_cue']

        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(self.dt)*noise*noise)*np.random.randn(self.N_in) + 0.1
        x_t[2:] = 0 
        
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)
        # ----------------------------------
        # Compute values
        # ----------------------------------
        
        #during the period of the stimulus, the 2 stimulus channel are set to a random intensity between 0 and 0.6.
        #optogenetic stimulation can be simulated by setting the opto parameter which is input to the channel during the stimulation time
        if stim_onset < t < stim_onset + stim_dur:
            x_t[0] += int_0
            x_t[1] += int_1
            if self.N_in == 4:
                x_t[3] += self.opto
        
        
        #before the response window, the output is set to the low value
        if t <= go_onset + post_cue:
            y_t =+ self.lo
            
        #the go-cue is represented by a short spike before the response window
        if  go_onset < t < go_onset + go_duration:
            x_t[2] = 0.5

        if go_onset < t < go_onset + post_cue:
            mask_t = np.zeros(self.N_out)

        if t > go_onset + post_cue:
            if int_0 > int_1:
                y_t[0] = self.hi
                y_t[1] = self.lo
            elif int_0 < int_1:
                y_t[0] = self.lo
                y_t[1] = self.hi
            elif int_0 == int_1 == 0.0:
                y_t[0] = self.lo
                y_t[1] = self.lo
            elif int_0 == int_1:
                y_t[0] = rand_out
                y_t[1] = 1-rand_out

        return x_t, y_t, mask_t
    
   
    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.
        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.
        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".
        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.
        
        """

        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)
        return np.mean(np.equal(truth, chosen))
    
    def psychometric_curve(self, correct_output, train_params):
        """Calculates the percentage of choice 1 made by the model, depending on the coherence between input 0 and 1.'
        --> psychometric curve.
        Catch trials are excluded."""

        diff1_2 = np.array([])
        chosen = np.array([])
        for i in range(len(train_params)):
            if train_params[i]['intensity_0']!=0.0 or train_params[i]['intensity_1']!=0.0:
                diff1_2 = np.append(diff1_2, round(train_params[i]['intensity_0']-train_params[i]['intensity_1'], 2))
                chosen = np.append(chosen, np.argmax(correct_output[i,249,:], axis=0))

        bins = np.array([-0.6, -0.4, -0.2,  0. ,  0.2,  0.4,  0.6])
        digitized = np.digitize(diff1_2, bins)
        bin_means = (1-np.array([chosen[digitized == k].mean() for k in range(1, len(bins)+1)]))*100
        
        return np.array([bin_means, bins], dtype=object)