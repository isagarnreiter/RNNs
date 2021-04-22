# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:53:42 2020

@author: Isabelle
"""

from __future__ import division

from psychrnn.tasks.task import Task
import numpy as np

class PerceptualDiscrimination(Task):
    """Two alternative forced choice (2AFC) binary discrimination task. 
    On each trial the network receives two simultaneous noisy inputs into each of two input channels. The network must determine which channel has the higher mean input and respond by driving the corresponding output unit to 1.
    Takes two channels of noisy input (:attr:`N_in` = 2).
    Two channel output (:attr:`N_out` = 2) with a one hot encoding (high value is 1, low value is .2) towards the higher mean channel.
    Loosely based on `Britten, Kenneth H., et al. "The analysis of visual motion: a comparison of neuronal and psychophysical performance." Journal of Neuroscience 12.12 (1992): 4745-4765 <https://www.jneurosci.org/content/12/12/4745>`_
    Args:
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.
        coherence (float, optional): Amount by which the means of the two channels will differ. By default None.
        direction (int, optional): Either 0 or 1, indicates which input channel will have higher mean input. By default None.
    """

    def __init__(self, dt, tau, T, N_batch, coherence = None, direction = None):
        super(PerceptualDiscrimination,self).__init__(2, 2, dt, tau, T, N_batch)
        
        self.coherence = coherence

        self.direction = direction

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
                * **coherence** (*float*) -- Amount by which the means of the two channels will differ. :attr:`self.coherence` if not None, otherwise ``np.random.exponential(scale=1/5)``.
                * **direction** (*int*) -- Either 0 or 1, indicates which input channel will have higher mean input. :attr:`self.direction` if not None, otherwise ``np.random.choice([0, 1])``.
                * **stim_noise** (*float*) -- Scales the stimlus noise. Set to .1.
                * **onset_time** (*float*) -- Stimulus onset time at 1/4th of total trial length
                * **stim_duration** (*float*) -- Stimulus duration of 1/2 of total trial length.
        """

        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
        params = dict()
        prob_catch_trial = np.random.random()
        if prob_catch_trial > 0.85:
           params['intensity_0'] = 0.0
           params['intensity_1'] = 0.0
        else:
            choices = [0.0, 0.2, 0.4, 0.6]
            params['intensity_0'] = np.random.choice(choices)
            params['intensity_1'] = np.random.choice(choices)
        params['random_output'] = np.random.choice([0,1])
        params['stim_noise'] = 0.1
        params['onset_time'] = 0
        params['stim_duration'] = 500
        params['go_cue_onset'] = 1500
        params['go_cue_duration'] = self.T/100
        params['post_go_cue'] = self.T / 20

        params['intensity_opto'] = 0.6
        params['duration_opto'] = 20
        params['repetition_opto'] = 100
        
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
        
        int_opto= params['intensity_opto'] = 0.6
        dur_opto = params['duration_opto'] = 20
        rep_opto = params['repetition_opto'] = 100
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(self.dt)*noise*noise)*np.random.randn(self.N_in) + 0.1
        x_t[2] = 0
        x_t[3] = 0
        
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)
        # ----------------------------------
        # Compute values
        # ----------------------------------
        if stim_onset < t < stim_onset + stim_dur:
            x_t[0] += int_0
            x_t[1] += int_1
        
        if t <= go_onset + post_cue:
            y_t =+ self.lo
            
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

        if t%rep_opto == 0 and t<go_onset:
            x_t[3] = int_opto
            
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
        --> psychometric curve."""
  
        diff1_2 = []
        chosen = []
        for i in range(len(train_params)):
            if train_params[i]['intensity_0']!=0.0 or train_params[i]['intensity_1']!=0.0:
                diff1_2.append(round(train_params[i]['intensity_0']-train_params[i]['intensity_1'], 2))
                if correct_output[i, 249, 0] > 0.8:
                    chosen.append(1)
                else:
                    chosen.append(0)
                    
        diff1_2 = np.array(diff1_2)
        chosen = np.array(chosen)
        bins = np.array([-0.6, -0.4, -0.2,  0. ,  0.2,  0.4,  0.6])
        digitized = np.digitize(diff1_2, bins)
        bin_means = np.array([chosen[digitized == j].mean() for j in range(1, len(bins)+1)])
        bin_means = bin_means*100
    
        chosen=np.array(chosen)
        frac_choice_1 = len(chosen[chosen==1])/len(chosen)
        
        return bin_means, bins, frac_choice_1
                