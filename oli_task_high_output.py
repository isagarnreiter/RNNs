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
        
        params['intensity'] = np.random.uniform(0.8, 1.4)
        
        params['coherence'] = np.random.uniform(params['intensity']/2, params['intensity'])+0.1
        params['direction'] = np.random.choice([0, 1])
        params['stim_noise'] = 0.1
        params['onset_time'] = 0
        params['stim_duration'] = 500
        params['go_cue_onset'] = 1500
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
        inte = params['intensity']
        coh = params['coherence']
        stim_onset = params['onset_time']
        stim_dur = params['stim_duration']
        dire = params['direction']
        noise = params['stim_noise']
        go_onset = params['go_cue_onset']
        go_duration = params['go_cue_duration']
        post_cue = params['post_go_cue']
        
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(self.dt)*noise*noise)*np.random.randn(self.N_in)
        x_t[2] = 0

        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)
        # ----------------------------------
        # Compute values
        # ----------------------------------
        if stim_onset < t < stim_onset + stim_dur:
            x_t[dire] += coh
            x_t[1-dire] += (inte - coh)
        
        if t <= go_onset + post_cue:
            y_t =+ 0
            
            
        if  go_onset < t < go_onset + go_duration:
            x_t[2] = 0.5

        if go_onset < t < go_onset + post_cue:
            mask_t = np.zeros(self.N_out)

        if t > go_onset + post_cue:
            y_t[dire] = self.hi
            y_t[1-dire] = self.lo

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
    
    def psychometric_curve(self, correct_output, output_mask, train_params, bin_nb):
        """Calculates the percentage of choice 1 made by the model, depending on the coherence between input 0 and 1.'
        --> psychometric curve."""
  
        
        coherence = []
        
        for i in range(len(train_params)):
            coherence.append(train_params[i]['coherence']/train_params[i]['intensity'])
            coherence[i] = coherence[i] - (1-coherence[i])
            if train_params[i]['direction'] == 1:
                coherence[i] = coherence[i]
            else:
                coherence[i] = - coherence[i]
        
        chosen = np.argmax(np.mean(correct_output*output_mask, axis=1), axis = 1)
        
        bins = np.linspace(-1, 1, bin_nb+1)
        digitized = np.digitize(coherence, bins)
        bin_means = np.array([chosen[digitized == i].mean() for i in range(1, len(bins))])
        bin_means = bin_means*100
    
        return bin_means
                
