# -*- coding: utf-8 -*-
import logging
import sys

import numpy as np

import grayskull.agents.linear.base

log = logging.getLogger(name=__name__)


class LinearGuessing(grayskull.agents.linear.base.LinearAgent):
    """
    Generates 10000 random settings for a linear model's weights and
    choose the best (where "best" is defined as the configuration that
    leads to the highest per-episode reward).
    """
    def __init__(self, n_guesses=10000, *args, **kwargs):
        """
        Parameters
        ----------
        n_guesses : int, optional
            How many guesses to try
        *args, **kwargs
            Passed on to super class
        """
        super(LinearGuessing, self).__init__(*args, **kwargs)

        self.n_guesses = n_guesses

        # make n_guesses guesses
        # this will be a (n_guesses, unrolled observation_space) array
        self.weights = self.guess_weights(n_guesses)

        # track the episode
        self.episode = 0

        # track the rewards
        self.rewards = np.zeros(n_guesses, dtype=np.float)

        # set the initial weights
        self.set_weights(self.weights[self.episode])

    def guess_weights(self, n_guesses):
        """
        Randomly guess at possible weights

        Parameters
        ----------
        n_guesses : int
            The number of guesses to return

        Returns
        -------
        numpy array of floats (n_guesses, n_params)
            `n_guesses` random guesses at model parameters, one per row
        """
        # TODO: Work out appropriate distribution for the params
        return np.random.rand(n_guesses, self.n_params) * 2 - 1

    def set_weights(self, weights):
        """
        Set the model weights

        Parameters
        ----------
        weights : numpy array (n_params, )
            The weights to set
        """
        self.model = weights

    def react(self,
              observation,
              action,
              reward,
              done,
              new_observation,
              timed_out):
        """
        Incorporate feedback from simulation

        Parameters
        ----------
        observation : a game state (usually an image)

        action : int
            The action that was taken

        reward : int
            The reward that was given

        done : bool
            Whether this ends the episode

        new_observation : a game state (usually an image)

        timed_out : bool
            Whether this ends an episode because of timeout
        """
        self.rewards[self.episode] += reward

        if not (done or timed_out):
            return

        self.episode += 1

        if self.episode >= self.n_guesses:
            best_episode = np.argmax(self.rewards)
            best_reward = self.rewards[best_episode]
            best_weghts = self.weights[best_episode]

            log.info('Best weights gave reward {}.'.format(best_reward))
            log.debug('Setting linear agent weights to {}'.format(best_weghts))

            self.set_weights(self.weights[best_episode])
            sys.exit(0)

        self.set_weights(self.weights[self.episode])
