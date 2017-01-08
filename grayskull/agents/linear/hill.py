# -*- coding: utf-8 -*-
import logging
import sys

import numpy as np

import grayskull.agents.linear.base
import grayskull.errors


log = logging.getLogger(name=__name__)


class LinearHill(grayskull.agents.linear.base.LinearAgent):
    """
    Start with a random setting of the parameters, add a small amount of noise
    to the parameters, and evaluate the new parameter configuration. If it
    performs better than the old configuration, discard the old configuration
    and accept the new one.

    See: https://openai.com/requests-for-research/#cartpole
    """
    def __init__(self, learning_rate=0.5, top_score=200.0, *args, **kwargs):
        """
        Parameters
        ----------
        learning_rate : float, optional
            How much to change the parameters.
            Default: 0.5
        top_score : float, optional
            If we reach the top score, stop tweaking the params
            Default: 200.0 (CartPole's top score)
        *args, **kwargs
            Passed on to super class
        """
        super(LinearHill, self).__init__(*args, **kwargs)

        # how much to change the parameters after each episode
        self.learning_rate = learning_rate

        # track the episode
        self.episode = 0

        # track the episode rewards
        self.current_reward = 0.0

        # track the best reward and weights
        self.best_reward = -np.inf
        self.best_weights = self.params

        # what is the solved score?
        self.top_score = top_score

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
        self.current_reward += reward

        if not (done or timed_out):
            return

        self.episode += 1

        # check whether we've climbed the hill
        if self.current_reward >= self.best_reward:
            self.best_weights = self.params
            self.best_reward = self.current_reward

        # check if we've solved the game
        if self.current_reward >= self.top_score:
            self.params = self.best_weights
            log.debug('Best weights: {}'.format(self.params))
            raise grayskull.errors.SolvedGame()

        # reset and try new parameters
        self.current_reward = 0.0

        # adjust the parameters
        gradient = (np.random.rand(self.n_params) * 2 - 1) * self.learning_rate

        self.params = self.best_weights + gradient
