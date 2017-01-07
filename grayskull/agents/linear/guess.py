# -*- coding: utf-8 -*-
import logging

import numpy as np

import grayskull.agents.linear.base

log = logging.getLogger(name=__name__)


class LinearGuessing(grayskull.agents.linear.base.LinearAgent):
    def __init__(self, n_guesses=10, *args, **kwargs):
        super(Guessing, self)__init__(*args, **kwargs)

        self.n_guesses = n_guesses

        # make n_guesses guesses
        # this will be a (n_guesses, unrolled observation_space) array
        self.weights = self.guess_weights()

        # track the episode
        self.episode = 0

        # track the rewards
        self.rewards = np.zeros(n_guesses, dtype=np.float)

        # set the initial weights
        self.set_weights(self.episode)

    def guess_weights(self):
        # TODO: Work out appropriate distribution for the params
        return np.random.rand((self.n_guesses, self.n_params)) * 2 - 1

    def set_weights(self, episode):
        self.model = self.weights[episode]

    def react(self, observation, action, reward, done, new_observation):
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
        """
        self.rewards[self.episode] += reward

        if not done:
            return

        self.episode += 1

        if self.episode >= self.n_guesses:
            best_episode = np.argmax(self.rewards)
            best_reward = self.rewards[best_episode]
            best_weghts = self.weights[best_episode]

            log.info('Best weights gave reward {}.'.format(best_reward))
            log.debug('Setting linear agent weights to {}'.format(best_weghts))

            self.set_weights(best_episode)
            return

        self.set_weights(self.episode)
