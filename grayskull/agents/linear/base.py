# -*- coding: utf-8 -*-
import logging

import gym.spaces.discrete
import numpy as np

import grayskull.agents.base
import grayskull.errors

log = logging.getLogger(name=__name__)


class LinearAgent(grayskull.agents.base.Agent):
    def __init__(self,
                 action_space,
                 observation_space,
                 *args,
                 **kwargs):
        """
        A base agent for the OpenAI requests for research on CartPole.

        Parameters
        ----------
        action_space : gym.spaces.Discrete
            The possible actions
        observation_space : numpy array like
            The space of possible observations
        *args, **kwargs
            Passed on to super class
        """
        super(LinearAgent, self).__init__(action_space, *args, **kwargs)

        # if action space is more than 2 actions, these agents don't work
        if not hasattr(action_space, 'n') or action_space.n > 2:
            msg = (
                'This agent only supports games with 2 possible actions (see '
                'CartPole and Acrobot for examples of 2-action games)'
            )
            raise grayskull.errors.IncompatibleGameError(msg)

        # get the input size
        self.n_params = np.prod(observation_space.shape)

        # set up the model
        self.model = np.zeros(self.n_params, dtype=np.float)

    def act(self, observation):
        """
        Return an action based on the last observed state

        Parameters
        ----------
        observation : a game state (usually an image)

        Returns
        -------
        an action from self.actions
        """
        return 0 if np.dot(self.model, observation.ravel()) < 0 else 1

