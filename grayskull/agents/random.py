# -*- coding: utf-8 -*-
import logging

import grayskull.agents.base


log = logging.getLogger(name=__name__)


class Random(grayskull.agents.base.Agent):
    """
    Chooses a random action at every step.
    """
    def __init__(self, *args, **kwargs):
        super(Random, self).__init__(*args, **kwargs)

    def act(self, observation):
        """
        Return a random action

        Parameters
        ----------
        observation : a game state (usually an image)

        Returns
        -------
        an action from self.actions
        """
        return self.actions.sample()

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
        pass
