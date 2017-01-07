# -*- coding: utf-8 -*-
import logging

import grayskull.agents.base


log = logging.getLogger(name=__name__)


class Random(grayskull.agents.base.Agent):
    """
    Chooses a random action at every step.
    """
    def __init__(self, *args, **kwargs):
        """
        Randomly guess an action and don't learn

        Parameters
        ----------
        *args, **kwargs
            Passed on to super class
        """
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
