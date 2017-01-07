# -*- coding: utf-8 -*-
import logging
import pickle


log = logging.getLogger(name=__name__)


class Agent(object):
    """
    Any Agent that can make actions from observations and then update the
    action-maker from rewards and new observations
    """
    def __init__(self, action_space, **kwargs):
        super(Agent, self).__init__()
        self.actions = action_space

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
        raise NotImplementedError('Override me!')

    def react(self,
              observation,
              action,
              reward,
              done,
              new_observation,
              timed_out):
        """
        Incorporate feedback from simulation. By default, just ignore.

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
        pass

    def save(self, path):
        """
        Save the Agent (after learning)

        Parameters
        ----------
        path : str
            Where to save the agent
        """
        with open(path, mode='w') as fout:
            pickle.dump(self, fout)
