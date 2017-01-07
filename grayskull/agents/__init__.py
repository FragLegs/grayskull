# -*- coding: utf-8 -*-
import logging
import pickle

log = logging.getLogger(name=__name__)


def load(path):
    """
    Load an Agent from a path

    Parameters
    ----------
    path : str
        Where to load the agent from

    Returns
    -------
    Agent
        The saved agent
    """
    with open(path, mode='r') as fin:
        return pickle.load(fin)
