# -*- coding: utf-8 -*-
import logging


log = logging.getLogger(name=__name__)


class IncompatibleGameError(Exception):
    """
    Raised when the user tries to train an agent on a game that is not suitable
    for that agent.
    """
    pass


class SolvedGame(Exception):
    """
    Raised to indicate that the simulation is done because we've solved the
    game
    """
    pass
