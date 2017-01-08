# -*- coding: utf-8 -*-
import logging

import grayskull.agents.linear.guess
import grayskull.agents.linear.hill
import grayskull.agents.random

log = logging.getLogger(name=__name__)


AGENTS = {
    'random': grayskull.agents.random.Random,
    'linear_guessing': grayskull.agents.linear.guess.LinearGuessing,
    'linear_hill': grayskull.agents.linear.hill.LinearHill,
}
