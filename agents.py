# -*- coding: utf-8 -*-
from grayskull.agents.agents import AGENTS

print('')
for name, agent in sorted(AGENTS.items()):
    print('{}:\n{}'.format(name, agent.__doc__))
