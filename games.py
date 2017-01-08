# -*- coding: utf-8 -*-
import gym

print('\n'.join(
    [game for game in sorted(gym.envs.registry.env_specs.keys())]
))
