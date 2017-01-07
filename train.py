# -*- coding: utf-8 -*-
import argparse
import json
import logging

import gym
# import universe

import grayskull.agents.linear.guess
import grayskull.agents.random

# import utils

log = logging.getLogger(name=__name__)


AGENTS = {
    'random': grayskull.agents.random.Random,
    'linear_guessing': grayskull.agents.linear.guess.LinearGuessing,
}

GAMES = [
    game for game in sorted(gym.envs.registry.env_specs.keys())
]


def main(game='CartPole-v0',
         agent='random',
         agent_args={},
         render=False,
         monitor=False,
         **kwargs):

    # set up the game and agent
    env = gym.make(game)
    agent_name = agent
    agent = AGENTS[agent_name](
        action_space=env.action_space,
        observation_space=env.observation_space,
        **agent_args
    )

    # determine the max number of steps per episode from the environment
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    episode = 0

    # run many episodes
    while True:
        # reset the environment
        observation = env.reset()
        done = False

        # track the total reward
        total_reward = 0.0
        step = 1

        # run steps until the episode is done or times out
        while step <= max_steps and not done:
            if render:
                env.render()

            # choose an action
            action = agent.act(observation)

            # take the action
            new_observation, reward, done, _ = env.step(action)
            total_reward += reward

            # learn from the action
            agent.react(
                observation,
                action,
                reward,
                done,
                new_observation,
                step == max_steps
            )

            # make the new observation the current one
            observation = new_observation

            step += 1

        log.info('Episode {}: {}'.format(episode, total_reward))
        episode += 1


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Train an agent on a game'
    parser = argparse.ArgumentParser(description=desc)

    game_help = 'Which game to train on'
    parser.add_argument(
        '-g',
        '--game',
        choices=GAMES,
        default='CartPole-v0',
        help=game_help
    )

    agent_help = 'Which agent to use'
    parser.add_argument(
        '-a',
        '--agent',
        choices=AGENTS.keys(),
        default='random',
        help=agent_help
    )

    agent_args_help = 'Additional args to pass to the agent'
    parser.add_argument(
        '--agent-args',
        type=json.loads,
        default='{}',
        help=agent_args_help
    )

    render_help = 'Whether to render the screen'
    parser.add_argument('-r',
                        '--render',
                        action='store_true',
                        help=render_help)

    upload_help = 'Whether to upload'
    parser.add_argument('-u',
                        '--upload',
                        action='store_true',
                        help=upload_help)

    monitor_help = 'Record video and stats'
    parser.add_argument('--monitor',
                        action='store_true',
                        help=monitor_help)

    seed_help = ('Set the random seed')
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help=seed_help)

    verbosity_help = 'Verbosity level (default: %(default)s)'
    choices = [logging.getLevelName(logging.DEBUG),
               logging.getLevelName(logging.INFO),
               logging.getLevelName(logging.WARN),
               logging.getLevelName(logging.ERROR)]

    parser.add_argument('-v',
                        '--verbosity',
                        choices=choices,
                        help=verbosity_help,
                        default=logging.getLevelName(logging.INFO))

    # Parse the command line arguments
    args = parser.parse_args()

    # Set the logging to console level
    gym.undo_logger_setup()
    logging.basicConfig(level=args.verbosity)

    return args


if __name__ == '__main__':
    main(**parse_args().__dict__)
