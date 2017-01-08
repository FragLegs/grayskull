# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import time

import gym
# import universe

from grayskull.agents.agents import AGENTS
import grayskull.errors
# import utils

log = logging.getLogger(name=__name__)


GAMES = [
    game for game in sorted(gym.envs.registry.env_specs.keys())
]


def main(game='CartPole-v0',
         agent='random',
         agent_args={},
         episodes=-1,
         render=False,
         monitor=False,
         **kwargs):
    """
    Run the simulation

    Parameters
    ----------
    game : str, optional
        The name of a gym or universe game to play
        Default: CartPole-v0
    agent : str, optional
        The name of an agent to use
        Default: random
    agent_args : dict, optional
        Additional arguments to pass to the agent
    episodes : int, optional
        How many episodes to run the simulation (or -1 for "forever")
        Default: -1
    render : bool, optional
        Whether to draw the sim on screen
        Default: False
    monitor : bool, optional
        Whether to monitor performance with gym (not yet implemented)
        Default: False
    **kwargs : keyword args
        Ignored
    """
    # create a folder for saving the agent and checkpoints
    date = time.strftime('%y-%m-%d-%H-%M-%S', time.localtime())
    results_path = os.path.join('results', agent, game, date)
    os.makedirs(results_path)

    # set up the game and agent
    env = gym.make(game)
    agent_name = agent

    try:
        agent = AGENTS[agent_name](
            action_space=env.action_space,
            observation_space=env.observation_space,
            results_path=results_path,
            **agent_args
        )
    except grayskull.errors.IncompatibleGameError as e:
        log.error(e)
        return

    # determine the max number of steps per episode from the environment
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    episode = 0

    try:
        # run many episodes
        while episodes == -1 or episode < episodes:
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

            log.debug('Episode {}: {}'.format(episode, total_reward))
            episode += 1
    finally:
        yn = raw_input('Save agent? ')
        if yn.lower() == 'y':
            agent.save(os.path.join(results_path, 'final.pkl'))


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Train an agent on a game'
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    game_help = 'Which game to train on'
    parser.add_argument(
        '-g',
        '--game',
        choices=GAMES,
        default='CartPole-v0',
        help=game_help,
        metavar='GAME'
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

    episodes_help = 'How many episodes to run (-1 means run forever)'
    parser.add_argument(
        '-e',
        '--episodes',
        type=int,
        help=episodes_help,
        default=-1
    )

    render_help = 'Whether to render the screen'
    parser.add_argument(
        '-r',
        '--render',
        action='store_true',
        help=render_help
    )

    monitor_help = 'Record video and stats'
    parser.add_argument(
        '--monitor',
        action='store_true',
        help=monitor_help
    )

    seed_help = ('Set the random seed')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help=seed_help
    )

    verbosity_help = 'Verbosity level (default: %(default)s)'
    choices = [
        logging.getLevelName(logging.DEBUG),
        logging.getLevelName(logging.INFO),
        logging.getLevelName(logging.WARN),
        logging.getLevelName(logging.ERROR)
    ]

    parser.add_argument(
        '-v',
        '--verbosity',
        choices=choices,
        help=verbosity_help,
        default=logging.getLevelName(logging.INFO)
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Set the logging to console level
    gym.undo_logger_setup()
    logging.basicConfig(level=args.verbosity)

    return args


if __name__ == '__main__':
    main(**parse_args().__dict__)
