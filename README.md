# Grayskull

Grayskull is a project for reinforcement learning agents trying to master the OpenAI universe.

## Installing

Start by installing [gym](https://github.com/openai/gym#installing-everything) and [universe](https://github.com/openai/universe#installation) from the github source (in a virtualenv, of course). It may take a few tries to get all of the dependencies installed.

Then, you can install grayskull by cloning this repo and calling `pip install .` in the root directory.


## Usage

Grayskull allows you to train an RL agent on any game in [OpenAI Gym](https://gym.openai.com/) or [OpenAI Universe](https://universe.openai.com/) (not yet implemented) with the `train.py` script. All command line options are viewable by running `python train.py -h`. That information is also reproduced here:

    usage: train.py [-h] [-g GAME] [-a {random,linear_guessing}]
                    [--agent-args AGENT_ARGS] [-e EPISODES] [-r] [--monitor]
                    [--seed SEED] [-v {DEBUG,INFO,WARNING,ERROR}]

    Train an agent on a game

    optional arguments:
      -h, --help            show this help message and exit
      -g GAME, --game GAME  Which game to train on (default: CartPole-v0)
      -a {random,linear_guessing}, --agent {random,linear_guessing}
                            Which agent to use (default: random)
      --agent-args AGENT_ARGS
                            Additional args to pass to the agent (default: {})
      -e EPISODES, --episodes EPISODES
                            How many episodes to run (-1 means run forever)
                            (default: -1)
      -r, --render          Whether to render the screen (default: False)
      --monitor             Record video and stats (default: False)
      --seed SEED           Set the random seed (default: None)
      -v {DEBUG,INFO,WARNING,ERROR}, --verbosity {DEBUG,INFO,WARNING,ERROR}
                            Verbosity level (default: INFO)

## Available Games

Currently, all of the games in gym are supported. You can list them by calling `python games.py`

## Available Agents

Available agents and their descriptions can be viewed by running `python agents.py`
