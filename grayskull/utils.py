# -*- coding: utf-8 -*-
import collections
import logging

import cursesmenu

log = logging.getLogger(name=__name__)


def choose_game(games):
    """
    Given a list of games, allow the user to choose one

    Parameters
    ----------
    games : iterable of str
        The available games

    Returns
    -------
    str
        The chosen game
    """
    # if there is any subcategorization to choose from
    if any(['.' in game for game in games]):
        # make a list of sublists
        categories = collections.defaultdict(list)

        for game in games:
            category, game_name = game.split('.', 1)
            categories[category].append(game_name)

        # have user choose a category
        selected_option = cursesmenu.SelectionMenu.get_selection(
            sorted(categories.keys()), 'Select a category'
        )
        category = sorted(categories.keys())[selected_option]

        # have user choose game from this category
        return category + '.' + choose_game(categories[category])

    # have the user choose one of the games
    selected_option = cursesmenu.SelectionMenu.get_selection(
        sorted(games), 'Select a game'
    )
    return sorted(games)[selected_option]
