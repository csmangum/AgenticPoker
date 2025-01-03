import logging
from typing import Any, Dict, List, Union

from game.player import Player


def log_chip_movements(
    players: List[Player],
    initial_chips: Union[Dict[Player, int], Dict[str, int], dict],
    handle_mocks: bool = False,
) -> None:
    """
    Log chip movements for players from their initial amounts.

    Args:
        players: List of players to check
        initial_chips: Dictionary of initial chip amounts (either by Player or name)
        handle_mocks: Whether to handle test mock objects
    """
    for player in players:
        try:
            initial = (
                initial_chips[player.name]
                if isinstance(next(iter(initial_chips)), str)
                else initial_chips[player]
            )

            if player.chips != initial:
                net_change = player.chips - initial
                logging.info(
                    f"{player.name}: ${initial} → ${player.chips} ({net_change:+d})"
                )
        except (TypeError, AttributeError) as e:
            if handle_mocks:
                logging.info(
                    f"{player.name}: Chip movement tracking skipped in test mode"
                )
            else:
                raise e
