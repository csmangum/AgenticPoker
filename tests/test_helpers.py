from unittest.mock import patch, MagicMock
from data.enums import ActionType
from data.types.action_decision import ActionDecision
from game.betting import betting_round

def run_betting_scenario(mock_game, players, scenario):
    """Run a betting round based on a predefined scenario.

    Args:
        mock_game: The mock game instance.
        players: List of player instances.
        scenario: Dictionary defining the actions for each player.

    Returns:
        Tuple containing the player queue and the list of players.
    """
    # Patch the logger to avoid cluttering test output
    with patch("game.betting.BettingLogger") as mock_logger:
        mock_logger.log_debug = MagicMock()
        mock_logger.log_player_turn = MagicMock()
        mock_logger.log_player_action = MagicMock()
        mock_logger.log_state_after_action = MagicMock()
        mock_logger.log_line_break = MagicMock()
        mock_logger.log_message = MagicMock()

        # Set up player actions based on the scenario
        for player_index, actions in scenario.items():
            player = players[player_index]
            action_sequence = (ActionDecision(**action) for action in actions)
            player.decide_action = MagicMock(side_effect=action_sequence)
            player.execute = MagicMock(side_effect=lambda d, g: execute_side_effect(player, d, g))

        # Set up player queue and pot manager
        table = mock_game.table
        table.players = players
        mock_game.pot = MagicMock()
        mock_game.pot.pot = 0

        # Run the betting round
        betting_round(mock_game)

        return table, players

def execute_side_effect(player, decision, game):
    """Execute the side effect of a player's action decision.

    Args:
        player: The player instance.
        decision: The action decision made by the player.
        game: The game instance.
    """
    if decision.action_type == ActionType.FOLD:
        player.folded = True
    elif decision.action_type in (ActionType.CALL, ActionType.RAISE):
        player.place_bet(decision.raise_amount, game)
        if player.chips == 0:
            player.is_all_in = True
