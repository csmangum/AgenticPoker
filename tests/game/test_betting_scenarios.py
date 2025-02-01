from typing import List
from unittest.mock import MagicMock, patch

import pytest

from data.types.action_decision import ActionDecision, ActionType
from game.betting import betting_round
from game.table import Table
from tests.mocks.mock_agent import MockAgent
from tests.test_helpers import run_betting_scenario


@pytest.fixture
def mock_game(game_state):
    """Create a mock game instance with basic state."""
    game = MagicMock()
    game.round_state = game_state.round_state
    game.current_bet = game_state.min_bet
    game.small_blind = game_state.small_blind
    game.big_blind = game_state.big_blind
    return game


@pytest.fixture
def setup_players(player_factory) -> List[MockAgent]:
    """Create a list of mock players with specific behaviors.

    Uses player_factory to create consistent player instances with:
    - Standard names (Alice, Bob, Charlie, Randy)
    - 1000 chips each
    - No initial bets
    - Not folded
    """
    players = [
        player_factory(name="Alice", chips=1000),
        player_factory(name="Bob", chips=1000),
        player_factory(name="Charlie", chips=1000),
        player_factory(name="Randy", chips=1000),
    ]

    # Configure initial states
    for player in players:
        player.bet = 0
        player.folded = False

    return players


@patch("game.betting.BettingLogger")
def test_betting_round_raise_and_call(mock_logger, mock_game, setup_players):
    """Test that betting round correctly completes after raise and call.

    Scenario:
    1. Alice raises to 300
    2. Bob folds
    3. Charlie folds
    4. Randy calls 300
    5. Betting round should complete

    Expected:
    - Betting round should complete after Randy calls
    - needs_to_act should be empty
    - Correct chip counts and bets should be recorded
    """
    scenario = {
        0: [{"action_type": ActionType.RAISE, "raise_amount": 300, "reasoning": "Strong hand"}],
        1: [{"action_type": ActionType.FOLD, "reasoning": "Weak hand"}],
        2: [{"action_type": ActionType.FOLD, "reasoning": "Weak hand"}],
        3: [{"action_type": ActionType.CALL, "reasoning": "Decent hand"}],
    }

    table, players = run_betting_scenario(mock_game, setup_players, scenario)

    # Verify betting round completed correctly
    assert (
        len(table.needs_to_act) == 0
    ), "needs_to_act should be empty after round completion"

    # Verify player states
    assert players[0].bet == 300, "Alice's bet should be 300"
    assert players[3].bet == 300, "Randy's bet should be 300"
    assert players[1].folded, "Bob should have folded"
    assert players[2].folded, "Charlie should have folded"

    # Verify chip counts
    assert players[0].chips == 700, "Alice should have 700 chips remaining"
    assert players[1].chips == 1000, "Bob should have 1000 chips remaining"
    assert players[2].chips == 1000, "Charlie should have 1000 chips remaining"
    assert players[3].chips == 700, "Randy should have 700 chips remaining"

    # Verify last raiser
    assert table.last_raiser == players[0], "Alice should be the last raiser"


@patch("game.betting.BettingLogger")
def test_betting_round_raise_reraise_call(mock_logger, mock_game, setup_players):
    """Test betting round with raise, re-raise, and call sequence.

    Scenario:
    1. Alice raises to 300
    2. Bob re-raises to 600
    3. Charlie folds
    4. Randy folds
    5. Alice calls 600

    Expected:
    - Betting round should complete after Alice calls
    - needs_to_act should be empty
    - Correct chip counts and bets should be recorded
    """
    scenario = {
        0: [
            {"action_type": ActionType.RAISE, "raise_amount": 300, "reasoning": "Strong hand"},
            {"action_type": ActionType.CALL, "reasoning": "Still strong but not raising"},
        ],
        1: [{"action_type": ActionType.RAISE, "raise_amount": 600, "reasoning": "Stronger hand"}],
        2: [{"action_type": ActionType.FOLD, "reasoning": "Weak hand"}],
        3: [{"action_type": ActionType.FOLD, "reasoning": "Weak hand"}],
    }

    table, players = run_betting_scenario(mock_game, setup_players, scenario)

    # Verify betting round completed correctly
    assert (
        len(table.needs_to_act) == 0
    ), "needs_to_act should be empty after round completion"

    # Verify player states
    assert players[0].bet == 600, "Alice's bet should be 600"
    assert players[1].bet == 600, "Bob's bet should be 600"
    assert players[2].folded, "Charlie should have folded"
    assert players[3].folded, "Randy should have folded"

    # Verify chip counts
    assert players[0].chips == 400, "Alice should have 400 chips remaining"
    assert players[1].chips == 400, "Bob should have 400 chips remaining"
    assert players[2].chips == 1000, "Charlie should have 1000 chips remaining"
    assert players[3].chips == 1000, "Randy should have 1000 chips remaining"

    # Verify last raiser
    assert table.last_raiser == players[1], "Bob should be the last raiser"


@patch("game.betting.BettingLogger")
def test_betting_round_all_checks(mock_logger, mock_game, setup_players):
    """Test betting round where all players check.

    Scenario:
    1. Alice checks
    2. Bob checks
    3. Charlie checks
    4. Randy checks

    Expected:
    - Round completes with no bets placed
    - All players keep original chip counts
    - No last raiser set
    """
    scenario = {
        0: [{"action_type": ActionType.CHECK, "reasoning": "Checking"}],
        1: [{"action_type": ActionType.CHECK, "reasoning": "Checking"}],
        2: [{"action_type": ActionType.CHECK, "reasoning": "Checking"}],
        3: [{"action_type": ActionType.CHECK, "reasoning": "Checking"}],
    }

    table, players = run_betting_scenario(mock_game, setup_players, scenario)

    # Verify no bets placed
    for player in players:
        assert player.bet == 0, f"{player.name} should have no bet"
        assert player.chips == 1000, f"{player.name} should have all chips"
        assert not player.folded, f"{player.name} should not be folded"

    assert table.last_raiser is None, "No raiser should be set after all checks"
    assert len(table.needs_to_act) == 0, "needs_to_act should be empty"
