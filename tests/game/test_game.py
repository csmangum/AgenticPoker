from datetime import datetime
from unittest.mock import MagicMock, Mock

import pytest

from agents.agent import Agent
from data.types.action_decision import ActionType
from data.types.pot_types import SidePot
from game import AgenticPoker
from game.hand import Hand


@pytest.fixture
def mock_players():
    """Fixture to create mock players."""
    players = []
    for name in ["Alice", "Bob", "Charlie"]:
        player = Mock(spec=Agent)
        player.name = name
        player.chips = 1000
        player.folded = False
        player.bet = 0
        # Mock hand attribute
        player.hand = Mock(spec=Hand)
        player.hand.__eq__ = lambda self, other: False
        player.hand.__gt__ = lambda self, other: False

        # Create a proper place_bet method that updates chips
        def make_place_bet(p):
            def place_bet(amount):
                actual_amount = min(amount, p.chips)
                p.chips -= actual_amount
                p.bet += actual_amount
                return actual_amount

            return place_bet

        player.place_bet = make_place_bet(player)
        players.append(player)
    return players


@pytest.fixture
def game(mock_players):
    """Fixture to create game instance."""
    return AgenticPoker(
        players=mock_players,
        small_blind=50,
        big_blind=100,
        ante=10,
        session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )


def test_game_initialization(game, mock_players):
    """Test game initialization with valid parameters.

    Assumptions:
        - All players start with 1000 chips
        - Players have valid names and attributes
        - Session ID is automatically generated if not provided

    Prerequisites:
        - mock_players fixture is available
        - game fixture is initialized with standard betting structure
    """
    assert len(game.table) == 3
    assert game.small_blind == 50
    assert game.big_blind == 100
    assert game.ante == 10
    assert game.session_id is not None
    # Verify all players have initial chips
    for player in game.table:
        assert player.chips == 1000


def test_invalid_game_initialization(mock_players):
    """Test game initialization with invalid parameters.

    Assumptions:
        - Game requires at least 2 players
        - Players cannot have negative chips
        - Invalid parameters raise ValueError

    Prerequisites:
        - mock_players fixture is available
        - Mock class is available from unittest.mock
    """
    with pytest.raises(ValueError):
        AgenticPoker(
            players=[],  # Empty players list
            small_blind=50,
            big_blind=100,
            ante=10,
        )

    with pytest.raises(ValueError):
        # Create players with negative chips
        invalid_players = [
            Mock(spec=Agent, chips=-1000, name=f"Player{i}") for i in range(3)
        ]
        AgenticPoker(
            players=invalid_players,
            small_blind=50,
            big_blind=100,
            ante=10,
        )


def test_dealer_rotation(game, mock_players):
    """Test dealer button rotation between rounds.

    Assumptions:
        - Dealer button moves clockwise by one position
        - Rotation happens at end of each round
        - Dealer index wraps around table size

    Prerequisites:
        - game fixture is initialized
        - mock_players fixture provides at least 2 players
    """
    initial_dealer = game.dealer_index

    # Reset round which rotates dealer
    game._reset_round()

    # Verify dealer rotated
    expected_dealer = (initial_dealer + 1) % len(mock_players)
    assert game.dealer_index == expected_dealer


def test_round_initialization(game, mock_players):
    """Test initialization of a new round.

    Assumptions:
        - All player bets are reset to 0
        - All player folded states are reset to False
        - Each player receives a new hand
        - Pot is reset to 0

    Prerequisites:
        - game fixture is initialized
        - mock_players fixture is available
        - Players have hand attribute
    """
    game._initialize_round()

    # Verify round state
    assert game.pot.pot == 0
    assert all(player.bet == 0 for player in game.table)
    assert all(not player.folded for player in game.table)
    assert all(hasattr(player, "hand") for player in game.table)


def test_collect_blinds_and_antes(game, player_factory):
    """Test that blinds and antes are collected correctly and pot is updated properly.

    Assumptions:
        - Players have sufficient chips for blinds and antes
        - Dealer position determines blind positions
        - Antes are collected before blinds
        - All collected chips go to pot

    Prerequisites:
        - game fixture is initialized
        - player_factory fixture is available
        - Players can track chips and bets
    """
    # Create players with known chip stacks
    players = [
        player_factory(name="Alice", chips=1000),
        player_factory(name="Bob", chips=1000),
        player_factory(name="Charlie", chips=1000),
    ]
    game.table.players = players

    # Set up initial state
    game.small_blind = 50
    game.big_blind = 100
    game.ante = 10
    game.dealer_index = 0

    # Store initial chip counts
    initial_chips = {p: p.chips for p in game.table}

    # Collect blinds and antes
    game._collect_blinds_and_antes()

    # Calculate expected pot amount
    num_players = len(players)
    expected_antes = num_players * game.ante  # 3 players * $10 = $30
    expected_blinds = game.small_blind + game.big_blind  # $50 + $100 = $150
    expected_pot = expected_antes + expected_blinds  # $30 + $150 = $180

    # Verify pot amount
    assert (
        game.pot.pot == expected_pot
    ), f"Expected pot of ${expected_pot}, got ${game.pot.pot}"

    # Verify player chip counts
    dealer = game.table[0]  # Alice
    sb_player = game.table[1]  # Bob
    bb_player = game.table[2]  # Charlie

    # Check dealer's chips (only pays ante)
    assert dealer.chips == initial_chips[dealer] - game.ante

    # Check small blind player's chips (pays ante + small blind)
    assert sb_player.chips == initial_chips[sb_player] - game.ante - game.small_blind

    # Check big blind player's chips (pays ante + big blind)
    assert bb_player.chips == initial_chips[bb_player] - game.ante - game.big_blind

    # Verify current bet is set to big blind
    assert game.current_bet == game.big_blind


def test_pot_not_double_counted(game, player_factory):
    """Test that pot amounts are correctly tracked during betting.

    Assumptions:
        - Bets are added to pot incrementally
        - Player chip counts decrease as they bet
        - Pot total matches sum of all player contributions
        - End of betting round doesn't modify pot total

    Prerequisites:
        - game fixture is initialized
        - player_factory fixture is available
        - Players can place bets and fold
        - Pot can track total and add bets
    """
    # Create players with known chip stacks
    players = [
        player_factory(name="Alice", chips=1000),  # Dealer
        player_factory(name="Bob", chips=1000),  # Small Blind
        player_factory(name="Charlie", chips=1000),  # Big Blind
        player_factory(name="Randy", chips=1000),  # UTG
    ]
    game.table.players = players

    # Set up initial state
    game.small_blind = 50
    game.big_blind = 100
    game.ante = 10
    game.dealer_index = 0

    # Reset pot before collecting blinds
    game.pot.reset_pot()

    # Collect blinds and antes
    game._collect_blinds_and_antes()

    # Verify initial pot after blinds/antes
    # Each player pays 10 ante (40 total)
    # SB pays 50, BB pays 100
    expected_initial_pot = 40 + 50 + 100
    assert (
        game.pot.pot == expected_initial_pot
    ), f"Expected initial pot of ${expected_initial_pot}, got ${game.pot.pot}"

    # Simulate pre-draw betting round:
    # Alice raises to 300
    alice = players[0]
    alice.chips -= 300
    alice.bet += 300
    game.pot.add_to_pot(300)
    expected_pot = expected_initial_pot + 300
    assert (
        game.pot.pot == expected_pot
    ), f"Pot should be ${expected_pot} after Alice's bet"

    # Bob folds (already put in SB)
    bob = players[1]
    bob.folded = True

    # Charlie calls (already put in BB)
    charlie = players[2]
    charlie.chips -= 200  # Additional 200 to match 300 total
    charlie.bet += 200
    game.pot.add_to_pot(200)
    expected_pot = expected_pot + 200
    assert (
        game.pot.pot == expected_pot
    ), f"Pot should be ${expected_pot} after Charlie's bet"

    # Randy calls
    randy = players[3]
    randy.chips -= 300
    randy.bet += 300
    game.pot.add_to_pot(300)
    expected_pot = expected_pot + 300
    assert (
        game.pot.pot == expected_pot
    ), f"Pot should be ${expected_pot} after Randy's bet"

    # End betting round (should only clear bets, not modify pot)
    initial_pot = game.pot.pot
    game.pot.end_betting_round(game.table.players)
    assert game.pot.pot == initial_pot, "Pot should not change at end of betting round"

    # Verify player chip counts
    assert alice.chips == 1000 - 10 - 300, f"Alice's chips incorrect: {alice.chips}"
    assert bob.chips == 1000 - 10 - 50, f"Bob's chips incorrect: {bob.chips}"
    assert (
        charlie.chips == 1000 - 10 - 100 - 200
    ), f"Charlie's chips incorrect: {charlie.chips}"
    assert randy.chips == 1000 - 10 - 300, f"Randy's chips incorrect: {randy.chips}"


def test_post_draw_betting_skipped_all_all_in(game, player_factory):
    """Test that post-draw betting is skipped when all remaining players are all-in.

    Assumptions:
        - Players marked as all-in have chips = 0
        - Folded players don't affect all-in status check
        - Betting phase returns True to continue to showdown

    Prerequisites:
        - game fixture is initialized
        - player_factory fixture is available
        - Players can be created with is_all_in flag
    """
    # Create players in all-in state
    all_in1 = player_factory(name="AllIn1", chips=0, is_all_in=True, bet=500)
    all_in2 = player_factory(name="AllIn2", chips=0, is_all_in=True, bet=300)
    folded = player_factory(name="Folded", folded=True)

    game.table.players = [all_in1, all_in2, folded]

    # Run post-draw betting
    result = game._handle_post_draw_phase()

    # Verify betting was skipped and returned True to continue to showdown
    assert result is True


def test_play_game_max_rounds(game, player_factory):
    """Test that game stops after max_rounds is reached.

    Assumptions:
        - Players start with sufficient chips to play multiple rounds
        - Game state is properly initialized
        - Phase handlers are mocked to isolate max rounds logic
        - Game should stop BEFORE exceeding max_rounds

    Prerequisites:
        - game fixture is initialized
        - player_factory fixture is available
        - MagicMock is imported from unittest.mock
    """
    # Create players with different chip stacks
    players = [
        player_factory(name="P1", chips=1000),
        player_factory(name="P2", chips=1000),
        player_factory(name="P3", chips=1000),
    ]
    game.table.players = players

    # Set max rounds to 2
    max_rounds = 2

    # Mock the phase handlers to avoid full game logic
    game._handle_pre_draw_phase = MagicMock(return_value=True)
    game._handle_draw_phase = MagicMock(return_value=True)
    game._handle_post_draw_phase = MagicMock(return_value=True)
    game._handle_showdown = MagicMock()

    # Play game
    game.play_game(max_rounds=max_rounds)

    # Verify game stopped after max rounds
    assert (
        game.round_number == max_rounds
    ), f"Expected {max_rounds} rounds, got {game.round_number}"

    # Verify phase handlers were called expected number of times
    assert game._handle_pre_draw_phase.call_count == max_rounds
    assert game._handle_draw_phase.call_count == max_rounds
    assert game._handle_post_draw_phase.call_count == max_rounds
    assert game._handle_showdown.call_count == max_rounds


def test_play_game_elimination(game, player_factory):
    """Test that game ends when players are eliminated.

    Assumptions:
        - Game ends immediately when only one player has chips > 0
        - Eliminated players (chips = 0) are removed from table
        - Phase handlers should not be called if game ends before first round

    Prerequisites:
        - game fixture is initialized
        - player_factory fixture is available
        - MagicMock is imported from unittest.mock
    """
    # Create players with different chip stacks
    players = [
        player_factory(name="Winner", chips=1000),
        player_factory(name="Loser1", chips=0),
        player_factory(name="Loser2", chips=0),
    ]
    game.table.players = players

    # Mock the phase handlers
    game._handle_pre_draw_phase = MagicMock(return_value=True)
    game._handle_draw_phase = MagicMock(return_value=True)
    game._handle_post_draw_phase = MagicMock(return_value=True)
    game._handle_showdown = MagicMock()

    # Play game
    game.play_game()

    # Verify game ended after eliminations
    assert len(game.table.players) == 1
    assert game.table.players[0].name == "Winner"

    # Verify phase handlers were not called (game should end before first round)
    assert game._handle_pre_draw_phase.call_count == 0
    assert game._handle_draw_phase.call_count == 0
    assert game._handle_post_draw_phase.call_count == 0
    assert game._handle_showdown.call_count == 0


def test_handle_showdown(game, player_factory):
    """Test showdown handling and pot distribution.

    Assumptions:
        - Pot contains only main pot (no side pots)
        - All bets have been collected into pot before showdown
        - Player hands are properly mocked with P1 winning
        - Initial chips are tracked for each player

    Prerequisites:
        - game fixture is initialized
        - player_factory fixture is available
        - MagicMock is imported from unittest.mock
        - Pot is properly initialized
    """
    # Create players with mock hands
    p1 = player_factory(name="P1", chips=900)  # Lost 100 in betting
    p2 = player_factory(name="P2", chips=900)  # Lost 100 in betting

    # Create spies first
    compare_spy = MagicMock()
    show_spy = MagicMock()

    # Create base mock hand with compare_to method
    class MockHand:
        def __init__(self, name):
            self.name = name

        def compare_to(self, other):
            # Call spy first
            compare_spy(self.name, other.name)
            # P1's hand beats P2's hand
            if self.name == "P1" and other.name == "P2":
                return 1
            # P2's hand loses to P1's hand
            if self.name == "P2" and other.name == "P1":
                return -1
            # Equal to self
            return 0

        def show(self):
            # Call spy
            show_spy(self.name)
            return f"{self.name}'s hand"

    # Set up mock hands where P1 wins
    p1.hand = MockHand("P1")
    p2.hand = MockHand("P2")

    game.table.players = [p1, p2]
    game.initial_chips = {p1: 1000, p2: 1000}

    # Set up pot with 200 (100 from each player)
    game.pot.pot = 200

    # Run showdown
    game._handle_showdown()

    # Verify winner got the pot
    assert p1.chips == 1100  # Original 900 + 200 pot
    assert p2.chips == 900  # No change

    # Debug: Print all calls to compare_spy
    print("\nAll compare_spy calls:")
    for call in compare_spy.mock_calls:
        print(f"  {call}")

    # Verify hand comparison was called
    compare_spy.assert_called()
    show_spy.assert_called()

    # Verify at least one comparison happened between P1 and P2
    assert any(
        call[1] == ("P1", "P2") or call[1] == ("P2", "P1") 
        for call in compare_spy.mock_calls
    ), "No comparison between P1 and P2 found"
