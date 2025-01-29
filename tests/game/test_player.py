from unittest.mock import Mock

import pytest

from data.states.player_state import PlayerState
from data.types.action_decision import ActionDecision, ActionType
from data.types.player_types import PlayerPosition
from game.hand import Hand
from game.player import Player


class TestPlayer:
    @pytest.fixture
    def player(self):
        return Player("TestPlayer", 1000)

    @pytest.fixture
    def mock_game(self):
        """Create a properly configured mock game for testing."""
        mock = Mock()
        mock.current_bet = 0
        mock.pot = Mock()
        mock.pot.pot = 0

        # Configure config with proper values instead of Mock objects
        mock.config = Mock()
        mock.config.max_raises_per_round = 4
        mock.config.min_bet = 100

        # Configure round_state with proper values
        mock.round_state = Mock()
        mock.round_state.raise_count = 0

        return mock

    def test_player_initialization(self, player):
        """Test that a player is initialized with correct default values"""
        assert player.name == "TestPlayer"
        assert player.chips == 1000
        assert player.bet == 0
        assert player.folded is False
        assert isinstance(player.hand, Hand)

    def test_place_bet_normal(self, player, mock_game):
        """Test placing a valid bet"""
        bet_amount = 500
        actual_bet = player.place_bet(bet_amount, mock_game)

        assert actual_bet == bet_amount
        assert player.chips == 500  # Started with 1000
        assert player.bet == 500

    def test_place_bet_more_than_chips(self, player, mock_game):
        """Test placing a bet larger than available chips"""
        bet_amount = 1500  # More than initial 1000
        actual_bet = player.place_bet(bet_amount, mock_game)

        assert actual_bet == 1000  # Should only bet what's available
        assert player.chips == 0
        assert player.bet == 1000

    def test_place_negative_bet(self, player, mock_game):
        """Test that placing a negative bet raises ValueError"""
        with pytest.raises(ValueError):
            player.place_bet(-100, mock_game)

    def test_fold(self, player):
        """Test folding functionality"""
        assert player.folded is False
        player._fold()
        assert player.folded is True

    def test_reset_bet(self, player, mock_game):
        """Test resetting bet amount"""
        player.place_bet(500, mock_game)
        assert player.bet == 500

        player.reset_bet()
        assert player.bet == 0

    def test_reset_for_new_round(self, player, mock_game):
        """Test resetting player state for new round"""
        # Setup: place bet and fold
        player.place_bet(500, mock_game)
        player._fold()
        assert player.bet == 500
        assert player.folded is True

        # Reset for new round
        player.reset_for_new_round()
        assert player.bet == 0
        assert player.folded is False
        # Chips should remain unchanged
        assert player.chips == 500

    def test_str_representation(self, player):
        """Test string representation of player"""
        expected = "TestPlayer (chips: 1000, folded: False)"
        assert str(player) == expected

    def test_multiple_bets_same_round(self, player, mock_game):
        """Test placing multiple bets in the same round"""
        player.place_bet(300, mock_game)
        player.place_bet(200, mock_game)

        assert player.chips == 500
        assert player.bet == 500

    def test_bet_all_chips(self, player, mock_game):
        """Test betting all available chips"""
        actual_bet = player.place_bet(1000, mock_game)

        assert actual_bet == 1000
        assert player.chips == 0
        assert player.bet == 1000

    def test_zero_bet(self, player, mock_game):
        """Test placing a zero bet"""
        actual_bet = player.place_bet(0, mock_game)

        assert actual_bet == 0
        assert player.chips == 1000
        assert player.bet == 0
        assert mock_game.pot.pot == 0

    def test_initialize_with_zero_chips(self, mock_game):
        """Test creating a player with zero initial chips"""
        player = Player("BrokePlayer", 0)

        assert player.chips == 0
        # Verify can't bet anything
        actual_bet = player.place_bet(100, mock_game)
        assert actual_bet == 0
        assert player.bet == 0
        assert mock_game.pot.pot == 0

    def test_initialize_with_negative_chips(self):
        """Test that initializing with negative chips raises ValueError"""
        with pytest.raises(ValueError):
            Player("NegativePlayer", -100)

    def test_multiple_folds(self, player):
        """Test that folding multiple times keeps player folded"""
        player._fold()
        assert player.folded is True

        player._fold()  # Second fold
        assert player.folded is True

    def test_bet_after_fold(self, player, mock_game):
        """Test that a player can still bet after folding (shouldn't be prevented)"""
        player._fold()
        actual_bet = player.place_bet(500, mock_game)

        assert actual_bet == 500
        assert player.chips == 500
        assert player.bet == 500

    def test_decimal_chip_amount(self):
        """Test that initializing with decimal chip amount raises ValueError"""
        with pytest.raises(ValueError):
            Player("DecimalPlayer", 100.5)

    def test_reset_bet_multiple_times(self, player, mock_game):
        """Test resetting bet multiple times"""
        player.place_bet(500, mock_game)
        player.reset_bet()
        player.reset_bet()  # Second reset

        assert player.bet == 0
        assert player.chips == 500

    def test_name_empty_string(self):
        """Test creating a player with an empty name"""
        with pytest.raises(ValueError):
            Player("", 1000)

    def test_name_whitespace(self):
        """Test creating a player with whitespace name"""
        with pytest.raises(ValueError):
            Player("   ", 1000)

    def test_very_large_chip_amount(self, mock_game):
        """Test initialization and betting with very large chip amounts"""
        large_amount = 10**9  # 1 billion
        player = Player("RichPlayer", large_amount)

        bet_amount = 10**8  # 100 million
        actual_bet = player.place_bet(bet_amount, mock_game)

        assert actual_bet == bet_amount
        assert player.chips == large_amount - bet_amount
        assert player.bet == bet_amount

    def test_all_in_raise_behavior(self, player, mock_game):
        """Test that a player going all-in with a raise is handled correctly."""
        player.chips = 500
        mock_game.current_bet = 300
        mock_game.config.min_bet = 100
        mock_game.round_state.raise_count = 0
        mock_game.config.max_raises_per_round = 4

        # Player tries to raise to 600 but only has 500
        player._raise(600, mock_game)

        assert player.is_all_in
        assert player.chips == 0
        assert player.bet == 500

    def test_all_in_call_behavior(self, player, mock_game):
        """Test that a player going all-in with a call is handled correctly."""
        player.chips = 300
        mock_game.current_bet = 500
        mock_game.round_state.raise_count = 0
        mock_game.config.max_raises_per_round = 4

        # Player tries to call 500 but only has 300, should fold
        player._call(500, mock_game)

        assert player.folded is True
        assert player.chips == 300  # Should keep their chips when folding
        assert player.bet == 0  # No bet placed when folding

    def test_exact_call_all_in(self, player, mock_game):
        """Test that a player going all-in with an exact call amount is handled correctly."""
        player.chips = 500
        mock_game.current_bet = 500

        player._call(500, mock_game)

        assert player.is_all_in
        assert player.chips == 0
        assert player.bet == 500
        assert player.called is True

    def test_partial_raise_forces_all_in(self, player, mock_game):
        """Test that a player who can't make minimum raise goes all-in."""
        player.chips = 350
        mock_game.current_bet = 300
        mock_game.config.min_bet = 100
        mock_game.round_state.raise_count = 0
        mock_game.config.max_raises_per_round = 4

        # Player tries to raise but can't meet minimum raise requirement
        player._raise(400, mock_game)

        assert player.is_all_in
        assert player.chips == 0
        assert player.bet == 350  # All-in call amount

    def test_execute_raise_action(self, player, mock_game):
        """Test executing a raise action"""
        action = ActionDecision(action_type=ActionType.RAISE, raise_amount=500)
        player.execute(action, mock_game)
        assert player.bet == 500
        assert player.chips == 500

    def test_execute_call_action(self, player, mock_game):
        """Test executing a call action"""
        mock_game.current_bet = 300
        action = ActionDecision(action_type=ActionType.CALL)
        player.execute(action, mock_game)
        assert player.bet == 300
        assert player.called is True

    def test_execute_check_action(self, player, mock_game):
        """Test executing a check action"""
        action = ActionDecision(action_type=ActionType.CHECK)
        player.execute(action, mock_game)
        assert player.checked is True

    def test_execute_fold_action(self, player, mock_game):
        """Test executing a fold action"""
        action = ActionDecision(action_type=ActionType.FOLD)
        player.execute(action, mock_game)
        assert player.folded is True

    def test_position_property(self, player):
        """Test getting and setting player position"""
        player.position = PlayerPosition.DEALER
        assert player.position == PlayerPosition.DEALER

        player.position = PlayerPosition.SMALL_BLIND
        assert player.position == PlayerPosition.SMALL_BLIND

    def test_get_state(self, player):
        """Test getting player state"""
        state = player.get_state()
        assert isinstance(state, PlayerState)
        assert state.name == player.name
        assert state.chips == player.chips
        assert state.bet == player.bet
        assert state.folded == player.folded

    def test_player_equality(self):
        """Test player equality comparison"""
        player1 = Player("TestPlayer", 1000)
        player2 = Player("TestPlayer", 2000)  # Same name, different chips
        player3 = Player("OtherPlayer", 1000)

        assert player1 == player2  # Same name should be equal
        assert player1 != player3  # Different name should not be equal
        assert player1 != "TestPlayer"  # Different type should not be equal

    def test_player_hash(self):
        """Test player hash functionality"""
        player1 = Player("TestPlayer", 1000)
        player2 = Player("TestPlayer", 2000)
        player3 = Player("OtherPlayer", 1000)

        # Same name should hash to same value
        assert hash(player1) == hash(player2)
        # Different names should hash to different values
        assert hash(player1) != hash(player3)

        # Test using players in a set
        player_set = {player1, player2, player3}
        assert len(player_set) == 2  # player1 and player2 count as same
